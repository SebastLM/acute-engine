use std::format;
use std::collections::HashMap;

use crate::gguf::{GgufCtx, GgufValue, gguf_init_from_file};

// stores imp tensor info for quickly accessing the respective data 
struct AcuteTensorWeight {
   offset: u64,
   gguf_idx: usize, 
   data_size: u64,
}

fn llama_split_prefix(fname: &str) -> &str {
    // basename-00001-of-xxxxx.gguf
    let without_ext = fname.strip_suffix(".gguf").unwrap_or(fname);
    // splitting where '-' is found. Operates from right-to-left o no problem will arise if prefix contains '-'
    // basename-00001-of-xxxxx
    let mut parts = without_ext.rsplitn(4, '-');
    parts.next(); // total splits    'xxxxx'
    parts.next(); // "of" string 
    parts.next(); // current split 'xxxxx'

    parts.next().unwrap_or(fname)
}


fn get_splits_list(fname: &str, n_splits: usize) -> Vec<String> {
    let mut splits: Vec<String> = Vec::with_capacity(n_splits);
    let prefix = llama_split_prefix(fname);

    for id in 1..=n_splits {
        // zero padded 5 digits for both current split and total
        let split = format!("{}-{:05}-of-{:05}.gguf", prefix, id, n_splits);
        splits.push(split);
    }
    splits
}


// loads all the files into respective GgufCtx
// custom file list can be received
fn model_loader(fname: &str, mut splits: Vec<String>) -> Result<Vec<(String, GgufCtx)>, String> {
    
    // parsing main file
    let ctx0 = match gguf_init_from_file(fname) {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // finding number of splits
    let n_splits = match ctx0.find_kv("split.count") {
        Some(v) => match v {
            GgufValue::Uint32(n) => *n as usize,
            _ => return Err(format!("ERROR: split.count is not a u32")),
        }
        None => return Err(format!("ERROR: no value found for split.count")),
    };

    let mut ggufs = Vec::with_capacity(n_splits);
    
    // for AcuteDCtx
    let mut n_elements: i64 = 0;
    let mut n_bytes: u64 = 0;

    // create weight_map
    let mut weights_map: HashMap<String, AcuteTensorWeight> = HashMap::new(); // key is tensor name

    // add fname tensor's to weigth_map        
    let mut add_wm_ctx = |idx: usize, ctx: GgufCtx, path: String| -> Result<(), String> {
        for info in &ctx.tensor_infos {
            if weights_map.contains_key(&info.name) {
                return Err(format!("ERROR: tensor '{}' is duplicated", info.name));
            }

            n_elements += info.ne.iter().product::<i64>();
            n_bytes += info.nbytes();

            weights_map.insert(info.name.clone(), AcuteTensorWeight { offset: info.offset, gguf_idx: idx,  data_size: info.nbytes()});
        }
        ggufs.push((path, ctx));
        Ok(())
    };

    add_wm_ctx(0, ctx0, fname.to_string())?;

    if n_splits > 1 {
        
        if splits.is_empty() { // no custom splits given
            splits = get_splits_list(fname, n_splits);
        }

        if n_splits != splits.len() { return Err(format!("Error: invalid custom split len"))}

        // index 0 is always the already-processed first split (fname)
        for (idx, path) in splits.into_iter().enumerate().skip(1) {
            let split_ctx = match gguf_init_from_file(&path) {
                Ok(ctx) => ctx,
                Err(e) => return Err(e),
            };
            add_wm_ctx(idx, split_ctx, path)?;
        }
    }
    Ok(ggufs)
}
