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
    let mut splits: Vec<String> = Vec::with_capacity(n_splits - 1); // accounting for already processed first split
    let prefix = llama_split_prefix(fname);
    
    for id in 2..=n_splits {
        // zero padded 5 digits for both current split and total
        let split = format!("{}-{:05}-of-{:05}.gguf", prefix, id, n_splits);
        splits.push(split);
    }
    splits
}


// loads all the files into respective GgufCtx
// custom file list can be received
fn model_loader(fname: &str, splits: Vec<&str>) -> Result<Vec<GgufCtx>, String> {
    
    // parsing main file
    let gguf_ctx_01 = match gguf_init_from_file(fname) {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };
    
    // finding number of splits
    let n_splits = match gguf_ctx_01.find_kv("split.count") {
        Some(v) => match v {
            GgufValue::Uint32(n) => *n as usize,
            _ => return Err(format!("ERROR: split.count is not a u32")),
        }
        None => return Err(format!("ERROR: no value found for split.count")),
    };

    let mut gguf_ctxs = Vec::with_capacity(n_splits);
    gguf_ctxs.push(gguf_ctx_01);

    let weights_map: HashMap<String, AcuteTensorWeight> = HashMap::new(); // key is tensor name
    if n_splits > 1 {
        
        if !splits.is_empty() {
            if n_splits != splits.len() { return Err(format!("Error: invalid custom split len"))}

            for path in splits {
                if path.eq(fname) { continue; }
                
                let split_ctx = match gguf_init_from_file(path) {
                    Ok(ctx) => ctx,
                    Err(e) => return Err(e),
                };
                gguf_ctxs.push(split_ctx);
            }
        } else { // no custom splits given
            let generated_splits = get_splits_list(&fname, n_splits);
            
            for path in generated_splits {
                let split_ctx = match gguf_init_from_file(&path) {
                    Ok(ctx) => ctx,
                    Err(e) => return Err(e),
                };
                gguf_ctxs.push(split_ctx);
            }
        }
        
        // create weight_map // if n_splits = 1, dont create weight map, and simply pass it to the alocator, if its empty the alocator will use the weight_map else where use the Ggufctx
    }
    Ok(gguf_ctxs)
}
