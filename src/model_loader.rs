use std::format;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, SeekFrom};

use crate::acute::AcuteArena;
use crate::gguf::{GgmlType, GgufCtx, GgufTensorInfo, GgufValue, gguf_init_from_file};
use crate::tensor::Tensor;

// stores tensor info for quickly finding which split file (and where in it)
// a tensor's data lives, without re-scanning every GgufCtx
pub struct AcuteTensorWeight {
   pub offset: u64,
   pub gguf_idx: usize,
   pub data_size: u64,
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

// GGUF stores ne[0] as the fastest-varying (innermost) dim, padded with 1s
// out to GGML_MAX_DIMS. acute's Tensor is row-major with the last shape
// entry fastest-varying, so reverse the axes here and trim the padding
// dims (now leading 1s after the reverse) back off
fn ggml_shape_to_row_major(ne: &[i64; 4]) -> Vec<usize> {
    let mut shape: Vec<usize> = ne.iter().rev().map(|&d| d as usize).collect();
    while shape.len() > 1 && shape[0] == 1 {
        shape.remove(0);
    }
    shape
}


// loads all the files into respective GgufCtx, builds a name -> location
// index (weights_map) and totals up how many bytes the tensor data will
// need once actually read in
// custom file list can be received
pub fn model_loader(
    fname: &str,
    mut splits: Vec<String>,
) -> Result<(Vec<(String, GgufCtx)>, HashMap<String, AcuteTensorWeight>, u64), String> {

    // parsing main file
    let ctx0 = match gguf_init_from_file(fname) {
        Ok(ctx) => ctx,
        Err(e) => return Err(e),
    };

    // finding number of splits, single-file models don't carry this key at
    // all so its absence just means "one file", not an error
    let n_splits = match ctx0.find_kv("split.count") {
        Some(v) => match v {
            GgufValue::Uint32(n) => *n as usize,
            _ => return Err(format!("ERROR: split.count is not a u32")),
        }
        None => 1,
    };

    let mut ggufs = Vec::with_capacity(n_splits);

    // for AcuteArena sizing
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

    let _ = n_elements; // currently only n_bytes is needed to size the arena
    Ok((ggufs, weights_map, n_bytes))
}

fn load_tensor(
    file: &mut File,
    ctx: &GgufCtx,
    info: &GgufTensorInfo,
    arena: &mut AcuteArena,
) -> Result<Tensor<f32>, String> {
    if info.dtype != GgmlType::F32 {
        return Err(format!(
            "tensor '{}' has dtype {:?}, only F32 is supported for now",
            info.name, info.dtype
        ));
    }

    file.seek(SeekFrom::Start(ctx.data_offset + info.offset))
        .map_err(|e| format!("failed to seek to tensor '{}': {e}", info.name))?;

    let shape = ggml_shape_to_row_major(&info.ne);
    let n_elements: usize = shape.iter().product();

    Tensor::from_reader(arena, file, n_elements, shape.into_boxed_slice())
        .map_err(|e| format!("failed loading tensor '{}': {e}", info.name))
}

// parses fname (and its splits, if any), sizes an AcuteArena to fit every
// tensor's data, then reads each tensor's raw bytes straight from its GGUF
// file into that arena. only F32 is supported for now, anything quantized
// gets rejected with a clear error instead of being silently misread
pub fn load_model(fname: &str, splits: Vec<String>) -> Result<(AcuteArena, HashMap<String, Tensor<f32>>), String> {
    let (ggufs, weights_map, n_bytes) = model_loader(fname, splits)?;
    let mut arena = AcuteArena::new(n_bytes as usize)?;

    let mut tensors: HashMap<String, Tensor<f32>> = HashMap::with_capacity(weights_map.len());
    for (path, ctx) in &ggufs {
        let mut file = File::open(path).map_err(|e| format!("error reopening '{path}': {e}"))?;
        for info in &ctx.tensor_infos {
            let tensor = load_tensor(&mut file, ctx, info, &mut arena)?;
            tensors.insert(info.name.clone(), tensor);
        }
    }

    Ok((arena, tensors))
}
