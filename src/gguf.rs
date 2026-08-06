// Parse a GGUF file's header: magic, version, key-value metadata and
// per-tensor info (name, shape, dtype, byte offset/size within the data
// section). This is metadata only — no tensor bytes are read here (following llama architecture).
//
// model_loader will later use GgufCtx to build the ggml_ctx and the weights_map.
// For each GgufTensorInfo it will fseek() to `ctx.data_offset + info.offset` in the file 
// and copy the bytes into the memory arena that tensor_allocator hands out.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

const GGML_MAX_DIMS: usize = 4;
const GGML_MAX_NAME: usize = 64;

// ---------------------------------------------------------------------
// ggml tensor element types
// ---------------------------------------------------------------------

// Tensor element type, as tagged by the `type` field of a GGUF tensor
// info entry. Only variants we can size/validate correctly are listed;
// files using an unrecognized id (e.g. the newer IQ*/TQ*/MXFP4 block
// quant types) are rejected with a clear error instead of guessing at
// their block/type size.
// TODO: IQ*/TQ*/MXFP4
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    F64,
    Bf16,
    I8,
    I16,
    I32,
    I64,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlType {
    fn from_id(id: u32) -> Result<GgmlType, String> {
        use GgmlType::*;
        Ok(match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            30 => Bf16,
            other => return Err(format!("unsupported/unknown ggml type id {other}")),
        })
    }

    // number of elements per quantization block (1 for unquantized types)
    pub fn block_size(self) -> i64 {
        use GgmlType::*;
        match self {
            F32 | F16 | F64 | Bf16 | I8 | I16 | I32 | I64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => 32,
            Q2K | Q3K | Q4K | Q5K | Q6K | Q8K => 256,
        }
    }

    // size in bytes of one block
    pub fn type_size(self) -> u64 {
        use GgmlType::*;
        match self {
            F32 => 4,
            F16 => 2,
            F64 => 8,
            Bf16 => 2,
            I8 => 1,
            I16 => 2,
            I32 => 4,
            I64 => 8,
            Q4_0 => 18,
            Q4_1 => 20,
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34,
            Q8_1 => 36,
            Q2K => 84,
            Q3K => 110,
            Q4K => 144,
            Q5K => 176,
            Q6K => 210,
            Q8K => 292,
        }
    }
}

// ---------------------------------------------------------------------
// GGUF metadata (key-value) values
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl GgufValueType {
    fn from_id(id: u32) -> Result<GgufValueType, String> {
        use GgufValueType::*;
        Ok(match id {
            0 => Uint8,
            1 => Int8,
            2 => Uint16,
            3 => Int16,
            4 => Uint32,
            5 => Int32,
            6 => Float32,
            7 => Bool,
            8 => String,
            9 => Array,
            10 => Uint64,
            11 => Int64,
            12 => Float64,
            other => return Err(format!("invalid GGUF value type id {other}")),
        })
    }
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

// ---------------------------------------------------------------------
// tensor info
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    // element counts per dimension, ggml order (ne[0] = innermost/fastest-varying)
    pub ne: [i64; GGML_MAX_DIMS],
    pub dtype: GgmlType,
    // byte strides per dimension, same convention as ne
    pub nb: [u64; GGML_MAX_DIMS],
    // byte offset of this tensor's data, relative to the start of the data section
    pub offset: u64,
}

impl GgufTensorInfo {
    // total size in bytes of this tensor's data
    pub fn nbytes(&self) -> u64 {
        self.nb[GGML_MAX_DIMS - 1] * self.ne[GGML_MAX_DIMS - 1] as u64
    }
}

// ---------------------------------------------------------------------
// gguf context
// ---------------------------------------------------------------------

#[derive(Debug)]
pub struct GgufCtx {
    pub version: u32,
    pub kv: Vec<(String, GgufValue)>,
    pub tensor_infos: Vec<GgufTensorInfo>,

    pub alignment: usize,
    // file offset where the tensor data section starts
    pub data_offset: u64,
    // total padded size of the tensor data section, in bytes
    pub data_size: u64,
}

impl GgufCtx {
    pub fn find_kv(&self, key: &str) -> Option<&GgufValue> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

fn pad(x: u64, align: u64) -> u64 {
    (x + align - 1) / align * align
}

// ---------------------------------------------------------------------
// low level byte reader
// ---------------------------------------------------------------------

struct GgufReader<R: Read + Seek> {
    inner: R,
}

impl<R: Read + Seek> GgufReader<R> {
    fn new(inner: R) -> Self {
        GgufReader { inner }
    }

    fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; n];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| format!("failed to read {n} bytes: {e}"))?;
        Ok(buf.to_vec())
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_bool(&mut self) -> Result<bool, String> {
        Ok(self.read_u8()? != 0)
    }

    fn read_i8(&mut self) -> Result<i8, String> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u64()?;
        let len = usize::try_from(len).map_err(|_| format!("string length {len} does not fit in usize"))?;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes).map_err(|e| format!("invalid utf-8 string: {e}"))
    }

    fn read_value(&mut self, t: GgufValueType) -> Result<GgufValue, String> {
        use GgufValueType::*;
        Ok(match t {
            Uint8 => GgufValue::Uint8(self.read_u8()?),
            Int8 => GgufValue::Int8(self.read_i8()?),
            Uint16 => GgufValue::Uint16(self.read_u16()?),
            Int16 => GgufValue::Int16(self.read_i16()?),
            Uint32 => GgufValue::Uint32(self.read_u32()?),
            Int32 => GgufValue::Int32(self.read_i32()?),
            Float32 => GgufValue::Float32(self.read_f32()?),
            Bool => GgufValue::Bool(self.read_bool()?),
            String => GgufValue::String(self.read_string()?),
            Uint64 => GgufValue::Uint64(self.read_u64()?),
            Int64 => GgufValue::Int64(self.read_i64()?),
            Float64 => GgufValue::Float64(self.read_f64()?),
            Array => return Err("nested arrays are not valid GGUF values".to_string()),
        })
    }

    fn tell(&mut self) -> Result<u64, String> {
        self.inner
            .stream_position()
            .map_err(|e| format!("failed to get stream position: {e}"))
    }

    fn seek_to(&mut self, pos: u64) -> Result<(), String> {
        self.inner
            .seek(SeekFrom::Start(pos))
            .map_err(|e| format!("failed to seek to offset {pos}: {e}"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------
// parsing
// ---------------------------------------------------------------------

fn gguf_init_from_reader<R: Read + Seek>(r: &mut GgufReader<R>) -> Result<GgufCtx, String> {
    // magic
    let magic = r.read_bytes(4)?;
    if magic.as_slice() != GGUF_MAGIC {
        let printable: String = magic
            .iter()
            .map(|&b| if b.is_ascii_graphic() { b as char } else { '?' })
            .collect();
        return Err(format!(
            "invalid magic characters: '{printable}', expected 'GGUF'"
        ));
    }

    // header
    let version = r.read_u32()?;
    if version == 0 {
        return Err(format!("bad GGUF version: {version}"));
    }
    if version & 0x0000_FFFF == 0 {
        return Err(format!(
            "failed to load model: this GGUF file version {version} is extremely large, \
             is there a mismatch between the host and model endianness?"
        ));
    }
    if version == 1 {
        return Err("GGUFv1 is no longer supported, please use a more up-to-date version".to_string());
    }
    if version > GGUF_VERSION {
        return Err(format!(
            "this GGUF file is version {version} but this software only supports up to version {GGUF_VERSION}"
        ));
    }

    let n_tensors = r.read_u64()?;
    let n_tensors = usize::try_from(n_tensors)
        .map_err(|_| format!("tensor count {n_tensors} does not fit in usize"))?;

    let n_kv = r.read_u64()?;
    let n_kv = usize::try_from(n_kv).map_err(|_| format!("kv count {n_kv} does not fit in usize"))?;

    // kv pairs
    let mut kv: Vec<(String, GgufValue)> = Vec::with_capacity(n_kv);
    for i in 0..n_kv {
        let key = r.read_string()?;
        if key.is_empty() {
            return Err(format!("key {i} is empty"));
        }
        if kv.iter().any(|(k, _)| k == &key) {
            return Err(format!("duplicate key '{key}'"));
        }

        let mut value_type = GgufValueType::from_id(r.read_u32()?)?;
        let is_array = value_type == GgufValueType::Array;
        let n: u64 = if is_array {
            value_type = GgufValueType::from_id(r.read_u32()?)?;
            r.read_u64()?
        } else {
            1
        };
        let n = usize::try_from(n).map_err(|_| format!("array length {n} does not fit in usize"))?;

        let value = if is_array {
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                items.push(r.read_value(value_type)?);
            }
            GgufValue::Array(items)
        } else {
            r.read_value(value_type)?
        };

        kv.push((key, value));
    }

    let alignment = match kv.iter().find(|(k, _)| k == "general.alignment").map(|(_, v)| v) {
        Some(GgufValue::Uint32(v)) => *v as usize,
        Some(_) => return Err("key 'general.alignment' does not have type u32".to_string()),
        None => GGUF_DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        return Err(format!("alignment {alignment} is not a power of 2"));
    }

    // tensor info
    let mut tensor_infos: Vec<GgufTensorInfo> = Vec::with_capacity(n_tensors);
    for i in 0..n_tensors {
        let name = r.read_string()?;
        if name.len() >= GGML_MAX_NAME {
            return Err(format!(
                "tensor name {i} is too long: {} >= {GGML_MAX_NAME}",
                name.len()
            ));
        }
        if tensor_infos.iter().any(|t| t.name == name) {
            return Err(format!("duplicate tensor name '{name}'"));
        }

        let n_dims = r.read_u32()?;
        if n_dims as usize > GGML_MAX_DIMS {
            return Err(format!(
                "tensor '{name}' has invalid number of dimensions: {n_dims} > {GGML_MAX_DIMS}"
            ));
        }

        let mut ne = [1i64; GGML_MAX_DIMS];
        for (j, ne_j) in ne.iter_mut().enumerate().take(GGML_MAX_DIMS) {
            if j < n_dims as usize {
                *ne_j = r.read_u64()? as i64;
            }
            if *ne_j < 0 {
                return Err(format!(
                    "tensor '{name}' dimension {j} has invalid number of elements: {ne_j} < 0"
                ));
            }
        }

        if i64::MAX / ne[1] <= ne[0]
            || i64::MAX / ne[2] <= ne[0] * ne[1]
            || i64::MAX / ne[3] <= ne[0] * ne[1] * ne[2]
        {
            return Err(format!(
                "total number of elements in tensor '{name}' with shape ({}, {}, {}, {}) is >= {}",
                ne[0], ne[1], ne[2], ne[3], i64::MAX
            ));
        }

        let dtype = GgmlType::from_id(r.read_u32()?)
            .map_err(|e| format!("tensor '{name}' has invalid ggml type: {e}"))?;
        let type_size = dtype.type_size();
        let block_size = dtype.block_size();

        if ne[0] % block_size != 0 {
            return Err(format!(
                "tensor '{name}' of type {dtype:?} has {} elements per row, not a multiple of block size ({block_size})",
                ne[0]
            ));
        }

        let mut nb = [0u64; GGML_MAX_DIMS];
        nb[0] = type_size;
        nb[1] = nb[0] * (ne[0] / block_size) as u64;
        for j in 2..GGML_MAX_DIMS {
            nb[j] = nb[j - 1] * ne[j - 1] as u64;
        }

        let offset = r.read_u64()?;

        tensor_infos.push(GgufTensorInfo {
            name,
            ne,
            dtype,
            nb,
            offset,
        });
    }

    // data section must start aligned; account for any padding
    if !tensor_infos.is_empty() {
        let padded = pad(r.tell()?, alignment as u64);
        r.seek_to(padded)?;
    }
    let data_offset = r.tell()?;

    // total size of the data section, taking alignment into account
    let mut data_size: u64 = 0;
    for ti in &tensor_infos {
        if ti.offset != data_size {
            return Err(format!(
                "tensor '{}' has offset {}, expected {data_size}",
                ti.name, ti.offset
            ));
        }
        let padded_size = pad(ti.nbytes(), alignment as u64);
        data_size = data_size
            .checked_add(padded_size)
            .ok_or_else(|| format!("tensor '{}' size overflow", ti.name))?;
    }

    Ok(GgufCtx {
        version,
        kv,
        tensor_infos,
        alignment,
        data_offset,
        data_size,
    })
}

pub fn gguf_init_from_file(path: &str) -> Result<GgufCtx, String> {
    let file = File::open(path).map_err(|e| format!("error opening file '{path}': {e}"))?;
    let mut reader = GgufReader::new(file);
    gguf_init_from_reader(&mut reader).map_err(|e| format!("'{path}': {e}"))
}