// tensor.data points at the start of the tensor's element data, either
// owned by an AcuteArena (from_arena/from_reader) or leaked out of a Vec
// (new/from_nested, for arena-less scratch/test tensors, see the note on
// new). tensor.data is a non contigous view.
// option to make it continous is possible with Tensor.contiguous()
// when transposing, permuting is done

use std::fmt;
use std::io::Read;
use std::ops::{Add, Div, Mul, Sub};
use std::ptr;

use crate::acute::AcuteArena;

pub trait TensorElement:
    Copy + Default + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
        fn zero() -> Self;
}

impl TensorElement for f32 {
    fn zero() -> Self { 0.0 }
}
impl TensorElement for f64 {
    fn zero() -> Self { 0.0 }
}

pub struct Tensor<T: TensorElement> {
    // points at the first element of this tensor's data. never freed by
    // Tensor itself, arena-backed tensors get reclaimed when the arena is
    // dropped, leaked ones live until the process exits
    data: *mut T,
    len: usize,
    shape: Box<[usize]>,
    /// needed jump to move to next dim
    stride: Box<[usize]>,

    // TODO: add Q, K, V. store offset for each, all will be stored inside data
}



fn validate_shape(shape: &[usize], data_len: usize) {
    assert!(!shape.is_empty());

    let expected: usize = shape.iter().product();
    if expected != data_len {
        panic!("Invalid shape for Tensor: shape {:?} implies {} elements, data has {}", shape, expected, data_len);
    }
}

fn is_valid_permutation(axes: &[usize], n: usize) -> bool {
    let mut seen = vec![false; n];
    for &a in axes {
        if a >= n || seen[a] { return false; }
        seen[a] = true;
    }
    true
}

// A tree of arbitrarily deep nested Vec`s, used only as the
// path for building a Tensor from a nested literal of unknown rank
#[derive(Debug)]
pub enum NestedVec<T: TensorElement> {
    Value(T),
    List(Vec<NestedVec<T>>),
}

impl<T: TensorElement> NestedVec<T> {
    fn flatten(&self, capacity: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(capacity);
        self.flatten_into(&mut out);
        out
    }

    fn flatten_into(&self, out: &mut Vec<T>) {
        match self {
            NestedVec::Value(v) => out.push(*v),
            NestedVec::List(items) => {
                for item in items {
                    item.flatten_into(out);
                }
            }
        }
    }
}

impl<T: TensorElement> From<T> for NestedVec<T> {
    fn from(v: T) -> Self {
        NestedVec::Value(v)
    }
}

impl<T: TensorElement, U: Into<NestedVec<T>>> From<Vec<U>> for NestedVec<T> {
    fn from(v: Vec<U>) -> Self {
        NestedVec::List(v.into_iter().map(Into::into).collect())
    }
}

impl<T, U, const N: usize> From<[U; N]> for NestedVec<T>
where
    T: TensorElement,
    U: Into<NestedVec<T>>,
{
    fn from(arr: [U; N]) -> Self {
        NestedVec::List(arr.into_iter().map(Into::into).collect())
    }
}


impl<T: TensorElement> Tensor<T> {

    // Fast path: data is already a flat buffer, moved in directly. no arena
    // around yet so the Vec's storage gets leaked into a raw pointer instead
    // of copied, cheap but never reclaimed. exists for scratch/test tensors
    // that don't have an arena to live in, once ops.rs allocates its
    // outputs from a scratch arena too, prefer from_arena over this
    pub fn new(data: Vec<T>, shape: impl Into<Box<[usize]>>, stride: Option<Box<[usize]>>) -> Tensor<T> {
        let shape = shape.into();
        validate_shape(&shape, data.len());

        let (data, len) = Self::leak(data);

        let mut new = Tensor {
            data,
            len,
            shape,
            stride: match stride {
                        None => Box::new([]),
                        Some(s) => s,
                    }
        };

        if new.stride.len() == 0 { new.update_stride(); }
        new
    }

    // Convenience path: build from a nested Vec literal of any rank
    // (`Vec<T>`, `Vec<Vec<T>>`, `Vec<Vec<Vec<T>>>`, ... unbounded depth).
    // Costs one flattening pass — not the hot path, use `new` for that.
    pub fn from_nested(data: impl Into<NestedVec<T>>, shape: impl Into<Box<[usize]>>) -> Tensor<T> {
        let shape = shape.into();
        let expected: usize = shape.iter().product();
        let flat = data.into().flatten(expected);
        Self::new(flat, shape, None)
    }

    // reserves space for data.len() elements inside arena and copies data
    // into it. memory is owned by arena, not by this Tensor, stays valid
    // exactly as long as arena does
    pub fn from_arena(
        arena: &mut AcuteArena,
        data: &[T],
        shape: impl Into<Box<[usize]>>,
        stride: Option<Box<[usize]>>,
    ) -> Result<Tensor<T>, String> {
        let shape = shape.into();
        validate_shape(&shape, data.len());

        let ptr = arena.alloc_tensor_data::<T>(data.len())?;
        // ptr was just carved out of the arena with room for exactly
        // data.len() elements, nothing else aliases it yet
        unsafe { ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()); }

        let mut new = Tensor {
            data: ptr,
            len: data.len(),
            shape,
            stride: stride.unwrap_or_default(),
        };
        if new.stride.is_empty() { new.update_stride(); }
        Ok(new)
    }

    // reads n_elements values of T straight from reader into freshly
    // arena-allocated space, no intermediate Vec, file bytes land directly
    // where the tensor will live. needs T's in-memory layout to match the
    // reader's byte layout (true for f32/f64 off a little-endian source,
    // e.g. a GGUF tensor data section, on a little-endian host)
    pub fn from_reader<R: Read>(
        arena: &mut AcuteArena,
        reader: &mut R,
        n_elements: usize,
        shape: impl Into<Box<[usize]>>,
    ) -> Result<Tensor<T>, String> {
        let shape = shape.into();
        validate_shape(&shape, n_elements);

        let ptr = arena.alloc_tensor_data::<T>(n_elements)?;
        let byte_len = n_elements * size_of::<T>();

        // ptr was just carved out of the arena with room for byte_len
        // bytes, nothing else aliases it yet
        let buf = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, byte_len) };
        reader.read_exact(buf).map_err(|e| format!("failed reading tensor data: {e}"))?;

        let mut new = Tensor { data: ptr, len: n_elements, shape, stride: Box::new([]) };
        new.update_stride();
        Ok(new)
    }

    fn leak(data: Vec<T>) -> (*mut T, usize) {
        let len = data.len();
        (Box::leak(data.into_boxed_slice()).as_mut_ptr(), len)
    }

    pub fn len(&self) -> usize { self.len }

    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn as_slice(&self) -> &[T] {
        // data always points at len valid, initialized T's, every
        // constructor establishes that
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // see as_slice, &mut self guarantees no other live view
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
    }

    pub fn shape(&self) -> &[usize] { &self.shape }

    pub fn stride(&self) -> &[usize] { &self.stride }

    // shape -> [1, 2, 3, ..., k]
    // stride -> [k!, ..., k * (k-1), k, 1] (assuming shape increases by 1)
    // when this happens is because the shape is contiguous
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1;
        for i in (0..self.shape.len()).rev() {
            if self.shape[i] <= 1 { continue; }
            if self.stride[i] != expected { return false; }
            expected *= self.shape[i];
        }
        true
    }

    /**
    * self.data changed. previous none contiguous form is lost.
    */
    pub fn make_contiguous(&mut self) {
        if self.is_contiguous() { return; }

        let total: usize = self.shape.iter().product();
        let mut data = Vec::with_capacity(total);
        let mut idx = vec![0usize; self.shape.len()];

        for _ in 0..total  {
            //row * stride[] + col * stride[]...
            let offset: usize = idx.iter().zip(self.stride.iter())
                                    .map(|(&i, &s)| i * s).sum();
            data.push(self.as_slice()[offset]);
            for d in (0..idx.len()).rev() {
                idx[d] += 1;
                if idx[d] < self.shape[d] { break; }
                idx[d] = 0;
            }

        }
        // same leak-on-replace tradeoff as new, the old region (arena- or
        // leak-owned) just gets abandoned, nothing here does per-tensor free
        let (data, len) = Self::leak(data);
        self.data = data;
        self.len = len;
        self.update_stride();
    }

    pub fn reshape(&mut self, shape: Box<[usize]>) {
        validate_shape(&shape, self.len);
        self.shape = shape;
        self.update_stride();
    }

    fn update_stride(&mut self) {
        let n = self.shape.len();
        let mut stride = vec![1usize; n];

        for i in (0..n-1).rev() {
            stride[i] = stride[i+1] * self.shape[i+1];
        }
        self.stride = stride.into_boxed_slice();
    }

    // TODO: add optimizations
    pub fn transpose(&mut self,dim0: usize, dim1: usize) {
        let mut axes: Vec<usize> = (0..self.shape.len()).collect();
        axes.swap(dim0, dim1);
        self.permute(&axes);
    }

    pub fn permute(&mut self, axes: &[usize]) {
        assert_eq!(axes.len(), self.shape.len(), "axes length must match tensor rank");
        debug_assert!(is_valid_permutation(axes, self.shape.len()));

        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let new_stride: Vec<usize> = axes.iter().map(|&a| self.stride[a]).collect();
        self.shape = new_shape.into_boxed_slice();
        self.stride = new_stride.into_boxed_slice();
    }
}

impl<T: TensorElement + fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.as_slice())
            .field("shape", &self.shape)
            .field("stride", &self.stride)
            .finish()
    }
}



#[cfg(test)]
#[path = "../tests/unit/tensor_test.rs"]
mod tests;
