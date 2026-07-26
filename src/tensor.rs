// tensor.data is a non contigous vector.
// option to make it continous is possible with Tensor.contiguous()
// when transposing, permuting is done

pub trait TensorElement: Copy + Default {}

impl TensorElement for f32 {}
impl TensorElement for f64 {}


#[derive(Debug)]
pub struct Tensor<T: TensorElement> {
    /// stored as a 1D vector
    pub data: Vec<T>, 
    shape: Box<[usize]>,
    /// needed jump to move to next dim
    stride: Box<[usize]>,
}



fn validate_shape(shape: &[usize], data_len: usize) {
    assert!(!shape.is_empty());

    let expected: usize = shape.iter().product();
    if expected != data_len {
        panic!("Invalid shape for Tensor: shape {:?} implies {} elements, data has {}", shape, expected, data_len);
    }
}

pub fn is_valid_permutation(axes: &[usize], n: usize) -> bool {
    let mut seen = vec![false; n];
    for &a in axes {
        if a >= n || seen[a] { return false; }
        seen[a] = true;
    }
    true
}


impl<T: TensorElement> Tensor<T> {

    pub fn new(data: Vec<T>, shape: impl Into<Box<[usize]>>) -> Tensor<T> {
        let shape = shape.into();
        validate_shape(&shape, data.len());

        let mut new = Tensor { 
            data,
            shape,
            stride: Box::new([]),
        };
        new.update_stride();
        new
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
        // row
        let mut idx = vec![0usize; self.shape.len()];

        for _ in 0..total  {
            //row * stride[]
            let offset: usize = idx.iter().zip(self.stride.iter())
                                    .map(|(&i, &s)| i * s).sum();
            data.push(self.data[offset]);
            for d in (0..idx.len()).rev() {
                idx[d] += 1;
                if idx[d] < self.shape[d] { break; }
                idx[d] = 0;
            }
            
        }
        self.data = data;
        self.update_stride();
    }

    pub fn reshape(&mut self, shape: Box<[usize]>) {
        validate_shape(&shape, self.data.len());
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



#[cfg(test)]
#[path = "../tests/unit/tensor_test.rs"]
mod tests;
