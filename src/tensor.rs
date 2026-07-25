// TODO: add f64 support via generics
#[derive(Debug)]
pub struct Tensor {
    /// stored as a 1D vector
    pub data: Vec<f32>, 
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

impl Tensor {
    pub fn new(data: Vec<f32>, shape: impl Into<Box<[usize]>>) -> Tensor {
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

    // TODO: add transpose for 3D+ matrices
    pub fn transpose(&mut self) {
        let mut flat_t_matrix: Vec<f32> = Vec::new();
        for i in 0..self.shape[1] {
            flat_t_matrix.push(self.data[i]);
            for j in 1..self.shape[0] {
                flat_t_matrix.push(self.data[i + self.stride[0] * j]);
            }
        }
        self.data = flat_t_matrix;
        self.shape.reverse();
        self.update_stride();
    }
}
