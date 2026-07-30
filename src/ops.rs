use crate::tensor::TensorElement;
use crate::Tensor;

// TODO: allow scalar + tensor ops
// TODO: if one operand is non-contiguous, make it contiguous before adding

fn same_shape<T: TensorElement>(t1: &Tensor<T>, t2: &Tensor<T>) -> bool {
    t1.shape() == t2.shape()
}

// TODO: pub fn add() splits into contiguous and non-contiguous


// Elementwise add. Returns None if the shapes don't match.
// Requires both inputs to already be contiguous — it reads `data` directly
// instead of walking stride
fn contiguous_add<T: TensorElement>(t1: &Tensor<T>, t2: &Tensor<T>) -> Option<Tensor<T>> {
    if !same_shape(t1, t2) {
        return None;
    }
    debug_assert!(
        t1.is_contiguous() && t2.is_contiguous(),
        "contiguous_add requires contiguous inputs"
    );

    let data: Vec<T> = t1
        .data
        .iter()
        .zip(t2.data.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    Some(Tensor::new(data, t1.shape().to_vec().into_boxed_slice(), None))
}
