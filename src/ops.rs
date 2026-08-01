use crate::tensor::TensorElement;
use crate::Tensor;
use smallvec::SmallVec;

// TODO: allow scalar + tensor ops
// TODO: if one operand is non-contiguous, make it contiguous before adding


// TODO: add more checks 
// + in the future make proper error handling, not inside a function like this with repeating string literals
fn invalid_add<T: TensorElement>(t1: &Tensor<T>, t2: &Tensor<T>) -> (bool, &'static str) {
    if t1.shape() != t2.shape() {
        return (false, "\"[acute]\": shapes are different, unable to add them together");
    }
    (true, "") // compiler usually reuses the same empty string literal everywhere in the program ...
}

// add op return None if the shapes don't match
//
pub fn add<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Result<Tensor<T>,&'static str> {
    let invalid_op = invalid_add(t1, t2);
    if invalid_op.0 == false { return Err(invalid_op.1); }

    if t1.shape() == t2.shape() && t1.stride() == t2.stride() {
        return Ok(contiguous_add(t1, t2));
    } else if !t1.is_contiguous() || !t2.is_contiguous() {
        return Ok(strided_add(t1, t2));
    }
    Err("*** Unknown error adding elements ***")
}

// Requires both inputs to already be contiguous
// it reads data directly instead of walking stride
fn contiguous_add<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T> {
    debug_assert_eq!(
        t1.stride(), t2.stride(),
        "contiguous_add requires both operands to share the same physical layout"
    );

    let data: Vec<T> = t1
        .data
        .iter()
        .zip(t2.data.iter())
        .map(|(&a, &b)| a + b)
        .collect();
 
    Tensor::new(
        data,
        t1.shape().to_vec().into_boxed_slice(),
        Some(t1.stride().to_vec().into_boxed_slice())
    )
}

// TODO: for t3.stride(), see which one(t1 or t2) will work more in combination with t3
// if we choose the stride of the tensor that will be more in contanct with t3, we will end up in the fast path{contiguous_add()} more often
fn strided_add<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T> {
    let total = t1.shape().iter().product();
    let mut data = vec![T::default(); total];
    let shape = &t1.shape(); // t2 shape == t1 shape
    // number of dimensions very is small, this way we store it on the stack
    let mut idx: SmallVec<[usize; 8]>  = SmallVec::from_elem(0usize, t1.shape().len());
    for _ in 0..total  {
        // row * stride[] + col * stride[]...
        let offset1: usize = idx.iter().zip(t1.stride().iter())
                                .map(|(&i, &s)| i * s).sum();
        let offset2: usize = idx.iter().zip(t2.stride().iter())
                                .map(|(&i, &s)| i * s).sum();

        data[offset1] = t1.data[offset1] + t2.data[offset2];
        for d in (0..idx.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] { break; }
            idx[d] = 0;
        }
    }
    Tensor::new(
        data,
        shape.to_vec().into_boxed_slice(),
        Some(t1.stride().to_vec().into_boxed_slice())
    )
}



pub fn sub<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Result<Tensor<T>,&'static str> {
    let invalid_op = invalid_add(t1, t2);
    if invalid_op.0 == false { return Err(invalid_op.1); }
    
    if t1.shape() == t2.shape() && t1.stride() == t2.stride() {
        return Ok(contiguous_sub(t1, t2));
    } else if !t1.is_contiguous() || !t2.is_contiguous() {
        return Ok(strided_sub(t1, t2));
    }
    Err("*** Unknown error subtracting elements ***")
}

// Requires both inputs to already be contiguous
// it reads data directly instead of walking stride
fn contiguous_sub<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T> {
    debug_assert_eq!(
        t1.stride(), t2.stride(),
        "contiguous_sub requires both operands to share the same physical layout"
    );

    let data: Vec<T> = t1
        .data
        .iter()
        .zip(t2.data.iter())
        .map(|(&a, &b)| a - b)
        .collect();
 
    Tensor::new(
        data,
        t1.shape().to_vec().into_boxed_slice(),
        Some(t1.stride().to_vec().into_boxed_slice())
    )
}

// TODO: for t3.stride(), see which one(t1 or t2) will work more in combination with t3
// if we choose the stride of the tensor that will be more in contanct with t3, we will end up in the fast path{contiguous_add()} more often
fn strided_sub<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T> {
    let total = t1.shape().iter().product();
    let mut data = vec![T::default(); total];
    let shape = &t1.shape(); // t2 shape == t1 shape
    // number of dimensions very is small, this way we store it on the stack
    let mut idx: SmallVec<[usize; 8]>  = SmallVec::from_elem(0usize, t1.shape().len());
    for _ in 0..total  {
        // row * stride[] + col * stride[]...
        let offset1: usize = idx.iter().zip(t1.stride().iter())
                                .map(|(&i, &s)| i * s).sum();
        let offset2: usize = idx.iter().zip(t2.stride().iter())
                                .map(|(&i, &s)| i * s).sum();

        data[offset1] = t1.data[offset1] - t2.data[offset2];
        for d in (0..idx.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] { break; }
            idx[d] = 0;
        }
    }
    Tensor::new(
        data,
        shape.to_vec().into_boxed_slice(),
        Some(t1.stride().to_vec().into_boxed_slice())
    )
}

#[cfg(test)]
#[path = "../tests/unit/ops_test.rs"]
mod tests;