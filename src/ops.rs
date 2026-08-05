use crate::tensor::TensorElement;
use crate::Tensor;
use smallvec::SmallVec;

// TODO: allow scalar + tensor ops
// TODO: if one operand is non-contiguous, make it contiguous before adding


// TODO: add more checks 
// + in the future make proper error handling, not inside a function like this with repeating string literals
fn invalid_binop<T: TensorElement>(t1: &Tensor<T>, t2: &Tensor<T>) -> (bool, &'static str) {
    if t1.shape() != t2.shape() {
        return (false, "\"[acute]\": shapes are different, unable to add them together");
    }
    (true, "") // compiler usually reuses the same empty string literal everywhere in the program ...
}

// add op return None if the shapes don't match
//
pub fn add<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Result<Tensor<T>,&'static str> {
    let invalid_op = invalid_binop(t1, t2);
    if invalid_op.0 == false { return Err(invalid_op.1); }

    if t1.shape() == t2.shape() && t1.stride() == t2.stride() {
        return Ok(contiguous_add(t1, t2));
    } else if !t1.is_contiguous() || !t2.is_contiguous() {
        return Ok(strided_add(t1, t2));
    }
    Err("\"[acute]\": incompatible elements for add")
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
    let invalid_op = invalid_binop(t1, t2);
    if invalid_op.0 == false { return Err(invalid_op.1); }
    
    if t1.shape() == t2.shape() && t1.stride() == t2.stride() {
        return Ok(contiguous_sub(t1, t2));
    } else if !t1.is_contiguous() || !t2.is_contiguous() {
        return Ok(strided_sub(t1, t2));
    }
    Err("\"[acute]\": incompatible elements for sub")
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



fn mul<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Result<Tensor<T>,&'static str> {
    let len  = t1.shape().len();
    if len == 1 && t1.shape() == t2.shape() { return Ok(ele_w_d1_mul(t1, t2)); } // currently only element wise op's
    if t1.shape().len() == t2.shape().len() {
        if len >= 3 && t1.shape()[0] == t2.shape()[0] {
            if t1.shape()[len - 1] == t2.shape()[len - 2] {
                return Ok(strided_mul(t1, t2));
            }
        } else if len == 2 {
            if t1.shape()[1] == t2.shape()[0] {
                return Ok(strided_mul(t1, t2));
            }

        }
    }
    Err("\"[acute]\": incompatible elements for mul")
}


// TODO: dot product multiplication


// element_wise multiplication for matrices with 1 dimension
fn ele_w_d1_mul<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T> {
    let len = t1.data.len();
    let mut data =  vec![T::default(); len];
    for i in 0..len {
        data[i] = t1.data[i] * t2.data[i];
    }
    Tensor::new(
        data,
        t1.shape().to_vec().into_boxed_slice(),
        Some(vec![1].into_boxed_slice())
    )
}


fn dot_d1_mul<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T> {
    let mut val: T = T::zero();
    for i in 0..t1.data.len() {
        val = val + t1.data[i] * t2.data[i];
    }
    let data =  vec![val];
    Tensor::new(
        data, // In the future we won't be storing a vector for a single val, 
              // but rather a pointer to the memory containing the actual data
        t1.shape().to_vec().into_boxed_slice(),
        Some(vec![1].into_boxed_slice())
    )
}

// element wise mul for  2+ dimensions
fn strided_mul<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T> {
    let len = t2.shape().len();
    let total = t1.shape().iter().product::<usize>() * t2.shape()[len - 1];
    let shape1 = &t1.shape();
    let shape2 = &t2.shape();

    let mut shape = shape1.to_vec().into_boxed_slice();
    shape[len - 1] = shape2[len - 1];

    let mut data = vec![T::default(); shape.iter().product()]; // batch*m*n output cells
    let mut idx1: SmallVec<[usize; 8]>  = SmallVec::from_elem(0usize, t1.shape().len());
    let mut idx2: SmallVec<[usize; 8]>  = SmallVec::from_elem(0usize, t2.shape().len());

    let mut pos = 0;
    for _ in 0..total  {
        // row * stride[] + col * stride[]...
        let offset1: usize = idx1.iter().zip(t1.stride().iter())
                                .map(|(&i, &s)| i * s).sum::<usize>()
                                + idx2[len - 2] * t1.stride()[len - 1];

        let offset2: usize = idx1.iter().zip(t2.stride().iter()).take(len - 2)
                                .map(|(&i, &s)| i * s).sum::<usize>()
                                + idx2[len - 2] * t2.stride()[len - 2]
                                + idx2[len - 1] * t2.stride()[len - 1];

        data[pos] = data[pos] + t1.data[offset1] * t2.data[offset2];

        idx2[len - 2] += 1;
        if idx2[len - 2] >= shape2[len - 2] {
            idx2[len - 2] = 0;
            pos += 1; // output matrix position calculated
            idx2[len - 1] += 1;

            if idx2[len - 1] >= shape2[len - 1] {
                idx2[len - 1] = 0;
                // finished every column for this row -> advance idx1's row (and batch dims, if any)
                for d in (0..idx1.len() - 1).rev() {
                    idx1[d] += 1;
                    if idx1[d] < shape1[d] { break; }
                    idx1[d] = 0;
                }
            }
        }
    }

    Tensor::new(
        data,
        shape,
        None
    )
}





#[cfg(test)]
#[path = "../tests/unit/ops_test.rs"]
mod tests;