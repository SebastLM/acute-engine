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
    let invalid_op = invalid_binop(t1, t2);
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


pub fn mul<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Result<Tensor<T>,&'static str> {
    let r1 = t1.shape().len();
    let r2 = t2.shape().len();

    if r1 == 1 && t1.shape() == t2.shape() {
        return Ok(ele_w_d1_mul(t1, t2)); // element-wise, 1D only
    }

    // batched matmul: t1 [..batch, m, k] x t2 [..batch, k, n] -> [..batch, m, n]
    // batch dims (everything before the trailing two) must match exactly;
    // plain 2D matmul is just the case where there are no batch dims at all.
    if r1 >= 2 && r1 == r2
        && t1.shape()[..r1 - 2] == t2.shape()[..r2 - 2]
        && t1.shape()[r1 - 1] == t2.shape()[r2 - 2]
    {
        return Ok(strided_mul(t1, t2));
    }

    Err("\"[acute]\": shapes are incompatible for multiplication")
}


// element_wise
fn ele_w_d1_mul<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T> {
    let data: Vec<T> = t1.data.iter().zip(t2.data.iter()).map(|(&a, &b)| a * b).collect();
    Tensor::new(
        data,
        t1.shape().to_vec().into_boxed_slice(),
        None
    )
}

// Batched matrix multiply. t1: [..batch, m, k], t2: [..batch, k, n] -> [..batch, m, n]
// Walks each operand's own stride directly (no densification needed first),
// so this works whether t1/t2 are contiguous or the result of transpose/permute.
fn strided_mul<T: TensorElement> (t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T> {
    let r1 = t1.shape().len();
    let r2 = t2.shape().len();
    let m = t1.shape()[r1 - 2];
    let k = t1.shape()[r1 - 1];
    let n = t2.shape()[r2 - 1];
    debug_assert_eq!(k, t2.shape()[r2 - 2], "strided_mul requires t1's last dim to match t2's second-to-last dim");

    let batch_shape = &t1.shape()[..r1 - 2];
    let batch_total: usize = batch_shape.iter().product();

    let mut out_shape: Vec<usize> = batch_shape.to_vec();
    out_shape.push(m);
    out_shape.push(n);

    let mut data = vec![T::default(); batch_total * m * n];
    let mut batch_idx: SmallVec<[usize; 8]> = SmallVec::from_elem(0usize, batch_shape.len());
    let mut pos = 0;

    for _ in 0..batch_total {
        let batch_offset1: usize = batch_idx.iter().zip(t1.stride().iter())
                                    .map(|(&b, &s)| b * s).sum();
        let batch_offset2: usize = batch_idx.iter().zip(t2.stride().iter())
                                    .map(|(&b, &s)| b * s).sum();

        for i in 0..m {
            for j in 0..n {
                let mut acc = T::default();
                for kk in 0..k {
                    let offset1 = batch_offset1 + i * t1.stride()[r1 - 2] + kk * t1.stride()[r1 - 1];
                    let offset2 = batch_offset2 + kk * t2.stride()[r2 - 2] + j * t2.stride()[r2 - 1];
                    acc = acc + t1.data[offset1] * t2.data[offset2];
                }
                data[pos] = acc;
                pos += 1;
            }
        }

        for d in (0..batch_idx.len()).rev() {
            batch_idx[d] += 1;
            if batch_idx[d] < batch_shape[d] { break; }
            batch_idx[d] = 0;
        }
    }

    // freshly built, written in canonical row-major order -> canonical stride, not an input's
    Tensor::new(data, out_shape.into_boxed_slice(), None)
}

#[cfg(test)]
#[path = "../tests/unit/ops_test.rs"]
mod tests;