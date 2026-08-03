use super::*;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape.to_vec().into_boxed_slice(), None)
}

/// Independent copy of `t`, used to build a "known good" dense reference
/// tensor to check strided results against, without disturbing the original.
fn clone_tensor(t: &Tensor<f32>) -> Tensor<f32> {
    Tensor::new(
        t.data.clone(),
        t.shape().to_vec().into_boxed_slice(),
        Some(t.stride().to_vec().into_boxed_slice()),
    )
}

#[test]
fn add_rejects_mismatched_shapes() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    let err = add(&a, &b).unwrap_err();
    assert!(err.contains("shapes are different"));
}

#[test]
fn add_sums_elementwise_for_freshly_built_contiguous_tensors() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = t(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    assert_eq!(a.stride(), b.stride()); // both fresh -> identical canonical stride -> fast path

    let c = add(&a, &b).expect("shapes match");

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn add_1d_tensors() {
    let a = t(vec![1.0, 2.0, 3.0], &[3]);
    let b = t(vec![4.0, 5.0, 6.0], &[3]);

    let c = add(&a, &b).expect("shapes match");

    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn add_takes_the_strided_path_when_only_one_operand_is_non_contiguous() {
    // a: built [2,3] then transposed -> logical shape [3,2], stride [1,3] (non-contiguous)
    let mut a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    a.transpose(0, 1);
    assert!(!a.is_contiguous());

    // b: freshly built [3,2] -> contiguous, different stride than a
    let b = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert!(b.is_contiguous());
    assert_ne!(a.stride(), b.stride());

    let mut c = add(&a, &b).expect("shapes match");
    c.make_contiguous();

    // a logical view (post-transpose): [[1,4],[2,5],[3,6]]
    // b logical view:                  [[1,2],[3,4],[5,6]]
    // elementwise sum:                 [[2,6],[5,9],[8,12]]
    assert_eq!(c.shape(), &[3, 2]);
    assert_eq!(c.data, vec![2.0, 6.0, 5.0, 9.0, 8.0, 12.0]);
}

#[test]
fn add_uses_fast_path_when_both_operands_share_identical_non_canonical_stride() {
    // both built [2,3] then transposed the same way -> identical, non-canonical stride
    let mut a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    a.transpose(0, 1);
    let mut b = t(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
    b.transpose(0, 1);

    assert!(!a.is_contiguous());
    assert!(!b.is_contiguous());
    assert_eq!(a.stride(), b.stride());

    let mut c = add(&a, &b).expect("shapes match");
    c.make_contiguous();

    // a logical: [[1,4],[2,5],[3,6]], b logical: [[10,40],[20,50],[30,60]]
    assert_eq!(c.data, vec![11.0, 44.0, 22.0, 55.0, 33.0, 66.0]);
}

#[test]
fn add_strided_path_matches_dense_reference_for_a_3d_case_with_both_operands_non_contiguous() {
    // a: [2,3,4] transposed on (0,1) -> shape [3,2,4], stride [4,12,1]
    let mut a = t((1..=24).map(|v| v as f32).collect(), &[2, 3, 4]);
    a.transpose(0, 1);

    // b: [4,3,2] permuted (1,2,0) -> shape [3,2,4], stride [2,1,6] (different from a's)
    let mut b = t((1..=24).map(|v| (v * 10) as f32).collect(), &[4, 3, 2]);
    b.permute(&[1, 2, 0]);

    assert_eq!(a.shape(), b.shape());
    assert_ne!(a.stride(), b.stride());
    assert!(!a.is_contiguous());
    assert!(!b.is_contiguous());

    // independently densify both operands and add plainly, as a trusted oracle
    let mut a_dense = clone_tensor(&a);
    let mut b_dense = clone_tensor(&b);
    a_dense.make_contiguous();
    b_dense.make_contiguous();
    let expected: Vec<f32> = a_dense
        .data
        .iter()
        .zip(b_dense.data.iter())
        .map(|(&x, &y)| x + y)
        .collect();

    let mut c = add(&a, &b).expect("shapes match");
    c.make_contiguous();

    assert_eq!(c.data, expected);
}

#[test]
fn sub_rejects_mismatched_shapes() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    let err = sub(&a, &b).unwrap_err();
    assert!(err.contains("shapes are different"));
}

#[test]
fn sub_subtracts_elementwise_for_freshly_built_contiguous_tensors() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = t(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    assert_eq!(a.stride(), b.stride()); // both fresh -> identical canonical stride -> fast path

    let c = sub(&a, &b).expect("shapes match");

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data, vec![-9.0, -18.0, -27.0, -36.0]);
}

#[test]
fn sub_1d_tensors() {
    let a = t(vec![1.0, 2.0, 3.0], &[3]);
    let b = t(vec![4.0, 5.0, 6.0], &[3]);

    let c = sub(&a, &b).expect("shapes match");

    assert_eq!(c.data, vec![-3.0, -3.0, -3.0]);
}

#[test]
fn sub_preserves_operand_order_through_the_strided_path() {
    // a: built [2,3] then transposed -> logical shape [3,2], stride [1,3] (non-contiguous)
    let mut a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    a.transpose(0, 1);
    assert!(!a.is_contiguous());

    // b: freshly built [3,2] -> contiguous, different stride than a
    let b = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert!(b.is_contiguous());
    assert_ne!(a.stride(), b.stride());

    let mut c = sub(&a, &b).expect("shapes match");
    c.make_contiguous();

    // a logical view (post-transpose): [[1,4],[2,5],[3,6]]
    // b logical view:                  [[1,2],[3,4],[5,6]]
    // elementwise a - b:               [[0,2],[-1,1],[-2,0]]
    assert_eq!(c.shape(), &[3, 2]);
    assert_eq!(c.data, vec![0.0, 2.0, -1.0, 1.0, -2.0, 0.0]);
}

#[test]
fn sub_uses_fast_path_when_both_operands_share_identical_non_canonical_stride() {
    // both built [2,3] then transposed the same way -> identical, non-canonical stride
    let mut a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    a.transpose(0, 1);
    let mut b = t(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
    b.transpose(0, 1);

    assert!(!a.is_contiguous());
    assert!(!b.is_contiguous());
    assert_eq!(a.stride(), b.stride());

    let mut c = sub(&a, &b).expect("shapes match");
    c.make_contiguous();

    // a logical: [[1,4],[2,5],[3,6]], b logical: [[10,40],[20,50],[30,60]]
    assert_eq!(c.data, vec![-9.0, -36.0, -18.0, -45.0, -27.0, -54.0]);
}

#[test]
fn sub_strided_path_matches_dense_reference_for_a_3d_case_with_both_operands_non_contiguous() {
    // a: [2,3,4] transposed on (0,1) -> shape [3,2,4], stride [4,12,1]
    let mut a = t((1..=24).map(|v| v as f32).collect(), &[2, 3, 4]);
    a.transpose(0, 1);

    // b: [4,3,2] permuted (1,2,0) -> shape [3,2,4], stride [2,1,6] (different from a's)
    let mut b = t((1..=24).map(|v| (v * 10) as f32).collect(), &[4, 3, 2]);
    b.permute(&[1, 2, 0]);

    assert_eq!(a.shape(), b.shape());
    assert_ne!(a.stride(), b.stride());
    assert!(!a.is_contiguous());
    assert!(!b.is_contiguous());

    // independently densify both operands and subtract plainly, as a trusted oracle
    let mut a_dense = clone_tensor(&a);
    let mut b_dense = clone_tensor(&b);
    a_dense.make_contiguous();
    b_dense.make_contiguous();
    let expected: Vec<f32> = a_dense
        .data
        .iter()
        .zip(b_dense.data.iter())
        .map(|(&x, &y)| x - y)
        .collect();

    let mut c = sub(&a, &b).expect("shapes match");
    c.make_contiguous();

    assert_eq!(c.data, expected);
}

#[test]
fn mul_rejects_incompatible_inner_dimensions() {
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]); // last dim of a (3) != second-to-last of b (2)

    let err = mul(&a, &b).unwrap_err();
    assert!(err.contains("incompatible"));
}

#[test]
fn mul_1d_is_elementwise() {
    let a = t(vec![1.0, 2.0, 3.0], &[3]);
    let b = t(vec![4.0, 5.0, 6.0], &[3]);

    let c = mul(&a, &b).expect("shapes match");

    assert_eq!(c.shape(), &[3]);
    assert_eq!(c.data, vec![4.0, 10.0, 18.0]);
}

#[test]
fn mul_2d_matmul_for_freshly_built_contiguous_tensors() {
    // a = [[1,2,3],[4,5,6]], b = [[7,8],[9,10],[11,12]]
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

    let c = mul(&a, &b).expect("shapes compatible");

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn mul_takes_the_strided_path_when_operands_are_non_contiguous() {
    // a: built [3,2] then transposed -> logical [[1,3,5],[2,4,6]], shape [2,3], non-contiguous
    let mut a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    a.transpose(0, 1);
    assert!(!a.is_contiguous());

    // b: built [2,3] then transposed -> logical [[7,10],[8,11],[9,12]], shape [3,2], non-contiguous
    let mut b = t(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3]);
    b.transpose(0, 1);
    assert!(!b.is_contiguous());

    let c = mul(&a, &b).expect("shapes compatible");

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data, vec![76.0, 103.0, 100.0, 136.0]);
}

#[test]
fn mul_batched_3d_matmul() {
    // batch of 2: [2,3] x [3,2] -> [2,2] per batch
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 2, 3]);
    let b = t(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 3, 2]);

    let c = mul(&a, &b).expect("shapes compatible");

    assert_eq!(c.shape(), &[2, 2, 2]);
    assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0, 2.0, 3.0, 4.0, 5.0]);
}
