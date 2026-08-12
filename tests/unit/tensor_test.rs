use super::*;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape.to_vec().into_boxed_slice(), None)
}

// --- construction & validation ---

#[test]
fn new_computes_row_major_stride() {
    let tensor = t(vec![0.0; 24], &[2, 3, 4]);
    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.stride(), &[12, 4, 1]);
}

#[test]
#[should_panic(expected = "Invalid shape for Tensor")]
fn new_panics_when_shape_does_not_match_data_len() {
    t(vec![1.0, 2.0, 3.0], &[2, 2]);
}

#[test]
#[should_panic]
fn new_panics_on_empty_shape() {
    t(vec![1.0], &[]);
}

#[test]
fn shape_and_stride_getters_return_slices() {
    let tensor = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let shape: &[usize] = tensor.shape();
    let stride: &[usize] = tensor.stride();
    assert_eq!(shape, &[2, 2]);
    assert_eq!(stride, &[2, 1]);
}

// --- reshape ---

#[test]
fn reshape_recomputes_stride_for_new_shape() {
    let mut tensor = t((0..24).map(|x| x as f32).collect(), &[2, 3, 4]);
    tensor.reshape(vec![4, 6].into_boxed_slice());
    assert_eq!(tensor.shape(), &[4, 6]);
    assert_eq!(tensor.stride(), &[6, 1]);
}

#[test]
#[should_panic(expected = "Invalid shape for Tensor")]
fn reshape_panics_on_element_count_mismatch() {
    let mut tensor = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    tensor.reshape(vec![3, 3].into_boxed_slice());
}

// --- is_valid_permutation ---

#[test]
fn is_valid_permutation_accepts_a_true_permutation() {
    assert!(is_valid_permutation(&[2, 0, 1], 3));
}

#[test]
fn is_valid_permutation_rejects_duplicate_axis() {
    assert!(!is_valid_permutation(&[0, 0, 2], 3));
}

#[test]
fn is_valid_permutation_rejects_out_of_range_axis() {
    assert!(!is_valid_permutation(&[0, 1, 3], 3));
}

// --- is_contiguous ---

#[test]
fn fresh_tensor_is_contiguous() {
    let tensor = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(tensor.is_contiguous());
}

// --- permute / transpose ---
//
// These encode the spec discussed for `permute`: reordering axes must reuse
// the *existing* strides in the new order, not recompute canonical strides
// for the new shape. `permute` currently calls `update_stride()` after
// reordering `shape`, which recomputes fresh canonical strides instead of
// permuting `self.stride` alongside it -- so the two tests below are
// expected to FAIL until that's fixed. They're left in as the regression
// check for when it is.

#[test]
fn permute_reorders_stride_in_lockstep_with_shape() {
    let mut tensor = t(vec![0.0; 8], &[2, 4]); // stride starts as [4, 1]
    tensor.permute(&[1, 0]);
    assert_eq!(tensor.shape(), &[4, 2]);
    assert_eq!(tensor.stride(), &[1, 4]); // old stride, reordered -- not canonical [2, 1]
}

#[test]
fn permute_preserves_element_correspondence() {
    // 2x4 row-major: data[i*4 + j] is logical element (i, j)
    let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
    let mut tensor = t(data.clone(), &[2, 4]);
    tensor.permute(&[1, 0]); // transpose to logical [4, 2]

    let stride = tensor.stride().to_vec();
    for i in 0..2 {
        for j in 0..4 {
            // after permute, logical (j, i) must still read the original (i, j)
            let offset = j * stride[0] + i * stride[1];
            assert_eq!(
                tensor.as_slice()[offset], data[i * 4 + j],
                "logical ({j},{i}) should read original ({i},{j})"
            );
        }
    }
}

#[test]
#[should_panic(expected = "axes length must match tensor rank")]
fn permute_panics_when_axes_len_mismatches_rank() {
    let mut tensor = t(vec![0.0; 8], &[2, 4]);
    tensor.permute(&[0, 1, 2]);
}

// debug_assert! only fires in debug builds -- this test passes under plain
// `cargo test` but would silently no-op under `cargo test --release`.
#[test]
#[should_panic]
fn permute_debug_asserts_on_invalid_permutation() {
    let mut tensor = t(vec![0.0; 8], &[2, 4]);
    tensor.permute(&[0, 0]);
}

#[test]
fn transpose_swaps_only_the_requested_axes() {
    let mut tensor = t(vec![0.0; 24], &[2, 3, 4]);
    tensor.transpose(0, 2);
    assert_eq!(tensor.shape(), &[4, 3, 2]);
}

// --- make_contiguous ---

#[test]
fn make_contiguous_is_a_no_op_when_already_dense() {
    let mut tensor = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let before = tensor.as_slice().to_vec();
    tensor.make_contiguous();
    assert_eq!(tensor.as_slice(), (before).as_slice());
    assert!(tensor.is_contiguous());
}

// Depends on `permute` correctly permuting strides (see note above) --
// expected to FAIL until that's fixed.
#[test]
fn make_contiguous_materializes_a_permuted_tensor() {
    let data: Vec<f32> = (0..8).map(|x| x as f32).collect(); // 2x4
    let mut tensor = t(data, &[2, 4]);
    tensor.permute(&[1, 0]); // logical 4x2 transpose

    tensor.make_contiguous();

    assert!(tensor.is_contiguous());
    assert_eq!(tensor.stride(), &[2, 1]);
    // row-major dense 4x2 transpose of [[0,1,2,3],[4,5,6,7]]
    assert_eq!(tensor.as_slice(), (vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]).as_slice());
}
