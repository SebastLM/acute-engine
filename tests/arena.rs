use acute_engine::acute::AcuteArena;
use acute_engine::Tensor;
use std::io::Cursor;

#[test]
fn arena_alloc_and_reader_roundtrip() {
    let mut arena = AcuteArena::new(4096).expect("mmap failed");

    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut cursor = Cursor::new(bytes);

    let tensor: Tensor<f32> =
        Tensor::from_reader(&mut arena, &mut cursor, 4, vec![4].into_boxed_slice())
            .expect("read into arena failed");

    assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(arena.used(), 16);
    assert_eq!(arena.n_objects(), 1);
}
