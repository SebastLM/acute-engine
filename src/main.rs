mod ops;
mod tensor;

use std::println;

use tensor::Tensor;


fn main() {
    let data: Vec<f32> = vec![1.00, 3.232, 4.343434, 2.3434325456, 4.298769, 8.359684, 1.396739, 5.36890984];
    let mut tensor1 = Tensor::new(data, vec![2, 4].into_boxed_slice());
    tensor1.transpose(1, 0);
    println!("{:?}", tensor1.shape());
    tensor1.make_contiguous();
    println!("{:?}", tensor1.data);
    tensor1.transpose(1, 0);
    println!("{:?}", tensor1.shape());
    tensor1.make_contiguous();
    println!("{:?}", tensor1.data);

    println!("\n\n\n\n\n\n");

    let data2: Vec<f32> = vec![1.0, 4.298769, 3.232, 8.359684, 4.343434, 1.396739, 2.3434325456, 5.36890984];
    tensor1 = Tensor::new(data2, vec![4, 2].into_boxed_slice());
    tensor1.transpose(1, 0);
    println!("{:?}", tensor1.shape());
    tensor1.make_contiguous();
    println!("{:?}", tensor1.data);
    tensor1.transpose(1, 0);
    println!("{:?}", tensor1.shape());
    tensor1.make_contiguous();
    println!("{:?}", tensor1.data);

}
