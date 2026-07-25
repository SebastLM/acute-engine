mod ops;
mod tensor;

use std::println;

use tensor::Tensor;


fn main() {
    let data = vec![1.00, 3.232, 4.343434, 2.3434325456, 4.298769, 8.359684, 1.396739, 5.36890984];
    let mut tensor1 = Tensor::new(data, vec![2, 4].into_boxed_slice());
    tensor1.transpose();
    println!("{:?}", tensor1.shape());
    println!("{:?}", tensor1.data);
    tensor1.transpose();
    println!("{:?}", tensor1.shape());
}
