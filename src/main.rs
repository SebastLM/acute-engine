mod ops;
mod tensor;

use std::{println, vec};

use acute_engine::ops::add;
use tensor::Tensor;



fn main() {
    let data: Vec<f32> = vec![1.00, 3.232, 4.343434, 2.3434325456, 4.298769, 8.359684, 1.396739, 5.36890984];
    let mut tensor1 = Tensor::new(data, vec![2, 4].into_boxed_slice(), None);
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
    tensor1 = Tensor::new(data2, vec![4, 2].into_boxed_slice(), None);
    tensor1.transpose(1, 0);
    println!("{:?}", tensor1.shape());
    tensor1.make_contiguous();
    println!("{:?}", tensor1.data);
    tensor1.transpose(1, 0);
    println!("{:?}", tensor1.shape());
    tensor1.make_contiguous();
    println!("{:?}", tensor1.data);
    

    println!("\n\n\n\n\n\n\n\n\n\n\n\n\n 
                *** contiguous add ***
            ");
    let data: Vec<f32> = vec![1.0,2.0,3.0, 4.0,5.0,6.0];
    let t1 = Tensor::new(data.clone(), vec![3,2], None);
    let t2 = Tensor::new(data, vec![3,2], None);

    let t3 = match ops::add(&t1, &t2)  {
        Ok(r) => r,
        Err(s) => {
            println!("{s}");
            Tensor::new(vec![], vec![0], None)
        },
    };
    println!("{:?}", t1.data);
    println!("{:?}", t3.data);


    println!("\n\n\n\n\n\n\n\n\n\n\n\n\n 
                *** non-contiguous add ***
            ");

    let data: Vec<[[f64; 4]; 3]> = vec![
                    [
                        [ 1.0,  2.0,  3.0,  4.0],
                        [ 5.0,  6.0,  7.0,  8.0],
                        [ 9.0, 10.0, 11.0, 12.0],
                    ],
                    [
                        [13.0, 14.0, 15.0, 16.0],
                        [17.0, 18.0, 19.0, 20.0],
                        [21.0, 22.0, 23.0, 24.0],
                    ],
                ];
    let shape: [usize; 3]  = [2, 3, 4];

    let mut t1 = Tensor::from_nested(data.clone(), shape);
    t1.transpose(0, 1); // shape [2,3,4] -> [3,2,4]; stride no longer canonical, so t1 is non-contiguous

    let flat: Vec<f64> = data.into_iter().flatten().flatten().collect();
    let t2 = Tensor::new(flat, vec![3, 2, 4], None); // freshly built -> contiguous, same shape as t1 post-transpose

    println!("t1 contiguous? {}  shape {:?}  stride {:?}", t1.is_contiguous(), t1.shape(), t1.stride());
    println!("t2 contiguous? {}  shape {:?}  stride {:?}", t2.is_contiguous(), t2.shape(), t2.stride());

    let t3 = match ops::add(&t1, &t2)  {
        Ok(r) => r,
        Err(s) => {
            println!("{s}");
            Tensor::new(vec![], vec![0], None)
        },
    };
    println!("{:?}", t1.data);
    println!("{:?}", t3.data);

}
