use acute_engine::tensor::Tensor;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_construction");

    for &(a, b, d) in &[(10, 10, 10), (50, 50, 40), (100, 100, 50)] {
        let n = a * b * d;

        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let nested: Vec<Vec<Vec<f32>>> = (0..a)
            .map(|_| (0..b).map(|_| (0..d).map(|i| i as f32).collect()).collect())
            .collect();

        
        group.bench_with_input(BenchmarkId::new("new_flat", n), &flat, |bencher, data| {
            bencher.iter_batched(
                || data.clone(),
                |owned| black_box(Tensor::new(owned, vec![a, b, d])),
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("from_nested", n), &nested, |bencher, data| {
            bencher.iter_batched(
                || data.clone(),
                |owned| black_box(Tensor::from_nested(owned, vec![a, b, d])),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_construction);
criterion_main!(benches);
