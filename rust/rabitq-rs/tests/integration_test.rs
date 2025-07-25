use ndarray::{Array, Array1, Array2};
use rabitq_rs::estimator::SingleCentroidEstimator;
use rabitq_rs::quantizer::{MetricType, quantize_split_single};
use rabitq_rs::rotator::Rotator;
use rabitq_rs::{BatchBinEstimator, RabitqConfig, SingleEstimator};
use rand::distr::{Distribution, Uniform};

#[test]
fn test_quantization_and_estimation() {
    const DIM: usize = 1024;
    const PADDED_DIM: usize = 1024;
    const NUM_VECTORS: usize = 100;
    const EX_BITS: usize = 4;

    // 1. Generate random vectors
    let mut rng = rand::rng();
    let unif = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let vectors: Array2<f32> = Array::from_shape_fn((NUM_VECTORS, DIM), |_| unif.sample(&mut rng));
    let query_vec: Array1<f32> = Array::from_shape_fn(DIM, |_| unif.sample(&mut rng));

    // 2. Create a rotator and rotate vectors
    let rotator = Rotator::new(DIM, PADDED_DIM).expect("Failed to create rotator");
    let mut rotated_vectors = Array2::<f32>::zeros((NUM_VECTORS, rotator.padded_dim()));
    for i in 0..NUM_VECTORS {
        let mut rotated_row = rotated_vectors.row_mut(i);
        rotator.rotate(
            vectors.row(i).as_slice().unwrap(),
            rotated_row.as_slice_mut().unwrap(),
        );
    }
    let mut rotated_query = Array1::<f32>::zeros(rotator.padded_dim());
    rotator.rotate(
        query_vec.as_slice().unwrap(),
        rotated_query.as_slice_mut().unwrap(),
    );

    // 3. Quantize with centroid at origin
    let mut centroid = vec![0.0f32; rotator.padded_dim()];
    for i in 0..rotator.padded_dim() {
        centroid[i] = unif.sample(&mut rng);
    }

    let (bin_codes, ex_codes) = quantize_split_single(
        rotated_vectors.row(0).as_slice().unwrap(),
        &centroid,
        EX_BITS,
        MetricType::L2,
        &RabitqConfig::new(),
    );

    // 4. Query
    let query = SingleEstimator::new(
        rotated_query.as_slice().unwrap(),
        rotator.padded_dim(),
        EX_BITS,
        &RabitqConfig::new(),
        0,
    );

    // 4.1 estimate using 1-bit encoding
    println!("rotator dim: {:?}", bin_codes.len());
    let g_add = (&rotated_query - &Array1::from(centroid.clone()))
        .pow2()
        .sum();
    let g_err = (&rotated_query - &Array1::from(centroid))
        .pow2()
        .sum()
        .sqrt();
    let (dist, low_dist, ip) = query.est_dist(&bin_codes, g_add, g_err);
    // Calculate the exact distance
    let l2_dist = (query_vec - rotated_vectors.row(0)).pow2().sum().sqrt();
    println!("acc dist: {:}", l2_dist);
    println!(
        "1-bit dist: {:?}, low_dist: {:?}, ip: {:?}",
        dist.sqrt(),
        low_dist.sqrt(),
        ip
    );

    query.set_g_add(g_err, 0f32);
    let estimated_dist = query.distance_boosting(&ex_codes, ip);

    println!("5-bit estimated dist: {:}", estimated_dist.sqrt());
}

#[test]
fn test_batch_bin_estimator() {
    const DIM: usize = 1024;
    const PADDED_DIM: usize = 1024;
    const NUM_VECTORS: usize = 32; // BatchBinEstimator::K_BATCH_SIZE
    const EX_BITS: usize = 4;

    // 1. Generate random vectors
    let mut rng = rand::rng();
    let unif = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let vectors: Array2<f32> = Array::from_shape_fn((NUM_VECTORS, DIM), |_| unif.sample(&mut rng));
    let query_vec: Array1<f32> = Array::from_shape_fn(DIM, |_| unif.sample(&mut rng));

    // 2. Create a rotator and rotate vectors
    let rotator = Rotator::new(DIM, PADDED_DIM).expect("Failed to create rotator");
    let mut rotated_vectors = Array2::<f32>::zeros((NUM_VECTORS, rotator.padded_dim()));
    for i in 0..NUM_VECTORS {
        let mut rotated_row = rotated_vectors.row_mut(i);
        rotator.rotate(
            vectors.row(i).as_slice().unwrap(),
            rotated_row.as_slice_mut().unwrap(),
        );
    }
    let mut rotated_query = Array1::<f32>::zeros(rotator.padded_dim());
    rotator.rotate(
        query_vec.as_slice().unwrap(),
        rotated_query.as_slice_mut().unwrap(),
    );

    // 3. Quantize vectors
    let centroid = vec![0.0f32; rotator.padded_dim()];
    let mut batch_bin_codes = Vec::new();
    let config = RabitqConfig::new();

    for i in 0..NUM_VECTORS {
        let (bin_codes, _) = quantize_split_single(
            rotated_vectors.row(i).as_slice().unwrap(),
            &centroid,
            EX_BITS,
            MetricType::L2,
            &config,
        );
        batch_bin_codes.extend_from_slice(&bin_codes);
    }

    // 4. Create BatchBinEstimator and estimate
    let mut estimator =
        BatchBinEstimator::new(rotated_query.as_slice().unwrap(), rotator.padded_dim());
    let g_add = (&rotated_query - &Array1::from(centroid.clone()))
        .pow2()
        .sum()
        .sqrt();
    estimator.set_g_add(g_add);

    let mut estimated_distances = vec![0.0f32; NUM_VECTORS];
    estimator.batch_est(&batch_bin_codes, &mut estimated_distances);

    // 5. Calculate exact distances and compare
    println!("Batch Bin Estimator Results:");
    for i in 0..NUM_VECTORS {
        let l2_dist = (query_vec.clone() - vectors.row(i)).pow2().sum().sqrt();
        println!(
            "Vector {}: Exact L2 Dist = {:.4}, Estimated Dist = {:.4}",
            i,
            l2_dist,
            estimated_distances[i].sqrt()
        );
    }
}

#[test]
fn test_single_centroid_estimator() {
    const DIM: usize = 1024;
    const PADDED_DIM: usize = 1024;
    const NUM_VECTORS: usize = 1;
    const EX_BITS: usize = 4;

    // 1. Generate random vectors
    let mut rng = rand::rng();
    let unif = Uniform::new(-1.0f32, 1.0f32).unwrap();
    let vectors: Array2<f32> =
        Array::from_shape_fn((NUM_VECTORS, DIM), |_| unif.sample(&mut rng));
    let query_vec: Array1<f32> = Array::from_shape_fn(DIM, |_| unif.sample(&mut rng));

    // 2. Create a rotator and rotate vectors
    let rotator = Rotator::new(DIM, PADDED_DIM).expect("Failed to create rotator");
    let mut rotated_vectors = Array2::<f32>::zeros((NUM_VECTORS, rotator.padded_dim()));
    let mut rotated_row = rotated_vectors.row_mut(0);
    rotator.rotate(
        vectors.row(0).as_slice().unwrap(),
        rotated_row.as_slice_mut().unwrap(),
    );

    let mut rotated_query = Array1::<f32>::zeros(rotator.padded_dim());
    rotator.rotate(
        query_vec.as_slice().unwrap(),
        rotated_query.as_slice_mut().unwrap(),
    );

    // 3. Quantize with centroid
    let mut centroid = vec![0.0f32; rotator.padded_dim()];
    for i in 0..rotator.padded_dim() {
        centroid[i] = unif.sample(&mut rng);
    }

    let (bin_codes, ex_codes) = quantize_split_single(
        rotated_vectors.row(0).as_slice().unwrap(),
        &centroid,
        EX_BITS,
        MetricType::L2,
        &RabitqConfig::new(),
    );

    // 4. Query using SingleCentroidEstimator
    let centroid_query = SingleCentroidEstimator::new(
        rotated_query.as_slice().unwrap(),
        &centroid,
        rotator.padded_dim(),
        EX_BITS,
        &RabitqConfig::new(),
        0,
    );

    // 4.1 estimate using 1-bit encoding
    let g_add = (&rotated_query - &Array1::from(centroid.clone()))
        .pow2()
        .sum();
    let g_err = (&rotated_query - &Array1::from(centroid.clone()))
        .pow2()
        .sum()
        .sqrt();

    let (dist, low_dist, ip) = centroid_query.est_dist(&bin_codes, g_add, g_err);

    // Calculate the exact distance
    let l2_dist = (query_vec.clone() - vectors.row(0)).pow2().sum();
    println!("[SingleCentroidEstimator] acc dist: {:}", l2_dist);
    println!(
        "[SingleCentroidEstimator] 1-bit dist: {:?}, low_dist: {:?}, ip: {:?}",
        dist,
        low_dist,
        ip
    );
    // calculate the exact inner product
    let ip = (&rotated_query - &Array1::from(centroid.clone()))
        .dot(&vectors.row(0));
    println!("ip: {:?}", ip);

    // 4.2 distance boosting
    let estimated_dist = centroid_query.distance_boosting(&ex_codes, ip);
    println!(
        "[SingleCentroidEstimator] 5-bit estimated dist: {:}",
        estimated_dist.sqrt()
    );
    let full_dist = centroid_query.full_dist(&bin_codes, &ex_codes, g_add, g_err);
    println!(
        "[SingleCentroidEstimator] full dist: {:}, {:}, {:}",
        full_dist.0, full_dist.1, full_dist.2
    );

    // 5. Query using SingleEstimator for comparison
    let single_query = SingleEstimator::new(
        rotated_query.as_slice().unwrap(),
        rotator.padded_dim(),
        EX_BITS,
        &RabitqConfig::new(),
        0,
    );
    single_query.set_g_add(g_err, 0f32); // g_err is norm, ip is 0 for L2

    let (single_dist, single_low_dist, single_ip) =
        single_query.est_dist(&bin_codes, g_add, g_err);
    println!(
        "[SingleEstimator] 1-bit dist: {:?}, low_dist: {:?}, ip: {:?}",
        single_dist,
        single_low_dist,
        single_ip
    );

    let single_full_dist_res = single_query.full_dist(&bin_codes, &ex_codes, g_add, g_err);
    println!(
        "[SingleEstimator] Full estimated dist: {:}, {:}, {:}",
        single_full_dist_res.0, single_full_dist_res.1, single_full_dist_res.2
    );
}
