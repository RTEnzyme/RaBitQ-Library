use rabitq_sys::*;

use crate::RabitqConfig;

pub fn select_excode_ipfunc(ex_bits: usize) -> ex_ipfunc {
    unsafe { rabitq_select_excode_ipfunc(ex_bits) }
}

pub struct SplitBatchEstimator {
    ptr: *mut SplitBatchQuery,
    padded_dim: usize,
}

impl SplitBatchEstimator {
    pub fn new(
        rotated_query: &[f32],
        padded_dim: usize,
        ex_bits: usize,
        metric_type: MetricType,
        use_hacc: bool,
    ) -> Self {
        let ptr = unsafe {
            rabitq_split_batch_query_new(
                rotated_query.as_ptr(),
                padded_dim,
                ex_bits,
                metric_type,
                use_hacc,
            )
        };
        Self { ptr, padded_dim }
    }

    pub fn set_g_add(&mut self, norm: f32, ip: f32) {
        unsafe {
            rabitq_split_batch_query_set_g_add(self.ptr, norm, ip);
        }
    }

    pub fn estdist(&self, batch_data: &[u8], use_hacc: bool) -> (f32, f32, f32) {
        let mut est_distance = 0.0;
        let mut low_distance = 0.0;
        let mut ip_x0_qr = 0.0;
        unsafe {
            rabitq_split_batch_estdist(
                batch_data.as_ptr() as *const i8,
                self.ptr,
                self.padded_dim,
                &mut est_distance,
                &mut low_distance,
                &mut ip_x0_qr,
                use_hacc,
            );
        }
        (est_distance, low_distance, ip_x0_qr)
    }

    pub fn distance_boosting(
        &self,
        ex_data: &[u8],
        ip_func: Option<unsafe extern "C" fn(*const f32, *const u8, usize) -> f32>,
        ex_bits: usize,
        ip_x0_qr: f32,
    ) -> f32 {
        unsafe {
            rabitq_split_distance_boosting_with_batch_query(
                ex_data.as_ptr() as *const i8,
                ip_func,
                self.ptr,
                self.padded_dim,
                ex_bits,
                ip_x0_qr,
            )
        }
    }
}

impl Drop for SplitBatchEstimator {
    fn drop(&mut self) {
        unsafe {
            rabitq_split_batch_query_free(self.ptr);
        }
    }
}

pub struct BatchBinEstimator {
    ptr: *mut BatchQuery,
    padded_dim: usize,
}

impl BatchBinEstimator {
    pub const K_BATCH_SIZE: usize = 32;
    pub fn new(rotated_query: &[f32], padded_dim: usize) -> Self {
        let ptr = unsafe { rabitq_batch_query_new(rotated_query.as_ptr(), padded_dim) };
        Self { ptr, padded_dim }
    }

    pub fn delta(&self) -> f32 {
        unsafe { rabitq_batch_query_delta(self.ptr) }
    }

    pub fn sum_vl_lut(&self) -> f32 {
        unsafe { rabitq_batch_query_sum_vl_lut(self.ptr) }
    }

    pub fn k1xsumq(&self) -> f32 {
        unsafe { rabitq_batch_query_k1xsumq(self.ptr) }
    }

    pub fn g_add(&self) -> f32 {
        unsafe { rabitq_batch_query_g_add(self.ptr) }
    }

    pub fn set_g_add(&mut self, sqr_norm: f32) {
        unsafe { rabitq_batch_query_set_g_add(self.ptr, sqr_norm) }
    }

    pub fn lut(&self) -> &[u8] {
        let data = unsafe { rabitq_batch_query_lut(self.ptr) };
        let len = (1 << 16) * std::mem::size_of::<f32>();
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    pub fn batch_est(&self, batch_data: &[u8], est_distance: &mut [f32]) {
        assert_eq!(
            est_distance.len(),
            Self::K_BATCH_SIZE,
            "The length of est_distance slice must be {}",
            Self::K_BATCH_SIZE
        );
        unsafe {
            rabitq_qg_batch_estdist(
                batch_data.as_ptr() as *const i8,
                self.ptr,
                self.padded_dim,
                est_distance.as_mut_ptr(),
            )
        };
    }
}

impl Drop for BatchBinEstimator {
    fn drop(&mut self) {
        unsafe {
            rabitq_batch_query_free(self.ptr);
        }
    }
}

pub struct SingleEstimator {
    ptr: *mut SplitSingleQuery,
    padded_dim: usize,
    ex_bits: usize,
    ip_func: Option<unsafe extern "C" fn(*const f32, *const u8, usize) -> f32>,
}

impl SingleEstimator {
    pub fn new(
        rotated_query: &[f32],
        padded_dim: usize,
        ex_bits: usize,
        config: &RabitqConfig,
        metric_type: MetricType,
    ) -> Self {
        let ptr = unsafe {
            rabitq_split_single_query_new(
                rotated_query.as_ptr(),
                padded_dim,
                ex_bits,
                config.ptr as *const _,
                metric_type,
            )
        };
        Self {
            ptr,
            padded_dim,
            ex_bits,
            ip_func: select_excode_ipfunc(ex_bits),
        }
    }

    pub fn query_bin(&self) -> &[u64] {
        let data = unsafe { rabitq_split_single_query_query_bin(self.ptr) };
        let len = self.padded_dim / 8; // a u64 is 8 bytes
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    pub fn delta(&self) -> f32 {
        unsafe { rabitq_split_single_query_delta(self.ptr) }
    }

    pub fn vl(&self) -> f32 {
        unsafe { rabitq_split_single_query_vl(self.ptr) }
    }

    pub fn set_g_add(&self, norm: f32, ip: f32) {
        unsafe { rabitq_split_single_query_set_g_add(self.ptr, norm, ip) };
    }

    pub fn est_dist(&self, bin_data: &[u8], g_add: f32, g_error: f32) -> (f32, f32, f32) {
        let mut est_dist = 0.0;
        let mut ip_x0_qr: f32 = 0.0;
        let mut low_dist: f32 = 0.0;
        unsafe {
            rabitq_split_single_estdist(
                bin_data.as_ptr() as *const i8,
                self.ptr,
                self.padded_dim,
                &mut ip_x0_qr,
                &mut est_dist,
                &mut low_dist,
                g_add,
                g_error,
            )
        }
        (est_dist, low_dist, ip_x0_qr)
    }

    pub fn full_dist(
        &self,
        bin_data: &[u8],
        ex_data: &[u8],
        g_add: f32,
        g_error: f32,
    ) -> (f32, f32, f32) {
        let mut est_dist = 0.0;
        let mut ip_x0_qr: f32 = 0.0;
        let mut low_dist: f32 = 0.0;
        unsafe {
            rabitq_split_single_fulldist(
                bin_data.as_ptr() as *const i8,
                ex_data.as_ptr() as *const i8,
                self.ip_func,
                self.ptr,
                self.padded_dim,
                self.ex_bits,
                &mut est_dist,
                &mut low_dist,
                &mut ip_x0_qr,
                g_add,
                g_error,
            );
        }
        (est_dist, low_dist, ip_x0_qr)
    }

    pub fn distance_boosting(&self, ex_data: &[u8], ip_x0_qr: f32) -> f32 {
        unsafe {
            rabitq_split_distance_boosting_with_single_query(
                ex_data.as_ptr() as *const i8,
                self.ip_func,
                self.ptr,
                self.padded_dim,
                self.ex_bits,
                ip_x0_qr,
            )
        }
    }

    
}

pub struct SingleCentroidEstimator {
    ptr: *mut SingleCentroidQuery,
    padded_dim: usize,
    ex_bits: usize,
    ip_func: Option<unsafe extern "C" fn(*const f32, *const u8, usize) -> f32>,
}

impl SingleCentroidEstimator {
    pub fn new(
        rotated_query: &[f32],
        centroid: &[f32],
        padded_dim: usize,
        ex_bits: usize,
        config: &RabitqConfig,
        metric_type: MetricType,
    ) -> Self {
        let ptr = unsafe {
            rabitq_single_centroid_query_new(
                rotated_query.as_ptr(),
                centroid.as_ptr(),
                padded_dim,
                ex_bits,
                config.ptr,
                metric_type,
            )
        };
        
        Self {
            ptr,
            padded_dim,
            ex_bits,
            ip_func: select_excode_ipfunc(ex_bits),
        }
    }

    pub fn query_bin(&self) -> &[u64] {
        let data = unsafe { rabitq_single_centroid_query_query_bin(self.ptr) };
        // From C++: QueryBin_(padded_dim * kNumBits / 64, 0) where kNumBits = 4
        let len = self.padded_dim * 4 / 64;
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    pub fn rotated_query(&self) -> &[f32] {
        let data = unsafe { rabitq_single_centroid_query_rotated_query(self.ptr) };
        unsafe { std::slice::from_raw_parts(data, self.padded_dim) }
    }

    pub fn delta(&self) -> f32 {
        unsafe { rabitq_single_centroid_query_delta(self.ptr) }
    }

    pub fn vl(&self) -> f32 {
        unsafe { rabitq_single_centroid_query_vl(self.ptr) }
    }

    pub fn k1xsumq(&self) -> f32 {
        unsafe { rabitq_single_centroid_query_k1xsumq(self.ptr) }
    }

    pub fn kbxsumq(&self) -> f32 {
        unsafe { rabitq_single_centroid_query_kbxsumq(self.ptr) }
    }

    pub fn g_add(&self) -> f32 {
        unsafe { rabitq_single_centroid_query_g_add(self.ptr) }
    }

    pub fn g_error(&self) -> f32 {
        unsafe { rabitq_single_centroid_query_g_error(self.ptr) }
    }

    pub fn set_g_add(&self, norm: f32, ip: f32) {
        unsafe { rabitq_single_centroid_query_set_g_add(self.ptr, norm, ip) }
    }

    pub fn set_g_error(&self, norm: f32) {
        unsafe { rabitq_single_centroid_query_set_g_error(self.ptr, norm) }
    }

    pub fn est_dist(&self, bin_data: &[u8], g_add: f32, g_error: f32) -> (f32, f32, f32) {
        let mut ip_x0_qr = 0.0;
        let mut est_dist = 0.0;
        let mut low_dist = 0.0;
        unsafe {
            rabitq_single_centroid_estdist(
                bin_data.as_ptr() as *const i8,
                self.ptr,
                self.padded_dim,
                &mut ip_x0_qr,
                &mut est_dist,
                &mut low_dist,
                g_add,
                g_error,
            );
        }
        (est_dist, low_dist, ip_x0_qr)
    }

    pub fn distance_boosting(&self, ex_data: &[u8], ip_x0_qr: f32) -> f32 {
        unsafe {
            rabitq_split_distance_boosting_with_centroid_query(
                ex_data.as_ptr() as *const i8,
                self.ip_func,
                self.ptr,
                self.padded_dim,
                self.ex_bits,
                ip_x0_qr,
            )
        }
    }

    pub fn full_dist(
        &self,
        bin_data: &[u8],
        ex_data: &[u8],
        g_add: f32,
        g_error: f32,
    ) -> (f32, f32, f32) {
        let mut est_dist = 0.0;
        let mut ip_x0_qr: f32 = 0.0;
        let mut low_dist: f32 = 0.0;
        unsafe {
            rabitq_single_centroid_fulldist(
                self.ptr as *mut rabitq_sys::SingleCentroidQuery,
                bin_data.as_ptr() as *const i8,
                ex_data.as_ptr() as *const i8,
                self.ip_func,
                self.padded_dim,
                self.ex_bits,
                &mut est_dist,
                &mut low_dist,
                &mut ip_x0_qr,
                g_add,
                g_error
            );
        }
        (est_dist, low_dist, ip_x0_qr)
    }
}

impl Drop for SingleCentroidEstimator {
    fn drop(&mut self) {
        unsafe {
            rabitq_single_centroid_query_free(self.ptr);
        }
    }
}
