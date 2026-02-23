/// FIR filter design and FFT-based zero-phase filtering.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::constants::{SAMPLE_RATE, NUM_TAPS, LUMA_BW, I_BW, Q_BW};

/// Design a FIR lowpass filter using windowed-sinc (Hamming window).
///
/// Equivalent to scipy.signal.firwin(num_taps, cutoff / nyquist).
pub fn design_lowpass(cutoff_hz: f64, num_taps: usize) -> Vec<f32> {
    let nyquist = SAMPLE_RATE / 2.0;
    let normalized_cutoff = cutoff_hz / nyquist; // 0..1

    // Windowed-sinc FIR design
    let half = (num_taps - 1) as f64 / 2.0;
    let mut coeffs = vec![0.0f64; num_taps];

    for i in 0..num_taps {
        let n = i as f64 - half;
        // Hamming window
        let window = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (num_taps - 1) as f64).cos();

        // Sinc function
        let sinc = if n.abs() < 1e-10 {
            normalized_cutoff
        } else {
            (std::f64::consts::PI * normalized_cutoff * n).sin() / (std::f64::consts::PI * n)
        };

        coeffs[i] = sinc * window;
    }

    // Normalize so sum = 1
    let sum: f64 = coeffs.iter().sum();
    for c in &mut coeffs {
        *c /= sum;
    }

    coeffs.iter().map(|&c| c as f32).collect()
}

/// Find the next power of 2 >= n (simple fast FFT size).
fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Precomputed filter kernel in frequency domain: |H(f)|^2 stored as real values.
/// This allows reuse across multiple rows without recomputing the FFT of the filter.
pub struct FilterKernel {
    /// |H(f)|^2 for each frequency bin (length = fft_n)
    h_squared: Vec<f32>,
    /// FFT size used
    fft_n: usize,
    /// Expected input signal length
    signal_len: usize,
}

impl FilterKernel {
    /// Create a precomputed filter kernel for signals of length `signal_len`.
    pub fn new(coeffs: &[f32], signal_len: usize) -> Self {
        let fft_n = next_pow2(signal_len + coeffs.len() - 1);

        // FFT of filter coefficients
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_n);

        let mut h_complex: Vec<Complex<f32>> = coeffs
            .iter()
            .map(|&c| Complex::new(c, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)).take(fft_n - coeffs.len()))
            .collect();
        fft.process(&mut h_complex);

        // |H(f)|^2 = H * conj(H) = re^2 + im^2
        let h_squared: Vec<f32> = h_complex.iter().map(|c| c.norm_sqr()).collect();

        Self {
            h_squared,
            fft_n,
            signal_len,
        }
    }

    /// FFT size this kernel was built for.
    pub fn fft_n(&self) -> usize {
        self.fft_n
    }

    /// Expected signal length.
    pub fn signal_len(&self) -> usize {
        self.signal_len
    }

    /// Get the |H(f)|^2 values.
    pub fn h_squared(&self) -> &[f32] {
        &self.h_squared
    }
}

/// Apply zero-phase FFT filter to a single row using a precomputed kernel.
/// The caller provides scratch buffers to avoid allocation.
pub fn apply_filter_kernel(
    kernel: &FilterKernel,
    signal: &[f32],
    output: &mut [f32],
    scratch: &mut Vec<Complex<f32>>,
    planner: &mut FftPlanner<f32>,
) {
    let fft_n = kernel.fft_n();
    let n = signal.len().min(kernel.signal_len());

    // Prepare complex input
    scratch.clear();
    scratch.extend(signal[..n].iter().map(|&s| Complex::new(s, 0.0)));
    scratch.resize(fft_n, Complex::new(0.0, 0.0));

    // Forward FFT
    let fft_fwd = planner.plan_fft_forward(fft_n);
    fft_fwd.process(scratch);

    // Multiply by |H(f)|^2
    let h2 = kernel.h_squared();
    for (s, &h) in scratch.iter_mut().zip(h2.iter()) {
        *s = Complex::new(s.re * h, s.im * h);
    }

    // Inverse FFT
    let fft_inv = planner.plan_fft_inverse(fft_n);
    fft_inv.process(scratch);

    // Normalize and copy to output
    let scale = 1.0 / fft_n as f32;
    for i in 0..n {
        output[i] = scratch[i].re * scale;
    }
}

/// Apply zero-phase FFT filter to multiple rows in parallel using rayon.
/// Each row in `data` has length `row_len`. Filtered in-place.
pub fn filter_rows_parallel(kernel: &FilterKernel, data: &mut [f32], row_len: usize) {
    use rayon::prelude::*;

    data.par_chunks_mut(row_len).for_each(|row| {
        let mut planner = FftPlanner::new();
        let mut scratch = Vec::with_capacity(kernel.fft_n());
        let mut output = vec![0.0f32; row_len];
        apply_filter_kernel(kernel, row, &mut output, &mut scratch, &mut planner);
        row.copy_from_slice(&output);
    });
}

/// Precomputed filter set for the standard NTSC filters.
pub struct NtscFilters {
    pub luma: FilterKernel,
    pub i_channel: FilterKernel,
    pub q_channel: FilterKernel,
}

impl NtscFilters {
    /// Create filter kernels for the given signal length (typically ACTIVE_SAMPLES + 2*PAD
    /// for encoder, or SAMPLES_PER_LINE for decoder).
    pub fn new(signal_len: usize) -> Self {
        let fir_y = design_lowpass(LUMA_BW, NUM_TAPS);
        let fir_i = design_lowpass(I_BW, NUM_TAPS);
        let fir_q = design_lowpass(Q_BW, NUM_TAPS);

        Self {
            luma: FilterKernel::new(&fir_y, signal_len),
            i_channel: FilterKernel::new(&fir_i, signal_len),
            q_channel: FilterKernel::new(&fir_q, signal_len),
        }
    }

    /// Create filter kernels for a different signal length (e.g. decoder uses
    /// SAMPLES_PER_LINE + padding for I/Q).
    pub fn new_separate(luma_len: usize, iq_len: usize) -> Self {
        let fir_y = design_lowpass(LUMA_BW, NUM_TAPS);
        let fir_i = design_lowpass(I_BW, NUM_TAPS);
        let fir_q = design_lowpass(Q_BW, NUM_TAPS);

        Self {
            luma: FilterKernel::new(&fir_y, luma_len),
            i_channel: FilterKernel::new(&fir_i, iq_len),
            q_channel: FilterKernel::new(&fir_q, iq_len),
        }
    }
}
