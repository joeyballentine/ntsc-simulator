/// FIR filter design and FFT-based zero-phase filtering using real FFT.

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex, ComplexToReal};
use std::sync::Arc;

use crate::constants::{SAMPLE_RATE, NUM_TAPS, LUMA_BW, I_BW, Q_BW};

/// Design a FIR lowpass filter using windowed-sinc (Hamming window).
pub fn design_lowpass(cutoff_hz: f64, num_taps: usize) -> Vec<f32> {
    let nyquist = SAMPLE_RATE / 2.0;
    let normalized_cutoff = cutoff_hz / nyquist;

    let half = (num_taps - 1) as f64 / 2.0;
    let mut coeffs = vec![0.0f64; num_taps];

    for i in 0..num_taps {
        let n = i as f64 - half;
        let window =
            0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (num_taps - 1) as f64).cos();
        let sinc = if n.abs() < 1e-10 {
            normalized_cutoff
        } else {
            (std::f64::consts::PI * normalized_cutoff * n).sin() / (std::f64::consts::PI * n)
        };
        coeffs[i] = sinc * window;
    }

    let sum: f64 = coeffs.iter().sum();
    for c in &mut coeffs {
        *c /= sum;
    }

    coeffs.iter().map(|&c| c as f32).collect()
}

/// Find the next power of 2 >= n.
fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Precomputed filter kernel in frequency domain using real FFT.
/// Stores |H(f)|^2 in the compact R2C format (fft_n/2 + 1 complex bins).
pub struct FilterKernel {
    /// |H(f)|^2 for each frequency bin (length = fft_n/2 + 1)
    h_squared: Vec<f32>,
    /// FFT size used
    fft_n: usize,
    /// Expected input signal length
    signal_len: usize,
    /// Number of complex frequency bins (fft_n/2 + 1)
    freq_bins: usize,
}

impl FilterKernel {
    /// Create a precomputed filter kernel for signals of length `signal_len`.
    pub fn new(coeffs: &[f32], signal_len: usize) -> Self {
        let fft_n = next_pow2(signal_len + coeffs.len() - 1);
        let freq_bins = fft_n / 2 + 1;

        // R2C FFT of filter coefficients
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(fft_n);

        let mut input = vec![0.0f32; fft_n];
        input[..coeffs.len()].copy_from_slice(coeffs);
        let mut spectrum = vec![Complex::new(0.0f32, 0.0); freq_bins];
        r2c.process(&mut input, &mut spectrum).unwrap();

        // |H(f)|^2 = re^2 + im^2
        let h_squared: Vec<f32> = spectrum.iter().map(|c| c.norm_sqr()).collect();

        Self {
            h_squared,
            fft_n,
            signal_len,
            freq_bins,
        }
    }

    pub fn fft_n(&self) -> usize {
        self.fft_n
    }

    pub fn signal_len(&self) -> usize {
        self.signal_len
    }

    pub fn freq_bins(&self) -> usize {
        self.freq_bins
    }

    pub fn h_squared(&self) -> &[f32] {
        &self.h_squared
    }
}

/// Reusable scratch buffers for FFT filtering, avoiding per-call allocation.
pub struct FilterScratch {
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    time_buf: Vec<f32>,
    freq_buf: Vec<Complex<f32>>,
    fft_n: usize,
}

impl FilterScratch {
    /// Create scratch buffers for a given FFT size.
    pub fn new(fft_n: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(fft_n);
        let c2r = planner.plan_fft_inverse(fft_n);
        let freq_bins = fft_n / 2 + 1;

        Self {
            r2c,
            c2r,
            time_buf: vec![0.0f32; fft_n],
            freq_buf: vec![Complex::new(0.0, 0.0); freq_bins],
            fft_n,
        }
    }

    /// Apply zero-phase filter to a single row in-place.
    pub fn apply(&mut self, kernel: &FilterKernel, row: &mut [f32]) {
        let n = row.len().min(kernel.signal_len());

        // Copy signal into time buffer, zero-pad
        self.time_buf[..n].copy_from_slice(&row[..n]);
        for v in &mut self.time_buf[n..] {
            *v = 0.0;
        }

        // Forward R2C FFT
        self.r2c
            .process(&mut self.time_buf, &mut self.freq_buf)
            .unwrap();

        // Multiply by |H(f)|^2
        let h2 = kernel.h_squared();
        for (s, &h) in self.freq_buf.iter_mut().zip(h2.iter()) {
            s.re *= h;
            s.im *= h;
        }

        // Inverse C2R FFT
        self.c2r
            .process(&mut self.freq_buf, &mut self.time_buf)
            .unwrap();

        // Normalize and copy back
        let scale = 1.0 / self.fft_n as f32;
        for i in 0..n {
            row[i] = self.time_buf[i] * scale;
        }
    }
}

/// Apply zero-phase FFT filter to multiple rows sequentially, reusing scratch buffers.
pub fn filter_rows_sequential(kernel: &FilterKernel, data: &mut [f32], row_len: usize, scratch: &mut FilterScratch) {
    for row in data.chunks_mut(row_len) {
        scratch.apply(kernel, row);
    }
}

/// Precomputed filter set for the standard NTSC filters.
pub struct NtscFilters {
    pub luma: FilterKernel,
    pub i_channel: FilterKernel,
    pub q_channel: FilterKernel,
}

impl NtscFilters {
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
}
