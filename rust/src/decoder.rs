//! NTSC composite video decoder: composite signal -> RGB frames.

use realfft::RealFftPlanner;

use crate::constants::*;
use crate::filters::{self, FilterKernel, FilterScratch};

const RIGHT_PAD: usize = NUM_TAPS;

/// Precomputed data for the decoder.
/// Each thread should have its own Decoder to avoid contention on scratch buffers.
pub struct Decoder {
    /// Filter for full-line luma (910 samples)
    luma_filter: FilterKernel,
    /// Filters for I/Q (910 + RIGHT_PAD samples)
    i_filter: FilterKernel,
    q_filter: FilterKernel,
    /// Absolute line numbers for visible lines
    abs_lines: [usize; VISIBLE_LINES],
    /// Reference lines for 1H comb
    ref_lines: [usize; VISIBLE_LINES],
    /// FFT scratch for luma (sized for SAMPLES_PER_LINE FFT)
    luma_scratch: FilterScratch,
    /// FFT scratch for I/Q (sized for SAMPLES_PER_LINE + RIGHT_PAD FFT)
    iq_scratch: FilterScratch,
    // --- Pre-allocated working buffers ---
    full_lines: Vec<f32>,
    y_full: Vec<f32>,
    chroma_full: Vec<f32>,
    burst_phases: Vec<f32>,
    i_raw: Vec<f32>,
    q_raw: Vec<f32>,
    output: Vec<u8>,
    /// Cached output dimensions for buffer invalidation
    cached_out_dims: (usize, usize),
}

impl Decoder {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();

        let fir_y = filters::design_lowpass(LUMA_BW, NUM_TAPS);
        let fir_i = filters::design_lowpass(I_BW, NUM_TAPS);
        let fir_q = filters::design_lowpass(Q_BW, NUM_TAPS);

        let iq_padded_len = SAMPLES_PER_LINE + RIGHT_PAD;

        let luma_filter = FilterKernel::new(&fir_y, SAMPLES_PER_LINE);
        let i_filter = FilterKernel::new(&fir_i, iq_padded_len);

        let luma_fft_n = luma_filter.fft_n();
        let iq_fft_n = i_filter.fft_n();

        Self {
            luma_filter,
            i_filter,
            q_filter: FilterKernel::new(&fir_q, iq_padded_len),
            abs_lines: build_abs_lines(),
            ref_lines: build_ref_lines(),
            luma_scratch: FilterScratch::with_planner(&mut planner, luma_fft_n),
            iq_scratch: FilterScratch::with_planner(&mut planner, iq_fft_n),
            full_lines: vec![0.0f32; VISIBLE_LINES * SAMPLES_PER_LINE],
            y_full: vec![0.0f32; VISIBLE_LINES * SAMPLES_PER_LINE],
            chroma_full: vec![0.0f32; VISIBLE_LINES * SAMPLES_PER_LINE],
            burst_phases: vec![0.0f32; VISIBLE_LINES],
            i_raw: vec![0.0f32; VISIBLE_LINES * iq_padded_len],
            q_raw: vec![0.0f32; VISIBLE_LINES * iq_padded_len],
            output: Vec::new(),
            cached_out_dims: (0, 0),
        }
    }

    /// Decode a composite NTSC signal back to an RGB frame.
    /// Returns a reference to the internal output buffer (valid until the next call).
    pub fn decode_frame(
        &mut self,
        signal: &[f32],
        output_width: usize,
        output_height: usize,
        comb_1h: bool,
    ) -> &[u8] {
        let spl = SAMPLES_PER_LINE;

        // Ensure output buffer is the right size
        let out_size = output_width * output_height * 3;
        if self.cached_out_dims != (output_width, output_height) {
            self.output.resize(out_size, 0);
            self.cached_out_dims = (output_width, output_height);
        }

        // Extract full lines for all visible lines (480 x 910)
        for vis in 0..VISIBLE_LINES {
            let abs = self.abs_lines[vis];
            let sig_start = abs * spl;
            let dst_start = vis * spl;
            self.full_lines[dst_start..dst_start + spl]
                .copy_from_slice(&signal[sig_start..sig_start + spl]);
        }

        // Comb filter for luma/chroma separation
        if comb_1h {
            for vis in 0..VISIBLE_LINES {
                let ref_line = self.ref_lines[vis];
                let ref_start = ref_line * spl;
                let cur_start = vis * spl;
                for n in 0..spl {
                    let cur = self.full_lines[cur_start + n];
                    let ref_val = signal[ref_start + n];
                    self.y_full[cur_start + n] = (cur + ref_val) * 0.5;
                    self.chroma_full[cur_start + n] = (cur - ref_val) * 0.5;
                }
            }
        } else {
            for vis in 0..VISIBLE_LINES {
                let base = vis * spl;
                self.y_full[base] = self.full_lines[base] * 0.5;
                self.y_full[base + 1] = self.full_lines[base + 1] * 0.5;
                self.chroma_full[base] = self.full_lines[base] * 0.5;
                self.chroma_full[base + 1] = self.full_lines[base + 1] * 0.5;

                for n in 2..spl {
                    let cur = self.full_lines[base + n];
                    let delayed = self.full_lines[base + n - 2];
                    self.y_full[base + n] = (cur + delayed) * 0.5;
                    self.chroma_full[base + n] = (cur - delayed) * 0.5;
                }
            }
        }

        // Lowpass luma at 4.2 MHz
        filters::filter_rows_sequential(
            &self.luma_filter,
            &mut self.y_full,
            spl,
            &mut self.luma_scratch,
        );

        // Undo composite voltage scaling on luma
        for v in &mut self.y_full {
            *v = (*v - COMPOSITE_OFFSET) / COMPOSITE_SCALE;
        }

        // Burst phase detection using carrier LUT
        for vis in 0..VISIBLE_LINES {
            let abs_line = self.abs_lines[vis];
            let line_phase = std::f32::consts::PI * abs_line as f32;
            let base = vis * spl;

            let phi = std::f32::consts::FRAC_PI_2 * BURST_START as f32 + line_phase;
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();

            let cos_lut = [cos_phi, -sin_phi, -cos_phi, sin_phi];
            let sin_lut = [sin_phi, cos_phi, -sin_phi, -cos_phi];

            let mut cos_corr = 0.0f32;
            let mut sin_corr = 0.0f32;

            for n in 0..BURST_SAMPLES {
                let burst_val = self.full_lines[base + BURST_START + n];
                cos_corr += burst_val * cos_lut[n & 3];
                sin_corr += burst_val * sin_lut[n & 3];
            }

            self.burst_phases[vis] = sin_corr.atan2(cos_corr) - std::f32::consts::PI;
        }

        // Zero out chroma in blanking region
        for vis in 0..VISIBLE_LINES {
            let base = vis * spl;
            self.chroma_full[base..base + ACTIVE_START].fill(0.0);
        }

        // Product detection using carrier LUT, then pad right with reflect
        let iq_padded_len = spl + RIGHT_PAD;

        for vis in 0..VISIBLE_LINES {
            let abs_line = self.abs_lines[vis];
            let line_phase = std::f32::consts::PI * abs_line as f32;
            let bp = self.burst_phases[vis];
            let chroma_base = vis * spl;
            let iq_base = vis * iq_padded_len;

            let i_phi = line_phase + bp + I_PHASE_RAD;
            let q_phi = line_phase + bp + Q_PHASE_RAD;

            let i_cos = i_phi.cos();
            let i_sin = i_phi.sin();
            let q_cos = q_phi.cos();
            let q_sin = q_phi.sin();

            let i_lut = [i_cos, -i_sin, -i_cos, i_sin];
            let q_lut = [q_cos, -q_sin, -q_cos, q_sin];

            for n in 0..spl {
                let chroma_val = self.chroma_full[chroma_base + n];
                self.i_raw[iq_base + n] = 2.0 * chroma_val * i_lut[n & 3];
                self.q_raw[iq_base + n] = 2.0 * chroma_val * q_lut[n & 3];
            }

            // Reflect padding on right
            for j in 0..RIGHT_PAD {
                self.i_raw[iq_base + spl + j] = self.i_raw[iq_base + spl - 2 - j];
                self.q_raw[iq_base + spl + j] = self.q_raw[iq_base + spl - 2 - j];
            }
        }

        // Low-pass filter I and Q
        filters::filter_rows_sequential(
            &self.i_filter,
            &mut self.i_raw,
            iq_padded_len,
            &mut self.iq_scratch,
        );
        filters::filter_rows_sequential(
            &self.q_filter,
            &mut self.q_raw,
            iq_padded_len,
            &mut self.iq_scratch,
        );

        // Crop to active region and convert YIQ -> RGB
        let need_resize = output_width != ACTIVE_SAMPLES || output_height != VISIBLE_LINES;

        if !need_resize {
            for vis in 0..VISIBLE_LINES {
                let y_base = vis * spl + ACTIVE_START;
                let iq_base = vis * iq_padded_len + ACTIVE_START;
                let out_base = vis * ACTIVE_SAMPLES * 3;

                for n in 0..ACTIVE_SAMPLES {
                    let y = self.y_full[y_base + n];
                    let i = self.i_raw[iq_base + n] / COMPOSITE_SCALE;
                    let q = self.q_raw[iq_base + n] / COMPOSITE_SCALE;

                    let r = (YIQ_TO_RGB[0][0] * y + YIQ_TO_RGB[0][1] * i + YIQ_TO_RGB[0][2] * q).clamp(0.0, 1.0);
                    let g = (YIQ_TO_RGB[1][0] * y + YIQ_TO_RGB[1][1] * i + YIQ_TO_RGB[1][2] * q).clamp(0.0, 1.0);
                    let b = (YIQ_TO_RGB[2][0] * y + YIQ_TO_RGB[2][1] * i + YIQ_TO_RGB[2][2] * q).clamp(0.0, 1.0);

                    self.output[out_base + n * 3] = (r * 255.0 + 0.5) as u8;
                    self.output[out_base + n * 3 + 1] = (g * 255.0 + 0.5) as u8;
                    self.output[out_base + n * 3 + 2] = (b * 255.0 + 0.5) as u8;
                }
            }
        } else {
            // Bilinear resize with fused YIQ->RGB
            let x_scale = if output_width > 1 {
                (ACTIVE_SAMPLES - 1) as f32 / (output_width - 1) as f32
            } else {
                0.0
            };
            let y_scale = if output_height > 1 {
                (VISIBLE_LINES - 1) as f32 / (output_height - 1) as f32
            } else {
                0.0
            };

            let inv_comp_scale = 1.0 / COMPOSITE_SCALE;

            for dy in 0..output_height {
                let sy = dy as f32 * y_scale;
                let sy0 = (sy as usize).min(VISIBLE_LINES - 1);
                let sy1 = (sy0 + 1).min(VISIBLE_LINES - 1);
                let yt = sy - sy0 as f32;
                let yt_inv = 1.0 - yt;

                for dx in 0..output_width {
                    let sx = dx as f32 * x_scale;
                    let sx0 = (sx as usize).min(ACTIVE_SAMPLES - 1);
                    let sx1 = (sx0 + 1).min(ACTIVE_SAMPLES - 1);
                    let xt = sx - sx0 as f32;
                    let xt_inv = 1.0 - xt;

                    let w00 = xt_inv * yt_inv;
                    let w01 = xt * yt_inv;
                    let w10 = xt_inv * yt;
                    let w11 = xt * yt;

                    let y = self.y_full[sy0 * spl + ACTIVE_START + sx0] * w00
                        + self.y_full[sy0 * spl + ACTIVE_START + sx1] * w01
                        + self.y_full[sy1 * spl + ACTIVE_START + sx0] * w10
                        + self.y_full[sy1 * spl + ACTIVE_START + sx1] * w11;

                    let i = (self.i_raw[sy0 * iq_padded_len + ACTIVE_START + sx0] * w00
                        + self.i_raw[sy0 * iq_padded_len + ACTIVE_START + sx1] * w01
                        + self.i_raw[sy1 * iq_padded_len + ACTIVE_START + sx0] * w10
                        + self.i_raw[sy1 * iq_padded_len + ACTIVE_START + sx1] * w11)
                        * inv_comp_scale;

                    let q = (self.q_raw[sy0 * iq_padded_len + ACTIVE_START + sx0] * w00
                        + self.q_raw[sy0 * iq_padded_len + ACTIVE_START + sx1] * w01
                        + self.q_raw[sy1 * iq_padded_len + ACTIVE_START + sx0] * w10
                        + self.q_raw[sy1 * iq_padded_len + ACTIVE_START + sx1] * w11)
                        * inv_comp_scale;

                    let r = (YIQ_TO_RGB[0][0] * y + YIQ_TO_RGB[0][1] * i + YIQ_TO_RGB[0][2] * q).clamp(0.0, 1.0);
                    let g = (YIQ_TO_RGB[1][0] * y + YIQ_TO_RGB[1][1] * i + YIQ_TO_RGB[1][2] * q).clamp(0.0, 1.0);
                    let b = (YIQ_TO_RGB[2][0] * y + YIQ_TO_RGB[2][1] * i + YIQ_TO_RGB[2][2] * q).clamp(0.0, 1.0);

                    let idx = (dy * output_width + dx) * 3;
                    self.output[idx] = (r * 255.0 + 0.5) as u8;
                    self.output[idx + 1] = (g * 255.0 + 0.5) as u8;
                    self.output[idx + 2] = (b * 255.0 + 0.5) as u8;
                }
            }
        }

        &self.output
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}
