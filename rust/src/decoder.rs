/// NTSC composite video decoder: composite signal -> RGB frames.

use crate::constants::*;
use crate::filters::{self, FilterKernel};

use rayon::prelude::*;

const RIGHT_PAD: usize = NUM_TAPS;

/// Precomputed data for the decoder (reuse across frames).
pub struct Decoder {
    /// Filters for full-line luma (910 samples)
    luma_filter: FilterKernel,
    /// Filters for I/Q (910 + RIGHT_PAD samples)
    i_filter: FilterKernel,
    q_filter: FilterKernel,
    /// Absolute line numbers for visible lines
    abs_lines: [usize; VISIBLE_LINES],
    /// Reference lines for 1H comb
    ref_lines: [usize; VISIBLE_LINES],
}

impl Decoder {
    pub fn new() -> Self {
        let fir_y = filters::design_lowpass(LUMA_BW, NUM_TAPS);
        let fir_i = filters::design_lowpass(I_BW, NUM_TAPS);
        let fir_q = filters::design_lowpass(Q_BW, NUM_TAPS);

        let iq_padded_len = SAMPLES_PER_LINE + RIGHT_PAD;

        Self {
            luma_filter: FilterKernel::new(&fir_y, SAMPLES_PER_LINE),
            i_filter: FilterKernel::new(&fir_i, iq_padded_len),
            q_filter: FilterKernel::new(&fir_q, iq_padded_len),
            abs_lines: build_abs_lines(),
            ref_lines: build_ref_lines(),
        }
    }

    /// Decode a composite NTSC signal back to an RGB frame.
    ///
    /// `signal` is a flat f32 array of length >= TOTAL_LINES * SAMPLES_PER_LINE.
    /// Returns RGB u8 data of size `output_height x output_width x 3`.
    pub fn decode_frame(
        &self,
        signal: &[f32],
        _frame_number: u32,
        output_width: usize,
        output_height: usize,
        comb_1h: bool,
    ) -> Vec<u8> {
        let spl = SAMPLES_PER_LINE;

        // Extract full lines for all visible lines (480 × 910)
        let mut full_lines = vec![0.0f32; VISIBLE_LINES * spl];
        for vis in 0..VISIBLE_LINES {
            let abs = self.abs_lines[vis];
            let sig_start = abs * spl;
            let dst_start = vis * spl;
            full_lines[dst_start..dst_start + spl]
                .copy_from_slice(&signal[sig_start..sig_start + spl]);
        }

        // Comb filter for luma/chroma separation
        let mut y_full = vec![0.0f32; VISIBLE_LINES * spl];
        let mut chroma_full = vec![0.0f32; VISIBLE_LINES * spl];

        if comb_1h {
            // 1H comb: use adjacent line in same field
            for vis in 0..VISIBLE_LINES {
                let ref_line = self.ref_lines[vis];
                let ref_start = ref_line * spl;
                let cur_start = vis * spl;
                for n in 0..spl {
                    let cur = full_lines[cur_start + n];
                    let ref_val = signal[ref_start + n];
                    y_full[cur_start + n] = (cur + ref_val) * 0.5;
                    chroma_full[cur_start + n] = (cur - ref_val) * 0.5;
                }
            }
        } else {
            // Horizontal 2-sample delay comb at 4xfsc
            for vis in 0..VISIBLE_LINES {
                let base = vis * spl;
                // First 2 samples: delayed = 0
                y_full[base] = full_lines[base] * 0.5;
                y_full[base + 1] = full_lines[base + 1] * 0.5;
                chroma_full[base] = full_lines[base] * 0.5;
                chroma_full[base + 1] = full_lines[base + 1] * 0.5;

                for n in 2..spl {
                    let cur = full_lines[base + n];
                    let delayed = full_lines[base + n - 2];
                    y_full[base + n] = (cur + delayed) * 0.5;
                    chroma_full[base + n] = (cur - delayed) * 0.5;
                }
            }
        }

        // Lowpass luma at 4.2 MHz
        filters::filter_rows_parallel(&self.luma_filter, &mut y_full, spl);

        // Undo composite voltage scaling on luma
        for v in &mut y_full {
            *v = (*v - COMPOSITE_OFFSET) / COMPOSITE_SCALE;
        }

        // Burst phase detection
        let mut burst_phases = vec![0.0f32; VISIBLE_LINES];
        let omega = std::f32::consts::PI / 2.0;

        for vis in 0..VISIBLE_LINES {
            let abs_line = self.abs_lines[vis];
            let line_phase = std::f32::consts::PI * abs_line as f32;
            let base = vis * spl;

            let mut cos_corr = 0.0f32;
            let mut sin_corr = 0.0f32;

            for n in 0..BURST_SAMPLES {
                let sample_idx = BURST_START + n;
                let burst_val = full_lines[base + sample_idx];
                let burst_omega = omega * sample_idx as f32 + line_phase;
                cos_corr += burst_val * burst_omega.cos();
                sin_corr += burst_val * burst_omega.sin();
            }

            burst_phases[vis] = sin_corr.atan2(cos_corr) - std::f32::consts::PI;
        }

        // Zero out chroma in blanking region
        for vis in 0..VISIBLE_LINES {
            let base = vis * spl;
            for n in 0..ACTIVE_START {
                chroma_full[base + n] = 0.0;
            }
        }

        // Product detection on full lines, then pad right with reflect
        let iq_padded_len = spl + RIGHT_PAD;
        let mut i_raw = vec![0.0f32; VISIBLE_LINES * iq_padded_len];
        let mut q_raw = vec![0.0f32; VISIBLE_LINES * iq_padded_len];

        i_raw
            .par_chunks_mut(iq_padded_len)
            .zip(q_raw.par_chunks_mut(iq_padded_len))
            .enumerate()
            .for_each(|(vis, (i_row, q_row))| {
                let abs_line = self.abs_lines[vis];
                let line_phase = std::f32::consts::PI * abs_line as f32;
                let bp = burst_phases[vis];
                let chroma_base = vis * spl;

                for n in 0..spl {
                    let full_omega = omega * n as f32 + line_phase + bp;
                    let chroma_val = chroma_full[chroma_base + n];
                    i_row[n] = 2.0 * chroma_val * (full_omega + I_PHASE_RAD).cos();
                    q_row[n] = 2.0 * chroma_val * (full_omega + Q_PHASE_RAD).cos();
                }

                // Reflect padding on right
                for j in 0..RIGHT_PAD {
                    i_row[spl + j] = i_row[spl - 2 - j];
                    q_row[spl + j] = q_row[spl - 2 - j];
                }
            });

        // Low-pass filter I and Q
        filters::filter_rows_parallel(&self.i_filter, &mut i_raw, iq_padded_len);
        filters::filter_rows_parallel(&self.q_filter, &mut q_raw, iq_padded_len);

        // Crop to active region and convert YIQ -> RGB
        let mut rgb_float = vec![0.0f32; VISIBLE_LINES * ACTIVE_SAMPLES * 3];

        for vis in 0..VISIBLE_LINES {
            let y_base = vis * spl + ACTIVE_START;
            let iq_base = vis * iq_padded_len + ACTIVE_START;
            let rgb_base = vis * ACTIVE_SAMPLES * 3;

            for n in 0..ACTIVE_SAMPLES {
                let y = y_full[y_base + n];
                let i = i_raw[iq_base + n] / COMPOSITE_SCALE;
                let q = q_raw[iq_base + n] / COMPOSITE_SCALE;

                let r = YIQ_TO_RGB[0][0] * y + YIQ_TO_RGB[0][1] * i + YIQ_TO_RGB[0][2] * q;
                let g = YIQ_TO_RGB[1][0] * y + YIQ_TO_RGB[1][1] * i + YIQ_TO_RGB[1][2] * q;
                let b = YIQ_TO_RGB[2][0] * y + YIQ_TO_RGB[2][1] * i + YIQ_TO_RGB[2][2] * q;

                rgb_float[rgb_base + n * 3] = r.clamp(0.0, 1.0);
                rgb_float[rgb_base + n * 3 + 1] = g.clamp(0.0, 1.0);
                rgb_float[rgb_base + n * 3 + 2] = b.clamp(0.0, 1.0);
            }
        }

        // Resize to output dimensions using bilinear interpolation
        if output_width == ACTIVE_SAMPLES && output_height == VISIBLE_LINES {
            // No resize needed, just convert to u8
            return rgb_float.iter().map(|&v| (v * 255.0).round() as u8).collect();
        }

        bilinear_resize_rgb(
            &rgb_float,
            ACTIVE_SAMPLES,
            VISIBLE_LINES,
            output_width,
            output_height,
        )
    }
}

/// Bilinear resize of an RGB float image to output dimensions, returning u8.
fn bilinear_resize_rgb(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; dst_w * dst_h * 3];

    let x_scale = if dst_w > 1 {
        (src_w - 1) as f32 / (dst_w - 1) as f32
    } else {
        0.0
    };
    let y_scale = if dst_h > 1 {
        (src_h - 1) as f32 / (dst_h - 1) as f32
    } else {
        0.0
    };

    for dy in 0..dst_h {
        let sy = dy as f32 * y_scale;
        let y0 = (sy as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let yt = sy - y0 as f32;

        for dx in 0..dst_w {
            let sx = dx as f32 * x_scale;
            let x0 = (sx as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let xt = sx - x0 as f32;

            for c in 0..3 {
                let v00 = src[(y0 * src_w + x0) * 3 + c];
                let v01 = src[(y0 * src_w + x1) * 3 + c];
                let v10 = src[(y1 * src_w + x0) * 3 + c];
                let v11 = src[(y1 * src_w + x1) * 3 + c];

                let v = v00 * (1.0 - xt) * (1.0 - yt)
                    + v01 * xt * (1.0 - yt)
                    + v10 * (1.0 - xt) * yt
                    + v11 * xt * yt;

                out[(dy * dst_w + dx) * 3 + c] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    out
}
