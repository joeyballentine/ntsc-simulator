/// NTSC composite video encoder: RGB frames -> composite signal.

use crate::constants::*;
use crate::filters::{self, FilterKernel, NtscFilters};

use rayon::prelude::*;

const PAD: usize = NUM_TAPS;
const PADDED_ACTIVE: usize = ACTIVE_SAMPLES + 2 * PAD;

/// Precomputed data for the encoder (reuse across frames).
pub struct Encoder {
    /// Filters for padded active region
    active_filters: NtscFilters,
    /// Filter for VSB lowpass on padded active composite
    vsb_filter: FilterKernel,
    /// Absolute line numbers for visible lines
    abs_lines: [usize; VISIBLE_LINES],
    /// Blank burst lines
    blank_burst_lines: Vec<usize>,
}

impl Encoder {
    pub fn new() -> Self {
        let fir_y = filters::design_lowpass(LUMA_BW, NUM_TAPS);

        Self {
            active_filters: NtscFilters::new(PADDED_ACTIVE),
            vsb_filter: FilterKernel::new(&fir_y, PADDED_ACTIVE),
            abs_lines: build_abs_lines(),
            blank_burst_lines: blank_burst_lines(),
        }
    }

    /// Encode a single RGB frame to a composite NTSC signal.
    ///
    /// `frame_rgb` is row-major RGB u8 data, dimensions `height x width x 3`.
    /// Returns a flat f32 signal of length TOTAL_LINES * SAMPLES_PER_LINE.
    pub fn encode_frame(
        &self,
        frame_rgb: &[u8],
        width: usize,
        height: usize,
        frame_number: u32,
    ) -> Vec<f32> {
        // 1. RGB -> YIQ
        let (y_img, i_img, q_img) = rgb_to_yiq(frame_rgb, width, height);

        // 2. Map visible lines to source rows
        let src_rows_f1 = build_visible_line_map(height, 0); // even indices -> 240 rows
        let src_rows_f2 = build_visible_line_map(height, 1); // odd indices -> 240 rows

        // 3. Resample each field's rows to ACTIVE_SAMPLES
        let y_f1 = resample_rows(&y_img, width, &src_rows_f1, ACTIVE_SAMPLES);
        let i_f1 = resample_rows(&i_img, width, &src_rows_f1, ACTIVE_SAMPLES);
        let q_f1 = resample_rows(&q_img, width, &src_rows_f1, ACTIVE_SAMPLES);

        let y_f2 = resample_rows(&y_img, width, &src_rows_f2, ACTIVE_SAMPLES);
        let i_f2 = resample_rows(&i_img, width, &src_rows_f2, ACTIVE_SAMPLES);
        let q_f2 = resample_rows(&q_img, width, &src_rows_f2, ACTIVE_SAMPLES);

        // 4. Interleave fields: visible line 0 from f1, 1 from f2, 2 from f1, ...
        let mut y_all = vec![0.0f32; VISIBLE_LINES * ACTIVE_SAMPLES];
        let mut i_all = vec![0.0f32; VISIBLE_LINES * ACTIVE_SAMPLES];
        let mut q_all = vec![0.0f32; VISIBLE_LINES * ACTIVE_SAMPLES];

        for row in 0..240 {
            let even = row * 2;
            let odd = row * 2 + 1;
            y_all[even * ACTIVE_SAMPLES..(even + 1) * ACTIVE_SAMPLES]
                .copy_from_slice(&y_f1[row * ACTIVE_SAMPLES..(row + 1) * ACTIVE_SAMPLES]);
            y_all[odd * ACTIVE_SAMPLES..(odd + 1) * ACTIVE_SAMPLES]
                .copy_from_slice(&y_f2[row * ACTIVE_SAMPLES..(row + 1) * ACTIVE_SAMPLES]);
            i_all[even * ACTIVE_SAMPLES..(even + 1) * ACTIVE_SAMPLES]
                .copy_from_slice(&i_f1[row * ACTIVE_SAMPLES..(row + 1) * ACTIVE_SAMPLES]);
            i_all[odd * ACTIVE_SAMPLES..(odd + 1) * ACTIVE_SAMPLES]
                .copy_from_slice(&i_f2[row * ACTIVE_SAMPLES..(row + 1) * ACTIVE_SAMPLES]);
            q_all[even * ACTIVE_SAMPLES..(even + 1) * ACTIVE_SAMPLES]
                .copy_from_slice(&q_f1[row * ACTIVE_SAMPLES..(row + 1) * ACTIVE_SAMPLES]);
            q_all[odd * ACTIVE_SAMPLES..(odd + 1) * ACTIVE_SAMPLES]
                .copy_from_slice(&q_f2[row * ACTIVE_SAMPLES..(row + 1) * ACTIVE_SAMPLES]);
        }

        // 5. Pad with edge values and bandwidth-limit
        let mut y_padded = pad_edge_2d(&y_all, VISIBLE_LINES, ACTIVE_SAMPLES, PAD);
        let mut i_padded = pad_edge_2d(&i_all, VISIBLE_LINES, ACTIVE_SAMPLES, PAD);
        let mut q_padded = pad_edge_2d(&q_all, VISIBLE_LINES, ACTIVE_SAMPLES, PAD);

        filters::filter_rows_parallel(&self.active_filters.luma, &mut y_padded, PADDED_ACTIVE);
        filters::filter_rows_parallel(&self.active_filters.i_channel, &mut i_padded, PADDED_ACTIVE);
        filters::filter_rows_parallel(&self.active_filters.q_channel, &mut q_padded, PADDED_ACTIVE);

        // Unpad
        unpad_2d(&y_padded, VISIBLE_LINES, PADDED_ACTIVE, PAD, &mut y_all);
        unpad_2d(&i_padded, VISIBLE_LINES, PADDED_ACTIVE, PAD, &mut i_all);
        unpad_2d(&q_padded, VISIBLE_LINES, PADDED_ACTIVE, PAD, &mut q_all);

        // 6. Generate carriers and modulate chroma
        let frame_phase = std::f32::consts::PI * frame_number as f32;
        let omega = std::f32::consts::PI / 2.0; // 2π * FSC / SAMPLE_RATE = π/2

        // Build active_voltage = (Y + chroma) * COMPOSITE_SCALE + COMPOSITE_OFFSET
        // Using carrier LUT optimization: cos(ω*n + φ) cycles through 4-element pattern
        let mut active_voltage = vec![0.0f32; VISIBLE_LINES * ACTIVE_SAMPLES];

        active_voltage
            .par_chunks_mut(ACTIVE_SAMPLES)
            .enumerate()
            .for_each(|(vis_line, row)| {
                let abs_line = if vis_line % 2 == 0 {
                    20 + vis_line / 2
                } else {
                    283 + vis_line / 2
                };
                let line_phase = std::f32::consts::PI * abs_line as f32 + frame_phase;

                let y_row = &y_all[vis_line * ACTIVE_SAMPLES..(vis_line + 1) * ACTIVE_SAMPLES];
                let i_row = &i_all[vis_line * ACTIVE_SAMPLES..(vis_line + 1) * ACTIVE_SAMPLES];
                let q_row = &q_all[vis_line * ACTIVE_SAMPLES..(vis_line + 1) * ACTIVE_SAMPLES];

                for n in 0..ACTIVE_SAMPLES {
                    let sample_idx = ACTIVE_START + n;
                    let phase = omega * sample_idx as f32 + line_phase;
                    let i_carrier = (phase + I_PHASE_RAD).cos();
                    let q_carrier = (phase + Q_PHASE_RAD).cos();
                    let chroma = i_row[n] * i_carrier + q_row[n] * q_carrier;
                    row[n] = (y_row[n] + chroma) * COMPOSITE_SCALE + COMPOSITE_OFFSET;
                }
            });

        // 7. VSB filter: 4.2 MHz lowpass on composite
        let mut av_padded = pad_edge_2d(&active_voltage, VISIBLE_LINES, ACTIVE_SAMPLES, PAD);
        filters::filter_rows_parallel(&self.vsb_filter, &mut av_padded, PADDED_ACTIVE);
        unpad_2d(
            &av_padded,
            VISIBLE_LINES,
            PADDED_ACTIVE,
            PAD,
            &mut active_voltage,
        );

        // 8. Build the full 525-line signal
        let mut signal = vec![BLANKING_V; TOTAL_LINES * SAMPLES_PER_LINE];

        // Write blanking structure
        write_blanking_structure(&mut signal);

        // Write colorburst for all visible lines
        self.write_burst_visible(&mut signal, frame_phase);

        // Write active video into the correct lines
        for vis_line in 0..VISIBLE_LINES {
            let abs_line = self.abs_lines[vis_line];
            let sig_start = abs_line * SAMPLES_PER_LINE + ACTIVE_START;
            let av_start = vis_line * ACTIVE_SAMPLES;
            signal[sig_start..sig_start + ACTIVE_SAMPLES]
                .copy_from_slice(&active_voltage[av_start..av_start + ACTIVE_SAMPLES]);
        }

        // Write burst on blank lines
        self.write_burst_blank_lines(&mut signal, frame_phase);

        signal
    }

    fn write_burst_visible(&self, signal: &mut [f32], frame_phase: f32) {
        let omega = std::f32::consts::PI / 2.0;
        for vis_line in 0..VISIBLE_LINES {
            let abs_line = self.abs_lines[vis_line];
            let line_phase = std::f32::consts::PI * abs_line as f32 + frame_phase;
            let sig_base = abs_line * SAMPLES_PER_LINE;
            for n in 0..BURST_SAMPLES {
                let sample_idx = BURST_START + n;
                let phase = omega * sample_idx as f32 + line_phase;
                signal[sig_base + sample_idx] = BLANKING_V + (-phase.cos()) * BURST_V;
            }
        }
    }

    fn write_burst_blank_lines(&self, signal: &mut [f32], frame_phase: f32) {
        let omega = std::f32::consts::PI / 2.0;
        for &ln in &self.blank_burst_lines {
            let line_phase = std::f32::consts::PI * ln as f32 + frame_phase;
            let sig_base = ln * SAMPLES_PER_LINE;
            for n in 0..BURST_SAMPLES {
                let sample_idx = BURST_START + n;
                let phase = omega * sample_idx as f32 + line_phase;
                signal[sig_base + sample_idx] = BLANKING_V + (-phase.cos()) * BURST_V;
            }
        }
    }
}

/// Convert RGB u8 frame to separate Y, I, Q float32 planes.
fn rgb_to_yiq(frame: &[u8], width: usize, height: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = width * height;
    let mut y = vec![0.0f32; pixels];
    let mut i = vec![0.0f32; pixels];
    let mut q = vec![0.0f32; pixels];

    for p in 0..pixels {
        let r = frame[p * 3] as f32 / 255.0;
        let g = frame[p * 3 + 1] as f32 / 255.0;
        let b = frame[p * 3 + 2] as f32 / 255.0;

        y[p] = RGB_TO_YIQ[0][0] * r + RGB_TO_YIQ[0][1] * g + RGB_TO_YIQ[0][2] * b;
        i[p] = RGB_TO_YIQ[1][0] * r + RGB_TO_YIQ[1][1] * g + RGB_TO_YIQ[1][2] * b;
        q[p] = RGB_TO_YIQ[2][0] * r + RGB_TO_YIQ[2][1] * g + RGB_TO_YIQ[2][2] * b;
    }

    (y, i, q)
}

/// Build mapping from field visible lines to source rows.
/// `field`: 0 = even (field 1), 1 = odd (field 2)
/// Returns 240 source row indices.
fn build_visible_line_map(src_height: usize, field: usize) -> Vec<usize> {
    let mut rows = Vec::with_capacity(240);
    for idx in 0..240 {
        let vis = idx * 2 + field;
        let src_row = (vis * src_height / VISIBLE_LINES).min(src_height - 1);
        rows.push(src_row);
    }
    rows
}

/// Resample rows via linear interpolation from `src_width` to `target_width`.
/// `src_rows` contains the source row indices to extract.
fn resample_rows(
    plane: &[f32],
    src_width: usize,
    src_rows: &[usize],
    target_width: usize,
) -> Vec<f32> {
    let num_rows = src_rows.len();
    let mut out = vec![0.0f32; num_rows * target_width];

    if src_width == target_width {
        for (i, &row) in src_rows.iter().enumerate() {
            let src_start = row * src_width;
            out[i * target_width..(i + 1) * target_width]
                .copy_from_slice(&plane[src_start..src_start + src_width]);
        }
        return out;
    }

    // Precompute interpolation indices and weights
    let scale = (src_width - 1) as f32 / (target_width - 1) as f32;
    let mut x0s = vec![0usize; target_width];
    let mut x1s = vec![0usize; target_width];
    let mut ts = vec![0.0f32; target_width];

    for j in 0..target_width {
        let x = j as f32 * scale;
        let x0 = (x as usize).min(src_width - 1);
        let x1 = (x0 + 1).min(src_width - 1);
        x0s[j] = x0;
        x1s[j] = x1;
        ts[j] = x - x0 as f32;
    }

    for (i, &row) in src_rows.iter().enumerate() {
        let src_base = row * src_width;
        let out_base = i * target_width;
        for j in 0..target_width {
            let v0 = plane[src_base + x0s[j]];
            let v1 = plane[src_base + x1s[j]];
            out[out_base + j] = v0 + (v1 - v0) * ts[j];
        }
    }

    out
}

/// Pad each row with edge values on left and right.
fn pad_edge_2d(
    data: &[f32],
    num_rows: usize,
    row_len: usize,
    pad: usize,
) -> Vec<f32> {
    let padded_len = row_len + 2 * pad;
    let mut out = vec![0.0f32; num_rows * padded_len];

    for r in 0..num_rows {
        let src_start = r * row_len;
        let dst_start = r * padded_len;
        let left_val = data[src_start];
        let right_val = data[src_start + row_len - 1];

        // Left pad
        for j in 0..pad {
            out[dst_start + j] = left_val;
        }
        // Copy data
        out[dst_start + pad..dst_start + pad + row_len]
            .copy_from_slice(&data[src_start..src_start + row_len]);
        // Right pad
        for j in 0..pad {
            out[dst_start + pad + row_len + j] = right_val;
        }
    }

    out
}

/// Extract unpadded data from padded 2D array.
fn unpad_2d(
    padded: &[f32],
    num_rows: usize,
    padded_row_len: usize,
    pad: usize,
    output: &mut [f32],
) {
    let row_len = padded_row_len - 2 * pad;
    for r in 0..num_rows {
        let src_start = r * padded_row_len + pad;
        let dst_start = r * row_len;
        output[dst_start..dst_start + row_len]
            .copy_from_slice(&padded[src_start..src_start + row_len]);
    }
}

/// Write sync pulses and blanking for all 525 lines.
fn write_blanking_structure(signal: &mut [f32]) {
    let fp = FRONT_PORCH_SAMPLES;
    let hs = HSYNC_SAMPLES;
    let hl = HALF_LINE_SAMPLES;

    // Normal hsync for all lines
    for ln in 0..TOTAL_LINES {
        let base = ln * SAMPLES_PER_LINE;
        let start = base + fp;
        let end = start + hs;
        for s in &mut signal[start..end] {
            *s = SYNC_TIP_V;
        }
    }

    // Helper closures
    let write_eq = |signal: &mut [f32], ln: usize, pos: usize| {
        let base = ln * SAMPLES_PER_LINE + pos;
        let end = (base + EQ_PULSE_SAMPLES).min((ln + 1) * SAMPLES_PER_LINE);
        for s in &mut signal[base..end] {
            *s = SYNC_TIP_V;
        }
    };

    let write_broad = |signal: &mut [f32], ln: usize, pos: usize| {
        let base = ln * SAMPLES_PER_LINE + pos;
        let end = (base + VSYNC_PULSE_SAMPLES).min((ln + 1) * SAMPLES_PER_LINE);
        for s in &mut signal[base..end] {
            *s = SYNC_TIP_V;
        }
    };

    let fill_blanking = |signal: &mut [f32], ln: usize| {
        let base = ln * SAMPLES_PER_LINE;
        for s in &mut signal[base..base + SAMPLES_PER_LINE] {
            *s = BLANKING_V;
        }
    };

    // Field 1 (lines 0-8)
    // Pre-eq pulses: lines 0-2
    for ln in 0..3 {
        fill_blanking(signal, ln);
        write_eq(signal, ln, 0);
        write_eq(signal, ln, hl);
    }

    // Vsync broad pulses: lines 3-5
    for ln in 3..6 {
        fill_blanking(signal, ln);
        write_broad(signal, ln, 0);
        write_broad(signal, ln, hl);
    }

    // Post-eq pulses: lines 6-8
    for ln in 6..9 {
        fill_blanking(signal, ln);
        write_eq(signal, ln, 0);
        write_eq(signal, ln, hl);
    }

    // Field 2 (lines 262-270)
    fill_blanking(signal, 262);
    write_eq(signal, 262, hl);

    for ln in 263..265 {
        fill_blanking(signal, ln);
        write_eq(signal, ln, 0);
        write_eq(signal, ln, hl);
    }

    // Line 265: EQ->Vsync transition
    fill_blanking(signal, 265);
    write_eq(signal, 265, 0);
    write_broad(signal, 265, hl);

    // Lines 266-267: broad pulses
    for ln in 266..268 {
        fill_blanking(signal, ln);
        write_broad(signal, ln, 0);
        write_broad(signal, ln, hl);
    }

    // Line 268: Vsync->Post-eq transition
    fill_blanking(signal, 268);
    write_broad(signal, 268, 0);
    write_eq(signal, 268, hl);

    // Line 269: EQ pulses
    fill_blanking(signal, 269);
    write_eq(signal, 269, 0);
    write_eq(signal, 269, hl);

    // Line 270: EQ at 0 only
    fill_blanking(signal, 270);
    write_eq(signal, 270, 0);
}
