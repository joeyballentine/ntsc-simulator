//! NTSC composite video encoder: RGB frames -> composite signal.

use realfft::RealFftPlanner;

use crate::constants::*;
use crate::filters::{self, FilterKernel, FilterScratch, NtscFilters};

const PAD: usize = NUM_TAPS;
const PADDED_ACTIVE: usize = ACTIVE_SAMPLES + 2 * PAD;

/// Precomputed data for the encoder (reuse across frames).
/// Each thread should have its own Encoder to avoid contention on scratch buffers.
pub struct Encoder {
    /// Filters for padded active region
    active_filters: NtscFilters,
    /// Filter for VSB lowpass on padded active composite
    vsb_filter: FilterKernel,
    /// Absolute line numbers for visible lines
    abs_lines: [usize; VISIBLE_LINES],
    /// Reusable FFT scratch buffer (sized for PADDED_ACTIVE)
    fft_scratch: FilterScratch,
    // --- Pre-allocated working buffers ---
    y_all: Vec<f32>,
    i_all: Vec<f32>,
    q_all: Vec<f32>,
    padded_buf: Vec<f32>,
    active_voltage: Vec<f32>,
    signal: Vec<f32>,
    // --- Pre-allocated interpolation tables ---
    src_rows_f1: Vec<usize>,
    src_rows_f2: Vec<usize>,
    x0s: Vec<usize>,
    x1s: Vec<usize>,
    ts: Vec<f32>,
    /// Cached source dimensions for interpolation table invalidation
    cached_src_dims: (usize, usize),
}

impl Encoder {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fir_y = filters::design_lowpass(LUMA_BW, NUM_TAPS);
        let active_filters = NtscFilters::new(PADDED_ACTIVE);
        let fft_n = active_filters.luma.fft_n();

        Self {
            vsb_filter: FilterKernel::new(&fir_y, PADDED_ACTIVE),
            active_filters,
            abs_lines: build_abs_lines(),
            fft_scratch: FilterScratch::with_planner(&mut planner, fft_n),
            y_all: vec![0.0f32; VISIBLE_LINES * ACTIVE_SAMPLES],
            i_all: vec![0.0f32; VISIBLE_LINES * ACTIVE_SAMPLES],
            q_all: vec![0.0f32; VISIBLE_LINES * ACTIVE_SAMPLES],
            padded_buf: vec![0.0f32; VISIBLE_LINES * PADDED_ACTIVE],
            active_voltage: vec![0.0f32; VISIBLE_LINES * ACTIVE_SAMPLES],
            signal: vec![BLANKING_V; TOTAL_LINES * SAMPLES_PER_LINE],
            src_rows_f1: vec![0usize; 240],
            src_rows_f2: vec![0usize; 240],
            x0s: vec![0usize; ACTIVE_SAMPLES],
            x1s: vec![0usize; ACTIVE_SAMPLES],
            ts: vec![0.0f32; ACTIVE_SAMPLES],
            cached_src_dims: (0, 0),
        }
    }

    /// Rebuild interpolation tables when source dimensions change.
    fn update_interp_tables(&mut self, width: usize, height: usize) {
        if self.cached_src_dims == (width, height) {
            return;
        }

        for idx in 0..240 {
            let vis = idx * 2;
            self.src_rows_f1[idx] = (vis * height / VISIBLE_LINES).min(height - 1);
            let vis = idx * 2 + 1;
            self.src_rows_f2[idx] = (vis * height / VISIBLE_LINES).min(height - 1);
        }

        if width != ACTIVE_SAMPLES {
            let scale = (width - 1) as f32 / (ACTIVE_SAMPLES - 1) as f32;
            for j in 0..ACTIVE_SAMPLES {
                let x = j as f32 * scale;
                let x0 = (x as usize).min(width - 1);
                self.x0s[j] = x0;
                self.x1s[j] = (x0 + 1).min(width - 1);
                self.ts[j] = x - x0 as f32;
            }
        }

        self.cached_src_dims = (width, height);
    }

    /// Encode a single RGB frame to a composite NTSC signal.
    /// Returns a reference to the internal signal buffer (valid until the next call).
    pub fn encode_frame(
        &mut self,
        frame_rgb: &[u8],
        width: usize,
        height: usize,
        frame_number: u32,
    ) -> &[f32] {
        self.update_interp_tables(width, height);

        // 1. RGB -> YIQ (directly into pre-allocated y/i/q planes via resampling)
        self.resample_yiq_interleaved(frame_rgb, width);

        // 2. Bandwidth-limit: pad, filter, unpad for Y, I, Q
        pad_edge_into(&self.y_all, VISIBLE_LINES, ACTIVE_SAMPLES, PAD, &mut self.padded_buf);
        filters::filter_rows_sequential(&self.active_filters.luma, &mut self.padded_buf, PADDED_ACTIVE, &mut self.fft_scratch);
        unpad_into(&self.padded_buf, VISIBLE_LINES, PADDED_ACTIVE, PAD, &mut self.y_all);

        pad_edge_into(&self.i_all, VISIBLE_LINES, ACTIVE_SAMPLES, PAD, &mut self.padded_buf);
        filters::filter_rows_sequential(&self.active_filters.i_channel, &mut self.padded_buf, PADDED_ACTIVE, &mut self.fft_scratch);
        unpad_into(&self.padded_buf, VISIBLE_LINES, PADDED_ACTIVE, PAD, &mut self.i_all);

        pad_edge_into(&self.q_all, VISIBLE_LINES, ACTIVE_SAMPLES, PAD, &mut self.padded_buf);
        filters::filter_rows_sequential(&self.active_filters.q_channel, &mut self.padded_buf, PADDED_ACTIVE, &mut self.fft_scratch);
        unpad_into(&self.padded_buf, VISIBLE_LINES, PADDED_ACTIVE, PAD, &mut self.q_all);

        // 3. Generate carriers using LUT and modulate chroma
        let frame_phase = std::f32::consts::PI * frame_number as f32;

        for vis_line in 0..VISIBLE_LINES {
            let abs_line = if vis_line % 2 == 0 {
                20 + vis_line / 2
            } else {
                283 + vis_line / 2
            };
            let line_phase = std::f32::consts::PI * abs_line as f32 + frame_phase;

            let i_phi = std::f32::consts::FRAC_PI_2 * ACTIVE_START as f32 + line_phase + I_PHASE_RAD;
            let q_phi = std::f32::consts::FRAC_PI_2 * ACTIVE_START as f32 + line_phase + Q_PHASE_RAD;

            let i_cos = i_phi.cos();
            let i_sin = i_phi.sin();
            let q_cos = q_phi.cos();
            let q_sin = q_phi.sin();

            let i_lut = [i_cos, -i_sin, -i_cos, i_sin];
            let q_lut = [q_cos, -q_sin, -q_cos, q_sin];

            let base = vis_line * ACTIVE_SAMPLES;
            let y_row = &self.y_all[base..base + ACTIVE_SAMPLES];
            let i_row = &self.i_all[base..base + ACTIVE_SAMPLES];
            let q_row = &self.q_all[base..base + ACTIVE_SAMPLES];
            let row = &mut self.active_voltage[base..base + ACTIVE_SAMPLES];

            for n in 0..ACTIVE_SAMPLES {
                let i_carrier = i_lut[n & 3];
                let q_carrier = q_lut[n & 3];
                let chroma = i_row[n] * i_carrier + q_row[n] * q_carrier;
                row[n] = (y_row[n] + chroma) * COMPOSITE_SCALE + COMPOSITE_OFFSET;
            }
        }

        // 4. VSB filter: 4.2 MHz lowpass on composite
        pad_edge_into(&self.active_voltage, VISIBLE_LINES, ACTIVE_SAMPLES, PAD, &mut self.padded_buf);
        filters::filter_rows_sequential(&self.vsb_filter, &mut self.padded_buf, PADDED_ACTIVE, &mut self.fft_scratch);
        unpad_into(&self.padded_buf, VISIBLE_LINES, PADDED_ACTIVE, PAD, &mut self.active_voltage);

        // 5. Build the full 525-line signal (reuse pre-allocated buffer)
        self.signal.fill(BLANKING_V);

        write_blanking_structure(&mut self.signal);
        self.write_burst_visible(frame_phase);

        for vis_line in 0..VISIBLE_LINES {
            let abs_line = self.abs_lines[vis_line];
            let sig_start = abs_line * SAMPLES_PER_LINE + ACTIVE_START;
            let av_start = vis_line * ACTIVE_SAMPLES;
            self.signal[sig_start..sig_start + ACTIVE_SAMPLES]
                .copy_from_slice(&self.active_voltage[av_start..av_start + ACTIVE_SAMPLES]);
        }

        self.write_burst_blank_lines(frame_phase);

        &self.signal
    }

    fn write_burst_visible(&mut self, frame_phase: f32) {
        for vis_line in 0..VISIBLE_LINES {
            let abs_line = self.abs_lines[vis_line];
            let line_phase = std::f32::consts::PI * abs_line as f32 + frame_phase;
            write_burst_line(&mut self.signal, abs_line, line_phase);
        }
    }

    fn write_burst_blank_lines(&mut self, frame_phase: f32) {
        for &ln in &BLANK_BURST_LINES {
            let line_phase = std::f32::consts::PI * ln as f32 + frame_phase;
            write_burst_line(&mut self.signal, ln, line_phase);
        }
    }

    /// RGB to YIQ conversion + resampling + field interleaving in one pass.
    fn resample_yiq_interleaved(&mut self, frame_rgb: &[u8], width: usize) {
        for field_row in 0..240 {
            // Field 1 (even visible lines)
            let vis_even = field_row * 2;
            let src_row = self.src_rows_f1[field_row];
            write_yiq_row(
                frame_rgb, width, src_row,
                &mut self.y_all, &mut self.i_all, &mut self.q_all,
                vis_even, &self.x0s, &self.x1s, &self.ts,
            );

            // Field 2 (odd visible lines)
            let vis_odd = field_row * 2 + 1;
            let src_row = self.src_rows_f2[field_row];
            write_yiq_row(
                frame_rgb, width, src_row,
                &mut self.y_all, &mut self.i_all, &mut self.q_all,
                vis_odd, &self.x0s, &self.x1s, &self.ts,
            );
        }
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Write colorburst on a single line using the carrier LUT.
fn write_burst_line(signal: &mut [f32], line_num: usize, line_phase: f32) {
    let phi = std::f32::consts::FRAC_PI_2 * BURST_START as f32 + line_phase;
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();
    let lut = [-cos_phi, sin_phi, cos_phi, -sin_phi];

    let sig_base = line_num * SAMPLES_PER_LINE + BURST_START;
    for n in 0..BURST_SAMPLES {
        signal[sig_base + n] = BLANKING_V + lut[n & 3] * BURST_V;
    }
}

/// Convert one source row to YIQ and resample to ACTIVE_SAMPLES, writing into the given visible line.
#[inline]
#[allow(clippy::too_many_arguments)]
fn write_yiq_row(
    frame_rgb: &[u8],
    width: usize,
    src_row: usize,
    y_all: &mut [f32],
    i_all: &mut [f32],
    q_all: &mut [f32],
    vis_line: usize,
    x0s: &[usize],
    x1s: &[usize],
    ts: &[f32],
) {
    let row_base = src_row * width * 3;
    let out_base = vis_line * ACTIVE_SAMPLES;

    if width == ACTIVE_SAMPLES {
        for j in 0..ACTIVE_SAMPLES {
            let px = row_base + j * 3;
            let r = frame_rgb[px] as f32 * (1.0 / 255.0);
            let g = frame_rgb[px + 1] as f32 * (1.0 / 255.0);
            let b = frame_rgb[px + 2] as f32 * (1.0 / 255.0);
            y_all[out_base + j] = RGB_TO_YIQ[0][0] * r + RGB_TO_YIQ[0][1] * g + RGB_TO_YIQ[0][2] * b;
            i_all[out_base + j] = RGB_TO_YIQ[1][0] * r + RGB_TO_YIQ[1][1] * g + RGB_TO_YIQ[1][2] * b;
            q_all[out_base + j] = RGB_TO_YIQ[2][0] * r + RGB_TO_YIQ[2][1] * g + RGB_TO_YIQ[2][2] * b;
        }
    } else {
        for j in 0..ACTIVE_SAMPLES {
            let px0 = row_base + x0s[j] * 3;
            let px1 = row_base + x1s[j] * 3;
            let t = ts[j];
            let t_inv = 1.0 - t;

            let r = (frame_rgb[px0] as f32 * t_inv + frame_rgb[px1] as f32 * t) * (1.0 / 255.0);
            let g = (frame_rgb[px0 + 1] as f32 * t_inv + frame_rgb[px1 + 1] as f32 * t) * (1.0 / 255.0);
            let b = (frame_rgb[px0 + 2] as f32 * t_inv + frame_rgb[px1 + 2] as f32 * t) * (1.0 / 255.0);

            y_all[out_base + j] = RGB_TO_YIQ[0][0] * r + RGB_TO_YIQ[0][1] * g + RGB_TO_YIQ[0][2] * b;
            i_all[out_base + j] = RGB_TO_YIQ[1][0] * r + RGB_TO_YIQ[1][1] * g + RGB_TO_YIQ[1][2] * b;
            q_all[out_base + j] = RGB_TO_YIQ[2][0] * r + RGB_TO_YIQ[2][1] * g + RGB_TO_YIQ[2][2] * b;
        }
    }
}

/// Pad each row with edge values, writing into a pre-allocated output buffer.
fn pad_edge_into(
    data: &[f32],
    num_rows: usize,
    row_len: usize,
    pad: usize,
    out: &mut [f32],
) {
    let padded_len = row_len + 2 * pad;
    for r in 0..num_rows {
        let src_start = r * row_len;
        let dst_start = r * padded_len;
        let left_val = data[src_start];
        let right_val = data[src_start + row_len - 1];

        out[dst_start..dst_start + pad].fill(left_val);
        out[dst_start + pad..dst_start + pad + row_len]
            .copy_from_slice(&data[src_start..src_start + row_len]);
        out[dst_start + pad + row_len..dst_start + padded_len].fill(right_val);
    }
}

/// Extract unpadded data from padded 2D array into a pre-allocated output buffer.
fn unpad_into(
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
        signal[start..end].fill(SYNC_TIP_V);
    }

    let write_eq = |signal: &mut [f32], ln: usize, pos: usize| {
        let base = ln * SAMPLES_PER_LINE + pos;
        let end = (base + EQ_PULSE_SAMPLES).min((ln + 1) * SAMPLES_PER_LINE);
        signal[base..end].fill(SYNC_TIP_V);
    };

    let write_broad = |signal: &mut [f32], ln: usize, pos: usize| {
        let base = ln * SAMPLES_PER_LINE + pos;
        let end = (base + VSYNC_PULSE_SAMPLES).min((ln + 1) * SAMPLES_PER_LINE);
        signal[base..end].fill(SYNC_TIP_V);
    };

    let fill_blanking = |signal: &mut [f32], ln: usize| {
        let base = ln * SAMPLES_PER_LINE;
        signal[base..base + SAMPLES_PER_LINE].fill(BLANKING_V);
    };

    for ln in 0..3 {
        fill_blanking(signal, ln);
        write_eq(signal, ln, 0);
        write_eq(signal, ln, hl);
    }
    for ln in 3..6 {
        fill_blanking(signal, ln);
        write_broad(signal, ln, 0);
        write_broad(signal, ln, hl);
    }
    for ln in 6..9 {
        fill_blanking(signal, ln);
        write_eq(signal, ln, 0);
        write_eq(signal, ln, hl);
    }

    fill_blanking(signal, 262);
    write_eq(signal, 262, hl);

    for ln in 263..265 {
        fill_blanking(signal, ln);
        write_eq(signal, ln, 0);
        write_eq(signal, ln, hl);
    }

    fill_blanking(signal, 265);
    write_eq(signal, 265, 0);
    write_broad(signal, 265, hl);

    for ln in 266..268 {
        fill_blanking(signal, ln);
        write_broad(signal, ln, 0);
        write_broad(signal, ln, hl);
    }

    fill_blanking(signal, 268);
    write_broad(signal, 268, 0);
    write_eq(signal, 268, hl);

    fill_blanking(signal, 269);
    write_eq(signal, 269, 0);
    write_eq(signal, 269, hl);

    fill_blanking(signal, 270);
    write_eq(signal, 270, 0);
}
