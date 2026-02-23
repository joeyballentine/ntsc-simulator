/// NTSC-M composite video constants derived from the specification.

// --- Frequency and Timing ---
pub const FSC: f64 = 3_579_545.06; // Subcarrier frequency (Hz)
pub const SAMPLE_RATE: f64 = 4.0 * FSC; // 14,318,180.24 Hz (4x subcarrier)
pub const LINE_FREQ: f64 = 15_734.264; // Horizontal line frequency (Hz)

// --- Line Structure ---
pub const TOTAL_LINES: usize = 525;
pub const VISIBLE_LINES: usize = 480;

pub const SAMPLES_PER_LINE: usize = 910;

// --- Horizontal Blanking Timing (in samples at 4xfsc) ---
const fn us_to_samples(us: f64) -> usize {
    (us * SAMPLE_RATE / 1e6 + 0.5) as usize
}

pub const FRONT_PORCH_US: f64 = 1.5;
pub const HSYNC_US: f64 = 4.7;
pub const BREEZEWAY_US: f64 = 0.6;
pub const BURST_US: f64 = 2.79;
pub const BACK_PORCH_US: f64 = 1.31;

pub const FRONT_PORCH_SAMPLES: usize = us_to_samples(FRONT_PORCH_US); // ~21
pub const HSYNC_SAMPLES: usize = us_to_samples(HSYNC_US); // ~67
pub const BREEZEWAY_SAMPLES: usize = us_to_samples(BREEZEWAY_US); // ~9
pub const BURST_SAMPLES: usize = us_to_samples(BURST_US); // ~36
pub const BACK_PORCH_SAMPLES: usize = us_to_samples(BACK_PORCH_US); // ~23

pub const HBLANK_SAMPLES: usize =
    FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES;
pub const ACTIVE_SAMPLES: usize = SAMPLES_PER_LINE - HBLANK_SAMPLES; // ~754

// --- Colorburst ---
pub const BURST_AMPLITUDE_IRE: f64 = 20.0;

// --- IRE Levels and Voltage Mapping ---
pub const SYNC_TIP_V: f32 = 0.0;
pub const BLANKING_V: f32 = 0.2857;

pub const SYNC_TIP_IRE: f64 = -40.0;
pub const WHITE_IRE: f64 = 100.0;

// Composite signal scaling
pub const COMPOSITE_SCALE: f32 = 0.66071429;
pub const COMPOSITE_OFFSET: f32 = 0.33928571;

// --- YIQ Color Matrices ---
// RGB to YIQ (row-major: row = output channel)
pub const RGB_TO_YIQ: [[f32; 3]; 3] = [
    [0.299, 0.587, 0.114],       // Y
    [0.595901, -0.274557, -0.321344], // I
    [0.211537, -0.522736, 0.311200],  // Q
];

// YIQ to RGB (row-major: row = output channel)
pub const YIQ_TO_RGB: [[f32; 3]; 3] = [
    [1.0, 0.956, 0.621],   // R
    [1.0, -0.272, -0.647], // G
    [1.0, -1.106, 1.703],  // B
];

// --- Chroma Modulation Angles ---
pub const I_PHASE_DEG: f64 = 123.0;
pub const Q_PHASE_DEG: f64 = 33.0;
pub const I_PHASE_RAD: f32 = (I_PHASE_DEG * std::f64::consts::PI / 180.0) as f32;
pub const Q_PHASE_RAD: f32 = (Q_PHASE_DEG * std::f64::consts::PI / 180.0) as f32;

// --- Bandwidth Limits (Hz) ---
pub const LUMA_BW: f64 = 4.2e6;
pub const I_BW: f64 = 1.5e6;
pub const Q_BW: f64 = 0.5e6;

// --- Vertical Sync Timing ---
pub const EQ_PULSE_US: f64 = 2.3;
pub const VSYNC_PULSE_US: f64 = 27.1;
pub const EQ_PULSE_SAMPLES: usize = us_to_samples(EQ_PULSE_US);
pub const VSYNC_PULSE_SAMPLES: usize = us_to_samples(VSYNC_PULSE_US);
pub const HALF_LINE_SAMPLES: usize = SAMPLES_PER_LINE / 2; // 455

// --- FIR Filter ---
pub const NUM_TAPS: usize = 101;

// --- Derived Constants ---
pub const ACTIVE_START: usize =
    FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES;
pub const BURST_START: usize = FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES;

pub const BURST_V: f32 = (BURST_AMPLITUDE_IRE / 140.0) as f32;

/// Blank (non-visible, non-vblank) lines that need colorburst.
pub fn blank_burst_lines() -> Vec<usize> {
    let mut lines = Vec::new();
    for ln in 9..20 {
        lines.push(ln);
    }
    for ln in 260..262 {
        lines.push(ln);
    }
    for ln in 271..283 {
        lines.push(ln);
    }
    for ln in 523..525 {
        lines.push(ln);
    }
    lines.retain(|&ln| ln < TOTAL_LINES);
    lines
}

/// Build mapping: absolute line number -> visible line index (or -1 if not visible).
/// Field 1: lines 20-259 -> visible 0, 2, 4, ...
/// Field 2: lines 283-522 -> visible 1, 3, 5, ...
pub fn build_line_to_visible() -> [i32; TOTAL_LINES] {
    let mut mapping = [-1i32; TOTAL_LINES];
    for ln in 20..260 {
        mapping[ln] = ((ln - 20) * 2) as i32;
    }
    for ln in 283..523 {
        mapping[ln] = ((ln - 283) * 2 + 1) as i32;
    }
    mapping
}

/// Build absolute line numbers for all 480 visible lines (interleaved fields).
pub fn build_abs_lines() -> [usize; VISIBLE_LINES] {
    let mut abs_lines = [0usize; VISIBLE_LINES];
    // Field 1: even visible indices -> lines 20..259
    for i in 0..240 {
        abs_lines[i * 2] = 20 + i;
    }
    // Field 2: odd visible indices -> lines 283..522
    for i in 0..240 {
        abs_lines[i * 2 + 1] = 283 + i;
    }
    abs_lines
}

/// Build reference lines for 1H comb filter (adjacent line in same field).
pub fn build_ref_lines() -> [usize; VISIBLE_LINES] {
    let abs_lines = build_abs_lines();
    let mut ref_lines = [0usize; VISIBLE_LINES];
    for i in 0..VISIBLE_LINES {
        if abs_lines[i] > 0 {
            ref_lines[i] = abs_lines[i] - 1;
        } else {
            ref_lines[i] = abs_lines[i] + 1;
        }
    }
    // First line of each field: use next line since there's no previous
    ref_lines[0] = abs_lines[0] + 1; // field 1 first line
    ref_lines[1] = abs_lines[1] + 1; // field 2 first line
    ref_lines
}
