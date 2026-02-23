//! NTSC-M composite video constants derived from the specification.

// --- Frequency and Timing ---
pub const FSC: f64 = 3_579_545.06; // Subcarrier frequency (Hz)
pub const SAMPLE_RATE: f64 = 4.0 * FSC; // 14,318,180.24 Hz (4x subcarrier)

// --- Line Structure ---
pub const TOTAL_LINES: usize = 525;
pub const VISIBLE_LINES: usize = 480;

pub const SAMPLES_PER_LINE: usize = 910;

// --- Horizontal Blanking Timing (in samples at 4xfsc) ---
const fn us_to_samples(us: f64) -> usize {
    (us * SAMPLE_RATE / 1e6 + 0.5) as usize
}

pub const FRONT_PORCH_SAMPLES: usize = us_to_samples(1.5); // ~21
pub const HSYNC_SAMPLES: usize = us_to_samples(4.7); // ~67
pub const BREEZEWAY_SAMPLES: usize = us_to_samples(0.6); // ~9
pub const BURST_SAMPLES: usize = us_to_samples(2.79); // ~36
pub const BACK_PORCH_SAMPLES: usize = us_to_samples(1.31); // ~23

pub const HBLANK_SAMPLES: usize =
    FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES;
pub const ACTIVE_SAMPLES: usize = SAMPLES_PER_LINE - HBLANK_SAMPLES; // ~754

// --- Colorburst ---
pub const BURST_AMPLITUDE_IRE: f64 = 20.0;

// --- IRE Levels and Voltage Mapping ---
pub const SYNC_TIP_V: f32 = 0.0;
pub const BLANKING_V: f32 = 0.2857;

// Composite signal scaling
pub const COMPOSITE_SCALE: f32 = 0.66071429;
pub const COMPOSITE_OFFSET: f32 = 0.339_285_7;

// --- YIQ Color Matrices ---
pub const RGB_TO_YIQ: [[f32; 3]; 3] = [
    [0.299, 0.587, 0.114],       // Y
    [0.595901, -0.274557, -0.321344], // I
    [0.211537, -0.522736, 0.311200],  // Q
];

pub const YIQ_TO_RGB: [[f32; 3]; 3] = [
    [1.0, 0.956, 0.621],   // R
    [1.0, -0.272, -0.647], // G
    [1.0, -1.106, 1.703],  // B
];

// --- Chroma Modulation Angles ---
pub const I_PHASE_RAD: f32 = (123.0 * std::f64::consts::PI / 180.0) as f32;
pub const Q_PHASE_RAD: f32 = (33.0 * std::f64::consts::PI / 180.0) as f32;

// --- Bandwidth Limits (Hz) ---
pub const LUMA_BW: f64 = 4.2e6;
pub const I_BW: f64 = 1.5e6;
pub const Q_BW: f64 = 0.5e6;

// --- Vertical Sync Timing ---
pub const EQ_PULSE_SAMPLES: usize = us_to_samples(2.3);
pub const VSYNC_PULSE_SAMPLES: usize = us_to_samples(27.1);
pub const HALF_LINE_SAMPLES: usize = SAMPLES_PER_LINE / 2; // 455

// --- FIR Filter ---
pub const NUM_TAPS: usize = 101;

// --- Derived Constants ---
pub const ACTIVE_START: usize =
    FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES;
pub const BURST_START: usize = FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES;

pub const BURST_V: f32 = (BURST_AMPLITUDE_IRE / 140.0) as f32;

/// Blank (non-visible, non-vblank) lines that need colorburst.
pub const BLANK_BURST_LINES: [usize; 25] = [
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    260, 261,
    271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282,
];

/// Build absolute line numbers for all 480 visible lines (interleaved fields).
pub fn build_abs_lines() -> [usize; VISIBLE_LINES] {
    let mut abs_lines = [0usize; VISIBLE_LINES];
    // Field 1: even visible indices -> lines 20..259
    let mut i = 0;
    while i < 240 {
        abs_lines[i * 2] = 20 + i;
        i += 1;
    }
    // Field 2: odd visible indices -> lines 283..522
    i = 0;
    while i < 240 {
        abs_lines[i * 2 + 1] = 283 + i;
        i += 1;
    }
    abs_lines
}

/// Build reference lines for 1H comb filter (adjacent line in same field).
pub fn build_ref_lines() -> [usize; VISIBLE_LINES] {
    let abs_lines = build_abs_lines();
    let mut ref_lines = [0usize; VISIBLE_LINES];
    let mut i = 0;
    while i < VISIBLE_LINES {
        if abs_lines[i] > 0 {
            ref_lines[i] = abs_lines[i] - 1;
        } else {
            ref_lines[i] = abs_lines[i] + 1;
        }
        i += 1;
    }
    // First line of each field: use next line since there's no previous
    ref_lines[0] = abs_lines[0] + 1; // field 1 first line
    ref_lines[1] = abs_lines[1] + 1; // field 2 first line
    ref_lines
}
