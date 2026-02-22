"""NTSC-M composite video constants derived from the specification."""

import numpy as np

# --- Frequency and Timing ---
FSC = 3_579_545.06                      # Subcarrier frequency (Hz)
SAMPLE_RATE = 4 * FSC                   # 14,318,180.24 Hz (4x subcarrier)
LINE_FREQ = 15_734.264                  # Horizontal line frequency (Hz)
FRAME_RATE = 30000 / 1001               # 29.97 fps
FIELD_RATE = 60000 / 1001               # 59.94 fields/s

# --- Line Structure ---
TOTAL_LINES = 525                       # Total lines per frame
LINES_PER_FIELD = 262                   # Integer lines per field (+ half-line)
VISIBLE_LINES = 480                     # Active picture lines
VBLANK_LINES = TOTAL_LINES - VISIBLE_LINES  # 45 lines vertical blanking

SAMPLES_PER_LINE = 910                  # 63.556 us * SAMPLE_RATE ≈ 910
LINE_PERIOD_US = 63.556                 # Microseconds per line

# --- Horizontal Blanking Timing (in samples at 4xfsc) ---
def _us_to_samples(us):
    return int(round(us * SAMPLE_RATE / 1e6))

FRONT_PORCH_US = 1.5
HSYNC_US = 4.7
BREEZEWAY_US = 0.6
BURST_US = 2.79                         # 10 cycles of subcarrier
BACK_PORCH_US = 1.31                    # Remainder of back porch after burst

FRONT_PORCH_SAMPLES = _us_to_samples(FRONT_PORCH_US)    # ~21
HSYNC_SAMPLES = _us_to_samples(HSYNC_US)                 # ~67
BREEZEWAY_SAMPLES = _us_to_samples(BREEZEWAY_US)         # ~9
BURST_SAMPLES = _us_to_samples(BURST_US)                 # ~36 (9 cycles * 4 samples)
BACK_PORCH_SAMPLES = _us_to_samples(BACK_PORCH_US)       # ~23

HBLANK_SAMPLES = (FRONT_PORCH_SAMPLES + HSYNC_SAMPLES +
                  BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES)
ACTIVE_SAMPLES = SAMPLES_PER_LINE - HBLANK_SAMPLES       # ~754

# Active picture (704 pixels at 4xfsc, ~49.17 us)
ACTIVE_WIDTH = 704                      # Active picture samples (standard)

# --- Colorburst ---
BURST_CYCLES = 10                       # 10 cycles of subcarrier in burst
BURST_AMPLITUDE_IRE = 20                # ±20 IRE

# --- IRE Levels and Voltage Mapping ---
# Voltage range: 0V (sync tip) to 1V (white)
SYNC_TIP_V = 0.0
BLANKING_V = 0.2857
BLACK_V = 0.3393                        # 7.5 IRE setup
WHITE_V = 1.0

SYNC_TIP_IRE = -40
BLANKING_IRE = 0
BLACK_IRE = 7.5
WHITE_IRE = 100

# Conversion: voltage = (ire - SYNC_TIP_IRE) / (WHITE_IRE - SYNC_TIP_IRE)
def ire_to_voltage(ire):
    """Convert IRE units to voltage (0-1 range)."""
    return (ire - SYNC_TIP_IRE) / (WHITE_IRE - SYNC_TIP_IRE)

def voltage_to_ire(v):
    """Convert voltage (0-1 range) to IRE units."""
    return v * (WHITE_IRE - SYNC_TIP_IRE) + SYNC_TIP_IRE

# Composite signal scaling from spec:
# composite_voltage = (luma + chroma) * 0.66071429 + 0.33928571
# This maps: luma=0 (with chroma=0) -> 0.3393V (black with setup)
#            luma=1 (with chroma=0) -> 1.0V (white)
COMPOSITE_SCALE = 0.66071429
COMPOSITE_OFFSET = 0.33928571

# --- YIQ Color Matrix ---
# RGB to YIQ
RGB_TO_YIQ = np.array([
    [0.299,     0.587,     0.114    ],   # Y
    [0.595901, -0.274557, -0.321344 ],   # I
    [0.211537, -0.522736,  0.311200 ],   # Q
], dtype=np.float32)

# YIQ to RGB
YIQ_TO_RGB = np.array([
    [1.0,  0.956,  0.621],   # R
    [1.0, -0.272, -0.647],   # G
    [1.0, -1.106,  1.703],   # B
], dtype=np.float32)

# --- Chroma Modulation Angles ---
# I axis at 123° from burst reference, Q axis at 33°
I_PHASE_DEG = 123.0
Q_PHASE_DEG = 33.0
I_PHASE_RAD = np.radians(I_PHASE_DEG)
Q_PHASE_RAD = np.radians(Q_PHASE_DEG)

# --- Bandwidth Limits (Hz) ---
LUMA_BW = 4.2e6
I_BW = 1.5e6
Q_BW = 0.5e6
CHROMA_BW_LOW = 2.0e6                  # Chroma bandpass lower edge
CHROMA_BW_HIGH = 4.2e6                 # Chroma bandpass upper edge

# --- Gamma ---
GAMMA = 2.2

# --- Vertical Sync Timing ---
# Equalizing pulse: half-line width, narrow pulse
EQ_PULSE_US = 2.3                       # Equalizing pulse width
VSYNC_PULSE_US = 27.1                   # Broad pulse width (serration)
EQ_PULSE_SAMPLES = _us_to_samples(EQ_PULSE_US)
VSYNC_PULSE_SAMPLES = _us_to_samples(VSYNC_PULSE_US)
HALF_LINE_SAMPLES = SAMPLES_PER_LINE // 2  # 455 samples
