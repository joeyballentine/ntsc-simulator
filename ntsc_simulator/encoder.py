"""NTSC composite video encoder: RGB frames -> 1D composite signal."""

import numpy as np
from scipy.signal import filtfilt, firwin

from .constants import (
    FSC, SAMPLE_RATE, SAMPLES_PER_LINE, TOTAL_LINES, VISIBLE_LINES,
    FRONT_PORCH_SAMPLES, HSYNC_SAMPLES, BREEZEWAY_SAMPLES,
    BURST_SAMPLES, BACK_PORCH_SAMPLES, ACTIVE_SAMPLES,
    SYNC_TIP_V, BLANKING_V,
    COMPOSITE_SCALE, COMPOSITE_OFFSET,
    RGB_TO_YIQ, I_PHASE_RAD, Q_PHASE_RAD, GAMMA,
    EQ_PULSE_SAMPLES, VSYNC_PULSE_SAMPLES, HALF_LINE_SAMPLES,
    BURST_AMPLITUDE_IRE, LUMA_BW, I_BW, Q_BW,
)

_OMEGA_PER_SAMPLE = 2.0 * np.pi * FSC / SAMPLE_RATE  # π/2

# Precompute filter coefficients
_NYQ = SAMPLE_RATE / 2
_NUM_TAPS = 201
_FIR_Y = firwin(_NUM_TAPS, LUMA_BW / _NYQ)
_FIR_I = firwin(_NUM_TAPS, I_BW / _NYQ)
_FIR_Q = firwin(_NUM_TAPS, Q_BW / _NYQ)

# Precompute active region sample indices (absolute position within a line)
_ACTIVE_START = (FRONT_PORCH_SAMPLES + HSYNC_SAMPLES +
                 BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES)
_ACTIVE_INDICES = np.arange(ACTIVE_SAMPLES, dtype=np.float64) + _ACTIVE_START

# Precompute burst sample indices
_BURST_START = FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES
_BURST_INDICES = np.arange(BURST_SAMPLES, dtype=np.float64) + _BURST_START


def rgb_to_yiq(frame):
    """Convert an RGB frame (H x W x 3, uint8) to YIQ (float64)."""
    rgb = frame.astype(np.float64) / 255.0
    rgb = np.power(rgb, GAMMA)
    return rgb @ RGB_TO_YIQ.T


def _resample_rows(data_2d, target_width):
    """Resample all rows of a 2D array to target_width using linear interpolation."""
    num_rows, src_width = data_2d.shape
    if src_width == target_width:
        return data_2d
    x_old = np.linspace(0, 1, src_width)
    x_new = np.linspace(0, 1, target_width)
    # Vectorized: interpolate each row
    out = np.empty((num_rows, target_width), dtype=np.float64)
    for i in range(num_rows):
        out[i] = np.interp(x_new, x_old, data_2d[i])
    return out


def _filtfilt_2d(coeffs, data_2d):
    """Apply zero-phase FIR filter to each row of a 2D array."""
    if data_2d.shape[1] <= 3 * len(coeffs):
        from scipy.signal import lfilter
        return lfilter(coeffs, 1.0, data_2d, axis=1)
    return filtfilt(coeffs, 1.0, data_2d, axis=1)


def _build_visible_line_map(src_height):
    """Build mapping from visible line index (0-479) to source row index.

    Returns array of shape (480,) with source row indices.
    """
    visible = np.arange(VISIBLE_LINES)
    if src_height == VISIBLE_LINES:
        return visible
    return np.minimum((visible * src_height // VISIBLE_LINES), src_height - 1)


def _build_line_to_visible():
    """Build array mapping absolute line number -> visible line index (-1 if not visible)."""
    mapping = np.full(TOTAL_LINES, -1, dtype=np.int32)
    # Field 1: lines 20-259 -> visible 0, 2, 4, ...
    for ln in range(20, 260):
        mapping[ln] = (ln - 20) * 2
    # Field 2: lines 283-522 -> visible 1, 3, 5, ...
    for ln in range(283, 523):
        mapping[ln] = (ln - 283) * 2 + 1
    return mapping


_LINE_TO_VISIBLE = _build_line_to_visible()


def encode_frame(frame, frame_number=0, field2_frame=None):
    """Encode a single RGB frame to a composite NTSC signal (vectorized).

    Args:
        frame: RGB image for field 1 (and field 2 if field2_frame is None).
        frame_number: Frame index.
        field2_frame: Optional separate RGB image for field 2 (telecine/interlace).
    """
    h, w, _ = frame.shape
    yiq1 = rgb_to_yiq(frame)  # (H, W, 3)

    if field2_frame is not None:
        yiq2 = rgb_to_yiq(field2_frame)
    else:
        yiq2 = yiq1

    # Field 1 = even visible lines (0,2,4...478) -> 240 lines from yiq1
    # Field 2 = odd visible lines (1,3,5...479) -> 240 lines from yiq2
    src_rows_f1 = _build_visible_line_map(h)[0::2]  # 240 rows for field 1
    src_rows_f2 = _build_visible_line_map(yiq2.shape[0])[1::2]  # 240 rows for field 2

    # Resample each field's rows to active width
    y_f1 = _resample_rows(yiq1[src_rows_f1, :, 0], ACTIVE_SAMPLES)
    i_f1 = _resample_rows(yiq1[src_rows_f1, :, 1], ACTIVE_SAMPLES)
    q_f1 = _resample_rows(yiq1[src_rows_f1, :, 2], ACTIVE_SAMPLES)

    y_f2 = _resample_rows(yiq2[src_rows_f2, :, 0], ACTIVE_SAMPLES)
    i_f2 = _resample_rows(yiq2[src_rows_f2, :, 1], ACTIVE_SAMPLES)
    q_f2 = _resample_rows(yiq2[src_rows_f2, :, 2], ACTIVE_SAMPLES)

    # Interleave fields: visible line 0 from f1, 1 from f2, 2 from f1, ...
    y_all = np.empty((VISIBLE_LINES, ACTIVE_SAMPLES), dtype=np.float64)
    i_all = np.empty_like(y_all)
    q_all = np.empty_like(y_all)
    y_all[0::2] = y_f1;  y_all[1::2] = y_f2
    i_all[0::2] = i_f1;  i_all[1::2] = i_f2
    q_all[0::2] = q_f1;  q_all[1::2] = q_f2

    # Bandwidth-limit all rows at once
    y_all = _filtfilt_2d(_FIR_Y, y_all)
    i_all = _filtfilt_2d(_FIR_I, i_all)
    q_all = _filtfilt_2d(_FIR_Q, q_all)

    # Build carrier phases for all 480 visible lines.
    # visible_line -> absolute_line_num is needed for the phase.
    # Field 1: visible 0,2,4...478 -> line 20,21,...259
    # Field 2: visible 1,3,5...479 -> line 283,284,...522
    abs_lines = np.empty(VISIBLE_LINES, dtype=np.int32)
    abs_lines[0::2] = np.arange(20, 260)   # field 1
    abs_lines[1::2] = np.arange(283, 523)  # field 2

    # line_start_phase for each visible line
    line_phases = (np.pi * abs_lines).reshape(-1, 1)  # (480, 1)

    # Carrier for active region: phase = omega * index + line_phase + angle
    active_phase = _OMEGA_PER_SAMPLE * _ACTIVE_INDICES.reshape(1, -1) + line_phases
    i_carrier = np.cos(active_phase + I_PHASE_RAD)  # (480, 754)
    q_carrier = np.cos(active_phase + Q_PHASE_RAD)

    # Modulate and composite
    chroma = i_all * i_carrier + q_all * q_carrier
    active_voltage = (y_all + chroma) * COMPOSITE_SCALE + COMPOSITE_OFFSET  # (480, 754)

    # --- Build the full 525-line signal ---
    signal = np.full((TOTAL_LINES, SAMPLES_PER_LINE), BLANKING_V, dtype=np.float64)

    # Write blanking structure for all lines
    _write_blanking_structure(signal)

    # Write colorburst for all non-vblank lines
    _write_burst_all(signal, abs_lines, line_phases.ravel())

    # Write active video into the correct lines
    for vis_idx in range(VISIBLE_LINES):
        ln = abs_lines[vis_idx]
        signal[ln, _ACTIVE_START:_ACTIVE_START + ACTIVE_SAMPLES] = active_voltage[vis_idx]

    # Write burst on blank (non-vblank, non-visible) lines too
    _write_burst_blank_lines(signal)

    return signal.ravel()


def _write_blanking_structure(signal):
    """Write sync pulses and blanking for all 525 lines (vectorized)."""
    fp = FRONT_PORCH_SAMPLES
    hs = HSYNC_SAMPLES

    # Normal hsync for all lines first
    signal[:, fp:fp + hs] = SYNC_TIP_V

    # Overwrite vblank lines with special sync patterns
    # Pre-eq pulses: lines 0-2, 262-264
    for ln in list(range(0, 3)) + list(range(262, 265)):
        signal[ln, :] = BLANKING_V
        signal[ln, 0:EQ_PULSE_SAMPLES] = SYNC_TIP_V
        signal[ln, HALF_LINE_SAMPLES:HALF_LINE_SAMPLES + EQ_PULSE_SAMPLES] = SYNC_TIP_V

    # Vsync broad pulses: lines 3-5, 265-267
    for ln in list(range(3, 6)) + list(range(265, 268)):
        signal[ln, :] = BLANKING_V
        signal[ln, 0:VSYNC_PULSE_SAMPLES] = SYNC_TIP_V
        signal[ln, HALF_LINE_SAMPLES:HALF_LINE_SAMPLES + VSYNC_PULSE_SAMPLES] = SYNC_TIP_V

    # Post-eq pulses: lines 6-8, 268-270
    for ln in list(range(6, 9)) + list(range(268, 271)):
        signal[ln, :] = BLANKING_V
        signal[ln, 0:EQ_PULSE_SAMPLES] = SYNC_TIP_V
        signal[ln, HALF_LINE_SAMPLES:HALF_LINE_SAMPLES + EQ_PULSE_SAMPLES] = SYNC_TIP_V


def _write_burst_all(signal, abs_lines, line_phases):
    """Write colorburst on all visible lines."""
    burst_v = BURST_AMPLITUDE_IRE / 140.0
    for idx in range(len(abs_lines)):
        ln = abs_lines[idx]
        phase = _OMEGA_PER_SAMPLE * _BURST_INDICES + line_phases[idx]
        signal[ln, _BURST_START:_BURST_START + BURST_SAMPLES] = BLANKING_V + (-np.cos(phase) * burst_v)


def _write_burst_blank_lines(signal):
    """Write colorburst on blank (non-visible, non-vblank special) lines."""
    burst_v = BURST_AMPLITUDE_IRE / 140.0
    # Lines 9-19 and 271-282 are normal blank lines that need burst
    blank_lines = list(range(9, 20)) + list(range(260, 262)) + list(range(271, 283)) + list(range(523, 525))
    for ln in blank_lines:
        if ln >= TOTAL_LINES:
            continue
        phase = _OMEGA_PER_SAMPLE * _BURST_INDICES + np.pi * ln
        signal[ln, _BURST_START:_BURST_START + BURST_SAMPLES] = BLANKING_V + (-np.cos(phase) * burst_v)
