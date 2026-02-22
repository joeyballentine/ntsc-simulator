"""NTSC composite video encoder: RGB frames -> 1D composite signal."""

import numpy as np
from scipy.signal import firwin
from scipy.fft import rfft, irfft, next_fast_len

from .constants import (
    FSC, SAMPLE_RATE, SAMPLES_PER_LINE, TOTAL_LINES, VISIBLE_LINES,
    FRONT_PORCH_SAMPLES, HSYNC_SAMPLES, BREEZEWAY_SAMPLES,
    BURST_SAMPLES, BACK_PORCH_SAMPLES, ACTIVE_SAMPLES,
    SYNC_TIP_V, BLANKING_V,
    COMPOSITE_SCALE, COMPOSITE_OFFSET,
    RGB_TO_YIQ, I_PHASE_RAD, Q_PHASE_RAD,
    EQ_PULSE_SAMPLES, VSYNC_PULSE_SAMPLES, HALF_LINE_SAMPLES,
    BURST_AMPLITUDE_IRE, LUMA_BW, I_BW, Q_BW,
)

_F = np.float32
_OMEGA_PER_SAMPLE = _F(2.0 * np.pi * FSC / SAMPLE_RATE)  # π/2

# Precompute filter coefficients (float32, 101 taps — halved from 201
# for ~2x faster filtering with negligible perceptual difference)
_NYQ = SAMPLE_RATE / 2
_NUM_TAPS = 101
_FIR_Y = firwin(_NUM_TAPS, LUMA_BW / _NYQ).astype(np.float32)
_FIR_I = firwin(_NUM_TAPS, I_BW / _NYQ).astype(np.float32)
_FIR_Q = firwin(_NUM_TAPS, Q_BW / _NYQ).astype(np.float32)

# Precompute active region sample indices (absolute position within a line)
_ACTIVE_START = (FRONT_PORCH_SAMPLES + HSYNC_SAMPLES +
                 BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES)
_ACTIVE_INDICES = np.arange(ACTIVE_SAMPLES, dtype=np.float32) + _ACTIVE_START

# Precompute burst sample indices
_BURST_START = FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES
_BURST_INDICES = np.arange(BURST_SAMPLES, dtype=np.float32) + _BURST_START

# Float32 versions of constants used in hot-path arithmetic
_BLANKING_F32 = _F(BLANKING_V)
_SYNC_TIP_F32 = _F(SYNC_TIP_V)
_COMP_SCALE = _F(COMPOSITE_SCALE)
_COMP_OFFSET = _F(COMPOSITE_OFFSET)
_BURST_V = _F(BURST_AMPLITUDE_IRE / 140.0)
_I_PHASE = _F(I_PHASE_RAD)
_Q_PHASE = _F(Q_PHASE_RAD)
_PI = _F(np.pi)

# Precompute blank line numbers for vectorized burst writing
_BLANK_BURST_LINES = np.array(
    list(range(9, 20)) + list(range(260, 262)) +
    list(range(271, 283)) + list(range(523, 525)),
    dtype=np.int32,
)
_BLANK_BURST_LINES = _BLANK_BURST_LINES[_BLANK_BURST_LINES < TOTAL_LINES]


def rgb_to_yiq(frame):
    """Convert an RGB frame (H x W x 3, uint8) to YIQ (float32)."""
    rgb = frame.astype(np.float32) * _F(1.0 / 255.0)
    return rgb @ RGB_TO_YIQ.T


def _resample_rows(data_2d, target_width):
    """Resample all rows to target_width via vectorized linear interpolation.

    Fully vectorized with numpy fancy indexing — no Python loop.
    """
    num_rows, src_width = data_2d.shape
    if src_width == target_width:
        return data_2d
    x_new = np.linspace(0, src_width - 1, target_width, dtype=np.float32)
    x0 = np.floor(x_new).astype(np.intp)
    x1 = np.minimum(x0 + 1, src_width - 1)
    t = x_new - x0.astype(np.float32)
    return data_2d[:, x0] * (_F(1.0) - t) + data_2d[:, x1] * t


def _fft_filtfilt_2d(coeffs, data_2d):
    """FFT-based zero-phase FIR filter applied to each row of a 2D array.

    Equivalent to scipy.signal.filtfilt(coeffs, 1.0, data_2d, axis=1) but
    computes |H(f)|^2 * X(f) via FFT for O(N log N) instead of O(N*M).

    For a symmetric FIR filter, filtfilt's forward-backward application
    yields zero phase shift and squared magnitude response |H(f)|^2.
    """
    n_cols = data_2d.shape[1]
    fft_n = next_fast_len(n_cols + len(coeffs) - 1)
    H = rfft(coeffs, n=fft_n)
    H2 = (H * np.conj(H)).real  # |H(f)|^2 — real-valued
    X = rfft(data_2d, n=fft_n, axis=1)
    return irfft(X * H2, n=fft_n, axis=1)[:, :n_cols]


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
    y_all = np.empty((VISIBLE_LINES, ACTIVE_SAMPLES), dtype=np.float32)
    i_all = np.empty_like(y_all)
    q_all = np.empty_like(y_all)
    y_all[0::2] = y_f1;  y_all[1::2] = y_f2
    i_all[0::2] = i_f1;  i_all[1::2] = i_f2
    q_all[0::2] = q_f1;  q_all[1::2] = q_f2

    # Bandwidth-limit all rows at once.
    # Pad with edge values to avoid FIR filter startup artifacts at left/right edges.
    _PAD = _NUM_TAPS
    y_pad = np.pad(y_all, ((0, 0), (_PAD, _PAD)), mode='edge')
    i_pad = np.pad(i_all, ((0, 0), (_PAD, _PAD)), mode='edge')
    q_pad = np.pad(q_all, ((0, 0), (_PAD, _PAD)), mode='edge')
    y_all = _fft_filtfilt_2d(_FIR_Y, y_pad)[:, _PAD:-_PAD]
    i_all = _fft_filtfilt_2d(_FIR_I, i_pad)[:, _PAD:-_PAD]
    q_all = _fft_filtfilt_2d(_FIR_Q, q_pad)[:, _PAD:-_PAD]

    # Build carrier phases for all 480 visible lines.
    # visible_line -> absolute_line_num is needed for the phase.
    # Field 1: visible 0,2,4...478 -> line 20,21,...259
    # Field 2: visible 1,3,5...479 -> line 283,284,...522
    abs_lines = np.empty(VISIBLE_LINES, dtype=np.int32)
    abs_lines[0::2] = np.arange(20, 260)   # field 1
    abs_lines[1::2] = np.arange(283, 523)  # field 2

    # line_start_phase for each visible line.
    # In real NTSC, there are 227.5 subcarrier cycles per line, so the phase
    # advances by π each line AND by π each frame (525 × 227.5 = 119437.5
    # cycles/frame — the half-cycle means phase flips every frame).
    # This causes the dot crawl / rainbow pattern to alternate every frame,
    # producing the characteristic ~15 Hz shimmer.
    frame_phase = _PI * frame_number
    line_phases = (_PI * abs_lines.astype(np.float32) + frame_phase).reshape(-1, 1)  # (480, 1)

    # Carrier for active region: phase = omega * index + line_phase + angle
    active_phase = _OMEGA_PER_SAMPLE * _ACTIVE_INDICES.reshape(1, -1) + line_phases
    i_carrier = np.cos(active_phase + _I_PHASE)  # (480, 754)
    q_carrier = np.cos(active_phase + _Q_PHASE)

    # Modulate and composite
    chroma = i_all * i_carrier + q_all * q_carrier
    active_voltage = (y_all + chroma) * _COMP_SCALE + _COMP_OFFSET  # (480, 754)

    # Apply 4.2 MHz lowpass to composite signal (vestigial sideband effect)
    av_pad = np.pad(active_voltage, ((0, 0), (_PAD, _PAD)), mode='edge')
    active_voltage = _fft_filtfilt_2d(_FIR_Y, av_pad)[:, _PAD:-_PAD]

    # --- Build the full 525-line signal ---
    signal = np.full((TOTAL_LINES, SAMPLES_PER_LINE), _BLANKING_F32, dtype=np.float32)

    # Write blanking structure for all lines
    _write_blanking_structure(signal)

    # Write colorburst for all visible lines (vectorized)
    _write_burst_all(signal, abs_lines, line_phases.ravel())

    # Write active video into the correct lines (vectorized)
    signal[abs_lines, _ACTIVE_START:_ACTIVE_START + ACTIVE_SAMPLES] = active_voltage

    # Write burst on blank (non-vblank, non-visible) lines too (vectorized)
    _write_burst_blank_lines(signal, frame_phase)

    return signal.ravel()


def _write_eq_pulse(signal, ln, pos):
    """Write an equalizing pulse at the given sample position on a line."""
    end = min(pos + EQ_PULSE_SAMPLES, SAMPLES_PER_LINE)
    signal[ln, pos:end] = _SYNC_TIP_F32


def _write_broad_pulse(signal, ln, pos):
    """Write a vsync broad pulse at the given sample position on a line."""
    end = min(pos + VSYNC_PULSE_SAMPLES, SAMPLES_PER_LINE)
    signal[ln, pos:end] = _SYNC_TIP_F32


def _write_blanking_structure(signal):
    """Write sync pulses and blanking for all 525 lines (vectorized)."""
    fp = FRONT_PORCH_SAMPLES
    hs = HSYNC_SAMPLES
    hl = HALF_LINE_SAMPLES

    # Normal hsync for all lines first
    signal[:, fp:fp + hs] = _SYNC_TIP_F32

    # --- Field 1 (lines 0-8): sync pulses at sample 0, no half-line offset ---
    # Pre-eq pulses: lines 0-2
    for ln in range(0, 3):
        signal[ln, :] = _BLANKING_F32
        _write_eq_pulse(signal, ln, 0)
        _write_eq_pulse(signal, ln, hl)

    # Vsync broad pulses: lines 3-5
    for ln in range(3, 6):
        signal[ln, :] = _BLANKING_F32
        _write_broad_pulse(signal, ln, 0)
        _write_broad_pulse(signal, ln, hl)

    # Post-eq pulses: lines 6-8
    for ln in range(6, 9):
        signal[ln, :] = _BLANKING_F32
        _write_eq_pulse(signal, ln, 0)
        _write_eq_pulse(signal, ln, hl)

    # --- Field 2 (lines 262-270): half-line offset ---
    # Line 262: first half = blanking, second half = EQ pulse at hl
    signal[262, :] = _BLANKING_F32
    _write_eq_pulse(signal, 262, hl)

    # Lines 263-264: EQ pulse at 0 AND at hl (same as field 1 eq)
    for ln in range(263, 265):
        signal[ln, :] = _BLANKING_F32
        _write_eq_pulse(signal, ln, 0)
        _write_eq_pulse(signal, ln, hl)

    # Line 265: EQ->Vsync transition: EQ at 0, broad pulse at hl
    signal[265, :] = _BLANKING_F32
    _write_eq_pulse(signal, 265, 0)
    _write_broad_pulse(signal, 265, hl)

    # Lines 266-267: broad pulse at 0 AND hl
    for ln in range(266, 268):
        signal[ln, :] = _BLANKING_F32
        _write_broad_pulse(signal, ln, 0)
        _write_broad_pulse(signal, ln, hl)

    # Line 268: Vsync->Post-eq transition: broad at 0, EQ at hl
    signal[268, :] = _BLANKING_F32
    _write_broad_pulse(signal, 268, 0)
    _write_eq_pulse(signal, 268, hl)

    # Line 269: EQ at 0 AND hl
    signal[269, :] = _BLANKING_F32
    _write_eq_pulse(signal, 269, 0)
    _write_eq_pulse(signal, 269, hl)

    # Line 270: EQ at 0, second half = blanking (no pulse)
    signal[270, :] = _BLANKING_F32
    _write_eq_pulse(signal, 270, 0)


def _write_burst_all(signal, abs_lines, line_phases):
    """Write colorburst on all visible lines (vectorized — no Python loop)."""
    phases = (_OMEGA_PER_SAMPLE * _BURST_INDICES.reshape(1, -1)
              + line_phases.reshape(-1, 1))
    burst_values = _BLANKING_F32 + (-np.cos(phases) * _BURST_V)
    signal[abs_lines, _BURST_START:_BURST_START + BURST_SAMPLES] = burst_values


def _write_burst_blank_lines(signal, frame_phase=_F(0.0)):
    """Write colorburst on blank (non-visible, non-vblank special) lines (vectorized)."""
    line_phases = _PI * _BLANK_BURST_LINES.astype(np.float32) + frame_phase
    phases = (_OMEGA_PER_SAMPLE * _BURST_INDICES.reshape(1, -1)
              + line_phases.reshape(-1, 1))
    burst_values = _BLANKING_F32 + (-np.cos(phases) * _BURST_V)
    signal[_BLANK_BURST_LINES, _BURST_START:_BURST_START + BURST_SAMPLES] = burst_values
