"""NTSC composite video decoder: 1D composite signal -> RGB frames."""

import numpy as np
from scipy.signal import firwin
from scipy.fft import rfft, irfft, next_fast_len

from .constants import (
    FSC, SAMPLE_RATE, SAMPLES_PER_LINE, TOTAL_LINES, VISIBLE_LINES,
    FRONT_PORCH_SAMPLES, HSYNC_SAMPLES, BREEZEWAY_SAMPLES,
    BURST_SAMPLES, BACK_PORCH_SAMPLES, ACTIVE_SAMPLES,
    COMPOSITE_SCALE, COMPOSITE_OFFSET,
    YIQ_TO_RGB, I_PHASE_RAD, Q_PHASE_RAD,
    I_BW, Q_BW, LUMA_BW,
)

_F = np.float32
_OMEGA_PER_SAMPLE = _F(2.0 * np.pi * FSC / SAMPLE_RATE)

# Precompute filter coefficients (float32, 101 taps)
_NYQ = SAMPLE_RATE / 2
_NUM_TAPS = 101
_FIR_Y = firwin(_NUM_TAPS, LUMA_BW / _NYQ).astype(np.float32)
_FIR_I = firwin(_NUM_TAPS, I_BW / _NYQ).astype(np.float32)
_FIR_Q = firwin(_NUM_TAPS, Q_BW / _NYQ).astype(np.float32)

# Precompute offsets
_ACTIVE_START = (FRONT_PORCH_SAMPLES + HSYNC_SAMPLES +
                 BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES)
_BURST_START = FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES
_ACTIVE_INDICES = np.arange(ACTIVE_SAMPLES, dtype=np.float32) + _ACTIVE_START
_BURST_INDICES = np.arange(BURST_SAMPLES, dtype=np.float32) + _BURST_START

_RIGHT_PAD = _NUM_TAPS

# Float32 constants for hot-path arithmetic
_COMP_SCALE = _F(COMPOSITE_SCALE)
_COMP_OFFSET = _F(COMPOSITE_OFFSET)
_I_PHASE = _F(I_PHASE_RAD)
_Q_PHASE = _F(Q_PHASE_RAD)
_PI = _F(np.pi)

# Precompute visible line -> absolute line mapping
_ABS_LINES = np.empty(VISIBLE_LINES, dtype=np.int32)
_ABS_LINES[0::2] = np.arange(20, 260)
_ABS_LINES[1::2] = np.arange(283, 523)

# Precompute 1H comb reference lines (adjacent line in same field).
# First line of each field uses the next line since there's no previous.
_REF_LINES = _ABS_LINES - 1
_REF_LINES[0] = _ABS_LINES[0] + 1   # field 1 first line: use next
_REF_LINES[1] = _ABS_LINES[1] + 1   # field 2 first line: use next


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


def decode_frame(signal, frame_number=0, output_width=640, output_height=480,
                 comb_1h=False):
    """Decode a composite NTSC signal back to an RGB frame (vectorized).

    Args:
        signal: 1D composite signal array for one frame (525 lines).
        frame_number: Frame index.
        output_width: Output frame width in pixels.
        output_height: Output frame height in pixels.
        comb_1h: Use 1H (one-line delay) comb filter instead of horizontal
                 2-sample comb. Reduces cross-color rainbow artifacts but
                 introduces hanging-dot artifacts at vertical transitions.

    Returns:
        RGB frame as numpy array (output_height x output_width x 3, uint8).
    """
    # Reshape signal into lines
    signal_2d = signal[:TOTAL_LINES * SAMPLES_PER_LINE].reshape(TOTAL_LINES, SAMPLES_PER_LINE)

    # Convert to float32 if needed (e.g. when loaded from float64 .npy file)
    if signal_2d.dtype != np.float32:
        signal_2d = signal_2d.astype(np.float32)

    # Extract FULL lines for all visible lines (910 samples each).
    # Processing the entire line gives the comb filter and FIR filters
    # valid signal data to settle on before reaching the active region,
    # eliminating edge artifacts on the left side of the picture.
    full_lines = signal_2d[_ABS_LINES].copy()  # (480, 910)

    # --- Comb filter for luma/chroma separation (vectorized) ---
    if comb_1h:
        # 1H comb: use adjacent line in same field as reference.
        # Subcarrier phase flips π between consecutive lines, so adding
        # cancels chroma and subtracting cancels luma.
        ref_lines = signal_2d[_REF_LINES]  # (480, 910)
        y_full = (full_lines + ref_lines) * _F(0.5)
        chroma_full = (full_lines - ref_lines) * _F(0.5)
    else:
        # Horizontal 2-sample delay comb at 4xfsc
        delayed = np.zeros_like(full_lines)
        delayed[:, 2:] = full_lines[:, :-2]
        y_full = (full_lines + delayed) * _F(0.5)
        chroma_full = (full_lines - delayed) * _F(0.5)

    # Lowpass luma at 4.2 MHz to remove residual subcarrier energy
    y_full = _fft_filtfilt_2d(_FIR_Y, y_full)

    # Undo composite voltage scaling on luma
    y_full = (y_full - _COMP_OFFSET) / _COMP_SCALE

    # --- Burst phase detection (vectorized) ---
    burst_all = full_lines[:, _BURST_START:_BURST_START + BURST_SAMPLES]
    line_phases = _PI * _ABS_LINES.astype(np.float32)  # (480,)

    # Reference carriers at burst positions for all lines
    burst_omega = _OMEGA_PER_SAMPLE * _BURST_INDICES.reshape(1, -1) + line_phases.reshape(-1, 1)
    cos_ref = np.cos(burst_omega)  # (480, burst_samples)
    sin_ref = np.sin(burst_omega)

    cos_corr = np.sum(burst_all * cos_ref, axis=1)  # (480,)
    sin_corr = np.sum(burst_all * sin_ref, axis=1)
    burst_phase = np.arctan2(sin_corr, cos_corr) - _PI  # (480,)

    # --- Chroma demodulation (vectorized) ---
    # Zero out chroma in the blanking region (before active start).
    # The burst was already used for phase detection above, so we null it
    # to prevent the colorburst from leaking through the FIR filter into
    # the left edge of the active picture.
    chroma_full[:, :_ACTIVE_START] = 0.0

    # Carrier phase for each line's sample range
    full_indices = np.arange(SAMPLES_PER_LINE, dtype=np.float32)
    full_omega = (_OMEGA_PER_SAMPLE * full_indices.reshape(1, -1)
                  + (line_phases + burst_phase).reshape(-1, 1))

    # Product detection on full lines (910 samples)
    i_raw = _F(2.0) * chroma_full * np.cos(full_omega + _I_PHASE)
    q_raw = _F(2.0) * chroma_full * np.cos(full_omega + _Q_PHASE)

    # Pad the demodulated baseband I/Q on the right with reflected values.
    # The active region ends at sample 910 (= SAMPLES_PER_LINE) with no
    # blanking after it. Reflect padding preserves signal continuity for
    # the filter's backward pass better than edge or zero padding.
    i_raw = np.pad(i_raw, ((0, 0), (0, _RIGHT_PAD)), mode='reflect')
    q_raw = np.pad(q_raw, ((0, 0), (0, _RIGHT_PAD)), mode='reflect')

    # Low-pass filter padded lines, then strip padding
    i_full = _fft_filtfilt_2d(_FIR_I, i_raw)[:, :SAMPLES_PER_LINE]
    q_full = _fft_filtfilt_2d(_FIR_Q, q_raw)[:, :SAMPLES_PER_LINE]

    # Crop to active region after filtering
    y_all = y_full[:, _ACTIVE_START:_ACTIVE_START + ACTIVE_SAMPLES]
    i_demod = i_full[:, _ACTIVE_START:_ACTIVE_START + ACTIVE_SAMPLES]
    q_demod = q_full[:, _ACTIVE_START:_ACTIVE_START + ACTIVE_SAMPLES]

    # Undo composite scaling on chroma
    i_demod /= _COMP_SCALE
    q_demod /= _COMP_SCALE

    # --- YIQ to RGB (vectorized) ---
    yiq = np.stack([y_all, i_demod, q_demod], axis=-1)  # (480, 754, 3)
    rgb = yiq @ YIQ_TO_RGB.T

    np.clip(rgb, 0.0, 1.0, out=rgb)

    # Convert to uint8
    output = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    # Resize to output dimensions
    if output_width != ACTIVE_SAMPLES or output_height != VISIBLE_LINES:
        import cv2
        output = cv2.resize(output, (output_width, output_height),
                            interpolation=cv2.INTER_LINEAR)

    return output
