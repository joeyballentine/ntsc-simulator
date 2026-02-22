"""NTSC composite video decoder: 1D composite signal -> RGB frames."""

import numpy as np
from scipy.signal import filtfilt, firwin

from .constants import (
    FSC, SAMPLE_RATE, SAMPLES_PER_LINE, TOTAL_LINES, VISIBLE_LINES,
    FRONT_PORCH_SAMPLES, HSYNC_SAMPLES, BREEZEWAY_SAMPLES,
    BURST_SAMPLES, BACK_PORCH_SAMPLES, ACTIVE_SAMPLES,
    COMPOSITE_SCALE, COMPOSITE_OFFSET,
    YIQ_TO_RGB, I_PHASE_RAD, Q_PHASE_RAD, GAMMA,
    I_BW, Q_BW,
)

_OMEGA_PER_SAMPLE = 2.0 * np.pi * FSC / SAMPLE_RATE

# Precompute filter coefficients for I/Q demodulation
_NYQ = SAMPLE_RATE / 2
_NUM_TAPS = 201
_FIR_I = firwin(_NUM_TAPS, I_BW / _NYQ)
_FIR_Q = firwin(_NUM_TAPS, Q_BW / _NYQ)

# Precompute offsets
_ACTIVE_START = (FRONT_PORCH_SAMPLES + HSYNC_SAMPLES +
                 BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES)
_BURST_START = FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES
_ACTIVE_INDICES = np.arange(ACTIVE_SAMPLES, dtype=np.float64) + _ACTIVE_START
_BURST_INDICES = np.arange(BURST_SAMPLES, dtype=np.float64) + _BURST_START

# Precompute visible line -> absolute line mapping
_ABS_LINES = np.empty(VISIBLE_LINES, dtype=np.int32)
_ABS_LINES[0::2] = np.arange(20, 260)
_ABS_LINES[1::2] = np.arange(283, 523)


def _filtfilt_2d(coeffs, data_2d):
    """Apply zero-phase FIR filter to each row of a 2D array."""
    if data_2d.shape[1] <= 3 * len(coeffs):
        from scipy.signal import lfilter
        return lfilter(coeffs, 1.0, data_2d, axis=1)
    return filtfilt(coeffs, 1.0, data_2d, axis=1)


def decode_frame(signal, frame_number=0, output_width=640, output_height=480):
    """Decode a composite NTSC signal back to an RGB frame (vectorized).

    Args:
        signal: 1D composite signal array for one frame (525 lines).
        frame_number: Frame index.
        output_width: Output frame width in pixels.
        output_height: Output frame height in pixels.

    Returns:
        RGB frame as numpy array (output_height x output_width x 3, uint8).
    """
    # Reshape signal into lines
    signal_2d = signal[:TOTAL_LINES * SAMPLES_PER_LINE].reshape(TOTAL_LINES, SAMPLES_PER_LINE)

    # Extract active regions for all visible lines at once
    active_all = signal_2d[_ABS_LINES, _ACTIVE_START:_ACTIVE_START + ACTIVE_SAMPLES].copy()
    # Shape: (480, 754)

    # --- Comb filter for luma/chroma separation (vectorized) ---
    delayed = np.zeros_like(active_all)
    delayed[:, 2:] = active_all[:, :-2]
    y_all = (active_all + delayed) / 2.0
    chroma_all = (active_all - delayed) / 2.0

    # Undo composite voltage scaling on luma
    y_all = (y_all - COMPOSITE_OFFSET) / COMPOSITE_SCALE

    # --- Burst phase detection (vectorized) ---
    burst_all = signal_2d[_ABS_LINES, _BURST_START:_BURST_START + BURST_SAMPLES]
    line_phases = np.pi * _ABS_LINES.astype(np.float64)  # (480,)

    # Reference carriers at burst positions for all lines
    burst_omega = _OMEGA_PER_SAMPLE * _BURST_INDICES.reshape(1, -1) + line_phases.reshape(-1, 1)
    cos_ref = np.cos(burst_omega)  # (480, burst_samples)
    sin_ref = np.sin(burst_omega)

    cos_corr = np.sum(burst_all * cos_ref, axis=1)  # (480,)
    sin_corr = np.sum(burst_all * sin_ref, axis=1)
    burst_phase = np.arctan2(sin_corr, cos_corr) - np.pi  # (480,)

    # --- Chroma demodulation (vectorized) ---
    # Carrier phase for each line's active region
    active_omega = (_OMEGA_PER_SAMPLE * _ACTIVE_INDICES.reshape(1, -1)
                    + (line_phases + burst_phase).reshape(-1, 1))

    # Product detection
    i_raw = 2.0 * chroma_all * np.cos(active_omega + I_PHASE_RAD)
    q_raw = 2.0 * chroma_all * np.cos(active_omega + Q_PHASE_RAD)

    # Low-pass filter all rows at once
    i_demod = _filtfilt_2d(_FIR_I, i_raw)
    q_demod = _filtfilt_2d(_FIR_Q, q_raw)

    # Undo composite scaling on chroma
    i_demod /= COMPOSITE_SCALE
    q_demod /= COMPOSITE_SCALE

    # --- YIQ to RGB (vectorized) ---
    yiq = np.stack([y_all, i_demod, q_demod], axis=-1)  # (480, 754, 3)
    rgb = yiq @ YIQ_TO_RGB.T

    # Clip and inverse gamma
    np.clip(rgb, 0.0, 1.0, out=rgb)
    np.power(rgb, 1.0 / GAMMA, out=rgb)

    # Convert to uint8
    output = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    # Resize to output dimensions
    if output_width != ACTIVE_SAMPLES or output_height != VISIBLE_LINES:
        import cv2
        output = cv2.resize(output, (output_width, output_height),
                            interpolation=cv2.INTER_LINEAR)

    return output
