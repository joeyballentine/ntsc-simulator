"""NTSC composite video decoder: 1D composite signal -> RGB frames."""

import numpy as np

from .constants import (
    FSC, SAMPLE_RATE, SAMPLES_PER_LINE, TOTAL_LINES, VISIBLE_LINES,
    FRONT_PORCH_SAMPLES, HSYNC_SAMPLES, BREEZEWAY_SAMPLES,
    BURST_SAMPLES, BACK_PORCH_SAMPLES, ACTIVE_SAMPLES,
    SYNC_TIP_V, BLANKING_V,
    COMPOSITE_SCALE, COMPOSITE_OFFSET,
    YIQ_TO_RGB, I_PHASE_RAD, Q_PHASE_RAD, GAMMA,
)
from .filters import lowpass_i, lowpass_q

# Angular frequency per sample (fsc is exactly sr/4, so this is π/2)
_OMEGA_PER_SAMPLE = 2.0 * np.pi * FSC / SAMPLE_RATE


def _comb_luma(signal):
    """Extract luma using a 1H comb filter: Y[n] = (S[n] + S[n-2]) / 2.

    At 4xfsc, the subcarrier period is exactly 4 samples. Averaging samples
    2 apart cancels the subcarrier (cos and -cos average to 0), leaving luma.
    """
    delayed = np.zeros_like(signal)
    delayed[2:] = signal[:-2]
    return (signal + delayed) / 2.0


def _comb_chroma(signal):
    """Extract chroma using a 1H comb filter: C[n] = (S[n] - S[n-2]) / 2.

    The complement of the luma comb — subtracting cancels the DC/luma component
    and preserves the alternating subcarrier.
    """
    delayed = np.zeros_like(signal)
    delayed[2:] = signal[:-2]
    return (signal - delayed) / 2.0


def _extract_burst_phase(line_signal, line_start_phase):
    """Extract the colorburst phase from a line's burst region.

    Returns the phase offset to correct the demodulation carriers.
    """
    burst_start = FRONT_PORCH_SAMPLES + HSYNC_SAMPLES + BREEZEWAY_SAMPLES
    burst_end = burst_start + BURST_SAMPLES
    burst_region = line_signal[burst_start:burst_end]

    if len(burst_region) < 4:
        return 0.0

    # Use absolute sample positions (matching the encoder)
    t = np.arange(len(burst_region), dtype=np.float64) + burst_start
    omega_t = _OMEGA_PER_SAMPLE * t + line_start_phase

    # Correlate burst with sin/cos references
    cos_corr = np.sum(burst_region * np.cos(omega_t))
    sin_corr = np.sum(burst_region * np.sin(omega_t))

    # Burst is nominally -cos (phase π). Measured phase relative to reference:
    phase = np.arctan2(sin_corr, cos_corr)
    return phase - np.pi


def decode_line(line_signal, line_start_phase, burst_phase=0.0):
    """Decode a single line of composite signal to YIQ.

    Args:
        line_signal: Signal samples for one line.
        line_start_phase: Nominal subcarrier phase at sample 0 of this line.
        burst_phase: Phase correction from burst detection.

    Returns:
        Tuple of (Y, I, Q) arrays for the active picture region.
    """
    active_start = (FRONT_PORCH_SAMPLES + HSYNC_SAMPLES +
                    BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES)
    active_end = min(active_start + ACTIVE_SAMPLES, len(line_signal))
    active = line_signal[active_start:active_end].copy()

    if len(active) < 10:
        return np.zeros(ACTIVE_SAMPLES), np.zeros(ACTIVE_SAMPLES), np.zeros(ACTIVE_SAMPLES)

    num_samples = len(active)

    # --- Luma extraction via comb filter ---
    y_signal = _comb_luma(active)

    # Undo composite voltage scaling: voltage = luma * SCALE + OFFSET
    y_signal = (y_signal - COMPOSITE_OFFSET) / COMPOSITE_SCALE

    # --- Chroma extraction via comb filter ---
    chroma = _comb_chroma(active)

    # Absolute sample indices for the active region
    t = np.arange(num_samples, dtype=np.float64) + active_start
    omega_t = _OMEGA_PER_SAMPLE * t + line_start_phase + burst_phase

    # Product detection: multiply by 2x reference carriers
    i_raw = 2.0 * chroma * np.cos(omega_t + I_PHASE_RAD)
    q_raw = 2.0 * chroma * np.cos(omega_t + Q_PHASE_RAD)

    # Low-pass filter to recover baseband I and Q
    i_demod = lowpass_i(i_raw, zero_phase=True)
    q_demod = lowpass_q(q_raw, zero_phase=True)

    # Undo the composite voltage scaling on chroma channels too.
    # The encoder applied: voltage = (Y + I*cos + Q*cos) * SCALE + OFFSET
    # The comb filter extracts chroma part which is scaled by SCALE.
    # Product detection gives I*SCALE (after LPF), so divide it out.
    i_demod = i_demod / COMPOSITE_SCALE
    q_demod = q_demod / COMPOSITE_SCALE

    # Pad or trim to ACTIVE_SAMPLES
    if num_samples < ACTIVE_SAMPLES:
        y_signal = np.pad(y_signal, (0, ACTIVE_SAMPLES - num_samples))
        i_demod = np.pad(i_demod, (0, ACTIVE_SAMPLES - num_samples))
        q_demod = np.pad(q_demod, (0, ACTIVE_SAMPLES - num_samples))
    else:
        y_signal = y_signal[:ACTIVE_SAMPLES]
        i_demod = i_demod[:ACTIVE_SAMPLES]
        q_demod = q_demod[:ACTIVE_SAMPLES]

    return y_signal, i_demod, q_demod


def decode_frame(signal, frame_number=0, output_width=640, output_height=480):
    """Decode a composite NTSC signal back to an RGB frame.

    Args:
        signal: 1D composite signal array for one frame (525 lines).
        frame_number: Frame index (for interlace field identification).
        output_width: Output frame width in pixels.
        output_height: Output frame height in pixels.

    Returns:
        RGB frame as numpy array (output_height x output_width x 3, uint8).
    """
    output = np.zeros((VISIBLE_LINES, ACTIVE_SAMPLES, 3), dtype=np.float64)

    for line_num in range(TOTAL_LINES):
        line_offset = line_num * SAMPLES_PER_LINE
        if line_offset + SAMPLES_PER_LINE > len(signal):
            break

        line_signal = signal[line_offset:line_offset + SAMPLES_PER_LINE]

        visible_line = _line_to_visible(line_num)
        if visible_line is None or visible_line >= VISIBLE_LINES:
            continue

        line_start_phase = np.pi * line_num
        burst_phase = _extract_burst_phase(line_signal, line_start_phase)
        y, i, q = decode_line(line_signal, line_start_phase, burst_phase)

        # YIQ to RGB
        yiq = np.stack([y, i, q], axis=-1)
        rgb = yiq @ YIQ_TO_RGB.T

        # Clip and apply inverse gamma
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = np.power(rgb, 1.0 / GAMMA)

        output[visible_line] = rgb

    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    if output_width != ACTIVE_SAMPLES or output_height != VISIBLE_LINES:
        import cv2
        output = cv2.resize(output, (output_width, output_height),
                            interpolation=cv2.INTER_LINEAR)

    return output


def _line_to_visible(line_num):
    """Map an absolute line number to a visible line index (0-479)."""
    if line_num < 20:
        return None
    if line_num < 263:
        field_line = line_num - 20
        if field_line >= 240:
            return None
        return field_line * 2
    if line_num < 283:
        return None
    field_line = line_num - 283
    if field_line >= 240:
        return None
    return field_line * 2 + 1
