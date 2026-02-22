"""NTSC composite video encoder: RGB frames -> 1D composite signal."""

import numpy as np

from .constants import (
    FSC, SAMPLE_RATE, SAMPLES_PER_LINE, TOTAL_LINES, VISIBLE_LINES,
    FRONT_PORCH_SAMPLES, HSYNC_SAMPLES, BREEZEWAY_SAMPLES,
    BURST_SAMPLES, BACK_PORCH_SAMPLES, ACTIVE_SAMPLES,
    SYNC_TIP_V, BLANKING_V,
    COMPOSITE_SCALE, COMPOSITE_OFFSET,
    RGB_TO_YIQ, I_PHASE_RAD, Q_PHASE_RAD, GAMMA,
    EQ_PULSE_SAMPLES, VSYNC_PULSE_SAMPLES, HALF_LINE_SAMPLES,
    BURST_AMPLITUDE_IRE,
)
from .filters import lowpass_luma, lowpass_i, lowpass_q

# Precompute the angular frequency ratio (fsc is exactly 1/4 of sample rate)
_OMEGA_PER_SAMPLE = 2.0 * np.pi * FSC / SAMPLE_RATE  # = π/2


def rgb_to_yiq(frame):
    """Convert an RGB frame (H x W x 3, uint8) to YIQ (float64, [0-1] luma).

    Applies gamma correction (linearization) before matrix multiply.
    """
    rgb = frame.astype(np.float64) / 255.0
    rgb = np.power(rgb, GAMMA)
    yiq = rgb @ RGB_TO_YIQ.T
    return yiq


def _resample_line(line_data, target_samples):
    """Resample a 1D array to target number of samples."""
    if len(line_data) == target_samples:
        return line_data
    x_old = np.linspace(0, 1, len(line_data))
    x_new = np.linspace(0, 1, target_samples)
    return np.interp(x_new, x_old, line_data)


def _carrier(sample_indices, phase_offset, line_start_phase):
    """Generate cosine carrier at given absolute sample positions within a line."""
    phase = _OMEGA_PER_SAMPLE * sample_indices + line_start_phase + phase_offset
    return np.cos(phase)


def encode_frame(frame, frame_number=0):
    """Encode a single RGB frame to a composite NTSC signal.

    Args:
        frame: RGB image as numpy array (H x W x 3, uint8).
        frame_number: Frame index (affects interlace field order).

    Returns:
        1D numpy array of composite signal voltage values (0-1 range).
    """
    h, w, _ = frame.shape
    yiq = rgb_to_yiq(frame)

    signal = np.full(TOTAL_LINES * SAMPLES_PER_LINE, BLANKING_V, dtype=np.float64)

    # Precompute the sample offset where active video begins
    active_start = (FRONT_PORCH_SAMPLES + HSYNC_SAMPLES +
                    BREEZEWAY_SAMPLES + BURST_SAMPLES + BACK_PORCH_SAMPLES)

    for line_num in range(TOTAL_LINES):
        line_offset = line_num * SAMPLES_PER_LINE
        line_signal = signal[line_offset:line_offset + SAMPLES_PER_LINE]

        # Phase of subcarrier at sample 0 of this line.
        # 910 samples/line at 4xfsc = 227.5 cycles, so phase advances π each line.
        line_start_phase = np.pi * line_num

        if _is_vblank_line(line_num):
            _encode_vblank_line(line_signal, line_num, line_start_phase)
            continue

        visible_line = _line_to_visible(line_num)
        if visible_line is None or visible_line >= h:
            _encode_blank_line(line_signal, line_start_phase)
            continue

        # Map visible line to source row
        src_row = int(visible_line * h / VISIBLE_LINES) if h != VISIBLE_LINES else visible_line
        src_row = min(src_row, h - 1)

        y_line = _resample_line(yiq[src_row, :, 0], ACTIVE_SAMPLES)
        i_line = _resample_line(yiq[src_row, :, 1], ACTIVE_SAMPLES)
        q_line = _resample_line(yiq[src_row, :, 2], ACTIVE_SAMPLES)

        # Bandwidth limit Y, I, Q
        y_line = lowpass_luma(y_line)
        i_line = lowpass_i(i_line)
        q_line = lowpass_q(q_line)

        # Absolute sample indices for the active region within the line
        active_indices = np.arange(ACTIVE_SAMPLES, dtype=np.float64) + active_start

        # Modulate chroma onto carrier using absolute positions
        i_carrier = _carrier(active_indices, I_PHASE_RAD, line_start_phase)
        q_carrier = _carrier(active_indices, Q_PHASE_RAD, line_start_phase)
        chroma = i_line * i_carrier + q_line * q_carrier

        # Composite = luma + chroma, scaled to voltage
        composite = y_line + chroma
        active_voltage = composite * COMPOSITE_SCALE + COMPOSITE_OFFSET

        # Build the full line with blanking, sync, burst, active
        _encode_active_line(line_signal, active_voltage, line_start_phase)

    return signal


def _is_vblank_line(line_num):
    """Check if a line number is in the vertical blanking interval."""
    if line_num < 20:
        return True
    if 262 <= line_num < 283:
        return True
    return False


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


def _encode_active_line(line_signal, active_voltage, line_start_phase):
    """Write a complete active video line with sync, burst, and picture."""
    pos = 0

    # Front porch
    line_signal[pos:pos + FRONT_PORCH_SAMPLES] = BLANKING_V
    pos += FRONT_PORCH_SAMPLES

    # Horizontal sync
    line_signal[pos:pos + HSYNC_SAMPLES] = SYNC_TIP_V
    pos += HSYNC_SAMPLES

    # Breezeway
    line_signal[pos:pos + BREEZEWAY_SAMPLES] = BLANKING_V
    pos += BREEZEWAY_SAMPLES

    # Colorburst — use absolute sample positions for phase consistency
    burst_indices = np.arange(BURST_SAMPLES, dtype=np.float64) + pos
    burst_phase = _OMEGA_PER_SAMPLE * burst_indices + line_start_phase
    burst_v = BURST_AMPLITUDE_IRE / 140.0
    line_signal[pos:pos + BURST_SAMPLES] = BLANKING_V + (-np.cos(burst_phase) * burst_v)
    pos += BURST_SAMPLES

    # Back porch
    line_signal[pos:pos + BACK_PORCH_SAMPLES] = BLANKING_V
    pos += BACK_PORCH_SAMPLES

    # Active video — already ACTIVE_SAMPLES wide, which equals SAMPLES_PER_LINE - pos
    line_signal[pos:pos + len(active_voltage)] = active_voltage


def _encode_blank_line(line_signal, line_start_phase):
    """Write a blank line (sync + burst + blanking level)."""
    line_signal[:] = BLANKING_V

    pos = 0
    line_signal[pos:pos + FRONT_PORCH_SAMPLES] = BLANKING_V
    pos += FRONT_PORCH_SAMPLES
    line_signal[pos:pos + HSYNC_SAMPLES] = SYNC_TIP_V
    pos += HSYNC_SAMPLES
    line_signal[pos:pos + BREEZEWAY_SAMPLES] = BLANKING_V
    pos += BREEZEWAY_SAMPLES

    burst_indices = np.arange(BURST_SAMPLES, dtype=np.float64) + pos
    burst_phase = _OMEGA_PER_SAMPLE * burst_indices + line_start_phase
    burst_v = BURST_AMPLITUDE_IRE / 140.0
    line_signal[pos:pos + BURST_SAMPLES] = BLANKING_V + (-np.cos(burst_phase) * burst_v)


def _encode_vblank_line(line_signal, line_num, line_start_phase):
    """Encode vertical blanking lines including vsync and equalizing pulses."""
    if line_num < 3:
        _write_eq_pulses(line_signal)
    elif line_num < 6:
        _write_vsync_pulses(line_signal)
    elif line_num < 9:
        _write_eq_pulses(line_signal)
    elif line_num < 20:
        _encode_blank_line(line_signal, line_start_phase)
    elif 262 <= line_num < 265:
        _write_eq_pulses(line_signal)
    elif 265 <= line_num < 268:
        _write_vsync_pulses(line_signal)
    elif 268 <= line_num < 271:
        _write_eq_pulses(line_signal)
    else:
        _encode_blank_line(line_signal, line_start_phase)


def _write_eq_pulses(line_signal):
    """Write two equalizing pulses per line (half-line rate)."""
    line_signal[:] = BLANKING_V
    line_signal[0:EQ_PULSE_SAMPLES] = SYNC_TIP_V
    line_signal[HALF_LINE_SAMPLES:HALF_LINE_SAMPLES + EQ_PULSE_SAMPLES] = SYNC_TIP_V


def _write_vsync_pulses(line_signal):
    """Write two broad (vertical sync) pulses per line."""
    line_signal[:] = BLANKING_V
    line_signal[0:VSYNC_PULSE_SAMPLES] = SYNC_TIP_V
    line_signal[HALF_LINE_SAMPLES:HALF_LINE_SAMPLES + VSYNC_PULSE_SAMPLES] = SYNC_TIP_V
