"""Signal degradation effects for simulating weak/noisy NTSC reception."""

import numpy as np

from .constants import BLANKING_V, TOTAL_LINES, SAMPLES_PER_LINE


def add_noise(amplitude):
    """Additive white Gaussian noise (snow).

    Args:
        amplitude: Noise amplitude in signal units (e.g. 0.05 = subtle,
            0.2 = heavy snow).

    Returns:
        Transform function: fn(signal, sample_rate) -> signal.
    """
    def transform(signal, sample_rate):
        noise = np.random.normal(0, amplitude, len(signal))
        return signal + noise

    return transform


def add_ghosting(amplitude, delay_us=2.0):
    """Multipath ghost — adds a delayed, attenuated copy of the signal.

    Args:
        amplitude: Ghost strength 0-1.
        delay_us: Ghost delay in microseconds (default 2.0 us, ~28 samples).

    Returns:
        Transform function: fn(signal, sample_rate) -> signal.
    """
    def transform(signal, sample_rate):
        delay_samples = int(round(delay_us * sample_rate / 1e6))
        if delay_samples <= 0 or delay_samples >= len(signal):
            return signal
        ghost = np.zeros_like(signal)
        ghost[delay_samples:] = signal[:-delay_samples]
        return signal + amplitude * ghost

    return transform


def add_attenuation(strength):
    """Compress signal toward blanking level, reducing contrast and saturation.

    Args:
        strength: 0 = no change, 1 = flat at blanking level.

    Returns:
        Transform function: fn(signal, sample_rate) -> signal.
    """
    def transform(signal, sample_rate):
        return BLANKING_V + (signal - BLANKING_V) * (1 - strength)

    return transform


def add_jitter(amplitude):
    """Per-line horizontal timing instability.

    Shifts lines by whole subcarrier cycles (4 samples) so the picture
    wobbles horizontally without altering decoded color. Uses a Gaussian
    distribution so most shifts are small with occasional larger ones.

    Args:
        amplitude: Standard deviation of shift in subcarrier cycles.
            0.1 = very subtle, 0.5 = moderate, 2.0 = heavy.

    Returns:
        Transform function: fn(signal, sample_rate) -> signal.
    """
    def transform(signal, sample_rate):
        expected = TOTAL_LINES * SAMPLES_PER_LINE
        if len(signal) < expected:
            return signal
        grid = signal[:expected].reshape(TOTAL_LINES, SAMPLES_PER_LINE)
        shifts = np.round(np.random.normal(0, amplitude, TOTAL_LINES)).astype(int)
        out = np.empty_like(grid)
        for line in range(TOTAL_LINES):
            cycles = shifts[line]
            out[line] = np.roll(grid[line], cycles * 4) if cycles else grid[line]
        result = out.flatten()
        if len(signal) > expected:
            result = np.concatenate([result, signal[expected:]])
        return result

    return transform
