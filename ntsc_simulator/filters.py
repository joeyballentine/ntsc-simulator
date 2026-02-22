"""FIR low-pass and bandpass filters for NTSC signal processing."""

import numpy as np
from scipy.signal import firwin, lfilter
from scipy.fft import rfft, irfft, next_fast_len

from .constants import SAMPLE_RATE


def design_lowpass(cutoff_hz, num_taps=101):
    """Design a FIR low-pass filter.

    Args:
        cutoff_hz: Cutoff frequency in Hz.
        num_taps: Number of filter taps (odd for symmetric).

    Returns:
        FIR filter coefficients (1D array, float32).
    """
    nyquist = SAMPLE_RATE / 2
    return firwin(num_taps, cutoff_hz / nyquist).astype(np.float32)


def design_bandpass(low_hz, high_hz, num_taps=101):
    """Design a FIR bandpass filter.

    Args:
        low_hz: Lower cutoff frequency in Hz.
        high_hz: Upper cutoff frequency in Hz.
        num_taps: Number of filter taps (odd for symmetric).

    Returns:
        FIR filter coefficients (1D array, float32).
    """
    nyquist = SAMPLE_RATE / 2
    return firwin(num_taps, [low_hz / nyquist, high_hz / nyquist],
                  pass_zero=False).astype(np.float32)


def apply_filter_zero_phase(coeffs, signal):
    """Apply a FIR filter with zero phase distortion using FFT.

    Uses |H(f)|^2 multiplication in frequency domain, equivalent to
    scipy.signal.filtfilt but faster for longer filters.
    """
    n = len(signal)
    fft_n = next_fast_len(n + len(coeffs) - 1)
    H = rfft(coeffs, n=fft_n)
    H2 = (H * np.conj(H)).real
    X = rfft(signal, n=fft_n)
    return irfft(X * H2, n=fft_n)[:n]


def apply_filter_causal(coeffs, signal):
    """Apply a FIR filter causally (introduces phase delay).

    Used during decoding to simulate real-time processing.
    """
    return lfilter(coeffs, 1.0, signal)


# Pre-designed filters (lazily cached)
_filter_cache = {}


def get_filter(name, num_taps=101):
    """Get a pre-designed filter by name.

    Names: 'luma', 'i_channel', 'q_channel', 'chroma_bandpass'
    """
    from .constants import LUMA_BW, I_BW, Q_BW, CHROMA_BW_LOW, CHROMA_BW_HIGH

    key = (name, num_taps)
    if key not in _filter_cache:
        if name == 'luma':
            _filter_cache[key] = design_lowpass(LUMA_BW, num_taps)
        elif name == 'i_channel':
            _filter_cache[key] = design_lowpass(I_BW, num_taps)
        elif name == 'q_channel':
            _filter_cache[key] = design_lowpass(Q_BW, num_taps)
        elif name == 'chroma_bandpass':
            _filter_cache[key] = design_bandpass(CHROMA_BW_LOW, CHROMA_BW_HIGH, num_taps)
        else:
            raise ValueError(f"Unknown filter: {name}")
    return _filter_cache[key]


def lowpass_luma(signal, zero_phase=True, num_taps=101):
    """Apply luma low-pass filter (4.2 MHz)."""
    coeffs = get_filter('luma', num_taps)
    if zero_phase:
        return apply_filter_zero_phase(coeffs, signal)
    return apply_filter_causal(coeffs, signal)


def lowpass_i(signal, zero_phase=True, num_taps=101):
    """Apply I channel low-pass filter (1.5 MHz)."""
    coeffs = get_filter('i_channel', num_taps)
    if zero_phase:
        return apply_filter_zero_phase(coeffs, signal)
    return apply_filter_causal(coeffs, signal)


def lowpass_q(signal, zero_phase=True, num_taps=101):
    """Apply Q channel low-pass filter (0.5 MHz)."""
    coeffs = get_filter('q_channel', num_taps)
    if zero_phase:
        return apply_filter_zero_phase(coeffs, signal)
    return apply_filter_causal(coeffs, signal)


def bandpass_chroma(signal, zero_phase=False, num_taps=101):
    """Apply chroma bandpass filter (~2-4.2 MHz)."""
    coeffs = get_filter('chroma_bandpass', num_taps)
    if zero_phase:
        return apply_filter_zero_phase(coeffs, signal)
    return apply_filter_causal(coeffs, signal)
