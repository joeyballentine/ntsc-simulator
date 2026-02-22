"""Tests for ntsc_simulator.filters."""

import numpy as np
import pytest

from ntsc_simulator.filters import (
    design_lowpass, design_bandpass,
    apply_filter_zero_phase, apply_filter_causal,
    get_filter, _filter_cache,
    lowpass_luma, lowpass_i, lowpass_q, bandpass_chroma,
)
from ntsc_simulator.constants import SAMPLE_RATE


class TestDesignLowpass:
    def test_shape(self):
        coeffs = design_lowpass(1e6, num_taps=101)
        assert coeffs.shape == (101,)

    def test_dtype(self):
        coeffs = design_lowpass(1e6)
        assert coeffs.dtype == np.float32

    def test_sums_to_one(self):
        coeffs = design_lowpass(1e6, num_taps=101)
        assert np.sum(coeffs) == pytest.approx(1.0, abs=0.01)

    def test_different_taps(self):
        c51 = design_lowpass(1e6, num_taps=51)
        c201 = design_lowpass(1e6, num_taps=201)
        assert c51.shape == (51,)
        assert c201.shape == (201,)


class TestDesignBandpass:
    def test_shape(self):
        coeffs = design_bandpass(1e6, 3e6, num_taps=101)
        assert coeffs.shape == (101,)

    def test_dtype(self):
        coeffs = design_bandpass(1e6, 3e6)
        assert coeffs.dtype == np.float32

    def test_dc_rejected(self):
        # Bandpass should reject DC (sum ≈ 0)
        coeffs = design_bandpass(1e6, 3e6, num_taps=101)
        assert abs(np.sum(coeffs)) < 0.01


class TestApplyFilterZeroPhase:
    def test_preserves_dc(self):
        coeffs = design_lowpass(1e6, num_taps=51)
        dc_signal = np.ones(1000, dtype=np.float32) * 0.5
        filtered = apply_filter_zero_phase(coeffs, dc_signal)
        # Middle of signal should be close to DC value
        mid = len(filtered) // 2
        np.testing.assert_allclose(filtered[mid - 100:mid + 100], 0.5, atol=0.01)

    def test_attenuates_high_freq(self):
        coeffs = design_lowpass(1e6, num_taps=101)
        t = np.arange(4000, dtype=np.float32) / SAMPLE_RATE
        # Signal at 5 MHz (above 1 MHz cutoff)
        high_freq = np.sin(2 * np.pi * 5e6 * t).astype(np.float32)
        filtered = apply_filter_zero_phase(coeffs, high_freq)
        # RMS of filtered should be much less than RMS of input
        rms_in = np.sqrt(np.mean(high_freq**2))
        rms_out = np.sqrt(np.mean(filtered[200:-200]**2))
        assert rms_out < rms_in * 0.1

    def test_output_length(self):
        coeffs = design_lowpass(1e6, num_taps=51)
        signal = np.random.default_rng(0).random(500).astype(np.float32)
        filtered = apply_filter_zero_phase(coeffs, signal)
        assert len(filtered) == len(signal)


class TestApplyFilterCausal:
    def test_preserves_dc(self):
        coeffs = design_lowpass(1e6, num_taps=51)
        dc_signal = np.ones(1000, dtype=np.float32) * 0.5
        filtered = apply_filter_causal(coeffs, dc_signal)
        # After settling (past filter length), should be close to DC value
        np.testing.assert_allclose(filtered[200:], 0.5, atol=0.01)

    def test_output_length(self):
        coeffs = design_lowpass(1e6, num_taps=51)
        signal = np.random.default_rng(0).random(500).astype(np.float32)
        filtered = apply_filter_causal(coeffs, signal)
        assert len(filtered) == len(signal)


class TestGetFilter:
    def test_known_names(self):
        for name in ['luma', 'i_channel', 'q_channel', 'chroma_bandpass']:
            coeffs = get_filter(name)
            assert isinstance(coeffs, np.ndarray)
            assert coeffs.dtype == np.float32

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown filter"):
            get_filter('nonexistent')

    def test_caching(self):
        c1 = get_filter('luma', num_taps=101)
        c2 = get_filter('luma', num_taps=101)
        assert c1 is c2  # Same object from cache

    def test_different_taps_not_cached_together(self):
        c1 = get_filter('luma', num_taps=51)
        c2 = get_filter('luma', num_taps=101)
        assert c1 is not c2
        assert c1.shape != c2.shape


class TestConvenienceFunctions:
    def test_lowpass_luma_length(self):
        signal = np.random.default_rng(0).random(1000).astype(np.float32)
        result = lowpass_luma(signal)
        assert len(result) == len(signal)

    def test_lowpass_i_length(self):
        signal = np.random.default_rng(0).random(1000).astype(np.float32)
        result = lowpass_i(signal)
        assert len(result) == len(signal)

    def test_lowpass_q_length(self):
        signal = np.random.default_rng(0).random(1000).astype(np.float32)
        result = lowpass_q(signal)
        assert len(result) == len(signal)

    def test_bandpass_chroma_length(self):
        signal = np.random.default_rng(0).random(1000).astype(np.float32)
        result = bandpass_chroma(signal)
        assert len(result) == len(signal)

    def test_causal_mode(self):
        signal = np.random.default_rng(0).random(1000).astype(np.float32)
        zp = lowpass_luma(signal, zero_phase=True)
        causal = lowpass_luma(signal, zero_phase=False)
        # They should differ (causal has phase delay)
        assert not np.allclose(zp, causal)
