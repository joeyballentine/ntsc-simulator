"""Tests for ntsc_simulator.effects."""

import numpy as np
import pytest

from ntsc_simulator.effects import add_noise, add_ghosting, add_attenuation, add_jitter
from ntsc_simulator.constants import (
    BLANKING_V, SAMPLE_RATE, TOTAL_LINES, SAMPLES_PER_LINE,
)


@pytest.fixture
def flat_signal():
    """A flat signal at blanking+0.3V, sized for one NTSC frame."""
    return np.full(TOTAL_LINES * SAMPLES_PER_LINE, BLANKING_V + 0.3,
                   dtype=np.float32)


class TestAddNoise:
    def test_output_shape(self, flat_signal):
        transform = add_noise(0.1)
        result = transform(flat_signal, SAMPLE_RATE)
        assert result.shape == flat_signal.shape

    def test_signal_differs(self, flat_signal):
        transform = add_noise(0.1)
        result = transform(flat_signal, SAMPLE_RATE)
        assert not np.array_equal(result, flat_signal)

    def test_mean_shift_is_small(self, flat_signal):
        transform = add_noise(0.05)
        result = transform(flat_signal, SAMPLE_RATE)
        mean_diff = abs(np.mean(result) - np.mean(flat_signal))
        assert mean_diff < 0.01

    def test_zero_amplitude(self, flat_signal):
        transform = add_noise(0.0)
        result = transform(flat_signal, SAMPLE_RATE)
        np.testing.assert_array_equal(result, flat_signal)


class TestAddGhosting:
    def test_output_shape(self, flat_signal):
        transform = add_ghosting(0.3, delay_us=2.0)
        result = transform(flat_signal, SAMPLE_RATE)
        assert result.shape == flat_signal.shape

    def test_echo_at_delay(self):
        # Create a signal with a single impulse
        n = TOTAL_LINES * SAMPLES_PER_LINE
        signal = np.zeros(n, dtype=np.float32)
        signal[1000] = 1.0

        delay_us = 2.0
        delay_samples = int(round(delay_us * SAMPLE_RATE / 1e6))
        transform = add_ghosting(0.5, delay_us=delay_us)
        result = transform(signal, SAMPLE_RATE)

        # Original impulse preserved
        assert result[1000] == pytest.approx(1.0, abs=1e-6)
        # Ghost at delay offset
        assert result[1000 + delay_samples] == pytest.approx(0.5, abs=1e-6)

    def test_zero_amplitude(self, flat_signal):
        transform = add_ghosting(0.0, delay_us=2.0)
        result = transform(flat_signal, SAMPLE_RATE)
        np.testing.assert_allclose(result, flat_signal, atol=1e-6)


class TestAddAttenuation:
    def test_zero_strength_no_change(self, flat_signal):
        transform = add_attenuation(0.0)
        result = transform(flat_signal, SAMPLE_RATE)
        np.testing.assert_allclose(result, flat_signal, atol=1e-7)

    def test_full_strength_flat_at_blanking(self, flat_signal):
        transform = add_attenuation(1.0)
        result = transform(flat_signal, SAMPLE_RATE)
        np.testing.assert_allclose(result, BLANKING_V, atol=1e-6)

    def test_half_strength(self, flat_signal):
        transform = add_attenuation(0.5)
        result = transform(flat_signal, SAMPLE_RATE)
        expected = BLANKING_V + (flat_signal - BLANKING_V) * 0.5
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_output_shape(self, flat_signal):
        transform = add_attenuation(0.5)
        result = transform(flat_signal, SAMPLE_RATE)
        assert result.shape == flat_signal.shape


class TestAddJitter:
    def test_output_shape(self, flat_signal):
        transform = add_jitter(0.5)
        result = transform(flat_signal, SAMPLE_RATE)
        assert result.shape == flat_signal.shape

    def test_signal_differs(self):
        # Use a signal with non-uniform per-line content
        rng = np.random.default_rng(42)
        signal = rng.random(TOTAL_LINES * SAMPLES_PER_LINE).astype(np.float32)
        transform = add_jitter(1.0)
        result = transform(signal, SAMPLE_RATE)
        assert not np.array_equal(result, signal)

    def test_short_signal_unchanged(self):
        short = np.ones(100, dtype=np.float32)
        transform = add_jitter(1.0)
        result = transform(short, SAMPLE_RATE)
        np.testing.assert_array_equal(result, short)
