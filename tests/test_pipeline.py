"""Tests for ntsc_simulator.pipeline."""

import numpy as np
import pytest

from ntsc_simulator.pipeline import SignalPipeline
from ntsc_simulator.constants import SAMPLE_RATE


class TestSignalPipeline:
    def test_empty_pipeline_length(self):
        p = SignalPipeline()
        assert len(p) == 0

    def test_add_increases_length(self):
        p = SignalPipeline()
        p.add(lambda s, sr: s)
        assert len(p) == 1
        p.add(lambda s, sr: s)
        assert len(p) == 2

    def test_add_returns_self(self):
        p = SignalPipeline()
        result = p.add(lambda s, sr: s)
        assert result is p

    def test_chaining(self):
        p = SignalPipeline()
        p.add(lambda s, sr: s).add(lambda s, sr: s).add(lambda s, sr: s)
        assert len(p) == 3

    def test_process_applies_in_order(self):
        p = SignalPipeline()
        p.add(lambda s, sr: s * 2)
        p.add(lambda s, sr: s + 1)

        signal = np.array([1.0, 2.0, 3.0])
        result = p.process(signal)
        # (signal * 2) + 1
        np.testing.assert_array_equal(result, [3.0, 5.0, 7.0])

    def test_process_empty_pipeline(self):
        p = SignalPipeline()
        signal = np.array([1.0, 2.0, 3.0])
        result = p.process(signal)
        np.testing.assert_array_equal(result, signal)

    def test_process_default_sample_rate(self):
        received_rates = []
        def capture_rate(s, sr):
            received_rates.append(sr)
            return s
        p = SignalPipeline()
        p.add(capture_rate)
        p.process(np.array([1.0]))
        assert received_rates[0] == pytest.approx(SAMPLE_RATE)

    def test_process_custom_sample_rate(self):
        received_rates = []
        def capture_rate(s, sr):
            received_rates.append(sr)
            return s
        p = SignalPipeline()
        p.add(capture_rate)
        p.process(np.array([1.0]), sample_rate=48000)
        assert received_rates[0] == 48000

    def test_clear(self):
        p = SignalPipeline()
        p.add(lambda s, sr: s * 2)
        p.add(lambda s, sr: s + 1)
        assert len(p) == 2
        p.clear()
        assert len(p) == 0

    def test_clear_then_process_is_identity(self):
        p = SignalPipeline()
        p.add(lambda s, sr: s * 99)
        p.clear()
        signal = np.array([1.0, 2.0])
        result = p.process(signal)
        np.testing.assert_array_equal(result, signal)
