"""Tests for ntsc_simulator.encoder."""

import numpy as np
import pytest

from ntsc_simulator.encoder import (
    encode_frame, rgb_to_yiq, _resample_rows,
    _build_visible_line_map, _build_line_to_visible,
)
from ntsc_simulator.constants import (
    TOTAL_LINES, SAMPLES_PER_LINE, VISIBLE_LINES, ACTIVE_SAMPLES,
)


class TestRgbToYiq:
    def test_output_shape(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        yiq = rgb_to_yiq(frame)
        assert yiq.shape == (64, 64, 3)

    def test_output_dtype(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        yiq = rgb_to_yiq(frame)
        assert yiq.dtype == np.float32

    def test_pure_white(self):
        frame = np.full((1, 1, 3), 255, dtype=np.uint8)
        yiq = rgb_to_yiq(frame)
        assert yiq[0, 0, 0] == pytest.approx(1.0, abs=0.01)  # Y ≈ 1
        assert abs(yiq[0, 0, 1]) < 0.02  # I ≈ 0
        assert abs(yiq[0, 0, 2]) < 0.02  # Q ≈ 0

    def test_pure_black(self):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        yiq = rgb_to_yiq(frame)
        assert yiq[0, 0, 0] == pytest.approx(0.0, abs=0.01)
        assert yiq[0, 0, 1] == pytest.approx(0.0, abs=0.01)
        assert yiq[0, 0, 2] == pytest.approx(0.0, abs=0.01)

    def test_red_has_positive_i(self):
        frame = np.array([[[255, 0, 0]]], dtype=np.uint8)
        yiq = rgb_to_yiq(frame)
        assert yiq[0, 0, 1] > 0  # I is positive for red


class TestResampleRows:
    def test_identity_same_width(self):
        data = np.random.default_rng(0).random((10, 50)).astype(np.float32)
        result = _resample_rows(data, 50)
        np.testing.assert_array_equal(result, data)

    def test_output_shape(self):
        data = np.random.default_rng(0).random((10, 50)).astype(np.float32)
        result = _resample_rows(data, 100)
        assert result.shape == (10, 100)

    def test_upsample_preserves_endpoints(self):
        data = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        result = _resample_rows(data, 7)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert result[0, -1] == pytest.approx(2.0, abs=1e-5)


class TestBuildVisibleLineMap:
    def test_output_shape(self):
        mapping = _build_visible_line_map(480)
        assert mapping.shape == (VISIBLE_LINES,)

    def test_identity_at_480(self):
        mapping = _build_visible_line_map(480)
        np.testing.assert_array_equal(mapping, np.arange(480))

    def test_values_in_range(self):
        for h in [64, 240, 480, 720]:
            mapping = _build_visible_line_map(h)
            assert np.all(mapping >= 0)
            assert np.all(mapping < h)


class TestBuildLineToVisible:
    def test_output_shape(self):
        mapping = _build_line_to_visible()
        assert mapping.shape == (TOTAL_LINES,)

    def test_blanking_lines_are_negative(self):
        mapping = _build_line_to_visible()
        # Lines 0-19 are blanking (field 1 vblank)
        assert np.all(mapping[:20] == -1)

    def test_visible_range(self):
        mapping = _build_line_to_visible()
        visible_values = mapping[mapping >= 0]
        assert visible_values.min() == 0
        assert visible_values.max() == VISIBLE_LINES - 1

    def test_field1_even_field2_odd(self):
        mapping = _build_line_to_visible()
        # Field 1 lines 20-259 map to even visible indices
        field1 = mapping[20:260]
        assert np.all(field1 % 2 == 0)
        # Field 2 lines 283-522 map to odd visible indices
        field2 = mapping[283:523]
        assert np.all(field2 % 2 == 1)


class TestEncodeFrame:
    def test_output_is_1d(self, sample_frame):
        signal = encode_frame(sample_frame)
        assert signal.ndim == 1

    def test_output_length(self, sample_frame):
        signal = encode_frame(sample_frame)
        assert len(signal) == TOTAL_LINES * SAMPLES_PER_LINE

    def test_output_dtype(self, sample_frame):
        signal = encode_frame(sample_frame)
        assert signal.dtype == np.float32

    def test_voltage_range(self, sample_frame):
        signal = encode_frame(sample_frame)
        # Sync tip can go to 0, white up to 1, but chroma can overshoot slightly
        assert signal.min() >= -0.1
        assert signal.max() <= 1.2

    def test_blanking_level_present(self, sample_frame):
        signal = encode_frame(sample_frame)
        from ntsc_simulator.constants import BLANKING_V
        # Many samples should be near blanking level (horizontal blanking regions)
        near_blanking = np.abs(signal - BLANKING_V) < 0.02
        assert np.sum(near_blanking) > 1000

    def test_different_frame_numbers(self, sample_frame):
        s0 = encode_frame(sample_frame, frame_number=0)
        s1 = encode_frame(sample_frame, frame_number=1)
        # Phase alternates per frame, so signals differ
        assert not np.array_equal(s0, s1)

    def test_field2_frame(self, sample_frame):
        rng = np.random.default_rng(99)
        field2 = rng.integers(0, 256, sample_frame.shape, dtype=np.uint8)
        signal = encode_frame(sample_frame, field2_frame=field2)
        assert signal.ndim == 1
        assert len(signal) == TOTAL_LINES * SAMPLES_PER_LINE
