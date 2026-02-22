"""Tests for ntsc_simulator.decoder."""

import numpy as np
import pytest

from ntsc_simulator.decoder import decode_frame
from ntsc_simulator.encoder import encode_frame
from ntsc_simulator.constants import TOTAL_LINES, SAMPLES_PER_LINE


class TestDecodeFrame:
    def test_output_shape_default(self, encoded_signal):
        frame = decode_frame(encoded_signal)
        assert frame.shape == (480, 640, 3)

    def test_output_shape_custom(self, encoded_signal):
        frame = decode_frame(encoded_signal, output_width=320, output_height=240)
        assert frame.shape == (240, 320, 3)

    def test_output_dtype(self, encoded_signal):
        frame = decode_frame(encoded_signal)
        assert frame.dtype == np.uint8

    def test_values_in_range(self, encoded_signal):
        frame = decode_frame(encoded_signal)
        assert frame.min() >= 0
        assert frame.max() <= 255

    def test_comb_1h_mode(self, encoded_signal):
        frame = decode_frame(encoded_signal, comb_1h=True)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    def test_both_comb_modes_valid(self, encoded_signal):
        frame_default = decode_frame(encoded_signal, comb_1h=False)
        frame_comb = decode_frame(encoded_signal, comb_1h=True)
        # Both produce valid output
        assert frame_default.dtype == np.uint8
        assert frame_comb.dtype == np.uint8
        # But results differ
        assert not np.array_equal(frame_default, frame_comb)

    def test_frame_number_accepted(self, encoded_signal):
        # Decoding with different frame_number should not error
        f0 = decode_frame(encoded_signal, frame_number=0)
        f1 = decode_frame(encoded_signal, frame_number=5)
        assert f0.shape == f1.shape
        assert f0.dtype == np.uint8
