"""Tests for ntsc_simulator.colorbars."""

import numpy as np
import pytest

from ntsc_simulator.colorbars import generate_colorbars


class TestGenerateColorbars:
    def test_output_shape(self):
        frame = generate_colorbars()
        assert frame.shape == (480, 640, 3)

    def test_output_dtype(self):
        frame = generate_colorbars()
        assert frame.dtype == np.uint8

    def test_custom_dimensions(self):
        frame = generate_colorbars(width=320, height=240)
        assert frame.shape == (240, 320, 3)

    def test_top_left_is_white_75(self):
        frame = generate_colorbars(width=640, height=480)
        # Top-left corner should be white at 75% (191, 191, 191)
        np.testing.assert_array_equal(frame[0, 0], [191, 191, 191])

    def test_first_bar_is_white(self):
        frame = generate_colorbars(width=700, height=480)
        bar_width = 700 // 7  # 100 pixels per bar
        # Sample middle of first bar, top region
        mid_x = bar_width // 2
        np.testing.assert_array_equal(frame[10, mid_x], [191, 191, 191])

    def test_last_bar_is_blue(self):
        frame = generate_colorbars(width=700, height=480)
        bar_width = 700 // 7
        # Last bar extends to width edge
        x = 6 * bar_width + bar_width // 2
        np.testing.assert_array_equal(frame[10, x], [0, 0, 191])

    def test_bar_order(self):
        """Verify the 7-bar color order: White, Yellow, Cyan, Green, Magenta, Red, Blue."""
        frame = generate_colorbars(width=700, height=480)
        bar_width = 700 // 7
        expected = [
            [191, 191, 191],  # White
            [191, 191, 0],    # Yellow
            [0, 191, 191],    # Cyan
            [0, 191, 0],      # Green
            [191, 0, 191],    # Magenta
            [191, 0, 0],      # Red
            [0, 0, 191],      # Blue
        ]
        for i, color in enumerate(expected):
            x = i * bar_width + bar_width // 2
            np.testing.assert_array_equal(frame[10, x], color,
                                          err_msg=f"Bar {i} mismatch")

    def test_large_dimensions(self):
        frame = generate_colorbars(width=1920, height=1080)
        assert frame.shape == (1080, 1920, 3)
        assert frame.dtype == np.uint8
