"""Integration tests: encode-then-decode roundtrips."""

import numpy as np
import pytest

from ntsc_simulator.encoder import encode_frame
from ntsc_simulator.decoder import decode_frame
from ntsc_simulator.pipeline import SignalPipeline
from ntsc_simulator.effects import add_noise, add_ghosting, add_attenuation
from ntsc_simulator.colorbars import generate_colorbars


class TestRoundtrip:
    def test_solid_color_roundtrip(self):
        """Encode a solid color, decode, and check the mean color is close."""
        # Solid mid-gray
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        signal = encode_frame(frame, frame_number=0)
        decoded = decode_frame(signal, frame_number=0,
                               output_width=640, output_height=480)

        # Mean of decoded should be near 128 (within NTSC degradation tolerance)
        mean_color = decoded.mean(axis=(0, 1))
        np.testing.assert_allclose(mean_color, 128, atol=30)

    def test_colorbars_roundtrip_shape(self):
        """Encode colorbars and decode — verify output shape and dtype."""
        bars = generate_colorbars(640, 480)
        signal = encode_frame(bars)
        decoded = decode_frame(signal, output_width=640, output_height=480)

        assert decoded.shape == (480, 640, 3)
        assert decoded.dtype == np.uint8

    def test_colorbars_not_black(self):
        """Decoded colorbars should not be all black."""
        bars = generate_colorbars(640, 480)
        signal = encode_frame(bars)
        decoded = decode_frame(signal, output_width=640, output_height=480)

        assert decoded.mean() > 10

    def test_roundtrip_with_effects(self):
        """Roundtrip with noise + attenuation still produces valid output."""
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        signal = encode_frame(frame)

        pipeline = SignalPipeline()
        pipeline.add(add_noise(0.02))
        pipeline.add(add_attenuation(0.1))
        signal = pipeline.process(signal)

        decoded = decode_frame(signal, output_width=640, output_height=480)
        assert decoded.shape == (480, 640, 3)
        assert decoded.dtype == np.uint8
        assert decoded.max() > 0  # Not all black

    def test_field2_frame_roundtrip(self):
        """Encoding with separate field2_frame works end-to-end."""
        frame1 = np.full((64, 64, 3), 100, dtype=np.uint8)
        frame2 = np.full((64, 64, 3), 200, dtype=np.uint8)
        signal = encode_frame(frame1, frame_number=0, field2_frame=frame2)
        decoded = decode_frame(signal, frame_number=0,
                               output_width=640, output_height=480)

        assert decoded.shape == (480, 640, 3)
        assert decoded.dtype == np.uint8

    def test_white_brighter_than_black(self):
        """White frame should decode brighter than black frame."""
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        black = np.zeros((64, 64, 3), dtype=np.uint8)

        sig_w = encode_frame(white)
        sig_b = encode_frame(black)

        dec_w = decode_frame(sig_w, output_width=640, output_height=480)
        dec_b = decode_frame(sig_b, output_width=640, output_height=480)

        assert dec_w.mean() > dec_b.mean()

    def test_roundtrip_with_ghosting(self):
        """Roundtrip with ghosting effect produces valid output."""
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        signal = encode_frame(frame)

        pipeline = SignalPipeline()
        pipeline.add(add_ghosting(0.3, delay_us=2.0))
        signal = pipeline.process(signal)

        decoded = decode_frame(signal, output_width=640, output_height=480)
        assert decoded.shape == (480, 640, 3)
        assert decoded.dtype == np.uint8
