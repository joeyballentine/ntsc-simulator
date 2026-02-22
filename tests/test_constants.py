"""Tests for ntsc_simulator.constants."""

import numpy as np
import pytest

from ntsc_simulator.constants import (
    FSC, SAMPLE_RATE, TOTAL_LINES, VISIBLE_LINES, SAMPLES_PER_LINE,
    FRAME_RATE, FIELD_RATE, LINE_FREQ,
    SYNC_TIP_IRE, BLANKING_IRE, WHITE_IRE,
    SYNC_TIP_V, BLANKING_V, WHITE_V,
    RGB_TO_YIQ, YIQ_TO_RGB,
    LUMA_BW, I_BW, Q_BW,
    GAMMA,
    ire_to_voltage, voltage_to_ire, _us_to_samples,
)


class TestIreVoltageConversion:
    def test_roundtrip_identity(self):
        for ire in [-40, -20, 0, 7.5, 50, 100]:
            assert voltage_to_ire(ire_to_voltage(ire)) == pytest.approx(ire)

    def test_sync_tip(self):
        assert ire_to_voltage(SYNC_TIP_IRE) == pytest.approx(SYNC_TIP_V, abs=1e-4)

    def test_blanking(self):
        assert ire_to_voltage(BLANKING_IRE) == pytest.approx(BLANKING_V, abs=1e-4)

    def test_white(self):
        assert ire_to_voltage(WHITE_IRE) == pytest.approx(WHITE_V, abs=1e-4)

    def test_voltage_to_ire_sync_tip(self):
        assert voltage_to_ire(SYNC_TIP_V) == pytest.approx(SYNC_TIP_IRE, abs=1e-4)

    def test_voltage_to_ire_white(self):
        assert voltage_to_ire(WHITE_V) == pytest.approx(WHITE_IRE, abs=1e-4)


class TestUsToSamples:
    def test_one_line(self):
        # One line is ~63.556 us -> 910 samples
        assert _us_to_samples(63.556) == SAMPLES_PER_LINE

    def test_zero(self):
        assert _us_to_samples(0) == 0

    def test_positive(self):
        # 1 us at ~14.318 MHz -> ~14 samples
        result = _us_to_samples(1.0)
        assert isinstance(result, int)
        assert result == round(SAMPLE_RATE / 1e6)


class TestColorMatrices:
    def test_rgb_to_yiq_shape(self):
        assert RGB_TO_YIQ.shape == (3, 3)

    def test_yiq_to_rgb_shape(self):
        assert YIQ_TO_RGB.shape == (3, 3)

    def test_approximate_inverse(self):
        product = YIQ_TO_RGB @ RGB_TO_YIQ
        np.testing.assert_allclose(product, np.eye(3), atol=0.02)

    def test_white_rgb_to_yiq(self):
        white_rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        yiq = RGB_TO_YIQ @ white_rgb
        assert yiq[0] == pytest.approx(1.0, abs=0.01)  # Y = 1
        assert abs(yiq[1]) < 0.01  # I ≈ 0
        assert abs(yiq[2]) < 0.01  # Q ≈ 0

    def test_dtype(self):
        assert RGB_TO_YIQ.dtype == np.float32
        assert YIQ_TO_RGB.dtype == np.float32


class TestKeyConstants:
    def test_fsc(self):
        assert FSC == pytest.approx(3_579_545.06)

    def test_sample_rate(self):
        assert SAMPLE_RATE == pytest.approx(4 * FSC)

    def test_total_lines(self):
        assert TOTAL_LINES == 525

    def test_visible_lines(self):
        assert VISIBLE_LINES == 480

    def test_samples_per_line(self):
        assert SAMPLES_PER_LINE == 910

    def test_frame_rate(self):
        assert FRAME_RATE == pytest.approx(29.97, abs=0.01)

    def test_field_rate(self):
        assert FIELD_RATE == pytest.approx(59.94, abs=0.01)

    def test_luma_bw(self):
        assert LUMA_BW == 4.2e6

    def test_i_bw(self):
        assert I_BW == 1.5e6

    def test_q_bw(self):
        assert Q_BW == 0.5e6

    def test_gamma(self):
        assert GAMMA == 2.2
