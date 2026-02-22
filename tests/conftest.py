"""Shared fixtures for NTSC simulator tests."""

import numpy as np
import pytest

from ntsc_simulator.colorbars import generate_colorbars
from ntsc_simulator.encoder import encode_frame


@pytest.fixture
def sample_frame():
    """Small 64x64 RGB frame for fast tests."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def colorbars_frame():
    """SMPTE color bars at 640x480."""
    return generate_colorbars(640, 480)


@pytest.fixture
def encoded_signal(sample_frame):
    """Pre-encoded composite signal from sample_frame."""
    return encode_frame(sample_frame, frame_number=0)
