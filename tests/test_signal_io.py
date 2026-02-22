"""Tests for ntsc_simulator.signal_io."""

import struct

import numpy as np
import pytest

from ntsc_simulator.signal_io import export_signal, import_signal, export_wav
from ntsc_simulator.constants import SAMPLE_RATE


class TestExportImportRoundtrip:
    def test_roundtrip_preserves_data(self, tmp_path):
        signal = np.random.default_rng(42).random(1000).astype(np.float32)
        filepath = str(tmp_path / "test.npy")

        export_signal(signal, filepath)
        loaded, sr = import_signal(filepath)

        np.testing.assert_allclose(loaded, signal.astype(np.float64), atol=1e-6)
        assert sr == int(SAMPLE_RATE)

    def test_roundtrip_preserves_shape(self, tmp_path):
        signal = np.ones(500, dtype=np.float32)
        filepath = str(tmp_path / "test.npy")

        export_signal(signal, filepath)
        loaded, _ = import_signal(filepath)

        assert loaded.shape == signal.shape

    def test_stored_as_float64(self, tmp_path):
        signal = np.ones(100, dtype=np.float32)
        filepath = str(tmp_path / "test.npy")
        export_signal(signal, filepath)
        raw = np.load(filepath)
        assert raw.dtype == np.float64


class TestExportWav:
    def test_creates_valid_wav(self, tmp_path):
        signal = np.random.default_rng(42).random(1000).astype(np.float32)
        filepath = str(tmp_path / "test.wav")

        export_wav(signal, filepath, sample_rate=48000)

        with open(filepath, 'rb') as f:
            data = f.read()

        # Check RIFF header
        assert data[:4] == b'RIFF'
        assert data[8:12] == b'WAVE'

        # Check fmt chunk
        assert data[12:16] == b'fmt '
        fmt_size = struct.unpack('<I', data[16:20])[0]
        assert fmt_size == 16

        # Audio format = 3 (IEEE float)
        audio_format = struct.unpack('<H', data[20:22])[0]
        assert audio_format == 3

        # Channels = 1
        channels = struct.unpack('<H', data[22:24])[0]
        assert channels == 1

        # Sample rate
        sr = struct.unpack('<I', data[24:28])[0]
        assert sr == 48000

        # Bits per sample = 32
        bps = struct.unpack('<H', data[34:36])[0]
        assert bps == 32

    def test_correct_data_size(self, tmp_path):
        n_samples = 500
        signal = np.ones(n_samples, dtype=np.float32) * 0.5
        filepath = str(tmp_path / "test.wav")

        export_wav(signal, filepath)

        with open(filepath, 'rb') as f:
            data = f.read()

        # data chunk starts at offset 36
        assert data[36:40] == b'data'
        data_size = struct.unpack('<I', data[40:44])[0]
        expected = n_samples * 4  # 32-bit float = 4 bytes
        assert data_size == expected

    def test_custom_sample_rate(self, tmp_path):
        signal = np.ones(100, dtype=np.float32)
        filepath = str(tmp_path / "test.wav")

        export_wav(signal, filepath, sample_rate=44100)

        with open(filepath, 'rb') as f:
            f.seek(24)
            sr = struct.unpack('<I', f.read(4))[0]
        assert sr == 44100
