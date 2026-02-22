"""WAV file export/import for composite NTSC signals."""

import struct

import numpy as np
from scipy.signal import resample_poly
from math import gcd

from .constants import SAMPLE_RATE


def export_wav(signal, filepath, sample_rate=None):
    """Export a composite signal as a 32-bit float WAV file.

    Writes a proper WAVE_FORMAT_IEEE_FLOAT file from scratch (format tag 3).
    The signal voltage range (0-1) is mapped to (-1, 1) audio range.

    Args:
        signal: 1D numpy array of composite signal.
        filepath: Output WAV file path.
        sample_rate: Sample rate (defaults to NTSC 4xfsc).
    """
    if sample_rate is None:
        sample_rate = int(SAMPLE_RATE)

    # Normalize signal from [0, 1] voltage to [-1, 1] audio range
    audio = np.clip(signal.astype(np.float64) * 2.0 - 1.0, -1.0, 1.0)
    _write_float_wav(audio.astype(np.float32).tobytes(), sample_rate, filepath)


def export_wav_preview(signal, filepath, preview_rate=48000):
    """Export a downsampled version of the composite signal for audio inspection.

    This resamples the 14.3 MHz signal down to a standard audio rate so it can
    be opened in tools like Audition/Audacity. Not usable for decoding — purely
    for visual waveform inspection.

    Args:
        signal: 1D numpy array of composite signal (at NTSC sample rate).
        filepath: Output WAV file path.
        preview_rate: Target sample rate (default 48000 Hz).
    """
    native_rate = int(SAMPLE_RATE)

    # Find rational resampling ratio: preview_rate / native_rate = up / down
    g = gcd(preview_rate, native_rate)
    up = preview_rate // g
    down = native_rate // g

    # resample_poly handles the anti-alias filtering internally
    audio = np.clip(signal.astype(np.float64) * 2.0 - 1.0, -1.0, 1.0)
    resampled = resample_poly(audio, up, down).astype(np.float32)

    _write_float_wav(resampled.tobytes(), preview_rate, filepath)


def _write_float_wav(audio_bytes, sample_rate, filepath):
    """Write raw float32 audio bytes as a WAVE_FORMAT_IEEE_FLOAT file."""
    num_channels = 1
    bits_per_sample = 32
    block_align = num_channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    data_size = len(audio_bytes)

    fmt_chunk = struct.pack('<4sIHHIIHH',
        b'fmt ', 16, 3, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
    )
    data_chunk_header = struct.pack('<4sI', b'data', data_size)
    riff_size = 4 + len(fmt_chunk) + len(data_chunk_header) + data_size

    with open(filepath, 'wb') as f:
        f.write(struct.pack('<4sI4s', b'RIFF', riff_size, b'WAVE'))
        f.write(fmt_chunk)
        f.write(data_chunk_header)
        f.write(audio_bytes)


def import_wav(filepath):
    """Import a composite signal from a WAV file.

    Supports both IEEE float (format 3) and PCM (format 1) WAV files.

    Args:
        filepath: Input WAV file path.

    Returns:
        Tuple of (signal, sample_rate) where signal is a 1D numpy array
        in voltage range [0, 1].
    """
    with open(filepath, 'rb') as f:
        # Read RIFF header
        riff_id, riff_size, wave_id = struct.unpack('<4sI4s', f.read(12))
        if riff_id != b'RIFF' or wave_id != b'WAVE':
            raise ValueError("Not a valid WAV file")

        format_tag = None
        sample_rate = None
        bits_per_sample = None
        audio_data = None

        # Read chunks
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id, chunk_size = struct.unpack('<4sI', chunk_header)

            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                format_tag = struct.unpack('<H', fmt_data[0:2])[0]
                num_channels = struct.unpack('<H', fmt_data[2:4])[0]
                sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
            elif chunk_id == b'data':
                audio_data = f.read(chunk_size)
            else:
                # Skip unknown chunks
                f.read(chunk_size)
                # WAV chunks are word-aligned
                if chunk_size % 2 != 0:
                    f.read(1)

        if format_tag is None or audio_data is None:
            raise ValueError("WAV file missing fmt or data chunk")

        if format_tag == 3:  # IEEE float
            if bits_per_sample == 32:
                audio = np.frombuffer(audio_data, dtype=np.float32).astype(np.float64)
            elif bits_per_sample == 64:
                audio = np.frombuffer(audio_data, dtype=np.float64)
            else:
                raise ValueError(f"Unsupported float bit depth: {bits_per_sample}")
        elif format_tag == 1:  # PCM
            if bits_per_sample == 16:
                audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float64) / 32768.0
            elif bits_per_sample == 32:
                audio = np.frombuffer(audio_data, dtype=np.int32).astype(np.float64) / 2147483648.0
            else:
                raise ValueError(f"Unsupported PCM bit depth: {bits_per_sample}")
        else:
            raise ValueError(f"Unsupported WAV format tag: {format_tag}")

    # Convert from [-1, 1] audio range back to [0, 1] voltage range
    signal = (audio + 1.0) / 2.0

    return signal, sample_rate
