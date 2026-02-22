"""NumPy file export/import for composite NTSC signals."""

import struct

import numpy as np

from .constants import SAMPLE_RATE


def export_signal(signal, filepath):
    """Export a composite signal as a NumPy .npy file.

    Saves the raw float64 signal array directly.

    Args:
        signal: 1D numpy array of composite signal.
        filepath: Output .npy file path.
    """
    np.save(filepath, signal.astype(np.float64))


def import_signal(filepath):
    """Import a composite signal from a NumPy .npy file.

    Args:
        filepath: Input .npy file path.

    Returns:
        Tuple of (signal, sample_rate) where signal is a 1D numpy array
        in voltage range [0, 1].
    """
    signal = np.load(filepath).astype(np.float64)
    return signal, int(SAMPLE_RATE)


def export_wav(signal, filepath, sample_rate=48000):
    """Export the composite signal as a WAV file for viewing in audio editors.

    Every sample is preserved — the WAV header simply declares a standard
    audio sample rate so programs like Audacity can open it.  The signal
    plays back much slower than real-time but every sample is visible.

    Args:
        signal: 1D numpy array of composite signal in [0, 1] voltage range.
        filepath: Output WAV file path.
        sample_rate: Declared sample rate in the WAV header (default 48000).
    """
    audio = np.clip(signal.astype(np.float64) * 2.0 - 1.0, -1.0, 1.0)
    audio_bytes = audio.astype(np.float32).tobytes()

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
