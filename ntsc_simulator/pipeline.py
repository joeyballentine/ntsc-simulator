"""Middleware pipeline for signal transforms between encode and decode."""

from .constants import SAMPLE_RATE


class SignalPipeline:
    """A chain of signal transforms applied to the composite signal.

    Each transform is a callable: fn(signal_array, sample_rate) -> signal_array
    """

    def __init__(self):
        self.transforms = []

    def add(self, transform_fn):
        """Add a transform to the pipeline.

        Args:
            transform_fn: Callable (signal, sample_rate) -> signal.
        """
        self.transforms.append(transform_fn)
        return self  # Allow chaining

    def process(self, signal, sample_rate=None):
        """Apply all transforms to the signal in order.

        Args:
            signal: 1D numpy array of composite signal.
            sample_rate: Sample rate (defaults to NTSC 4xfsc).

        Returns:
            Transformed signal array.
        """
        if sample_rate is None:
            sample_rate = SAMPLE_RATE
        for fn in self.transforms:
            signal = fn(signal, sample_rate)
        return signal

    def clear(self):
        """Remove all transforms."""
        self.transforms.clear()

    def __len__(self):
        return len(self.transforms)
