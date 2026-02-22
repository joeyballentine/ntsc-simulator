"""NTSC Composite Video Simulator."""

from .encoder import encode_frame
from .decoder import decode_frame
from .pipeline import SignalPipeline
from .signal_io import export_signal, import_signal, export_wav
from .colorbars import generate_colorbars
from .effects import add_noise, add_ghosting, add_attenuation, add_jitter
