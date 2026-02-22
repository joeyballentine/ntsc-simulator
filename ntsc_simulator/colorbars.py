"""SMPTE color bar test pattern generator."""

import numpy as np


def generate_colorbars(width=640, height=480):
    """Generate a standard SMPTE color bar test pattern.

    The pattern consists of 7 vertical bars (left to right):
    White, Yellow, Cyan, Green, Magenta, Red, Blue

    Args:
        width: Output image width.
        height: Output image height.

    Returns:
        RGB frame as numpy array (height x width x 3, uint8).
    """
    # SMPTE color bar colors at 75% amplitude (standard)
    # Order: White, Yellow, Cyan, Green, Magenta, Red, Blue
    colors_75 = np.array([
        [191, 191, 191],   # White (75%)
        [191, 191,   0],   # Yellow
        [  0, 191, 191],   # Cyan
        [  0, 191,   0],   # Green
        [191,   0, 191],   # Magenta
        [191,   0,   0],   # Red
        [  0,   0, 191],   # Blue
    ], dtype=np.uint8)

    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Main color bars (top 2/3)
    bar_height = height * 2 // 3
    bar_width = width // 7

    for i, color in enumerate(colors_75):
        x_start = i * bar_width
        x_end = (i + 1) * bar_width if i < 6 else width
        frame[0:bar_height, x_start:x_end] = color

    # Bottom section: simplified SMPTE pattern
    # Reverse order bars at 75% for the middle strip
    strip_height = height // 12
    strip_top = bar_height

    reverse_colors = np.array([
        [  0,   0, 191],   # Blue
        [  0,   0,   0],   # Black
        [191,   0, 191],   # Magenta
        [  0,   0,   0],   # Black
        [  0, 191, 191],   # Cyan
        [  0,   0,   0],   # Black
        [191, 191, 191],   # White
    ], dtype=np.uint8)

    for i, color in enumerate(reverse_colors):
        x_start = i * bar_width
        x_end = (i + 1) * bar_width if i < 6 else width
        frame[strip_top:strip_top + strip_height, x_start:x_end] = color

    # Bottom strip: PLUGE (Picture Line-Up Generation Equipment)
    pluge_top = strip_top + strip_height
    pluge_section_width = width // 4

    # -4% black, 0% black, +4% black, 100% white reference
    pluge_colors = [
        (0, 0, 0),        # Superblack (below black)
        (16, 16, 16),     # Black
        (36, 36, 36),     # Slightly above black
        (255, 255, 255),  # White reference
    ]

    for i, color in enumerate(pluge_colors):
        x_start = i * pluge_section_width
        x_end = (i + 1) * pluge_section_width if i < 3 else width
        frame[pluge_top:height, x_start:x_end] = color

    return frame
