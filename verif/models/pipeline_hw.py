"""
pipeline_hw.py — Bit-exact model of what the board sends to the OLED.

Chains the fixed-point filter models (filters_hw.py) the way the RTL modules
will be chained on the board, then converts the result to the RGB565 pixels
received by oled_controller:

    original  : RGB                                  -> RGB565
    grayscale : RGB -> grayscale                     -> gray RGB565
    threshold : RGB -> grayscale -> threshold        -> gray RGB565
    gaussian  : R, G, B -> gaussian_filter (×3)      -> colour RGB565
    sobel     : RGB -> grayscale -> sobel_filter     -> gray RGB565
    median    : R, G, B -> median_filter (×3)        -> colour RGB565

The 3x3 RTL filters process one 8-bit channel: sobel and threshold work on
the luminance (grayscale stage first), gaussian and median are applied to
each channel, as in python/filters.py (VELA-TN-007 §6.2, rtl/filter_engine.v).
Input images are raw 8-bit pixels (as read from the SD card), not the
normalized images used by the CNN.

Used by python/gui_simulator.py (hardware view) and verified by
verif/tests/test_pipeline_hw.py.
"""

import numpy as np

from filters_hw import gaussian_u8, grayscale_u8, median_u8, sobel_u8, threshold_u8

# REQ: VELA-FR-009

FILTERS = ("original", "grayscale", "sobel", "gaussian", "median", "threshold")

# Mode numbers of rtl/filter_engine.v (VELA-TN-007 §6.2); 6 and 7 behave as copy
ENGINE_MODES = {0: "original", 1: "grayscale", 2: "threshold", 3: "gaussian",
                4: "sobel", 5: "median", 6: "original", 7: "original"}


def rgb888_to_rgb565(image):
    """
    RGB 8 bits per channel -> RGB565 words (5 MSBs of R, 6 of G, 5 of B).

    Args:
        image: np.ndarray (3, H, W) uint8
    Returns:
        np.ndarray (H, W) uint16
    """
    r, g, b = (image[c].astype(np.uint16) for c in range(3))
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def gray_to_rgb565(image):
    """
    8-bit gray -> RGB565 with R = G = B: {Y[7:3], Y[7:2], Y[7:3]} (VELA-TN-003 §11).

    Args:
        image: np.ndarray (1, H, W) uint8
    Returns:
        np.ndarray (H, W) uint16
    """
    return rgb888_to_rgb565(np.repeat(image, 3, axis=0))


def rgb565_to_rgb888(pixels):
    """
    RGB565 words -> 8-bit RGB for on-screen display (bit replication:
    a 5-bit value v becomes (v << 3) | (v >> 2), so 31 -> 255 and 0 -> 0).

    Args:
        pixels: np.ndarray (H, W) uint16
    Returns:
        np.ndarray (3, H, W) uint8
    """
    p = pixels.astype(np.uint16)
    r5, g6, b5 = (p >> 11) & 0x1F, (p >> 5) & 0x3F, p & 0x1F
    r = (r5 << 3) | (r5 >> 2)
    g = (g6 << 2) | (g6 >> 4)
    b = (b5 << 3) | (b5 >> 2)
    return np.stack([r, g, b]).astype(np.uint8)


def filter_frame(image, name, threshold=128, mag_shift=2):
    """
    RGB565 frame sent to the OLED for a given filter mode.

    Args:
        image:     np.ndarray (3, H, W) uint8, raw RGB pixels
        name:      one of FILTERS
        threshold: THRESHOLD parameter of threshold.v
        mag_shift: MAG_SHIFT parameter of sobel_filter.v (default 2, as in the RTL)
    Returns:
        np.ndarray (H, W) uint16, RGB565 pixels in raster order
    """
    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}, expected one of {FILTERS}")
    if name == "original":
        return rgb888_to_rgb565(image)
    if name in ("gaussian", "median"):
        stage = gaussian_u8 if name == "gaussian" else median_u8
        return rgb888_to_rgb565(np.concatenate([stage(image[c:c + 1]) for c in range(3)]))

    gray = grayscale_u8(image)
    stage = {
        "grayscale": lambda y: y,
        "threshold": lambda y: threshold_u8(y, threshold),
        "sobel":     lambda y: sobel_u8(y, mag_shift),
    }[name]
    return gray_to_rgb565(stage(gray))
