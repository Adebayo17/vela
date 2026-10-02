"""
filters_hw.py — Bit-exact integer models of the RTL image filters.

Ces fonctions reproduisent exactement l'arithmétique entière des modules
Verilog (rtl/*.v). Elles servent de référence pour les testbenches RTL :
la sortie de la simulation doit être identique, pixel pour pixel.

Le golden model flottant (filters.py) reste la référence fonctionnelle ;
les tests vérifient que ces modèles entiers restent proches de lui.

Toutes les fonctions prennent et retournent des np.ndarray uint8.
"""

import numpy as np

# REQ: VELA-FR-008

# Coefficients BT.601 en virgule fixe Q0.8 : 0.299, 0.587, 0.114 -> /256
GRAY_COEF_R = 77
GRAY_COEF_G = 150
GRAY_COEF_B = 29
GRAY_ROUND = 128


def grayscale_u8(image):
    """
    RGB -> luminance, identique à rtl/grayscale.v.

    Y = (77*R + 150*G + 29*B + 128) >> 8     (+128 : arrondi au plus proche)

    Args:
        image: np.ndarray (3, H, W) uint8
    Returns:
        np.ndarray (1, H, W) uint8
    """
    rgb = image.astype(np.uint32)
    y = (GRAY_COEF_R * rgb[0] + GRAY_COEF_G * rgb[1] + GRAY_COEF_B * rgb[2] + GRAY_ROUND) >> 8
    return y[np.newaxis, :, :].astype(np.uint8)
