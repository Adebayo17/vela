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


def threshold_u8(image, thresh=128):
    """
    Seuillage binaire, identique à rtl/threshold.v.

    Y = 255 si pixel > thresh, sinon 0

    Équivaut à filters.threshold(image / 255, thresh / 255) * 255.

    Args:
        image: np.ndarray (1, H, W) uint8
        thresh: int [0, 255], paramètre THRESHOLD du module
    Returns:
        np.ndarray (1, H, W) uint8, valeurs 0 ou 255
    """
    return np.where(image > thresh, 255, 0).astype(np.uint8)


def windows_3x3_u8(image):
    """
    Fenêtres 3x3 produites par rtl/window3x3.v, une par pixel (ordre raster).

    Bords : voisins hors image = 0, comme le padding de filters.py.

    Args:
        image: np.ndarray (1, H, W) uint8
    Returns:
        np.ndarray (H*W, 9) uint8, fenêtre en ordre ligne par ligne
        [haut-gauche, haut, haut-droite, gauche, centre, ..., bas-droite]
    """
    _, h, w = image.shape
    padded = np.pad(image[0], 1, mode="constant", constant_values=0)
    return np.array([padded[i:i+3, j:j+3].ravel()
                     for i in range(h) for j in range(w)], dtype=np.uint8)


GAUSS_KERNEL = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1], dtype=np.uint32)   # somme = 16
GAUSS_ROUND = 8


def gaussian_u8(image):
    """
    Flou gaussien 3x3, identique à rtl/gaussian_filter.v.

    Y = (somme(noyau * fenêtre) + 8) >> 4     (+8 : arrondi au plus proche)
    Noyau [1 2 1; 2 4 2; 1 2 1], bords à 0 (comme filters.gaussian).

    Args:
        image: np.ndarray (1, H, W) uint8
    Returns:
        np.ndarray (1, H, W) uint8
    """
    _, h, w = image.shape
    acc = windows_3x3_u8(image).astype(np.uint32) @ GAUSS_KERNEL
    return ((acc + GAUSS_ROUND) >> 4).astype(np.uint8).reshape(1, h, w)
