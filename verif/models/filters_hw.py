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


def sobel_u8(image, mag_shift=0):
    """
    Détection de contours Sobel 3x3, identique à rtl/sobel_filter.v.

    Gx = [-1 0 1; -2 0 2; -1 0 1], Gy = [-1 -2 -1; 0 0 0; 1 2 1]
    Y  = min((|Gx| + |Gy|) >> mag_shift, 255)

    |Gx| + |Gy| peut atteindre 1530 : la sortie 8 bits est saturée.
    Avec mag_shift = 0, Y == clip(filters.sobel(image), 0, 255) exactement.

    Args:
        image: np.ndarray (1, H, W) uint8
        mag_shift: int, paramètre MAG_SHIFT du module (division par 2**mag_shift)
    Returns:
        np.ndarray (1, H, W) uint8
    """
    _, h, w = image.shape
    win = windows_3x3_u8(image).astype(np.int32)
    gx = (win[:, 2] + 2 * win[:, 5] + win[:, 8]) - (win[:, 0] + 2 * win[:, 3] + win[:, 6])
    gy = (win[:, 6] + 2 * win[:, 7] + win[:, 8]) - (win[:, 0] + 2 * win[:, 1] + win[:, 2])
    mag = (np.abs(gx) + np.abs(gy)) >> mag_shift
    return np.minimum(mag, 255).astype(np.uint8).reshape(1, h, w)


# Réseau de sélection de la médiane de 9 valeurs (« opt_med9 », 19 compare-échanges).
# Chaque paire (a, b) : p[a], p[b] = min, max. La médiane finit en p[4].
# Regroupé par étage de pipeline de rtl/median_filter.v (3 niveaux par étage).
MEDIAN9_STAGES = [
    [(1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7), (1, 2), (4, 5), (7, 8)],   # étage 1
    [(0, 3), (5, 8), (4, 7), (3, 6), (1, 4), (2, 5), (4, 7)],                   # étage 2
    [(4, 2), (6, 4), (4, 2)],                                                   # étage 3
]


def median9_network(values):
    """
    Applique MEDIAN9_STAGES, comme le RTL.

    Args:
        values: np.ndarray (N, 9), une fenêtre par ligne
    Returns:
        np.ndarray (N,), la médiane de chaque fenêtre
    """
    p = np.array(values, copy=True)
    for stage in MEDIAN9_STAGES:
        for a, b in stage:
            lo = np.minimum(p[:, a], p[:, b])
            hi = np.maximum(p[:, a], p[:, b])
            p[:, a], p[:, b] = lo, hi
    return p[:, 4]


def median_u8(image):
    """
    Filtre médian 3x3, identique à rtl/median_filter.v.

    Médiane des 9 pixels de la fenêtre, bords à 0 (comme filters.median).

    Args:
        image: np.ndarray (1, H, W) uint8
    Returns:
        np.ndarray (1, H, W) uint8
    """
    _, h, w = image.shape
    return median9_network(windows_3x3_u8(image)).astype(np.uint8).reshape(1, h, w)
