"""
filters.py — Image filters: golden model Python

Filtres implémentés :
    - grayscale   : RGB → luminance (0.299R + 0.587G + 0.114B)
    - sobel       : détection de contours (gradient 3x3)
    - gaussian    : lissage gaussien (noyau 3x3)
    - median      : filtre médian (fenêtre 3x3)
    - threshold   : seuillage binaire

Toutes les fonctions prennent une image CHW (C, H, W) en float32
et retournent le résultat dans le même format.
"""

import numpy as np


def grayscale(image):
    """
    Convertit une image RGB en niveaux de gris.
    
    Args:
        image: np.ndarray (3, H, W) float32, valeurs normalisées
    Returns:
        np.ndarray (1, H, W) float32, luminance
    """
    r, g, b = image[0], image[1], image[2]
    y = (0.299 * r) + (0.587 * g) + (0.114 * b)

    return y[np.newaxis, :, :].astype(np.float32)   # (1, H, W) for CHW


def sobel(image):
    """
    Détection de contours par filtre de Sobel (3x3).
    
    Appliqué sur l'image en niveaux de gris.
    Gx = [-1 0 1; -2 0 2; -1 0 1]
    Gy = [-1 -2 -1; 0 0 0; 1 2 1]
    Magnitude = |Gx| + |Gy|
    
    Args:
        image: np.ndarray (1, H, W) float32, grayscale
    Returns:
        np.ndarray (1, H, W) float32, edge magnitude
    """
    if image.shape[0] != 1:
        image = grayscale(image)

    gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)

    gy_kernel = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=np.float32)

    _, h, w = image.shape
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)),
                    mode="constant", constant_values=0)
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            window = padded[0, i:i+3, j:j+3]
            gx = (window * gx_kernel).sum()
            gy = (window * gy_kernel).sum()
            output[0, i, j] = abs(gx) + abs(gy)

    return output.astype(np.float32)


def gaussian(image, kernel=None):
    """
    Lissage gaussien (noyau 3x3).
    
    Noyau par défaut : [1,2,1; 2,4,2; 1,2,1] / 16
    
    Args:
        image: np.ndarray (C, H, W) float32
        kernel: np.ndarray (3, 3) optionnel
    Returns:
        np.ndarray (C, H, W) float32
    """
    if kernel is None:
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 16.0

    c, h, w = image.shape
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)),
                    mode="constant", constant_values=0)

    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            window = padded[:, i:i+3, j:j+3]    # (C, 3, 3)
            output[:, i, j] = (window * kernel).sum(axis=(1, 2))
    
    return output.astype(np.float32)



def median(image, kernel_size=3):
    """
    Filtre médian (fenêtre 3x3).
    
    Pour chaque pixel, trie les 9 voisins et prend la valeur médiane.
    
    Args:
        image: np.ndarray (C, H, W) float32
        kernel_size: int, taille de la fenêtre (3)
    Returns:
        np.ndarray (C, H, W) float32
    """
    c, h, w = image.shape
    pad = kernel_size // 2
    padded = np.pad(image, ((0, 0), (pad, pad), (pad, pad)),
                    mode="constant", constant_values=0)
    output = np.zeros_like(image)

    for ch in range(c):
        for i in range(h):
            for j in range(w):
                window = padded[ch, i:i+kernel_size, j:j+kernel_size]
                output[ch, i, j] = np.median(window)

    return output.astype(np.float32)


def threshold(image, thresh=0.5):
    """
    Seuillage binaire.
    
    Output = 1.0 si pixel > thresh, sinon 0.0
    
    Args:
        image: np.ndarray (1, H, W) float32, grayscale
        thresh: float, seuil [0, 1]
    Returns:
        np.ndarray (1, H, W) float32, image binaire
    """
    result = np.where(image > thresh, 1.0, 0.0)
    return result.astype(np.float32)
