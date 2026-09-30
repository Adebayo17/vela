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
    # TODO: Phase 1b
    raise NotImplementedError("grayscale — à implémenter")


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
    # TODO: Phase 1b
    raise NotImplementedError("sobel — à implémenter")


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
    # TODO: Phase 1b
    raise NotImplementedError("gaussian — à implémenter")


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
    # TODO: Phase 1b
    raise NotImplementedError("median — à implémenter")


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
    # TODO: Phase 1b
    raise NotImplementedError("threshold — à implémenter")
