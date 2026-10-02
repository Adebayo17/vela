"""
test_filters.py — Unit tests for image filters (Phase 1b)

Tests use synthetic images with known expected outputs
to validate each filter independently of CIFAR-10 data.
"""

import numpy as np
import pytest
from vela_testing import requirement
from filters import *


# ─── Grayscale ───────────────────────────────────────────────────────────────

class TestGrayscale:
    """Tests for grayscale conversion."""

    @requirement("VELA-FR-007")
    def test_output_shape_single_channel(self):
        """Grayscale doit produire 1 canal en sortie."""
        img = np.ones((3, 8, 8), dtype=np.float32)
        result = grayscale(img)
        assert result.shape == (1, 8, 8)

    @requirement("VELA-FR-007")
    def test_luminance_weights(self):
        """Vérifie les coefficients ITU-R BT.601 : 0.299R + 0.587G + 0.114B."""
        # Image où R=100, G=0, B=0 partout
        img = np.zeros((3, 4, 4), dtype=np.float32)
        img[0] = 100.0  # R only
        result = grayscale(img)
        expected = 0.299 * 100.0
        assert np.allclose(result[0], expected, atol=1e-4)

    @requirement("VELA-FR-007")
    def test_uniform_rgb_gives_same_value(self):
        """Si R=G=B=v, le grayscale doit valoir v (car 0.299+0.587+0.114=1)."""
        v = 42.0
        img = np.full((3, 5, 5), v, dtype=np.float32)
        result = grayscale(img)
        assert np.allclose(result[0], v, atol=1e-4)

    @requirement("VELA-FR-007")
    def test_output_dtype_float32(self):
        """Le résultat doit être en float32."""
        img = np.ones((3, 4, 4), dtype=np.float32)
        result = grayscale(img)
        assert result.dtype == np.float32


# ─── Sobel ───────────────────────────────────────────────────────────────────

class TestSobel:
    """Tests for Sobel edge detection."""

    @requirement("VELA-FR-007")
    def test_output_shape(self):
        """Sobel conserve les dimensions spatiales, produit 1 canal."""
        img = np.ones((3, 8, 8), dtype=np.float32)
        result = sobel(img)
        assert result.shape[0] == 1          # mono-canal
        assert result.shape[1:] == (8, 8)    # mêmes dimensions

    @requirement("VELA-FR-007")
    def test_uniform_image_zero_edges(self):
        """Sur une image uniforme, aucun gradient → sortie ≈ 0 partout."""
        img = np.full((1, 8, 8), 50.0, dtype=np.float32)
        result = sobel(img)
        assert np.allclose(result[:, 2:-2, 2:-2], 0.0, atol=1e-4)

    @requirement("VELA-FR-007")
    def test_vertical_edge_detected(self):
        """Un bord vertical net doit produire une réponse forte au centre."""
        img = np.zeros((1, 8, 8), dtype=np.float32)
        img[0, :, 4:] = 100.0  # moitié droite à 100
        result = sobel(img)
        # La colonne du bord (col 4) doit avoir les valeurs les plus fortes
        edge_col = result[0, 1:-1, 4]  # ignorer les bords paddés
        non_edge  = result[0, 1:-1, 0]
        assert edge_col.mean() > non_edge.mean()

    @requirement("VELA-FR-007")
    def test_output_non_negative(self):
        """Sobel utilise |Gx|+|Gy|, donc la sortie est toujours ≥ 0."""
        rng = np.random.default_rng(42)
        img = rng.random((3, 10, 10)).astype(np.float32)
        result = sobel(img)
        assert (result >= -1e-6).all()


# ─── Gaussian ────────────────────────────────────────────────────────────────

class TestGaussian:
    """Tests for Gaussian blur."""

    @requirement("VELA-FR-007")
    def test_output_shape_preserved(self):
        """Gaussian conserve exactement la forme d'entrée."""
        img = np.ones((3, 8, 8), dtype=np.float32)
        result = gaussian(img)
        assert result.shape == img.shape

    @requirement("VELA-FR-007")
    def test_uniform_image_unchanged(self):
        """Un flou sur une image uniforme ne change rien."""
        v = 75.0
        img = np.full((3, 8, 8), v, dtype=np.float32)
        result = gaussian(img)
        # Les pixels intérieurs doivent rester à v
        # (les bords sont affectés par le zero-padding)
        assert np.allclose(result[:, 1:-1, 1:-1], v, atol=1e-4)

    @requirement("VELA-FR-007")
    def test_impulse_response(self):
        """Un pixel isolé doit être étalé selon le kernel gaussien."""
        img = np.zeros((1, 7, 7), dtype=np.float32)
        img[0, 3, 3] = 16.0  # impulsion au centre
        result = gaussian(img)
        # Le centre doit recevoir 4/16 * 16 = 4.0 (poids central du kernel)
        assert abs(result[0, 3, 3] - 4.0) < 1e-4
        # Les voisins directs reçoivent 2/16 * 16 = 2.0
        assert abs(result[0, 3, 2] - 2.0) < 1e-4
        assert abs(result[0, 2, 3] - 2.0) < 1e-4

    @requirement("VELA-FR-007")
    def test_reduces_noise(self):
        """Le blur doit réduire la variance (aplatir le bruit)."""
        rng = np.random.default_rng(42)
        img = rng.random((1, 20, 20)).astype(np.float32) * 100
        result = gaussian(img)
        # Variance intérieure doit diminuer après flou
        assert result[:, 2:-2, 2:-2].var() < img[:, 2:-2, 2:-2].var()


# ─── Median ──────────────────────────────────────────────────────────────────

class TestMedian:
    """Tests for Median filter."""

    @requirement("VELA-FR-007")
    def test_output_shape_preserved(self):
        """Median conserve la forme d'entrée."""
        img = np.ones((3, 8, 8), dtype=np.float32)
        result = median(img)
        assert result.shape == img.shape

    @requirement("VELA-FR-007")
    def test_uniform_image_unchanged(self):
        """Image uniforme → médiane = valeur uniforme."""
        v = 50.0
        img = np.full((1, 6, 6), v, dtype=np.float32)
        result = median(img)
        assert np.allclose(result[:, 1:-1, 1:-1], v, atol=1e-4)

    @requirement("VELA-FR-007")
    def test_removes_salt_and_pepper(self):
        """Le filtre médian doit éliminer le bruit sel-et-poivre."""
        img = np.full((1, 9, 9), 100.0, dtype=np.float32)
        # Ajouter quelques pixels aberrants isolés
        img[0, 4, 4] = 0.0    # "pepper"
        img[0, 2, 6] = 255.0  # "salt"
        result = median(img)
        # Les pixels aberrants doivent être remplacés par la médiane locale (100)
        assert abs(result[0, 4, 4] - 100.0) < 1e-4
        assert abs(result[0, 2, 6] - 100.0) < 1e-4

    @requirement("VELA-FR-007")
    def test_preserves_edges(self):
        """Contrairement au gaussien, le médian préserve les bords nets."""
        img = np.zeros((1, 10, 10), dtype=np.float32)
        img[0, :, 5:] = 100.0  # bord vertical net
        result = median(img)
        # Loin du bord, les valeurs doivent rester 0 et 100
        assert abs(result[0, 5, 2] - 0.0) < 1e-4
        assert abs(result[0, 5, 7] - 100.0) < 1e-4


# ─── Threshold ───────────────────────────────────────────────────────────────

class TestThreshold:
    """Tests for binary threshold."""

    @requirement("VELA-FR-007")
    def test_output_shape_preserved(self):
        """Threshold conserve la forme d'entrée."""
        img = np.ones((3, 8, 8), dtype=np.float32) * 0.5
        result = threshold(img, thresh=0.5)
        assert result.shape == img.shape

    @requirement("VELA-FR-007")
    def test_binary_output(self):
        """La sortie ne contient que des 0.0 et des 1.0."""
        rng = np.random.default_rng(42)
        img = rng.random((3, 10, 10)).astype(np.float32)
        result = threshold(img, thresh=0.5)
        unique_vals = set(np.unique(result))
        assert unique_vals <= {0.0, 1.0}

    @requirement("VELA-FR-007")
    def test_known_threshold(self):
        """Vérifie le seuil exact : > thresh → 1, sinon 0."""
        img = np.array([[[0.3, 0.5, 0.7],
                         [0.1, 0.9, 0.5],
                         [0.6, 0.4, 0.8]]], dtype=np.float32)
        result = threshold(img, thresh=0.5)
        expected = np.array([[[0.0, 0.0, 1.0],
                              [0.0, 1.0, 0.0],
                              [1.0, 0.0, 1.0]]], dtype=np.float32)
        assert np.array_equal(result, expected)

    @requirement("VELA-FR-007")
    def test_all_above_threshold(self):
        """Si tous les pixels > thresh, tout vaut 1."""
        img = np.full((1, 5, 5), 0.8, dtype=np.float32)
        result = threshold(img, thresh=0.3)
        assert np.all(result == 1.0)

    @requirement("VELA-FR-007")
    def test_all_below_threshold(self):
        """Si tous les pixels ≤ thresh, tout vaut 0."""
        img = np.full((1, 5, 5), 0.2, dtype=np.float32)
        result = threshold(img, thresh=0.5)
        assert np.all(result == 0.0)
