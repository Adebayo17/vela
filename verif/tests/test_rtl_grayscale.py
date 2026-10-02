"""
test_rtl_grayscale.py — grayscale.v vs golden model co-simulation (Phase 2a)

Chaque module Verilog est simulé avec Icarus Verilog et sa sortie est
comparée pixel pour pixel au modèle entier bit-exact (filters_hw.py).
Les modèles entiers sont eux-mêmes comparés au golden model flottant
(filters.py) pour garantir que l'approximation virgule fixe reste correcte.
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from filters import grayscale
from filters_hw import grayscale_u8


def random_rgb(h, w, seed):
    """Random uint8 RGB image (3, H, W), corners forced to extreme values."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)
    img[:, 0, 0] = 0
    img[:, 0, -1] = 255
    img[:, -1, 0] = (255, 0, 0)
    img[:, -1, -1] = (0, 255, 0)
    return img


def pack_rgb(img):
    """(3, H, W) uint8 -> flat list of 24-bit RRGGBB words, raster order."""
    rgb = img.astype(np.uint32)
    return ((rgb[0] << 16) | (rgb[1] << 8) | rgb[2]).ravel()


# ─── Grayscale ───────────────────────────────────────────────────────────────

class TestGrayscaleHw:
    """Integer model vs floating-point golden model (no simulator needed)."""

    @requirement("VELA-FR-008")
    def test_close_to_float_model(self):
        """Le modèle entier arrondi reste à < 1 LSB du modèle flottant.

        Erreur max = 0.5 (arrondi) + 0.45 (coefficients quantifiés sur 8 bits).
        """
        img = random_rgb(32, 32, seed=1)
        ref = grayscale(img.astype(np.float32))[0]
        hw = grayscale_u8(img)[0].astype(np.float32)
        assert np.max(np.abs(ref - hw)) < 1.0

    @requirement("VELA-FR-008")
    def test_white_stays_white(self):
        """77+150+29 = 256 : un pixel blanc doit donner exactement 255."""
        img = np.full((3, 1, 1), 255, dtype=np.uint8)
        assert grayscale_u8(img)[0, 0, 0] == 255


@requires_iverilog
class TestGrayscaleRtl:
    """grayscale.v simulated with Icarus Verilog vs grayscale_u8."""

    @requirement("VELA-FR-008")
    def test_matches_model_continuous_stream(self, tmp_path):
        """Flux continu (valid_in toujours à 1) : sortie bit-exact."""
        img = random_rgb(24, 24, seed=2)
        out = run_testbench("tb_grayscale", ["grayscale.v"], pack_rgb(img), 24, tmp_path)
        np.testing.assert_array_equal(out, grayscale_u8(img).ravel())

    @requirement("VELA-FR-008")
    def test_matches_model_with_valid_gaps(self, tmp_path):
        """valid_in avec trous aléatoires : aucun pixel perdu ni dupliqué."""
        img = random_rgb(24, 24, seed=3)
        out = run_testbench("tb_grayscale", ["grayscale.v"], pack_rgb(img), 24, tmp_path,
                            seed=12345)
        np.testing.assert_array_equal(out, grayscale_u8(img).ravel())

    @requirement("VELA-FR-008")
    def test_exhaustive_gray_ramp(self, tmp_path):
        """R=G=B=v pour v in 0..255 : Y doit valoir v (somme des coefs = 256)."""
        v = np.arange(256, dtype=np.uint8)
        img = np.stack([v, v, v])[:, np.newaxis, :]   # (3, 1, 256)
        out = run_testbench("tb_grayscale", ["grayscale.v"], pack_rgb(img), 24, tmp_path)
        np.testing.assert_array_equal(out, v)
