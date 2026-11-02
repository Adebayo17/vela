"""
test_rtl_sobel.py — sobel_filter.v vs golden model co-simulation (Phase 2a)

|Gx| + |Gy| peut atteindre 1530 alors que la sortie fait 8 bits : le module
divise la magnitude par 2^MAG_SHIFT (2 par défaut) puis sature à 255.
Avec MAG_SHIFT = 0, le modèle entier doit être exactement
clip(golden model, 0, 255) ; le RTL doit être identique au modèle entier.
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from filters import sobel
from filters_hw import sobel_u8

SIZES = [(8, 6), (5, 3), (24, 24), (32, 32)]   # (width, height)


def random_gray(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(1, h, w), dtype=np.uint8)


def low_contrast(h, w, seed):
    """Random image in [100, 140): gradients mostly < 255, so no saturation."""
    rng = np.random.default_rng(seed)
    return rng.integers(100, 140, size=(1, h, w), dtype=np.uint8)


def max_gradient_image():
    """5x5 image whose center window gives the maximum |Gx| + |Gy| = 1530."""
    img = np.zeros((1, 5, 5), dtype=np.uint8)
    img[0, 1:4, 1:4] = [[0, 0, 255], [0, 0, 255], [255, 255, 255]]
    return img


def simulate(frames, tmp_path, seed=0, mag_shift=None):
    """Run tb_filter3x3 with sobel_filter on a list of (1, H, W) frames."""
    _, h, w = frames[0].shape
    pixels = np.concatenate([f.ravel() for f in frames])
    defines = {"DUT": "sobel_filter"}
    if mag_shift is not None:
        defines["DUT_EXTRA_PARAMS"] = f",.MAG_SHIFT({mag_shift})"
    return run_testbench("tb_filter3x3", ["sobel_filter.v", "window3x3.v"], pixels, 8,
                         tmp_path, seed=seed, params={"IMG_WIDTH": w, "IMG_HEIGHT": h},
                         defines=defines)


# ─── Integer model vs float golden model ────────────────────────────────────

class TestSobelHw:
    """Integer model vs floating-point golden model (no simulator needed)."""

    @requirement("VELA-FR-008")
    def test_equals_clipped_float_model(self):
        """MAG_SHIFT = 0 : identique au golden model saturé à 255."""
        img = random_gray(32, 32, seed=1)
        ref = np.clip(sobel(img.astype(np.float32)), 0, 255)
        np.testing.assert_array_equal(sobel_u8(img, mag_shift=0), ref.astype(np.uint8))

    @requirement("VELA-FR-008")
    def test_low_contrast_not_saturated(self):
        """Faible contraste à l'intérieur : aucune saturation, égalité stricte."""
        img = low_contrast(16, 16, seed=2)
        ref = sobel(img.astype(np.float32))[0, 1:-1, 1:-1]
        assert ref.max() < 255
        np.testing.assert_array_equal(sobel_u8(img, mag_shift=0)[0, 1:-1, 1:-1],
                                      ref.astype(np.uint8))

    @requirement("VELA-FR-008")
    def test_max_gradient_saturates(self):
        """Magnitude maximale 1530 : 255 (saturé) avec 0 et 2, 1530 >> 3 = 191 avec 3."""
        img = max_gradient_image()
        assert sobel(img.astype(np.float32))[0, 2, 2] == 1530
        assert sobel_u8(img, mag_shift=0)[0, 2, 2] == 255
        assert sobel_u8(img, mag_shift=2)[0, 2, 2] == 255     # 1530 >> 2 = 382
        assert sobel_u8(img, mag_shift=3)[0, 2, 2] == 191

    @requirement("VELA-FR-008")
    def test_flat_image_has_no_edges(self):
        """Image uniforme : magnitude nulle à l'intérieur."""
        img = np.full((1, 6, 6), 77, dtype=np.uint8)
        assert np.all(sobel_u8(img)[0, 1:-1, 1:-1] == 0)


# ─── RTL simulation ──────────────────────────────────────────────────────────

@requires_iverilog
class TestSobelRtl:
    """sobel_filter.v simulated with Icarus Verilog vs sobel_u8."""

    @requirement("VELA-FR-008")
    @pytest.mark.parametrize("w,h", SIZES)
    def test_single_frame(self, tmp_path, w, h):
        """Image aléatoire (fort contraste, beaucoup de saturation) : bit-exact."""
        img = random_gray(h, w, seed=w * h)
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, sobel_u8(img).ravel())

    @requirement("VELA-FR-008")
    def test_low_contrast(self, tmp_path):
        """Faible contraste : gradients non saturés, valeurs exactes."""
        img = low_contrast(24, 24, seed=3)
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, sobel_u8(img).ravel())

    @requirement("VELA-FR-008")
    @pytest.mark.parametrize("flip", ["none", "lr", "ud", "both"])
    def test_gradient_signs(self, tmp_path, flip):
        """Fenêtre max retournée : Gx et Gy positifs ou négatifs (valeur absolue)."""
        img = max_gradient_image()[0]
        img = {"none": img, "lr": img[:, ::-1], "ud": img[::-1, :],
               "both": img[::-1, ::-1]}[flip]
        img = np.ascontiguousarray(img)[np.newaxis]
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, sobel_u8(img).ravel())

    @requirement("VELA-FR-008")
    @pytest.mark.parametrize("mag_shift", [0, 1, 3])
    def test_mag_shift(self, tmp_path, mag_shift):
        """Paramètre MAG_SHIFT : magnitude divisée par 2^N avant saturation."""
        img = random_gray(8, 8, seed=4)
        img[:, 2:7, 2:7] = max_gradient_image()
        out = simulate([img], tmp_path, mag_shift=mag_shift)
        np.testing.assert_array_equal(out, sobel_u8(img, mag_shift).ravel())

    @requirement("VELA-FR-008")
    def test_default_mag_shift_is_2(self, tmp_path):
        """Sans surcharge, le module divise par 4 (MAG_SHIFT = 2) : diffère de 0 et 3."""
        img = random_gray(8, 8, seed=7)
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, sobel_u8(img, mag_shift=2).ravel())
        assert not np.array_equal(out, sobel_u8(img, mag_shift=0).ravel())
        assert not np.array_equal(out, sobel_u8(img, mag_shift=3).ravel())

    @requirement("VELA-FR-008")
    def test_consecutive_frames_with_valid_gaps(self, tmp_path):
        """Images à la suite + trous aléatoires dans valid_in."""
        frames = [low_contrast(24, 24, seed=40), random_gray(24, 24, seed=41)]
        out = simulate(frames, tmp_path, seed=999)
        expected = np.concatenate([sobel_u8(f).ravel() for f in frames])
        np.testing.assert_array_equal(out, expected)
