"""
test_rtl_gaussian.py — gaussian_filter.v vs golden model co-simulation (Phase 2a)

Le modèle entier (arrondi au plus proche) doit rester à ≤ 0.5 LSB du golden
model flottant, et le RTL doit être identique au modèle entier, bords compris.
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from filters import gaussian
from filters_hw import gaussian_u8

SIZES = [(8, 6), (5, 3), (24, 24), (32, 32)]   # (width, height)


def random_gray(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(1, h, w), dtype=np.uint8)


def simulate(frames, tmp_path, seed=0):
    """Run tb_filter3x3 with gaussian_filter on a list of (1, H, W) frames."""
    _, h, w = frames[0].shape
    pixels = np.concatenate([f.ravel() for f in frames])
    return run_testbench("tb_filter3x3", ["gaussian_filter.v", "window3x3.v"], pixels, 8,
                         tmp_path, seed=seed, params={"IMG_WIDTH": w, "IMG_HEIGHT": h},
                         defines={"DUT": "gaussian_filter"})


# ─── Integer model vs float golden model ────────────────────────────────────

class TestGaussianHw:
    """Integer model vs floating-point golden model (no simulator needed)."""

    @requirement("VELA-FR-008")
    def test_close_to_float_model(self):
        """Arrondi au plus proche : écart ≤ 0.5 LSB, bords compris."""
        img = random_gray(32, 32, seed=1)
        ref = gaussian(img.astype(np.float32))
        hw = gaussian_u8(img).astype(np.float32)
        assert np.max(np.abs(ref - hw)) <= 0.5

    @requirement("VELA-FR-008")
    def test_white_interior_stays_white(self):
        """Somme du noyau = 16 : un intérieur blanc reste à 255 (pas de débordement)."""
        img = np.full((1, 5, 5), 255, dtype=np.uint8)
        assert np.all(gaussian_u8(img)[0, 1:-1, 1:-1] == 255)

    @requirement("VELA-FR-008")
    def test_impulse_gives_kernel(self):
        """Un seul pixel à 16 donne le noyau [1 2 1; 2 4 2; 1 2 1] autour de lui."""
        img = np.zeros((1, 5, 5), dtype=np.uint8)
        img[0, 2, 2] = 16
        expected = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(gaussian_u8(img)[0, 1:4, 1:4], expected)


# ─── RTL simulation ──────────────────────────────────────────────────────────

@requires_iverilog
class TestGaussianRtl:
    """gaussian_filter.v simulated with Icarus Verilog vs gaussian_u8."""

    @requirement("VELA-FR-008")
    @pytest.mark.parametrize("w,h", SIZES)
    def test_single_frame(self, tmp_path, w, h):
        """Une image, flux continu : sortie bit-exact, bords compris."""
        img = random_gray(h, w, seed=w * h)
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, gaussian_u8(img).ravel())

    @requirement("VELA-FR-008")
    def test_white_image(self, tmp_path):
        """Image blanche : valeurs maximales, aucun débordement dans l'additionneur."""
        img = np.full((1, 6, 8), 255, dtype=np.uint8)
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, gaussian_u8(img).ravel())

    @requirement("VELA-FR-008")
    def test_impulse(self, tmp_path):
        """Impulsion : chaque poids du noyau est vérifié individuellement."""
        img = np.zeros((1, 6, 8), dtype=np.uint8)
        img[0, 2, 3] = 160
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, gaussian_u8(img).ravel())

    @requirement("VELA-FR-008")
    def test_consecutive_frames_with_valid_gaps(self, tmp_path):
        """Images à la suite + trous aléatoires dans valid_in."""
        frames = [random_gray(24, 24, seed=s) for s in (30, 31)]
        out = simulate(frames, tmp_path, seed=777)
        expected = np.concatenate([gaussian_u8(f).ravel() for f in frames])
        np.testing.assert_array_equal(out, expected)
