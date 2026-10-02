"""
test_rtl_window3x3.py — window3x3.v vs reference windows (Phase 2a)

Le générateur de fenêtres est le cœur commun de sobel, gaussian et median.
On vérifie, pour chaque pixel, les 9 valeurs de la fenêtre (bords à 0),
sur plusieurs tailles d'image, avec trous dans valid_in et plusieurs
images consécutives (flush de fin d'image + ready_in).
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from filters_hw import windows_3x3_u8

SIZES = [(8, 6), (5, 3), (24, 24), (32, 32)]   # (width, height)


def random_gray(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(1, h, w), dtype=np.uint8)


def simulate(frames, tmp_path, seed=0):
    """Run tb_window3x3 on a list of (1, H, W) frames, return (N, 9) windows."""
    _, h, w = frames[0].shape
    pixels = np.concatenate([f.ravel() for f in frames])
    return run_testbench("tb_window3x3", ["window3x3.v"], pixels, 8, tmp_path, seed=seed,
                         params={"IMG_WIDTH": w, "IMG_HEIGHT": h}, out_width=72)


# ─── Reference model ────────────────────────────────────────────────────────

class TestWindowModel:
    """windows_3x3_u8 vs the padding used by the float golden model."""

    @requirement("VELA-FR-008")
    def test_center_is_pixel(self):
        """La valeur centrale de chaque fenêtre est le pixel lui-même."""
        img = random_gray(5, 7, seed=1)
        np.testing.assert_array_equal(windows_3x3_u8(img)[:, 4], img.ravel())

    @requirement("VELA-FR-008")
    def test_corner_zero_padding(self):
        """Coin haut-gauche : ligne du haut et colonne de gauche à 0."""
        img = np.full((1, 4, 4), 9, dtype=np.uint8)
        np.testing.assert_array_equal(windows_3x3_u8(img)[0], [0, 0, 0, 0, 9, 9, 0, 9, 9])


# ─── RTL simulation ──────────────────────────────────────────────────────────

@requires_iverilog
class TestWindowRtl:
    """window3x3.v simulated with Icarus Verilog vs windows_3x3_u8."""

    @requirement("VELA-FR-008")
    @pytest.mark.parametrize("w,h", SIZES)
    def test_single_frame(self, tmp_path, w, h):
        """Une image, flux continu : 9 valeurs exactes pour chaque pixel."""
        img = random_gray(h, w, seed=w * h)
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, windows_3x3_u8(img))

    @requirement("VELA-FR-008")
    def test_position_ramp(self, tmp_path):
        """Pixel = 1 + indice : une erreur de décalage se voit sur la valeur."""
        h, w = 6, 8
        img = (np.arange(h * w) + 1).astype(np.uint8).reshape(1, h, w)
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, windows_3x3_u8(img))

    @requirement("VELA-FR-008")
    def test_consecutive_frames(self, tmp_path):
        """3 images à la suite : aucune donnée ne fuit d'une image à l'autre."""
        frames = [random_gray(6, 8, seed=s) for s in (10, 11, 12)]
        out = simulate(frames, tmp_path)
        expected = np.concatenate([windows_3x3_u8(f) for f in frames])
        np.testing.assert_array_equal(out, expected)

    @requirement("VELA-FR-008")
    def test_consecutive_frames_with_valid_gaps(self, tmp_path):
        """Images à la suite + trous aléatoires dans valid_in."""
        frames = [random_gray(24, 24, seed=s) for s in (20, 21)]
        out = simulate(frames, tmp_path, seed=4242)
        expected = np.concatenate([windows_3x3_u8(f) for f in frames])
        np.testing.assert_array_equal(out, expected)
