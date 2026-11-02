"""
test_rtl_median.py — median_filter.v vs golden model co-simulation (Phase 2a)

La médiane de 9 entiers est un entier : le modèle entier doit être identique
au golden model flottant. Le réseau de tri (MEDIAN9_STAGES) est prouvé de
façon exhaustive, puis le RTL est comparé au modèle bit à bit.
"""

import itertools

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from filters import median
from filters_hw import median9_network, median_u8

SIZES = [(8, 6), (5, 3), (24, 24), (32, 32)]   # (width, height)


def random_gray(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(1, h, w), dtype=np.uint8)


def salt_and_pepper(h, w, seed, density=0.1):
    """Mid-gray image with isolated 0/255 impulses: what a median filter removes."""
    rng = np.random.default_rng(seed)
    img = np.full((1, h, w), 128, dtype=np.uint8)
    noise = rng.random((h, w))
    img[0][noise < density / 2] = 0
    img[0][noise > 1 - density / 2] = 255
    return img


def simulate(frames, tmp_path, seed=0):
    """Run tb_filter3x3 with median_filter on a list of (1, H, W) frames."""
    _, h, w = frames[0].shape
    pixels = np.concatenate([f.ravel() for f in frames])
    return run_testbench("tb_filter3x3", ["median_filter.v", "window3x3.v"], pixels, 8,
                         tmp_path, seed=seed, params={"IMG_WIDTH": w, "IMG_HEIGHT": h},
                         defines={"DUT": "median_filter"})


# ─── Sorting network + integer model vs float golden model ──────────────────

class TestMedianHw:
    """Sorting network proof and integer model vs golden model (no simulator)."""

    @requirement("VELA-FR-008")
    def test_network_all_permutations(self):
        """Les 9! = 362 880 ordres de 9 valeurs distinctes donnent la médiane."""
        perms = np.array(list(itertools.permutations(range(9))), dtype=np.int32)
        assert np.all(median9_network(perms) == 4)

    @requirement("VELA-FR-008")
    def test_network_all_binary_inputs(self):
        """Les 512 entrées 0/1 (principe 0-1 : couvre aussi les valeurs égales)."""
        bins = np.array(list(itertools.product((0, 1), repeat=9)), dtype=np.int32)
        np.testing.assert_array_equal(median9_network(bins), np.median(bins, axis=1))

    @requirement("VELA-FR-008")
    def test_equals_float_model(self):
        """Modèle entier identique au golden model flottant, bords compris."""
        img = random_gray(32, 32, seed=1)
        ref = median(img.astype(np.float32))
        np.testing.assert_array_equal(median_u8(img), ref.astype(np.uint8))

    @requirement("VELA-FR-008")
    def test_removes_isolated_impulses(self):
        """Bruit poivre et sel peu dense : l'intérieur redevient uniforme."""
        img = salt_and_pepper(16, 16, seed=2, density=0.05)
        assert np.all(median_u8(img)[0, 2:-2, 2:-2] == 128)


# ─── RTL simulation ──────────────────────────────────────────────────────────

@requires_iverilog
class TestMedianRtl:
    """median_filter.v simulated with Icarus Verilog vs median_u8."""

    @requirement("VELA-FR-008")
    @pytest.mark.parametrize("w,h", SIZES)
    def test_single_frame(self, tmp_path, w, h):
        """Image aléatoire : sortie bit-exact, bords compris."""
        img = random_gray(h, w, seed=w * h)
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, median_u8(img).ravel())

    @requirement("VELA-FR-008")
    def test_salt_and_pepper(self, tmp_path):
        """Bruit poivre et sel : nombreuses valeurs égales dans les fenêtres."""
        img = salt_and_pepper(24, 24, seed=3, density=0.3)
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, median_u8(img).ravel())

    @requirement("VELA-FR-008")
    def test_few_levels(self, tmp_path):
        """Seulement 3 niveaux de gris : beaucoup d'égalités, teste les comparaisons."""
        rng = np.random.default_rng(4)
        img = rng.choice(np.array([10, 20, 30], dtype=np.uint8), size=(1, 24, 24))
        out = simulate([img], tmp_path)
        np.testing.assert_array_equal(out, median_u8(img).ravel())

    @requirement("VELA-FR-008")
    def test_consecutive_frames_with_valid_gaps(self, tmp_path):
        """Images à la suite + trous aléatoires dans valid_in (pipeline de 3 étages)."""
        frames = [random_gray(24, 24, seed=50), salt_and_pepper(24, 24, seed=51)]
        out = simulate(frames, tmp_path, seed=31337)
        expected = np.concatenate([median_u8(f).ravel() for f in frames])
        np.testing.assert_array_equal(out, expected)
