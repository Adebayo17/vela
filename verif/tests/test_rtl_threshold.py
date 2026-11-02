"""
test_rtl_threshold.py — threshold.v vs golden model co-simulation (Phase 2a)

Le seuillage n'a que 256 entrées possibles : chaque test les couvre toutes,
pour plusieurs valeurs du paramètre THRESHOLD (bornes 0/255 et voisinage
du seuil par défaut 128).
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from filters import threshold
from filters_hw import threshold_u8

ALL_PIXELS = np.arange(256, dtype=np.uint8)[np.newaxis, np.newaxis, :]   # (1, 1, 256)
THRESHOLDS = [0, 1, 127, 128, 254, 255]


# ─── Integer model vs float golden model ────────────────────────────────────

class TestThresholdHw:
    """Integer model vs floating-point golden model (no simulator needed)."""

    @requirement("VELA-FR-008")
    @pytest.mark.parametrize("thresh", THRESHOLDS)
    def test_matches_float_model(self, thresh):
        """threshold_u8(p, T) == threshold(p/255, T/255) * 255 pour tout p."""
        ref = threshold(ALL_PIXELS.astype(np.float32) / 255.0, thresh=thresh / 255.0) * 255
        hw = threshold_u8(ALL_PIXELS, thresh)
        np.testing.assert_array_equal(hw, ref.astype(np.uint8))

    @requirement("VELA-FR-008")
    def test_strict_comparison(self):
        """Comparaison stricte : pixel == seuil donne 0."""
        assert threshold_u8(np.array([[[128]]], dtype=np.uint8), 128)[0, 0, 0] == 0
        assert threshold_u8(np.array([[[129]]], dtype=np.uint8), 128)[0, 0, 0] == 255


# ─── RTL simulation ──────────────────────────────────────────────────────────

@requires_iverilog
class TestThresholdRtl:
    """threshold.v simulated with Icarus Verilog vs threshold_u8."""

    @requirement("VELA-FR-008")
    @pytest.mark.parametrize("thresh", THRESHOLDS)
    def test_exhaustive_all_pixels(self, tmp_path, thresh):
        """Les 256 valeurs de pixel, pour chaque seuil : sortie bit-exact."""
        out = run_testbench("tb_threshold", ["threshold.v"], ALL_PIXELS.ravel(), 8, tmp_path,
                            params={"THRESHOLD": thresh})
        np.testing.assert_array_equal(out, threshold_u8(ALL_PIXELS, thresh).ravel())

    @requirement("VELA-FR-008")
    def test_default_threshold_is_128(self, tmp_path):
        """Sans surcharge, le module utilise THRESHOLD = 128."""
        out = run_testbench("tb_threshold", ["threshold.v"], ALL_PIXELS.ravel(), 8, tmp_path)
        np.testing.assert_array_equal(out, threshold_u8(ALL_PIXELS, 128).ravel())

    @requirement("VELA-FR-008")
    def test_matches_model_with_valid_gaps(self, tmp_path):
        """valid_in avec trous aléatoires : aucun pixel perdu ni dupliqué."""
        rng = np.random.default_rng(4)
        img = rng.integers(0, 256, size=(1, 24, 24), dtype=np.uint8)
        out = run_testbench("tb_threshold", ["threshold.v"], img.ravel(), 8, tmp_path,
                            seed=6789, params={"THRESHOLD": 100})
        np.testing.assert_array_equal(out, threshold_u8(img, 100).ravel())
