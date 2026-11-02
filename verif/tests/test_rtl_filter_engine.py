"""
test_rtl_filter_engine.py — filter_engine.v co-simulation (Phase 3, VELA-TN-007 §6.2)

Le moteur lit l'image source (mots 0x00RRGGBB), applique le chemin du mode
choisi et écrit le résultat RGB565 dans la zone image de la mémoire
d'affichage. La référence est pipeline_hw.filter_frame (modèles entiers des
filtres, bit-exact) ; la zone texte, pré-remplie avec un motif témoin, doit
rester intacte (VELA-FR-013).
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from pipeline_hw import ENGINE_MODES, filter_frame

SOURCES = ["filter_engine.v", "grayscale.v", "threshold.v", "sobel_filter.v",
           "gaussian_filter.v", "median_filter.v", "window3x3.v"]
MARK = 0xA5A5_0000


def random_rgb(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)


def source_words(img):
    """(3, H, W) uint8 -> source memory words 0x00RRGGBB, raster order."""
    r, g, b = (img[c].astype(np.uint32).ravel() for c in range(3))
    return (r << 16) | (g << 8) | b


def simulate(img, mode, tmp_path, disp_width=None, **plusargs):
    """
    Run tb_filter_engine; return (display memory (H, DISP_WIDTH) uint32, cycles).
    plusargs: MODE2, GAP2, BUSY_START (see the testbench header).
    """
    _, h, w = img.shape
    disp_width = disp_width or w + 4
    tmp_path.mkdir(parents=True, exist_ok=True)
    words = run_testbench("tb_filter_engine", SOURCES, source_words(img), 32, tmp_path,
                          params={"SRC_WIDTH": w, "SRC_HEIGHT": h, "DISP_WIDTH": disp_width},
                          plusargs={"MODE": mode, **plusargs}, raw=True, timeout=300)
    mem = np.array(words[:-1], dtype=np.uint32).reshape(h, disp_width)
    return mem, words[-1]


def check_display(mem, img, mode):
    """Image zone = filter_frame (RGB565, upper half-word 0); text zone untouched."""
    _, h, w = img.shape
    expected = filter_frame(img, ENGINE_MODES[mode]).astype(np.uint32)
    np.testing.assert_array_equal(mem[:, :w], expected)
    disp_width = mem.shape[1]
    marker = MARK | np.arange(h * disp_width, dtype=np.uint32).reshape(h, disp_width)
    np.testing.assert_array_equal(mem[:, w:], marker[:, w:])


@requires_iverilog
class TestFilterEngineRtl:
    """filter_engine.v (with the real filter modules) simulated with Icarus Verilog."""

    @requirement("VELA-FR-011")
    @requirement("VELA-FR-013")
    @pytest.mark.parametrize("mode", range(8))
    def test_each_mode(self, tmp_path, mode):
        """Chaque mode : zone image bit-exact avec le modèle, zone texte intacte."""
        img = random_rgb(6, 8, seed=10 + mode)
        mem, _ = simulate(img, mode, tmp_path)
        check_display(mem, img, mode)

    @requirement("VELA-FR-011")
    @requirement("VELA-FR-013")
    @pytest.mark.parametrize("mode", [0, 3, 4, 5])
    def test_full_size_image(self, tmp_path, mode):
        """Taille réelle : image 64×64 dans une mémoire d'affichage de 96 colonnes."""
        img = np.repeat(np.repeat(random_rgb(32, 32, seed=mode), 2, axis=1), 2, axis=2)   # ×2
        mem, cycles = simulate(img, mode, tmp_path, disp_width=96)
        check_display(mem, img, mode)
        assert cycles <= 64 * 64 + 64 + 8 + 4, cycles     # TN007 §6.2: ~41 us

    @requirement("VELA-FR-011")
    @pytest.mark.parametrize("mode1,mode2", [(0, 4), (4, 0), (3, 5), (5, 3), (2, 2),
                                             (2, 4), (1, 4), (0, 3), (0, 5)])
    def test_restart_without_reset(self, tmp_path, mode1, mode2):
        """Deux images à la suite, la seconde lancée 1 cycle après done : résultat
        exact et même durée qu'une image seule. Seul le chemin choisi reçoit les
        pixels, donc aucun filtre 3x3 n'est encore en vidange de fin d'image
        (sinon le démarrage attendrait qu'il soit prêt)."""
        img = random_rgb(6, 8, seed=40)
        mem, cycles = simulate(img, mode1, tmp_path / "chain", MODE2=mode2, GAP2=1)
        check_display(mem, img, mode2)
        _, alone = simulate(img, mode2, tmp_path / "alone")
        assert cycles == alone, (cycles, alone)

    @requirement("VELA-FR-011")
    def test_start_while_busy_ignored(self, tmp_path):
        """Un front de start pendant le traitement est ignoré : le mode reste le premier."""
        img = random_rgb(6, 8, seed=50)
        mem, _ = simulate(img, 4, tmp_path, BUSY_START=0)
        check_display(mem, img, 4)

    @requirement("VELA-FR-011")
    @pytest.mark.parametrize("mode", [0, 4])
    def test_duration(self, tmp_path, mode):
        """Durée = W·H cycles + latence du chemin (≤ W + 8), pas de cycle perdu."""
        img = random_rgb(6, 8, seed=60)
        _, cycles = simulate(img, mode, tmp_path)
        assert 8 * 6 <= cycles <= 8 * 6 + 8 + 8 + 4, cycles
