"""
test_pipeline_hw.py — Board output model (pipeline_hw.py) and GUI hardware view

Vérifie la chaîne affichée par le GUI en vue HW : conversions RGB565,
enchaînement grayscale -> filtre RTL bit-exact, et composition de l'image de
l'écran sans renormalisation. Pas de simulateur.
"""

import numpy as np
import pytest
from vela_testing import requirement
from filters_hw import gaussian_u8, grayscale_u8, median_u8, sobel_u8, threshold_u8
from pipeline_hw import (FILTERS, filter_frame, gray_to_rgb565, rgb565_to_rgb888,
                         rgb888_to_rgb565)


def rgb(r, g, b):
    """Single RGB pixel as a (3, 1, 1) uint8 image."""
    return np.array([r, g, b], dtype=np.uint8).reshape(3, 1, 1)


def random_rgb(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)


# ─── RGB565 conversions ──────────────────────────────────────────────────────

class TestRgb565:

    @requirement("VELA-FR-009")
    @pytest.mark.parametrize("pixel,word", [
        ((255, 255, 255), 0xFFFF), ((0, 0, 0), 0x0000),
        ((255, 0, 0), 0xF800), ((0, 255, 0), 0x07E0), ((0, 0, 255), 0x001F),
        ((0x58, 0x9F, 0x85), 0x5CF0),   # R5 = 11, G6 = 39, B5 = 16
    ])
    def test_rgb888_to_rgb565_reference_values(self, pixel, word):
        """Valeurs de référence : 5 bits de R, 6 de G, 5 de B (poids forts)."""
        assert rgb888_to_rgb565(rgb(*pixel))[0, 0] == word

    @requirement("VELA-FR-009")
    def test_gray_to_rgb565_bit_layout(self):
        """Les 256 niveaux : {Y[7:3], Y[7:2], Y[7:3]} (TN003 §11)."""
        y = np.arange(256, dtype=np.uint8)
        expected = ((y >> 3).astype(np.uint16) << 11) | ((y >> 2).astype(np.uint16) << 5) | (y >> 3)
        np.testing.assert_array_equal(gray_to_rgb565(y.reshape(1, 1, 256))[0], expected)

    @requirement("VELA-FR-009")
    def test_rgb565_roundtrip_all_codes(self):
        """Les 65 536 codes : RGB565 -> RGB888 -> RGB565 redonne le code."""
        codes = np.arange(1 << 16, dtype=np.uint16).reshape(256, 256)
        np.testing.assert_array_equal(rgb888_to_rgb565(rgb565_to_rgb888(codes)), codes)

    @requirement("VELA-FR-009")
    def test_rgb565_to_rgb888_full_scale(self):
        """Réplication des bits : 0 -> 0 et la valeur max -> 255 sur chaque canal."""
        out = rgb565_to_rgb888(np.array([[0x0000, 0xFFFF]], dtype=np.uint16))
        np.testing.assert_array_equal(out[:, 0, 0], [0, 0, 0])
        np.testing.assert_array_equal(out[:, 0, 1], [255, 255, 255])


# ─── Filter chain ────────────────────────────────────────────────────────────

class TestFilterFrame:

    @requirement("VELA-FR-009")
    @pytest.mark.parametrize("name,stage", [
        ("grayscale", lambda y: y),
        ("threshold", lambda y: threshold_u8(y, 128)),
        ("gaussian", gaussian_u8),
        ("sobel", sobel_u8),
        ("median", median_u8),
    ])
    def test_gray_modes_chain(self, name, stage):
        """Mode filtre = grayscale_u8 -> filtre RTL bit-exact -> RGB565 gris."""
        img = random_rgb(24, 24, seed=1)
        expected = gray_to_rgb565(stage(grayscale_u8(img)))
        np.testing.assert_array_equal(filter_frame(img, name), expected)

    @requirement("VELA-FR-009")
    def test_original_mode(self):
        """Mode original : l'image brute convertie en RGB565, sans traitement."""
        img = random_rgb(24, 24, seed=2)
        np.testing.assert_array_equal(filter_frame(img, "original"), rgb888_to_rgb565(img))

    @requirement("VELA-FR-009")
    def test_parameters_are_forwarded(self):
        """THRESHOLD et MAG_SHIFT changent bien le résultat."""
        img = random_rgb(24, 24, seed=3)
        assert not np.array_equal(filter_frame(img, "threshold", threshold=50),
                                  filter_frame(img, "threshold", threshold=200))
        assert not np.array_equal(filter_frame(img, "sobel", mag_shift=0),
                                  filter_frame(img, "sobel", mag_shift=3))

    @requirement("VELA-FR-009")
    def test_unknown_filter_rejected(self):
        with pytest.raises(ValueError):
            filter_frame(random_rgb(4, 4, seed=4), "blur")

    @requirement("VELA-FR-009")
    def test_all_modes_shape(self):
        """Chaque mode produit une image RGB565 de même taille que l'entrée."""
        img = random_rgb(24, 24, seed=5)
        for name in FILTERS:
            out = filter_frame(img, name)
            assert out.shape == (24, 24) and out.dtype == np.uint16


# ─── GUI frame composition ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gui():
    """python/gui_simulator.py module (importing it opens no window)."""
    return pytest.importorskip("gui_simulator")


class TestGuiFrame:
    """compose_frame() of python/gui_simulator.py (no Tk window needed)."""

    @requirement("VELA-FR-007")
    def test_hw_view_copies_pixels_unchanged(self, gui):
        """Vue HW : les pixels de l'écran sont recopiés tels quels (pas de min/max)."""
        img = rgb565_to_rgb888(filter_frame(random_rgb(24, 24, seed=6), "sobel"))
        frame = gui.compose_frame(img, rescale=False)
        y0 = (gui.OLED_H - gui.TEXT_ZONE - 24) // 2
        x0 = (gui.OLED_W - 24) // 2
        np.testing.assert_array_equal(frame[y0:y0 + 24, x0:x0 + 24], img.transpose(1, 2, 0))
        assert frame.shape == (64, 96, 3)

    @requirement("VELA-FR-007")
    def test_hw_view_rejects_float_images(self, gui):
        """Vue HW : une image non uint8 est refusée (elle aurait besoin d'une mise à l'échelle)."""
        with pytest.raises(ValueError):
            gui.compose_frame(np.zeros((3, 24, 24), dtype=np.float32), rescale=False)

    @requirement("VELA-FR-007")
    def test_golden_view_rescales(self, gui):
        """Vue golden : image normalisée (valeurs négatives) remise sur [0, 255]."""
        img = np.linspace(-2.0, 3.0, 3 * 24 * 24, dtype=np.float32).reshape(3, 24, 24)
        frame = gui.compose_frame(img, rescale=True)
        y0 = (gui.OLED_H - gui.TEXT_ZONE - 24) // 2
        x0 = (gui.OLED_W - 24) // 2
        region = frame[y0:y0 + 24, x0:x0 + 24]
        assert region.min() == 0 and region.max() == 255
