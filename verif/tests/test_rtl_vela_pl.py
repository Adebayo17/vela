"""
test_rtl_vela_pl.py — vela_pl.v system co-simulation (Phase 3, VELA-TN-007 §6)

Moteur de filtres, lecteur d'affichage et contrôleur OLED réunis, avec les
deux mémoires modélisées ; le bus SPI est relu comme le ferait le SSD1331.
Ce que l'écran reçoit après l'init doit être la mémoire d'affichage, image
après image : zone image = sortie du moteur (pipeline_hw.filter_frame), zone
texte = ce que l'ARM y a écrit, jamais modifiée par la logique programmable.
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from oled_hw import SSD1331_INIT, decode_monitor, sclk_half_period
from pipeline_hw import ENGINE_MODES, filter_frame

SOURCES = ["vela_pl.v", "filter_engine.v", "display_reader.v", "oled_controller.v",
           "spi_master.v", "grayscale.v", "threshold.v", "sobel_filter.v",
           "gaussian_filter.v", "median_filter.v", "window3x3.v"]
W, H, DW = 8, 6, 12                     # image zone, display width (small, for speed)
FRAMES = 3
OLD = 0xFFFF                            # image zone before the engine runs


def random_rgb(seed):
    """Values ≤ 250: no copy / filter output can be 0xFFFF (= OLD)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 251, size=(3, H, W), dtype=np.uint8)


def display_memory(img, mode):
    """Expected display memory (H, DW): engine output | ARM text pattern."""
    mem = (0x0800 + np.arange(H * DW)).reshape(H, DW).astype(np.uint32)
    mem[:, :W] = filter_frame(img, ENGINE_MODES[mode])
    return mem


def simulate(img, mode, start_at, tmp_path):
    """Run tb_vela_pl; return (pixels received by the OLED, done cycle, pixel cycles)."""
    r, g, b = (img[c].astype(np.uint32).ravel() for c in range(3))
    nbytes = len(SSD1331_INIT) + 2 * DW * H * FRAMES
    words = run_testbench("tb_vela_pl", SOURCES, (r << 16) | (g << 8) | b, 32, tmp_path,
                          params={"SRC_WIDTH": W, "SRC_HEIGHT": H, "DISP_WIDTH": DW},
                          plusargs={"MODE": mode, "START_AT": start_at, "NBYTES": nbytes,
                                    "HALF": sclk_half_period(100_000_000),
                                    "MAXCYC": 200_000},
                          tb_sources=["spi_monitor.v"], out_width=48, raw=True, timeout=600)
    spi, events = decode_monitor(words)
    init = [(dc, byte) for dc, byte, _ in spi[:len(SSD1331_INIT)]]
    assert init == [(0, byte) for byte in SSD1331_INIT]
    data = spi[len(SSD1331_INIT):]
    assert all(dc == 1 for dc, _, _ in data)
    pixels = np.array([(data[k][1] << 8) | data[k + 1][1] for k in range(0, len(data), 2)])
    cycles = np.array([data[k][2] for k in range(0, len(data), 2)])
    done = [c for sig, _, c in events if sig == 5]
    assert len(done) == 1, done
    return pixels, done[0], cycles


@requires_iverilog
class TestVelaPlRtl:
    """vela_pl.v: engine + display_reader + oled_controller, SPI decoded."""

    @requirement("VELA-FR-012")
    @requirement("VELA-FR-013")
    @pytest.mark.parametrize("mode", [0, 3, 4])
    def test_engine_before_display(self, tmp_path, mode):
        """Moteur lancé pendant l'init de l'OLED : chaque image reçue = la mémoire
        d'affichage finale (zone image filtrée, zone texte intacte), en raster."""
        img = random_rgb(seed=mode)
        pixels, _, _ = simulate(img, mode, start_at=10, tmp_path=tmp_path)
        expected = np.tile(display_memory(img, mode).ravel(), FRAMES)
        np.testing.assert_array_equal(pixels, expected)

    @requirement("VELA-FR-012")
    @requirement("VELA-FR-013")
    @pytest.mark.parametrize("mode", [0, 5])
    def test_engine_during_streaming(self, tmp_path, mode):
        """Moteur lancé au milieu de la 1re image envoyée : aucun pixel perdu ni
        décalé ; chaque position passe une seule fois de l'ancienne valeur à la
        nouvelle ; la zone texte ne change jamais ; dernière image = nouvelle."""
        img = random_rgb(seed=20 + mode)
        new = display_memory(img, mode).ravel()
        old = new.copy().reshape(H, DW)
        old[:, :W] = OLD
        old = old.ravel()
        pixels, done, cycles = simulate(img, mode, start_at=14_000, tmp_path=tmp_path)
        # The engine really ran while the first frame was being sent
        assert cycles[0] < done < cycles[DW * H - 1], (cycles[0], done, cycles[DW * H - 1])
        frames = pixels.reshape(FRAMES, DW * H)
        assert np.all((frames == old) | (frames == new))
        is_new = frames == new
        assert np.all(is_new[1:] >= is_new[:-1])          # old -> new, never back
        np.testing.assert_array_equal(frames[-1], new)
        text = (np.arange(DW * H) % DW) >= W
        assert np.all(frames[:, text] == new[text])
