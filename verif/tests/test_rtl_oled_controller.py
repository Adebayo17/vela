"""
test_rtl_oled_controller.py — oled_controller.v (+ spi_master.v) system checks

Le bus SPI est relu par spi_monitor.v et comparé à ce que le SSD1331 doit
recevoir (oled_hw.expected_transactions) : 31 commandes (dc = 0, la dernière
étant Display ON) puis 2 octets par pixel RGB565 (dc = 1). Les signaux
d'alimentation et de reset sont horodatés pour vérifier l'ordre de la
datasheet (VCC allumé après le reset et l'init) et les délais.

Les tests fonctionnels utilisent des délais courts ; un test vérifie les
délais réels à 100 MHz (20 ms, 100 µs, 25 ms, 100 ms).
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from oled_hw import (DISPLAY_ON_S, POWER_UP_S, RESET_S, SSD1331_INIT, VCC_ON_S,
                     decode_monitor, expected_transactions, sclk_half_period)

CLK = 100_000_000
PMODEN, VCCEN, RES, TREADY = 1, 2, 3, 4
SHORT = {"power_up": 200, "reset": 50, "vcc_on": 300, "display_on": 400}   # cycles


def simulate(pixels, tmp_path, seed=0, delays=SHORT):
    """
    Run tb_oled_controller; return (spi, events) decoded from the monitor output.

    delays: dict of delay overrides in clock cycles (short, for speed), or
    None to keep the RTL defaults derived from CLK_FREQ.
    """
    params = {"CLK_FREQ": CLK, "SPI_FREQ": 6_250_000}
    if delays is not None:
        params.update({"POWER_UP_CYCLES": delays["power_up"], "RESET_CYCLES": delays["reset"],
                       "VCC_ON_CYCLES": delays["vcc_on"],
                       "DISPLAY_ON_CYCLES": delays["display_on"]})
        wait = delays["power_up"] + 2 * delays["reset"] + delays["vcc_on"] + delays["display_on"]
    else:
        wait = round((POWER_UP_S + 2 * RESET_S + VCC_ON_S + DISPLAY_ON_S) * CLK)
    nbytes = len(expected_transactions(pixels))
    maxcyc = wait + (nbytes + 4) * 400 + 10_000
    words = run_testbench("tb_oled_controller", ["oled_controller.v", "spi_master.v"],
                          pixels, 16, tmp_path, seed=seed, params=params,
                          plusargs={"NBYTES": nbytes, "HALF": sclk_half_period(CLK),
                                    "MAXCYC": maxcyc},
                          tb_sources=["spi_monitor.v"], out_width=48, raw=True,
                          timeout=600)
    return decode_monitor(words)


def event_cycle(events, sig, value):
    """Cycle of the first change of `sig` to `value`."""
    return next(c for s, v, c in events if s == sig and v == value)


def random_rgb565(n, seed):
    rng = np.random.default_rng(seed)
    return [int(p) for p in rng.integers(0, 1 << 16, size=n)]


@requires_iverilog
class TestOledControllerRtl:
    """oled_controller.v + spi_master.v simulated, SPI bus decoded by spi_monitor."""

    @requirement("VELA-FR-009")
    def test_init_sequence(self, tmp_path):
        """Les 31 commandes (init + Display ON) sont envoyées dans l'ordre, dc = 0."""
        spi, _ = simulate([0x0000], tmp_path)
        init = [(dc, b) for dc, b, _ in spi[:len(SSD1331_INIT)]]
        assert init == [(0, b) for b in SSD1331_INIT]

    @requirement("VELA-FR-009")
    def test_pixels_after_init(self, tmp_path):
        """Pixels RGB565 : octet haut puis octet bas, en mode données (dc = 1)."""
        pixels = [0xF800, 0x07E0, 0x001F, 0xFFFF, 0x0000, 0x1234]   # R, G, B, white, black, mixed
        spi, _ = simulate(pixels, tmp_path)
        assert [(dc, b) for dc, b, _ in spi] == expected_transactions(pixels)

    @requirement("VELA-FR-009")
    def test_pixels_with_tvalid_gaps(self, tmp_path):
        """tvalid avec trous aléatoires : aucun pixel perdu ni dupliqué."""
        pixels = random_rgb565(40, seed=5)
        spi, _ = simulate(pixels, tmp_path, seed=777)
        assert [(dc, b) for dc, b, _ in spi] == expected_transactions(pixels)

    @requirement("VELA-FR-009")
    def test_power_and_reset_sequence(self, tmp_path):
        """Ordre de la datasheet : logique, reset, init, puis VCC, Display ON, pixels."""
        d = SHORT
        spi, events = simulate([0x0000], tmp_path)
        t_pmoden = event_cycle(events, PMODEN, 1)
        t_res_low = event_cycle(events, RES, 0)
        t_res_high = event_cycle(events, RES, 1)
        t_vccen = event_cycle(events, VCCEN, 1)
        t_tready = event_cycle(events, TREADY, 1)
        last_init = spi[len(SSD1331_INIT) - 2][2]      # last ROM byte (before 0xAF)
        t_display_on = spi[len(SSD1331_INIT) - 1][2]   # 0xAF
        assert spi[len(SSD1331_INIT) - 1][1] == 0xAF
        # logic supply, wait, reset pulse, wait, first command
        assert d["power_up"] <= t_res_low - t_pmoden <= d["power_up"] + 2
        assert d["reset"] <= t_res_high - t_res_low <= d["reset"] + 2
        assert spi[0][2] - t_res_high >= d["reset"]
        # VCC only after the reset and the init commands, then the settling delay
        assert t_vccen > last_init and t_vccen > t_res_high
        assert t_display_on - t_vccen >= d["vcc_on"]
        # pixels accepted only after the display turn-on delay
        assert t_tready - t_display_on >= d["display_on"]

    @requirement("VELA-FR-009")
    def test_vccen_low_during_reset_and_init(self, tmp_path):
        """VCCEN ne change qu'une fois (0 -> 1), après la dernière commande d'init."""
        spi, events = simulate([0x0000], tmp_path)
        vcc_changes = [(v, c) for s, v, c in events if s == VCCEN]
        assert [v for v, _ in vcc_changes] == [1]
        assert vcc_changes[0][1] > spi[len(SSD1331_INIT) - 2][2]

    @requirement("VELA-FR-009")
    def test_tready_only_after_init(self, tmp_path):
        """s_axis_tready ne monte qu'après le dernier octet d'init (Display ON)."""
        spi, events = simulate([0x0000], tmp_path)
        last_init = spi[len(SSD1331_INIT) - 1][2]
        assert event_cycle(events, TREADY, 1) > last_init

    @pytest.mark.slow   # ~14.5 million simulated cycles: run by `make test-all`
    @requirement("VELA-FR-009", "VELA-NFR-001")
    def test_real_delays_at_100mhz(self, tmp_path):
        """Délais par défaut à 100 MHz : 20 ms, 100 µs, 25 ms, 100 ms (≈ 14,5 M cycles)."""
        spi, events = simulate([0x0000], tmp_path, delays=None)
        t_pmoden = event_cycle(events, PMODEN, 1)
        t_res_low = event_cycle(events, RES, 0)
        t_res_high = event_cycle(events, RES, 1)
        t_vccen = event_cycle(events, VCCEN, 1)
        t_tready = event_cycle(events, TREADY, 1)
        t_display_on = spi[len(SSD1331_INIT) - 1][2]
        assert abs((t_res_low - t_pmoden) / CLK - POWER_UP_S) < 1e-6
        assert abs((t_res_high - t_res_low) / CLK - RESET_S) < 1e-6
        # VCC -> Display ON: 25 ms + the start of the SPI byte (< 1 byte time)
        assert 0 <= (t_display_on - t_vccen) / CLK - VCC_ON_S < 2e-6
        # Display ON -> tready: 100 ms + the rest of the 0xAF byte
        assert 0 <= (t_tready - t_display_on) / CLK - DISPLAY_ON_S < 2e-6
        assert [(dc, b) for dc, b, _ in spi] == expected_transactions([0x0000])
