"""
test_rtl_oled_controller.py — oled_controller.v (+ spi_master.v) system checks

Le bus SPI est relu par spi_monitor.v et comparé à ce que le SSD1331 doit
recevoir (oled_hw.expected_transactions) : 31 commandes d'init (dc = 0) puis
2 octets par pixel RGB565 (dc = 1). Les signaux d'alimentation et de reset
sont horodatés pour vérifier l'ordre et les délais.

Les tests fonctionnels utilisent des délais courts (POWER_UP_CYCLES,
RESET_CYCLES) ; un test vérifie les délais réels à 100 MHz (20 ms, 100 µs).
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from oled_hw import (POWER_UP_S, RESET_S, SSD1331_INIT, decode_monitor,
                     expected_transactions, sclk_half_period)

CLK = 100_000_000
PMODEN, VCCEN, RES, TREADY = 1, 2, 3, 4


def simulate(pixels, tmp_path, seed=0, power_up=200, reset=50):
    """
    Run tb_oled_controller; return (spi, events) decoded from the monitor output.

    power_up / reset: delay overrides in clock cycles (short, for speed), or
    None to keep the RTL defaults derived from CLK_FREQ (20 ms, 100 us).
    """
    params = {"CLK_FREQ": CLK, "SPI_FREQ": 6_250_000}
    if power_up is not None:
        params.update({"POWER_UP_CYCLES": power_up, "RESET_CYCLES": reset})
        wait = power_up + 2 * reset
    else:
        wait = round((POWER_UP_S + 2 * RESET_S) * CLK)
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
        """Les 31 octets d'init sont envoyés dans l'ordre, en mode commande (dc = 0)."""
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
        """Ordre : alimentations, attente, reset bas, reset haut, attente, 1re commande."""
        power_up, reset = 200, 50
        spi, events = simulate([0x0000], tmp_path, power_up=power_up, reset=reset)
        t_pmoden = event_cycle(events, PMODEN, 1)
        t_vccen = event_cycle(events, VCCEN, 1)
        t_res_low = event_cycle(events, RES, 0)
        t_res_high = event_cycle(events, RES, 1)
        t_first_cmd = spi[0][2]
        assert t_pmoden == t_vccen
        assert power_up <= t_res_low - t_pmoden <= power_up + 2
        assert reset <= t_res_high - t_res_low <= reset + 2
        assert t_first_cmd - t_res_high >= reset

    @requirement("VELA-FR-009")
    def test_tready_only_after_init(self, tmp_path):
        """s_axis_tready ne monte qu'après le dernier octet d'init (Display ON)."""
        spi, events = simulate([0x0000], tmp_path)
        last_init = spi[len(SSD1331_INIT) - 1][2]
        assert event_cycle(events, TREADY, 1) > last_init

    @pytest.mark.slow   # ~2 million simulated cycles: run by `make test-all`
    @requirement("VELA-FR-009", "VELA-NFR-001")
    def test_real_delays_at_100mhz(self, tmp_path):
        """Délais par défaut à 100 MHz : 20 ms d'attente, 100 µs de reset (≈ 2 M cycles)."""
        spi, events = simulate([0x0000], tmp_path, power_up=None)
        t_pmoden = event_cycle(events, PMODEN, 1)
        t_res_low = event_cycle(events, RES, 0)
        t_res_high = event_cycle(events, RES, 1)
        assert abs((t_res_low - t_pmoden) / CLK - POWER_UP_S) < 1e-6
        assert abs((t_res_high - t_res_low) / CLK - RESET_S) < 1e-6
        assert [(dc, b) for dc, b, _ in spi] == expected_transactions([0x0000])
