"""
test_rtl_spi_master.py — spi_master.v protocol checks (VELA display path)

Le moniteur SPI (verif/tb/spi_monitor.v) relit le bus comme le ferait le
SSD1331 : octets reconstruits, mode 0, demi-période de SCLK. La demi-période
attendue est calculée ici à partir de la fréquence spécifiée, pas à partir
de la formule du RTL.
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench
from oled_hw import decode_monitor, sclk_half_period


def simulate(data, tmp_path, clk_freq=100_000_000, spi_freq=6_250_000, seed=0):
    """Send `data` bytes through spi_master, return the decoded (dc, byte, cycle) list."""
    words = run_testbench("tb_spi_master", ["spi_master.v"], data, 8, tmp_path, seed=seed,
                          params={"CLK_FREQ": clk_freq, "SPI_FREQ": spi_freq},
                          plusargs={"HALF": sclk_half_period(clk_freq, spi_freq)},
                          tb_sources=["spi_monitor.v"], out_width=48, raw=True)
    spi, _ = decode_monitor(words)
    return spi


@requires_iverilog
class TestSpiMasterRtl:
    """spi_master.v simulated with Icarus Verilog, bus decoded by spi_monitor."""

    @requirement("VELA-FR-009")
    def test_all_byte_values(self, tmp_path):
        """Les 256 valeurs, à la suite : chaque octet relu à l'identique, MSB d'abord."""
        data = list(range(256))
        spi = simulate(data, tmp_path)
        assert [b for _, b, _ in spi] == data

    @requirement("VELA-FR-009")
    def test_bit_order_patterns(self, tmp_path):
        """Motifs asymétriques : une inversion MSB/LSB ou un décalage se verrait."""
        data = [0x80, 0x01, 0xF0, 0x0F, 0xAA, 0x55, 0xC3, 0x3C]
        spi = simulate(data, tmp_path)
        assert [b for _, b, _ in spi] == data

    @requirement("VELA-FR-009")
    def test_gaps_and_spurious_starts(self, tmp_path):
        """Pauses aléatoires entre octets + impulsions start pendant busy (ignorées)."""
        rng = np.random.default_rng(1)
        data = [int(b) for b in rng.integers(0, 256, size=64)]
        spi = simulate(data, tmp_path, seed=2024)
        assert [b for _, b, _ in spi] == data

    @requirement("VELA-FR-009", "VELA-NFR-001")
    @pytest.mark.parametrize("clk_freq,spi_freq", [
        (100_000_000, 6_250_000),    # VELA target: half period 8 cycles
        (125_000_000, 6_250_000),    # board oscillator: half period 10 cycles
        (100_000_000, 12_500_000),   # faster SPI: half period 4 cycles
    ])
    def test_sclk_frequency(self, tmp_path, clk_freq, spi_freq):
        """Demi-période de SCLK conforme à CLK_FREQ / SPI_FREQ (vérifiée par le moniteur)."""
        data = [0x12, 0x34, 0x56]
        spi = simulate(data, tmp_path, clk_freq=clk_freq, spi_freq=spi_freq)
        assert [b for _, b, _ in spi] == data

    @requirement("VELA-FR-009")
    def test_back_to_back_byte_spacing(self, tmp_path):
        """Octets enchaînés : écart régulier entre débuts d'octet (aucune perte de cycle aléatoire)."""
        spi = simulate([0x00] * 8, tmp_path)
        starts = np.diff([c for _, _, c in spi])
        assert len(set(starts)) == 1
        # 16 half periods of 8 cycles + end step + restart overhead: well under 2x a byte
        assert 16 * 8 <= starts[0] < 2 * 16 * 8
