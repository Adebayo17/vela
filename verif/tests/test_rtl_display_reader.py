"""
test_rtl_display_reader.py — display_reader.v co-simulation (Phase 3, VELA-TN-007 §6.3)

Le flux AXI-Stream doit être la mémoire d'affichage lue en boucle, dans
l'ordre raster, sans pixel perdu ni dupliqué, quels que soient tready (OLED)
et grant (moteur prioritaire). Le banc vérifie aussi les règles AXI-Stream et
qu'aucune lecture n'a lieu avant le premier tready ni quand grant est bas.
"""

import numpy as np
import pytest
from vela_testing import requirement
from rtl_sim import requires_iverilog, run_testbench


def memory_words(n, seed):
    """Random words: RGB565 in bits 15:0, random garbage in bits 31:16."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(np.uint32)


def simulate(words, count, tmp_path, seed=0, tready_at=20):
    return run_testbench("tb_display_reader", ["display_reader.v"], words, 32, tmp_path,
                         seed=seed, params={"N_PIXELS": len(words)}, out_width=16,
                         plusargs={"COUNT": count, "TREADY_AT": tready_at})


def expected_stream(words, count):
    """Pixel k of the stream = bits 15:0 of word k mod N."""
    return (words[np.arange(count) % len(words)] & 0xFFFF).astype(np.uint32)


@requires_iverilog
class TestDisplayReaderRtl:
    """display_reader.v simulated with Icarus Verilog."""

    @requirement("VELA-FR-012")
    def test_continuous_stream(self, tmp_path):
        """tready et grant toujours hauts : 3 images complètes dans l'ordre, bouclage."""
        words = memory_words(20, seed=1)
        out = simulate(words, 60, tmp_path)
        np.testing.assert_array_equal(out, expected_stream(words, 60))

    @requirement("VELA-FR-012")
    @pytest.mark.parametrize("seed", [3, 17, 2024])
    def test_random_tready_and_grant(self, tmp_path, seed):
        """tready et grant aléatoires : ni perte ni doublon, règles AXI-Stream respectées."""
        words = memory_words(24, seed=seed)
        out = simulate(words, 100, tmp_path, seed=seed)
        np.testing.assert_array_equal(out, expected_stream(words, 100))

    @requirement("VELA-FR-012")
    def test_starts_at_word_zero_after_late_tready(self, tmp_path):
        """Premier tready tardif (fin de l'init OLED) : le flux commence au mot 0."""
        words = memory_words(16, seed=5)
        out = simulate(words, 20, tmp_path, tready_at=500)
        np.testing.assert_array_equal(out, expected_stream(words, 20))

    @requirement("VELA-FR-012")
    def test_full_display_size(self, tmp_path):
        """Taille réelle 96×64 (6 144 mots) : bouclage après le dernier mot."""
        words = memory_words(96 * 64, seed=7)
        out = simulate(words, 96 * 64 + 10, tmp_path, seed=11)
        np.testing.assert_array_equal(out, expected_stream(words, 96 * 64 + 10))
