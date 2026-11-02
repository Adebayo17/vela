"""
test_sw_host.py — Portable ARM code (sw/vela_app/src) vs the Python reference

Le code C qui tournera sur l'ARM (prétraitement, rangement en mots, softmax,
image ×2, dessin du texte) est compilé avec gcc sur le PC et comparé aux
références Python, avant tout essai sur la carte (VELA-TN-007 §10.2) :
  - prétraitement = data_loader.preprocess + cnn_hw_model.input_raw ;
  - mots de la fenêtre image = cnn_hw_model.pack_image_words ;
  - le texte ne touche que la zone texte (colonnes 64-95, VELA-FR-013) ;
  - golden_ref.h est à jour par rapport à tools/gen_sw_golden.py.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from vela_testing import requirement
from hls_sim import DATA_DIR, WEIGHTS_FILE, requires_weights

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "sw" / "vela_app" / "src"
HARNESS = ROOT / "verif" / "sw" / "host_harness.c"
GCC = shutil.which("gcc")

pytestmark = pytest.mark.skipif(GCC is None, reason="gcc not installed")

N_IMAGES = 1000
MARK = 0xA5A5_0000
DW, DH, TEXT_X0 = 96, 64, 64


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    exe = tmp_path_factory.mktemp("sw") / "host_harness"
    subprocess.run([GCC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    f"-I{SRC}", str(HARNESS), str(SRC / "vela_image.c"), str(SRC / "vela_text.c"),
                    "-lm", "-o", str(exe)], check=True, capture_output=True, text=True)
    return exe


def run(exe, mode, data=b""):
    return subprocess.run([str(exe), mode], input=data, capture_output=True, check=True).stdout


@pytest.fixture(scope="module")
def cifar():
    """Raw pixels (N, 3, 32, 32) uint8 and labels of the first test images."""
    path = DATA_DIR / "test_batch.bin"
    if not path.exists():
        pytest.skip(f"CIFAR-10 data not found in {DATA_DIR}")
    from data_loader import load_batch
    images, labels = load_batch(str(path))
    return images[:N_IMAGES].astype(np.uint8), labels[:N_IMAGES]


@pytest.fixture(scope="module")
def c_preprocessed(harness, cifar):
    """C raw codes (N, 1728) int16 and packed words (N, 864) uint32."""
    pixels, _ = cifar
    out = np.frombuffer(run(harness, "pre", pixels.tobytes()), dtype=np.uint8)
    per_image = 1728 * 2 + 864 * 4
    out = out.reshape(len(pixels), per_image)
    raw = out[:, :1728 * 2].copy().view(np.int16)
    words = out[:, 1728 * 2:].copy().view(np.uint32)
    return raw, words


@pytest.fixture(scope="module")
def py_raw(cifar):
    """Raw codes of the golden preprocessing (python/data_loader.py, float32)."""
    from cnn_hw_model import input_raw
    from data_loader import preprocess
    pixels, _ = cifar
    return np.stack([input_raw(preprocess(p.astype(np.float32))).reshape(-1) for p in pixels])


@pytest.fixture(scope="module")
def py_raw_f64(cifar):
    """Same preprocessing computed in float64, the precision used by the C code."""
    from cnn_hw_model import input_raw
    from data_loader import crop_center
    pixels, _ = cifar
    out = []
    for p in pixels:
        c = crop_center(p.astype(np.float64))
        out.append(input_raw((c - c.mean()) / max(c.std(), 1.0 / np.sqrt(c.size))).reshape(-1))
    return np.stack(out)


class TestPreprocessing:

    @requirement("VELA-FR-003")
    def test_raw_codes_exact_in_double(self, c_preprocessed, py_raw_f64):
        """Codes bruts C = même prétraitement Python calculé en float64, exactement
        (1 000 images × 1 728 valeurs)."""
        raw, _ = c_preprocessed
        np.testing.assert_array_equal(raw, py_raw_f64)

    @requirement("VELA-FR-003")
    def test_raw_codes_close_to_golden(self, c_preprocessed, py_raw):
        """Le golden model normalise en float32, le C en double : un code peut
        différer d'1 LSB quand la valeur tombe à la limite d'un arrondi
        (mesuré : 196 valeurs sur 1 728 000, soit 0,011 %)."""
        raw, _ = c_preprocessed
        diff = np.abs(raw.astype(np.int32) - py_raw.astype(np.int32))
        assert diff.max() <= 1, diff.max()
        assert np.count_nonzero(diff) <= 5e-4 * diff.size, np.count_nonzero(diff)

    @requirement("VELA-FR-003")
    def test_input_raw_rounding_and_saturation(self, harness):
        """Arrondi au plus proche (égalité vers le haut) et saturation 16 bits,
        valeurs absentes des images de test (|x| > 16 sature)."""
        from cnn_hw_model import input_raw
        lsb = 2.0 ** -11
        x = np.array([0.0, 0.5 * lsb, -0.5 * lsb, 1.5 * lsb, -1.5 * lsb, 0.49 * lsb,
                      15.99951171875, 16.0, 20.0, 1e9, -16.0, -16.0001, -20.0, -1e9,
                      3.14159, -2.71828], dtype=np.float64)
        out = np.frombuffer(run(harness, "raw", x.tobytes()), dtype=np.int16)
        np.testing.assert_array_equal(out, input_raw(x))

    @requirement("VELA-FR-003")
    def test_packing(self, c_preprocessed):
        """Mots de la fenêtre image = pack_image_words des mêmes codes (bit-exact)."""
        raw, words = c_preprocessed
        lo = raw[:, 0::2].view(np.uint16).astype(np.uint32)
        hi = raw[:, 1::2].view(np.uint16).astype(np.uint32)
        np.testing.assert_array_equal(words, lo | (hi << 16))

    @requirement("VELA-NFR-012")
    @requires_weights
    def test_same_cnn_class_as_python_codes(self, c_preprocessed, py_raw):
        """Les codes du C donnent la même classe (modèle bit-exact de l'IP) que
        ceux du golden model, sur les 1 000 images : l'écart d'arrondi est sans
        effet sur le résultat."""
        from cnn_hw_model import emulate_tn004
        from weights_loader import load_weights
        weights = load_weights(str(WEIGHTS_FILE))
        raw, _ = c_preprocessed
        n = N_IMAGES
        c_pred = emulate_tn004(raw[:n].reshape(n, 3, 24, 24) / 2.0 ** 11, weights)["pred"]
        py_pred = emulate_tn004(py_raw[:n].reshape(n, 3, 24, 24) / 2.0 ** 11, weights)["pred"]
        np.testing.assert_array_equal(c_pred, py_pred)


class TestDisplayData:

    @requirement("VELA-FR-010")
    def test_image_x2(self, harness, cifar):
        """Image ×2 : chaque pixel devient un carré 2×2, mots 0x00RRGGBB, 64×64."""
        pixels, _ = cifar
        img = pixels[7]
        out = np.frombuffer(run(harness, "x2", img.tobytes()), dtype=np.uint32).reshape(64, 64)
        big = img.repeat(2, axis=1).repeat(2, axis=2).astype(np.uint32)
        np.testing.assert_array_equal(out, (big[0] << 16) | (big[1] << 8) | big[2])

    @requirement("VELA-FR-010")
    def test_softmax(self, harness):
        """Probabilités = softmax(logits / 2^9), comme le PS doit les afficher
        (logits modérés : toutes les probabilités comptent)."""
        logits = np.array([-300, 120, 0, 409, -500, 700, 17, -1, 512, 205], np.int16)
        probs = np.frombuffer(run(harness, "soft", logits.tobytes()), dtype=np.float64)
        z = logits / 512.0
        ref = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        np.testing.assert_allclose(probs, ref, rtol=1e-12)

    @requirement("VELA-FR-010")
    def test_softmax_extreme_logits(self, harness):
        """Logits extrêmes (±64) : pas de débordement de exp (exp(64·…) sans
        décalage du maximum), probabilités finies, somme = 1."""
        logits = np.array([-32768, 32767, 0, 32767, -1, 1, 100, -100, 32000, -32000], np.int16)
        probs = np.frombuffer(run(harness, "soft", logits.tobytes()), dtype=np.float64)
        assert np.all(np.isfinite(probs)) and abs(probs.sum() - 1.0) < 1e-12
        z = logits / 512.0
        ref = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        np.testing.assert_allclose(probs, ref, rtol=1e-12, atol=1e-300)

    @requirement("VELA-FR-010")
    def test_unpack_logits(self, harness):
        """Décodage des 5 mots de logits = unpack_logit_words (TN005 §4)."""
        from cnn_hw_model import unpack_logit_words
        rng = np.random.default_rng(3)
        words = rng.integers(0, 1 << 32, size=5, dtype=np.uint64).astype(np.uint32)
        out = np.frombuffer(run(harness, "unpack", words.tobytes()), dtype=np.int16)
        np.testing.assert_array_equal(out / 512.0, unpack_logit_words(words))


@pytest.fixture(scope="module")
def disp(harness):
    """Display memory after the text drawing scenario of host_harness.c."""
    return np.frombuffer(run(harness, "text"), dtype=np.uint32).reshape(DH, DW)


class TestTextZone:

    @requirement("VELA-FR-013")
    def test_image_zone_untouched(self, disp):
        """Texte dessiné partout (dans la zone image, à cheval, hors écran) :
        les colonnes 0-63 gardent leur motif témoin."""
        marker = MARK | np.arange(DH * DW, dtype=np.uint32).reshape(DH, DW)
        np.testing.assert_array_equal(disp[:, :TEXT_X0], marker[:, :TEXT_X0])

    @requirement("VELA-FR-013")
    def test_text_zone_written(self, disp):
        """La zone texte est entièrement repeinte (plus aucun motif témoin) et
        ne contient que des pixels RGB565 (16 bits de poids fort à 0)."""
        zone = disp[:, TEXT_X0:]
        assert np.all(zone >> 16 == 0)

    @requirement("VELA-FR-010")
    def test_glyph_pixels(self, disp):
        """« C » de « CNN » en (64, 4) : ligne 0 = .###. (0x0E), ligne 1 = #...#."""
        yellow, black = 0xFFE0, 0x0000
        np.testing.assert_array_equal(disp[4, 64:70], [black, yellow, yellow, yellow, black, black])
        np.testing.assert_array_equal(disp[5, 64:70], [yellow, black, black, black, yellow, black])


@requirement("VELA-NFR-012")
@requires_weights
def test_golden_ref_up_to_date(cifar):
    """golden_ref.h = sortie actuelle de tools/gen_sw_golden.py (sinon : make sw-golden)."""
    import gen_sw_golden
    assert (SRC / "golden_ref.h").read_text() == gen_sw_golden.header_text(), \
        "golden_ref.h is stale: run `make sw-golden`"
