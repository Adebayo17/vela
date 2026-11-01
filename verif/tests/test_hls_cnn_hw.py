"""
test_hls_cnn_hw.py — Fixed-point CNN (hls/cnn_hw.cpp) vs its models

Phase 2b, étape 3. Deux comparaisons, comme pour les filtres RTL :

  ② bit-exact : le C++ ap_fixed (compilé avec g++ et les en-têtes de Vitis,
    comme une C-simulation Vitis HLS) = le modèle verif/models/cnn_hw_model.py
    (formats de TN004), couche par couche et en sortie, saturation comprise.
  ① fidélité : même classe que le golden model flottant sur ≥ 99 % des images
    (VELA-NFR-006), précision ≥ 60 % (FR-005), aucune classe à 0 % (FR-006).
"""

import numpy as np
import pytest
from vela_testing import requirement
from hls_sim import (DATA_DIR, WEIGHTS_FILE, build_hw, requires_ap_fixed, requires_gxx,
                     requires_weights, run_hw)
from cnn import cnn_forward
from cnn_hw_model import emulate_tn004

pytestmark = [requires_gxx, requires_weights, requires_ap_fixed]


@pytest.fixture(scope="module")
def hw_exe(tmp_path_factory):
    return build_hw(tmp_path_factory.mktemp("cnn_hw"))


@pytest.fixture(scope="module")
def weights():
    from weights_loader import load_weights
    return load_weights(str(WEIGHTS_FILE))


def load_images(n):
    if not (DATA_DIR / "test_batch.bin").exists():
        pytest.skip(f"CIFAR-10 data not found in {DATA_DIR}")
    from data_loader import load_dataset
    return load_dataset(str(DATA_DIR), max_images=n)


# ─── ② Bit-exact vs the fixed-point model ────────────────────────────────────

@requirement("VELA-NFR-006")
def test_layers_bit_exact(hw_exe, weights, tmp_path):
    """pool1, pool2, pool3 de l'image 0 : identiques au bit près au modèle."""
    images, _ = load_images(1)
    _, _, _, layers = run_hw(hw_exe, images, tmp_path, trace=True)
    model = emulate_tn004(images, weights)
    for name in ("pool1", "pool2", "pool3"):
        np.testing.assert_array_equal(layers[name], model[name], err_msg=name)


@requirement("VELA-NFR-006")
def test_outputs_bit_exact(hw_exe, weights, tmp_path):
    """200 images : scores exacts, logits (logit_t) et classes identiques au modèle."""
    images, _ = load_images(200)
    logits, scores, preds = run_hw(hw_exe, images, tmp_path)
    model = emulate_tn004(images, weights)
    np.testing.assert_array_equal(scores, model["scores"])
    np.testing.assert_array_equal(logits, model["logits"])
    np.testing.assert_array_equal(preds, model["pred"])


@requirement("VELA-NFR-006")
def test_saturation_bit_exact(hw_exe, weights, tmp_path):
    """Images x5 : entrée et activations saturent ; le C++ sature comme le modèle."""
    images, _ = load_images(20)
    images = images * 5.0
    assert np.abs(images).max() > 16          # beyond input_t range (+-16)
    logits, scores, preds = run_hw(hw_exe, images, tmp_path)
    model = emulate_tn004(images, weights)
    np.testing.assert_array_equal(scores, model["scores"])
    np.testing.assert_array_equal(logits, model["logits"])
    np.testing.assert_array_equal(preds, model["pred"])


# ─── ① Fidelity to the float golden model ────────────────────────────────────

@requirement("VELA-NFR-006", "VELA-FR-001")
def test_agreement_with_golden_200_images(hw_exe, weights, tmp_path):
    """200 images : même classe que le golden model flottant sur ≥ 99 %."""
    images, _ = load_images(200)
    _, _, preds = run_hw(hw_exe, images, tmp_path)
    golden = np.array([cnn_forward(img, weights)[1] for img in images])
    assert (preds == golden).mean() >= 0.99


@pytest.mark.slow
@requirement("VELA-NFR-006", "VELA-FR-005", "VELA-FR-006")
def test_10000_images(hw_exe, weights, tmp_path):
    """10 000 images : accord ≥ 99 %, précision ≥ 60 %, aucune classe à 0 %, bit-exact."""
    images, labels = load_images(10000)
    _, _, preds = run_hw(hw_exe, images, tmp_path)
    golden = np.array([cnn_forward(img, weights)[1] for img in images])
    assert (preds == golden).mean() >= 0.99
    assert (preds == labels).mean() >= 0.60
    for c in range(10):
        assert (preds[labels == c] == c).mean() > 0, f"class {c} at 0 %"
    np.testing.assert_array_equal(preds, emulate_tn004(images, weights)["pred"])
