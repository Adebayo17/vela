"""
test_hls_cnn_ref.py — C++ double reference (hls/cnn_ref.cpp) vs golden model

Phase 2b, étape 1 : le CNN porté en C++ double doit calculer la même chose
que python/cnn.py. Comparaison couche par couche (pour localiser une erreur),
puis sur des images CIFAR-10 : logits, probabilités, classe prédite.

Le golden model calcule en float32, la référence C++ en double : les écarts
attendus sont de l'ordre de l'arrondi float32 (~1e-6 relatif).
"""

import numpy as np
import pytest
from vela_testing import requirement
from hls_sim import DATA_DIR, WEIGHTS_FILE, build_reference, requires_gxx, requires_weights, run_reference
from cnn import cnn_forward, conv2d_fast, maxpool2d, relu

pytestmark = [requires_gxx, requires_weights]


@pytest.fixture(scope="module")
def ref_exe(tmp_path_factory):
    return build_reference(tmp_path_factory.mktemp("cnn_ref"))


@pytest.fixture(scope="module")
def weights():
    from weights_loader import load_weights
    return load_weights(str(WEIGHTS_FILE))


def load_images(n):
    if not (DATA_DIR / "test_batch.bin").exists():
        pytest.skip(f"CIFAR-10 data not found in {DATA_DIR}")
    from data_loader import load_dataset
    return load_dataset(str(DATA_DIR), max_images=n)


def assert_close(actual, expected, name, rel=1e-5):
    """Equal up to float32 rounding, relative to the layer's dynamic range."""
    scale = max(float(np.abs(expected).max()), 1e-6)
    err = float(np.abs(actual - expected).max())
    assert err <= rel * scale, f"{name}: max error {err:.3g} > {rel * scale:.3g}"


@requirement("VELA-FR-001")
def test_layers_match_golden_model(ref_exe, weights, tmp_path):
    """Chaque couche intermédiaire (image 0) = la fonction Python correspondante."""
    images, _ = load_images(1)
    _, _, _, layers = run_reference(ref_exe, images, tmp_path, trace=True)

    x = images[0].astype(np.float32)
    conv1 = conv2d_fast(x, weights["conv1_w"], weights["conv1_b"])
    pool1 = maxpool2d(relu(conv1))
    conv2 = conv2d_fast(pool1, weights["conv2_w"], weights["conv2_b"])
    pool2 = maxpool2d(relu(conv2))
    conv3 = conv2d_fast(pool2, weights["conv3_w"], weights["conv3_b"])
    pool3 = maxpool2d(relu(conv3))
    vec = pool3.transpose(1, 2, 0).flatten()       # HWC order, as in cnn_forward

    for name, golden in [("conv1", conv1), ("pool1", pool1), ("conv2", conv2),
                         ("pool2", pool2), ("conv3", conv3), ("pool3", pool3), ("vec", vec)]:
        assert_close(layers[name], golden, name)


@requirement("VELA-FR-001", "VELA-FR-004")
def test_outputs_match_golden_model(ref_exe, weights, tmp_path):
    """50 images : logits et probabilités égaux à l'arrondi près, même classe."""
    images, _ = load_images(50)
    logits, probs, preds = run_reference(ref_exe, images, tmp_path)
    for i, img in enumerate(images):
        p_ref, pred_ref = cnn_forward(img, weights)
        assert_close(probs[i], p_ref, f"probs[{i}]", rel=1e-4)
        assert preds[i] == pred_ref, f"image {i}: C++ {preds[i]} vs Python {pred_ref}"


@requirement("VELA-FR-004")
def test_output_contract(ref_exe, tmp_path):
    """Probabilités dans [0, 1], somme 1, classe = argmax dans 0..9."""
    images, _ = load_images(20)
    logits, probs, preds = run_reference(ref_exe, images, tmp_path)
    assert np.all((probs >= 0) & (probs <= 1))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_array_equal(preds, logits.argmax(axis=1))
    assert np.all((preds >= 0) & (preds <= 9))


@pytest.mark.slow
@requirement("VELA-FR-001", "VELA-FR-005")
def test_1000_images_same_predictions(ref_exe, weights, tmp_path):
    """1000 images : mêmes classes que le golden model, précision ≥ 60 %."""
    images, labels = load_images(1000)
    _, _, preds = run_reference(ref_exe, images, tmp_path)
    preds_ref = np.array([cnn_forward(img, weights)[1] for img in images])
    disagree = np.flatnonzero(preds != preds_ref)
    assert disagree.size == 0, f"{disagree.size} image(s) differ: {disagree[:10]}"
    assert (preds == labels).mean() >= 0.60
