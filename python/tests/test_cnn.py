"""
test_cnn.py — Integration tests for the full CNN forward pass.

Requirements covered:
  VELA-FR-004 : cnn_forward() returns (probs array of shape (10,), pred int in [0,9])
  VELA-FR-001 : all layers integrated correctly in the pipeline

These tests use loaded weights (fixture from conftest.py).
They are skipped automatically if CNN_coeff_3x3.txt is not accessible.
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cnn import cnn_forward
from data_loader import preprocess, CIFAR10_CLASSES
from .helpers import requirement


class TestCnnForwardOutputContract:
    """VELA-FR-004 : output contract of cnn_forward()."""

    @requirement("VELA-FR-004")
    def test_output_types(self, loaded_weights, synthetic_rgb_image_32x32):
        """cnn_forward must return (np.ndarray, int)."""
        img_pp = preprocess(synthetic_rgb_image_32x32)
        probs, pred = cnn_forward(img_pp, loaded_weights)
        assert isinstance(probs, np.ndarray), "probs must be np.ndarray"
        assert isinstance(pred, int), "pred must be int"

    @requirement("VELA-FR-004")
    def test_probs_shape(self, loaded_weights, synthetic_rgb_image_32x32):
        img_pp = preprocess(synthetic_rgb_image_32x32)
        probs, _ = cnn_forward(img_pp, loaded_weights)
        assert probs.shape == (10,), f"Expected (10,), got {probs.shape}"

    @requirement("VELA-FR-004")
    def test_probs_sum_to_one(self, loaded_weights, synthetic_rgb_image_32x32):
        img_pp = preprocess(synthetic_rgb_image_32x32)
        probs, _ = cnn_forward(img_pp, loaded_weights)
        assert abs(probs.sum() - 1.0) < 1e-5

    @requirement("VELA-FR-004")
    def test_pred_in_valid_range(self, loaded_weights, synthetic_rgb_image_32x32):
        img_pp = preprocess(synthetic_rgb_image_32x32)
        _, pred = cnn_forward(img_pp, loaded_weights)
        assert 0 <= pred <= 9, f"Class index must be in [0,9], got {pred}"

    @requirement("VELA-FR-004")
    def test_pred_equals_argmax_of_probs(self, loaded_weights,
                                          synthetic_rgb_image_32x32):
        img_pp = preprocess(synthetic_rgb_image_32x32)
        probs, pred = cnn_forward(img_pp, loaded_weights)
        assert pred == int(np.argmax(probs))

    @requirement("VELA-FR-004")
    def test_deterministic_on_same_input(self, loaded_weights,
                                          synthetic_rgb_image_32x32):
        """Same input must always produce the same output (no randomness)."""
        img_pp = preprocess(synthetic_rgb_image_32x32)
        probs1, pred1 = cnn_forward(img_pp, loaded_weights)
        probs2, pred2 = cnn_forward(img_pp, loaded_weights)
        np.testing.assert_array_equal(probs1, probs2)
        assert pred1 == pred2

    @requirement("VELA-FR-004")
    def test_different_images_can_give_different_predictions(
            self, loaded_weights, sample_images_10):
        """
        The network must not always predict the same class regardless of input.
        With 10 images from 10 different classes, at least 2 distinct
        predictions are expected.
        """
        images, _ = sample_images_10
        predictions = set()
        for img in images:
            _, pred = cnn_forward(img, loaded_weights)
            predictions.add(pred)
        assert len(predictions) >= 2, (
            "Network predicts the same class for all inputs — likely a bug "
            "in weights loading or reshape order."
        )


# ═══════════════════════════════════════════════════════════════════════════════
"""
test_system.py — System-level accuracy tests.

Requirements covered:
  VELA-FR-005 : top-1 accuracy >= 60% on CIFAR-10 test set
  VELA-FR-006 : confusion matrix and per-class accuracy

Marked as @pytest.mark.slow — excluded from fast CI runs with:
  pytest -m "not slow"

Run explicitly with:
  pytest tests/test_system.py -v
  python top.py --images 1000
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cnn import cnn_forward
from data_loader import load_dataset, CIFAR10_CLASSES
from .helpers import requirement


class TestSystemAccuracy:

    @pytest.mark.slow
    @requirement("VELA-FR-005")
    def test_VELAFR005_accuracy_100_images(self, loaded_weights,
                                            sample_images_100):
        """
        VELA-FR-005 fast smoke test : accuracy >= 55% on 100 images.
        (Slightly lower threshold than 1000-image test to account for
        statistical variance on a small sample.)
        """
        images, labels = sample_images_100
        correct = sum(
            1 for img, lbl in zip(images, labels)
            if cnn_forward(img, loaded_weights)[1] == lbl
        )
        accuracy = correct / len(images)
        assert accuracy >= 0.55, (
            f"Accuracy on 100 images: {accuracy*100:.1f}% < 55%. "
            "Check reshape order (CHW vs HWC) and weight loading."
        )

    @pytest.mark.slow
    @requirement("VELA-FR-005")
    def test_VELAFR005_accuracy_1000_images(self, loaded_weights,
                                             data_dir, data_available):
        """
        VELA-FR-005 main criterion : accuracy >= 60% on 1000 images.
        This is the Sprint 1 exit criterion.
        """
        if not data_available:
            pytest.skip("CIFAR-10 data not available")
        images, labels = load_dataset(data_dir, max_images=1000)
        correct = sum(
            1 for img, lbl in zip(images, labels)
            if cnn_forward(img, loaded_weights)[1] == lbl
        )
        accuracy = correct / 1000
        assert accuracy >= 0.60, (
            f"Accuracy on 1000 images: {accuracy*100:.1f}% < 60%. "
            "Sprint 1 exit criterion NOT met."
        )

    @pytest.mark.slow
    @requirement("VELA-FR-006")
    def test_VELAFR006_no_class_has_zero_accuracy(self, loaded_weights,
                                                    sample_images_100):
        """
        VELA-FR-006 : per-class accuracy must be non-zero for all classes.
        Zero accuracy on any class indicates a systematic ordering bug.
        """
        images, labels = sample_images_100
        per_class_correct = {i: 0 for i in range(10)}
        per_class_total   = {i: 0 for i in range(10)}

        for img, lbl in zip(images, labels):
            _, pred = cnn_forward(img, loaded_weights)
            per_class_total[int(lbl)] += 1
            if pred == int(lbl):
                per_class_correct[int(lbl)] += 1

        for cls_idx in range(10):
            total = per_class_total[cls_idx]
            if total == 0:
                continue   # class not present in sample — skip
            acc = per_class_correct[cls_idx] / total
            assert acc > 0.0, (
                f"Class '{CIFAR10_CLASSES[cls_idx]}' has 0% accuracy "
                f"on {total} images. This is a systematic bug."
            )