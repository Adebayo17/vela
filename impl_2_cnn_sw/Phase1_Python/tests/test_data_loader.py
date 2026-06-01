"""
test_data_loader.py — TDD tests for data_loader.py

Requirements covered:
  VELA-FR-002 : load pre-trained weights (weight file format)
  VELA-FR-003 : preprocess images — centre crop 24x24 + global normalisation

Each test follows the AAA pattern:
  Arrange  — set up inputs
  Act      — call the function under test
  Assert   — verify the output

Tests using only synthetic data (no fixture files needed) run in all
environments including CI. Tests requiring CIFAR-10 files are gated by
the `data_available` fixture and auto-skip if files are absent.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import (
    crop_center, normalize, preprocess,
    IMAGE_SIZE_ORIG, IMAGE_SIZE_CROP, N_CHANNELS, RECORD_SIZE
)
from conftest import requirement


# ── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    """VELA-FR-003 : verify that module constants match CIFAR-10 spec."""

    @requirement("VELA-FR-003")
    def test_image_size_original(self):
        assert IMAGE_SIZE_ORIG == 32, "Original CIFAR-10 image must be 32x32"

    @requirement("VELA-FR-003")
    def test_image_size_crop(self):
        assert IMAGE_SIZE_CROP == 24, "Target crop size must be 24x24"

    @requirement("VELA-FR-003")
    def test_n_channels(self):
        assert N_CHANNELS == 3, "CIFAR-10 images are RGB (3 channels)"

    @requirement("VELA-FR-003")
    def test_record_size(self):
        # 1 label byte + 3 * 32 * 32 pixel bytes = 3073
        assert RECORD_SIZE == 3073


# ── crop_center ──────────────────────────────────────────────────────────────

class TestCropCenter:
    """VELA-FR-003 : centre crop from 32x32 to 24x24."""

    @requirement("VELA-FR-003")
    def test_output_shape(self, synthetic_rgb_image_32x32):
        # Arrange
        img = synthetic_rgb_image_32x32          # (3, 32, 32)
        # Act
        cropped = crop_center(img, crop_size=24)
        # Assert
        assert cropped.shape == (3, 24, 24), (
            f"Expected (3, 24, 24), got {cropped.shape}"
        )

    @requirement("VELA-FR-003")
    def test_crop_is_centered(self, synthetic_rgb_image_32x32):
        """
        The crop must be centred.
        For 32→24 : offset = (32-24)//2 = 4
        Pixel at (0, 0, 0) in cropped = pixel at (0, 4, 4) in original.
        """
        img = synthetic_rgb_image_32x32
        cropped = crop_center(img, crop_size=24)
        expected_offset = (32 - 24) // 2   # = 4
        np.testing.assert_array_equal(
            cropped[:, 0, 0],
            img[:, expected_offset, expected_offset],
            err_msg="Top-left pixel of crop does not match expected offset"
        )

    @requirement("VELA-FR-003")
    def test_crop_preserves_channels(self, synthetic_rgb_image_32x32):
        cropped = crop_center(synthetic_rgb_image_32x32)
        assert cropped.shape[0] == 3, "Channel count must be preserved after crop"

    @requirement("VELA-FR-003")
    def test_crop_does_not_modify_original(self, synthetic_rgb_image_32x32):
        img_copy = synthetic_rgb_image_32x32.copy()
        crop_center(synthetic_rgb_image_32x32)
        np.testing.assert_array_equal(
            synthetic_rgb_image_32x32, img_copy,
            err_msg="crop_center must not modify the input array"
        )

    @requirement("VELA-FR-003")
    def test_crop_to_same_size_returns_identical(self):
        """Cropping to the same size as input must return the full image."""
        img = np.ones((3, 24, 24), dtype=np.float32)
        cropped = crop_center(img, crop_size=24)
        np.testing.assert_array_equal(cropped, img)


# ── normalize ────────────────────────────────────────────────────────────────

class TestNormalize:
    """
    VELA-FR-003 : global normalisation m' = (m - µ) / max(σ, 1/√N)
    across ALL channels simultaneously (not per-channel).
    """

    @requirement("VELA-FR-003")
    def test_output_shape_preserved(self):
        img = np.random.rand(3, 24, 24).astype(np.float32)
        out = normalize(img)
        assert out.shape == img.shape, "normalize must preserve shape"

    @requirement("VELA-FR-003")
    def test_mean_approximately_zero(self):
        """After normalisation, global mean must be ≈ 0."""
        rng = np.random.default_rng(seed=0)
        img = rng.uniform(0, 255, (3, 24, 24)).astype(np.float32)
        out = normalize(img)
        assert abs(out.mean()) < 0.01, (
            f"Normalised mean should be ≈ 0, got {out.mean():.6f}"
        )

    @requirement("VELA-FR-003")
    def test_std_approximately_one(self):
        """After normalisation, global std must be ≈ 1."""
        rng = np.random.default_rng(seed=1)
        img = rng.uniform(0, 255, (3, 24, 24)).astype(np.float32)
        out = normalize(img)
        assert abs(out.std() - 1.0) < 0.01, (
            f"Normalised std should be ≈ 1, got {out.std():.6f}"
        )

    @requirement("VELA-FR-003")
    def test_uniform_image_does_not_divide_by_zero(self):
        """
        A perfectly uniform image has σ = 0.
        The formula uses max(σ, 1/√N) to avoid division by zero.
        Result must be finite (no NaN or Inf).
        """
        img = np.full((3, 24, 24), 128.0, dtype=np.float32)
        out = normalize(img)
        assert np.all(np.isfinite(out)), "normalize must not produce NaN/Inf"

    @requirement("VELA-FR-003")
    def test_normalisation_is_global_not_per_channel(self):
        """
        µ and σ are computed across ALL channels together, not per channel.
        This preserves the relative colour balance between R, G, B.
        Verification: channel 0 all-zeros, channel 1 all-ones.
        Per-channel norm would make both channels zero.
        Global norm preserves the difference.
        """
        img = np.zeros((3, 24, 24), dtype=np.float32)
        img[0, :, :] = 0.0
        img[1, :, :] = 255.0
        img[2, :, :] = 128.0
        out = normalize(img)
        # Channels should NOT all be the same after global normalisation
        ch0_mean = out[0].mean()
        ch1_mean = out[1].mean()
        assert ch0_mean < ch1_mean, (
            "Global normalisation must preserve relative channel differences. "
            f"Got ch0={ch0_mean:.3f}, ch1={ch1_mean:.3f} — they should differ."
        )

    @requirement("VELA-FR-003")
    def test_output_dtype_is_float32(self):
        img = np.random.rand(3, 24, 24).astype(np.float32)
        out = normalize(img)
        assert out.dtype == np.float32, "normalize output must be float32"


# ── preprocess (pipeline complet) ────────────────────────────────────────────

class TestPreprocess:
    """
    VELA-FR-003 : full preprocessing pipeline = crop_center + normalize.
    """

    @requirement("VELA-FR-003")
    def test_output_shape(self, synthetic_rgb_image_32x32):
        out = preprocess(synthetic_rgb_image_32x32)
        assert out.shape == (3, 24, 24), (
            f"preprocess output must be (3, 24, 24), got {out.shape}"
        )

    @requirement("VELA-FR-003")
    def test_output_is_normalised(self, synthetic_rgb_image_32x32):
        out = preprocess(synthetic_rgb_image_32x32)
        assert abs(out.mean()) < 0.01
        assert abs(out.std() - 1.0) < 0.1

    @requirement("VELA-FR-003")
    def test_output_dtype(self, synthetic_rgb_image_32x32):
        out = preprocess(synthetic_rgb_image_32x32)
        assert out.dtype == np.float32


# ── load_batch (requires data files) ────────────────────────────────────────

class TestLoadBatch:
    """
    VELA-FR-002 : load CIFAR-10 binary batch files.
    These tests are skipped automatically if test_batch.bin is not available.
    """

    @requirement("VELA-FR-002")
    def test_batch_shape(self, data_dir, data_available):
        if not data_available:
            pytest.skip("CIFAR-10 data not available")
        from data_loader import load_batch
        import os
        imgs, labs = load_batch(os.path.join(data_dir, "test_batch.bin"))
        assert imgs.shape == (10000, 3, 32, 32), (
            f"Expected (10000, 3, 32, 32), got {imgs.shape}"
        )
        assert labs.shape == (10000,)

    @requirement("VELA-FR-002")
    def test_labels_in_valid_range(self, data_dir, data_available):
        if not data_available:
            pytest.skip("CIFAR-10 data not available")
        from data_loader import load_batch
        import os
        _, labs = load_batch(os.path.join(data_dir, "test_batch.bin"))
        assert labs.min() >= 0
        assert labs.max() <= 9
        assert len(set(labs.tolist())) == 10, "All 10 classes must be present"

    @requirement("VELA-FR-002")
    def test_pixel_values_in_valid_range(self, data_dir, data_available):
        if not data_available:
            pytest.skip("CIFAR-10 data not available")
        from data_loader import load_batch
        import os
        imgs, _ = load_batch(os.path.join(data_dir, "test_batch.bin"))
        assert imgs.min() >= 0.0
        assert imgs.max() <= 255.0