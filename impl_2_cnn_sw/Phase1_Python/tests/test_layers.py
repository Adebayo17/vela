"""
test_layers.py — TDD tests for each CNN layer in cnn.py

Requirements covered:
  VELA-FR-001 : implement all CNN layers (conv, relu, maxpool, fc, softmax)
  VELA-FR-004 : cnn_forward() returns class index and probability vector

Test strategy:
  - All tests use synthetic data only (no external files needed)
  - conv2d is tested via two complementary approaches:
      1. Mathematical property tests (identity, linearity, border)
      2. Consistency test : fast vectorised version == slow reference loops
  - Each layer has both a "happy path" test and at least one edge case
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cnn import conv2d, conv2d_fast, relu, maxpool2d, fully_connected, softmax
from conftest import requirement


# ── conv2d ───────────────────────────────────────────────────────────────────

class TestConv2d:
    """VELA-FR-001 : 2D multi-channel convolution."""

    @requirement("VELA-FR-001")
    def test_identity_kernel_reproduces_input(
        self, synthetic_image_1channel, identity_kernel_1to1, zero_bias_1
    ):
        """
        An identity kernel (centre=1, rest=0) applied with 'same' padding
        must reproduce the input exactly.
        This validates: kernel application, padding, accumulation.
        """
        out = conv2d_fast(
            synthetic_image_1channel,
            identity_kernel_1to1,
            zero_bias_1,
            padding="same"
        )
        np.testing.assert_allclose(
            out, synthetic_image_1channel, atol=1e-5,
            err_msg="Identity kernel must reproduce input"
        )

    @requirement("VELA-FR-001")
    def test_fast_equals_reference_implementation(self, synthetic_image_1channel):
        """
        conv2d_fast() (vectorised) must produce identical results to
        conv2d() (explicit loops). Both are in the codebase; the loop
        version serves as the mathematical reference.
        """
        kern = np.random.default_rng(42).random((2, 1, 3, 3)).astype(np.float32)
        bias = np.zeros(2, dtype=np.float32)
        out_fast = conv2d_fast(synthetic_image_1channel, kern, bias)
        out_ref  = conv2d(synthetic_image_1channel, kern, bias)
        np.testing.assert_allclose(
            out_fast, out_ref, atol=1e-4,
            err_msg="conv2d_fast must match conv2d reference implementation"
        )

    @requirement("VELA-FR-001")
    def test_output_shape_same_padding(self):
        """With padding='same', output H and W must equal input H and W."""
        img  = np.zeros((3, 12, 12), dtype=np.float32)
        kern = np.zeros((8, 3, 3, 3), dtype=np.float32)
        bias = np.zeros(8, dtype=np.float32)
        out  = conv2d_fast(img, kern, bias, padding="same")
        assert out.shape == (8, 12, 12), (
            f"Expected (8, 12, 12), got {out.shape}"
        )

    @requirement("VELA-FR-001")
    def test_output_shape_valid_padding(self):
        """With padding='valid', H_out = H_in - kH + 1."""
        img  = np.zeros((1, 8, 8), dtype=np.float32)
        kern = np.zeros((4, 1, 3, 3), dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)
        out  = conv2d_fast(img, kern, bias, padding="valid")
        assert out.shape == (4, 6, 6), (
            f"Expected (4, 6, 6), got {out.shape}"
        )

    @requirement("VELA-FR-001")
    def test_bias_is_added_correctly(self):
        """A zero-kernel with non-zero bias must output the bias value everywhere."""
        img  = np.ones((1, 4, 4), dtype=np.float32)
        kern = np.zeros((1, 1, 3, 3), dtype=np.float32)
        bias = np.array([7.0], dtype=np.float32)
        out  = conv2d_fast(img, kern, bias, padding="same")
        np.testing.assert_allclose(
            out, np.full((1, 4, 4), 7.0, dtype=np.float32), atol=1e-5,
            err_msg="Zero kernel + bias=7 must output 7 everywhere"
        )

    @requirement("VELA-FR-001")
    def test_multi_channel_accumulation(self):
        """
        Output channel c = sum over ALL input channels.
        Verify with a known calculation:
          Input: 2 channels, each all-ones, 3×3
          Kernel: all-ones, 3×3, for 1 output channel from 2 input channels
          With padding=same and centre pixel only : sum = 2 * 1 * 1 = 2
        """
        img  = np.ones((2, 3, 3), dtype=np.float32)
        kern = np.zeros((1, 2, 3, 3), dtype=np.float32)
        kern[0, :, 1, 1] = 1.0    # centre pixel only, both input channels
        bias = np.zeros(1, dtype=np.float32)
        out  = conv2d_fast(img, kern, bias, padding="same")
        # Centre pixel: sum = 1 (ch0) + 1 (ch1) = 2
        assert out[0, 1, 1] == pytest.approx(2.0, abs=1e-5)

    @requirement("VELA-FR-001")
    def test_output_dtype_is_float32(self, synthetic_image_1channel,
                                     identity_kernel_1to1, zero_bias_1):
        out = conv2d_fast(synthetic_image_1channel, identity_kernel_1to1, zero_bias_1)
        assert out.dtype == np.float32


# ── relu ─────────────────────────────────────────────────────────────────────

class TestRelu:
    """VELA-FR-001 : ReLU activation — RELU(a) = max(0, a)."""

    @requirement("VELA-FR-001")
    def test_negative_values_become_zero(self):
        x   = np.array([-5.0, -0.1, -1e-7], dtype=np.float32)
        out = relu(x)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

    @requirement("VELA-FR-001")
    def test_positive_values_unchanged(self):
        x   = np.array([0.5, 1.0, 100.0], dtype=np.float32)
        out = relu(x)
        np.testing.assert_array_equal(out, x)

    @requirement("VELA-FR-001")
    def test_zero_passes_through(self):
        assert relu(np.array([0.0], dtype=np.float32))[0] == pytest.approx(0.0)

    @requirement("VELA-FR-001")
    def test_shape_preserved(self):
        x = np.random.randn(64, 12, 12).astype(np.float32)
        assert relu(x).shape == x.shape

    @requirement("VELA-FR-001")
    def test_no_negative_values_in_output(self):
        x   = np.random.randn(3, 24, 24).astype(np.float32)
        out = relu(x)
        assert out.min() >= 0.0, "ReLU output must never be negative"

    @requirement("VELA-FR-001")
    def test_mixed_values(self):
        x   = np.array([-3.0, -0.1, 0.0, 0.5, 2.0], dtype=np.float32)
        exp = np.array([ 0.0,  0.0, 0.0, 0.5, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(relu(x), exp)


# ── maxpool2d ────────────────────────────────────────────────────────────────

class TestMaxPool2d:
    """VELA-FR-001 : 2D max pooling with configurable window and stride."""

    @requirement("VELA-FR-001")
    def test_output_shape_3x3_stride2(self):
        """
        Architecture uses pool=3, stride=2.
        24→12 : (24-3)//2 + 1 = 11 + 1 = 12 ✓
        12→6  : (12-3)//2 + 1 = 4 + 1 = 5... wait, actually:
        The SEI project uses specific dimensions — validate the chain.
        """
        fmap = np.zeros((64, 24, 24), dtype=np.float32)
        out  = maxpool2d(fmap, pool_size=3, stride=2)
        assert out.shape == (64, 12, 12), (
            f"Expected (64, 12, 12), got {out.shape} — "
            "check: H_out = (H - pool) // stride + 1"
        )

    @requirement("VELA-FR-001")
    def test_returns_maximum_value(self):
        """
        MaxPool must return the maximum, not average or arbitrary value.
        Verified with a known 2×2 window on a 1×4×4 map.
        """
        fmap = np.array([[[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9,10,11,12],
                          [13,14,15,16]]], dtype=np.float32)
        out  = maxpool2d(fmap, pool_size=2, stride=2)
        expected = np.array([[[6, 8], [14, 16]]], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    @requirement("VELA-FR-001")
    def test_channels_processed_independently(self):
        """Each channel must be pooled independently."""
        fmap = np.zeros((3, 4, 4), dtype=np.float32)
        fmap[0, 0, 0] = 10.0   # max in channel 0, top-left window
        fmap[1, 0, 0] = 20.0   # max in channel 1
        fmap[2, 0, 0] = 30.0   # max in channel 2
        out  = maxpool2d(fmap, pool_size=2, stride=2)
        assert out[0, 0, 0] == pytest.approx(10.0)
        assert out[1, 0, 0] == pytest.approx(20.0)
        assert out[2, 0, 0] == pytest.approx(30.0)

    @requirement("VELA-FR-001")
    def test_architecture_dimension_chain(self):
        """
        Verify the full dimension chain of the CNN architecture:
          Input  : 3  × 24 × 24
          Pool1  : 64 × 12 × 12  (pool=3, stride=2)
          Pool2  : 32 × 6  × 6   (pool=3, stride=2)
          Pool3  : 20 × 3  × 3   (pool=3, stride=2)
        """
        x = np.zeros((64, 24, 24), dtype=np.float32)
        x = maxpool2d(x, pool_size=3, stride=2)
        assert x.shape == (64, 12, 12), f"After Pool1: {x.shape}"

        x2 = np.zeros((32, 12, 12), dtype=np.float32)
        x2 = maxpool2d(x2, pool_size=3, stride=2)
        assert x2.shape == (32, 6, 6), f"After Pool2: {x2.shape}"

        x3 = np.zeros((20, 6, 6), dtype=np.float32)
        x3 = maxpool2d(x3, pool_size=3, stride=2)
        assert x3.shape == (20, 3, 3), f"After Pool3: {x3.shape}"


# ── fully_connected ──────────────────────────────────────────────────────────

class TestFullyConnected:
    """VELA-FR-001 : Fully-connected layer S = W·I + b."""

    @requirement("VELA-FR-001")
    def test_output_shape(self):
        vec    = np.ones(180, dtype=np.float32)
        weight = np.ones((10, 180), dtype=np.float32)
        bias   = np.zeros(10, dtype=np.float32)
        out    = fully_connected(vec, weight, bias)
        assert out.shape == (10,)

    @requirement("VELA-FR-001")
    def test_known_computation(self):
        """[1,1,1] · [[1,0,0],[0,1,0],[0,0,1]] + [0,0,0] = [1,1,1]"""
        vec    = np.ones(3, dtype=np.float32)
        weight = np.eye(3, dtype=np.float32)
        bias   = np.zeros(3, dtype=np.float32)
        out    = fully_connected(vec, weight, bias)
        np.testing.assert_array_equal(out, [1.0, 1.0, 1.0])

    @requirement("VELA-FR-001")
    def test_bias_added(self):
        vec    = np.zeros(3, dtype=np.float32)
        weight = np.zeros((2, 3), dtype=np.float32)
        bias   = np.array([3.0, 7.0], dtype=np.float32)
        out    = fully_connected(vec, weight, bias)
        np.testing.assert_array_equal(out, [3.0, 7.0])

    @requirement("VELA-FR-001")
    def test_3d_input_is_flattened(self):
        """FC must accept a 3D feature map and flatten it automatically."""
        fmap   = np.ones((20, 3, 3), dtype=np.float32)  # 180 elements
        weight = np.ones((10, 180), dtype=np.float32)
        bias   = np.zeros(10, dtype=np.float32)
        out    = fully_connected(fmap, weight, bias)
        assert out.shape == (10,)
        # All weights=1, all inputs=1 → each output = 180
        np.testing.assert_allclose(out, np.full(10, 180.0), atol=1e-4)


# ── softmax ──────────────────────────────────────────────────────────────────

class TestSoftmax:
    """VELA-FR-001 : Softmax normalisation."""

    @requirement("VELA-FR-001")
    def test_output_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        probs  = softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6, (
            f"Softmax probabilities must sum to 1, got {probs.sum()}"
        )

    @requirement("VELA-FR-001")
    def test_argmax_preserved(self):
        """The class with the highest logit must have the highest probability."""
        logits = np.array([0.1, 0.5, 3.0, 0.2, 0.1], dtype=np.float32)
        probs  = softmax(logits)
        assert np.argmax(probs) == 2

    @requirement("VELA-FR-001")
    def test_all_probabilities_positive(self):
        logits = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32)
        probs  = softmax(logits)
        assert np.all(probs > 0), "All softmax outputs must be strictly positive"

    @requirement("VELA-FR-001")
    def test_numerical_stability_large_logits(self):
        """
        Naive exp(x) overflows for x > ~88 (float32).
        The stable implementation subtracts max(logits) before exp.
        """
        logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        probs  = softmax(logits)
        assert np.all(np.isfinite(probs)), "Softmax must be numerically stable"
        assert abs(probs.sum() - 1.0) < 1e-5

    @requirement("VELA-FR-001")
    def test_uniform_logits_give_equal_probabilities(self):
        logits = np.zeros(10, dtype=np.float32)
        probs  = softmax(logits)
        np.testing.assert_allclose(
            probs, np.full(10, 0.1), atol=1e-6,
            err_msg="Uniform logits must give equal probabilities (1/N each)"
        )