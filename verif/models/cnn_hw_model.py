"""
cnn_hw_model.py — Bit-exact model of the fixed-point HLS CNN (hls/cnn_hw.cpp).

Batched NumPy version of the CNN (same computation as python/cnn.py) with
optional quantization of the activations, used for two purposes:

- tools/cnn_fixed_point_analysis.py: value ranges and precision sweep that
  chose the ap_fixed types (VELA-TN-004);
- verif/tests/test_hls_cnn_hw.py: reference of hls/cnn_hw.cpp, which must
  match emulate_tn004() bit for bit.

Why bit-exact comparison is possible: products of 16-bit values and their
sums (< 2^11 in magnitude, resolution 2^-24) need fewer than 53 bits, so
float64 computes them exactly, like the exact ap_fixed<40, 14> accumulator;
and rounding / saturation are monotonic, so quantizing a conv output before
or after ReLU + maxpool gives the same value.

The formats below are those of VELA-TN-004 §5 (written from the spec, not
read from the C++).
"""

import math

import numpy as np

# REQ: VELA-NFR-006

CHUNK = 100   # images per batch (memory of the conv2 patches)

# (width, integer bits incl. sign) of each tensor — VELA-TN-004 §5
TN004_FORMATS = {
    "weight": (16, 3),
    "input": (16, 5),
    "pool1": (16, 6),
    "pool2": (16, 7),
    "pool3": (16, 7),
    "logit": (16, 7),
}


# ─── Layers (batched, float64) ───────────────────────────────────────────────

def conv3x3_same(x, w, b):
    """x (N, Cin, H, W), w (Cout, Cin, 3, 3), b (Cout,) -> (N, Cout, H, W)."""
    n, cin, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    # im2col: (N, H, W, Cin, 3, 3) -> (N*H*W, Cin*9), then one BLAS matrix product
    patches = np.stack([xp[:, :, m:m + h, k:k + wd] for m in range(3) for k in range(3)], axis=-1)
    cols = patches.transpose(0, 2, 3, 1, 4).reshape(n * h * wd, cin * 9)
    out = cols @ w.reshape(w.shape[0], cin * 9).T + b
    return out.reshape(n, h, wd, -1).transpose(0, 3, 1, 2)


def conv3x3_abs_bound(x, w, b):
    """Largest possible partial sum of each accumulator: sum |x w| + |b|."""
    return conv3x3_same(np.abs(x), np.abs(w), np.abs(b))


def maxpool_same(x):
    """3x3 / stride 2 TensorFlow SAME maxpool: padding (-inf) at bottom/right."""
    n, c, h, w = x.shape
    ho, wo = math.ceil(h / 2), math.ceil(w / 2)
    ph, pw = (ho - 1) * 2 + 3 - h, (wo - 1) * 2 + 3 - w
    xp = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), constant_values=-np.inf)
    out = np.full((n, c, ho, wo), -np.inf)
    for u in range(3):
        for v in range(3):
            out = np.maximum(out, xp[:, :, u:u + 2 * ho:2, v:v + 2 * wo:2])
    return out


def forward(images, w, quant_act=None, record=None):
    """
    Batched forward pass. quant_act(name, tensor) quantizes the activations
    stored between layers ("input", "pool1".."pool3"; None = float).
    record(name, tensor) collects values (also accumulator bounds "acc*").
    Returns the exact logits (N, 10), before any output quantization.
    """
    q = quant_act or (lambda name, t: t)
    rec = record
    x = q("input", images)
    if rec:
        rec("input", x)
    for i, (wk, bk) in enumerate([("conv1_w", "conv1_b"), ("conv2_w", "conv2_b"),
                                  ("conv3_w", "conv3_b")], start=1):
        if rec:   # accumulator bound only when recording (doubles the conv cost)
            rec(f"acc{i}", conv3x3_abs_bound(x, w[wk], w[bk]))
        c = conv3x3_same(x, w[wk], w[bk])
        x = q(f"pool{i}", maxpool_same(np.maximum(c, 0.0)))
        if rec:
            rec(f"conv{i}", c)
            rec(f"pool{i}", x)
    vec = x.transpose(0, 2, 3, 1).reshape(x.shape[0], -1)      # HWC flatten
    logits = vec @ w["fc_w"].T + w["fc_b"]
    if rec:
        rec("acc_fc", np.abs(vec) @ np.abs(w["fc_w"]).T + np.abs(w["fc_b"]))
        rec("logits", logits)
    return logits


def run_batched(images, w, **kw):
    return np.concatenate([forward(images[i:i + CHUNK], w, **kw)
                           for i in range(0, len(images), CHUNK)])


# ─── Fixed point ─────────────────────────────────────────────────────────────

def int_bits(max_abs):
    """ap_fixed integer bits I (sign included) so that |v| <= max_abs fits."""
    return max(1, math.floor(math.log2(max_abs)) + 2) if max_abs > 0 else 1


def quantize(t, width, ibits, mode):
    """
    Emulate ap_fixed<width, ibits, mode, AP_SAT>.
    mode 'trn': AP_TRN (toward -inf); 'rnd': AP_RND (round half up).
    """
    step = 2.0 ** (ibits - width)
    scaled = t / step
    scaled = np.floor(scaled) if mode == "trn" else np.floor(scaled + 0.5)
    lo, hi = -(2 ** (width - 1)), 2 ** (width - 1) - 1
    return np.clip(scaled, lo, hi) * step


def emulate_tn004(images, weights):
    """
    Bit-exact model of hls/cnn_hw.cpp (TN004 formats, AP_RND + AP_SAT).

    Args:
        images:  np.ndarray (N, 3, 24, 24), preprocessed (float)
        weights: dict from python/weights_loader.load_weights()
    Returns:
        dict with
            scores (N, 10): exact FC outputs (acc_t)
            logits (N, 10): scores quantized to logit_t
            pred   (N,):    argmax of the scores (first maximum)
            pool1, pool2, pool3: quantized activations of the first image
    """
    ww, wi = TN004_FORMATS["weight"]
    wq = {k: quantize(v.astype(np.float64), ww, wi, "rnd") for k, v in weights.items()}

    def quant_act(name, t):
        width, ibits = TN004_FORMATS[name]
        return quantize(t, width, ibits, "rnd")

    first = {}

    def record(name, t):
        if name.startswith("pool") and name not in first:
            first[name] = t[0].copy()

    images = np.asarray(images, dtype=np.float64)
    scores = run_batched(images[:1], wq, quant_act=quant_act, record=record)
    if len(images) > 1:
        scores = np.concatenate([scores, run_batched(images[1:], wq, quant_act=quant_act)])
    lw, li = TN004_FORMATS["logit"]
    return {"scores": scores, "logits": quantize(scores, lw, li, "rnd"),
            "pred": scores.argmax(axis=1), **first}


# ─── PS side: what the software writes into the IP (VELA-TN-005) ────────────

def input_raw(images):
    """
    Raw 16-bit codes of input_t (ap_fixed<16, 5, AP_RND, AP_SAT>) for
    preprocessed images: round(x * 2^11) with ties rounded up, saturated to
    [-32768, 32767]. raw / 2^11 equals quantize(x, 16, 5, "rnd").
    """
    width, ibits = TN004_FORMATS["input"]
    scaled = np.floor(np.asarray(images, dtype=np.float64) * 2.0 ** (width - ibits) + 0.5)
    return np.clip(scaled, -(2 ** (width - 1)), 2 ** (width - 1) - 1).astype(np.int16)


def pack_image_words(image):
    """
    32-bit words of the AXI4-Lite image window (offset 0x1000) for one image
    (3, 24, 24): CHW order, two values per word, element 2n in bits 15:0 and
    element 2n+1 in bits 31:16 (Vitis HLS register map).
    """
    raw = input_raw(image).reshape(-1).view(np.uint16).astype(np.uint32)
    return raw[0::2] | (raw[1::2] << 16)


def unpack_logit_words(words):
    """10 logit_t values from the 5 words of the logits window (offset 0x20)."""
    words = np.asarray(words, dtype=np.uint32)
    halves = np.stack([words & 0xFFFF, words >> 16], axis=1).reshape(-1).astype(np.uint16)
    width, ibits = TN004_FORMATS["logit"]
    return halves.view(np.int16).astype(np.float64) / 2.0 ** (width - ibits)
