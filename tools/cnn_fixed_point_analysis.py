"""
cnn_fixed_point_analysis.py — Value ranges and fixed-point sweep of the CNN

Phase 2b, step 2: size the ap_fixed types of the HLS CNN.

1. Ranges: runs the float model on the test images and records, for every
   tensor (input, weights, biases, conv outputs, pooled activations, logits),
   its min / max, and for every accumulator the largest partial sum
   (sum of |input x weight| + |bias|, a bound on any intermediate value).
2. Sweep: emulates the fixed-point CNN (weights and activations quantized,
   exact wide accumulators, as with a large ap_fixed accumulator) for several
   bit widths and two quantization modes (truncation = AP_TRN, the ap_fixed
   default; rounding = AP_RND), with saturation, and measures the agreement
   with the float model and the accuracy.

The batched model is checked against python/cnn.py before use.

Usage:
    python tools/cnn_fixed_point_analysis.py [--images N] [--report FILE]
                                             [--configs W_WEIGHTS:W_ACT:MODE,...]
    Without --configs, sweeps 8..16 bits x 8..16 bits x {trn, rnd} (slow);
    e.g. --configs 16:16:rnd,12:12:rnd evaluates only these.
"""

import argparse

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from cnn import cnn_forward                    # noqa: E402
from data_loader import load_dataset           # noqa: E402
from weights_loader import load_weights        # noqa: E402

CHUNK = 100   # images per batch (memory of the conv2 patches)


# ─── Batched float model (same computation as python/cnn.py) ─────────────────

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
    stored between layers (None = float). record(name, tensor) collects values.
    Returns the logits (N, 10).
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


# ─── Fixed-point helpers ─────────────────────────────────────────────────────

def int_bits(max_abs):
    """ap_fixed integer bits I (sign included) so that |v| <= max_abs fits."""
    return max(1, math.floor(math.log2(max_abs)) + 2) if max_abs > 0 else 1


def quantize(t, width, ibits, mode):
    """Emulate ap_fixed<width, ibits, mode, AP_SAT>: mode 'trn' (floor) or 'rnd'."""
    step = 2.0 ** (ibits - width)
    scaled = t / step
    scaled = np.floor(scaled) if mode == "trn" else np.floor(scaled + 0.5)
    lo, hi = -(2 ** (width - 1)), 2 ** (width - 1) - 1
    return np.clip(scaled, lo, hi) * step


# ─── Analysis ────────────────────────────────────────────────────────────────

def main(n_images=1000, report_path=ROOT / "build" / "cnn_fixed_point.md", configs=None):
    t0 = time.time()
    w = {k: v.astype(np.float64) for k, v in load_weights(str(ROOT / "data" / "CNN_coeff_3x3.txt")).items()}
    images, labels = load_dataset(str(ROOT / "data" / "cifar-10-batches-bin"), max_images=n_images)
    images = images.astype(np.float64)
    out = [f"# CNN fixed-point analysis ({n_images} CIFAR-10 test images)", ""]

    # 0. Check the batched model against the golden model
    logits_f = run_batched(images, w)
    pred_f = logits_f.argmax(axis=1)
    pred_golden = np.array([cnn_forward(img, WEIGHTS32)[1]
                            for img in images])
    same = int((pred_f == pred_golden).sum())
    acc_f = float((pred_f == labels).mean())
    out += [f"Batched float model vs python/cnn.py: {same}/{n_images} identical predictions, "
            f"accuracy {100 * acc_f:.1f} %.", ""]
    assert same == n_images, "batched model differs from the golden model"

    # 1. Ranges
    stats = {}

    def record(name, t):
        s = stats.setdefault(name, [np.inf, -np.inf])
        s[0], s[1] = min(s[0], float(t.min())), max(s[1], float(t.max()))

    for i in range(0, n_images, CHUNK):
        forward(images[i:i + CHUNK], w, record=record)

    out += ["## Value ranges", "",
            "| Tensor | min | max | max \\|v\\| | integer bits I (sign incl.) |",
            "| :--- | ---: | ---: | ---: | ---: |"]
    for name, arr in w.items():
        stats[name] = [float(arr.min()), float(arr.max())]
    order = ["input", "conv1_w", "conv1_b", "acc1", "conv1", "pool1",
             "conv2_w", "conv2_b", "acc2", "conv2", "pool2",
             "conv3_w", "conv3_b", "acc3", "conv3", "pool3",
             "fc_w", "fc_b", "acc_fc", "logits"]
    ibits = {}
    for name in order:
        lo, hi = stats[name]
        m = max(abs(lo), abs(hi))
        ibits[name] = int_bits(m)
        out.append(f"| {name} | {lo:.4g} | {hi:.4g} | {m:.4g} | {ibits[name]} |")
    out += ["", "acc* = bound on any partial sum of the accumulator (sum |x w| + |b|).", ""]

    # 2. Sweep: same width for all activations, per-tensor integer bits from the ranges
    act_ibits = {"input": ibits["input"], "pool1": ibits["pool1"],
                 "pool2": ibits["pool2"], "pool3": ibits["pool3"]}
    weight_ibits = max(ibits[k] for k in w)          # one weight format for all layers
    out += ["## Fixed-point sweep", "",
            f"Activation integer bits: {act_ibits}; weight integer bits: {weight_ibits} "
            "(all weights and biases share one format). Accumulators exact (wide).", "",
            "| weights W | activations W | mode | agreement with float | accuracy |",
            "| ---: | ---: | :--- | ---: | ---: |"]
    results = []
    if configs is None:
        configs = [(ww, wa, mode) for ww in (8, 10, 12, 14, 16)
                   for wa in (8, 10, 12, 14, 16) for mode in ("trn", "rnd")]
    for ww, wa, mode in configs:
        wq = {k: quantize(v, ww, weight_ibits, mode) for k, v in w.items()}
        qa = lambda name, t, wa=wa, mode=mode: quantize(t, wa, act_ibits[name], mode)
        pred_q = run_batched(images, wq, quant_act=qa).argmax(axis=1)
        agree = float((pred_q == pred_f).mean())
        acc = float((pred_q == labels).mean())
        results.append((ww, wa, mode, agree, acc))
        out.append(f"| {ww} | {wa} | {mode} | {100 * agree:.1f} % | {100 * acc:.1f} % |")
        print(f"weights {ww:2d} bits, activations {wa:2d} bits, {mode}: "
              f"agreement {100 * agree:5.1f} %, accuracy {100 * acc:5.1f} %", flush=True)

    out += ["", f"Run time: {time.time() - t0:.0f} s"]
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(out) + "\n")
    print("\n".join(out[:out.index("## Fixed-point sweep")]))
    print(f"Report: {report_path}")
    return stats, ibits, results


WEIGHTS32 = load_weights(str(ROOT / "data" / "CNN_coeff_3x3.txt"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--images", type=int, default=1000)
    ap.add_argument("--report", default=str(ROOT / "build" / "cnn_fixed_point.md"))
    ap.add_argument("--configs", default=None,
                    help="comma-separated W_WEIGHTS:W_ACT:MODE, e.g. 16:16:rnd,12:12:rnd")
    args = ap.parse_args()
    cfg = None
    if args.configs:
        cfg = [(int(a), int(b), m) for a, b, m in (c.split(":") for c in args.configs.split(","))]
    main(args.images, args.report, cfg)
