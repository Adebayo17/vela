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

The batched model (verif/models/cnn_hw_model.py) is checked against
python/cnn.py before use.

Usage:
    python tools/cnn_fixed_point_analysis.py [--images N] [--report FILE]
                                             [--configs W_WEIGHTS:W_ACT:MODE,...]
    Without --configs, sweeps 8..16 bits x 8..16 bits x {trn, rnd} (slow);
    e.g. --configs 16:16:rnd,12:12:rnd evaluates only these.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "verif" / "models"))

from cnn import cnn_forward                    # noqa: E402
from data_loader import load_dataset           # noqa: E402
from weights_loader import load_weights        # noqa: E402
from cnn_hw_model import (CHUNK, forward, int_bits, quantize,   # noqa: E402
                          run_batched)

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
        out.append(f"| {ww} | {wa} | {mode} | {100 * agree:.2f} % | {100 * acc:.2f} % |")
        print(f"weights {ww:2d} bits, activations {wa:2d} bits, {mode}: "
              f"agreement {100 * agree:6.2f} %, accuracy {100 * acc:6.2f} %", flush=True)

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
