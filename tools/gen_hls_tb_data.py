"""
gen_hls_tb_data.py — Stimuli and expected results for the Vitis HLS testbench.

Writes, for the first N CIFAR-10 test images:
    <out>/tb_images.bin    N x 3 x 24 x 24 float64 (preprocessed images, CHW)
    <out>/tb_expected.bin  N x 21 float64: 10 logits (logit_t), 10 exact
                           scores (acc_t), predicted class

The expected values come from the bit-exact model of the fixed-point CNN
(verif/models/cnn_hw_model.py, TN004 formats), not from the C++ code, so
hls/testbench.cpp checks cnn_hw() against an independent reference in both
C-simulation and C/RTL co-simulation.

Usage:
    python tools/gen_hls_tb_data.py [N] [out_dir]      (default: 20, build/hls)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "verif" / "models"))

from cnn_hw_model import emulate_tn004      # noqa: E402
from data_loader import load_dataset        # noqa: E402
from weights_loader import load_weights     # noqa: E402


def main(n=20, out_dir=ROOT / "build" / "hls"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images, _ = load_dataset(str(ROOT / "data" / "cifar-10-batches-bin"), max_images=n)
    model = emulate_tn004(images, load_weights(str(ROOT / "data" / "CNN_coeff_3x3.txt")))
    np.ascontiguousarray(images, dtype="<f8").tofile(out_dir / "tb_images.bin")
    expected = np.concatenate([model["logits"], model["scores"],
                               model["pred"][:, None].astype(np.float64)], axis=1)
    np.ascontiguousarray(expected, dtype="<f8").tofile(out_dir / "tb_expected.bin")
    print(f"Wrote {n} images and expected results to {out_dir}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20,
         sys.argv[2] if len(sys.argv) > 2 else ROOT / "build" / "hls")
