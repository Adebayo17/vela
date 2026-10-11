"""
hls_sim.py — Build and run the C++ CNN code from pytest (no Vitis needed).

Flow:
    1. generate the weights header (tools/gen_weights_header.py)
    2. compile hls/*.cpp + the batch runner with g++
    3. write the images as raw float64, run the executable, read the results

Used by verif/tests/test_hls_cnn_ref.py.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
HLS_DIR = ROOT_DIR / "hls"
WEIGHTS_FILE = ROOT_DIR / "data" / "CNN_coeff_3x3.txt"
DATA_DIR = ROOT_DIR / "data" / "cifar-10-batches-bin"

GXX = shutil.which("g++")

requires_gxx = pytest.mark.skipif(GXX is None, reason="g++ not installed")
requires_weights = pytest.mark.skipif(not WEIGHTS_FILE.exists(),
                                      reason=f"weights not found: {WEIGHTS_FILE}")

# Shapes of the CnnTrace fields, in struct order (hls/cnn_ref.h)
TRACE_FIELDS = [
    ("conv1", (64, 24, 24)), ("pool1", (64, 12, 12)),
    ("conv2", (32, 12, 12)), ("pool2", (32, 6, 6)),
    ("conv3", (20, 6, 6)), ("pool3", (20, 3, 3)),
    ("vec", (180,)),
]


def build_reference(workdir):
    """Generate the weights header and compile the double reference runner."""
    gen_dir = workdir / "generated"
    subprocess.run([sys.executable, str(ROOT_DIR / "tools" / "gen_weights_header.py"),
                    str(WEIGHTS_FILE), str(gen_dir / "cnn_weights.h")],
                   check=True, capture_output=True, text=True)
    exe = workdir / "cnn_ref_main"
    result = subprocess.run(
        [GXX, "-O2", "-std=c++14", "-Wall", "-Wextra", "-Werror",
         f"-I{HLS_DIR}", f"-I{gen_dir}",
         str(HLS_DIR / "cnn_ref.cpp"), str(HLS_DIR / "tb" / "cnn_ref_main.cpp"),
         "-o", str(exe)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return exe


def run_reference(exe, images, workdir, trace=False):
    """
    Run the C++ reference on preprocessed images.

    Args:
        exe:     executable from build_reference()
        images:  np.ndarray (N, 3, 24, 24), preprocessed (float)
        workdir: pathlib.Path for the exchange files
        trace:   also return the intermediate layers of the first image
    Returns:
        logits (N, 10), probs (N, 10), preds (N,) [, trace dict]
    """
    images = np.ascontiguousarray(images, dtype="<f8")
    n = images.shape[0]
    img_file, out_file, trace_file = workdir / "images.bin", workdir / "out.bin", workdir / "trace.bin"
    images.tofile(img_file)
    args = [str(exe), str(img_file), str(n), str(out_file)] + ([str(trace_file)] if trace else [])
    result = subprocess.run(args, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and "DONE" in result.stdout, result.stdout + result.stderr

    out = np.fromfile(out_file, dtype="<f8").reshape(n, 21)
    logits, probs, preds = out[:, :10], out[:, 10:20], out[:, 20].astype(int)
    if not trace:
        return logits, probs, preds

    flat = np.fromfile(trace_file, dtype="<f8")
    layers, pos = {}, 0
    for name, shape in TRACE_FIELDS:
        size = int(np.prod(shape))
        layers[name] = flat[pos:pos + size].reshape(shape)
        pos += size
    assert pos == flat.size, "CnnTrace layout mismatch"
    return logits, probs, preds, layers
