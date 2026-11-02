"""
hls_sim.py — Build and run the C++ CNN code from pytest (no Vitis needed).

Flow:
    1. generate the weights header (tools/gen_weights_header.py)
    2. compile hls/*.cpp + the batch runner with g++
    3. write the images as raw float64, run the executable, read the results

Used by verif/tests/test_hls_cnn_ref.py (double reference) and
verif/tests/test_hls_cnn_hw.py (fixed-point version, compiled with the
ap_fixed headers of Vitis: same code path as a Vitis HLS C-simulation).
"""

import os
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


def find_vitis_include():
    """Directory holding ap_fixed.h: $VITIS_INCLUDE, else next to vitis_hls / v++."""
    candidates = [os.environ.get("VITIS_INCLUDE")]
    for tool in ("vitis_hls", "v++", "vitis-run"):
        exe = shutil.which(tool)
        if exe:
            install = Path(exe).resolve().parents[1]          # .../<tool>/2024.2
            candidates += [install / "include",
                           install.parents[1] / "Vitis" / install.name / "include"]
    for c in candidates:
        if c and (Path(c) / "ap_fixed.h").exists():
            return Path(c)
    return None


VITIS_INCLUDE = find_vitis_include()
requires_ap_fixed = pytest.mark.skipif(VITIS_INCLUDE is None,
                                       reason="ap_fixed.h not found (set VITIS_INCLUDE)")

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


def generate_weights(workdir):
    """Write the weights header into workdir/generated, return that directory."""
    gen_dir = workdir / "generated"
    subprocess.run([sys.executable, str(ROOT_DIR / "tools" / "gen_weights_header.py"),
                    str(WEIGHTS_FILE), str(gen_dir / "cnn_weights.h")],
                   check=True, capture_output=True, text=True)
    return gen_dir


def build_reference(workdir):
    """Generate the weights header and compile the double reference runner."""
    gen_dir = generate_weights(workdir)
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


# Optimization level of the ap_fixed build. -O2 by default; the CI sets -O1:
# with the open-source ap_fixed headers (2019), g++ -O2 needs more than the
# 7 GB of a GitHub runner, -O1 about 1.6 GB. The level does not change the
# arithmetic (no -ffast-math): the bit-exact tests check it.
HLS_OPT = os.environ.get("VELA_HLS_OPT", "-O2")


def build_hw(workdir):
    """Compile the fixed-point CNN runner (g++ + Vitis ap_fixed headers)."""
    gen_dir = generate_weights(workdir)
    exe = workdir / "cnn_hw_main"
    result = subprocess.run(
        [GXX, HLS_OPT, "-std=c++14", "-Wall", "-Wextra", "-Werror",
         "-isystem", str(VITIS_INCLUDE),       # no warnings from the Vitis headers
         # False positive inside the Vitis 2024.2 headers (ap_private.h, member
         # read in clearUnusedBits() after inlining at -O2): -isystem cannot hide
         # it because GCC reports it at the inlining site, in our files.
         "-Wno-uninitialized", "-Wno-maybe-uninitialized",
         # Loop labels name the loops for the HLS directives (step 4);
         # plain C++ never jumps to them.
         "-Wno-unused-label",
         "-Wno-unknown-pragmas",               # #pragma HLS, read by Vitis only
         f"-I{HLS_DIR}", f"-I{gen_dir}",
         str(HLS_DIR / "cnn_hw.cpp"), str(HLS_DIR / "tb" / "cnn_hw_main.cpp"),
         "-o", str(exe)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]
    return exe


def run_hw(exe, images, workdir, trace=False):
    """
    Run the fixed-point CNN on preprocessed images.

    Returns:
        logits (N, 10) as logit_t values, scores (N, 10) exact acc_t values,
        preds (N,) [, dict pool1/pool2/pool3 of the first image]
    """
    images = np.ascontiguousarray(images, dtype="<f8")
    n = images.shape[0]
    img_file, out_file, trace_file = workdir / "images.bin", workdir / "out.bin", workdir / "trace.bin"
    images.tofile(img_file)
    args = [str(exe), str(img_file), str(n), str(out_file)] + ([str(trace_file)] if trace else [])
    result = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0 and "DONE" in result.stdout, result.stdout + result.stderr

    out = np.fromfile(out_file, dtype="<f8").reshape(n, 21)
    logits, scores, preds = out[:, :10], out[:, 10:20], out[:, 20].astype(int)
    if not trace:
        return logits, scores, preds
    flat = np.fromfile(trace_file, dtype="<f8")
    layers, pos = {}, 0
    for name, shape in [("pool1", (64, 12, 12)), ("pool2", (32, 6, 6)), ("pool3", (20, 3, 3))]:
        size = int(np.prod(shape))
        layers[name] = flat[pos:pos + size].reshape(shape)
        pos += size
    assert pos == flat.size, "trace layout mismatch"
    return logits, scores, preds, layers
