"""
rtl_sim.py — Run Verilog testbenches with Icarus Verilog from pytest.

Flow (one call to run_testbench):
    1. write input pixels to a .hex file   ($readmemh format)
    2. compile tb + DUT with iverilog
    3. run the simulation with vvp (+IN, +OUT, +N, +SEED plusargs)
    4. read output pixels back from the .hex file
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
RTL_DIR = ROOT_DIR / "rtl"
TB_DIR = ROOT_DIR / "verif" / "tb"

IVERILOG = shutil.which("iverilog")
VVP = shutil.which("vvp")

requires_iverilog = pytest.mark.skipif(
    IVERILOG is None or VVP is None,
    reason="Icarus Verilog (iverilog/vvp) not installed",
)


def write_hex(path, words, width_bits):
    """Write one hex word per line, zero-padded to width_bits."""
    digits = (width_bits + 3) // 4
    with open(path, "w") as f:
        for w in words:
            f.write(f"{int(w):0{digits}x}\n")


def read_hex(path):
    """Read one hex word per line, as Python ints (any width)."""
    with open(path) as f:
        return [int(line, 16) for line in f if line.strip()]


def unpack_bytes(words, n_bytes):
    """Split each word into n_bytes bytes, MSB first -> np.ndarray (len(words), n_bytes) uint8."""
    return np.array([[(w >> (8 * (n_bytes - 1 - i))) & 0xFF for i in range(n_bytes)]
                     for w in words], dtype=np.uint8)


def run_testbench(tb_name, sources, in_words, in_width, workdir, seed=0, params=None,
                  out_width=8, defines=None):
    """
    Compile and simulate a file-driven testbench.

    Args:
        tb_name:   top module name, e.g. "tb_grayscale" (file verif/tb/<tb_name>.v)
        sources:   list of DUT file names in rtl/, e.g. ["grayscale.v"]
        in_words:  iterable of input words
        in_width:  input word width in bits
        workdir:   pathlib.Path for temporary files (pytest tmp_path)
        seed:      random seed for valid_in gaps (0 = continuous stream)
        params:    dict of testbench parameter overrides, e.g. {"THRESHOLD": 100}
        out_width: output word width in bits
        defines:   dict of preprocessor macros, e.g. {"DUT": "gaussian_filter"}
    Returns:
        np.ndarray uint32 of output words if out_width <= 32,
        else np.ndarray (N, out_width/8) uint8 (bytes MSB first)
    """
    in_words = list(in_words)
    in_file = workdir / "in.hex"
    out_file = workdir / "out.hex"
    sim_bin = workdir / f"{tb_name}.vvp"

    write_hex(in_file, in_words, in_width)

    files = [str(TB_DIR / f"{tb_name}.v")] + [str(RTL_DIR / s) for s in sources]
    overrides = [f"-P{tb_name}.{k}={v}" for k, v in (params or {}).items()]
    overrides += [f"-D{k}={v}" for k, v in (defines or {}).items()]
    subprocess.run(
        [IVERILOG, "-g2012", "-Wall", "-s", tb_name, "-o", str(sim_bin)] + overrides + files,
        check=True, capture_output=True, text=True,
    )

    result = subprocess.run(
        [VVP, "-n", str(sim_bin),
         f"+IN={in_file}", f"+OUT={out_file}", f"+N={len(in_words)}", f"+SEED={seed}"],
        check=True, capture_output=True, text=True, timeout=60,
    )
    assert "ERROR" not in result.stdout, result.stdout
    assert "DONE" in result.stdout, result.stdout

    words = read_hex(out_file)
    if out_width <= 32:
        return np.array(words, dtype=np.uint32)
    return unpack_bytes(words, out_width // 8)
