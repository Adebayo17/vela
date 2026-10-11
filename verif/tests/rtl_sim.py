"""
rtl_sim.py — Run Verilog testbenches with Icarus Verilog from pytest.

Flow (one call to run_testbench):
    1. write input pixels to a .hex file   ($readmemh format)
    2. compile tb + DUT with iverilog
    3. run the simulation with vvp (+IN, +OUT, +N, +SEED plusargs)
    4. read output pixels back from the .hex file

Waveforms: with VELA_WAVES=1 in the environment, each simulation also dumps
a VCD trace and keeps a copy of in.hex / out.hex in build/waves/<test name>/,
with a GTKWave save file (wave.gtkw) preloading the testbench signals.
Without it, everything stays in pytest's tmp dir.
"""

import os
import re

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
RTL_DIR = ROOT_DIR / "rtl"
TB_DIR = ROOT_DIR / "verif" / "tb"

WAVES_DIR = ROOT_DIR / "build" / "waves"

IVERILOG = shutil.which("iverilog")
VVP = shutil.which("vvp")

requires_iverilog = pytest.mark.skipif(
    IVERILOG is None or VVP is None,
    reason="Icarus Verilog (iverilog/vvp) not installed",
)


def waves_dir():
    """build/waves/<test name>/ if VELA_WAVES is set, else None."""
    if not os.environ.get("VELA_WAVES"):
        return None
    # PYTEST_CURRENT_TEST = "verif/tests/test_rtl_x.py::TestX::test_y[8-6] (call)"
    current = os.environ.get("PYTEST_CURRENT_TEST", "manual").split(" ")[0]
    module = Path(current.split("::")[0]).stem
    test = current.split("::")[-1]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{module}__{test}").strip("_")
    path = WAVES_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# Testbench bookkeeping variables, hidden from the GTKWave view
TB_INTERNALS = {"in_file", "out_file", "vcd_file", "n", "seed", "fd", "sent", "received", "cycles"}

# Display order in GTKWave: control first, then inputs, then outputs
SIGNAL_ORDER = ["clk", "rst_n", "valid_in", "ready_in", "r_in", "g_in", "b_in", "pixel_in",
                "valid_out"]


def _signal_rank(name):
    if name in SIGNAL_ORDER:
        return SIGNAL_ORDER.index(name)
    return len(SIGNAL_ORDER) + (0 if name.endswith("_in") else 1)


def write_gtkw(vcd_path, tb_name):
    """
    Write a GTKWave save file next to the VCD, preloading the testbench-level
    signals (clk, rst_n, inputs, outputs) so the waveform is readable at once:
        gtkwave build/waves/<test>/wave.gtkw
    1-bit signals in binary, buses in hex. DUT internals stay available in
    GTKWave's tree (tb_xxx > dut).
    """
    signals, depth = [], 0
    with open(vcd_path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "$scope":
                depth += 1
            elif tok[0] == "$upscope":
                depth -= 1
            elif tok[0] == "$var" and depth == 1:
                width, name = int(tok[2]), tok[4]
                rng = tok[5] if len(tok) > 6 and tok[5].startswith("[") else ""
                if name not in TB_INTERNALS and tok[1] not in ("integer", "parameter"):
                    signals.append((_signal_rank(name), width, f"{tb_name}.{name}{rng}"))
            elif tok[0] == "$enddefinitions":
                break

    lines = [f'[dumpfile] "{vcd_path.resolve()}"']   # absolute: gtkwave resolves from cwd
    for _, width, sig in sorted(signals):
        lines += ["@28" if width == 1 else "@22", sig]   # 28 = binary, 22 = hex
    (vcd_path.parent / "wave.gtkw").write_text("\n".join(lines) + "\n")


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
                  out_width=8, defines=None, plusargs=None, tb_sources=None, timeout=60,
                  raw=False):
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
        plusargs:  dict of extra simulation plusargs, e.g. {"HALF": 8} -> +HALF=8
        tb_sources: extra verification files in verif/tb/, e.g. ["spi_monitor.v"]
        timeout:   simulation time limit in seconds
        raw:       return the output words as a list of Python ints (any width)
    Returns:
        np.ndarray uint32 of output words if out_width <= 32,
        else np.ndarray (N, out_width/8) uint8 (bytes MSB first)
    """
    in_words = list(in_words)
    in_file = workdir / "in.hex"
    out_file = workdir / "out.hex"
    sim_bin = workdir / f"{tb_name}.vvp"

    write_hex(in_file, in_words, in_width)

    files = [str(TB_DIR / f"{tb_name}.v")] + [str(TB_DIR / s) for s in (tb_sources or [])]
    files += [str(RTL_DIR / s) for s in sources]
    overrides = [f"-P{tb_name}.{k}={v}" for k, v in (params or {}).items()]
    overrides += [f"-D{k}={v}" for k, v in (defines or {}).items()]
    subprocess.run(
        [IVERILOG, "-g2012", "-Wall", "-s", tb_name, "-o", str(sim_bin)] + overrides + files,
        check=True, capture_output=True, text=True,
    )

    wave_dir = waves_dir()
    vcd_arg = [f"+VCD={wave_dir / 'wave.vcd'}"] if wave_dir else []

    result = subprocess.run(
        [VVP, "-n", str(sim_bin),
         f"+IN={in_file}", f"+OUT={out_file}", f"+N={len(in_words)}", f"+SEED={seed}"]
        + [f"+{k}={v}" for k, v in (plusargs or {}).items()] + vcd_arg,
        check=True, capture_output=True, text=True, timeout=timeout,
    )
    if wave_dir:
        shutil.copy(in_file, wave_dir / "in.hex")
        shutil.copy(out_file, wave_dir / "out.hex")
        write_gtkw(wave_dir / "wave.vcd", tb_name)
    assert "ERROR" not in result.stdout, result.stdout
    assert "DONE" in result.stdout, result.stdout

    words = read_hex(out_file)
    if raw:
        return words
    if out_width <= 32:
        return np.array(words, dtype=np.uint32)
    return unpack_bytes(words, out_width // 8)
