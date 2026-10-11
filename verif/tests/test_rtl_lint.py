"""
test_rtl_lint.py — VELA-NFR-005: every RTL file is plain, lint-clean Verilog-2001

For each file of rtl/:
    - Verilator --lint-only -Wall reports no warning (submodules found in rtl/);
    - Icarus Verilog compiles it in Verilog-2005 mode (-g2005), which rejects
      SystemVerilog syntax such as casts N'(x).
"""

import shutil
import subprocess

import pytest
from vela_testing import requirement
from rtl_sim import IVERILOG, RTL_DIR

VERILATOR = shutil.which("verilator")
RTL_FILES = sorted(p.name for p in RTL_DIR.glob("*.v"))


@requirement("VELA-NFR-005")
@pytest.mark.skipif(VERILATOR is None, reason="Verilator not installed")
@pytest.mark.parametrize("name", RTL_FILES)
def test_verilator_lint_clean(name):
    """Lint Verilator -Wall sans avertissement."""
    result = subprocess.run([VERILATOR, "--lint-only", "-Wall", "--quiet", "-y", str(RTL_DIR),
                             str(RTL_DIR / name)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@requirement("VELA-NFR-005")
@pytest.mark.skipif(IVERILOG is None, reason="Icarus Verilog not installed")
@pytest.mark.parametrize("name", RTL_FILES)
def test_verilog_2005_compatible(name):
    """Compilation iverilog -g2005 : aucune syntaxe SystemVerilog."""
    result = subprocess.run([IVERILOG, "-g2005", "-y", str(RTL_DIR), "-o", "/dev/null",
                             str(RTL_DIR / name)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
