"""
hls_impl_summary.py — Check the out-of-context implementation of the CNN IP.

Reads <dir>/summary.txt and <dir>/utilization.rpt (tools/hls_impl.tcl) and
checks, after place and route on xc7z020:

    VELA-NFR-009 : <= 50 % of DSPs (110), BRAM 36 Kb tiles (70) and LUTs (26 600)
    VELA-NFR-010 : WNS >= 0 and WHS >= 0 at 100 MHz (WHS between internal
                   registers; port hold is not meaningful out of context)

Prints a table, writes <dir>/summary.md, exits with code 1 if one fails.

Usage:
    python tools/hls_impl_summary.py [build/hls/impl]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_summary import DEVICE, parse_utilization   # noqa: E402

BUDGET = 0.50


def _num(text):
    return None if text == "NA" else float(text)


def main(impl_dir):
    impl_dir = Path(impl_dir)
    _, wns, whs, whs_io = (impl_dir / "summary.txt").read_text().split()
    wns, whs, whs_io = _num(wns), _num(whs), _num(whs_io)
    u = parse_utilization(impl_dir / "utilization.rpt")

    rows = [("LUT", u["lut"], DEVICE["lut"]), ("FF", u["ff"], DEVICE["ff"]),
            ("BRAM 36 Kb tiles", u["bram"], DEVICE["bram"]), ("DSP", u["dsp"], DEVICE["dsp"])]
    lines = ["| Resource | Used | Available | % |", "| :--- | ---: | ---: | ---: |"]
    lines += [f"| {n} | {v:g} | {a} | {100 * v / a:.1f} % |" for n, v, a in rows]
    fmax = 1000.0 / (10.0 - wns)
    lines += ["", f"WNS {wns:+.3f} ns (Fmax est. {fmax:.0f} MHz), "
                  f"WHS {whs:+.3f} ns (internal), WHS I/O {whs_io:+.3f} ns (info)", ""]

    res_ok = all(v <= BUDGET * a for n, v, a in rows if n != "FF")
    timing_ok = wns >= 0 and (whs is None or whs >= 0)
    lines += ["| Requirement | Result |", "| :--- | :--- |",
              f"| VELA-NFR-009 (<= 50 % DSP, BRAM, LUT) | {'PASS' if res_ok else 'FAIL'} |",
              f"| VELA-NFR-010 (100 MHz, WNS/WHS >= 0) | {'PASS' if timing_ok else 'FAIL'} |"]
    report = "\n".join(lines) + "\n"
    print(report)
    (impl_dir / "summary.md").write_text(report)
    return 0 if res_ok and timing_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build/hls/impl"))
