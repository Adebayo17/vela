"""
synth_summary.py — Summarize `make synth` results and check VELA-NFR-002 / 004.

Reads build/synth/<module>/{summary.txt, utilization.rpt}, prints a table,
writes build/synth/summary.md and exits with code 1 if a requirement fails.

    NFR-002 : every module WNS >= 0 and WHS >= 0 at 100 MHz
              (WHS between internal registers; port hold WHS_IO is shown for
              information only, see tools/synth.tcl)
    NFR-004 : filters use no DSP and, together, <= 10 % of LUTs and FFs;
              measured on filter_engine when present (it instantiates every
              filter actually used: gaussian and median x 3, VELA-TN-007 §6.2),
              else on the sum of the individual filter modules

Usage:
    python tools/synth_summary.py [build/synth]
"""

import re
import sys
from pathlib import Path

DEVICE = {"lut": 53200, "ff": 106400, "bram": 140, "dsp": 220}   # xc7z020
BUDGET = 0.10                                                     # NFR-004

# Hardware actually instantiated for the filter set: each 3x3 filter has its
# own window3x3 inside, so window3x3 is not counted on its own.
FILTERS = ["grayscale", "threshold", "gaussian_filter", "sobel_filter", "median_filter"]

ROWS = {
    "lut":     r"Slice LUTs\*?",
    "lutram":  r"LUT as Memory",
    "ff":      r"Slice Registers",
    "bram":    r"Block RAM Tile",
    "dsp":     r"DSPs",
}


def parse_utilization(path):
    """Extract the 'Used' column of the main rows of a report_utilization file."""
    text = path.read_text()
    values = {}
    for key, label in ROWS.items():
        m = re.search(rf"^\|\s*{label}\s*\|\s*([\d.]+)\s*\|", text, re.M)
        values[key] = float(m.group(1)) if m else 0.0
    return values


def _num(text):
    """Slack value in ns, or None when Vivado found no such path ("NA")."""
    return None if text == "NA" else float(text)


def _fmt(value):
    return "—" if value is None else f"{value:+.3f}"


def main(synth_dir):
    results = []
    for summary in sorted(Path(synth_dir).glob("*/summary.txt")):
        name, wns, whs, whs_io = summary.read_text().split()
        util = parse_utilization(summary.parent / "utilization.rpt")
        results.append({"name": name, "wns": _num(wns), "whs": _num(whs),
                        "whs_io": _num(whs_io), **util})
    if not results:
        print(f"No results in {synth_dir}")
        return 1

    lines = [
        "WHS = hold slack between internal registers (— : no such path). "
        "WHS I/O includes port-to-register paths, not meaningful out of context.",
        "",
        "| Module | LUT | LUTRAM | FF | BRAM | DSP | WNS (ns) | WHS (ns) | WHS I/O (ns, info) | Fmax est. (MHz) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        fmax = 1000.0 / (10.0 - r["wns"])
        lines.append(f"| {r['name']} | {r['lut']:.0f} | {r['lutram']:.0f} | {r['ff']:.0f} | "
                     f"{r['bram']:g} | {r['dsp']:.0f} | {_fmt(r['wns'])} | {_fmt(r['whs'])} | "
                     f"{_fmt(r['whs_io'])} | {fmax:.0f} |")

    by_name = {r["name"]: r for r in results}
    checks = []

    timing_ok = all(r["wns"] >= 0 and (r["whs"] is None or r["whs"] >= 0) for r in results)
    worst = min(results, key=lambda r: r["wns"])
    checks.append(("VELA-NFR-002", timing_ok,
                   f"worst WNS {worst['wns']:+.3f} ns ({worst['name']})"))

    present = [f for f in FILTERS if f in by_name]
    if "filter_engine" in by_name:
        eng = by_name["filter_engine"]
        res_ok = (eng["dsp"] == 0 and eng["lut"] <= BUDGET * DEVICE["lut"]
                  and eng["ff"] <= BUDGET * DEVICE["ff"])
        checks.append(("VELA-NFR-004", res_ok,
                       f"filter_engine (all filters as instantiated): {eng['lut']:.0f} LUT "
                       f"({100 * eng['lut'] / DEVICE['lut']:.1f} %), {eng['ff']:.0f} FF "
                       f"({100 * eng['ff'] / DEVICE['ff']:.1f} %), {eng['dsp']:.0f} DSP"))
    elif len(present) == len(FILTERS):
        lut = sum(by_name[f]["lut"] for f in FILTERS)
        ff = sum(by_name[f]["ff"] for f in FILTERS)
        dsp = sum(by_name[f]["dsp"] for f in FILTERS)
        res_ok = dsp == 0 and lut <= BUDGET * DEVICE["lut"] and ff <= BUDGET * DEVICE["ff"]
        checks.append(("VELA-NFR-004", res_ok,
                       f"filters: {lut:.0f} LUT ({100 * lut / DEVICE['lut']:.1f} %), "
                       f"{ff:.0f} FF ({100 * ff / DEVICE['ff']:.1f} %), {dsp:.0f} DSP"))
    else:
        checks.append(("VELA-NFR-004", None, "not evaluated: run all filters"))

    lines += ["", "| Requirement | Result | Detail |", "| :--- | :--- | :--- |"]
    for req, ok, detail in checks:
        status = "n/a" if ok is None else ("PASS" if ok else "FAIL")
        lines.append(f"| {req} | {status} | {detail} |")

    report = "\n".join(lines) + "\n"
    print(report)
    (Path(synth_dir) / "summary.md").write_text(report)
    return 0 if all(ok is not False for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build/synth"))
