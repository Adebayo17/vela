#!/usr/bin/env python3
"""
tools/traceability_scanner.py
─────────────────────────────
Scans the codebase to automatically build and verify the traceability matrix.

For each requirement ID found in the SRS document, the scanner checks whether:
  1. At least one test references it via @requirement("VELA-FR-XXX")
  2. At least one source file references it via # REQ: VELA-FR-XXX

Outputs a Markdown traceability report and returns exit code 1 if any
requirement is uncovered (used by CI to fail the build).

Usage:
    python tools/traceability_scanner.py \\
        --srs docs/SRS_VELA.md \\
        --tests impl_2_cnn_sw/Phase1_Python/tests/ \\
        --source impl_2_cnn_sw/Phase1_Python/ \\
        --output traceability_report.md \\
        --fail-on-uncovered

    # Preview without failing CI:
    python tools/traceability_scanner.py --srs docs/SRS_VELA.md --tests tests/
"""

import argparse
import ast
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ── Patterns ─────────────────────────────────────────────────────────────────

# Matches requirement IDs in SRS : "| VELA-FR-001 |" or "VELA-NFR-001"
SRS_REQ_PATTERN = re.compile(r'\b(VELA-(?:FR|NFR|T)-\d{3})\b')

# Matches inline code comment : # REQ: VELA-FR-001
CODE_REQ_PATTERN = re.compile(r'#\s*REQ:\s*(VELA-(?:FR|NFR|T)-\d{3})')

# Matches decorator argument string : @requirement("VELA-FR-001")
DECORATOR_STR_PATTERN = re.compile(r'@requirement\([^)]*"(VELA-(?:FR|NFR|T)-\d{3})"')


# ── Extraction functions ──────────────────────────────────────────────────────

def extract_requirements_from_srs(srs_path: str) -> set[str]:
    """
    Extract all requirement IDs defined in the SRS document.
    Only extracts FR-XXX and NFR-XXX (not T-XXX test IDs).
    """
    req_ids = set()
    with open(srs_path, encoding="utf-8") as f:
        for line in f:
            for match in SRS_REQ_PATTERN.finditer(line):
                rid = match.group(1)
                if "-T-" not in rid:     # exclude test IDs
                    req_ids.add(rid)
    return req_ids


def extract_requirements_from_tests(tests_dir: str) -> dict[str, list[dict]]:
    """
    Scan all test_*.py files and extract @requirement() decorator arguments.

    Returns:
        dict mapping requirement_id → list of {file, line, test_name}
    """
    coverage = defaultdict(list)

    for py_file in Path(tests_dir).rglob("test_*.py"):
        with open(py_file, encoding="utf-8") as f:
            source = f.read()

        # Method 1: regex on decorator strings (fast, handles all cases)
        for match in DECORATOR_STR_PATTERN.finditer(source):
            req_id = match.group(1)
            # Find approximate line number
            line_no = source[:match.start()].count('\n') + 1
            coverage[req_id].append({
                "file": str(py_file),
                "line": line_no,
                "test_name": _find_test_name_near(source, match.start()),
                "method": "decorator"
            })

        # Method 2: AST parsing for _requirement_ids attribute (more robust)
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            func = decorator.func
                            name = getattr(func, 'id', None) or getattr(func, 'attr', '')
                            if name == 'requirement':
                                for arg in decorator.args:
                                    if isinstance(arg, ast.Constant):
                                        req_id = str(arg.value)
                                        if SRS_REQ_PATTERN.match(req_id):
                                            # Avoid duplicates with regex method
                                            existing = [
                                                e["test_name"]
                                                for e in coverage[req_id]
                                            ]
                                            if node.name not in existing:
                                                coverage[req_id].append({
                                                    "file": str(py_file),
                                                    "line": node.lineno,
                                                    "test_name": node.name,
                                                    "method": "ast"
                                                })
        except SyntaxError:
            pass   # Skip files with syntax errors

    return dict(coverage)


def extract_requirements_from_source(source_dir: str) -> dict[str, list[dict]]:
    """
    Scan all .py source files (excluding tests/) for # REQ: comments.

    Returns:
        dict mapping requirement_id → list of {file, line}
    """
    source_refs = defaultdict(list)

    for py_file in Path(source_dir).rglob("*.py"):
        if "test" in py_file.name or "tests" in str(py_file):
            continue
        with open(py_file, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                for match in CODE_REQ_PATTERN.finditer(line):
                    req_id = match.group(1)
                    source_refs[req_id].append({
                        "file": str(py_file),
                        "line": line_no
                    })

    return dict(source_refs)


def _find_test_name_near(source: str, pos: int) -> str:
    """Find the name of the def nearest after a given position in source."""
    snippet = source[pos:pos + 200]
    match = re.search(r'def\s+(\w+)\s*\(', snippet)
    return match.group(1) if match else "<unknown>"


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(
    all_requirements: set[str],
    test_coverage: dict[str, list[dict]],
    source_refs: dict[str, list[dict]],
    output_path: str
) -> tuple[int, int]:
    """
    Generate a Markdown traceability matrix report.

    Returns:
        (covered_count, uncovered_count)
    """
    covered   = []
    uncovered = []

    for req_id in sorted(all_requirements):
        tests   = test_coverage.get(req_id, [])
        sources = source_refs.get(req_id, [])
        if tests:
            covered.append((req_id, tests, sources))
        else:
            uncovered.append((req_id, sources))

    lines = [
        "# VELA — Traceability Matrix Report",
        f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        f"**Total requirements:** {len(all_requirements)}  ",
        f"**Covered by tests:** {len(covered)}  ",
        f"**Uncovered:** {len(uncovered)}  ",
        f"**Coverage:** {len(covered)/max(len(all_requirements),1)*100:.1f}%\n",
        "---\n",
    ]

    # ── Covered requirements
    lines.append("## ✅ Covered requirements\n")
    lines.append("| Requirement | Test(s) | Source reference |")
    lines.append("|-------------|---------|-----------------|")

    for req_id, tests, sources in covered:
        test_links = "<br>".join(
            f"`{t['test_name']}` ({Path(t['file']).name}:{t['line']})"
            for t in tests[:3]   # show max 3
        )
        if len(tests) > 3:
            test_links += f"<br>... +{len(tests)-3} more"

        src_links = "<br>".join(
            f"`{Path(s['file']).name}:{s['line']}`"
            for s in sources[:2]
        ) if sources else "—"

        lines.append(f"| `{req_id}` | {test_links} | {src_links} |")

    # ── Uncovered requirements
    if uncovered:
        lines.append("\n## ❌ Uncovered requirements\n")
        lines.append("> These requirements have no associated test. "
                     "Write a test before implementing the feature (TDD).\n")
        lines.append("| Requirement | Source reference | Action required |")
        lines.append("|-------------|-----------------|-----------------|")

        for req_id, sources in uncovered:
            src_links = "<br>".join(
                f"`{Path(s['file']).name}:{s['line']}`"
                for s in sources[:2]
            ) if sources else "—"
            lines.append(
                f"| `{req_id}` | {src_links} | "
                f"Add `@requirement(\"{req_id}\")` to a test |"
            )
    else:
        lines.append("\n## ✅ All requirements are covered by at least one test.\n")

    # ── Summary by sprint
    lines.append("\n## Coverage by requirement category\n")
    fr_reqs  = [r for r in all_requirements if "-FR-" in r]
    nfr_reqs = [r for r in all_requirements if "-NFR-" in r]
    fr_cov   = sum(1 for r in fr_reqs  if r in test_coverage)
    nfr_cov  = sum(1 for r in nfr_reqs if r in test_coverage)

    lines.append("| Category | Total | Covered | % |")
    lines.append("|----------|-------|---------|---|")
    lines.append(
        f"| Functional (FR) | {len(fr_reqs)} | {fr_cov} | "
        f"{fr_cov/max(len(fr_reqs),1)*100:.0f}% |"
    )
    lines.append(
        f"| Non-functional (NFR) | {len(nfr_reqs)} | {nfr_cov} | "
        f"{nfr_cov/max(len(nfr_reqs),1)*100:.0f}% |"
    )

    # Write file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return len(covered), len(uncovered)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VELA traceability scanner — requirements ↔ tests matrix"
    )
    parser.add_argument("--srs",    required=True,
                        help="Path to SRS_VELA.md")
    parser.add_argument("--tests",  required=True,
                        help="Directory containing test_*.py files")
    parser.add_argument("--source", default=None,
                        help="Directory containing source .py files "
                             "(scanned for # REQ: comments)")
    parser.add_argument("--output", default="traceability_report.md",
                        help="Output report path")
    parser.add_argument("--fail-on-uncovered", action="store_true",
                        help="Exit with code 1 if any requirement has no test")
    args = parser.parse_args()

    print("VELA Traceability Scanner")
    print("=" * 50)

    # 1. Extract requirements from SRS
    print(f"[1] Scanning SRS: {args.srs}")
    all_reqs = extract_requirements_from_srs(args.srs)
    print(f"    Found {len(all_reqs)} requirement IDs")

    # 2. Extract test coverage
    print(f"[2] Scanning tests: {args.tests}")
    test_cov = extract_requirements_from_tests(args.tests)
    print(f"    Found references to {len(test_cov)} unique requirement IDs")

    # 3. Extract source references
    source_refs = {}
    if args.source:
        print(f"[3] Scanning source: {args.source}")
        source_refs = extract_requirements_from_source(args.source)
        print(f"    Found # REQ: comments for {len(source_refs)} IDs")

    # 4. Generate report
    print(f"[4] Generating report: {args.output}")
    covered, uncovered = generate_report(
        all_reqs, test_cov, source_refs, args.output
    )

    # 5. Print summary
    total = len(all_reqs)
    pct   = covered / max(total, 1) * 100
    print(f"\nResults: {covered}/{total} requirements covered ({pct:.1f}%)")

    if uncovered > 0:
        print(f"\n[WARNING] {uncovered} requirement(s) have no test:")
        for req_id in sorted(all_reqs):
            if req_id not in test_cov:
                print(f"  - {req_id}")

    if args.fail_on_uncovered and uncovered > 0:
        print(f"\n[FAIL] --fail-on-uncovered : {uncovered} uncovered requirement(s)")
        sys.exit(1)

    print(f"\n[OK] Report written to: {args.output}")
    sys.exit(0)


if __name__ == "__main__":
    main()