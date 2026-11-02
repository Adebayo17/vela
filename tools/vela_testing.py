"""
vela_testing.py — Shared test utilities (golden model + RTL verification).

Usage:
    from vela_testing import requirement

    @requirement("VELA-FR-007")
    def test_something(): ...
"""

import pytest

def requirement(*req_ids):
    ids_label = ",".join(req_ids)
    mark = pytest.mark.requirement(ids=list(req_ids), reason=f"Covers: {ids_label}")
    def decorator(func):
        func._requirement_ids = list(req_ids)
        return mark(func)
    return decorator