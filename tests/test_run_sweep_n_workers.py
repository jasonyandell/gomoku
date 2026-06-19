"""run_sweep --n-workers overrides the cell's generator count (#69).

n_workers (the parallel self-play generator count) was a hardcoded Cell field;
--n-workers N makes it a launch-time knob so generators scale without editing
the cell. Omitted → the cell value is unchanged. Mirrors the --epochs override.
"""
from __future__ import annotations

from scripts import run_sweep
from scripts.run_sweep import CELLS


def _resolve_cell_via_main(monkeypatch, argv):
    """Run main() with launch_cell stubbed, returning the cell it would launch."""
    captured = {}
    monkeypatch.setattr(run_sweep, "launch_cell",
                        lambda cell, **kw: captured.__setitem__("cell", cell))
    monkeypatch.setattr("sys.argv", ["run_sweep.py", *argv])
    run_sweep.main()
    return captured["cell"]


def test_n_workers_overrides_cell(monkeypatch):
    original = CELLS["SMOKE"].n_workers
    try:
        cell = _resolve_cell_via_main(
            monkeypatch, ["--cell", "SMOKE", "--n-workers", "13"])
        assert cell.n_workers == 13
    finally:
        CELLS["SMOKE"].n_workers = original


def test_n_workers_omitted_keeps_cell_default(monkeypatch):
    expected = CELLS["SMOKE"].n_workers
    cell = _resolve_cell_via_main(monkeypatch, ["--cell", "SMOKE"])
    assert cell.n_workers == expected


def test_n_workers_zero_is_honored(monkeypatch):
    """0 is a valid (if degenerate) override — guard uses `is not None`, not truthiness."""
    original = CELLS["SMOKE"].n_workers
    try:
        cell = _resolve_cell_via_main(
            monkeypatch, ["--cell", "SMOKE", "--n-workers", "0"])
        assert cell.n_workers == 0
    finally:
        CELLS["SMOKE"].n_workers = original
