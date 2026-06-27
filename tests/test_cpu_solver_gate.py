"""The CPU vcf solver is RETIRED as a runtime dependency.

Every public entry point (``solve_vcf`` / ``solve_vct`` and the ``*_from_planes``
wrappers) must raise :class:`gomoku.vcf.CpuSolverRetired` when reached without the
deliberate-use override ``GOMOKU_ALLOW_CPU_SOLVER=1``, and must run normally with
it. See ``wiki/topics/mega-vct-solver.md`` (the on-device GPU solver
``scripts.vct_metal.mega_vct_bb.solve_vct_mega_bb`` is the runtime replacement).

NB: ``tests/conftest.py`` sets the override session-wide (the suite is the kept
oracle), so this test deletes the env var locally via ``monkeypatch`` to exercise
the *closed* gate, then restores it.
"""

from __future__ import annotations

import numpy as np
import pytest

from gomoku import state_ops, vcf

ENTRY_POINTS = ["solve_vcf", "solve_vct", "solve_vcf_from_planes",
                "solve_vct_from_planes"]


def _empty_arg(name):
    """An empty input shaped for ``name``: a (2, N, N) board for the board
    solvers, a (HISTORY_PLY+1, N, N) plane stack for the ``*_from_planes`` ones
    (which read the defender from plane ``HISTORY_PLY``)."""
    n = vcf.BOARD_SIZE
    planes = 2 if not name.endswith("_from_planes") else state_ops.HISTORY_PLY + 1
    return np.zeros((planes, n, n), dtype=bool)


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_entry_point_raises_when_gated(monkeypatch, name):
    """Without the override every public entry point throws CpuSolverRetired."""
    monkeypatch.delenv("GOMOKU_ALLOW_CPU_SOLVER", raising=False)
    fn = getattr(vcf, name)
    with pytest.raises(vcf.CpuSolverRetired):
        fn(_empty_arg(name))


def test_message_points_at_replacement_and_override(monkeypatch):
    monkeypatch.delenv("GOMOKU_ALLOW_CPU_SOLVER", raising=False)
    with pytest.raises(vcf.CpuSolverRetired) as ei:
        vcf.solve_vct(_empty_arg("solve_vct"))
    msg = str(ei.value)
    assert "GOMOKU_ALLOW_CPU_SOLVER=1" in msg
    assert "solve_vct_mega_bb" in msg


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_override_unblocks_entry_point(monkeypatch, name):
    """With the override set the solver runs (empty board -> no forced win)."""
    monkeypatch.setenv("GOMOKU_ALLOW_CPU_SOLVER", "1")
    fn = getattr(vcf, name)
    res = fn(_empty_arg(name))
    assert res.has_forced_win is False
