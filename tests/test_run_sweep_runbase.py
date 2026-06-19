"""run_sweep --run-base / GOMOKU_RUN_DIR decoupling (epic #53, P3).

Default must stay REPO_ROOT so the derby and every existing caller are unchanged;
the autolab points it at ~/data so run DATA lives outside the ephemeral code
worktree it checks out to run.
"""
from __future__ import annotations

from scripts.run_sweep import CELLS, REPO_ROOT, cell_dirs, run_base


def test_run_base_default_is_repo_root(monkeypatch):
    monkeypatch.delenv("GOMOKU_RUN_DIR", raising=False)
    assert run_base() == REPO_ROOT
    d = cell_dirs(CELLS["SMOKE"])
    assert str(d["checkpoint_dir"]).startswith(str(REPO_ROOT))


def test_run_base_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GOMOKU_RUN_DIR", str(tmp_path))
    assert run_base() == tmp_path
    d = cell_dirs(CELLS["SMOKE"])
    assert str(d["checkpoint_dir"]).startswith(str(tmp_path))
    assert str(d["log_dir"]).startswith(str(tmp_path))
