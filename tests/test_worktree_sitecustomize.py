"""Worktree-aware import shim (scripts/worktree_sitecustomize.py).

Locks in the path-resolution logic that kills the editable-install gotcha:
`python scripts/x.py` / pytest from a worktree must resolve `gomoku` to THAT
worktree, not the editable-mapped main checkout. See
wiki/topics/worktree-hygiene.md § Editable-install gotcha.
"""
import importlib.util
import os
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "worktree_sitecustomize.py"


def _load():
    spec = importlib.util.spec_from_file_location("worktree_sitecustomize", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_find_worktree_root_walks_up_to_gomoku_pkg(tmp_path):
    mod = _load()
    root = tmp_path / "wt"
    (root / "gomoku").mkdir(parents=True)
    (root / "gomoku" / "__init__.py").write_text("")
    (root / "scripts").mkdir()
    # From a nested dir, the nearest ancestor holding gomoku/__init__.py wins.
    assert mod._find_worktree_root(str(root / "scripts")) == str(root)
    assert mod._find_worktree_root(str(root)) == str(root)


def test_find_worktree_root_none_when_absent(tmp_path):
    mod = _load()
    assert mod._find_worktree_root(str(tmp_path)) is None


def test_under_is_containment(tmp_path):
    mod = _load()
    base = tmp_path / "venv"
    base.mkdir()
    assert mod._under(str(base / "bin"), str(base)) is True
    assert mod._under(str(tmp_path / "other"), str(base)) is False


def test_activate_prepends_worktree_root(tmp_path, monkeypatch):
    mod = _load()
    root = tmp_path / "wt"
    (root / "gomoku").mkdir(parents=True)
    (root / "gomoku" / "__init__.py").write_text("")
    (root / "scripts").mkdir()  # the entry script's dir must exist (isdir gate)
    import sys
    monkeypatch.setattr(sys, "argv", [str(root / "scripts" / "x.py")])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    got = mod.activate()
    assert got == str(root)
    assert os.path.abspath(sys.path[0]) == str(root)


def test_install_check_reports_without_writing(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_site_packages", lambda: str(tmp_path))
    rc = mod._install(check_only=True)
    assert rc == 1  # MISSING -> nonzero, and nothing written
    assert not (tmp_path / "sitecustomize.py").exists()


def test_install_writes_and_is_idempotent(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_site_packages", lambda: str(tmp_path))
    assert mod._install(check_only=False) == 0
    dst = tmp_path / "sitecustomize.py"
    assert dst.exists() and "worktree" in dst.read_text().lower()
    # Second run is a no-op (already current).
    assert mod._install(check_only=False) == 0


def test_install_refuses_foreign_sitecustomize(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "_site_packages", lambda: str(tmp_path))
    (tmp_path / "sitecustomize.py").write_text("# someone else's startup hook\n")
    assert mod._install(check_only=False) == 2  # refused, not clobbered
