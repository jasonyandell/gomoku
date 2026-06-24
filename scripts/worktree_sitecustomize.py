"""Worktree-aware import shim for the shared gomoku venv.

THE FRICTION THIS KILLS
-----------------------
The shared venv installs `gomoku` editable via a MetaPathFinder that maps
`gomoku -> /Users/jason/code/gomoku/gomoku` (the MAIN checkout). `install()`
*appends* that finder to ``sys.meta_path``, so the stdlib ``PathFinder`` (which
searches ``sys.path``) is consulted FIRST. Consequence: running code from a
worktree only picks up the worktree's source when the worktree root happens to
be on ``sys.path`` — true for ``python -c`` from the worktree cwd (the ``''``
entry), but FALSE for ``python scripts/foo.py`` (``sys.path[0]`` is ``scripts/``)
and for ``pytest`` (entry script lives in ``.venv/bin``). The symptom is a real
``gomoku.external_engine.ExternalEnginePlayer`` instance that is silently the
MAIN checkout's older class — e.g. ``hasattr(eng, 'analyze') == False`` on a
branch that added ``analyze``. Hours get lost rediscovering this.

THE FIX
-------
Installed as ``sitecustomize.py`` in the venv's site-packages, this runs at
interpreter startup (every invocation: script, ``-m``, pytest, REPL) and
prepends the *enclosing worktree root* (nearest ancestor of the entry script,
else the cwd, that contains ``gomoku/__init__.py``) to ``sys.path``. PathFinder
then resolves ``gomoku`` to the worktree you are standing in. Never raises — an
import shim must not be able to break the interpreter.

INSTALL / REINSTALL (idempotent; safe to run anytime, e.g. after a venv rebuild):
    python scripts/worktree_sitecustomize.py            # install into active venv
    python scripts/worktree_sitecustomize.py --check    # report only, no write
"""
from __future__ import annotations

import os
import sys


def _under(path: str, base: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(base)]) == os.path.abspath(base)
    except Exception:
        return False


def _find_worktree_root(start: str) -> str | None:
    """Nearest ancestor of ``start`` containing ``gomoku/__init__.py``."""
    try:
        d = os.path.abspath(start)
    except Exception:
        return None
    while True:
        if os.path.isfile(os.path.join(d, "gomoku", "__init__.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def activate() -> str | None:
    """Prepend the enclosing worktree root to sys.path. Returns it (or None)."""
    candidates: list[str] = []
    # 1) the entry script's dir — the worktree that OWNS the code being run.
    #    Skip anything inside the venv itself (e.g. .venv/bin/pytest), which would
    #    otherwise resolve UP to the main checkout the venv lives in.
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 and argv0 not in ("-c", "-m", ""):
        sd = os.path.dirname(os.path.abspath(argv0))
        if os.path.isdir(sd) and not _under(sd, sys.prefix) and not _under(sd, sys.base_prefix):
            candidates.append(sd)
    # 2) the cwd — covers `python -m`, pytest, and the REPL.
    try:
        candidates.append(os.getcwd())
    except Exception:
        pass

    for c in candidates:
        root = _find_worktree_root(c)
        if not root:
            continue
        # Already first on the path (e.g. cwd '' already points here)? Done.
        if sys.path and os.path.abspath(sys.path[0] or ".") == root:
            return root
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != root]
        sys.path.insert(0, root)
        os.environ.setdefault("GOMOKU_WORKTREE_ROOT", root)
        return root
    return None


# --- install path ---------------------------------------------------------

def _site_packages() -> str:
    import sysconfig
    return sysconfig.get_paths()["purelib"]


def _install(check_only: bool) -> int:
    sp = _site_packages()
    dst = os.path.join(sp, "sitecustomize.py")
    src = os.path.abspath(__file__)
    with open(src, "r") as f:
        body = f.read()
    existing = None
    if os.path.isfile(dst):
        with open(dst, "r") as f:
            existing = f.read()
    if existing == body:
        print(f"[worktree-pathfix] already current: {dst}")
        return 0
    if check_only:
        state = "MISSING" if existing is None else "STALE"
        print(f"[worktree-pathfix] {state} at {dst} (run without --check to install)")
        return 1
    if existing is not None and "worktree" not in existing.lower():
        print(f"[worktree-pathfix] REFUSING to overwrite a foreign sitecustomize: {dst}")
        return 2
    with open(dst, "w") as f:
        f.write(body)
    print(f"[worktree-pathfix] installed -> {dst}")
    return 0


# When imported AS sitecustomize, activate at startup. When run as a script,
# (re)install into the active venv. When imported under its own name, do nothing.
if __name__ == "sitecustomize":
    try:
        activate()
    except Exception:
        pass
elif __name__ == "__main__":
    raise SystemExit(_install(check_only="--check" in sys.argv))
