"""Tests for the #51 pre-commit gate that auto-fires the workflow resilience
gauge (`scripts/check_workflow_resilience.mjs`) only when a staged change
touches `.claude/workflows/*.js`.

These drive the hook script directly via its testability seam
(`GOMOKU_PRECOMMIT_STAGED` — a newline-separated staged-file list) so no real
`git commit` has to be fabricated. To prove the hook does / does not invoke the
checker, we shadow `node` on PATH with a fake that records its invocation to a
sentinel file and exits with a controllable code. This makes the assertions
about "node was/wasn't run" and "exit code is propagated" non-tautological.

The third test is the smoke from the issue: the real
`node scripts/check_workflow_resilience.mjs` exits 0 on the current tree.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / "scripts" / "hooks" / "pre-commit"
CHECKER = REPO_ROOT / "scripts" / "check_workflow_resilience.mjs"


def test_hook_exists_and_is_executable() -> None:
    assert HOOK.is_file(), f"missing hook: {HOOK}"
    assert os.access(HOOK, os.X_OK), f"hook not executable: {HOOK}"


def _fake_node_dir(tmp_path: Path, sentinel: Path, exit_code: int) -> Path:
    """Create a dir containing a fake `node` that records its invocation to
    *sentinel* and exits with *exit_code*. Returned dir is meant to be prepended
    to PATH so it shadows the real node."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    fake = bindir / "node"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'printf "FAKE-NODE args=%s\\n" "$*" > "$SENTINEL"\n'
        f"exit {exit_code}\n"
    )
    fake.chmod(0o755)
    _ = sentinel  # passed to the subprocess via env, not used here
    return bindir


def _run_hook(staged: str, *, extra_env: dict[str, str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["GOMOKU_PRECOMMIT_STAGED"] = staged
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(HOOK)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_short_circuits_without_running_node_for_non_workflow_changes(tmp_path: Path) -> None:
    """Staged paths with no `.claude/workflows/*.js` -> exit 0 AND node is never
    spawned (normal commits stay untouched / cheap)."""
    sentinel = tmp_path / "sentinel"
    bindir = _fake_node_dir(tmp_path, sentinel, exit_code=0)
    proc = _run_hook(
        "gomoku/train.py\nREADME.md\n.claude/workflows/notes.md",
        extra_env={"SENTINEL": str(sentinel), "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"},
    )
    assert proc.returncode == 0, proc.stderr
    assert not sentinel.exists(), "node was invoked for a non-workflow change (should short-circuit)"


def test_fires_checker_and_passes_when_checker_passes(tmp_path: Path) -> None:
    """A staged `.claude/workflows/*.js` -> the checker IS invoked; a passing
    checker (exit 0) lets the commit through (hook exit 0)."""
    sentinel = tmp_path / "sentinel"
    bindir = _fake_node_dir(tmp_path, sentinel, exit_code=0)
    proc = _run_hook(
        "gomoku/train.py\n.claude/workflows/implement-backlog.js",
        extra_env={"SENTINEL": str(sentinel), "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"},
    )
    assert proc.returncode == 0, proc.stderr
    assert sentinel.exists(), "checker (node) was NOT invoked despite a staged workflow change"
    assert str(CHECKER) in sentinel.read_text(), "checker invoked with the wrong script path"


def test_propagates_nonzero_exit_from_checker(tmp_path: Path) -> None:
    """A staged workflow change + a FAILING checker (exit 1) -> the hook fails
    the commit with the same code. This is the load-bearing gate behavior."""
    sentinel = tmp_path / "sentinel"
    bindir = _fake_node_dir(tmp_path, sentinel, exit_code=1)
    proc = _run_hook(
        ".claude/workflows/reviewer-gated-fanout.js",
        extra_env={"SENTINEL": str(sentinel), "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"},
    )
    assert proc.returncode == 1, f"expected propagated exit 1, got {proc.returncode}: {proc.stderr}"
    assert sentinel.exists(), "checker (node) was NOT invoked despite a staged workflow change"


def test_only_matches_files_directly_under_workflows(tmp_path: Path) -> None:
    """The harness only loads top-level `.claude/workflows/*.js`; a nested path
    must NOT trigger the gauge (no false fire)."""
    sentinel = tmp_path / "sentinel"
    bindir = _fake_node_dir(tmp_path, sentinel, exit_code=1)  # would fail if (wrongly) fired
    proc = _run_hook(
        ".claude/workflows/sub/deep.js",
        extra_env={"SENTINEL": str(sentinel), "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"},
    )
    assert proc.returncode == 0, proc.stderr
    assert not sentinel.exists(), "nested non-top-level workflow path falsely fired the gauge"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_real_checker_exits_zero_on_current_tree() -> None:
    """Smoke: the real resilience checker is green on the committed workflows."""
    proc = subprocess.run(
        ["node", str(CHECKER)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"resilience checker FAILED:\n{proc.stdout}\n{proc.stderr}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
