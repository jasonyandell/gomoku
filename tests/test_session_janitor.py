"""Behavioral tests for scripts/session_janitor.sh — the SessionStart/PreCompact
worktree janitor wrapper wired in for GitHub issue #48.

These prove the three properties the issue demands, against the script's real
behavior (not a tautology):
  1. A normal run exits 0 and prints a `repo-hygiene:` line.
  2. A failing python interpreter is NON-FATAL — the wrapper STILL exits 0
     (a janitor hiccup must never block session start).
  3. The python-picker prefers `.venv/bin/python` when present and falls back
     to `python3` when it is absent.

They also assert the live `.claude/settings.json` (the gitignored, machine-local
hook config) wires the janitor command into BOTH the SessionStart and PreCompact
hook arrays IF that file exists. See the receipt / issue #48 note: settings.json
is NOT tracked, so the wrapper is the durable deliverable and the settings edit
is applied to the live file out-of-band.
"""
import json
import os
import subprocess
import sys

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
JANITOR = os.path.realpath(os.path.join(SCRIPTS, "session_janitor.sh"))
RECLAIM = os.path.realpath(os.path.join(SCRIPTS, "reclaim_worktrees.py"))


def _run(env_extra=None, project_dir=None):
    """Run the janitor wrapper and capture (returncode, stdout+stderr)."""
    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = project_dir or os.path.realpath(
        os.path.join(SCRIPTS, "..")
    )
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        ["bash", JANITOR],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_script_exists_and_executable():
    assert os.path.isfile(JANITOR), JANITOR
    assert os.access(JANITOR, os.X_OK), "session_janitor.sh must be executable"


def test_normal_run_exits_zero_and_emits_gauge():
    rc, out = _run()
    assert rc == 0, f"expected exit 0, got {rc}\n{out}"
    assert "repo-hygiene:" in out, f"missing hygiene gauge line:\n{out}"


def test_failing_python_is_nonfatal():
    # /bin/false errors on every invocation -> both --apply and --gauge fail.
    # The wrapper must swallow it and STILL exit 0, still printing a
    # repo-hygiene: line (the sentinel) so the human sees the janitor ran.
    rc, out = _run(env_extra={"GOMOKU_JANITOR_PY": "/bin/false"})
    assert rc == 0, f"non-fatal guard broken: exit {rc}\n{out}"
    assert "repo-hygiene:" in out, f"missing sentinel gauge line:\n{out}"


def test_missing_interpreter_is_nonfatal():
    rc, out = _run(env_extra={"GOMOKU_JANITOR_PY": "/no/such/interpreter"})
    assert rc == 0, f"non-fatal guard broken for missing interp: exit {rc}\n{out}"
    assert "repo-hygiene:" in out


def _pick_python_via_subprocess(tmp_path, make_venv, env_extra=None):
    """Invoke the wrapper end-to-end with PROJECT_DIR pointed at a fake repo
    whose reclaim_worktrees.py is a stub that PRINTS WHICH interpreter ran.

    This proves the picker chose venv vs python3 by observing the actual
    interpreter the wrapper executed — a real behavioral assertion, not a
    string match on the source.
    """
    project = tmp_path / "repo"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)

    # Stub reclaim_worktrees.py: on --gauge, print a repo-hygiene line that
    # records sys.executable's parent dir so we can tell venv from python3.
    stub = scripts / "reclaim_worktrees.py"
    stub.write_text(
        "import sys, os\n"
        "if '--gauge' in sys.argv:\n"
        "    print('repo-hygiene: stub via ' + os.path.realpath(sys.executable))\n"
    )

    venv_python_real = None
    if make_venv:
        venvbin = project / ".venv" / "bin"
        venvbin.mkdir(parents=True)
        fake_py = venvbin / "python"
        # A real, working python: symlink to the test's interpreter so the stub
        # actually runs and the -x check passes.
        os.symlink(os.path.realpath(sys.executable), fake_py)
        venv_python_real = os.path.realpath(str(fake_py))

    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = str(project)
    # Ensure GOMOKU_JANITOR_PY is NOT set so the venv-vs-python3 logic runs.
    env.pop("GOMOKU_JANITOR_PY", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        ["bash", JANITOR],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr, venv_python_real


def test_picker_prefers_venv_when_present(tmp_path):
    rc, out, venv_real = _pick_python_via_subprocess(tmp_path, make_venv=True)
    assert rc == 0, out
    assert "repo-hygiene: stub via" in out, out
    # The venv python is a symlink to sys.executable; realpath resolves both to
    # the same target, so the stub's reported interpreter must match it.
    assert os.path.realpath(sys.executable) in out, (
        f"expected venv interpreter {venv_real} to have run\n{out}"
    )


def test_picker_falls_back_to_python3_when_no_venv(tmp_path):
    rc, out, _ = _pick_python_via_subprocess(tmp_path, make_venv=False)
    assert rc == 0, out
    assert "repo-hygiene: stub via" in out, (
        "fallback python3 must have run the stub\n" + out
    )
    # With no .venv, the picker emits literal "python3"; resolve it to confirm
    # the fallback path actually executed something (not the venv).
    py3 = subprocess.run(
        ["bash", "-c", "command -v python3"], capture_output=True, text=True
    ).stdout.strip()
    if py3:  # python3 on PATH on this box; assert the fallback ran it
        assert os.path.realpath(py3) in out or "python3" in out, out


@pytest.mark.parametrize("array_key", ["SessionStart", "PreCompact"])
def test_live_settings_wires_janitor_if_present(array_key):
    """Once the janitor has been ACTIVATED (the manual, gitignored
    .claude/settings.json edit — #48's human step), it must be wired into BOTH
    hook arrays. This test does NOT assert that activation has happened — merging
    this branch lands only the wrapper, not the settings edit — so it SKIPS until
    session_janitor.sh is referenced anywhere in settings, then enforces both-hook
    correctness. Also skipped when the file is absent (gitignored; CI/worktrees)."""
    repo = os.path.realpath(os.path.join(SCRIPTS, ".."))
    settings_path = os.path.join(repo, ".claude", "settings.json")
    if not os.path.isfile(settings_path):
        pytest.skip("machine-local .claude/settings.json not present in this checkout")
    with open(settings_path) as f:
        cfg = json.load(f)  # asserts valid JSON
    hooks = cfg.get("hooks", {})
    all_commands = [
        h.get("command", "")
        for ak in ("SessionStart", "PreCompact")
        for group in hooks.get(ak, [])
        for h in group.get("hooks", [])
    ]
    if not any("session_janitor.sh" in c for c in all_commands):
        pytest.skip(
            "session_janitor.sh not yet wired into .claude/settings.json — "
            "manual activation step pending (#48)"
        )
    # Activated somewhere → it must be in THIS hook array too (both required).
    commands = [
        h.get("command", "")
        for group in hooks.get(array_key, [])
        for h in group.get("hooks", [])
    ]
    assert any("session_janitor.sh" in c for c in commands), (
        f"session_janitor.sh is activated but not wired into {array_key}: {commands}"
    )
