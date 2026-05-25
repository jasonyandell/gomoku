"""Regression test for the wave-mode SIGTERM/SIGINT deadlock.

Bug (observed 3x in the research lab): `gomoku.train --wave-mode` HANGS on
SIGTERM/SIGINT and has to be SIGKILL'd, but ONLY when its selfplay workers are
already dead/gone. The trainer blocks in the wave barrier (`_ingest_worker_wave`,
`while True: ... time.sleep(worker_poll_sec)`) waiting for games from every
worker for the current model version. With the workers orphaned (the derby-kill
sequence), `all(done)` is never satisfied, so the wait never returns — and the
signal handler only flips a flag that is checked at the EPOCH BOUNDARY, which is
reached *after* the ingest. A wait that never returns never sees the flag.

The fix routes a shutdown request raised inside the ingest waits to the same
clean-stop / resumable-save path used by `--max-wall-secs`, so SIGTERM causes a
prompt clean exit even with no workers.

This test reproduces the dead-worker case (an EMPTY worker-input-dir so no wave
ever arrives) and asserts the trainer exits cleanly within a few seconds of
SIGTERM, writing a resumable latest.pt. A small replay buffer keeps the
shutdown checkpoint cheap so the assertion measures signal-handling latency,
not serialization time.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# The trainer needs the repo on PYTHONPATH; the test file lives in tests/.
REPO_ROOT = Path(__file__).resolve().parent.parent

# How long after SIGTERM we allow before declaring a hang. The buggy version
# never exits (had to be SIGKILL'd); the fixed version exits in well under 1s
# of signal-handling latency. 20s is a wide margin for a loaded CI machine
# while still being far below "infinite".
EXIT_DEADLINE_S = 20.0


def _spawn_waveless_trainer(tmp_path: Path) -> subprocess.Popen:
    """Launch a wave-mode trainer pointed at an EMPTY worker-input-dir so it
    blocks on the first wave barrier (no worker ever delivers games). CPU-only,
    no real workers, no GPU."""
    input_dir = tmp_path / "records"
    ckpt_dir = tmp_path / "ckpt"
    input_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = input_dir / "worker_weights.pt"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["WANDB_MODE"] = "disabled"

    cmd = [
        sys.executable, "-u", "-m", "gomoku.train",
        "--size", "tiny",
        "--device", "cpu",
        "--epochs", "100",
        "--no-wandb",
        "--no-eval",
        "--wave-mode",
        "--wave-workers", "1",
        "--wave-games-per-worker", "2",
        "--worker-input-dir", str(input_dir),
        "--worker-weights-path", str(weights_path),
        "--checkpoint-dir", str(ckpt_dir),
        "--worker-poll-sec", "0.2",
        # Small buffer => the shutdown checkpoint is cheap, so we measure
        # signal latency, not serialization time.
        "--replay-buffer-size", "200",
        # A wall cap that is much longer than the test so the ONLY thing that
        # ends the run is our signal (not the cap).
        "--max-wall-secs", "600",
    ]
    # start_new_session so we can clean up the whole group on failure.
    return subprocess.Popen(cmd, env=env, start_new_session=True), ckpt_dir


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_wave_mode_exits_cleanly_on_signal_with_no_workers(tmp_path, sig):
    """No workers + SIGTERM/SIGINT -> trainer exits cleanly (not SIGKILL) and
    writes a resumable latest.pt. The pre-fix trainer hung here forever."""
    proc, ckpt_dir = _spawn_waveless_trainer(tmp_path)
    try:
        # Let it reach the wave barrier (model build + first poll loop).
        time.sleep(6.0)
        assert proc.poll() is None, (
            f"trainer exited early (rc={proc.returncode}) before we could "
            f"signal it; expected it to be blocked on the empty wave barrier"
        )

        t0 = time.time()
        proc.send_signal(sig)
        try:
            rc = proc.wait(timeout=EXIT_DEADLINE_S)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            pytest.fail(
                f"DEADLOCK: trainer did not exit within {EXIT_DEADLINE_S:.0f}s "
                f"of {sig!r}; had to SIGKILL (this is the bug)"
            )
        elapsed = time.time() - t0

        # Clean shutdown returns 0 (it's a normal exit after a resumable save),
        # not a signal-death code (which would be negative under Popen).
        assert rc == 0, (
            f"expected clean exit code 0 on {sig!r}, got {rc} "
            f"(negative => died from the signal, i.e. no clean handler ran)"
        )
        # The resumable checkpoint must be written so a slice can --resume.
        assert (ckpt_dir / "latest.pt").exists(), (
            "clean shutdown must leave a resumable latest.pt"
        )
        # Sanity: it really was prompt, not a slow drift to exit.
        assert elapsed < EXIT_DEADLINE_S
    finally:
        if proc.poll() is None:
            _kill_group(proc)
