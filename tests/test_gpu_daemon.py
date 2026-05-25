"""Tests for scripts/gpu_daemon.py — the serial GPU-lane scheduler.

All GPU-free: 'train' jobs are exercised only at the command-building layer; the
end-to-end daemon loop runs trivial `python -c` jobs as the `raw` kind, so these
tests never launch an MPS tenant and are safe to run while the box is busy.
"""
from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gpu_daemon as gd  # noqa: E402


@pytest.fixture
def queue(tmp_path):
    return gd.Queue(tmp_path / "lab_queue")


def _raw(note, cmd, **kw):
    return gd.normalize_job({"kind": "raw", "cmd": cmd, "note": note, **kw})


# --- normalize + validation -------------------------------------------------
def test_normalize_assigns_id_and_defaults():
    job = _raw("hi", ["python", "-c", "print(1)"])
    assert job["id"]
    assert "status" not in job  # status is stamped on write(), not by normalize
    assert job["tier"] == 2 and job["priority"] == 0.0
    assert job["submitted_at"]


def test_normalize_raw_accepts_string_cmd():
    job = gd.normalize_job({"kind": "raw", "cmd": "python -c 'print(1)'"})
    assert job["cmd"] == ["python", "-c", "print(1)"]


def test_normalize_rejects_bad_kind():
    with pytest.raises(ValueError):
        gd.normalize_job({"kind": "nonsense"})


def test_normalize_rejects_train_without_cell():
    with pytest.raises(ValueError):
        gd.normalize_job({"kind": "train"})


def test_normalize_rejects_empty_raw_cmd():
    with pytest.raises(ValueError):
        gd.normalize_job({"kind": "raw", "cmd": []})


# --- priority ordering ------------------------------------------------------
def test_sort_pending_tier_then_priority_then_fifo():
    jobs = [
        {"id": "a", "tier": 2, "priority": 0.0, "submitted_at": "2026-01-01T00:00:02Z"},
        {"id": "b", "tier": 1, "priority": 0.0, "submitted_at": "2026-01-01T00:00:03Z"},
        {"id": "c", "tier": 2, "priority": 9.0, "submitted_at": "2026-01-01T00:00:04Z"},
        {"id": "d", "tier": 2, "priority": 0.0, "submitted_at": "2026-01-01T00:00:01Z"},
    ]
    order = [j["id"] for j in gd.sort_pending(jobs)]
    # tier 1 first; then within tier 2: highest priority, then earliest submit
    assert order == ["b", "c", "d", "a"]


# --- command building -------------------------------------------------------
def test_build_cmd_train_slice():
    run_sweep = gd._import_run_sweep()
    cell = sorted(run_sweep.CELLS)[0]
    job = gd.normalize_job(
        {"kind": "train", "cell": cell, "max_wall_secs": 300, "final_eval": True}
    )
    cmd = gd.build_cmd(job)
    assert "run_sweep.py" in cmd[2]
    assert cmd[cmd.index("--cell") + 1] == cell
    assert "--max-wall-secs" in cmd and "300.0" in cmd
    assert "--final-eval" in cmd
    assert "--resume" not in cmd  # resume_from defaulted to None


def test_build_cmd_train_epochs_override():
    run_sweep = gd._import_run_sweep()
    cell = sorted(run_sweep.CELLS)[0]
    job = gd.normalize_job(
        {"kind": "train", "cell": cell, "overrides": {"epochs": 7}}
    )
    cmd = gd.build_cmd(job)
    assert cmd[cmd.index("--epochs") + 1] == "7"


def test_build_cmd_raw_passthrough():
    job = _raw("x", ["echo", "hi"])
    assert gd.build_cmd(job) == ["echo", "hi"]


def test_unknown_cell_rejected():
    with pytest.raises(ValueError):
        gd.normalize_job({"kind": "train", "cell": "definitely-not-a-cell-xyz"})


# --- queue io ---------------------------------------------------------------
def test_write_and_find_and_move(queue):
    job = _raw("io", ["true"])
    queue.write("pending", job)
    assert queue.find(job["id"])[0] == "pending"
    queue.move("pending", "done", job)
    found = queue.find(job["id"])
    assert found[0] == "done"
    # old pending file is gone
    assert not queue.job_path("pending", job["id"]).exists()


def test_events_appended(queue):
    job = _raw("ev", ["true"])
    queue.write("pending", job)
    queue.event(job["id"], "submit", {"kind": "raw"})
    queue.move("pending", "done", job)
    lines = queue.events_path.read_text().splitlines()
    assert any('"event": "submit"' in ln for ln in lines)
    assert any('"event": "move"' in ln for ln in lines)


# --- end-to-end daemon loop (raw jobs, no GPU) ------------------------------
@pytest.fixture
def restore_signals():
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


def _drain(queue, restore_signals):
    # preflight off: the real box may be busy with the derby; these raw jobs
    # don't touch MPS, so we explicitly own the (fake) lane.
    gd.Daemon(queue, poll_secs=0.05, preflight=False, once=True).run()


def test_daemon_runs_job_to_done(queue, restore_signals):
    job = _raw("ok", [sys.executable, "-c", "print('hello-daemon')"])
    queue.write("pending", job)
    _drain(queue, restore_signals)
    found = queue.find(job["id"])
    assert found[0] == "done"
    assert found[1]["returncode"] == 0
    log = queue.log_path(job["id"]).read_text()
    assert "hello-daemon" in log


def test_daemon_marks_failure(queue, restore_signals):
    job = _raw("boom", [sys.executable, "-c", "import sys; sys.exit(3)"])
    queue.write("pending", job)
    _drain(queue, restore_signals)
    found = queue.find(job["id"])
    assert found[0] == "failed"
    assert found[1]["returncode"] == 3


def test_daemon_respects_priority_order(queue, restore_signals):
    lo = _raw("lo", [sys.executable, "-c", "print('lo')"], tier=2, priority=0.0)
    hi = _raw("hi", [sys.executable, "-c", "print('hi')"], tier=1, priority=0.0)
    queue.write("pending", lo)
    queue.write("pending", hi)
    _drain(queue, restore_signals)
    # both done; the tier-1 job should have an earlier start timestamp
    hi_done = queue.find(hi["id"])[1]
    lo_done = queue.find(lo["id"])[1]
    assert hi_done["started_at"] <= lo_done["started_at"]


def test_reconcile_requeues_crashed_running_job(queue, restore_signals):
    # simulate a hard crash: a job left in running/ with no daemon
    job = _raw("crashed", [sys.executable, "-c", "print('recovered')"])
    queue.write("running", job)
    _drain(queue, restore_signals)
    found = queue.find(job["id"])
    assert found[0] == "done"  # reconciled -> pending -> run -> done
    assert found[1].get("requeued", 0) >= 1


def test_cancel_pending(queue, restore_signals):
    job = _raw("cancelme", [sys.executable, "-c", "print('x')"])
    queue.write("pending", job)
    queue.move("pending", "cancelled", job)  # what cmd_cancel does for pending
    assert queue.find(job["id"])[0] == "cancelled"
