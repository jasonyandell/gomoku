"""Tests for the autolab shared daemon (epic #53, P2 #56). CPU-only, no torch."""
from __future__ import annotations

import subprocess
import sys
import time

from gomoku.lab import daemon as D
from gomoku.lab import ledger as L

# A stdlib-only child that grabs an flock and holds it — no `gomoku` import, so it
# sidesteps the editable-install finder. Used to prove the kernel auto-frees the
# lock when the holder dies (the whole reason we dropped claim/lease bookkeeping).
_HOLD_CHILD = (
    "import fcntl,os,sys,time\n"
    "fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT,0o644)\n"
    "fcntl.flock(fd,fcntl.LOCK_EX)\n"
    "open(sys.argv[2],'w').close()\n"
    "time.sleep(60)\n"
)


def _seed(path, *experiments):
    for e in experiments:
        L.append(path, e)


# ---- the flock singleton ------------------------------------------------

def test_flock_excludes_second_instance(tmp_path):
    lk = str(tmp_path / "daemon-noop.lock")
    a = D.SingletonLock("noop", path=lk)
    assert a.acquire() is True
    b = D.SingletonLock("noop", path=lk)
    assert b.acquire() is False           # a live holder blocks the second
    a.release()
    assert b.acquire() is True            # freed → second can take it now
    b.release()


def test_lock_autofrees_after_holder_is_killed(tmp_path):
    lk = str(tmp_path / "daemon-x.lock")
    ready = tmp_path / "ready"
    proc = subprocess.Popen([sys.executable, "-c", _HOLD_CHILD, lk, str(ready)])
    try:
        end = time.time() + 10
        while not ready.exists() and time.time() < end:
            time.sleep(0.02)
        assert ready.exists(), "child never acquired the lock"
        assert D.probe_alive(lk) is True
        proc.kill()
        proc.wait()
        end = time.time() + 5
        while D.probe_alive(lk) and time.time() < end:
            time.sleep(0.02)
        assert D.probe_alive(lk) is False   # SIGKILL freed it — no stale cleanup
    finally:
        if proc.poll() is None:
            proc.kill()


def test_probe_alive_and_read_lockfile(tmp_path):
    lk = str(tmp_path / "daemon-noop.lock")
    lock = D.SingletonLock("noop", path=lk)
    lock.acquire()
    lock.write_meta(item="exp-7", status="running")
    assert D.probe_alive(lk) is True
    meta = D.read_lockfile(lk)
    assert meta["item"] == "exp-7" and meta["role"] == "noop" and meta["pid"] > 0
    lock.release()
    assert D.probe_alive(lk) is False        # free
    assert D.read_lockfile(lk)["item"] == "exp-7"  # stale metadata survives


def test_probe_alive_missing_file(tmp_path):
    assert D.probe_alive(str(tmp_path / "nope.lock")) is False
    assert D.read_lockfile(str(tmp_path / "nope.lock")) is None


def test_acquire_creates_missing_dir(tmp_path):
    lk = str(tmp_path / "deep" / "nested" / "daemon-noop.lock")
    lock = D.SingletonLock("noop", path=lk)
    assert lock.acquire() is True
    lock.release()


# ---- the loop -----------------------------------------------------------

def test_run_daemon_once_picks_runs_appends(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    led = str(tmp_path / "ledger.jsonl")
    _seed(led, L.experiment("e", "noop", priority=10))
    assert D.run_daemon(D.NoOpRole(), led, once=True) == 0
    e = L.fold(L.read_all(led)).experiments["e"]
    assert e["status"] == L.DONE and e["result"]["model"] == "noop://e"


def test_failed_run_chunk_writes_failed_result(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    led = str(tmp_path / "ledger.jsonl")
    _seed(led, L.experiment("e", "noop"))
    D.run_daemon(D.NoOpRole(fail=True), led, once=True)
    e = L.fold(L.read_all(led)).experiments["e"]
    assert e["status"] == L.FAILED and "RuntimeError" in e["result"]["error"]


def test_preflight_deferred_does_not_fail_the_lane(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    led = str(tmp_path / "ledger.jsonl")
    _seed(led, L.experiment("e", "noop"))
    D.run_daemon(D.NoOpRole(defer=True), led, once=True)
    st = L.fold(L.read_all(led))
    assert st.experiments["e"]["status"] == L.OPEN      # NOT failed
    assert any(ev["summary"].startswith("preflight deferred") for ev in st.events)


def test_followups_appended_and_picked_next(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    led = str(tmp_path / "ledger.jsonl")
    _seed(led, L.experiment("e", "noop", priority=50))
    role = D.NoOpRole(followups=[L.experiment("e-next", "noop", priority=50)])
    D.run_daemon(role, led, once=True)
    st = L.fold(L.read_all(led))
    assert st.experiments["e"]["status"] == L.DONE
    assert "e-next" in st.experiments and st.experiments["e-next"]["status"] == L.OPEN
    # a second iteration (plain NoOpRole) drains the continuation
    D.run_daemon(D.NoOpRole(), led, once=True)
    assert L.fold(L.read_all(led)).experiments["e-next"]["status"] == L.DONE


def test_repick_after_crash_takes_now_highest(tmp_path, monkeypatch):
    """No claim rows: a fresh daemon just re-picks the first-priority open item."""
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    led = str(tmp_path / "ledger.jsonl")
    _seed(led, L.experiment("lo", "noop", priority=10),
          L.experiment("hi", "noop", priority=90))
    D.run_daemon(D.NoOpRole(), led, once=True)          # picks hi
    assert L.fold(L.read_all(led)).experiments["hi"]["status"] == L.DONE
    _seed(led, L.experiment("urgent", "noop", priority=99))  # appears mid-stream
    D.run_daemon(D.NoOpRole(), led, once=True)          # fresh daemon re-picks
    assert L.fold(L.read_all(led)).experiments["urgent"]["status"] == L.DONE


def test_idle_when_nothing_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    led = str(tmp_path / "ledger.jsonl")
    _seed(led, L.experiment("a", "arena"))   # wrong role for a noop daemon
    assert D.run_daemon(D.NoOpRole(), led, once=True) == 0
    assert L.read_all(led)[-1]["type"] == L.EXPERIMENT   # nothing appended


def test_singleton_held_exits_nonzero_without_touching_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    led = str(tmp_path / "ledger.jsonl")
    _seed(led, L.experiment("e", "noop"))
    held = D.SingletonLock("noop")          # canonical path under AUTOLAB_HOME
    assert held.acquire() is True
    try:
        rc = D.run_daemon(D.NoOpRole(), led, once=True)
        assert rc == D.EXIT_SINGLETON_HELD
        assert L.fold(L.read_all(led)).experiments["e"]["status"] == L.OPEN
    finally:
        held.release()


def test_stop_file_breaks_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    led = str(tmp_path / "ledger.jsonl")
    stop = tmp_path / "STOP"
    stop.write_text("x")
    _seed(led, L.experiment("e", "noop"))
    D.run_daemon(D.NoOpRole(), led, stop_file=str(stop))   # not once: breaks on stop
    assert L.fold(L.read_all(led)).experiments["e"]["status"] == L.OPEN  # never ran


def test_signal_handlers_restored(tmp_path, monkeypatch):
    import signal
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    led = str(tmp_path / "ledger.jsonl")
    _seed(led, L.experiment("e", "noop"))
    sentinel = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        D.run_daemon(D.NoOpRole(), led, once=True)
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL  # restored
    finally:
        signal.signal(signal.SIGTERM, sentinel)


# ---- status surface -----------------------------------------------------

def test_status_board_live_vs_dead(tmp_path, monkeypatch):
    from gomoku.lab import status
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    led = str(tmp_path / "ledger.jsonl")
    _seed(led, L.experiment("t1", "train", priority=50),
          L.experiment("a1", "arena", priority=10))
    L.append(led, L.result("t1", model="hf://x@r1", metrics={"eval/model_elo": 1612}))
    # a live train daemon on t1; arena daemon down with stale meta
    train_lock = D.SingletonLock("train")
    train_lock.acquire()
    train_lock.write_meta(item="t1", status="running")
    dead = D.SingletonLock("arena")
    dead.acquire()
    dead.write_meta(item="a1", status="running")
    dead.release()
    try:
        board = status.lane_board(led)
        assert board["daemons"]["train"]["alive"] is True
        assert board["daemons"]["arena"]["alive"] is False
        text = status.format_board(board)
        assert "train" in text and "ALIVE" in text and "DOWN" in text
        assert "elo=1612" in text
    finally:
        train_lock.release()
