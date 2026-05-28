"""Unit tests for the wandb run re-lock fix (GH issue #1 / bead derby-d9q).

BUG: a trainer's wandb run re-locks with a ServerResponseError
'run ID <x> is in use' on resume after a SIGTERM-capped / interrupted chunk
never `wandb.finish()`'d cleanly — the run stays "running" server-side, so the
next slice's `resume(id=x, resume="allow")` collides and crash-loops
`gomoku.train` on startup.

FIX (both tested here, with wandb fully MOCKED — no network, no GPU):
  (a) the SIGTERM/cap stop path releases the run cleanly via
      `_release_wandb_run` -> mark_preempting() + finish().
  (c) `_init_wandb_run` recovers from the 'run ID in use' lock (retry once,
      then fall back to a fresh id) instead of crash-looping.

These exercise the two extracted helpers directly. CPU only, no torch model,
no wandb network — every wandb touch-point is a stub object.
"""
from __future__ import annotations

import pytest

from gomoku.train import (
    _init_wandb_run,
    _is_run_in_use_error,
    _release_wandb_run,
)


class _RunInUseError(Exception):
    """Stand-in for wandb's ServerResponseError with the lock message."""


class _FakeRun:
    """Minimal stand-in for a wandb run object."""

    def __init__(self, run_id="abc123"):
        self.id = run_id
        self.mark_preempting_calls = 0
        self.finish_calls = 0

    def mark_preempting(self):
        self.mark_preempting_calls += 1

    def finish(self):
        self.finish_calls += 1


class _FakeWandb:
    """Stub `wandb` module. `init` returns a fresh run, or raises the
    configured exceptions for the first N calls (to simulate the lock)."""

    def __init__(self, raise_first_n=0, exc=None, fresh_id="freshXYZ"):
        self.raise_first_n = raise_first_n
        self.exc = exc or _RunInUseError("run ID abc123 is in use")
        self.fresh_id = fresh_id
        self.init_calls = []  # list of kwargs dicts

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if len(self.init_calls) <= self.raise_first_n:
            raise self.exc
        rid = kwargs.get("id") or self.fresh_id
        return _FakeRun(run_id=rid)


# --------------------------------------------------------------------------
# Fix (a): the stop path releases the run cleanly.
# --------------------------------------------------------------------------

def test_release_wandb_run_marks_preempting_and_finishes():
    """The SIGTERM/cap stop path must mark the run preempting AND finish it so
    it is no longer 'running' server-side and the next resume can re-acquire."""
    run = _FakeRun()
    _release_wandb_run(run)
    assert run.mark_preempting_calls == 1
    assert run.finish_calls == 1


def test_release_wandb_run_is_noop_when_no_run():
    """No active run (e.g. --no-wandb): releasing must be a safe no-op, not
    an AttributeError on None."""
    _release_wandb_run(None)  # must not raise


def test_release_wandb_run_survives_finish_failure():
    """A wandb hiccup during release must NOT turn a clean capped-slice exit
    into a crash — finish() raising is swallowed (best-effort)."""

    class _FlakyRun(_FakeRun):
        def finish(self):
            raise RuntimeError("network blip during finish")

    run = _FlakyRun()
    _release_wandb_run(run)  # must not raise
    assert run.mark_preempting_calls == 1


def test_release_wandb_run_finishes_even_without_mark_preempting():
    """Older wandb without mark_preempting must still get finish() called."""

    class _NoMarkRun:
        def __init__(self):
            self.finish_calls = 0

        def finish(self):
            self.finish_calls += 1

    run = _NoMarkRun()
    _release_wandb_run(run)
    assert run.finish_calls == 1


# --------------------------------------------------------------------------
# Normal init (non-interrupted) — the resumed-single-run-per-lane model.
# --------------------------------------------------------------------------

def test_init_resumes_same_id_when_unlocked():
    """Normal completion / clean resume: when the id is free, we resume the
    SAME id with resume='allow' — the resumed-single-run-per-lane model is
    unchanged (embedded wandb_run_id + derby/* logging depend on this)."""
    wb = _FakeWandb(raise_first_n=0)
    run = _init_wandb_run(wb, project="gomoku", name="lane", run_id="abc123",
                          config={"x": 1})
    assert run.id == "abc123"
    assert len(wb.init_calls) == 1
    assert wb.init_calls[0]["id"] == "abc123"
    assert wb.init_calls[0]["resume"] == "allow"


def test_init_fresh_run_when_no_id():
    """First-ever slice (no embedded id): fresh run, resume=None."""
    wb = _FakeWandb(raise_first_n=0)
    run = _init_wandb_run(wb, project="gomoku", name="lane", run_id=None,
                          config={})
    assert wb.init_calls[0]["id"] is None
    assert wb.init_calls[0]["resume"] is None
    assert run.id == "freshXYZ"


# --------------------------------------------------------------------------
# Fix (c): recover from the 'run ID in use' lock instead of crash-looping.
# --------------------------------------------------------------------------

def test_init_retries_then_succeeds_on_transient_lock():
    """If the lock clears on the retry (server mid-release), we re-acquire the
    SAME id — no fresh run, model preserved."""
    wb = _FakeWandb(raise_first_n=1)  # first call locked, retry succeeds
    run = _init_wandb_run(wb, project="gomoku", name="lane", run_id="abc123",
                          config={})
    assert len(wb.init_calls) == 2
    assert run.id == "abc123"  # still the same id


def test_init_falls_back_to_fresh_id_on_persistent_lock(capsys):
    """Persistent 'run ID in use' lock: after a retry, fall back to a FRESH id
    with a logged warning rather than crash-looping the trainer."""
    wb = _FakeWandb(raise_first_n=2)  # both id-resume attempts locked
    run = _init_wandb_run(wb, project="gomoku", name="lane", run_id="abc123",
                          config={})
    # Three init calls: resume, retry-resume, then the fresh fallback.
    assert len(wb.init_calls) == 3
    assert wb.init_calls[2]["id"] is None  # fresh id
    assert wb.init_calls[2]["resume"] is None
    assert run.id == "freshXYZ"
    out = capsys.readouterr().out
    assert "fresh run id" in out.lower()


def test_init_does_not_swallow_unrelated_errors():
    """A non-lock error (e.g. auth failure) must propagate, not silently fall
    back to a fresh run — we only recover the specific lock collision."""
    wb = _FakeWandb(raise_first_n=1, exc=ValueError("bad api key"))
    with pytest.raises(ValueError, match="bad api key"):
        _init_wandb_run(wb, project="gomoku", name="lane", run_id="abc123",
                        config={})


def test_init_lock_without_id_propagates():
    """A lock-shaped error with NO run_id to recover (can't be the resume
    collision we handle) must propagate rather than loop."""
    wb = _FakeWandb(raise_first_n=1)
    with pytest.raises(_RunInUseError):
        _init_wandb_run(wb, project="gomoku", name="lane", run_id=None,
                        config={})


# --------------------------------------------------------------------------
# The error matcher.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "run ID abc123 is in use",
    "Run abc is in use",
    "ServerResponseError: run id z0i6qs0x is in use",
])
def test_is_run_in_use_error_matches_lock_messages(msg):
    assert _is_run_in_use_error(Exception(msg))


@pytest.mark.parametrize("msg", [
    "connection refused",
    "invalid api key",
    "",
])
def test_is_run_in_use_error_rejects_unrelated(msg):
    assert not _is_run_in_use_error(Exception(msg))
