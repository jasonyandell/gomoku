"""Tests for scripts/derby_broker.py — the derby's policy tick loop.

GPU-free and eval-free: the daemon submit/poll and the H2H verdict are injected
mocks, so the tick mechanics (submit, fold climb signal, conditional peak-stay
age-out, refill, climb-rate pick, verdict apply, loser-swap) are exercised with
no MPS and no real round-robin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gpu_daemon as gd  # noqa: E402
import derby_pool  # noqa: E402
import derby_broker as db  # noqa: E402


class FakeDaemon:
    """Records submits; poll returns a controllable state/result."""

    def __init__(self):
        self.submitted: list[tuple[str, dict]] = []
        self._n = 0
        self.poll_state = "running"
        self.poll_result: dict = {}

    def submit(self, queue, spec):
        self._n += 1
        jid = f"job-{self._n}"
        self.submitted.append((jid, dict(spec)))
        return jid

    def poll(self, queue, job_id):
        return {"state": self.poll_state, "returncode": 0,
                "result": dict(self.poll_result), "job": {"wall_secs": 300.0}}


def make_broker(tmp_path, fake, pool_dir=None, verdict_fn=None, **cfg):
    return db.Broker(
        derby_dir=tmp_path / "derby",
        queue=gd.Queue(tmp_path / "q"),
        pool_dir=pool_dir,
        config=cfg,
        submit_fn=fake.submit,
        poll_fn=fake.poll,
        verdict_fn=verdict_fn,
        sync_verdict=True,
    )


# --- submit / fold cycle ----------------------------------------------------
def test_tick_submits_a_chunk(tmp_path):
    fake = FakeDaemon()
    b = make_broker(tmp_path, fake)
    b.seed("control", "cellA")
    out = b.tick()
    assert out["action"] == "submitted" and out["lane"] == "control"
    assert b.state["current_job"]["job_id"] == "job-1"
    spec = fake.submitted[0][1]
    assert spec["cell"] == "cellA"
    assert spec["max_wall_secs"] == 300.0
    assert spec["resume_from"] == "auto"
    assert spec["kind"] == "train"


def test_no_double_submit_while_chunk_running(tmp_path):
    fake = FakeDaemon()  # poll stays "running"
    b = make_broker(tmp_path, fake)
    b.seed("c", "cellA")
    b.tick()  # submits job-1
    out = b.tick()  # chunk still running
    assert out["action"] == "chunk_running"
    assert len(fake.submitted) == 1  # serial discipline: no second chunk


def test_fold_updates_climb_and_peak(tmp_path):
    fake = FakeDaemon()
    b = make_broker(tmp_path, fake)
    b.seed("c", "cellA")
    b.tick()  # submit
    fake.poll_state, fake.poll_result = "done", {"model_elo": 1200.0}
    b.tick()  # fold + resubmit
    lane = b.state["lanes"]["c"]
    assert lane["chunks_done"] == 1
    assert lane["peak_elo"] == 1200.0
    assert lane["wall_secs_total"] == 300.0
    assert lane["chunks_since_peak"] == 0


def test_climb_rate_and_since_peak_track(tmp_path):
    fake = FakeDaemon()
    b = make_broker(tmp_path, fake)
    b.seed("c", "cellA")
    elos = [1000.0, 1100.0, 1050.0]  # up, up, down
    for e in elos:
        b.tick()
        fake.poll_state, fake.poll_result = "done", {"model_elo": e}
        b.tick()
        fake.poll_state = "running"
    lane = b.state["lanes"]["c"]
    assert lane["chunks_done"] == 3
    assert lane["peak_elo"] == 1100.0
    assert lane["chunks_since_peak"] == 1  # the 1050 didn't beat the 1100 peak
    assert lane["climb_rate"] is not None


# --- conditional peak-stay age-out (the "never cut a climber" rule) ----------
def test_age_out_only_when_past_ttl_AND_plateaued(tmp_path):
    b = make_broker(tmp_path, FakeDaemon(), ttl_secs=100.0, peak_window=2)
    b.seed("plateau", "cellA")
    b.seed("climber", "cellB")
    b.seed("young", "cellC")
    b.state["lanes"]["plateau"].update(wall_secs_total=200.0, chunks_since_peak=5)
    b.state["lanes"]["climber"].update(wall_secs_total=200.0, chunks_since_peak=0)  # past ttl, still peaking
    b.state["lanes"]["young"].update(wall_secs_total=50.0, chunks_since_peak=9)      # plateaued but under ttl
    aged = b._age_out_pass()
    assert aged == ["plateau"]
    assert b.state["lanes"]["plateau"]["status"] == "aged_out"
    assert b.state["lanes"]["climber"]["status"] == "running"
    assert b.state["lanes"]["young"]["status"] == "running"


# --- refill from the open-entry pool ----------------------------------------
def test_refill_claims_candidates_into_free_slots(tmp_path):
    pool = tmp_path / "pool"
    derby_pool.register("x", "cellX", "lever x", pool_dir=pool)
    derby_pool.register("y", "cellY", "lever y", pool_dir=pool)
    derby_pool.register("z", "cellZ", "lever z", pool_dir=pool)
    b = make_broker(tmp_path, FakeDaemon(), pool_dir=pool, pool_size=2)
    added = b._refill_pass()
    assert len(added) == 2
    assert len(b._running()) == 2
    # third stays available in the pool
    assert len(derby_pool.list_candidates(status="available", pool_dir=pool)) == 1


def test_no_refill_without_pool(tmp_path):
    b = make_broker(tmp_path, FakeDaemon(), pool_size=4)  # pool_dir None
    assert b._refill_pass() == []


# --- climb-rate priority pick -----------------------------------------------
def test_pick_entry_fee_then_steepest_climb(tmp_path):
    b = make_broker(tmp_path, FakeDaemon())
    b.seed("a", "cA")
    b.seed("b", "cB")
    assert b._pick_lane() == "a"  # neither measured -> entry fee, tie broken by name
    b.state["lanes"]["a"].update(climb_rate=10.0, chunks_done=1, chunks_since_peak=0)
    b.state["lanes"]["b"].update(climb_rate=50.0, chunks_done=1, chunks_since_peak=0)
    assert b._pick_lane() == "b"  # steeper climb wins


def test_pick_deprioritizes_plateaued(tmp_path):
    b = make_broker(tmp_path, FakeDaemon(), peak_window=3)
    b.seed("flat", "cA")
    b.seed("slow", "cB")
    b.state["lanes"]["flat"].update(climb_rate=99.0, chunks_done=5, chunks_since_peak=9)  # plateaued
    b.state["lanes"]["slow"].update(climb_rate=1.0, chunks_done=5, chunks_since_peak=0)   # still climbing
    assert b._pick_lane() == "slow"  # patience: plateaued sinks below a slow climber


# --- verdict apply (the trustworthy swap/crown signal) ----------------------
def test_apply_verdict_crowns_and_clears_needs_you(tmp_path):
    b = make_broker(tmp_path, FakeDaemon())
    b.state["needs_you"] = "stale"
    b._apply_verdict({"crowned": "w", "escalate": False,
                      "ranking": [("w", 30.0, 5.0)], "reason": "clear"})
    assert b.state["champion"] == "w"
    assert b.state["needs_you"] is None


def test_apply_verdict_escalate_raises_needs_you(tmp_path):
    b = make_broker(tmp_path, FakeDaemon())
    b._apply_verdict({"crowned": None, "escalate": True, "ranking": [], "reason": "too close"})
    assert b.state["champion"] is None
    assert "too close" in b.state["needs_you"]


def test_swap_loser_retires_clearly_dominated_lane(tmp_path):
    b = make_broker(tmp_path, FakeDaemon())
    b.seed("win", "cA")
    b.seed("lose", "cB")
    b._apply_verdict({"crowned": "win", "escalate": False,
                      "ranking": [("win", 40.0, 5.0), ("lose", -40.0, 5.0)], "reason": ""})
    assert b.state["lanes"]["lose"]["status"] == "aged_out"  # margin 80 >> hypot(5,5)
    assert b.state["lanes"]["win"]["status"] == "running"


def test_swap_loser_holds_when_inside_ci(tmp_path):
    b = make_broker(tmp_path, FakeDaemon())
    b.seed("win", "cA")
    b.seed("lose", "cB")
    b._apply_verdict({"crowned": "win", "escalate": False,
                      "ranking": [("win", 10.0, 20.0), ("lose", -10.0, 20.0)], "reason": ""})
    assert b.state["lanes"]["lose"]["status"] == "running"  # margin 20 < hypot(20,20)=28


def test_maybe_verdict_triggers_when_due(tmp_path):
    captured = {}

    def vf(peaks):
        captured["peaks"] = peaks
        return {"crowned": "a", "escalate": False,
                "ranking": [("a", 5.0, 1.0), ("b", -5.0, 1.0)], "reason": ""}

    b = make_broker(tmp_path, FakeDaemon(), verdict_fn=vf, verdict_period_secs=1.0)
    b.seed("a", "cA")
    b.seed("b", "cB")
    b.state["lanes"]["a"].update(wall_secs_total=100.0, peak_path="/tmp/a.pt")
    b.state["lanes"]["b"].update(wall_secs_total=100.0, peak_path="/tmp/b.pt")
    b._maybe_verdict()
    assert captured["peaks"] == {"a": "/tmp/a.pt", "b": "/tmp/b.pt"}
    assert b.state["champion"] == "a"


def test_maybe_verdict_skips_when_not_due(tmp_path):
    called = []
    b = make_broker(tmp_path, FakeDaemon(),
                    verdict_fn=lambda p: called.append(p) or {}, verdict_period_secs=1e9)
    b.seed("a", "cA")
    b.seed("b", "cB")
    b.state["lanes"]["a"].update(wall_secs_total=10.0, peak_path="/tmp/a.pt")
    b.state["lanes"]["b"].update(wall_secs_total=10.0, peak_path="/tmp/b.pt")
    b._maybe_verdict()
    assert called == []  # below the period -> no verdict run


# --- persistence ------------------------------------------------------------
def test_state_persists_across_reload(tmp_path):
    fake = FakeDaemon()
    b = make_broker(tmp_path, fake)
    b.seed("c", "cellA")
    b.tick()
    b2 = db.Broker(derby_dir=tmp_path / "derby", queue=gd.Queue(tmp_path / "q"))
    assert "c" in b2.state["lanes"]
    assert b2.state["current_job"]["job_id"] == "job-1"
