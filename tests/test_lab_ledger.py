"""Tests for the autolab ledger spine (epic #53, P1 #54)."""
from __future__ import annotations

import datetime as _dt

import pytest

from gomoku.lab import ledger as L


def _ts(seconds: int) -> str:
    """A deterministic ISO stamp `seconds` past a fixed epoch."""
    base = _dt.datetime(2026, 6, 18, 12, 0, 0, tzinfo=_dt.timezone.utc)
    return (base + _dt.timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _now(seconds: int) -> _dt.datetime:
    return L.parse_ts(_ts(seconds))


# ---- append: monotonic seq, append-only, fsync round-trip ---------------

def test_append_assigns_monotonic_seq(tmp_path):
    p = tmp_path / "sub" / "ledger.jsonl"  # also exercises mkdir -p
    r0 = L.append(p, L.experiment("exp-1", "train"))
    r1 = L.append(p, L.experiment("exp-2", "train"))
    r2 = L.append(p, L.claim("exp-1", "trainer@h/1", L.lease_until(60, now=_now(0))))
    assert [r0["seq"], r1["seq"], r2["seq"]] == [0, 1, 2]


def test_append_stamps_ts_when_absent(tmp_path):
    p = tmp_path / "l.jsonl"
    r = L.append(p, L.experiment("e", "train"))
    assert r["ts"]  # present and parseable
    assert L.parse_ts(r["ts"]).tzinfo is not None


def test_append_is_append_only(tmp_path):
    """A correction adds a line; it never rewrites the original."""
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("e", "train", priority=10), ts=_ts(0))
    first_line_before = p.read_text().splitlines()[0]
    L.append(p, L.correction("e", {"priority": 99}), ts=_ts(1))
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == first_line_before  # original byte-identical


def test_read_all_round_trips_and_skips_comments(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("e", "train"))
    with open(p, "a") as f:
        f.write("# a hand comment\n\n")  # comment + blank: ignored by read_all
    L.append(p, L.experiment("e2", "arena"))
    rows = L.read_all(p)
    assert [r["id"] for r in rows] == ["e", "e2"]


def test_read_all_missing_file_is_empty(tmp_path):
    assert L.read_all(tmp_path / "nope.jsonl") == []


# ---- fold: status lifecycle ---------------------------------------------

def test_fold_open_then_claimed_then_done(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("e", "train"), ts=_ts(0))
    assert L.fold(L.read_all(p)).experiments["e"]["status"] == L.OPEN
    L.append(p, L.claim("e", "trainer@h/1", L.lease_until(3600, now=_now(0))), ts=_ts(1))
    assert L.fold(L.read_all(p)).experiments["e"]["status"] == L.CLAIMED
    L.append(p, L.result("e", model="hf://x@rev1", metrics={"epochs": 1}), ts=_ts(2))
    e = L.fold(L.read_all(p)).experiments["e"]
    assert e["status"] == L.DONE
    assert e["result"]["model"] == "hf://x@rev1"


def test_fold_records_failed_result(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("e", "train"))
    L.append(p, L.claim("e", "t", L.lease_until(60)))
    L.append(p, L.result("e", status=L.FAILED, error="boom"))
    e = L.fold(L.read_all(p)).experiments["e"]
    assert e["status"] == L.FAILED and e["result"]["error"] == "boom"


# ---- corrections: financial-journal supersession ------------------------

def test_correction_supersedes_field_last_writer_wins(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("e", "train", priority=10))
    L.append(p, L.correction("e", {"priority": 50}, reason="bump"))
    L.append(p, L.correction("e", {"priority": 80}, reason="bump again"))
    e = L.fold(L.read_all(p)).experiments["e"]
    assert e["priority"] == 80
    assert e["_corrections"] == 2


def test_correction_can_reclaim_dead_lane(tmp_path):
    """Setting status back to open via a correction is how a crash is reclaimed."""
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("e", "train"))
    L.append(p, L.claim("e", "trainer@h/dead", L.lease_until(60, now=_now(0))))
    assert L.fold(L.read_all(p)).experiments["e"]["status"] == L.CLAIMED
    L.append(p, L.correction("e", {"status": L.OPEN}, reason="dead pid reclaim"))
    assert L.fold(L.read_all(p)).experiments["e"]["status"] == L.OPEN


def test_correction_on_eval(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.eval_row("ev-1", model="hf://x@rev1", metrics={"elo": 1500}))
    L.append(p, L.correction("ev-1", {"metrics": {"elo": 1620}}, reason="recount"))
    assert L.fold(L.read_all(p)).evals["ev-1"]["metrics"]["elo"] == 1620


# ---- claimable: lease semantics -----------------------------------------

def test_claimable_open_only_for_matching_role(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("t", "train"))
    L.append(p, L.experiment("a", "arena"))
    st = L.fold(L.read_all(p))
    assert [e["id"] for e in st.claimable("train", _now(0))] == ["t"]
    assert [e["id"] for e in st.claimable("arena", _now(0))] == ["a"]


def test_claimable_live_lease_excluded_expired_included(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("live", "train"), ts=_ts(0))
    L.append(p, L.claim("live", "t", L.lease_until(3600, now=_now(0))), ts=_ts(1))
    L.append(p, L.experiment("dead", "train"), ts=_ts(2))
    L.append(p, L.claim("dead", "t", L.lease_until(60, now=_now(0))), ts=_ts(3))
    st = L.fold(L.read_all(p))
    # at t+120s: live lease (until t+3600) still held; dead lease (until t+60) expired
    ids = {e["id"] for e in st.claimable("train", _now(120))}
    assert ids == {"dead"}


def test_claimable_done_never_reclaimed(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("e", "train"), ts=_ts(0))
    L.append(p, L.claim("e", "t", L.lease_until(60, now=_now(0))), ts=_ts(1))
    L.append(p, L.result("e", model="hf://x"), ts=_ts(2))
    # even long after lease expiry, a finished lane is not reclaimable
    assert L.fold(L.read_all(p)).claimable("train", _now(99999)) == []


# ---- pick: priority, starvation floor, FIFO -----------------------------

def test_pick_prefers_higher_priority(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("lo", "train", priority=10))
    L.append(p, L.experiment("hi", "train", priority=90))
    assert L.fold(L.read_all(p)).pick("train", _now(0))["id"] == "hi"


def test_pick_fifo_tiebreak_on_equal_priority(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("first", "train", priority=50))
    L.append(p, L.experiment("second", "train", priority=50))
    assert L.fold(L.read_all(p)).pick("train", _now(0))["id"] == "first"


def test_pick_starvation_floor_prefers_never_completed(tmp_path):
    """Same priority: a lane that has produced NO result outranks one that has.

    The daemon writes no claim rows (singleton via flock, recovery via re-pick),
    so the starvation signal is `_results` (completed slices), not `_claims`.
    Here 'attempted' ran once (a FAILED result) and was reopened by a correction;
    'fresh' has never produced a result, so it wins despite the same priority.
    """
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("attempted", "train", priority=50), ts=_ts(0))
    L.append(p, L.result("attempted", status=L.FAILED, error="x"), ts=_ts(1))
    L.append(p, L.correction("attempted", {"status": L.OPEN}, reason="reopen"), ts=_ts(2))
    L.append(p, L.experiment("fresh", "train", priority=50), ts=_ts(3))
    assert L.fold(L.read_all(p)).pick("train", _now(999))["id"] == "fresh"


def test_pick_none_when_nothing_claimable(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("e", "arena"))
    assert L.fold(L.read_all(p)).pick("train", _now(0)) is None


def test_pick_custom_priority_fn_is_the_p5_seam(tmp_path):
    p = tmp_path / "l.jsonl"
    L.append(p, L.experiment("a", "train", priority=10, config={"elo_per_hr": 5}))
    L.append(p, L.experiment("b", "train", priority=99, config={"elo_per_hr": 200}))
    st = L.fold(L.read_all(p))
    # a Δelo/hr-style policy ignores explicit priority and ranks by the metric
    pick = st.pick("train", _now(0), priority_fn=lambda e: e["config"]["elo_per_hr"])
    assert pick["id"] == "b"


# ---- Ledger handle ------------------------------------------------------

def test_ledger_handle_roundtrip(tmp_path):
    led = L.Ledger(tmp_path / "l.jsonl")
    led.append(L.experiment("e", "train", priority=5))
    led.append(L.experiment("e2", "train", priority=50))
    assert led.pick("train").id if False else led.pick("train")["id"] == "e2"
    assert len(led.read_all()) == 2


# ---- claimant / lease helpers -------------------------------------------

def test_claimant_format():
    c = L.claimant("trainer", host="m5max", pid=4412)
    assert c == "trainer@m5max/4412"


def test_lease_until_is_in_the_future():
    now = _now(0)
    lu = L.parse_ts(L.lease_until(3600, now=now))
    assert (lu - now).total_seconds() == pytest.approx(3600, abs=1)
