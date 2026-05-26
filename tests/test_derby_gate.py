"""Gate-logic tests for scripts/derby_gate.py.

CRITICAL: these MUST NOT run a real H2H eval (it needs real checkpoints and is
minutes-slow on MPS). We monkeypatch `derby_gate.run_round_robin` — the SOLE
function that touches the real eval — to return synthetic per-pair Δelo + CI
results, and assert ONLY the gate logic:

  (a) clear winner          -> crowned set,  escalate False
  (b) overlapping top pair  -> triggers escalation (run_round_robin re-called
                               with just the overlapping pair at escalate_to)
  (c) still overlapping     -> crowned None, escalate True
  (d) ranking order + mean-centering correct

The pure math (mean_centered_ratings, combined_ci, decide) is also exercised
directly — no mocking needed there.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import derby_gate as dg  # noqa: E402


def _pair(a, b, delta_a_vs_b, ci_half, n_games=200):
    """A synthetic per-pair result in run_round_robin's output shape."""
    return {"a": a, "b": b, "delta_elo_a_vs_b": float(delta_a_vs_b),
            "ci_half": float(ci_half), "a_wins": 0, "draws": 0, "b_wins": 0,
            "n_games": n_games}


# ---- pure math -----------------------------------------------------------


def test_mean_centered_ratings_order_and_centering():
    # A beats B by +120, A beats C by +90, B beats C by +60.
    # ratings: A=(120+90)/2=105, B=(-120+60)/2=-30, C=(-90-60)/2=-75.
    pairs = [_pair("A", "B", 120, 10), _pair("A", "C", 90, 10),
             _pair("B", "C", 60, 10)]
    ranking = dg.mean_centered_ratings(["A", "B", "C"], pairs)
    names = [t[0] for t in ranking]
    assert names == ["A", "B", "C"], "ranked best-first by mean Δelo"
    ratings = {t[0]: t[1] for t in ranking}
    assert ratings["A"] == pytest.approx(105.0)
    assert ratings["B"] == pytest.approx(-30.0)
    assert ratings["C"] == pytest.approx(-75.0)
    # Mean-centered: the per-lane ratings sum to ~0.
    assert sum(ratings.values()) == pytest.approx(0.0, abs=1e-9)


def test_mean_centered_ratings_ci_is_quadrature_over_opponents():
    # Two opponents, each pair CI half=30 -> lane CI = sqrt(30^2+30^2)/2.
    pairs = [_pair("A", "B", 100, 30), _pair("A", "C", 100, 30),
             _pair("B", "C", 0, 30)]
    ranking = dg.mean_centered_ratings(["A", "B", "C"], pairs)
    ci = {t[0]: t[2] for t in ranking}
    assert ci["A"] == pytest.approx(math.hypot(30, 30) / 2)


def test_decide_clear_vs_overlap():
    # Clear: margin 100 well outside combined CI hypot(10,10)~14.1.
    clear = [("A", 100.0, 10.0), ("B", 0.0, 10.0)]
    crowned, escalate, reason = dg.decide(clear)
    assert crowned == "A" and escalate is False and "clear" in reason
    # Overlap: margin 10 inside combined CI hypot(40,40)~56.6.
    overlap = [("A", 10.0, 40.0), ("B", 0.0, 40.0)]
    crowned, escalate, reason = dg.decide(overlap)
    assert crowned is None and escalate is True and "overlap" in reason


# ---- gate orchestration (run_round_robin mocked) -------------------------


def test_clear_winner_no_escalation(monkeypatch):
    """(a) clear winner -> crowned set, escalate False, and NO escalation call."""
    calls = []

    def fake_rr(peaks, *, games_per_pair, pairs=None, **kw):
        calls.append((games_per_pair, pairs))
        # A dominates the field, tight CIs -> separates cleanly.
        return [_pair("A", "B", 150, 12), _pair("A", "C", 130, 12),
                _pair("B", "C", 40, 12)]

    monkeypatch.setattr(dg, "run_round_robin", fake_rr)
    v = dg.verdict({"A": "a.pt", "B": "b.pt", "C": "c.pt"},
                   games_per_pair=200, escalate_to=400)
    assert v["crowned"] == "A"
    assert v["escalate"] is False
    assert v["ranking"][0][0] == "A"
    # Only the initial round-robin was played — no escalation.
    assert len(calls) == 1 and calls[0][0] == 200 and calls[0][1] is None


def test_overlap_triggers_escalation_then_resolves(monkeypatch):
    """(b) overlapping top pair -> escalation re-runs JUST that pair at
    escalate_to; the tighter CI then separates the leader (crowned, escalate
    False)."""
    calls = []

    def fake_rr(peaks, *, games_per_pair, pairs=None, **kw):
        calls.append((games_per_pair, pairs))
        if pairs is None:
            # Initial: A barely leads B, wide CIs -> overlap at the top.
            return [_pair("A", "B", 20, 60), _pair("A", "C", 200, 20),
                    _pair("B", "C", 180, 20)]
        # Escalation: re-run of the overlapping A-vs-B pair at escalate_to.
        # Same edge, much tighter CI -> A now separates from B.
        assert pairs == [("A", "B")], "only the overlapping pair is re-run"
        assert games_per_pair == 400, "re-run at escalate_to"
        return [_pair("A", "B", 60, 8, n_games=400)]

    monkeypatch.setattr(dg, "run_round_robin", fake_rr)
    v = dg.verdict({"A": "a.pt", "B": "b.pt", "C": "c.pt"},
                   games_per_pair=200, escalate_to=400)
    assert len(calls) == 2, "initial + one escalation"
    assert calls[1] == (400, [("A", "B")])
    assert v["crowned"] == "A"
    assert v["escalate"] is False
    assert "escalation" in v["reason"]
    # The escalated A-vs-B pair (n_games=400) replaced the original.
    ab = next(p for p in v["pairs"] if {p["a"], p["b"]} == {"A", "B"})
    assert ab["n_games"] == 400


def test_still_overlapping_after_escalation_no_verdict(monkeypatch):
    """(c) still overlapping after escalation -> crowned None, escalate True."""
    calls = []

    def fake_rr(peaks, *, games_per_pair, pairs=None, **kw):
        calls.append((games_per_pair, pairs))
        # Both initial and escalation return a buried margin (CI never shrinks
        # enough) -> no trustworthy separation either time.
        if pairs is None:
            return [_pair("A", "B", 10, 60), _pair("A", "C", 200, 20),
                    _pair("B", "C", 190, 20)]
        return [_pair("A", "B", 10, 55, n_games=400)]

    monkeypatch.setattr(dg, "run_round_robin", fake_rr)
    v = dg.verdict({"A": "a.pt", "B": "b.pt", "C": "c.pt"},
                   games_per_pair=200, escalate_to=400)
    assert len(calls) == 2, "initial + escalation both ran"
    assert v["crowned"] is None
    assert v["escalate"] is True
    assert "no verdict" in v["reason"] and "hold incumbent" in v["reason"]


def test_ranking_order_and_mean_centering_through_verdict(monkeypatch):
    """(d) ranking order + mean-centering surface correctly out of verdict()."""
    def fake_rr(peaks, *, games_per_pair, pairs=None, **kw):
        # C strongest, then A, then B; tight CIs so it resolves on the first pass.
        return [_pair("A", "B", 80, 10), _pair("A", "C", -100, 10),
                _pair("B", "C", -200, 10)]

    monkeypatch.setattr(dg, "run_round_robin", fake_rr)
    v = dg.verdict({"A": "a.pt", "B": "b.pt", "C": "c.pt"},
                   games_per_pair=200, escalate_to=400)
    names = [t[0] for t in v["ranking"]]
    assert names == ["C", "A", "B"], "ranked best-first"
    assert v["crowned"] == "C"
    ratings = {t[0]: t[1] for t in v["ranking"]}
    # C=(100+200)/2=150, A=(80-100)/2=-10, B=(-80-200)/2=-140.
    assert ratings["C"] == pytest.approx(150.0)
    assert ratings["A"] == pytest.approx(-10.0)
    assert ratings["B"] == pytest.approx(-140.0)
    assert sum(ratings.values()) == pytest.approx(0.0, abs=1e-9)


def test_two_lane_overlap_no_escalation_disabled(monkeypatch):
    """Escalation disabled (escalate_to <= games_per_pair) + overlap -> no
    verdict without a second eval call."""
    calls = []

    def fake_rr(peaks, *, games_per_pair, pairs=None, **kw):
        calls.append((games_per_pair, pairs))
        return [_pair("A", "B", 5, 50)]

    monkeypatch.setattr(dg, "run_round_robin", fake_rr)
    v = dg.verdict({"A": "a.pt", "B": "b.pt"},
                   games_per_pair=200, escalate_to=200)
    assert len(calls) == 1, "no escalation when escalate_to <= games_per_pair"
    assert v["crowned"] is None and v["escalate"] is True


def test_verdict_requires_two_lanes():
    with pytest.raises(ValueError):
        dg.verdict({"A": "a.pt"})
