"""Tests for the autolab arena role (epic #53, P4). Gate/HF injected — no GPU."""
from __future__ import annotations

import types

import pytest

from gomoku.lab import arena as A
from gomoku.lab import daemon as D


def _verdict(verdict, promote, *, wr=0.6, n=40):
    return types.SimpleNamespace(verdict=verdict, promote=promote, win_rate=wr,
                                 ci_lo=wr - 0.1, ci_hi=wr + 0.1, n_games=n, seed=0)


def _arena(verdict, *, resolver=None, **kw):
    rec = {"gate": [], "set": []}

    def gate_fn(*, candidate, peak, n_games):
        rec["gate"].append({"candidate": candidate, "peak": peak, "n_games": n_games})
        return verdict

    def setter(item, cand_ref, cand_path, v):
        rec["set"].append({"ref": cand_ref, "verdict": v.verdict})

    role = A.ArenaRole(gate_fn=gate_fn, champion_resolver=resolver or (lambda: (None, None)),
                       champion_setter=setter, **kw)
    return role, rec


def _item(base="local:///tmp/cand.pt", id="eval-mvp@0"):
    return {"id": id, "role": "arena", "base": base, "config": {"lane": "mvp"}}


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))


def test_first_candidate_becomes_champion(tmp_path):
    role, rec = _arena(_verdict("PROMOTE", True), resolver=lambda: (None, None))
    res = role.run_chunk(_item())
    assert rec["gate"][0]["peak"] is None            # no incumbent to play
    assert rec["set"] and rec["set"][0]["ref"] == "local:///tmp/cand.pt"
    assert res.metrics["verdict"] == "PROMOTE" and res.metrics["promoted"] is True
    assert res.metrics["vs"] == "(first champion)"


def test_promote_vs_champion_bumps_tag(tmp_path):
    role, rec = _arena(_verdict("PROMOTE", True),
                       resolver=lambda: ("champion", "/tmp/champ.pt"))
    role.run_chunk(_item())
    assert rec["gate"][0]["peak"] == "/tmp/champ.pt"
    assert len(rec["set"]) == 1                       # champion moved


def test_ambiguous_does_not_bump(tmp_path):
    role, rec = _arena(_verdict("AMBIGUOUS", False),
                       resolver=lambda: ("champion", "/tmp/champ.pt"))
    res = role.run_chunk(_item())
    assert rec["set"] == []                            # champion unchanged
    assert res.metrics["promoted"] is False and res.metrics["verdict"] == "AMBIGUOUS"


def test_revert_does_not_bump(tmp_path):
    role, rec = _arena(_verdict("REVERT", False),
                       resolver=lambda: ("champion", "/tmp/champ.pt"))
    role.run_chunk(_item())
    assert rec["set"] == []


def test_cotenancy_shrinks_n_games(tmp_path, monkeypatch):
    role, rec = _arena(_verdict("AMBIGUOUS", False), gate_n_games=40, shrink_n_games=12)
    monkeypatch.setattr(D, "probe_alive", lambda path: True)   # a trainer slice is live
    role.run_chunk(_item())
    assert rec["gate"][0]["n_games"] == 12


def test_full_n_games_when_box_free(tmp_path, monkeypatch):
    role, rec = _arena(_verdict("AMBIGUOUS", False), gate_n_games=40, shrink_n_games=12)
    monkeypatch.setattr(D, "probe_alive", lambda path: False)
    role.run_chunk(_item())
    assert rec["gate"][0]["n_games"] == 40


def test_eval_and_verdict_rows_share_ref(tmp_path):
    role, _ = _arena(_verdict("PROMOTE", True))
    res = role.run_chunk(_item(id="eval-mvp@0"))
    ev, vr = res.followups
    assert ev["type"] == "eval" and vr["type"] == "verdict"
    assert ev["ref"] == vr["ref"] == "ev-eval-mvp@0"
    assert vr["gate"] == "PROMOTE"


def test_resolve_model_local_strips_scheme(tmp_path):
    role, _ = _arena(_verdict("PROMOTE", True))
    assert role._resolve_model("local:///a/b/latest.pt") == "/a/b/latest.pt"
    assert role._resolve_model(None) is None
