"""Torch-free unit tests for the sliding-derby promotion gate (#39/#38).

The head-to-head executor is STUBBED with a tiny object that mimics
``HeadToHeadResult`` (``.win_rate``, ``.score``, ``.n_games``, ``.delta_elo``,
``.delta_ci_half``), so nothing here loads torch or plays a real game. We assert
the SYMMETRIC THREE-WAY verdict rule (RFC v2 CHANGE 3): PROMOTE only on a
statistically-clear win (ci_lo > 0.5+margin), REVERT only on a statistically-clear
LOSS (ci_hi < 0.5-margin), and AMBIGUOUS on a noisy CI-straddles-0.5 tie (the v2
fix — a near-tie must NOT read as "proven worse"). Plus the ``--gate-on
white_loss`` selector (RFC v2 CHANGE 2), the board json round-trip, the
append-only verdict log, atomic promotion, and dry-run immutability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import scripts.sliding_gate as sg


# --- A stub of HeadToHeadResult: only the fields the gate reads. ------------
@dataclass
class FakeH2H:
    wins: int
    draws: int
    losses: int

    @property
    def n_games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def win_rate(self) -> float:
        return self.score / max(self.n_games, 1)

    # The gate reads these via getattr with a 0.0 default; provide them.
    delta_elo = 0.0
    delta_ci_half = 0.0


def _eval_fn_for(result: FakeH2H):
    """Build an eval_fn stub that ignores all args and returns `result`."""

    def _fn(candidate, peak, **kwargs):
        return result

    return _fn


def _touch(p: Path, content: bytes = b"ckpt") -> Path:
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# Pure verdict rule
# ---------------------------------------------------------------------------
def test_promote_when_candidate_clearly_beats_peak():
    # 40 games, candidate wins 32 (80%) — CI lower bound well above 0.5. This is
    # the (a) "clear win => PROMOTE" case.
    v = sg.decide_verdict(
        candidate="cand.pt", peak="peak.pt",
        win_rate=0.80, score=32.0, n_games=40,
    )
    assert v.verdict == sg.PROMOTE
    assert v.promote is True
    assert v.revert is False and v.ambiguous is False
    assert v.ci_lo > 0.5
    assert v.gate_on == sg.GATE_ON_H2H, "default gate-on is h2h"


def test_revert_when_candidate_loses():
    # candidate wins only 12/40 (30%) — a clear loss; CI [0.181, 0.454] sits
    # ENTIRELY below 0.5, so ci_hi < 0.5 => REVERT (statistically WORSE).
    v = sg.decide_verdict(
        candidate="cand.pt", peak="peak.pt",
        win_rate=0.30, score=12.0, n_games=40,
    )
    assert v.verdict == sg.REVERT
    assert v.revert is True
    assert v.promote is False and v.ambiguous is False
    assert v.ci_hi < 0.5, "a clear loss must clear its OWN CI below 0.5"


def test_revert_on_decisive_loss_ci_hi_below_half_minus_margin():
    # 4/40 (10%) — a decisive loss; CI [0.040, 0.231]. ci_hi well below 0.5-margin
    # => REVERT under the symmetric rule (this is the (b) "clear loss" case).
    v = sg.decide_verdict(
        candidate="cand.pt", peak="peak.pt",
        win_rate=0.10, score=4.0, n_games=40, ci_margin=0.05,
    )
    assert v.ci_hi < 0.5 - 0.05, "must clear ci_hi < 0.5 - margin"
    assert v.verdict == sg.REVERT


def test_ambiguous_on_noisy_tie_ci_straddles_half():
    # candidate wins 22/40 (55%) — above 0.5 on the point estimate, but with only
    # 40 games the Wilson CI STRADDLES 0.5 ([0.398, 0.693]). Under the symmetric
    # v2 rule this is AMBIGUOUS ("can't tell yet") — crucially NOT REVERT (a noisy
    # near-tie must not read as "proven worse"). This is the key v2 fix.
    v = sg.decide_verdict(
        candidate="cand.pt", peak="peak.pt",
        win_rate=0.55, score=22.0, n_games=40,
    )
    assert v.ci_lo < 0.5 < v.ci_hi, "this case must actually straddle 0.5"
    assert v.verdict == sg.AMBIGUOUS, "a CI straddling 0.5 is AMBIGUOUS, not REVERT"
    assert v.verdict != sg.REVERT, "a noisy tie must NOT read as proven-worse"
    assert v.ambiguous is True
    assert v.promote is False and v.revert is False


def test_exact_half_is_ambiguous():
    # 20/40 = exactly 0.5 — the CI is symmetric about 0.5 ([0.352, 0.648]), so it
    # straddles: AMBIGUOUS under the v2 rule (was REVERT under the old binary one).
    v = sg.decide_verdict(
        candidate="cand.pt", peak="peak.pt",
        win_rate=0.50, score=20.0, n_games=40,
    )
    assert v.verdict == sg.AMBIGUOUS


def test_ci_margin_makes_gate_stricter():
    # A win that PROMOTEs at margin 0 (39/60, CI [0.524, 0.758]) no longer clears
    # the promote bar at margin 0.20 (ci_lo 0.524 is not > 0.70). Under the
    # SYMMETRIC v2 rule the borderline win falls into the AMBIGUOUS band — NOT
    # REVERT (the candidate is not proven worse; a strict margin just widens the
    # "can't tell" zone symmetrically).
    base = sg.decide_verdict(
        candidate="c.pt", peak="p.pt", win_rate=0.65, score=39.0, n_games=60,
    )
    strict = sg.decide_verdict(
        candidate="c.pt", peak="p.pt", win_rate=0.65, score=39.0, n_games=60,
        ci_margin=0.20,
    )
    assert base.verdict == sg.PROMOTE
    assert strict.verdict == sg.AMBIGUOUS
    assert strict.verdict != sg.REVERT
    assert strict.ci_margin == 0.20


def test_no_peak_seeds_ladder():
    v = sg.decide_verdict(
        candidate="cand.pt", peak=None, win_rate=1.0, score=0.0, n_games=0,
    )
    assert v.verdict == sg.PROMOTE
    assert "seed" in v.reason.lower()


# ---------------------------------------------------------------------------
# --gate-on white_loss: the PRIMARY metric becomes the white-DEFENSE rate
# (1 - white_loss) with its OWN Wilson CI (RFC v2 CHANGE 2).
# ---------------------------------------------------------------------------
def test_gate_on_white_loss_promote_when_defends_well():
    # Candidate did NOT lose 36 of 40 white games (white_loss_rate 0.10), so the
    # white-defense rate is 0.90 with CI [0.769, 0.960] — ci_lo well clear of 0.5.
    # Gating on white_loss => PROMOTE. The H2H signal is a logged guardrail.
    v = sg.decide_verdict(
        candidate="cand.pt", peak="peak.pt",
        win_rate=0.50, score=20.0, n_games=40,  # H2H is a dead-needle tie here
        delta_elo=-12.0,                          # guardrail: no big regression
        white_loss_rate=0.10,
        white_loss_score=36.0, white_loss_n=40,
        gate_on=sg.GATE_ON_WHITE_LOSS,
    )
    assert v.gate_on == sg.GATE_ON_WHITE_LOSS
    assert v.verdict == sg.PROMOTE, "strong white defense promotes on the moving signal"
    # The verdict's primary rate IS the white-defense rate, not H2H win_rate.
    assert abs(v.win_rate - 0.90) < 1e-9
    assert v.ci_lo > 0.5
    # H2H rides along as the logged guardrail.
    assert v.delta_elo == -12.0
    assert "white_loss" in v.reason and "guardrail" in v.reason.lower()


def test_gate_on_white_loss_revert_when_defends_badly():
    # Candidate lost 36 of 40 white games => white-defense rate 0.10, CI entirely
    # below 0.5 => statistically WORSE => REVERT under the symmetric rule.
    v = sg.decide_verdict(
        candidate="cand.pt", peak="peak.pt",
        win_rate=0.50, score=20.0, n_games=40,
        white_loss_rate=0.90, white_loss_score=4.0, white_loss_n=40,
        gate_on=sg.GATE_ON_WHITE_LOSS,
    )
    assert v.verdict == sg.REVERT
    assert v.ci_hi < 0.5


def test_gate_on_white_loss_ambiguous_when_noisy():
    # 5/10 white-defense wins => rate 0.5, CI [0.237, 0.763] straddles => AMBIGUOUS.
    v = sg.decide_verdict(
        candidate="cand.pt", peak="peak.pt",
        win_rate=0.50, score=20.0, n_games=40,
        white_loss_rate=0.50, white_loss_score=5.0, white_loss_n=10,
        gate_on=sg.GATE_ON_WHITE_LOSS,
    )
    assert v.verdict == sg.AMBIGUOUS


def test_gate_on_white_loss_no_white_games_is_ambiguous():
    # No white-side games => no signal => AMBIGUOUS (never a silent REVERT).
    v = sg.decide_verdict(
        candidate="cand.pt", peak="peak.pt",
        win_rate=0.50, score=20.0, n_games=40,
        white_loss_rate=None, white_loss_score=None, white_loss_n=0,
        gate_on=sg.GATE_ON_WHITE_LOSS,
    )
    assert v.verdict == sg.AMBIGUOUS
    assert "no white" in v.reason.lower() or "no signal" in v.reason.lower()


# ---------------------------------------------------------------------------
# Board json round-trip
# ---------------------------------------------------------------------------
def test_board_roundtrip(tmp_path: Path):
    board_p = tmp_path / "board.json"
    board = sg.default_board()
    v = sg.decide_verdict(
        candidate="cand.pt", peak="peak.pt", win_rate=0.80, score=32.0, n_games=40,
    )
    # run_gate sets v.peak to the snapshot path before recording a promotion; the
    # board's authoritative peak is that snapshot, not the candidate's own path.
    v.peak = "snapshot_peak.pt"
    sg.record_verdict_on_board(board, v, promoted=True)
    sg.save_board(board, board_p)

    loaded = sg.load_board(board_p)
    assert loaded["peak_path"] == "snapshot_peak.pt"
    assert loaded["lap"] == 1
    assert loaded["promotions"] == 1
    assert loaded["reverts"] == 0
    assert loaded["last_verdict"]["verdict"] == sg.PROMOTE
    assert loaded["last_verdict"]["applied"] is True
    assert len(loaded["history"]) == 1


def test_board_revert_increments_reverts(tmp_path: Path):
    board_p = tmp_path / "board.json"
    board = sg.default_board()
    board["peak_path"] = "old_peak.pt"
    v = sg.decide_verdict(
        candidate="cand.pt", peak="old_peak.pt", win_rate=0.30, score=12.0, n_games=40,
    )
    sg.record_verdict_on_board(board, v, promoted=False)
    sg.save_board(board, board_p)

    loaded = sg.load_board(board_p)
    assert loaded["peak_path"] == "old_peak.pt", "REVERT keeps the old peak"
    assert loaded["reverts"] == 1
    assert loaded["promotions"] == 0
    assert loaded["lap"] == 1


def test_load_board_missing_returns_fresh(tmp_path: Path):
    loaded = sg.load_board(tmp_path / "nope.json")
    assert loaded == sg.default_board()


# ---------------------------------------------------------------------------
# Verdict log — append-only
# ---------------------------------------------------------------------------
def test_verdict_log_appends(tmp_path: Path):
    log_p = tmp_path / "verdicts.jsonl"
    v1 = sg.decide_verdict(candidate="a.pt", peak="p.pt",
                           win_rate=0.80, score=32.0, n_games=40)
    v2 = sg.decide_verdict(candidate="b.pt", peak="p.pt",
                           win_rate=0.30, score=12.0, n_games=40)
    sg.append_verdict_log(log_p, v1, applied=True, ts="2026-01-01T00:00:00Z")
    sg.append_verdict_log(log_p, v2, applied=False, ts="2026-01-01T00:01:00Z")

    lines = log_p.read_text().strip().splitlines()
    assert len(lines) == 2
    r1, r2 = (json.loads(line) for line in lines)
    assert r1["candidate"] == "a.pt" and r1["verdict"] == sg.PROMOTE and r1["applied"] is True
    assert r2["candidate"] == "b.pt" and r2["verdict"] == sg.REVERT and r2["applied"] is False
    assert r1["ts"] == "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Atomic promotion of the peak file
# ---------------------------------------------------------------------------
def test_promote_candidate_copies_file(tmp_path: Path):
    cand = _touch(tmp_path / "cand.pt", b"new-champion-bytes")
    peak_out = tmp_path / "peak" / "peak.pt"
    out = sg.promote_candidate(str(cand), peak_out)
    assert Path(out) == peak_out
    assert peak_out.read_bytes() == b"new-champion-bytes"
    assert not peak_out.with_suffix(peak_out.suffix + ".tmp").exists(), "no tmp left"


def test_promote_candidate_missing_raises(tmp_path: Path):
    try:
        sg.promote_candidate(str(tmp_path / "absent.pt"), tmp_path / "peak.pt")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for a missing candidate")


# ---------------------------------------------------------------------------
# Full run_gate with a stubbed eval — applies / dry-run immutability
# ---------------------------------------------------------------------------
def _gate_kwargs(tmp_path: Path, candidate: Path, peak: Path | None):
    return dict(
        candidate=str(candidate),
        peak=(str(peak) if peak is not None else None),
        n_games=40, seed=7,
        board_path=tmp_path / "board.json",
        verdict_log=tmp_path / "verdicts.jsonl",
        peak_out=tmp_path / "peak_out.pt",
        sims=50, c_puct=1.5, opening_plies=4, device="cpu", n_workers=1,
        ci_margin=0.0, want_white_loss=False,
    )


def test_run_gate_promote_applies_and_persists(tmp_path: Path):
    cand = _touch(tmp_path / "cand.pt", b"candidate")
    peak = _touch(tmp_path / "peak.pt", b"peak")
    kw = _gate_kwargs(tmp_path, cand, peak)
    v = sg.run_gate(
        **kw, dry_run=False,
        eval_fn=_eval_fn_for(FakeH2H(wins=32, draws=0, losses=8)),
    )
    assert v.verdict == sg.PROMOTE
    # Peak file actually slid to the candidate bytes.
    assert kw["peak_out"].read_bytes() == b"candidate"
    # Board persisted with the new peak + a promotion + an applied history entry.
    board = sg.load_board(kw["board_path"])
    assert board["promotions"] == 1
    assert board["peak_path"] == str(kw["peak_out"])
    assert board["history"][-1]["applied"] is True
    # Verdict logged.
    assert kw["verdict_log"].exists()
    assert json.loads(kw["verdict_log"].read_text().splitlines()[0])["applied"] is True


def test_run_gate_dry_run_does_not_mutate(tmp_path: Path):
    cand = _touch(tmp_path / "cand.pt", b"candidate")
    peak = _touch(tmp_path / "peak.pt", b"peak")
    kw = _gate_kwargs(tmp_path, cand, peak)
    v = sg.run_gate(
        **kw, dry_run=True,
        eval_fn=_eval_fn_for(FakeH2H(wins=32, draws=0, losses=8)),
    )
    assert v.verdict == sg.PROMOTE, "dry-run still COMPUTES the verdict"
    # The peak was NOT moved, and the board was NOT written.
    assert not kw["peak_out"].exists(), "dry-run must not create the new peak"
    assert not kw["board_path"].exists(), "dry-run must not mutate the board"
    # But the verdict WAS logged (for human review), marked not-applied.
    rec = json.loads(kw["verdict_log"].read_text().splitlines()[0])
    assert rec["verdict"] == sg.PROMOTE and rec["applied"] is False


def test_run_gate_revert_keeps_old_peak(tmp_path: Path):
    cand = _touch(tmp_path / "cand.pt", b"candidate")
    peak = _touch(tmp_path / "peak.pt", b"peak")
    kw = _gate_kwargs(tmp_path, cand, peak)
    v = sg.run_gate(
        **kw, dry_run=False,
        eval_fn=_eval_fn_for(FakeH2H(wins=12, draws=0, losses=28)),
    )
    assert v.verdict == sg.REVERT
    assert not kw["peak_out"].exists(), "REVERT must not write a new peak"
    board = sg.load_board(kw["board_path"])
    assert board["reverts"] == 1 and board["promotions"] == 0


def test_run_gate_no_peak_seeds_without_eval(tmp_path: Path):
    cand = _touch(tmp_path / "cand.pt", b"candidate")
    kw = _gate_kwargs(tmp_path, cand, None)

    def _boom(*a, **k):  # must NOT be called when there is no peer to play
        raise AssertionError("eval_fn called on a fresh ladder (no peak)")

    v = sg.run_gate(**kw, dry_run=False, eval_fn=_boom)
    assert v.verdict == sg.PROMOTE
    board = sg.load_board(kw["board_path"])
    assert board["peak_path"] == str(kw["peak_out"])


def test_run_gate_ambiguous_keeps_old_peak(tmp_path: Path):
    # A noisy near-tie (22/40) is AMBIGUOUS: the peak does NOT move, the board
    # records neither a promotion nor a revert, and the JSONL carries the
    # AMBIGUOUS token so a caller can distinguish it from PROMOTE/REVERT.
    cand = _touch(tmp_path / "cand.pt", b"candidate")
    peak = _touch(tmp_path / "peak.pt", b"peak")
    kw = _gate_kwargs(tmp_path, cand, peak)
    v = sg.run_gate(
        **kw, dry_run=False,
        eval_fn=_eval_fn_for(FakeH2H(wins=22, draws=0, losses=18)),
    )
    assert v.verdict == sg.AMBIGUOUS
    assert not kw["peak_out"].exists(), "AMBIGUOUS must not slide the peak"
    board = sg.load_board(kw["board_path"])
    assert board["promotions"] == 0 and board["reverts"] == 0
    assert board["last_verdict"]["verdict"] == sg.AMBIGUOUS
    rec = json.loads(kw["verdict_log"].read_text().splitlines()[0])
    assert rec["verdict"] == sg.AMBIGUOUS and rec["applied"] is False


def _white_loss_fn_for(rate, defense_score, white_n):
    """Stub for candidate_white_loss_tally: returns a fixed white-defense tally
    without touching torch or playing games."""

    def _fn(candidate, peak, **kwargs):
        return (rate, defense_score, white_n)

    return _fn


def test_run_gate_gate_on_white_loss_primary_verdict(tmp_path: Path):
    # --gate-on white_loss: the verdict is driven by the candidate's white-defense
    # rate (here 36/40 NOT lost => promote), NOT the dead-needle H2H tie. The white
    # tally is computed UNCONDITIONALLY (even with want_white_loss=False).
    cand = _touch(tmp_path / "cand.pt", b"candidate")
    peak = _touch(tmp_path / "peak.pt", b"peak")
    kw = _gate_kwargs(tmp_path, cand, peak)
    kw["want_white_loss"] = False  # gate_on=white_loss must force the tally anyway
    v = sg.run_gate(
        **kw, dry_run=False, gate_on=sg.GATE_ON_WHITE_LOSS,
        eval_fn=_eval_fn_for(FakeH2H(wins=20, draws=0, losses=20)),  # H2H = 0.500 tie
        white_loss_fn=_white_loss_fn_for(0.10, 36.0, 40),            # strong defense
    )
    assert v.gate_on == sg.GATE_ON_WHITE_LOSS
    assert v.verdict == sg.PROMOTE, "promotes on the white-defense signal, not H2H"
    assert abs(v.win_rate - 0.90) < 1e-9, "primary metric is the white-defense rate"
    assert v.white_loss_rate == 0.10
    # The peak actually slid (PROMOTE applies regardless of which metric gated).
    assert kw["peak_out"].read_bytes() == b"candidate"
    board = sg.load_board(kw["board_path"])
    assert board["promotions"] == 1
    rec = json.loads(kw["verdict_log"].read_text().splitlines()[0])
    assert rec["gate_on"] == sg.GATE_ON_WHITE_LOSS and rec["verdict"] == sg.PROMOTE


def test_run_gate_gate_on_white_loss_revert_on_bad_defense(tmp_path: Path):
    # White-defense rate 4/40 => CI below 0.5 => REVERT, peak stays.
    cand = _touch(tmp_path / "cand.pt", b"candidate")
    peak = _touch(tmp_path / "peak.pt", b"peak")
    kw = _gate_kwargs(tmp_path, cand, peak)
    v = sg.run_gate(
        **kw, dry_run=False, gate_on=sg.GATE_ON_WHITE_LOSS,
        eval_fn=_eval_fn_for(FakeH2H(wins=20, draws=0, losses=20)),
        white_loss_fn=_white_loss_fn_for(0.90, 4.0, 40),  # loses most white games
    )
    assert v.verdict == sg.REVERT
    assert not kw["peak_out"].exists()
    board = sg.load_board(kw["board_path"])
    assert board["reverts"] == 1 and board["promotions"] == 0
