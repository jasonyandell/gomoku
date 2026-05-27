"""Unit tests for the per-color (black/white) eval breakdown.

The eval already alternates the model's color per game; these tests cover the
new per-color split added to `MatchResult` and the aggregation helper that
feeds it, without loading a model or touching the GPU.

Verified invariants:
  (a) per-color fields sum to the aggregate (black_w + white_w == wins, etc.),
  (b) the aggregate W/D/L fields are unchanged / still correct,
  (c) a known black-loss / white-loss split is reported correctly.
"""
from __future__ import annotations

import numpy as np

from gomoku.eval import MatchResult, _match_result_from_outcomes, play_match_pickers


def _all_split_fields(res: MatchResult) -> dict[str, int]:
    return {
        "black_w": res.black_w,
        "black_l": res.black_l,
        "black_d": res.black_d,
        "white_w": res.white_w,
        "white_l": res.white_l,
        "white_d": res.white_d,
    }


def test_match_result_defaults_backward_compatible():
    """Existing call sites construct MatchResult with only aggregates; the new
    split fields must default to 0 so nothing breaks."""
    res = MatchResult(n_games=4, wins=2, losses=1, draws=1)
    assert res.wins == 2 and res.losses == 1 and res.draws == 1
    assert all(v == 0 for v in _all_split_fields(res).values())
    # win_rate semantics unchanged: draws are half-wins.
    assert res.win_rate == (2 + 0.5 * 1) / 4


def test_split_sums_to_aggregate_and_aggregate_correct():
    """(a) split fields sum to the aggregate, (b) aggregate fields correct."""
    # Model color per game: True == model played black.
    outcomes = [
        (True, "win"),    # black win
        (False, "win"),   # white win
        (True, "loss"),   # black loss
        (False, "loss"),  # white loss
        (True, "draw"),   # black draw
        (False, "draw"),  # white draw
        (True, "win"),    # black win
        (False, "loss"),  # white loss
    ]
    res = _match_result_from_outcomes(outcomes, n_games=len(outcomes))

    # (b) aggregate fields correct
    assert res.n_games == 8
    assert res.wins == 3
    assert res.losses == 3
    assert res.draws == 2
    assert res.wins + res.losses + res.draws == res.n_games

    # (a) per-color fields sum to the aggregate
    assert res.black_w + res.white_w == res.wins
    assert res.black_l + res.white_l == res.losses
    assert res.black_d + res.white_d == res.draws


def test_known_black_white_loss_split_reported():
    """(c) a known black-loss / white-loss split is reported correctly.

    Construct a match where the model loses 1 game as black and 3 as white,
    and wins some on each color, and assert each per-color tally exactly."""
    outcomes = [
        (True, "win"),    # black win
        (True, "win"),    # black win
        (True, "loss"),   # black loss  -> 1 black loss
        (False, "win"),   # white win
        (False, "loss"),  # white loss
        (False, "loss"),  # white loss
        (False, "loss"),  # white loss  -> 3 white losses
        (True, "draw"),   # black draw
    ]
    res = _match_result_from_outcomes(outcomes, n_games=len(outcomes))

    assert res.black_w == 2
    assert res.black_l == 1
    assert res.black_d == 1
    assert res.white_w == 1
    assert res.white_l == 3
    assert res.white_d == 0

    # Aggregates match the split.
    assert res.wins == 3
    assert res.losses == 4
    assert res.draws == 1
    assert res.black_l + res.white_l == res.losses
    # The hypothesis-relevant signal: losses cluster on white here.
    assert res.white_l > res.black_l


def _const_picker(action: int):
    """A picker that always plays the same cell if legal, else the first legal
    action. Used to drive a deterministic synthetic match (no model)."""

    def pick(state, rng: np.random.Generator) -> int:
        legal = state.legal_actions()
        if action in legal:
            return int(action)
        return int(legal[0])

    return pick


def test_play_match_pickers_split_consistent_real_engine():
    """End-to-end through the real match engine (no model / GPU): the split
    must always sum to the aggregate and the aggregate must equal n_games."""
    # Two simple deterministic-ish pickers; the actual win/loss distribution
    # doesn't matter — only the bookkeeping invariants do.
    a = _const_picker(40)  # center-ish
    b = _const_picker(0)
    res = play_match_pickers(a, b, n_games=6, seed=0)

    assert res.n_games == 6
    assert res.wins + res.losses + res.draws == 6
    assert res.black_w + res.white_w == res.wins
    assert res.black_l + res.white_l == res.losses
    assert res.black_d + res.white_d == res.draws
    # With 6 games alternating colors, the model plays black 3 and white 3.
    black_games = res.black_w + res.black_l + res.black_d
    white_games = res.white_w + res.white_l + res.white_d
    assert black_games == 3
    assert white_games == 3
