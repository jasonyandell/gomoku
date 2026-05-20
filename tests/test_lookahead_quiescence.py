"""Regression test for the lookahead depth=0 quiescence fix.

The static `evaluate_position` in `gomoku.baselines` over-credits "live 4"
patterns (4 stones in a window with no opponent stones) without distinguishing
between unblockable open-fours and trivially-blockable half-open-fours. At
odd lookahead depths, the searcher exploits this by building threats the
opponent never gets to refute before the eval.

The fix in `_negamax`'s depth==0 branch applies a 1-ply quiescence: if the
opponent (just-moved side) has any immediate-win-completable 4-in-a-row,
play the forced block before evaluating. That makes the leaf eval reflect
post-block reality.

This test pins the specific bug-trigger position and verifies the corrected
leaf eval differs sharply from the raw `evaluate_position`. If the test
ever fails, either the quiescence guard was removed/reordered, or the
static eval was changed in a way that the bug is no longer reproducible
here. Either way, investigate.
"""
import numpy as np

import pytest

from gomoku.baselines import _negamax, evaluate_position
from gomoku.game import GameState, str_to_action


def _position_with_b_live_four() -> GameState:
    """Build the position from the lookahead-bug investigation: black has
    a live-four buildup on column e (e4-e5-e6-e7), and white must block e8
    or lose next move."""
    state = GameState.initial()
    # Move sequence (B then W alternating): leaves B with e3-blocked + the
    # vertical buildup. After B plays e7 next, B has e4-e5-e6-e7 = 4 in a
    # row blocked at e3, open at e8 — an immediate winning threat.
    for m in ["e5", "f5", "f4", "d6", "e4", "d4", "e6", "e3"]:
        state = state.apply(str_to_action(m))
    # Now B plays e7 to complete the live-four buildup.
    return state.apply(str_to_action("e7"))


def test_depth_zero_quiescence_blocks_immediate_threat():
    """At a leaf where opp has an immediate winning move, `_negamax(state, 0)`
    must apply the forced block (or report we lose if it's a fork) instead
    of returning the raw `evaluate_position` value."""
    state = _position_with_b_live_four()  # W to move, B has open winning move at e8
    raw = evaluate_position(state)
    quiesce = _negamax(state, 0, -np.inf, np.inf)
    # The raw eval over-credits B's threat (very large negative for W).
    # Quiescence applies W's block at e8 and re-evaluates, giving a value
    # much closer to neutral. The exact value isn't important — what matters
    # is the magnitude of the correction.
    assert raw < -10_000, f"raw evaluate_position should be very negative, got {raw}"
    assert quiesce > raw + 10_000, (
        f"quiescence should correct upward by >10k Elo-units; "
        f"raw={raw:.0f}, quiesce={quiesce:.0f}"
    )


def test_depth_zero_quiescence_no_op_when_no_threat():
    """In a quiet position (no immediate-win threat), quiescence should
    pass through to the raw `evaluate_position` value."""
    state = GameState.initial().apply(str_to_action("e5"))
    raw = evaluate_position(state)
    quiesce = _negamax(state, 0, -np.inf, np.inf)
    # Both should agree on a quiet position.
    assert quiesce == pytest.approx(raw, abs=1e-6)


def test_lookahead_depth_3_beats_heuristic():
    """Empirical check that the fix actually changes depth=3's playing
    strength against heuristic. Pre-fix, heuristic beat depth=3 100% of
    games. Post-fix, depth=3 should win meaningfully against heuristic.
    """
    from gomoku.eval import play_match_pickers
    from gomoku.match import build_player, parse_spec

    p3 = build_player(parse_spec("lookahead:depth=3"))
    ph = build_player(parse_spec("heuristic"))
    res = play_match_pickers(p3, ph, n_games=10, seed=0)
    # Pre-fix: 0 wins out of any sample size. Post-fix: should win majority.
    # Threshold is loose to keep the test robust under RNG.
    assert res.wins >= 4, (
        f"depth=3 should beat heuristic ≥40% post-fix; "
        f"got {res.wins}W-{res.losses}L-{res.draws}D"
    )
