"""Native lookahead parity (issue #110).

The C search (`gomoku._lookahead_native`) must be MOVE-IDENTICAL to the pure
Python `baselines._root_best_actions`: same tied-best action list, in the same
order, on the same positions. That holds because the port mirrors the stable
move ordering and every cut/tie decision, and all heuristic weights are
integer-valued doubles (exact arithmetic on both sides).

Skipped wholesale if the extension isn't built for the active board size.
"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku import baselines
from gomoku.baselines import (
    _root_best_actions,
    heuristic_player,
    lookahead_player,
    random_player,
)
from gomoku.game import GameState

native = pytest.importorskip("gomoku._lookahead_native")


def _positions(n, max_plies, seed):
    """Realistic non-terminal positions: mixed heuristic/random playouts."""
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        s = GameState.initial()
        plies = int(rng.integers(0, max_plies + 1))
        for _ in range(plies):
            if s.is_terminal()[0]:
                break
            picker = heuristic_player if rng.random() < 0.6 else random_player
            s = s.apply(picker(s, rng))
        if not s.is_terminal()[0]:
            out.append(s)
    return out


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_native_matches_python_tie_sets(depth):
    for s in _positions(12, 20, seed=100 * depth):
        py = _root_best_actions(s, depth)
        nat = native.best_actions(np.ascontiguousarray(s.board), depth, s.move_count)
        assert nat == py, f"depth={depth} move_count={s.move_count}"


def test_native_matches_python_on_forced_win_and_block():
    # A four-in-a-row for the side to move: both must take the win.
    s = GameState.initial()
    # black: 40, 41, 42, 43 / white: 0, 1, 2, 30 — black to move completes at 44 or 39
    for a in (40, 0, 41, 1, 42, 2, 43, 30):
        s = s.apply(a)
    py = _root_best_actions(s, 2)
    nat = native.best_actions(np.ascontiguousarray(s.board), 2, s.move_count)
    assert nat == py
    assert set(nat) <= {39, 44}


def test_lookahead_player_uses_native_and_matches_python_picks():
    if baselines._NATIVE_LOOKAHEAD is None:
        pytest.skip("native lookahead disabled in this environment")
    play = lookahead_player(depth=2)
    for s in _positions(8, 16, seed=999):
        pick_native = play(s, np.random.default_rng(5))
        py = _root_best_actions(s, 2)
        if len(py) == 1:
            pick_py = py[0]
        else:
            pick_py = int(np.random.default_rng(5).choice(np.asarray(py)))
        assert pick_native == pick_py
        assert s.legal_mask()[pick_native]


def test_native_deep_position_sanity():
    # Crowded midgame at depth 4 — the expensive derby-gate regime.
    for s in _positions(3, 30, seed=4242):
        py = _root_best_actions(s, 4)
        nat = native.best_actions(np.ascontiguousarray(s.board), 4, s.move_count)
        assert nat == py
