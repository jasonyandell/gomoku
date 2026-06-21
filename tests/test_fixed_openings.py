"""Tests for the fixed balanced opening-book self-play mode (``--fixed-openings``, #73).

The opening book is Rapfi's 9 balanced swap2 openings, placed directly (no
negotiation, no net, no choice head) so the net plays only post-opening from a
known-fair board. The coords are 15x15-specific, so run with the board size set
BEFORE pytest imports the package:

    GOMOKU_BOARD_SIZE=15 pytest tests/test_fixed_openings.py

In a default (9x9) run every test here is skipped (board size is a process-level
constant).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gomoku.board_config import BOARD_SIZE
from gomoku.mcts import make_torch_evaluator
from gomoku.model import build_model
from gomoku import self_play as sp

pytestmark = pytest.mark.skipif(
    BOARD_SIZE not in (9, 11, 13, 15),
    reason="fixed openings are defined for board sizes 9/11/13/15",
)


class _RngForcedIndex:
    """Minimal np.random.Generator stand-in: forces integers(0, n) -> a fixed k."""

    def __init__(self, k: int) -> None:
        self._k = k

    def integers(self, low, high=None):  # noqa: D401 - matches Generator.integers
        return self._k


# A single shared tiny CPU model + evaluator for the module (built once).
_DEV = torch.device("cpu")
_MODEL = build_model("tiny").to(_DEV)
_MODEL.eval()
_EV = make_torch_evaluator(_MODEL, _DEV)
_BOOK = sp._FAIR_OPENINGS.get(BOARD_SIZE, ())


def test_all_openings_2black_1white_white_to_move():
    """Each of the 9 openings: 2 black + 1 white, white to move, move_count=3, not terminal."""
    for k in range(len(_BOOK)):
        state, plies = sp._fixed_opening_state(_RngForcedIndex(k))
        assert plies == 3
        assert state.move_count == 3
        assert state.history == ()
        # plane 0 = side-to-move (white, the mover); plane 1 = opponent (black)
        assert int(state.board[0].sum()) == 1  # one white stone = mover
        assert int(state.board[1].sum()) == 2  # two black stones
        assert not state.is_terminal()[0]  # 3 scattered stones never win
        # stones land at the expected (x=col, y=row) cells with the right color
        opening = _BOOK[k]
        for i, (x, y) in enumerate(opening):
            plane = 1 if i % 2 == 0 else 0  # i=0,2 black (plane 1); i=1 white (plane 0)
            assert state.board[plane][y, x], f"opening {k} stone {i} at ({x},{y})"


def test_oracle_equivalence_to_swap2_to_normal():
    """Construction is byte-identical to swap2.OpeningState.to_normal() (SWAP outcome)."""
    from gomoku import swap2

    for k, opening in enumerate(_BOOK):
        state, _ = sp._fixed_opening_state(_RngForcedIndex(k))
        black = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        white = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        for i, (x, y) in enumerate(opening):
            (black if i % 2 == 0 else white)[y, x] = True
        ref = swap2._finish(black, white, opener_color=swap2.Color.BLACK).to_normal()
        assert np.array_equal(state.board, ref.board)
        assert state.move_count == ref.move_count


def test_generate_games_fixed_openings_no_swap2_no_choice(monkeypatch):
    """fixed_openings self-play makes records, never calls swap2, emits no choice examples."""

    def _boom(*a, **k):
        raise AssertionError("_swap2_opening_state must not be called for fixed_openings")

    monkeypatch.setattr(sp, "_swap2_opening_state", _boom)
    records = sp.generate_games(
        6, _EV, n_simulations=8, rng=np.random.default_rng(0), fixed_openings=True,
    )
    assert isinstance(records, list) and len(records) > 0
    for r in records:
        assert r.choice_examples == []


def test_fixed_openings_mutually_exclusive():
    """fixed_openings is rejected alongside swap2 or a random-opening prefix."""
    with pytest.raises(ValueError):
        sp.generate_games(1, _EV, n_simulations=4, rng=np.random.default_rng(0),
                          fixed_openings=True, swap2=True)
    with pytest.raises(ValueError):
        sp.generate_games(1, _EV, n_simulations=4, rng=np.random.default_rng(0),
                          fixed_openings=True, random_opening_moves=4)


def test_default_path_unbroken():
    """fixed_openings defaults OFF -> standard empty-board self-play still works."""
    records = sp.generate_games(2, _EV, n_simulations=4, rng=np.random.default_rng(1))
    assert isinstance(records, list) and len(records) > 0
