"""Tests for balanced (random-opening) eval mode.

Verified invariants:
  (a) random_opening_moves=0 reproduces current behavior (empty-board start).
  (b) random_opening_moves=K starts each game with K stones pre-placed.
  (c) Both games in a color-swap pair start from the SAME opening position
      (fairness guarantee — neither picker benefits from the draw).
  (d) Aggregate bookkeeping invariants still hold with openings enabled.
"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku.eval import play_match_pickers
from gomoku.game import GameState


# ---------------------------------------------------------------------------
# Simple deterministic pickers for testing (no model / GPU).
# ---------------------------------------------------------------------------

def _fixed_picker(action: int):
    """Always plays ``action`` if legal, else the first legal move."""
    def pick(state: GameState, rng: np.random.Generator) -> int:
        legal = state.legal_actions()
        if action in legal:
            return int(action)
        return int(legal[0])
    return pick


class _OpeningCapture:
    """Wraps a picker and records the GameState at the moment move_count
    equals ``opening_moves`` — i.e., the first call each game where the
    picker must move FROM the post-opening position."""

    def __init__(self, inner, opening_moves: int):
        self.inner = inner
        self.opening_moves = opening_moves
        self.captured: list[GameState] = []
        self._captured_this_game = False
        self._prev_move_count = -1

    def __call__(self, state: GameState, rng: np.random.Generator) -> int:
        # Detect start of a new game: move_count went backwards (or to the
        # expected opening ply from a higher value on the previous call).
        if state.move_count <= self._prev_move_count or state.move_count == self.opening_moves:
            if state.move_count == self.opening_moves and not self._captured_this_game:
                self.captured.append(state)
                self._captured_this_game = True
            if state.move_count < self._prev_move_count:
                # New game started.
                self._captured_this_game = False
                if state.move_count == self.opening_moves:
                    self.captured.append(state)
                    self._captured_this_game = True
        elif state.move_count == self.opening_moves and not self._captured_this_game:
            self.captured.append(state)
            self._captured_this_game = True
        self._prev_move_count = state.move_count
        return self.inner(state, rng)


def _make_capturing_pickers(inner_a, inner_b, opening_moves: int):
    """Return (picker_a, picker_b, per_game_opening_states) where
    per_game_opening_states is a list that is populated as games are played.

    Each picker records every state it sees at move_count==opening_moves.
    By collecting from BOTH pickers we get one entry per game regardless of
    which side plays first from the opening position.
    """
    per_game: list[GameState] = []

    def make(inner):
        captured_game_ids: set[int] = set()
        prev_mc = [-1]
        game_id = [0]

        def pick(state: GameState, rng: np.random.Generator) -> int:
            # Detect a new game when move_count jumps back down.
            if state.move_count < prev_mc[0]:
                game_id[0] += 1
            prev_mc[0] = state.move_count

            gid = game_id[0]
            if state.move_count == opening_moves and gid not in captured_game_ids:
                captured_game_ids.add(gid)
                per_game.append(state)
            return inner(state, rng)

        return pick

    return make(inner_a), make(inner_b), per_game


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_zero_opening_moves_empty_board():
    """(a) random_opening_moves=0 gives each game an empty initial board."""
    first_states: list[GameState] = []
    prev_mc = [-1]
    game_seen = [False]

    def recording_a(state: GameState, rng: np.random.Generator) -> int:
        if state.move_count < prev_mc[0]:
            # New game
            game_seen[0] = False
        if state.move_count == 0 and not game_seen[0]:
            first_states.append(state)
            game_seen[0] = True
        prev_mc[0] = state.move_count
        legal = state.legal_actions()
        return int(legal[0])

    play_match_pickers(recording_a, _fixed_picker(0), n_games=4, seed=7,
                       random_opening_moves=0)
    assert all(s.move_count == 0 for s in first_states), (
        "With random_opening_moves=0 every game must start from an empty board."
    )
    # picker_a plays first in even games (g_idx=0,2); 2 of 4 games.
    assert len(first_states) == 2


def test_k_opening_moves_places_k_stones():
    """(b) With random_opening_moves=K, the picker first receives a state with
    K stones already placed."""
    K = 4
    pa, pb, opening_states = _make_capturing_pickers(
        _fixed_picker(112), _fixed_picker(0), opening_moves=K
    )
    play_match_pickers(pa, pb, n_games=6, seed=42, random_opening_moves=K)

    assert len(opening_states) == 6, (
        f"Expected one opening capture per game (6), got {len(opening_states)}"
    )
    for i, s in enumerate(opening_states):
        assert s.move_count == K, (
            f"Game {i}: expected {K} stones pre-placed, got move_count={s.move_count}"
        )


def test_color_swap_pair_shares_opening():
    """(c) Both games in a color-swap pair (g_idx=2k, 2k+1) start from the
    SAME opening position — the board content must be byte-identical."""
    K = 4
    pa, pb, opening_states = _make_capturing_pickers(
        _fixed_picker(112), _fixed_picker(0), opening_moves=K
    )
    play_match_pickers(
        pa, pb,
        n_games=6,   # 3 pairs: (0,1), (2,3), (4,5)
        seed=99,
        random_opening_moves=K,
    )

    assert len(opening_states) == 6, (
        f"Expected 6 opening captures (one per game), got {len(opening_states)}"
    )

    # Within each pair the boards must be identical.
    for pair in range(3):
        board_even = opening_states[pair * 2].board
        board_odd  = opening_states[pair * 2 + 1].board
        np.testing.assert_array_equal(
            board_even, board_odd,
            err_msg=(
                f"Pair {pair}: games {pair*2} and {pair*2+1} started from "
                "different opening positions — fairness violated."
            ),
        )

    # Across pairs the openings should differ (high probability for K>=4).
    pair_boards = [opening_states[p * 2].board for p in range(3)]
    n_distinct = len({b.tobytes() for b in pair_boards})
    assert n_distinct > 1, (
        "All pairs got the same opening — RNG may not be advancing between pairs."
    )


def test_aggregate_bookkeeping_with_openings():
    """(d) w+l+d == n_games and per-color split sums are correct with openings."""
    pa = _fixed_picker(112)
    pb = _fixed_picker(0)
    res = play_match_pickers(pa, pb, n_games=8, seed=5, random_opening_moves=4)

    assert res.n_games == 8
    assert res.wins + res.losses + res.draws == res.n_games
    assert res.black_w + res.white_w == res.wins
    assert res.black_l + res.white_l == res.losses
    assert res.black_d + res.white_d == res.draws
    black_games = res.black_w + res.black_l + res.black_d
    white_games = res.white_w + res.white_l + res.white_d
    assert black_games == 4
    assert white_games == 4


def test_deterministic_given_seed():
    """Same seed + same opening_moves must produce identical results."""
    pa = _fixed_picker(112)
    pb = _fixed_picker(0)
    r1 = play_match_pickers(pa, pb, n_games=6, seed=13, random_opening_moves=3)
    r2 = play_match_pickers(pa, pb, n_games=6, seed=13, random_opening_moves=3)
    assert (r1.wins, r1.losses, r1.draws) == (r2.wins, r2.losses, r2.draws)
