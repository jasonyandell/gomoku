"""Tests for the ``--swap2`` self-play wiring in ``gomoku.self_play``.

These formalize a known-good smoke of the swap2 self-play seam: a tiny CPU model
drives the swap2 negotiator as the opening oracle, then normal self-play runs
from the handed-off position. We assert the oracle contract, that swap2 self-play
generates records, that swap2 is mutually exclusive with the random-opening
prefix, that negotiated openings are valid + diverse, and that the default
(non-swap2) generation path is unbroken.

Everything is 'tiny' model + low sims on CPU so the whole module is fast; the
swap2 opening logic is board-size agnostic, so we use the default board size.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gomoku.board_config import BOARD_SIZE
from gomoku.game import GameState
from gomoku.mcts import make_torch_evaluator
from gomoku.model import build_model
from gomoku.swap2_search import negotiate
from gomoku import self_play as sp


# A single shared tiny CPU model + evaluator + oracle for the module. Build once
# (model construction dominates the runtime); the calls below are cheap.
_DEV = torch.device("cpu")
_MODEL = build_model("tiny").to(_DEV)
_MODEL.eval()
_EV = make_torch_evaluator(_MODEL, _DEV)
_ORACLE = sp._make_swap2_oracle(_EV)


def test_oracle_contract():
    """The swap2 oracle returns a proper (probs, value) over the initial state."""
    probs, value = _ORACLE(GameState.initial())
    probs = np.asarray(probs, dtype=np.float64)
    assert probs.shape == (BOARD_SIZE * BOARD_SIZE,)
    assert len(probs) == BOARD_SIZE**2
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs >= 0.0)
    assert -1.0 <= value <= 1.0


def test_swap2_selfplay_generates_records():
    """``generate_games(..., swap2=True)`` produces a non-empty record list."""
    records = sp.generate_games(
        6,
        _EV,
        n_simulations=8,
        rng=np.random.default_rng(0),
        swap2=True,
    )
    assert isinstance(records, list)
    assert len(records) > 0


def test_swap2_random_opening_mutually_exclusive():
    """swap2 negotiates the opening, so a random-opening prefix is rejected."""
    with pytest.raises(ValueError):
        sp.generate_games(
            1,
            _EV,
            n_simulations=4,
            rng=np.random.default_rng(0),
            swap2=True,
            random_opening_moves=3,
        )


def test_negotiated_openings_valid_and_diverse():
    """Negotiated openings are legal, non-terminal, 3-5 stones, and varied."""
    openings: set[tuple] = set()
    for seed in range(12):
        res = negotiate(_ORACLE, np.random.default_rng(seed))
        gs = res.normal_state
        n_stones = int(gs.move_count)
        assert 3 <= n_stones <= 5, f"opening had {n_stones} stones (expected 3-5)"
        # The handed-off position must be a legal, non-terminal start for play.
        done, _ = gs.is_terminal()
        assert not done, "negotiated opening handed off a terminal position"
        assert gs.legal_actions().size > 0
        # Canonical fingerprint of the absolute board (both planes) for diversity.
        openings.add((gs.board[0].tobytes(), gs.board[1].tobytes()))
    assert len(openings) >= 2, "negotiated openings were not diverse"


def test_swap2_off_is_default():
    """The default (no swap2) generation path still produces records."""
    records = sp.generate_games(
        2,
        _EV,
        n_simulations=4,
        rng=np.random.default_rng(0),
    )
    assert isinstance(records, list)
    assert len(records) > 0
