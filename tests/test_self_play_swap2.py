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
from gomoku.game import GameState, N_INPUT_PLANES
from gomoku.mcts import make_torch_evaluator
from gomoku.model import build_model
from gomoku.swap2 import Actor, N_CHOICES
from gomoku.swap2_search import ChoiceRecord, Swap2Result, negotiate
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


def test_swap2_records_choice_examples():
    """swap2 self-play threads negotiation CHOICE records out as ChoiceExamples.

    At least one game must carry a non-empty `choice_examples`, and each example
    must be well-formed: an N_CHOICES legal mask with 2-3 legal slots (responder
    {STAY,SWAP,PLACE2} or opener pick-color), a legal `chosen`, and a chooser_z in
    [-1, 1]. Non-swap2 records carry an empty list (byte-identical default).
    """
    records = sp.generate_games(
        12,
        _EV,
        n_simulations=8,
        rng=np.random.default_rng(0),
        swap2=True,
    )
    assert any(len(r.choice_examples) > 0 for r in records), (
        "expected at least one swap2 game to produce choice examples"
    )
    for r in records:
        for ce in r.choice_examples:
            assert ce.planes.shape == (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
            assert ce.legal_mask.shape == (N_CHOICES,)
            assert ce.legal_mask.dtype == bool
            n_legal = int(ce.legal_mask.sum())
            assert n_legal in (2, 3), f"legal slot count {n_legal} not in (2, 3)"
            assert 0 <= ce.chosen < N_CHOICES
            assert bool(ce.legal_mask[ce.chosen]), "chosen slot must be legal"
            assert -1.0 <= ce.chooser_z <= 1.0


def _synthetic_result(to_act: Actor, mover_actor: Actor) -> Swap2Result:
    """A minimal Swap2Result with ONE choice record at the given chooser node."""
    planes = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    legal = np.array([True, True, True], dtype=bool)
    cr = ChoiceRecord(
        planes=planes,
        to_act=to_act,
        legal_mask=legal,
        target=np.array([1.0, 0.0, 0.0]),
        chosen=0,
    )
    # normal_state/opener_color are unused by _choice_examples_for_game.
    return Swap2Result(
        normal_state=GameState.initial(),
        opener_color=None,
        mover_actor=mover_actor,
        choice_records=[cr],
    )


def test_choice_example_chooser_z_sign():
    """chooser_z = backup_sign(actor) * outcome_for_black.

    outcome_for_black is the outcome for the HAND-OFF MOVER (res.mover_actor). So
    when the chooser IS the mover (to_act == mover_actor) and the mover WON
    (outcome_for_black=+1), chooser_z must be > 0; the sign flips when EITHER the
    chooser is the OTHER actor OR the mover lost.
    """
    mover = Actor.OPENER
    other = Actor.RESPONDER

    # chooser == mover, mover won -> chooser_z > 0
    res = _synthetic_result(to_act=mover, mover_actor=mover)
    ce = sp._choice_examples_for_game(res, outcome_for_black=+1.0)[0]
    assert ce.chooser_z > 0

    # chooser == mover, mover LOST -> chooser_z < 0 (flips on outcome)
    ce = sp._choice_examples_for_game(res, outcome_for_black=-1.0)[0]
    assert ce.chooser_z < 0

    # chooser == OTHER actor, mover won -> chooser_z < 0 (flips on actor)
    res_other = _synthetic_result(to_act=other, mover_actor=mover)
    ce = sp._choice_examples_for_game(res_other, outcome_for_black=+1.0)[0]
    assert ce.chooser_z < 0

    # chooser == OTHER actor, mover lost -> chooser_z > 0 (double flip)
    ce = sp._choice_examples_for_game(res_other, outcome_for_black=-1.0)[0]
    assert ce.chooser_z > 0

    # No swap2 result -> no choice examples.
    assert sp._choice_examples_for_game(None, outcome_for_black=+1.0) == []
