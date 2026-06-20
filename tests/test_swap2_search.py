"""Tests for the v1 swap2 NEGOTIATOR (``gomoku.swap2_search``).

All oracles here are deterministic FAKES — no real net, no GPU. The board size
is whatever the test process resolves (default 9); every assertion is size-
agnostic. The focus is value attribution (sign correctness per branch and
minimax in the PLACE2 line), value-maximizing choice selection, the recorded
ChoiceRecord shape/legality, determinism, and legal-square sampling.
"""

from __future__ import annotations

import numpy as np
import pytest

from gomoku.board_config import BOARD_SIZE, N_ACTIONS
from gomoku.game import GameState, N_INPUT_PLANES
from gomoku.swap2 import (
    Actor,
    Color,
    Phase,
    PICK_BLACK,
    PICK_WHITE,
    RESP_PLACE2,
    RESP_STAY,
    RESP_SWAP,
    backup_sign,
    initial_opening,
)
from gomoku.swap2_search import (
    ChoiceRecord,
    Swap2Result,
    _choice_values,
    agent_act,
    negotiate,
)

# ---------------------------------------------------------------------------
# Fake oracles.
# ---------------------------------------------------------------------------


def _uniform_policy() -> np.ndarray:
    return np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float64)


def const_oracle(value: float):
    """Oracle: uniform policy, a constant value regardless of position."""

    def _o(gs: GameState) -> tuple[np.ndarray, float]:
        return _uniform_policy(), float(value)

    return _o


def first_legal_policy_oracle(value_fn):
    """Oracle whose policy spikes the first legal (empty) square deterministically,
    and whose value comes from ``value_fn(gs)``. The spiked-policy makes greedy
    placement rollouts fully determined."""

    def _o(gs: GameState) -> tuple[np.ndarray, float]:
        empty = ~(gs.board[0] | gs.board[1])
        flat = empty.reshape(-1)
        p = np.zeros(N_ACTIONS, dtype=np.float64)
        idx = int(np.flatnonzero(flat)[0])
        p[idx] = 1.0
        return p, float(value_fn(gs))

    return _o


# ---------------------------------------------------------------------------
# Value attribution: sign correctness per branch (white-box on _choice_values).
# ---------------------------------------------------------------------------


def test_choice_value_sign_swap_and_stay():
    """For the responder's P2_CHOICE node, SWAP and STAY must carry the correct
    backup sign relative to the resulting normal position's mover."""
    rng = np.random.default_rng(0)
    V = 0.37  # the (known) normal-position value from side-to-move perspective.

    # Build a P2_CHOICE state by placing 3 stones greedily.
    s = initial_opening()
    while s.phase is Phase.PLACE and not s.is_done():
        s = s.apply(int(np.flatnonzero(s.legal_square_mask())[0]))
    assert s.node.name == "P2_CHOICE"
    cs = s

    oracle = first_legal_policy_oracle(lambda gs: V)
    values, legal_mask = _choice_values(oracle, cs)

    # Recompute the expected sign per branch from the public primitives.
    for slot in (RESP_STAY, RESP_SWAP, RESP_PLACE2):
        assert legal_mask[slot]
        # Resolve the branch greedily the same way the module does, then check
        # the sign equals backup_sign(cs, done.mover_actor()) * V.
        from gomoku.swap2_search import _resolve_greedy

        done = _resolve_greedy(oracle, cs.apply(slot))
        expected = backup_sign(cs, done.mover_actor()) * V
        assert values[slot] == pytest.approx(expected)

    # Concretely with a CONST oracle, every greedy-resolved branch ends with the
    # OPENER to move (SWAP -> 2B+1W white(opener) moves; STAY -> 2B+2W
    # black(opener) moves; PLACE2 -> opener picks WHITE -> white(opener) moves),
    # so backup_sign(RESPONDER, OPENER) = -1 in all three -> every value is -V.
    assert values[RESP_SWAP] == pytest.approx(-V)
    assert values[RESP_STAY] == pytest.approx(-V)
    assert values[RESP_PLACE2] == pytest.approx(-V)


def test_choice_value_sign_opener_pick_both_colors():
    """At the opener's P1_PICK node, PICK_BLACK -> responder moves (sign -1),
    PICK_WHITE -> opener moves (sign +1), so values are -V and +V."""
    V = 0.51

    # Drive to P1_PICK: place 3, choose PLACE2, place 2.
    s = initial_opening()
    while s.node.name != "P2_CHOICE":
        s = s.apply(int(np.flatnonzero(s.legal_square_mask())[0]))
    s = s.apply(RESP_PLACE2)
    while s.node.name != "P1_PICK":
        s = s.apply(int(np.flatnonzero(s.legal_square_mask())[0]))
    assert s.node.name == "P1_PICK"
    pick = s

    oracle = first_legal_policy_oracle(lambda gs: V)
    values, legal_mask = _choice_values(oracle, pick)

    for slot in (PICK_BLACK, PICK_WHITE):
        from gomoku.swap2_search import _resolve_greedy

        done = _resolve_greedy(oracle, pick.apply(slot))
        expected = backup_sign(pick, done.mover_actor()) * V
        assert values[slot] == pytest.approx(expected)

    assert values[PICK_BLACK] == pytest.approx(-V)  # opener_color BLACK -> white(=responder) moves
    assert values[PICK_WHITE] == pytest.approx(+V)  # opener_color WHITE -> opener moves


# ---------------------------------------------------------------------------
# The searcher picks the value-maximizing choice.
# ---------------------------------------------------------------------------


def test_negotiate_picks_value_maximizing_branch_swap():
    """Make the SWAP outcome (total 3 stones, opener-to-move) the worst position
    from its side-to-move perspective. Because the responder re-signs by -1 (the
    opener moves in every greedy-resolved branch), a low v_norm for SWAP is the
    HIGH responder value -> greedy choice = SWAP, opener takes WHITE, 3 stones."""

    def value_fn(gs: GameState) -> float:
        # Total stones uniquely identifies the branch: 3=SWAP, 4=STAY, 5=PLACE2.
        total = int(gs.board.sum())
        return -1.0 if total == 3 else +1.0

    oracle = first_legal_policy_oracle(value_fn)
    rng = np.random.default_rng(1)
    res = negotiate(oracle, rng, greedy_choices=True, temperature=0.0)

    # SWAP -> opener plays WHITE the whole game, no new stone (2B+1W = 3).
    assert res.opener_color == Color.WHITE
    assert int(res.normal_state.board.sum()) == 3
    # The recorded responder choice was SWAP.
    assert res.choice_records[0].chosen == RESP_SWAP


def test_negotiate_picks_stay_when_its_outcome_is_responder_best():
    """Mirror: make the STAY outcome (4 stones, opener-to-move) the worst from its
    own side-to-move perspective. STAY is fully responder-controlled (no nested
    opener pick), so a low v_norm for total-4 -> the HIGH responder value -> the
    responder chooses STAY, board has 4 stones (2B+2W), opener plays BLACK."""

    def value_fn(gs: GameState) -> float:
        total = int(gs.board.sum())
        return -1.0 if total == 4 else +1.0

    oracle = first_legal_policy_oracle(value_fn)
    rng = np.random.default_rng(2)
    res = negotiate(oracle, rng, greedy_choices=True, temperature=0.0)

    # STAY adds one white stone: 2B + 2W = 4 stones, black(=opener)-to-move.
    assert res.opener_color == Color.BLACK
    assert int(res.normal_state.board.sum()) == 4
    assert res.choice_records[0].chosen == RESP_STAY


# ---------------------------------------------------------------------------
# Minimax in PLACE2: the opener's adversarial pick is reflected in the
# responder's PLACE2 value.
# ---------------------------------------------------------------------------


def test_place2_value_reflects_opener_best_pick():
    """In the PLACE2 line the opener picks a color AFTER the responder. The
    responder's value for PLACE2 must reflect the opener's BEST (adversarial)
    pick, not the responder-favorable one.

    We make the opener's two picks asymmetric in the chooser's own frame:
    construct an oracle whose normal-position value depends on opener_color so
    that one pick is great for the opener (so the opener takes it) and that same
    outcome is bad for the responder. The responder's PLACE2 value must therefore
    be the LOW (opener-favorable) value, not the high one.
    """
    # Drive to P1_PICK to learn the two resulting normal states' identities.
    s = initial_opening()
    while s.node.name != "P2_CHOICE":
        s = s.apply(int(np.flatnonzero(s.legal_square_mask())[0]))
    cs_choice = s  # the responder's P2_CHOICE node
    s2 = s.apply(RESP_PLACE2)
    while s2.node.name != "P1_PICK":
        s2 = s2.apply(int(np.flatnonzero(s2.legal_square_mask())[0]))
    pick = s2

    # Both picks leave white-to-move, identical board; they differ only in
    # opener_color (BLACK vs WHITE) -> different mover_actor:
    #   PICK_BLACK -> mover RESPONDER ; PICK_WHITE -> mover OPENER.
    # Value from side-to-move perspective is V_NORM in both (board identical),
    # but backup_sign re-signs to each node's chooser.
    V_NORM = 0.6

    oracle = first_legal_policy_oracle(lambda gs: V_NORM)

    # At the opener's PICK node:
    #   PICK_BLACK: opener value = backup_sign(OPENER, RESPONDER)*V_NORM = -V_NORM
    #   PICK_WHITE: opener value = backup_sign(OPENER, OPENER)*V_NORM   = +V_NORM
    # Opener maximizes -> picks WHITE (value +V_NORM). After WHITE, mover=OPENER.
    pick_vals, _ = _choice_values(oracle, pick)
    assert pick_vals[PICK_WHITE] == pytest.approx(+V_NORM)
    assert pick_vals[PICK_BLACK] == pytest.approx(-V_NORM)

    # Now the responder's PLACE2 value: resolve greedily -> opener picks WHITE ->
    # done.mover_actor() == OPENER. Responder value =
    #   backup_sign(RESPONDER, OPENER) * V_NORM = -V_NORM  (the opener-best, which
    #   is responder-bad). NOT +V_NORM (which it would be if the opener picked
    #   the responder-favorable BLACK).
    resp_vals, _ = _choice_values(oracle, cs_choice)
    assert resp_vals[RESP_PLACE2] == pytest.approx(-V_NORM)
    # Sanity: if minimax had (wrongly) let the responder pick BLACK, PLACE2 would
    # be +V_NORM.
    assert resp_vals[RESP_PLACE2] != pytest.approx(+V_NORM)


# ---------------------------------------------------------------------------
# negotiate() output: legal DONE-derived normal_state + well-formed records.
# ---------------------------------------------------------------------------


def test_negotiate_returns_legal_nonterminal_normal_state():
    oracle = const_oracle(0.0)
    rng = np.random.default_rng(7)
    res = negotiate(oracle, rng)

    assert isinstance(res, Swap2Result)
    gs = res.normal_state
    # Not terminal, has legal moves to continue.
    done, _ = gs.is_terminal()
    assert not done
    assert gs.legal_actions().size > 0
    # Plane shape sanity via to_planes.
    assert gs.to_planes().shape == (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    # opener/mover consistency.
    assert res.opener_color in (Color.BLACK, Color.WHITE)
    assert res.mover_actor in (Actor.OPENER, Actor.RESPONDER)


def test_choice_records_well_formed():
    oracle = const_oracle(0.1)
    rng = np.random.default_rng(11)
    res = negotiate(oracle, rng, greedy_choices=True)

    assert len(res.choice_records) >= 1  # P2_CHOICE always happens.
    for rec in res.choice_records:
        assert isinstance(rec, ChoiceRecord)
        assert rec.planes.shape == (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
        assert rec.legal_mask.shape == (3,)
        assert rec.target.shape == (3,)
        # target sums to 1 over legal slots, illegal slots are exactly 0.
        assert rec.target.sum() == pytest.approx(1.0)
        assert np.all(rec.target[~rec.legal_mask] == 0.0)
        assert rec.legal_mask[rec.chosen]
        # chosen is the argmax (greedy_choices=True) of the target over legal.
        assert rec.chosen == int(np.argmax(np.where(rec.legal_mask, rec.target, -np.inf)))


def test_choice_target_softmax_over_legal_only():
    """A nonzero choice_tau gives a soft target; illegal slots stay 0 and the
    legal slots are an honest softmax of the per-slot values."""
    oracle = const_oracle(0.0)  # all branches value 0 -> uniform softmax.
    rng = np.random.default_rng(3)
    res = negotiate(oracle, rng, choice_tau=1.0, greedy_choices=True)

    # The first record is the responder's 3-way choice: all values 0 (const
    # oracle) -> uniform over the 3 legal slots.
    first = res.choice_records[0]
    assert np.all(first.legal_mask)  # all 3 legal at P2_CHOICE
    assert np.allclose(first.target, 1.0 / 3.0)


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


def test_determinism_greedy_choices_temp0():
    oracle = first_legal_policy_oracle(lambda gs: 0.2)

    def run():
        rng = np.random.default_rng(123)
        return negotiate(oracle, rng, greedy_choices=True, temperature=0.0)

    a = run()
    b = run()
    assert a.opener_color == b.opener_color
    assert a.mover_actor == b.mover_actor
    assert np.array_equal(a.normal_state.board, b.normal_state.board)
    assert len(a.choice_records) == len(b.choice_records)
    for ra, rb in zip(a.choice_records, b.choice_records):
        assert ra.chosen == rb.chosen
        assert np.array_equal(ra.target, rb.target)


# ---------------------------------------------------------------------------
# Sampled placements are always legal/empty squares.
# ---------------------------------------------------------------------------


def test_sampled_placements_always_legal():
    """Across many seeds with a uniform-policy oracle and temperature, every
    placement lands on an empty square (no overlap, correct stone counts)."""
    oracle = const_oracle(0.0)
    for seed in range(25):
        rng = np.random.default_rng(seed)
        res = negotiate(oracle, rng, temperature=1.0)
        gs = res.normal_state
        # No square holds two stones.
        assert not np.any(gs.board[0] & gs.board[1])
        # Stone count is one of the legal swap2 totals: SWAP=3, STAY=4, PLACE2=5.
        total = int(gs.board.sum())
        assert total in (3, 4, 5)


def test_agent_act_single_node_place_and_choice():
    """agent_act handles one node at a time: PLACE -> (square, None) legal;
    CHOICE -> (slot, ChoiceRecord)."""
    oracle = const_oracle(0.0)
    rng = np.random.default_rng(5)

    s = initial_opening()
    assert s.phase is Phase.PLACE
    action, rec = agent_act(oracle, s, rng)
    assert rec is None
    assert s.legal_square_mask()[action]  # a legal empty square

    # Drive to a CHOICE node and check the record path.
    while s.phase is Phase.PLACE:
        s = s.apply(int(np.flatnonzero(s.legal_square_mask())[0]))
    assert s.phase is Phase.CHOICE
    slot, rec = agent_act(oracle, s, rng, greedy=True)
    assert rec is not None
    assert s.legal_choice_mask()[slot]
    assert rec.chosen == slot

    # DONE node raises.
    while not s.is_done():
        if s.phase is Phase.PLACE:
            s = s.apply(int(np.flatnonzero(s.legal_square_mask())[0]))
        else:
            s = s.apply(int(np.flatnonzero(s.legal_choice_mask())[0]))
    with pytest.raises(ValueError):
        agent_act(oracle, s, rng)
