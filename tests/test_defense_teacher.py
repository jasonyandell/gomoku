"""Correctness tests for the DEFENSIVE VCF teacher (gomoku/self_play._apply_defense_teacher).

The defensive teacher is the value-only mirror of the offensive VCF teacher: it
relabels a recorded position's value target to -1.0 when the OPPONENT has a
proven forced VCF win against the side to move (detected by solving the SAME
position with the two planes SWAPPED). A FALSE POSITIVE here poisons value
targets the WRONG way (labels a non-lost position as lost), so the headline test
is `test_no_false_positive_*`: over 400 random positions the defensive detector
must have ZERO false positives versus an independent referee.

The detector reuses `vcf.solve_vcf` verbatim on the swapped board; the referee
below is independent of that solver (it replays the claimed line move-by-move
using `state_ops.has_five_in_a_row`, asserting each attacker move is really a
four and the defender never has an instant five), mirroring `tests/test_vcf.py`.
"""

from __future__ import annotations

import numpy as np

from gomoku import state_ops, vcf
from gomoku.game import HISTORY_PLY, N_INPUT_PLANES
from gomoku.self_play import _apply_defense_teacher

N = state_ops.BOARD_SIZE


def make_planes(me_cells, opp_cells):
    """Build an (N_INPUT_PLANES, N, N) float32 plane stack with the side-to-move's
    stones at plane 0 and the opponent's stones at plane HISTORY_PLY (the layout
    `GameState.to_planes` emits and that `_apply_defense_teacher` reads)."""
    planes = np.zeros((N_INPUT_PLANES, N, N), dtype=np.float32)
    for r, c in me_cells:
        planes[0, r, c] = 1.0
    for r, c in opp_cells:
        planes[HISTORY_PLY, r, c] = 1.0
    return planes


def _swapped_board(planes):
    """The (2, N, N) board the defensive detector solves: opponent = attacker."""
    opp = planes[HISTORY_PLY].astype(bool)
    me = planes[0].astype(bool)
    return np.stack([opp, me], axis=0)


def _verify_attacker_line(board):
    """Independent referee: assert plane-0 of `board` (the opponent, as attacker)
    really has a forced continuous-four win. Replays the solver's claimed line
    using `state_ops.has_five_in_a_row` as a separate truth oracle. Returns True
    if the line is confirmed to reach five; raises on any inconsistency.

    Mirrors `tests/test_vcf._verify_line` but on the (swapped) board the
    defensive detector hands to `solve_vcf`.
    """
    res = vcf.solve_vcf(board)
    assert res.has_forced_win
    attacker = board[0].copy()
    defender = board[1].copy()
    move = res.winning_move
    if move is None:
        # depth-0 already-won case; attacker already has five.
        assert state_ops.has_five_in_a_row(attacker)
        return True
    for _ in range(40):  # generous safety bound on line length
        attacker[move // N, move % N] = True
        if state_ops.has_five_in_a_row(attacker):
            return True  # mate reached
        empty = ~(attacker | defender)
        comps = []
        for idx in np.flatnonzero(empty.reshape(-1)):
            test = attacker.copy()
            test[idx // N, idx % N] = True
            if state_ops.has_five_in_a_row(test):
                comps.append(int(idx))
        assert comps, "attacker move was not actually a four"
        for idx in np.flatnonzero(empty.reshape(-1)):
            test = defender.copy()
            test[idx // N, idx % N] = True
            assert not state_ops.has_five_in_a_row(test), \
                "defender had an immediate five — line is unsound"
        block = comps[0]
        defender[block // N, block % N] = True
        sub = vcf.solve_vcf(np.stack([attacker, defender], axis=0))
        assert sub.has_forced_win, "line broke: no continuation found"
        move = sub.winning_move
        if move is None:
            assert state_ops.has_five_in_a_row(attacker)
            return True
    raise AssertionError("line did not terminate")


# ---------------------------------------------------------------------------
# Positive sanity: a proven OPPONENT win relabels z = -1.0
# ---------------------------------------------------------------------------

def test_opponent_open_four_relabels_loss():
    # The OPPONENT (plane HISTORY_PLY) has an open four .XXXX. -> a proven forced
    # win against the side to move. The defensive teacher must relabel z to -1.0.
    planes = make_planes(
        me_cells=[(8, 8)],
        opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)],
    )
    # Referee agrees the opponent has a forced win.
    assert _verify_attacker_line(_swapped_board(planes))
    new_z, fired = _apply_defense_teacher(planes, 0.5, vcf_already_fired=False)
    assert fired is True
    assert new_z == -1.0


def test_quiet_position_does_not_fire():
    # A sparse position with no opponent four-making move: the cheap pre-scan must
    # short-circuit (no fire), z unchanged.
    planes = make_planes(me_cells=[(4, 4)], opp_cells=[(0, 0), (8, 8)])
    z_in = 0.3
    new_z, fired = _apply_defense_teacher(planes, z_in, vcf_already_fired=False)
    assert fired is False
    assert new_z == z_in


def test_skips_when_vcf_already_fired():
    # Gen-cost gate (a): if the offensive VCF teacher already fired (side to move
    # has a PROVEN win), the defensive solve is skipped entirely — even on a board
    # that WOULD otherwise look dangerous.
    planes = make_planes(
        me_cells=[(8, 8)],
        opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)],
    )
    z_in = 1.0
    new_z, fired = _apply_defense_teacher(planes, z_in, vcf_already_fired=True)
    assert fired is False
    assert new_z == z_in


# ---------------------------------------------------------------------------
# THE CRITICAL TEST: zero false positives over 400 random positions
# ---------------------------------------------------------------------------

def test_no_false_positive_random_positions():
    # Fuzz: random legal positions where neither side has five. Whenever the
    # defensive detector FIRES (claims the opponent has a forced win => z relabeled
    # to -1.0), the independent referee MUST confirm the opponent really has a
    # forced continuous-four win to five. ZERO false positives is the blocking gate
    # (a false positive labels a non-lost position as lost — poisoning value
    # targets the wrong way).
    rng = np.random.default_rng(20260527)
    n_fired = 0
    n_examined = 0
    for _ in range(400):
        n_stones = int(rng.integers(2, 16))
        cells = rng.choice(N * N, size=n_stones, replace=False)
        me_cells = [(int(x // N), int(x % N)) for x in cells[0::2]]
        opp_cells = [(int(x // N), int(x % N)) for x in cells[1::2]]
        planes = make_planes(me_cells, opp_cells)
        board = _swapped_board(planes)
        # Skip positions where either side already has five (not a legal
        # side-to-move position; the solver guards these but they aren't fuzz
        # targets).
        if state_ops.has_five_in_a_row(board[0]) or state_ops.has_five_in_a_row(board[1]):
            continue
        n_examined += 1
        new_z, fired = _apply_defense_teacher(planes, 0.0, vcf_already_fired=False)
        if fired:
            # Detector claims a proven opponent win -> referee must confirm it,
            # and the value must be the loss vertex.
            assert new_z == -1.0
            assert _verify_attacker_line(board), \
                "DEFENSIVE FALSE POSITIVE: detector fired but referee found no win"
            n_fired += 1
        else:
            # When it does NOT fire, z must be untouched.
            assert new_z == 0.0
    # Informational: the fuzz should have examined real positions; some seeds may
    # legitimately yield zero forced-win fires.
    assert n_examined > 0


# ---------------------------------------------------------------------------
# DEFAULT-OFF / byte-identical: the gate is off by default and never solves
# ---------------------------------------------------------------------------

def test_default_off_signature_and_no_solver_call(monkeypatch):
    # The `defense_teacher` flag defaults to False on every generator entry point,
    # so the default-off path never reaches `_apply_defense_teacher`. Independently,
    # prove the gate itself never invokes the solver when it would be a no-op: if
    # the offensive teacher already fired, NO solver/pre-scan call is made (so the
    # generation is bit-for-bit unchanged on a proven-win position).
    calls = {"prescan": 0, "solve": 0}
    real_prescan = vcf.has_four_threat
    real_solve = vcf.solve_vcf

    def counting_prescan(board):
        calls["prescan"] += 1
        return real_prescan(board)

    def counting_solve(board, **kw):
        calls["solve"] += 1
        return real_solve(board, **kw)

    monkeypatch.setattr(vcf, "has_four_threat", counting_prescan)
    monkeypatch.setattr(vcf, "solve_vcf", counting_solve)

    planes = make_planes(
        me_cells=[(8, 8)],
        opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)],
    )
    # vcf_already_fired=True => the gate short-circuits before any solver work.
    _apply_defense_teacher(planes, 1.0, vcf_already_fired=True)
    assert calls["prescan"] == 0
    assert calls["solve"] == 0


def test_generate_games_defense_teacher_defaults_off():
    # The public generator signatures must default `defense_teacher` to False so
    # the production default path is byte-identical (no relabel, no solver).
    import inspect
    from gomoku import self_play

    for name in (
        "generate_games",
        "generate_games_vs_baseline",
        "_generate_games_native",
        "_generate_games_gumbel",
        "_generate_games_native_gumbel",
    ):
        fn = getattr(self_play, name)
        sig = inspect.signature(fn)
        assert "defense_teacher" in sig.parameters, f"{name} missing defense_teacher"
        assert sig.parameters["defense_teacher"].default is False, \
            f"{name}.defense_teacher default is not False"
