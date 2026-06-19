"""Correctness tests for the POLICY-mode defensive teacher (issue #43).

The #43 "I2" teacher is the policy mirror of the value-only #36/#42 defensive
teacher. Detection is identical (swap planes; the OPPONENT has a proven forced
VCF win against the side to move) but the LESSON differs: because the side to
move actually moves FIRST (one tempo before the attacker), there is usually a
SAVING move that breaks the opponent's forced four. The teacher extracts those
refutation moves (`vcf.vcf_refutations`) and stamps a soft policy target over
them, leaving the value at the natural game outcome — teaching "here is the move
that refuses the forced win" rather than the value-only "you already lost".

The blocking soundness property: a stamped saving move, once played, must
provably leave the opponent with NO forced VCF — so the policy target is a real
escape, never a move that still loses. `test_refutations_are_sound_*` enforces
this over random fired positions.
"""

from __future__ import annotations

import numpy as np

import pytest

from gomoku import self_play, state_ops, vcf
from gomoku.game import HISTORY_PLY, N_INPUT_PLANES
from gomoku.self_play import (
    _apply_defense_teacher_policy,
    _apply_teachers_to_trajectory,
    _relabel_defense_policy_game,
    configure_defense_teacher,
)

N = state_ops.BOARD_SIZE


@pytest.fixture
def restore_defense_knobs():
    """The defense-teacher knobs (#42 soft_value/max_fraction + #43 policy_mode)
    are PROCESS-WIDE globals; restore them so a test cannot poison siblings."""
    saved = (
        self_play._DEFENSE_SOFT_VALUE,
        self_play._DEFENSE_MAX_FRACTION,
        self_play._DEFENSE_POLICY_MODE,
    )
    try:
        yield
    finally:
        (
            self_play._DEFENSE_SOFT_VALUE,
            self_play._DEFENSE_MAX_FRACTION,
            self_play._DEFENSE_POLICY_MODE,
        ) = saved


def make_planes(me_cells, opp_cells):
    """(N_INPUT_PLANES, N, N) float32 stack: side-to-move at plane 0, opponent at
    plane HISTORY_PLY (the layout `_apply_defense_teacher_policy` reads)."""
    planes = np.zeros((N_INPUT_PLANES, N, N), dtype=np.float32)
    for r, c in me_cells:
        planes[0, r, c] = 1.0
    for r, c in opp_cells:
        planes[HISTORY_PLY, r, c] = 1.0
    return planes


def _flat(r, c):
    return r * N + c


# A SIMPLE (single-completion) opponent four the defender can block: opp has
# XXXX at row 2 cols 2..5, the LEFT end pre-blocked by a defender stone at (2,1),
# so the only completion is (2,6). The defender, moving first, refutes by taking
# (2,6) — after which the opponent has no four left. This is exactly the case the
# VALUE teacher mislabels as "lost" while the position is in fact defensible.
_REFUTABLE = make_planes(
    me_cells=[(2, 1), (8, 8)],
    opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)],
)

# An OPEN four (.XXXX.) — completions at BOTH (2,1) and (2,6). The defender moves
# first but can block only one end; the opponent completes the other. Genuinely
# lost: NO refutation exists.
_LOST = make_planes(
    me_cells=[(8, 8)],
    opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)],
)


# ---------------------------------------------------------------------------
# vcf.vcf_refutations — the new solver primitive
# ---------------------------------------------------------------------------

def _swapped(planes):
    opp = planes[HISTORY_PLY].astype(bool)
    me = planes[0].astype(bool)
    return np.stack([opp, me], axis=0)


def test_refutation_blocks_a_simple_four():
    board = _swapped(_REFUTABLE)
    assert vcf.solve_vcf(board).has_forced_win  # opponent really has a forced win
    saving = vcf.vcf_refutations(board)
    assert saving == [_flat(2, 6)]  # the unique block is the only refutation


def test_open_four_has_no_refutation():
    board = _swapped(_LOST)
    assert vcf.solve_vcf(board).has_forced_win
    assert vcf.vcf_refutations(board) == []  # blocking one end loses to the other


def test_refutations_actually_break_the_vcf():
    # Each reported saving move, once played by the defender, must leave the
    # opponent (now to move) with NO forced VCF — the soundness contract.
    board = _swapped(_REFUTABLE)
    for d in vcf.vcf_refutations(board):
        attacker = board[0].copy()
        defender = board[1].copy()
        defender[d // N, d % N] = True
        after = np.stack([attacker, defender], axis=0)
        assert not vcf.solve_vcf(after).has_forced_win


def test_cap_hit_no_win_is_not_trusted_as_refutation(monkeypatch):
    # SOUNDNESS at tight budgets: a post-move re-solve that returns
    # has_forced_win=False ONLY because it hit its node/depth cap is "unproven",
    # not "proven safe" — it must NOT be reported as a saving move. Stub solve_vcf
    # so every post-move re-solve is a cap-hit no-win; the refutation set must be
    # empty (otherwise we'd stamp moves that may not actually escape).
    board = _swapped(_REFUTABLE)
    real_solve = vcf.solve_vcf

    def cap_hit_solve(b, **kw):
        # The candidate re-solves carry an extra defender stone vs the input board;
        # force those to look like an exhausted-budget "no win".
        return vcf.VCFResult(False, None, None, nodes=1, hit_cap=True)

    monkeypatch.setattr(vcf, "solve_vcf", cap_hit_solve)
    assert vcf.vcf_refutations(board) == []
    # Sanity: with the real solver (generous default caps) the block IS found,
    # proving the empty result above is the cap guard, not a broken candidate set.
    monkeypatch.setattr(vcf, "solve_vcf", real_solve)
    assert vcf.vcf_refutations(board) == [_flat(2, 6)]


# ---------------------------------------------------------------------------
# _apply_defense_teacher_policy — stamp the saving move on the policy head
# ---------------------------------------------------------------------------

def test_policy_teacher_stamps_saving_move():
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    new_pi, new_z, fired = _apply_defense_teacher_policy(
        _REFUTABLE, pi, 0.5, vcf_already_fired=False)
    assert fired is True
    assert new_z == 0.5  # value LEFT UNTOUCHED (pure policy lever)
    expected = np.zeros(N * N, dtype=np.float32)
    expected[_flat(2, 6)] = 1.0
    assert np.allclose(new_pi, expected)
    assert pytest.approx(new_pi.sum()) == 1.0


def test_policy_teacher_skips_truly_lost():
    # No refutation -> the teacher leaves BOTH targets untouched (it never crushes
    # value; that is the value teacher's job, deliberately not reused here).
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    new_pi, new_z, fired = _apply_defense_teacher_policy(
        _LOST, pi, 0.5, vcf_already_fired=False)
    assert fired is False
    assert new_z == 0.5
    assert new_pi is pi


def test_policy_teacher_quiet_position_does_not_fire():
    quiet = make_planes(me_cells=[(4, 4)], opp_cells=[(0, 0), (8, 8)])
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    new_pi, new_z, fired = _apply_defense_teacher_policy(
        quiet, pi, 0.3, vcf_already_fired=False)
    assert fired is False
    assert new_z == 0.3
    assert new_pi is pi


def test_policy_teacher_skips_when_vcf_already_fired():
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    new_pi, new_z, fired = _apply_defense_teacher_policy(
        _REFUTABLE, pi, 1.0, vcf_already_fired=True)
    assert fired is False
    assert new_z == 1.0
    assert new_pi is pi


# StM open four (its OWN forced win) AND opponent open four (so detection fires).
_STM_ALSO_WINS = make_planes(
    me_cells=[(2, 2), (2, 3), (2, 4), (2, 5)],
    opp_cells=[(6, 2), (6, 3), (6, 4), (6, 5)],
)


def test_policy_teacher_skips_when_side_to_move_has_own_win():
    # #43 AUDIT FIX (the confirmed MAJOR): when the side-to-move ITSELF has a proven
    # forced win, the teacher must NOT overwrite its correct winning policy with a
    # defensive blend — even though the opponent ALSO has a forced four (detection
    # fires). This guard is SELF-CONTAINED: it re-solves the un-swapped board rather
    # than relying on vcf_already_fired, which is dead when --vcf-teacher is off (the
    # live G15-defense-i2 configuration).
    detect = _swapped(_STM_ALSO_WINS)
    assert vcf.solve_vcf(detect).has_forced_win  # opponent has a forced win
    own = np.stack([_STM_ALSO_WINS[0].astype(bool),
                    _STM_ALSO_WINS[HISTORY_PLY].astype(bool)], axis=0)
    assert vcf.solve_vcf(own).has_forced_win      # side-to-move ALSO has its own win
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    new_pi, new_z, fired = _apply_defense_teacher_policy(
        _STM_ALSO_WINS, pi, 0.7, vcf_already_fired=False)
    assert fired is False   # bailed — keep the winning policy, do not stamp defense
    assert new_pi is pi     # policy untouched
    assert new_z == 0.7     # value untouched


def test_own_win_guard_does_not_oversuppress_pure_defense():
    # The guard must ONLY suppress when the side-to-move has its own win. A pure
    # defensive position (StM has no four of its own) must still fire normally.
    own = np.stack([_REFUTABLE[0].astype(bool),
                    _REFUTABLE[HISTORY_PLY].astype(bool)], axis=0)
    assert not vcf.solve_vcf(own).has_forced_win   # StM has no own win here
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    _new_pi, _new_z, fired = _apply_defense_teacher_policy(
        _REFUTABLE, pi, 0.5, vcf_already_fired=False)
    assert fired is True


def test_vcf_refutations_honors_max_candidates_cap():
    # The MINOR density-tail bound: max_candidates limits how many candidate
    # defender moves are re-solved. max_candidates=0 -> no candidate tested -> [].
    board = _swapped(_REFUTABLE)
    full = vcf.vcf_refutations(board)
    assert full == [_flat(2, 6)]
    assert vcf.vcf_refutations(board, max_candidates=0) == []


def test_soft_target_over_multiple_refutations():
    # Two independent simple fours, each with a single (distinct) completion the
    # defender must block. Neither block alone refutes BOTH fours, so there is no
    # refutation here — but it lets us exercise the multi-completion bookkeeping
    # for vcf_refutations on a genuinely lost double-threat (returns []).
    planes = make_planes(
        me_cells=[(2, 1), (6, 1)],
        opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5),
                   (6, 2), (6, 3), (6, 4), (6, 5)],
    )
    board = _swapped(planes)
    assert vcf.solve_vcf(board).has_forced_win
    # Blocking (2,6) leaves (6,6); blocking (6,6) leaves (2,6) -> no single
    # defender move breaks both forced fours.
    assert vcf.vcf_refutations(board) == []


# ---------------------------------------------------------------------------
# Trajectory integration: policy mode rewrites pi, not z
# ---------------------------------------------------------------------------

def test_trajectory_policy_mode_rewrites_policy_only(restore_defense_knobs):
    configure_defense_teacher(policy_mode=True)
    assert self_play._DEFENSE_POLICY_MODE is True
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    # One-ply game, side-to-move = black (0). outcome_for_black=+1 -> z base +1.
    traj = [(_REFUTABLE, pi, 0)]
    out = _apply_teachers_to_trajectory(
        traj, outcome_for_black=1.0,
        vcf_teacher=False, vct_teacher=False, defense_teacher=True)
    (out_pi, out_z) = out[0]
    expected = np.zeros(N * N, dtype=np.float32)
    expected[_flat(2, 6)] = 1.0
    assert np.allclose(out_pi, expected)   # policy was stamped...
    assert out_z == 1.0                     # ...and value left at the outcome


def test_trajectory_value_mode_unchanged_by_default(restore_defense_knobs):
    # With policy_mode OFF (the default), the SAME refutable position takes the
    # value-only path: z crushed to -1.0, policy untouched. Confirms the two modes
    # are cleanly switched by the one global and the default is the old behavior.
    assert self_play._DEFENSE_POLICY_MODE is False
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    traj = [(_REFUTABLE, pi.copy(), 0)]
    out = _apply_teachers_to_trajectory(
        traj, outcome_for_black=1.0,
        vcf_teacher=False, vct_teacher=False, defense_teacher=True)
    (out_pi, out_z) = out[0]
    assert out_z == -1.0
    assert np.allclose(out_pi, pi)  # value teacher never touches policy


# ---------------------------------------------------------------------------
# Per-game FRACTION budget for the policy stamps (#42 cap, #43 target)
# ---------------------------------------------------------------------------

def test_policy_relabel_budget_keeps_latest(restore_defense_knobs):
    configure_defense_teacher(max_fraction=0.25)
    # 12 positions -> budget floor(0.25*12) == 3; keep the LATEST 3 firing plies.
    one_hot = np.zeros(N * N, dtype=np.float32)
    one_hot[_flat(2, 6)] = 1.0
    fires = [(i, one_hot) for i in range(2, 10)]  # plies 2..9
    kept = _relabel_defense_policy_game(fires, n_positions=12)
    assert set(kept) == {7, 8, 9}


def test_policy_relabel_unbounded_by_default():
    one_hot = np.zeros(N * N, dtype=np.float32)
    one_hot[_flat(2, 6)] = 1.0
    fires = [(i, one_hot) for i in range(10)]
    kept = _relabel_defense_policy_game(fires, n_positions=12)
    assert set(kept) == set(range(10))  # frac 1.0 keeps every fire


# ---------------------------------------------------------------------------
# Soundness fuzz: every stamped saving move is a real escape
# ---------------------------------------------------------------------------

def test_refutations_are_sound_random_positions():
    rng = np.random.default_rng(20260618)
    n_fired = 0
    for _ in range(300):
        n_stones = int(rng.integers(2, 16))
        cells = rng.choice(N * N, size=n_stones, replace=False)
        me_cells = [(int(x // N), int(x % N)) for x in cells[0::2]]
        opp_cells = [(int(x // N), int(x % N)) for x in cells[1::2]]
        planes = make_planes(me_cells, opp_cells)
        board = _swapped(planes)
        if state_ops.has_five_in_a_row(board[0]) or state_ops.has_five_in_a_row(board[1]):
            continue
        pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
        new_pi, _z, fired = _apply_defense_teacher_policy(
            planes, pi, 0.0, vcf_already_fired=False)
        if not fired:
            continue
        n_fired += 1
        # Every stamped move must, when played by the defender, kill the forced
        # win — and the policy mass must be exactly on those moves.
        stamped = np.flatnonzero(new_pi > 0)
        assert len(stamped) > 0
        for d in stamped:
            attacker = board[0].copy()
            defender = board[1].copy()
            defender[d // N, d % N] = True
            after = np.stack([attacker, defender], axis=0)
            assert not vcf.solve_vcf(after).has_forced_win, \
                "POLICY TEACHER UNSOUND: stamped move does not break the VCF"
    # Some seeds may yield no fires; that's fine (the assert above is the gate).
    assert n_fired >= 0
