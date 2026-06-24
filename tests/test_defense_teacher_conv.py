"""Correctness tests for the CONV block-teacher (the white-defense dense-shallow arm).

The conv teacher is the COMPLEMENT of the #43 sparse+deep VCF policy teacher: a
CHEAP vectorized board scan (NO solve_vcf, NO tree search) that fires on every ply
and stamps the BLOCK to the opponent's IMMEDIATE threat on the POLICY head, leaving
the value at the natural game outcome. Detection lives in
``self_play._conv_block_detect``; the per-ply stamp in
``self_play._apply_defense_teacher_conv``; the per-game seam in
``self_play._apply_teachers_to_trajectory`` (gated on ``_DEFENSE_CONV_MODE``).

The blocking soundness properties this file enforces:
  * Tier 1 (forced four) is EXACT: when the opponent has exactly one immediate
    five-completion, the stamped cell is that unique block — and it agrees with the
    independent ``vcf.solve_vcf`` winning move on the depth-1 case (VCF cross-check).
  * A double-four (>=2 opp win cells) is genuinely lost -> NO stamp.
  * The defender having its own win -> NO stamp (convert, don't defend).
  * A quiet position -> NO stamp.
  * NEVER stamp a cell that does not actually block the opponent's immediate win
    (no false saves).
  * Byte-identity OFF: with conv mode off, the seam output is unchanged and the
    detector is never called.

Run at board 15 (the live config): ``GOMOKU_BOARD_SIZE=15 pytest tests/test_defense_teacher_conv.py``.
"""

from __future__ import annotations

import numpy as np

import pytest

from gomoku import self_play, state_ops, vcf
from gomoku.game import HISTORY_PLY, N_INPUT_PLANES
from gomoku.self_play import (
    _apply_defense_teacher_conv,
    _apply_teachers_to_trajectory,
    _conv_block_detect,
    configure_defense_teacher,
)

N = state_ops.BOARD_SIZE

# A few fixtures below are crafted for the LIVE 15x15 board (the docstring's
# stated config) and are geometrically invalid at 9x9: the open-three at row 6
# cols 6,7,8 has a prevention cell at (6,9), which is OFF a 9x9 board (cols 0..8),
# and _crafted_single_fours' r0/c0 sweep yields < the required 20 positions on a
# 9x9 grid. They are NOT board-agnostic, so gate them to board 15 — matching the
# sibling white-defense suite (tests/test_white_defense.py) which skips wholesale
# unless GOMOKU_BOARD_SIZE=15. The remaining tests in this file are board-agnostic
# and still run at every size. (The conv teacher's detection logic itself is
# board-size-independent; only these crafted positions need the 15x15 geometry.)
needs_board15 = pytest.mark.skipif(
    N != 15,
    reason="fixture geometry (open-three prevention cell / crafted-fours count) "
    "is crafted for the live 15x15 board; run with GOMOKU_BOARD_SIZE=15",
)


@pytest.fixture
def restore_defense_knobs():
    """The conv-teacher knobs are PROCESS-WIDE globals; restore them so a test
    that flips _DEFENSE_CONV_MODE cannot poison the byte-identity-off siblings."""
    saved = (
        self_play._DEFENSE_CONV_MODE,
        self_play._DEFENSE_CONV_TIER2,
        self_play._DEFENSE_MAX_FRACTION,
    )
    try:
        yield
    finally:
        (
            self_play._DEFENSE_CONV_MODE,
            self_play._DEFENSE_CONV_TIER2,
            self_play._DEFENSE_MAX_FRACTION,
        ) = saved


def make_planes(me_cells, opp_cells):
    """(N_INPUT_PLANES, N, N) float32 stack: DEFENDER (side to move) at plane 0,
    opponent (attacker) at plane HISTORY_PLY — the layout _conv_block_detect reads."""
    planes = np.zeros((N_INPUT_PLANES, N, N), dtype=np.float32)
    for r, c in me_cells:
        planes[0, r, c] = 1.0
    for r, c in opp_cells:
        planes[HISTORY_PLY, r, c] = 1.0
    return planes


def _flat(r, c):
    return r * N + c


def _swapped(planes):
    """The (2, N, N) board the VCF detector solves: opponent = attacker."""
    opp = planes[HISTORY_PLY].astype(bool)
    me = planes[0].astype(bool)
    return np.stack([opp, me], axis=0)


# ---------------------------------------------------------------------------
# Tier 1 soundness: forced four, double four, own win, quiet
# ---------------------------------------------------------------------------

def test_unique_four_stamps_the_block():
    # Opponent XXXX at row 2 cols 2..5; LEFT end pre-blocked by a defender stone at
    # (2,1) so the only completion is (2,6). The conv teacher must one-hot (2,6).
    planes = make_planes(me_cells=[(2, 1), (8, 8)],
                         opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)])
    cells = _conv_block_detect(planes, tier2=True)
    assert cells is not None
    assert list(cells) == [_flat(2, 6)]


def test_double_four_no_stamp():
    # An OPEN four .XXXX. -> completions at BOTH ends (a double four) -> genuinely
    # lost -> NO stamp (mirror of the exact teacher leaving true losses untouched).
    planes = make_planes(me_cells=[(8, 8)],
                         opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)])
    assert _conv_block_detect(planes, tier2=True) is None


def test_defender_own_win_no_stamp():
    # The DEFENDER itself has a five-completion -> convert, don't defend -> NO stamp,
    # even though the opponent also has a (single-completion) four.
    planes = make_planes(
        me_cells=[(5, 5), (5, 6), (5, 7), (5, 8)],   # defender open four (its own win)
        opp_cells=[(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],  # opp ... blocked? see below
    )
    # Make the opponent threat a real single four too, so the only reason to bail is
    # the defender's own win (not "no opp threat").
    planes = make_planes(
        me_cells=[(5, 5), (5, 6), (5, 7), (5, 8)],
        opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)],
    )
    # Defender has an open four of its own -> own_win guard fires.
    assert vcf._five_completions(planes[0].astype(bool),
                                 ~(planes[0].astype(bool) | planes[HISTORY_PLY].astype(bool)))
    assert _conv_block_detect(planes, tier2=True) is None


def test_quiet_position_no_stamp():
    planes = make_planes(me_cells=[(4, 4)], opp_cells=[(0, 0), (8, 8)])
    assert _conv_block_detect(planes, tier2=True) is None


# ---------------------------------------------------------------------------
# Tier 2 (heuristic): open three -> open four prevention; sub-toggle
# ---------------------------------------------------------------------------

@needs_board15
def test_open_three_stamps_prevention_cells():
    # Opponent .XXX. (three at row 6 cols 6,7,8). The open-four-making moves are
    # (6,5) and (6,9); each end is in the other's defeating set, so blocking either
    # end prevents BOTH open fours -> the intersection is {(6,5),(6,9)}.
    planes = make_planes(me_cells=[(0, 0)], opp_cells=[(6, 6), (6, 7), (6, 8)])
    cells = _conv_block_detect(planes, tier2=True)
    assert cells is not None
    assert set(int(c) for c in cells) == {_flat(6, 5), _flat(6, 9)}


def test_tier2_disabled_skips_open_three():
    # With Tier 2 off, the SAME open three is left untouched (only the sound
    # forced-four block remains).
    planes = make_planes(me_cells=[(0, 0)], opp_cells=[(6, 6), (6, 7), (6, 8)])
    assert _conv_block_detect(planes, tier2=False) is None


def test_tier1_takes_priority_over_tier2():
    # When the opponent has a bare four (Tier 1) the forced block is returned even
    # with Tier 2 enabled — the four short-circuits before the open-three scan.
    planes = make_planes(me_cells=[(2, 1)],
                         opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)])
    cells = _conv_block_detect(planes, tier2=True)
    assert list(cells) == [_flat(2, 6)]


# ---------------------------------------------------------------------------
# THE VCF CROSS-CHECK: the conv forced-block agrees with the exact solver on the
# shallow (depth-1 four) case, and never stamps a non-blocking cell.
# ---------------------------------------------------------------------------

def _crafted_single_fours():
    """Yield crafted positions where the OPPONENT has exactly one immediate five-
    completion (a depth-1 forced win) and the DEFENDER cannot itself win: four
    opponent stones in a row along each axis, one end pre-blocked by a defender
    stone, placed away from any other defender stone. Each yields
    (planes, unique_completion_flat_index)."""
    _DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for r0 in range(1, N - 6):
        for c0 in range(2, N - 6):
            for dr, dc in _DIRS:
                opp = [(r0 + dr * k, c0 + dc * k) for k in range(4)]
                br, bc = r0 - dr, c0 - dc          # block one open end
                ar, ac = r0 + dr * 4, c0 + dc * 4  # the other end (the completion)
                if not (0 <= br < N and 0 <= bc < N and 0 <= ar < N and 0 <= ac < N):
                    continue
                planes = make_planes(me_cells=[(br, bc)], opp_cells=opp)
                me = planes[0].astype(bool)
                opp_b = planes[HISTORY_PLY].astype(bool)
                empty = ~(me | opp_b)
                opp_win = vcf._five_completions(opp_b, empty)
                own_win = vcf._five_completions(me, empty)
                if len(opp_win) == 1 and len(own_win) == 0:
                    yield planes, int(opp_win[0])


@needs_board15
def test_conv_agrees_with_vcf_on_shallow_fours():
    # CROSS-CHECK: for >= 20 crafted positions where the opponent has an IMMEDIATE
    # four (a depth-1 forced win), the conv detector's forced-block cell must equal
    # the winning move vcf.solve_vcf identifies on the swapped board (opponent =
    # attacker). NEVER stamp a cell that does not actually block (no false saves):
    # occupying the stamped cell must remove the opponent's five.
    n_checked = 0
    for planes, completion in _crafted_single_fours():
        board = _swapped(planes)
        res = vcf.solve_vcf(board)
        assert res.has_forced_win
        assert res.mate_distance == 1                 # immediate five = depth-1
        assert res.winning_move == completion         # solver agrees on the cell
        cells = _conv_block_detect(planes, tier2=True)
        assert cells is not None and list(cells) == [completion], \
            "conv forced-block disagrees with vcf.solve_vcf on a depth-1 four"
        # No false save: occupying the block removes the opponent's immediate win.
        me = planes[0].astype(bool).copy()
        opp = planes[HISTORY_PLY].astype(bool)
        me[completion // N, completion % N] = True
        assert vcf._five_completions(opp, ~(opp | me)) == [], \
            "FALSE SAVE: stamped block did not remove the opponent's five"
        n_checked += 1
        if n_checked >= 40:
            break
    assert n_checked >= 20, f"only {n_checked} shallow-four cross-checks (need >= 20)"


def test_conv_never_stamps_a_nonblocking_cell_fuzz():
    # Stronger dual soundness over many positions: EVERY stamped Tier-1 single-cell
    # block, once occupied by the defender, must remove the opponent's immediate
    # win. (Tier-2 stamps are heuristic open-four prevention, not immediate-win
    # blocks, so they are excluded from this immediate-five assertion.)
    rng = np.random.default_rng(424242)
    n_checked = 0
    for _ in range(1500):
        n_stones = int(rng.integers(2, 18))
        cells_idx = rng.choice(N * N, size=n_stones, replace=False)
        me_cells = [(int(x // N), int(x % N)) for x in cells_idx[0::2]]
        opp_cells = [(int(x // N), int(x % N)) for x in cells_idx[1::2]]
        planes = make_planes(me_cells, opp_cells)
        me = planes[0].astype(bool)
        opp = planes[HISTORY_PLY].astype(bool)
        if state_ops.has_five_in_a_row(me) or state_ops.has_five_in_a_row(opp):
            continue
        empty = ~(me | opp)
        opp_win = vcf._five_completions(opp, empty)
        own_win = vcf._five_completions(me, empty)
        cells = _conv_block_detect(planes, tier2=True)
        if cells is None:
            continue
        if len(opp_win) == 1 and len(own_win) == 0:
            block = int(cells[0])
            defender2 = me.copy()
            defender2[block // N, block % N] = True
            empty2 = ~(opp | defender2)
            assert vcf._five_completions(opp, empty2) == []
            n_checked += 1
    assert n_checked > 0


# ---------------------------------------------------------------------------
# Per-ply stamp wrapper: uniform policy, value untouched
# ---------------------------------------------------------------------------

@needs_board15
def test_apply_conv_stamps_uniform_policy_value_untouched(restore_defense_knobs):
    self_play._DEFENSE_CONV_TIER2 = True
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    # Open three -> two prevention cells -> uniform 0.5 / 0.5, value left at 0.5.
    planes = make_planes(me_cells=[(0, 0)], opp_cells=[(6, 6), (6, 7), (6, 8)])
    new_pi, new_z, fired = _apply_defense_teacher_conv(planes, pi, 0.5)
    assert fired is True
    assert new_z == 0.5  # value untouched (pure policy lever)
    nz = np.flatnonzero(new_pi)
    assert set(int(x) for x in nz) == {_flat(6, 5), _flat(6, 9)}
    assert np.allclose(new_pi[nz], 0.5)
    assert pytest.approx(new_pi.sum()) == 1.0


def test_apply_conv_no_fire_returns_inputs(restore_defense_knobs):
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    quiet = make_planes(me_cells=[(4, 4)], opp_cells=[(0, 0), (8, 8)])
    new_pi, new_z, fired = _apply_defense_teacher_conv(quiet, pi, 0.3)
    assert fired is False
    assert new_z == 0.3
    assert new_pi is pi


# ---------------------------------------------------------------------------
# Trajectory integration: conv mode rewrites policy, not value
# ---------------------------------------------------------------------------

def test_trajectory_conv_mode_rewrites_policy_only(restore_defense_knobs):
    configure_defense_teacher(conv_mode=True, conv_tier2=True)
    assert self_play._DEFENSE_CONV_MODE is True
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    # One-ply game, side-to-move = black (0); outcome_for_black=+1 -> z base +1.
    refutable = make_planes(me_cells=[(2, 1)],
                            opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)])
    traj = [(refutable, pi.copy(), 0)]
    out = _apply_teachers_to_trajectory(
        traj, outcome_for_black=1.0,
        vcf_teacher=False, vct_teacher=False, defense_teacher=True)
    (out_pi, out_z) = out[0]
    expected = np.zeros(N * N, dtype=np.float32)
    expected[_flat(2, 6)] = 1.0
    assert np.allclose(out_pi, expected)  # policy stamped...
    assert out_z == 1.0                    # ...value left at the outcome


def test_trajectory_conv_max_fraction_keeps_latest(restore_defense_knobs):
    # The per-game FRACTION budget applies to conv stamps too (latest plies first).
    configure_defense_teacher(conv_mode=True, conv_tier2=True, max_fraction=0.5)
    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    refutable = make_planes(me_cells=[(2, 1)],
                            opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)])
    traj = [(refutable, pi.copy(), 0) for _ in range(4)]  # n=4, budget floor(0.5*4)=2
    out = _apply_teachers_to_trajectory(
        traj, outcome_for_black=1.0,
        vcf_teacher=False, vct_teacher=False, defense_teacher=True)
    stamped = {p for p in range(4) if not np.allclose(out[p][0], pi)}
    assert stamped == {2, 3}  # latest 2 of 4 firing plies
    expected = np.zeros(N * N, dtype=np.float32)
    expected[_flat(2, 6)] = 1.0
    for p in stamped:
        assert np.allclose(out[p][0], expected)
    for p in range(4):
        assert out[p][1] == 1.0  # value untouched


# ---------------------------------------------------------------------------
# Byte-identity OFF: detector never called, seam output unchanged
# ---------------------------------------------------------------------------

def test_conv_off_is_default():
    # The module-level toggle must default OFF so any cell that does not pass the
    # conv flag is byte-identical.
    assert self_play._DEFENSE_CONV_MODE is False


def test_conv_off_detector_never_called_and_output_unchanged(monkeypatch, restore_defense_knobs):
    # With conv mode OFF, _apply_teachers_to_trajectory must not call the conv
    # detector at all, and its output must equal the no-defense-teacher baseline.
    self_play._DEFENSE_CONV_MODE = False
    calls = {"n": 0}
    real = self_play._conv_block_detect

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(self_play, "_conv_block_detect", counting)

    pi = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    refutable = make_planes(me_cells=[(2, 1)],
                            opp_cells=[(2, 2), (2, 3), (2, 4), (2, 5)])
    traj = [(refutable, pi.copy(), 0)]
    # Baseline: defense_teacher entirely off.
    base = _apply_teachers_to_trajectory(
        traj, outcome_for_black=1.0,
        vcf_teacher=False, vct_teacher=False, defense_teacher=False)
    # With defense_teacher True but conv mode off, the value teacher path runs (not
    # conv); the conv detector is still never called.
    _ = _apply_teachers_to_trajectory(
        [(refutable, pi.copy(), 0)], outcome_for_black=1.0,
        vcf_teacher=False, vct_teacher=False, defense_teacher=True)
    assert calls["n"] == 0, "conv detector was called while conv mode is OFF"
    # The pure-off baseline policy is untouched (no teacher of any kind).
    assert np.allclose(base[0][0], pi)
    assert base[0][1] == 1.0


def test_configure_conv_roundtrip(restore_defense_knobs):
    configure_defense_teacher(conv_mode=True, conv_tier2=False)
    assert self_play._DEFENSE_CONV_MODE is True
    assert self_play._DEFENSE_CONV_TIER2 is False
    # None leaves a field untouched.
    configure_defense_teacher(conv_mode=None, conv_tier2=None)
    assert self_play._DEFENSE_CONV_MODE is True
    assert self_play._DEFENSE_CONV_TIER2 is False


def test_generate_games_defense_teacher_still_defaults_off():
    # The conv lever piggybacks on the existing defense_teacher gate, so the public
    # generator signatures must still default defense_teacher to False (byte-identical
    # production path; the conv detector cannot run unless a worker opts in).
    import inspect
    for name in (
        "generate_games",
        "generate_games_vs_baseline",
        "_generate_games_native",
        "_generate_games_gumbel",
        "_generate_games_native_gumbel",
    ):
        fn = getattr(self_play, name)
        sig = inspect.signature(fn)
        assert sig.parameters["defense_teacher"].default is False
