"""Merged per-ply oracle solve + staged escalation + oracle/search overlap.

The perf restructure (one bulk mega-solve per ply instead of a terminus call
plus a defense call — one solver dispatch costs one TAIL) must be BIT-IDENTICAL
to the historical two-call form. These tests prove the Python-side equivalence
with a deterministic composition-independent fake solver (win is a function of
the board alone, mirroring the megakernel's per-thread node budget — the GPU
side of the same property is exercised by a live-solver probe, see the wiki).
No MLX/Metal needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku import self_play as sp
from gomoku.game import HISTORY_PLY, N_ACTIONS, N_INPUT_PLANES
from gomoku.board_config import BOARD_SIZE


@pytest.fixture(autouse=True)
def _reset_globals():
    solver = sp._vct_terminus_solver
    veto, cands = sp._ORACLE_VETO_ENABLED, sp._VETO_MAX_CANDS
    overlap = sp._ORACLE_OVERLAP_ENABLED
    yield
    sp._vct_terminus_solver = solver
    sp.configure_oracle_veto(enabled=veto, max_cands=cands)
    sp.configure_oracle_overlap(enabled=overlap)


def fake_solver(boards, *, max_nodes=50, return_move=False, **_kw):
    """Composition-independent verdict: each board's result depends only on its
    own planes (the property the megakernel guarantees via its per-thread node
    counter). Win iff attacker has >= defender + 2 stones."""
    boards = np.asarray(boards)
    B = boards.shape[0]
    natt = boards[:, 0].reshape(B, -1).sum(axis=1)
    ndef = boards[:, 1].reshape(B, -1).sum(axis=1)
    win = natt >= ndef + 2
    hit = np.zeros(B, dtype=bool)
    empty = ~(boards[:, 0] | boards[:, 1]).reshape(B, -1)
    move = np.where(win & empty.any(axis=1), np.argmax(empty, axis=1), -1)
    if return_move:
        return win, hit, move.astype(np.int32)
    return win, hit


def _planes(me_flat=(), opp_flat=()):
    p = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for m in me_flat:
        p[0].reshape(-1)[m] = 1.0
    for m in opp_flat:
        p[HISTORY_PLY].reshape(-1)[m] = 1.0
    return p


def _random_planes(rng, n_me, n_opp):
    cells = rng.permutation(N_ACTIONS)[: n_me + n_opp]
    return _planes(me_flat=cells[:n_me], opp_flat=cells[n_me:])


def test_merged_solve_equals_separate_calls(monkeypatch):
    monkeypatch.setattr(sp, "_vct_terminus_solver", fake_solver)
    rng = np.random.default_rng(7)
    planes_list = [_random_planes(rng, k % 6, (k * 3) % 7) for k in range(9)]

    ref_win, ref_move = sp._vct_terminus_solve(planes_list)
    ref_maps, ref_masks = sp._vct_defense_solve(planes_list, max_cands=0)

    win_t, move_t, vmaps, vmasks = sp._oracle_ply_solve(
        planes_list, want_terminus=True, want_defense=True,
        defense_max_cands=0, profile=None)

    np.testing.assert_array_equal(win_t, ref_win)
    np.testing.assert_array_equal(move_t, ref_move)
    for a, b in zip(vmaps, ref_maps):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(vmasks, ref_masks):
        np.testing.assert_array_equal(a, b)


def test_merged_solve_single_consumer_modes(monkeypatch):
    monkeypatch.setattr(sp, "_vct_terminus_solver", fake_solver)
    rng = np.random.default_rng(11)
    planes_list = [_random_planes(rng, 4, 2) for _ in range(4)]

    win_t, move_t, vmaps, vmasks = sp._oracle_ply_solve(
        planes_list, want_terminus=True, want_defense=False)
    ref_win, ref_move = sp._vct_terminus_solve(planes_list)
    np.testing.assert_array_equal(win_t, ref_win)
    np.testing.assert_array_equal(move_t, ref_move)
    assert vmaps is None and vmasks is None

    win_t, move_t, vmaps, vmasks = sp._oracle_ply_solve(
        planes_list, want_terminus=False, want_defense=True,
        defense_max_cands=3)
    ref_maps, ref_masks = sp._vct_defense_solve(planes_list, max_cands=3)
    assert win_t is None and move_t is None
    for a, b in zip(vmaps, ref_maps):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(vmasks, ref_masks):
        np.testing.assert_array_equal(a, b)


def _reference_defense_children(planes_list, max_cands):
    """The historical per-cell loop (pre-vectorization), verbatim semantics."""
    n = BOARD_SIZE
    masks = [np.zeros(N_ACTIONS, dtype=bool) for _ in planes_list]
    child_boards, owners = [], []
    for pos_idx, p in enumerate(planes_list):
        p = np.asarray(p)
        me = p[0].astype(bool)
        opp = p[HISTORY_PLY].astype(bool)
        occupied = me | opp
        empty_flat = np.flatnonzero(~occupied.reshape(-1))
        cands = sp._defense_candidate_cells(empty_flat, occupied, max_cands)
        for m in cands:
            m = int(m)
            r, c = divmod(m, n)
            child_def = me.copy()
            child_def[r, c] = True
            child_boards.append(np.stack([opp, child_def], axis=0))
            owners.append((pos_idx, m))
            masks[pos_idx][m] = True
    if not child_boards:
        return None, owners, masks
    return np.stack(child_boards, axis=0).astype(bool), owners, masks


@pytest.mark.parametrize("max_cands", [0, 5])
def test_vectorized_children_match_reference_loop(max_cands):
    rng = np.random.default_rng(3)
    planes_list = [_random_planes(rng, k, k + 1) for k in range(0, 8, 2)]
    ref_batch, ref_owners, ref_masks = _reference_defense_children(
        planes_list, max_cands)
    arrs = [np.asarray(p) for p in planes_list]
    batch, opos, ocell, masks = sp._defense_children(arrs, max_cands)
    np.testing.assert_array_equal(batch, ref_batch)
    assert [(int(a), int(b)) for a, b in zip(opos, ocell)] == ref_owners
    for a, b in zip(masks, ref_masks):
        np.testing.assert_array_equal(a, b)


def test_escalation_completes_all_blunder_positions(monkeypatch):
    monkeypatch.setattr(sp, "_vct_terminus_solver", fake_solver)
    # Position 0: opp has 5 stones, me 1 -> every child (attacker=opp 5 vs
    # defender 2) wins for the attacker => stage 1 (K cells) all-blunder =>
    # must escalate to full breadth.
    # Position 1: opp 2 stones, me 1 -> no child wins => tested-safe, no
    # escalation.
    p0 = _planes(me_flat=(40,), opp_flat=(0, 1, 2, 3, 4))
    p1 = _planes(me_flat=(40,), opp_flat=(0, 1))
    planes_list = [p0, p1]
    K = 4
    _wt, _mt, vmaps, vmasks = sp._oracle_ply_solve(
        planes_list, want_terminus=False, want_defense=True,
        defense_max_cands=K)
    assert vmasks[0].sum() == K and vmasks[1].sum() == K
    assert vmaps[0][vmasks[0]].sum() == K          # all K tested cells lose
    assert vmaps[1][vmasks[1]].sum() == 0

    profile: dict = {}
    sp._escalate_all_blunder_positions(planes_list, [0, 1], vmaps, vmasks,
                                       profile)
    legal0 = sp._legal_mask_from_planes(p0)
    legal1 = sp._legal_mask_from_planes(p1)
    # position 0 escalated to full breadth; every legal cell now proven lost
    np.testing.assert_array_equal(vmasks[0] & legal0, legal0)
    assert np.all(vmaps[0][legal0] >= 0.5)
    # position 1 untouched (a tested-safe cell exists -> no terminus possible)
    assert vmasks[1].sum() == K
    assert profile.get("oracle_escalated_positions") == 1.0
    # full map == what a full-breadth stage-1 solve would have produced
    full_maps, full_masks = sp._vct_defense_solve([p0], max_cands=0)
    np.testing.assert_array_equal(vmaps[0], full_maps[0])
    np.testing.assert_array_equal(vmasks[0], full_masks[0])


def test_apply_oracle_partitions_terminus_then_veto(monkeypatch):
    monkeypatch.setattr(sp, "_vct_terminus_solver", fake_solver)
    sp.configure_oracle_veto(enabled=True, max_cands=0)
    # game 0: side to move has the fake "forced VCT" (me 4 vs opp 1) ->
    # attacker terminus fires pre-partition.
    # game 1: quiet (me 1, opp 1) -> survives with an all-zero blunder map.
    g0 = _planes(me_flat=(10, 11, 12, 13), opp_flat=(30,))
    g1 = _planes(me_flat=(10,), opp_flat=(30,))
    planes_list = [g0, g1]
    res = sp._oracle_ply_solve(planes_list, want_terminus=True,
                               want_defense=True, defense_max_cands=0)
    active, active_games = [5, 6], ["a", "b"]
    slot_of = {5: 0, 6: 1}
    trajectories: dict = {5: [], 6: []}
    completed: list = []
    vct_maps: dict = {5: [], 6: []}
    active, active_games, pending = sp._apply_oracle_partitions(
        res, active, active_games, planes_list, slot_of, ply=2,
        initial_plies={5: 0, 6: 0}, trajectories=trajectories,
        completed=completed, final_state={}, record_ownership=False,
        record_vct=False, vct_maps=vct_maps, profile=None)
    assert active == [6] and active_games == ["b"]
    assert len(completed) == 1 and completed[0][0] == 5
    assert len(trajectories[5]) == 1              # one-hot terminus example
    assert pending[6] is not None                 # full-breadth map recorded
    assert pending[6].sum() == 0.0                # ...with no blunders


def test_configure_oracle_overlap_default_off():
    assert sp._ORACLE_OVERLAP_ENABLED is False    # default OFF = byte-identical
    sp.configure_oracle_overlap(enabled=True)
    assert sp._ORACLE_OVERLAP_ENABLED is True
    sp.configure_oracle_overlap(enabled=None)
    assert sp._ORACLE_OVERLAP_ENABLED is True
    sp.configure_oracle_overlap(enabled=False)
    assert sp._ORACLE_OVERLAP_ENABLED is False


def test_configure_veto_max_cands():
    sp.configure_oracle_veto(enabled=True, max_cands=24)
    assert sp._ORACLE_VETO_ENABLED is True and sp._VETO_MAX_CANDS == 24
    sp.configure_oracle_veto(max_cands=0)
    assert sp._VETO_MAX_CANDS == 0
