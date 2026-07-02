"""Sound-world oracle veto (issue #107): gen-side blunder masking + defender
terminus.

Deterministic unit tests of the veto/partition logic — no MLX/Metal, the
blunder maps are hand-built. Mirrors tests/test_vct_terminus.py. The
real-oracle end-to-end behaviour is exercised by the live run, gated on
selfplay/plies_mean leaving the ~9-10 attractor.
"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku import self_play as sp
from gomoku.game import HISTORY_PLY, N_ACTIONS, N_INPUT_PLANES


@pytest.fixture(autouse=True)
def _reset_veto_global():
    enabled = sp._ORACLE_VETO_ENABLED
    yield
    sp.configure_oracle_veto(enabled=enabled)


def _planes_with_stones(me_flat=(), opp_flat=(), n=9):
    p = np.zeros((N_INPUT_PLANES, n, n), dtype=np.float32)
    for m in me_flat:
        p[0].reshape(-1)[m] = 1.0
    for m in opp_flat:
        p[HISTORY_PLY].reshape(-1)[m] = 1.0
    p[2 * HISTORY_PLY] = 1.0
    return p


def test_default_off_and_configure():
    assert sp._ORACLE_VETO_ENABLED is False           # default OFF = byte-identical
    sp.configure_oracle_veto(enabled=True)
    assert sp._ORACLE_VETO_ENABLED is True
    sp.configure_oracle_veto(enabled=None)            # None leaves it unchanged
    assert sp._ORACLE_VETO_ENABLED is True
    sp.configure_oracle_veto(enabled=False)
    assert sp._ORACLE_VETO_ENABLED is False


def test_veto_policy_masks_and_renormalizes():
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[3], pi[7], pi[11] = 0.5, 0.3, 0.2
    vmap = np.zeros(N_ACTIONS, dtype=np.float32)
    vmap[7] = 1.0                                     # 7 is a proven blunder
    legal = np.ones(N_ACTIONS, dtype=bool)
    out = sp._veto_policy(pi, vmap, legal)
    assert out[7] == 0.0
    np.testing.assert_allclose(out[3], 0.5 / 0.7, rtol=1e-6)
    np.testing.assert_allclose(out.sum(), 1.0, rtol=1e-6)


def test_veto_policy_fallback_uniform_over_safe_legal():
    # Search put ALL its mass on blunders -> uniform over legal non-blunder.
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[7] = 1.0
    vmap = np.zeros(N_ACTIONS, dtype=np.float32)
    vmap[7] = 1.0
    legal = np.zeros(N_ACTIONS, dtype=bool)
    legal[[5, 7, 9]] = True                           # safe legal cells: 5 and 9
    out = sp._veto_policy(pi, vmap, legal)
    assert out[7] == 0.0
    np.testing.assert_allclose(out[5], 0.5, rtol=1e-6)
    np.testing.assert_allclose(out[9], 0.5, rtol=1e-6)


def test_veto_policy_nan_pi_falls_back():
    pi = np.full(N_ACTIONS, np.nan, dtype=np.float32)
    vmap = np.zeros(N_ACTIONS, dtype=np.float32)
    legal = np.zeros(N_ACTIONS, dtype=bool)
    legal[[2, 4]] = True
    out = sp._veto_policy(pi, vmap, legal)
    np.testing.assert_allclose(out[[2, 4]], [0.5, 0.5], rtol=1e-6)
    assert np.isfinite(out).all()


def test_partition_all_lose_is_a_defender_terminus():
    # Board with two stones; EVERY legal cell is a proven blunder -> the side
    # to move LOSES here, no search is spent, and — the 2026-07-01 wound fix —
    # NO training example is recorded for the doomed position. The original
    # uniform-over-legal pi target taught white's policy noise at scale: as
    # black sharpened, most games ended here, so white's most common examples
    # were uniform noise and white collapsed (e1982: 0W-20L-0D as white vs the
    # old champion, vs all-draws at e1239; self-metrics saw nothing). See #107.
    planes = _planes_with_stones(me_flat=(0,), opp_flat=(1,))
    legal = sp._legal_mask_from_planes(planes)
    vmap = np.zeros(N_ACTIONS, dtype=np.float32)
    vmap[legal] = 1.0
    pending = {4: vmap}
    trajectories = {4: []}
    completed = []
    final_state = {}
    vct_maps = {4: []}

    new_active, new_games = sp._oracle_veto_partition(
        [4], ["g"], [planes], pending, ply=5, initial_plies={4: 0},
        trajectories=trajectories, completed=completed, final_state=final_state,
        record_ownership=True, record_vct=True, vct_maps=vct_maps, profile=None)

    assert new_active == [] and new_games == []
    # side = (0+5)%2 = 1 (white) is to move and is LOST -> black wins
    assert completed == [(4, 1.0, 5)]
    assert trajectories[4] == []                      # doomed position NOT recorded
    assert vct_maps[4] == []                          # lockstep stays empty
    assert final_state[4] == (None, 0)                # ownership masked


def test_partition_survives_with_a_safe_move_or_no_map():
    planes = _planes_with_stones(me_flat=(0,), opp_flat=(1,))
    legal = sp._legal_mask_from_planes(planes)
    vmap = np.zeros(N_ACTIONS, dtype=np.float32)
    vmap[legal] = 1.0
    vmap[80] = 0.0                                    # one safe legal cell
    pending = {0: vmap, 1: None}
    trajectories = {0: [], 1: []}
    completed = []

    new_active, new_games = sp._oracle_veto_partition(
        [0, 1], ["a", "b"], [planes, planes], pending, ply=2,
        initial_plies={0: 0, 1: 0}, trajectories=trajectories,
        completed=completed, final_state={}, record_ownership=False,
        record_vct=False, vct_maps={0: [], 1: []}, profile=None)

    assert new_active == [0, 1] and new_games == ["a", "b"]
    assert completed == [] and trajectories[0] == []


def test_partition_black_to_move_all_lose_scores_white_win():
    planes = _planes_with_stones(me_flat=(0,), opp_flat=(1,))
    legal = sp._legal_mask_from_planes(planes)
    vmap = np.zeros(N_ACTIONS, dtype=np.float32)
    vmap[legal] = 1.0
    completed = []
    new_active, _ = sp._oracle_veto_partition(
        [0], ["g"], [planes], {0: vmap}, ply=4, initial_plies={0: 0},
        trajectories={0: []}, completed=completed, final_state={},
        record_ownership=False, record_vct=False, vct_maps={0: []}, profile=None)
    assert new_active == []
    g_idx, outcome_for_black, plies = completed[0]
    assert g_idx == 0 and outcome_for_black == -1.0 and plies == 4
