"""VCT-terminus self-play (issue #98): end a game at the first cap50 VCT.

Deterministic unit tests of the termination/partition logic — the batched GPU
oracle is monkeypatched, so these run on CPU with no MLX/Metal. The real-oracle
end-to-end behaviour (games actually terminate early on 9x9) is exercised by the
smoke script, not here.
"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku import self_play as sp
from gomoku.game import N_ACTIONS


@pytest.fixture(autouse=True)
def _reset_terminus_globals():
    """Keep the process-wide terminus config from leaking across tests."""
    enabled, budget = sp._VCT_TERMINUS_ENABLED, sp._VCT_TERMINUS_BUDGET
    yield
    sp.configure_vct_terminus(enabled=enabled, budget=budget)


def test_default_off_and_configure():
    assert sp._VCT_TERMINUS_ENABLED is False          # default OFF = byte-identical
    sp.configure_vct_terminus(enabled=True, budget=250)
    assert sp._VCT_TERMINUS_ENABLED is True
    assert sp._VCT_TERMINUS_BUDGET == 250
    sp.configure_vct_terminus(enabled=False)
    assert sp._VCT_TERMINUS_ENABLED is False
    assert sp._VCT_TERMINUS_BUDGET == 250              # None leaves budget unchanged


def test_partition_terminates_vct_games_and_keeps_the_rest(monkeypatch):
    # Oracle verdict: only game-slot 1 has a forced VCT, winning move index 7.
    def fake_solve(planes_list):
        assert len(planes_list) == 3
        return np.array([False, True, False]), np.array([-1, 7, -1])
    monkeypatch.setattr(sp, "_vct_terminus_solve", fake_solve)

    active = [0, 1, 2]
    active_games = ["a", "b", "c"]
    planes_list = [np.zeros((2, 9, 9), np.float32) for _ in range(3)]
    trajectories = [[], [], []]
    completed = []
    final_state = {}
    ply = 5
    initial_plies = [0, 0, 0]

    new_active, new_games = sp._vct_terminus_partition(
        active, active_games, planes_list, ply, initial_plies,
        trajectories, completed, final_state, record_ownership=False, profile=None)

    # survivors (no VCT) continue; the VCT game is dropped
    assert new_active == [0, 2]
    assert new_games == ["a", "c"]

    # exactly one game completed: g_idx=1, side=(0+5)%2=1 (white) -> white wins ->
    # outcome_for_black = -1.0; plies = n_initial + ply + 1 = 6
    assert completed == [(1, -1.0, 6)]

    # the terminated game recorded ONE decisive example: one-hot on the oracle move
    assert len(trajectories[1]) == 1
    planes, pi, side = trajectories[1][0]
    assert side == 1
    assert pi.shape == (N_ACTIONS,)
    assert pi[7] == 1.0 and pi.sum() == 1.0
    assert planes.shape == (2, 9, 9)
    # survivors recorded nothing here (they get their normal MCTS example later)
    assert trajectories[0] == [] and trajectories[2] == []


def test_partition_black_vct_scores_black_win(monkeypatch):
    # side-to-move at an EVEN total ply is black (0); a black VCT -> +1 for black.
    monkeypatch.setattr(
        sp, "_vct_terminus_solve",
        lambda planes_list: (np.array([True]), np.array([12])))
    active, active_games = [0], ["g"]
    planes_list = [np.zeros((2, 9, 9), np.float32)]
    trajectories, completed, final_state = [[]], [], {}

    new_active, _ = sp._vct_terminus_partition(
        active, active_games, planes_list, ply=4, initial_plies=[0],
        trajectories=trajectories, completed=completed, final_state=final_state,
        record_ownership=False, profile=None)

    assert new_active == []                            # all terminated
    g_idx, outcome_for_black, plies = completed[0]
    assert g_idx == 0 and outcome_for_black == 1.0 and plies == 5
    _, pi, side = trajectories[0][0]
    assert side == 0 and pi[12] == 1.0


def test_ownership_masked_at_vct_terminus(monkeypatch):
    monkeypatch.setattr(
        sp, "_vct_terminus_solve",
        lambda planes_list: (np.array([True]), np.array([3])))
    final_state = {}
    sp._vct_terminus_partition(
        [0], ["g"], [np.zeros((2, 9, 9), np.float32)], ply=3, initial_plies=[0],
        trajectories=[[]], completed=[], final_state=final_state,
        record_ownership=True, profile=None)
    # a VCT board is not a real five-in-a-row -> ownership target MASKED (None)
    assert final_state[0] == (None, 0)


def test_no_vct_is_a_noop(monkeypatch):
    monkeypatch.setattr(
        sp, "_vct_terminus_solve",
        lambda planes_list: (np.array([False, False]), np.array([-1, -1])))
    active, active_games = [3, 4], ["x", "y"]
    trajectories, completed = [[], [], [], [], []], []
    new_active, new_games = sp._vct_terminus_partition(
        active, active_games, [np.zeros((2, 9, 9))] * 2, ply=2, initial_plies=[0] * 5,
        trajectories=trajectories, completed=completed, final_state={},
        record_ownership=False, profile=None)
    assert new_active == [3, 4] and new_games == ["x", "y"]
    assert completed == []
