"""Attacker-preserve mask (issue #116, sound-world idea #2): gen-side masking of
the recorded+played policy to the winmask when the mover has a clean proven VCT,
so the net learns to CLOSE on-policy.

Deterministic unit tests of the mask/solve logic — no MLX/Metal. `_preserve_policy`
is exercised with hand-built masks; `_attacker_preserve_solve` is exercised with a
fake solver + fake cell-unpacker (monkeypatched), mirroring tests/test_oracle_veto.py
and tests/test_oracle_merged_solve.py. The real-oracle end-to-end behaviour is
gated by the live run (selfplay/plies_mean + white-column recovery).
"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku import self_play as sp
from gomoku.game import HISTORY_PLY, N_ACTIONS, N_INPUT_PLANES


@pytest.fixture(autouse=True)
def _reset_preserve_global():
    enabled = sp._ATTACKER_PRESERVE_ENABLED
    yield
    sp.configure_attacker_preserve(enabled=enabled)


def _planes_with_stones(me_flat=(), opp_flat=(), n=9):
    p = np.zeros((N_INPUT_PLANES, n, n), dtype=np.float32)
    for m in me_flat:
        p[0].reshape(-1)[m] = 1.0
    for m in opp_flat:
        p[HISTORY_PLY].reshape(-1)[m] = 1.0
    p[2 * HISTORY_PLY] = 1.0
    return p


def test_default_off_and_configure():
    assert sp._ATTACKER_PRESERVE_ENABLED is False      # default OFF = byte-identical
    sp.configure_attacker_preserve(enabled=True)
    assert sp._ATTACKER_PRESERVE_ENABLED is True
    sp.configure_attacker_preserve(enabled=None)       # None leaves it unchanged
    assert sp._ATTACKER_PRESERVE_ENABLED is True
    sp.configure_attacker_preserve(enabled=False)
    assert sp._ATTACKER_PRESERVE_ENABLED is False


def test_preserve_policy_masks_to_winning_moves():
    # pi has mass on 3, 7, 11; only 3 and 7 preserve the forced win -> 11 dropped.
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[3], pi[7], pi[11] = 0.5, 0.3, 0.2
    pmask = np.zeros(N_ACTIONS, dtype=bool)
    pmask[[3, 7]] = True
    out = sp._preserve_policy(pi, pmask)
    assert out[11] == 0.0
    np.testing.assert_allclose(out[3], 0.5 / 0.8, rtol=1e-6)
    np.testing.assert_allclose(out[7], 0.3 / 0.8, rtol=1e-6)
    np.testing.assert_allclose(out.sum(), 1.0, rtol=1e-6)


def test_preserve_policy_fallback_uniform_over_winmask():
    # Search put ALL its mass on non-winning moves -> uniform over winning moves.
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[11] = 1.0
    pmask = np.zeros(N_ACTIONS, dtype=bool)
    pmask[[3, 7]] = True
    out = sp._preserve_policy(pi, pmask)
    assert out[11] == 0.0
    np.testing.assert_allclose(out[3], 0.5, rtol=1e-6)
    np.testing.assert_allclose(out[7], 0.5, rtol=1e-6)
    np.testing.assert_allclose(out.sum(), 1.0, rtol=1e-6)


def test_preserve_policy_nan_pi_falls_back_to_winmask():
    pi = np.full(N_ACTIONS, np.nan, dtype=np.float32)
    pmask = np.zeros(N_ACTIONS, dtype=bool)
    pmask[[2, 4]] = True
    out = sp._preserve_policy(pi, pmask)
    np.testing.assert_allclose(out[[2, 4]], [0.5, 0.5], rtol=1e-6)
    assert np.isfinite(out).all()


def test_attacker_preserve_solve_masks_every_proven_win(monkeypatch):
    # Gate is `win` ALONE (NOT win & ~hit_cap): complete-mode enumeration hits
    # the node cap on almost every real win, but win=True is always sound (0 FP)
    # and every winmask bit is an independently proven winning move.
    #   0 = clean win (winmask {3,7});  1 = no win -> None;
    #   2 = win + hit_cap (PARTIAL winmask {9}) -> still masked, soundly.
    win = np.array([True, False, True])
    hit = np.array([False, False, True])
    cells_table = {0: [3, 7], 2: [9]}

    def fake_getter():
        def fake(boards, *, max_nodes, complete, return_move):
            assert complete is True and return_move is True
            B = boards.shape[0]
            wm = np.zeros((B, 4), dtype=np.uint64)
            for i in range(B):
                wm[i, 0] = i           # encode the position index for the fake unpacker
            move = np.full(B, -1, dtype=np.int64)
            return win, hit, move, wm
        return fake

    monkeypatch.setattr(sp, "_preserve_solver", fake_getter)
    monkeypatch.setattr(sp, "_cells_from_words",
                        lambda words: cells_table.get(int(words[0]), []))

    planes = [_planes_with_stones(me_flat=(0,), opp_flat=(1,)) for _ in range(3)]
    out = sp._attacker_preserve_solve(planes)

    assert len(out) == 3
    assert out[1] is None              # no proven win -> no mask
    assert out[0] is not None and out[0].dtype == bool
    assert out[0][3] and out[0][7] and int(out[0].sum()) == 2
    assert out[2] is not None          # capped win STILL masks (partial winmask)
    assert out[2][9] and int(out[2].sum()) == 1


def test_attacker_preserve_solve_empty_winmask_is_none(monkeypatch):
    # A clean win whose winmask unpacks to no cells -> None (no mask that ply).
    win = np.array([True])
    hit = np.array([False])

    def fake_getter():
        def fake(boards, *, max_nodes, complete, return_move):
            B = boards.shape[0]
            return win, hit, np.full(B, -1), np.zeros((B, 4), dtype=np.uint64)
        return fake

    monkeypatch.setattr(sp, "_preserve_solver", fake_getter)
    monkeypatch.setattr(sp, "_cells_from_words", lambda words: [])
    out = sp._attacker_preserve_solve([_planes_with_stones(me_flat=(0,), opp_flat=(1,))])
    assert out == [None]
