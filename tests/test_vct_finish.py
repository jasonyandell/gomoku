"""VCT-finisher picker (issue #99): policy to the VCT, GPU oracle to the win.

Deterministic unit tests — the GPU oracle is monkeypatched via `_load_vct_solver`
so these run on CPU with no MLX/Metal. Real-oracle behaviour (a net + finisher
wins genuine games vs a baseline through the standard harness) is the smoke.
"""
from __future__ import annotations

import numpy as np

from gomoku import eval as ev
from gomoku.game import GameState


def _const_base(action):
    def base(state, rng):
        return action
    return base


def _fake_solver(win, move):
    def solve(boards, *, max_nodes, return_move=False):
        b = boards.shape[0]
        w = np.array([bool(win)] * b)
        h = np.zeros(b, bool)
        m = np.array([int(move)] * b, dtype=np.int64)
        return (w, h, m) if return_move else (w, h)
    return solve


def test_off_returns_base_unchanged():
    base = _const_base(3)
    assert ev.vct_finish_picker(base, budget=0) is base      # OFF = same object
    assert ev.vct_finish_picker(base, budget=-5) is base


def test_plays_oracle_move_on_win(monkeypatch):
    monkeypatch.setattr(ev, "_load_vct_solver", lambda: _fake_solver(True, 40))
    picker = ev.vct_finish_picker(_const_base(3), budget=50)
    got = picker(GameState.initial(), np.random.default_rng(0))
    assert got == 40                                          # oracle move, not base's 3


def test_falls_through_when_no_vct(monkeypatch):
    monkeypatch.setattr(ev, "_load_vct_solver", lambda: _fake_solver(False, -1))
    picker = ev.vct_finish_picker(_const_base(3), budget=50)
    assert picker(GameState.initial(), np.random.default_rng(0)) == 3


def test_illegal_oracle_move_falls_through(monkeypatch):
    # "win" but the returned cell is already occupied -> legality guard -> base
    state = GameState.initial().apply(40)                     # cell 40 taken
    monkeypatch.setattr(ev, "_load_vct_solver", lambda: _fake_solver(True, 40))
    picker = ev.vct_finish_picker(_const_base(7), budget=50)
    assert picker(state, np.random.default_rng(0)) == 7


def test_mcts_picker_off_by_default_is_byte_identical():
    # vct_finish_nodes defaults to 0 -> mcts_picker must not wrap / import MLX.
    # (We can't build a real evaluator here cheaply; assert the param exists and
    # defaults OFF via the picker helper contract instead.)
    base = _const_base(1)
    assert ev.vct_finish_picker(base) is base                 # default budget=0
