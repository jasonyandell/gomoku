"""Arena batched VCT finisher (issue #109).

CPU-only: the GPU oracle is monkeypatched. The real-oracle hybrid win-rate
jump is measured operationally (the #107 battery: bare 5W vs hybrid 14W over
heuristic), not asserted here.
"""
from __future__ import annotations

import numpy as np
import pytest

import gomoku.eval as geval
from gomoku.arena import NetAgent, build_agent
from gomoku.game import GameState
from gomoku.mcts import N_ACTIONS, make_random_evaluator


class _CountingEvaluator:
    def __init__(self):
        self.calls = 0
        self._inner = make_random_evaluator()

    def __call__(self, states):
        self.calls += 1
        return self._inner(states)


def _fake_solver(win_for: set[int]):
    """Oracle stub: boards whose index is in `win_for` have a forced VCT whose
    winning move is that board's first legal cell."""

    def solve(boards, max_nodes=0, return_move=True):
        boards = np.asarray(boards)
        B = boards.shape[0]
        win = np.zeros(B, dtype=bool)
        move = np.full(B, -1, dtype=np.int64)
        for i in range(B):
            if i in win_for:
                win[i] = True
                occupied = boards[i, 0] | boards[i, 1]
                move[i] = int(np.flatnonzero(~occupied.reshape(-1))[0])
        return win, np.zeros(B, dtype=bool), move

    return solve


def test_finisher_plays_oracle_move_and_skips_search(monkeypatch):
    monkeypatch.setattr(geval, "_load_vct_solver", lambda: _fake_solver({0, 1}))
    ev = _CountingEvaluator()
    agent = NetAgent(ev, sims=8, wave_size=4, vct_finish=50, label="net")
    states = [GameState.initial(), GameState.initial().apply(40)]
    rngs = [np.random.default_rng(i) for i in range(2)]
    picks = agent.pick_batch(states, rngs)
    assert picks[0] == 0          # first legal cell of an empty board
    assert picks[1] == 0          # cell 40 occupied -> first legal is still 0
    assert agent.vct_finish_fired == 2
    assert ev.calls == 0          # every game finished by the oracle: no MCTS


def test_finisher_falls_through_to_search(monkeypatch):
    monkeypatch.setattr(geval, "_load_vct_solver", lambda: _fake_solver({1}))
    ev = _CountingEvaluator()
    agent = NetAgent(ev, sims=8, wave_size=4, vct_finish=50, label="net")
    states = [GameState.initial(), GameState.initial().apply(0).apply(1)]
    rngs = [np.random.default_rng(i) for i in range(2)]
    picks = agent.pick_batch(states, rngs)
    assert agent.vct_finish_fired == 1
    assert ev.calls > 0           # game 0 went through the batched MCTS
    for s, a in zip(states, picks):
        assert 0 <= a < N_ACTIONS
        assert bool(np.asarray(s.legal_mask()).reshape(-1)[a])


def test_finisher_off_is_bare_agent():
    """vct_finish=0 never touches the solver loader (no MLX import)."""
    agent = NetAgent(make_random_evaluator(), sims=8, wave_size=4, label="net")
    assert agent._solver is None
    assert agent.vct_finish_fired == 0


def test_build_agent_rejects_unknown_model_kwargs():
    with pytest.raises(SystemExit, match="not supported by the batched arena"):
        build_agent("model:checkpoint=whatever.pt,reuse_tree=1")


def test_build_agent_parses_vct_finish(monkeypatch):
    seen = {}

    def fake_from_ckpt(checkpoint, **kw):
        seen.update(kw, checkpoint=checkpoint)
        return object()

    import gomoku.arena as arena_mod

    monkeypatch.setattr(arena_mod, "net_agent_from_checkpoint", fake_from_ckpt)
    build_agent("model:checkpoint=x.pt,vct_finish=50,sims=25")
    assert seen["checkpoint"] == "x.pt"
    assert seen["vct_finish"] == 50
    assert seen["sims"] == 25
