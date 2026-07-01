"""Batched eval arena (gomoku/arena.py, issue #105).

CPU-only: agents here are the random/heuristic pickers and stub evaluators —
no checkpoint, no torch device work, no external engine. The real-model MPS
speedup is measured operationally, not asserted in unit tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku.arena import (
    NetAgent,
    PickerAgent,
    _build_slots,
    build_agent,
    play_matches_batched,
)
from gomoku.baselines import heuristic_player, random_player
from gomoku.eval import play_match_pickers
from gomoku.game import GameState
from gomoku.mcts import N_ACTIONS, make_random_evaluator


def _rand_agent():
    return PickerAgent(random_player, label="random")


def _heur_agent():
    return PickerAgent(heuristic_player, label="heuristic")


# ---------------- result semantics ----------------


def test_arena_result_invariants():
    r = play_matches_batched(_rand_agent(), _heur_agent(), n_games=8, seed=3)
    assert r.n_games == 8
    assert r.wins + r.losses + r.draws == 8
    # Color split tallies the same outcomes.
    assert r.black_w + r.white_w == r.wins
    assert r.black_l + r.white_l == r.losses
    assert r.black_d + r.white_d == r.draws
    # A alternates colors: 4 games as black, 4 as white.
    assert r.black_w + r.black_l + r.black_d == 4
    assert r.white_w + r.white_l + r.white_d == 4


def test_arena_heuristic_beats_random():
    r = play_matches_batched(_heur_agent(), _rand_agent(), n_games=10, seed=0)
    assert r.win_rate > 0.7


def test_arena_deterministic_under_seed():
    a = play_matches_batched(_rand_agent(), _rand_agent(), n_games=6, seed=42)
    b = play_matches_batched(_rand_agent(), _rand_agent(), n_games=6, seed=42)
    assert a == b


def test_arena_win_accounting_matches_sequential_for_deterministic_players():
    # Heuristic vs heuristic is fully deterministic (no RNG consumed), so the
    # arena must reproduce the sequential engine's outcomes exactly game by
    # game — same openings (none), same seat alternation, same scoring.
    seq = play_match_pickers(heuristic_player, heuristic_player, n_games=4, seed=0)
    arena = play_matches_batched(_heur_agent(), _heur_agent(), n_games=4, seed=0)
    assert (arena.wins, arena.losses, arena.draws) == (seq.wins, seq.losses, seq.draws)
    assert (arena.black_w, arena.white_w) == (seq.black_w, seq.white_w)


# ---------------- slot setup ----------------


def test_slots_alternate_colors_and_start_empty():
    slots = _build_slots(4, seed=0)
    assert [s.a_is_black for s in slots] == [True, False, True, False]
    assert all(s.state.move_count == 0 for s in slots)
    assert [s.a_to_move for s in slots] == [True, False, True, False]


def test_slots_pair_share_random_opening():
    slots = _build_slots(6, seed=7, random_opening_moves=4)
    for k in range(3):
        a, b = slots[2 * k], slots[2 * k + 1]
        assert np.array_equal(a.state.board, b.state.board)
        assert a.a_is_black and not b.a_is_black
    # Different pairs get different openings (overwhelmingly likely).
    assert not np.array_equal(slots[0].state.board, slots[2].state.board)


def test_slots_odd_ply_start_state_flips_mover():
    start = GameState.initial().apply(40)  # 1 stone -> white to move
    slots = _build_slots(2, seed=0, start_state=start)
    # Game 0: A is black but white moves next -> B to move first.
    assert slots[0].a_is_black and not slots[0].a_to_move
    # Game 1: A is white -> A to move first.
    assert not slots[1].a_is_black and slots[1].a_to_move


# ---------------- cross-game batching ----------------


class _CountingEvaluator:
    """Uniform-prior evaluator that records every batch width."""

    def __init__(self):
        self.batch_sizes: list[int] = []
        self._inner = make_random_evaluator()

    def __call__(self, states):
        self.batch_sizes.append(len(states))
        return self._inner(states)


def test_net_agent_batches_across_games():
    ev = _CountingEvaluator()
    agent = NetAgent(ev, sims=16, wave_size=8, label="net")
    r = play_matches_batched(agent, _rand_agent(), n_games=8, seed=1)
    assert r.n_games == 8
    # The whole point: leaf evals must be batched well beyond one board.
    assert max(ev.batch_sizes) > 8


def test_net_agent_moves_are_legal():
    ev = make_random_evaluator()
    agent = NetAgent(ev, sims=8, wave_size=4, label="net")
    states = [GameState.initial(), GameState.initial().apply(0).apply(1)]
    rngs = [np.random.default_rng(i) for i in range(2)]
    actions = agent.pick_batch(states, rngs)
    for s, a in zip(states, actions):
        assert 0 <= a < N_ACTIONS
        assert bool(np.asarray(s.legal_mask()).reshape(-1)[a])


def test_arena_far_fewer_evaluator_calls_than_sequential():
    # Sequential engine: one evaluator call per simulation per move per game.
    # Arena: calls shrink by ~(n_games x wave_size). Assert a big gap, not an
    # exact factor (root expansions and endgame stragglers blur the edges).
    from gomoku.eval import mcts_picker

    seq_ev = _CountingEvaluator()
    seq_picker = mcts_picker(seq_ev, n_simulations=16)
    play_match_pickers(seq_picker, random_player, n_games=4, seed=5)

    arena_ev = _CountingEvaluator()
    agent = NetAgent(arena_ev, sims=16, wave_size=8, label="net")
    play_matches_batched(agent, _rand_agent(), n_games=4, seed=5)

    assert len(arena_ev.batch_sizes) < len(seq_ev.batch_sizes) / 4


# ---------------- spec parsing ----------------


def test_build_agent_picker_kinds():
    assert isinstance(build_agent("random"), PickerAgent)
    a = build_agent("lookahead:depth=2")
    assert isinstance(a, PickerAgent)
    assert a.label == "lookahead:depth=2"


def test_build_agent_model_requires_checkpoint():
    with pytest.raises(SystemExit):
        build_agent("model:sims=50")


def test_rapfi_sugar_parses(monkeypatch):
    # Intercept RapfiPool so no engine binary is needed.
    captured = {}

    class _FakePool:
        def __init__(self, **kw):
            captured.update(kw)

        def close(self):
            pass

    import gomoku.rapfi_pool as rp

    monkeypatch.setattr(rp, "RapfiPool", _FakePool)
    agent = build_agent("rapfi@50ms", engine_pool_size=4)
    assert captured["timeout_ms"] == 50
    assert captured["size"] == 4
    assert agent.label == "rapfi@50ms"
    agent.close()


# ---------------- explicit opening_states (the H2H gate seam, #106) ----------------


def test_slots_explicit_opening_states():
    s1 = GameState.initial().apply(10).apply(20)
    s2 = GameState.initial().apply(30).apply(40)
    slots = _build_slots(4, seed=0, opening_states=[s1, s2])
    assert np.array_equal(slots[0].state.board, s1.board)
    assert np.array_equal(slots[1].state.board, s1.board)
    assert np.array_equal(slots[2].state.board, s2.board)
    assert np.array_equal(slots[3].state.board, s2.board)
    # Even-ply openings: black to move, so A moves first iff A is black.
    assert [s.a_to_move for s in slots] == [True, False, True, False]


def test_slots_opening_states_too_short_raises():
    with pytest.raises(ValueError):
        _build_slots(6, seed=0, opening_states=[GameState.initial()])


def test_h2h_gate_arena_path(monkeypatch):
    # head_to_head_eval(use_arena=True) end-to-end with stub agents — no torch
    # model, no checkpoint files. Verifies the gate's arena wiring: opening
    # regeneration, slot layout, tally mapping into HeadToHeadResult.
    import gomoku.arena as arena_mod
    from scripts.delta_e_harness import head_to_head_eval

    def _fake_net_agent(ckpt, *, sims, c_puct, device=None, label=None):
        picker = heuristic_player if ckpt == "fork.pt" else random_player
        return PickerAgent(picker, label=label or ckpt)

    monkeypatch.setattr(arena_mod, "net_agent_from_checkpoint", _fake_net_agent)
    res = head_to_head_eval(
        "fork.pt", "c.pt", recipe_label="stub", window_epochs=0, wall_secs=None,
        n_games=8, sims=10, c_puct=1.5, seed=0, use_arena=True)
    assert res.n_games == 8
    assert res.wins + res.draws + res.losses == 8
    # heuristic fork should dominate the random parent.
    assert res.win_rate > 0.7
    assert res.delta_elo > 0


def test_h2h_opening_states_match_legacy_derivation():
    # The arena path must play the SAME openings a legacy run at this seed
    # would: pair k's opening = _random_opening_state(default_rng(seed + k)).
    from gomoku.self_play import _random_opening_state
    from scripts.delta_e_harness import _h2h_opening_states

    seed, plies = 123, 4
    got = _h2h_opening_states(3, plies, seed)
    for k in range(3):
        want, _ = _random_opening_state(np.random.default_rng(seed + k), plies)
        assert np.array_equal(got[k].board, want.board)
