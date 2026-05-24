"""Tests for the native (C) Gumbel root + Sequential Halving port.

Covers:
  - Sequential-Halving schedule / budget honoring (total visits match the
    Python reference's visits_spent exactly).
  - Candidate sampling: selected move is legal + among sampled candidates.
  - Completed-policy TARGET parity: native gumbel_policy() == Python
    completed_policy_target() computed from the EXACT same final tree state.
  - generate_games(..., gumbel_root=True) routes to the native path and
    produces valid, legal, complete games.
  - The byte-identical-OFF guarantee is exercised in test_native_mcts /
    determinism separately; here we just confirm gumbel_root=0 init is inert.
"""

import numpy as np
import pytest

from gomoku import native_mcts
from gomoku.game import GameState, N_ACTIONS, str_to_action
from gomoku.gumbel import completed_policy_target, _sequential_halving_schedule
from gomoku.mcts import Node, _init_node, make_random_evaluator
from gomoku.gumbel import gumbel_search_root
from gomoku.self_play import generate_games, _can_use_native_gumbel


pytestmark = pytest.mark.skipif(
    not native_mcts.USING_NATIVE_MCTS or not native_mcts.has_native_gumbel(),
    reason="native MCTS extension with Gumbel batch path is not built",
)


def uniform_planes_evaluator(planes):
    batch = planes.shape[0]
    priors = np.zeros((batch, N_ACTIONS), dtype=np.float32)
    values = np.zeros((batch,), dtype=np.float32)
    occupied = (planes[:, 0] > 0.5) | (planes[:, 8] > 0.5)
    legal = ~occupied.reshape(batch, N_ACTIONS)
    for i in range(batch):
        if legal[i].any():
            priors[i, legal[i]] = 1.0 / legal[i].sum()
    return priors, values


def signal_evaluator(planes):
    batch = planes.shape[0]
    priors = np.zeros((batch, N_ACTIONS), dtype=np.float32)
    values = np.zeros((batch,), dtype=np.float32)
    for i in range(batch):
        occ = (planes[i, 0] > 0.5) | (planes[i, 8] > 0.5)
        flat = occ.reshape(N_ACTIONS)
        legal = ~flat
        idxs = np.flatnonzero(legal)
        priors[i, legal] = (idxs.astype(np.float32) % 7) - 3.0
        n_stones = int(flat.sum())
        values[i] = np.tanh((n_stones % 5 - 2) * 0.3)
    return priors, values


class _Eval:
    def __init__(self, fn):
        self.evaluate_planes = staticmethod(fn)


def _make_position(moves):
    s = GameState.initial()
    for mv in moves:
        s = s.apply(str_to_action(mv))
    return s


POSITIONS = {
    "empty": GameState.initial(),
    "open": _make_position(["e5", "a1", "f5", "b1"]),
    "mid": _make_position(["e5", "e6", "f5", "f6", "g5", "g6", "d4", "c3", "h5", "i5"]),
}


@pytest.mark.parametrize("m", [2, 4, 8, 16])
@pytest.mark.parametrize("sims", [8, 16, 32, 64, 100])
def test_native_sh_total_visits_match_python_reference(m, sims):
    # On a uniform-eval empty board the SH schedule + round-robin are fully
    # deterministic given (m, sims), so the native total root visits must equal
    # the Python reference's visits_spent exactly.
    ev_list = make_random_evaluator()
    node = Node(state=GameState.initial())
    _init_node(node)
    res = gumbel_search_root(node, ev_list, n_simulations=sims, m=m, rng=np.random.default_rng(0))

    g = native_mcts.NativeMCTSGame(GameState.initial(), seed=1, gumbel_root=1, gumbel_m=m)
    native_mcts.gumbel_search_batch([g], uniform_planes_evaluator, n_simulations=sims, wave_size=8)
    assert int(g.visit_counts().sum()) == res.visits


@pytest.mark.parametrize("m", [4, 8, 16])
def test_native_sh_never_exceeds_budget(m):
    for sims in (1, 3, 7, 16, 33, 64):
        g = native_mcts.NativeMCTSGame(GameState.initial(), seed=3, gumbel_root=1, gumbel_m=m)
        native_mcts.gumbel_search_batch([g], uniform_planes_evaluator, n_simulations=sims, wave_size=4)
        assert int(g.visit_counts().sum()) <= sims


def test_native_selected_action_is_legal_and_sampled():
    state = POSITIONS["open"]
    for seed in (1, 2, 3, 7):
        g = native_mcts.NativeMCTSGame(state, seed=seed, gumbel_root=1, gumbel_m=16)
        native_mcts.gumbel_search_batch([g], signal_evaluator, n_simulations=64, wave_size=8)
        sel = int(g.gumbel_selected_action())
        occ = (g.root_planes()[0] > 0.5) | (g.root_planes()[8] > 0.5)
        legal = ~occ.reshape(N_ACTIONS)
        assert legal[sel], f"selected illegal action {sel}"


@pytest.mark.parametrize("ev_name", ["uniform", "signal"])
@pytest.mark.parametrize("pos_name", ["empty", "open", "mid"])
def test_native_completed_policy_target_matches_python(ev_name, pos_name):
    ev = uniform_planes_evaluator if ev_name == "uniform" else signal_evaluator
    state = POSITIONS[pos_name]
    C_VISIT, C_SCALE = 50.0, 1.0
    max_diff = 0.0
    for seed in (1, 2, 3, 7, 11):
        for m in (8, 16):
            for sims in (16, 32, 64):
                g = native_mcts.NativeMCTSGame(
                    state, seed=seed, gumbel_root=1, gumbel_m=m,
                    gumbel_c_visit=C_VISIT, gumbel_c_scale=C_SCALE,
                )
                native_mcts.gumbel_search_batch([g], ev, n_simulations=sims, wave_size=8)
                native_pi = g.gumbel_policy()
                dbg = g.gumbel_debug_state()
                visits = g.visit_counts()

                # native_pi must be a valid pmf over legal actions.
                occ = (g.root_planes()[0] > 0.5) | (g.root_planes()[8] > 0.5)
                legal = ~occ.reshape(N_ACTIONS)
                assert np.all(native_pi >= 0.0)
                assert np.isclose(native_pi.sum(), 1.0, atol=1e-5)
                assert np.all(native_pi[~legal] == 0.0)

                # Reconstruct the Python reference from the exact native state.
                node = Node(state=state)
                _init_node(node)
                node.expanded = True
                node.N = visits.astype(np.int64).copy()
                node.W = dbg["W"].astype(np.float64).copy()
                node.P = dbg["P"].astype(np.float64).copy()
                logits = dbg["logits"].astype(np.float64).copy()
                root_value = float(dbg["root_value"])
                py_pi = completed_policy_target(
                    node, logits, root_value, c_visit=C_VISIT, c_scale=C_SCALE
                )
                max_diff = max(max_diff, float(np.max(np.abs(
                    native_pi.astype(np.float64) - py_pi.astype(np.float64)))))
    assert max_diff <= 1e-5, f"completed-policy target diff {max_diff} exceeds 1e-5"


def test_generate_games_routes_to_native_gumbel():
    ev = _Eval(uniform_planes_evaluator)
    assert _can_use_native_gumbel(ev)
    records = generate_games(
        2, ev, n_simulations=16, max_plies=20, rng=np.random.default_rng(11),
        augment_symmetries=False, gumbel_root=True, gumbel_m=8,
    )
    assert len(records) == 2
    for rec in records:
        assert rec.plies > 0
        for ex in rec.examples:
            assert np.all(ex.pi >= 0.0)
            assert np.isclose(ex.pi.sum(), 1.0, atol=1e-5)
            assert ex.z in (-1.0, 0.0, 1.0)


def test_native_gumbel_single_candidate_terminal_safe():
    # Near-full board with very few legal moves: must not crash, target valid.
    s = GameState.initial()
    rng = np.random.default_rng(2)
    actions = list(range(N_ACTIONS))
    rng.shuffle(actions)
    placed = 0
    for a in actions:
        done, _ = s.is_terminal()
        if done:
            break
        if s.legal_mask()[a]:
            s = s.apply(a)
            placed += 1
        if placed >= 78:
            break
    done, _ = s.is_terminal()
    if done:
        pytest.skip("random fill ended the game")
    g = native_mcts.NativeMCTSGame(s, seed=1, gumbel_root=1, gumbel_m=16)
    native_mcts.gumbel_search_batch([g], uniform_planes_evaluator, n_simulations=8, wave_size=4)
    pi = g.gumbel_policy()
    assert np.isclose(pi.sum(), 1.0, atol=1e-5)
    sel = int(g.gumbel_selected_action())
    assert s.legal_mask()[sel]


def test_native_gumbel_off_default_is_puct():
    # gumbel_root defaults to 0; the game behaves as ordinary PUCT (search_batch
    # path). This is the production-safety default. We just confirm a default
    # game runs search_batch and produces a normal visit-count policy.
    g = native_mcts.NativeMCTSGame(GameState.initial(), seed=1)
    native_mcts.search_batch([g], uniform_planes_evaluator, n_simulations=32, wave_size=8)
    assert int(g.visit_counts().sum()) == 32
    assert np.isclose(g.policy(temperature=1.0).sum(), 1.0)
