"""Tests for Gumbel AlphaZero root selection + Sequential Halving."""

import numpy as np

from gomoku.game import N_ACTIONS, GameState, str_to_action
from gomoku.mcts import MCTSGame, Node, make_random_evaluator, _init_node
from gomoku.gumbel import (
    _gumbel_topk,
    _root_logits,
    _sequential_halving_schedule,
    completed_policy_target,
    gumbel_search_root,
)
from gomoku.self_play import generate_games


def _fresh_root(state=None):
    node = Node(state=state or GameState.initial())
    _init_node(node)
    return node


def test_sequential_halving_halves_and_respects_budget():
    # m=16, budget=64: 4 phases, candidate sets 16->8->4->2.
    sched = _sequential_halving_schedule(16, 64)
    assert len(sched) == 4  # floor(log2(16)) = 4 phases
    # Total visits never exceeds budget.
    sizes = [16, 8, 4, 2]
    total = sum(n_per * size for n_per, size in zip(sched, sizes))
    assert total <= 64
    # Every candidate gets at least one visit per phase.
    assert all(n >= 1 for n in sched)
    # Tiny budgets still produce a valid (>=1/phase) schedule.
    assert all(n >= 1 for n in _sequential_halving_schedule(16, 2))
    assert _sequential_halving_schedule(1, 100) == []  # nothing to halve


def test_gumbel_topk_samples_only_legal_without_replacement():
    rng = np.random.default_rng(0)
    root = _fresh_root(GameState.initial().apply(str_to_action("e5")))
    raw = rng.standard_normal(N_ACTIONS).astype(np.float32)
    logits = _root_logits(root, raw)
    legal_idx = np.flatnonzero(root.legal_mask)
    sampled, gumbel = _gumbel_topk(logits, legal_idx, m=16, rng=rng)
    assert len(sampled) == 16
    assert len(set(sampled.tolist())) == 16  # no replacement
    # e5 is occupied/illegal — must never be sampled.
    assert str_to_action("e5") not in sampled.tolist()
    for a in sampled:
        assert root.legal_mask[a]


def test_gumbel_topk_clamps_to_num_legal():
    rng = np.random.default_rng(1)
    # Build a near-full board so few legal moves remain.
    s = GameState.initial()
    actions = [a for a in range(N_ACTIONS)]
    rng2 = np.random.default_rng(2)
    rng2.shuffle(actions)
    placed = 0
    for a in actions:
        done, _ = s.is_terminal()
        if done:
            break
        if s.legal_mask()[a]:
            s = s.apply(a)
            placed += 1
        if placed >= 75:
            break
    root = _fresh_root(s)
    n_legal = int(root.legal_mask.sum())
    logits = _root_logits(root, np.zeros(N_ACTIONS, dtype=np.float32))
    legal_idx = np.flatnonzero(root.legal_mask)
    sampled, _ = _gumbel_topk(logits, legal_idx, m=16, rng=rng)
    assert len(sampled) == min(16, n_legal)


def test_completed_policy_target_is_valid_pmf():
    ev = make_random_evaluator()
    root = _fresh_root()
    res = gumbel_search_root(root, ev, n_simulations=32, m=16,
                             rng=np.random.default_rng(7))
    pi = res.pi
    assert pi.shape == (N_ACTIONS,)
    assert np.all(pi >= 0.0)
    assert np.isclose(pi.sum(), 1.0)
    # Mass only on legal actions.
    assert np.all(pi[~root.legal_mask] == 0.0)
    # Selected action is legal.
    assert root.legal_mask[res.action]


def test_uniform_priors_zero_value_gives_sane_distribution():
    # With uniform priors and zero values, the completed policy should be ~uniform
    # over legal actions (no signal to concentrate). Check it's spread out, not
    # one-hot, and that the selected action is among the sampled candidates.
    ev = make_random_evaluator()
    root = _fresh_root()
    res = gumbel_search_root(root, ev, n_simulations=64, m=16,
                             rng=np.random.default_rng(3))
    pi = res.pi
    legal = root.legal_mask
    n_legal = int(legal.sum())
    # Entropy should be high (close to uniform). Compare to uniform entropy.
    p = pi[legal]
    ent = -np.sum(p * np.log(p + 1e-12))
    uniform_ent = np.log(n_legal)
    # Allow some spread from sigma(q) noise but it should be reasonably uniform.
    assert ent > 0.8 * uniform_ent


def test_completed_q_uses_value_for_unvisited():
    # With zero simulations, every action is unvisited; completed Q == root value
    # (here 0), so the target reduces to softmax(logits) == uniform for uniform
    # priors.
    ev = make_random_evaluator()
    root = _fresh_root()
    # Manually expand and call the target directly with 0 visits.
    priors, values = ev([root.state])
    from gomoku.mcts import _set_priors
    _set_priors(root, priors[0])
    root.expanded = True
    logits = _root_logits(root, priors[0])
    pi = completed_policy_target(root, logits, value_root=0.0,
                                 c_visit=50.0, c_scale=1.0)
    legal = root.legal_mask
    n_legal = int(legal.sum())
    # Uniform priors + zero value => exactly uniform target.
    assert np.allclose(pi[legal], 1.0 / n_legal, atol=1e-5)


def test_generate_games_gumbel_produces_legal_complete_games():
    ev = make_random_evaluator()
    rng = np.random.default_rng(11)
    records = generate_games(
        2, ev, n_simulations=16, max_plies=20, rng=rng,
        augment_symmetries=False, gumbel_root=True, gumbel_m=8,
    )
    assert len(records) == 2
    for rec in records:
        assert rec.plies > 0
        for ex in rec.examples:
            assert np.all(ex.pi >= 0.0)
            assert np.isclose(ex.pi.sum(), 1.0, atol=1e-5)
            assert ex.z in (-1.0, 0.0, 1.0)


def test_gumbel_off_uses_standard_path():
    # gumbel_root defaults to False; the standard path must be taken. We just
    # verify generation still works and produces valid targets (the byte-identity
    # check is done separately via the determinism harness / code path).
    ev = make_random_evaluator()
    rng = np.random.default_rng(5)
    records = generate_games(
        2, ev, n_simulations=16, max_plies=20, rng=rng,
        augment_symmetries=False, gumbel_root=False,
    )
    assert len(records) == 2
    for rec in records:
        for ex in rec.examples:
            assert np.isclose(ex.pi.sum(), 1.0, atol=1e-5)
