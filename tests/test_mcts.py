import numpy as np

from gomoku.game import GameState, str_to_action
from gomoku.mcts import (
    MCTSGame,
    make_random_evaluator,
    policy_from_visits,
    run_batched_mcts,
)


def test_random_evaluator_priors_only_on_legal():
    ev = make_random_evaluator()
    s = GameState.initial().apply(str_to_action("e5"))
    priors, values = ev([s])
    assert priors.shape == (1, 81)
    # Position e5 is illegal now and should get zero prior.
    assert priors[0, str_to_action("e5")] == 0.0
    # All legal positions should sum to 1.
    assert np.isclose(priors[0].sum(), 1.0)


def test_mcts_visits_only_legal_actions():
    ev = make_random_evaluator()
    s = GameState.initial().apply(str_to_action("e5"))
    g = MCTSGame(s)
    run_batched_mcts([g], ev, n_simulations=50, add_root_noise=False)
    # The illegal action e5 should have zero visits.
    assert g.root.N[str_to_action("e5")] == 0
    # Total visits should equal n_simulations (modulo terminal-skip; here root is far from terminal).
    assert g.root.N.sum() == 50


def test_policy_from_visits_sums_to_one():
    ev = make_random_evaluator()
    g = MCTSGame(GameState.initial())
    run_batched_mcts([g], ev, n_simulations=30, add_root_noise=False)
    pi1 = policy_from_visits(g.root, temperature=1.0)
    pi0 = policy_from_visits(g.root, temperature=0.0)
    assert np.isclose(pi1.sum(), 1.0)
    assert np.isclose(pi0.sum(), 1.0)
    # Greedy policy should be one-hot (or split ties evenly summing to 1).
    assert pi0.max() > 0


def test_mcts_picks_winning_move_with_random_eval():
    """Even with a random-prior evaluator, MCTS should pick the winning move
    if it sees the terminal value from rolling it out."""
    # Set up: black has 4 in a row a5..d5, white to move would block at e5.
    # Then black plays e5 and wins. So if it's BLACK to move with a5..d5
    # already placed, the winning move is e5.
    s = GameState.initial()
    moves = [
        ("a5", "a1"),
        ("b5", "b1"),
        ("c5", "c1"),
        ("d5", "d1"),
    ]
    for b, w in moves:
        s = s.apply(str_to_action(b))
        s = s.apply(str_to_action(w))
    # Now it's black to move and e5 wins immediately.
    ev = make_random_evaluator()
    g = MCTSGame(s)
    run_batched_mcts([g], ev, n_simulations=200, add_root_noise=False)
    pi = policy_from_visits(g.root, temperature=0.0)
    assert int(np.argmax(pi)) == str_to_action("e5")


def test_batched_mcts_across_games():
    ev = make_random_evaluator()
    games = [MCTSGame(GameState.initial()) for _ in range(4)]
    run_batched_mcts(games, ev, n_simulations=20, add_root_noise=False)
    for g in games:
        assert g.root.N.sum() == 20


def test_wave_bfs_matches_sequential_byte_for_byte():
    """The cross-game BFS descent in `run_batched_mcts_waves` must produce the
    same per-action visit counts as a straight sequential wave that calls
    `_select_one_vloss` one descent at a time. Both descents on independent
    trees should be order-invariant; this test catches accidental reorderings
    (e.g., stack ordering, lazy-child-creation drift)."""
    from gomoku.mcts import (
        _backprop_value_only,
        _select_one_vloss,
        _set_priors,
    )
    from gomoku.mcts import run_batched_mcts_waves

    def sequential_reference(games, evaluator, *, n_simulations, wave_size):
        # Mimics the pre-refactor wave loop: one `_select_one_vloss` call at a
        # time across the (game, slot) cross-product, then a single batched
        # eval per wave.
        unexpanded = [g for g in games if not g.root.expanded and not g.root.is_terminal]
        if unexpanded:
            priors, _ = evaluator([g.root.state for g in unexpanded])
            for g, p in zip(unexpanded, priors):
                _set_priors(g.root, p)
                g.root.expanded = True
        sims_done = [0] * len(games)
        while True:
            wave_counts = [min(wave_size, n_simulations - sims_done[i]) for i in range(len(games))]
            if not any(wave_counts):
                return
            pending = []
            # OLD ordering: game-major (game 0 slot 0..W, game 1 slot 0..W, ...).
            # New code does slot-major BFS; per-game tree independence makes this
            # mathematically equivalent. So the test passes only if the BFS path
            # really respects that independence.
            for g_idx, g in enumerate(games):
                for _ in range(wave_counts[g_idx]):
                    pending.append(_select_one_vloss(g.root, g.c_puct, g.c_puct_base))
            to_eval = []
            for p in pending:
                if p.leaf.is_terminal:
                    _backprop_value_only(p.path, p.leaf.terminal_value)
                else:
                    to_eval.append(p)
            if to_eval:
                states = [p.leaf.state for p in to_eval]
                priors, values = evaluator(states)
                for p, prior, value in zip(to_eval, priors, values):
                    if not p.leaf.expanded:
                        _set_priors(p.leaf, prior)
                        p.leaf.expanded = True
                    _backprop_value_only(p.path, float(value))
            for i, w in enumerate(wave_counts):
                sims_done[i] += w

    # Build TWO identical sets of games (same starting state, same RNG seed).
    # Run sequential reference on one, new BFS-vectorized run on the other.
    for seed in (0, 1, 7):
        for wave_size, n_sim in ((1, 20), (8, 64), (16, 64)):
            ev = make_random_evaluator()  # deterministic — uniform priors, no rng draws

            games_ref = [
                MCTSGame(GameState.initial(), rng=np.random.default_rng(seed + i))
                for i in range(4)
            ]
            games_new = [
                MCTSGame(GameState.initial(), rng=np.random.default_rng(seed + i))
                for i in range(4)
            ]

            sequential_reference(games_ref, ev, n_simulations=n_sim, wave_size=wave_size)
            run_batched_mcts_waves(games_new, ev, n_simulations=n_sim, wave_size=wave_size,
                                   add_root_noise=False)

            for g_ref, g_new in zip(games_ref, games_new):
                assert np.array_equal(g_ref.root.N, g_new.root.N), (
                    f"seed={seed} wave={wave_size} sims={n_sim}: N mismatch")
                assert np.allclose(g_ref.root.W, g_new.root.W, atol=1e-6), (
                    f"seed={seed} wave={wave_size} sims={n_sim}: W mismatch")
