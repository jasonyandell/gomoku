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
