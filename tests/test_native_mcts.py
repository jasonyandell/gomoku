import numpy as np
import pytest

from gomoku import native_mcts
from gomoku.game import GameState, str_to_action
from gomoku.self_play import generate_games


pytestmark = pytest.mark.skipif(
    not native_mcts.USING_NATIVE_MCTS,
    reason="native MCTS extension is not built",
)


def uniform_planes_evaluator(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch = planes.shape[0]
    priors = np.zeros((batch, 81), dtype=np.float32)
    values = np.zeros((batch,), dtype=np.float32)
    occupied = (planes[:, 0] > 0.5) | (planes[:, 8] > 0.5)
    legal = ~occupied.reshape(batch, 81)
    for i in range(batch):
        if legal[i].any():
            priors[i, legal[i]] = 1.0 / legal[i].sum()
    return priors, values


def test_native_root_planes_match_game_state():
    state = GameState.initial()
    for move in ("e5", "a1", "f5", "b1", "g5"):
        state = state.apply(str_to_action(move))

    game = native_mcts.NativeMCTSGame(state)

    assert np.array_equal(game.root_planes(), state.to_planes())
    assert game.move_count == state.move_count


def test_native_mcts_visits_only_legal_actions():
    state = GameState.initial().apply(str_to_action("e5"))
    game = native_mcts.NativeMCTSGame(state)

    native_mcts.search_batch(
        [game],
        uniform_planes_evaluator,
        n_simulations=50,
        wave_size=8,
        add_root_noise=False,
    )

    visits = game.visit_counts()
    assert visits[str_to_action("e5")] == 0
    assert visits.sum() == 50
    assert np.isclose(game.policy(temperature=1.0).sum(), 1.0)


def test_native_mcts_picks_immediate_win_with_uniform_eval():
    state = GameState.initial()
    for black, white in [
        ("a5", "a1"),
        ("b5", "b1"),
        ("c5", "c1"),
        ("d5", "d1"),
    ]:
        state = state.apply(str_to_action(black))
        state = state.apply(str_to_action(white))

    game = native_mcts.NativeMCTSGame(state)
    native_mcts.search_batch(
        [game],
        uniform_planes_evaluator,
        n_simulations=200,
        wave_size=16,
        add_root_noise=False,
    )

    assert int(np.argmax(game.policy(temperature=0.0))) == str_to_action("e5")


def test_generate_games_native_smoke():
    class Evaluator:
        evaluate_planes = staticmethod(uniform_planes_evaluator)

    records = generate_games(
        2,
        Evaluator(),
        n_simulations=4,
        wave_size=2,
        max_plies=3,
        rng=np.random.default_rng(0),
        augment_symmetries=False,
    )

    assert len(records) == 2
    assert all(record.plies <= 3 for record in records)
    assert all(example.planes.shape == (17, 9, 9) for record in records for example in record.examples)
