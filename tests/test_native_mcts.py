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


def test_native_policy_finite_at_low_temperature_with_concentrated_visits():
    """Regression: pow(N, 1/tau) in the tau != 1.0 path used to cast a
    finite double to float32 *before* normalizing by the sum, which
    overflowed to +Inf when N was large enough (e.g. N ≈ 7200 at tau=0.1
    gives pow=3.7e38, above FLT_MAX). That produced +Inf entries in the
    emitted policy and ultimately NaN in `_sample_action`'s
    `pi/s` divide (Inf/Inf=NaN).

    This is the exact failure that killed all 8 WL3 self-play workers
    around e825 (wandb 0o75gws5). Concretely we feed an evaluator with a
    near-deterministic prior and run enough sims that one root child
    accumulates ~8000 visits, then ask for the tau=0.1 policy. The
    policy must be finite (no Inf, no NaN) and sum to ~1.
    """
    def peaked_evaluator(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = planes.shape[0]
        priors = np.full((batch, 81), -10.0, dtype=np.float32)
        priors[:, 40] = 10.0  # heavy peak at center
        values = np.zeros((batch,), dtype=np.float32)
        return priors, values

    game = native_mcts.NativeMCTSGame(
        GameState.initial(),
        dirichlet_alpha=0.01,
        dirichlet_eps=0.01,
        seed=1,
    )
    native_mcts.search_batch(
        [game],
        peaked_evaluator,
        n_simulations=8000,
        wave_size=64,
        add_root_noise=True,
    )

    visits = game.visit_counts()
    # We need at least one child to have crossed the float32-overflow
    # threshold for pow(N, 10), otherwise the regression isn't exercised.
    assert int(visits.max()) >= 7100, (
        f"need a concentrated visit count to trigger the overflow path; "
        f"got max={int(visits.max())}"
    )

    for tau in (0.1, 0.01):
        pi = game.policy(temperature=tau)
        assert np.all(np.isfinite(pi)), (
            f"policy at tau={tau} contains non-finite entries: "
            f"{pi[~np.isfinite(pi)]}"
        )
        assert np.isclose(pi.sum(), 1.0, atol=1e-5), (
            f"policy at tau={tau} doesn't sum to 1: sum={pi.sum()}"
        )


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
