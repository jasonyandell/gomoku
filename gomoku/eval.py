"""Evaluation: play matches between two policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gomoku.game import GameState
from gomoku.mcts import Evaluator, MCTSGame, policy_from_visits, run_batched_mcts


@dataclass
class MatchResult:
    n_games: int
    wins: int
    losses: int
    draws: int

    @property
    def win_rate(self) -> float:
        # Draws count as half-wins, standard chess scoring.
        return (self.wins + 0.5 * self.draws) / max(self.n_games, 1)


def _random_evaluator_pick(state: GameState, rng: np.random.Generator) -> int:
    legal = state.legal_actions()
    return int(rng.choice(legal))


def _mcts_pick(
    state: GameState,
    evaluator: Evaluator,
    *,
    n_simulations: int,
    c_puct: float,
    rng: np.random.Generator,
) -> int:
    g = MCTSGame(state, c_puct=c_puct, rng=rng)
    run_batched_mcts([g], evaluator, n_simulations=n_simulations, add_root_noise=False)
    pi = policy_from_visits(g.root, temperature=0.0)
    return int(np.argmax(pi))


def play_vs_random(
    evaluator: Evaluator,
    *,
    n_games: int,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    seed: int = 0,
) -> MatchResult:
    """Play `n_games` against a random opponent, alternating colors. Returns result for the model."""
    rng = np.random.default_rng(seed)
    wins = losses = draws = 0
    for g_idx in range(n_games):
        state = GameState.initial()
        model_is_black = (g_idx % 2 == 0)
        model_to_move = model_is_black
        # Track outcome from "black"'s perspective so we can convert per-side later.
        winner_side: int | None = None  # 0=black, 1=white
        ply = 0
        while True:
            if model_to_move:
                action = _mcts_pick(state, evaluator, n_simulations=n_simulations, c_puct=c_puct, rng=rng)
            else:
                action = _random_evaluator_pick(state, rng)
            side_just_moved = ply % 2
            state = state.apply(action)
            done, v = state.is_terminal()
            if done:
                if v == -1.0:
                    winner_side = side_just_moved
                break
            model_to_move = not model_to_move
            ply += 1

        if winner_side is None:
            draws += 1
        else:
            model_side = 0 if model_is_black else 1
            if winner_side == model_side:
                wins += 1
            else:
                losses += 1

    return MatchResult(n_games=n_games, wins=wins, losses=losses, draws=draws)


def play_match(
    eval_a: Evaluator,
    eval_b: Evaluator,
    *,
    n_games: int,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    seed: int = 0,
) -> MatchResult:
    """Play A vs B, alternating colors. Returns result from A's perspective."""
    rng = np.random.default_rng(seed)
    wins = losses = draws = 0
    for g_idx in range(n_games):
        a_is_black = (g_idx % 2 == 0)
        a_to_move = a_is_black
        state = GameState.initial()
        winner_side: int | None = None
        ply = 0
        while True:
            evaluator = eval_a if a_to_move else eval_b
            action = _mcts_pick(state, evaluator, n_simulations=n_simulations, c_puct=c_puct, rng=rng)
            side_just_moved = ply % 2
            state = state.apply(action)
            done, v = state.is_terminal()
            if done:
                if v == -1.0:
                    winner_side = side_just_moved
                break
            a_to_move = not a_to_move
            ply += 1

        if winner_side is None:
            draws += 1
        else:
            a_side = 0 if a_is_black else 1
            if winner_side == a_side:
                wins += 1
            else:
                losses += 1

    return MatchResult(n_games=n_games, wins=wins, losses=losses, draws=draws)
