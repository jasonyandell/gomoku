"""Evaluation: play matches between two policies.

Two layers:

- `play_match_pickers` is the generic engine. Each player is a
  `Callable[[GameState, np.random.Generator], int]` (a "picker"). Random,
  heuristic, lookahead, and the MCTS-driven model all plug into this.

- `play_match` and `play_vs_random` are thin wrappers kept for backward
  compatibility with the training loop and any existing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from gomoku.game import GameState
from gomoku.mcts import Evaluator, MCTSGame, policy_from_visits, run_batched_mcts


Picker = Callable[[GameState, np.random.Generator], int]


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


def _random_pick(state: GameState, rng: np.random.Generator) -> int:
    legal = state.legal_actions()
    return int(rng.choice(legal))


def mcts_picker(
    evaluator: Evaluator,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.5,
) -> Picker:
    """Wrap a leaf evaluator into a one-shot picker for use in matches.

    Builds a fresh MCTS tree per call (no cross-move reuse — matches don't need
    it and this keeps the wrapper simple).
    """

    def pick(state: GameState, rng: np.random.Generator) -> int:
        g = MCTSGame(state, c_puct=c_puct, rng=rng)
        run_batched_mcts([g], evaluator, n_simulations=n_simulations, add_root_noise=False)
        pi = policy_from_visits(g.root, temperature=0.0)
        return int(np.argmax(pi))

    return pick


def play_match_pickers(
    picker_a: Picker,
    picker_b: Picker,
    *,
    n_games: int,
    seed: int = 0,
) -> MatchResult:
    """Play A vs B, alternating colors. Returns result from A's perspective.

    Draws count as half-wins in `win_rate`.
    """
    rng = np.random.default_rng(seed)
    wins = losses = draws = 0
    for g_idx in range(n_games):
        a_is_black = (g_idx % 2 == 0)
        a_to_move = a_is_black
        state = GameState.initial()
        winner_side: int | None = None
        ply = 0
        while True:
            picker = picker_a if a_to_move else picker_b
            action = picker(state, rng)
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


def play_vs_random(
    evaluator: Evaluator,
    *,
    n_games: int,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    seed: int = 0,
) -> MatchResult:
    """Play `n_games` against a random opponent, alternating colors.

    Thin wrapper over `play_match_pickers` — kept for backward compatibility
    with the training loop.
    """
    model_picker = mcts_picker(evaluator, n_simulations=n_simulations, c_puct=c_puct)
    return play_match_pickers(model_picker, _random_pick, n_games=n_games, seed=seed)


def play_match(
    eval_a: Evaluator,
    eval_b: Evaluator,
    *,
    n_games: int,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    seed: int = 0,
) -> MatchResult:
    """Play A vs B (both MCTS-driven), alternating colors.

    Thin wrapper over `play_match_pickers` — kept for backward compatibility.
    """
    pa = mcts_picker(eval_a, n_simulations=n_simulations, c_puct=c_puct)
    pb = mcts_picker(eval_b, n_simulations=n_simulations, c_puct=c_puct)
    return play_match_pickers(pa, pb, n_games=n_games, seed=seed)
