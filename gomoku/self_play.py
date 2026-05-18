"""Parallel self-play game generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gomoku.game import GameState, augment
from gomoku.mcts import (
    Evaluator,
    MCTSGame,
    policy_from_visits,
    run_batched_mcts,
)


@dataclass
class SelfPlayExample:
    """One training example: a canonical state, MCTS policy, and outcome.

    All three are from the SAME side-to-move perspective (canonical).
    """

    planes: np.ndarray   # (3, 9, 9) float32
    pi: np.ndarray       # (81,) float32, sums to 1 over legal actions
    z: float             # value in [-1, 1]


@dataclass
class GameRecord:
    """A completed game's training examples plus some metadata."""

    examples: list[SelfPlayExample]
    plies: int
    outcome: float  # +1 if first-mover (black) won, -1 if second-mover (white) won, 0 draw


def _sample_action(pi: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an action from a probability distribution. Falls back to argmax if needed."""
    s = pi.sum()
    if s <= 0:
        return int(np.argmax(pi))
    pi = pi / s
    return int(rng.choice(len(pi), p=pi))


def generate_games(
    n_games: int,
    evaluator: Evaluator,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.5,
    temperature_moves: int = 8,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    max_plies: int = 81,
    rng: np.random.Generator | None = None,
    augment_symmetries: bool = True,
) -> list[GameRecord]:
    """Generate `n_games` self-play games in parallel.

    All games advance in lockstep: at each ply we batch-MCTS across active games,
    sample an action per game, apply it, and remove any games that ended.
    """
    rng = rng or np.random.default_rng()

    games = [MCTSGame(GameState.initial(), c_puct=c_puct,
                      dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
                      rng=np.random.default_rng(rng.integers(0, 2**31)))
             for _ in range(n_games)]

    # Per-game trajectory of (planes, pi, side_to_move_at_that_ply)
    # side_to_move is encoded as 0 for the player who moved first ("black"), 1 for the other.
    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    active: list[int] = list(range(n_games))
    completed: list[tuple[int, float, int]] = []  # (game_idx, outcome_for_black, plies)

    ply = 0
    while active and ply < max_plies:
        active_games = [games[i] for i in active]
        run_batched_mcts(
            active_games,
            evaluator,
            n_simulations=n_simulations,
            add_root_noise=True,
        )

        next_active: list[int] = []
        for slot_idx, g_idx in enumerate(active):
            g = active_games[slot_idx]
            tau = 1.0 if ply < temperature_moves else 0.0
            pi = policy_from_visits(g.root, tau)
            # Whose move was this? Plies are 0-indexed; even ply = "black" (first mover).
            side = ply % 2
            trajectories[g_idx].append((g.root.state.to_planes(), pi.copy(), side))

            action = _sample_action(pi, rng)
            g.advance_root(action)
            done, term_val = g.root.state.is_terminal()
            if done:
                # term_val is from the NEW root's side-to-move perspective.
                # The new side-to-move is the player who DIDN'T just move.
                # If term_val == -1, the player who just moved (side) won.
                # If term_val == 0, draw.
                if term_val == -1.0:
                    winner_side = side
                    outcome_for_black = 1.0 if winner_side == 0 else -1.0
                else:
                    outcome_for_black = 0.0
                completed.append((g_idx, outcome_for_black, ply + 1))
            else:
                next_active.append(g_idx)

        active = next_active
        ply += 1

    # Any games still active at max_plies are scored as draws.
    for g_idx in active:
        completed.append((g_idx, 0.0, ply))

    # Build records, applying symmetry augmentation.
    records: list[GameRecord] = []
    for g_idx, outcome_for_black, plies in sorted(completed):
        examples: list[SelfPlayExample] = []
        for planes, pi, side in trajectories[g_idx]:
            # z from this ply's side-to-move perspective
            z = outcome_for_black if side == 0 else -outcome_for_black
            if augment_symmetries:
                for aug_planes, aug_pi in augment(planes, pi):
                    examples.append(SelfPlayExample(aug_planes, aug_pi.astype(np.float32), z))
            else:
                examples.append(SelfPlayExample(planes, pi, z))
        records.append(GameRecord(examples=examples, plies=plies, outcome=outcome_for_black))

    return records
