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


# ---------------- Multi-process parallel match ----------------
#
# eval_worker plays n=20 games per baseline sequentially on one CPU core.
# When CPU is otherwise idle (trainer + self-play workers are on MPS), we
# can run those games in parallel via multiprocessing.Pool to cut the eval
# pass time by ~n_workers.
#
# The aggregate W/L/D counts are unbiased — each game's RNG is independent
# (seeded from game index + base seed) — but not byte-for-byte identical to
# the sequential path because the sequential version threads ONE shared RNG
# across all games, while the parallel version gives each game its own seed.

# Worker-process globals, populated by `_pool_init`.
_WORKER_MODEL_PICKER: Picker | None = None
_WORKER_OPP_PICKER: Picker | None = None


def _pool_init(checkpoint_path: str | None, sims: int, c_puct: float,
               device: str, opp_spec_str: str) -> None:
    """Run once in each worker on Pool startup. Loads the model + opp picker.

    Each worker holds its own copy of the model (forked-then-loaded, so the
    parent's model isn't copied — workers re-load from disk). Acceptable
    overhead at eval-cycle granularity since the pool only spins up once
    per cycle per baseline.
    """
    global _WORKER_MODEL_PICKER, _WORKER_OPP_PICKER

    # Lazy imports so the worker process doesn't have to ship pytorch around
    # in the fork.
    from gomoku.match import build_player, parse_spec
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import load_checkpoint

    if checkpoint_path:
        model, _ = load_checkpoint(checkpoint_path, device=device)
        model.eval()
        evaluator = make_torch_evaluator(model, device)
        _WORKER_MODEL_PICKER = mcts_picker(
            evaluator, n_simulations=sims, c_puct=c_puct
        )
    _WORKER_OPP_PICKER = build_player(parse_spec(opp_spec_str))


def _pool_play_one(args: tuple[int, int]) -> str:
    """Play one game in a worker. Returns 'win'/'loss'/'draw' from the model's
    perspective. The model is `_WORKER_MODEL_PICKER`, opponent is
    `_WORKER_OPP_PICKER` (both set by `_pool_init`)."""
    g_idx, game_seed = args
    assert _WORKER_MODEL_PICKER is not None and _WORKER_OPP_PICKER is not None
    rng = np.random.default_rng(game_seed)
    a_is_black = (g_idx % 2 == 0)
    a_to_move = a_is_black
    state = GameState.initial()
    winner_side: int | None = None
    ply = 0
    while True:
        picker = _WORKER_MODEL_PICKER if a_to_move else _WORKER_OPP_PICKER
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
        return "draw"
    a_side = 0 if a_is_black else 1
    return "win" if winner_side == a_side else "loss"


def play_match_parallel(
    *,
    checkpoint_path: str | None,
    opp_spec: str,
    n_games: int,
    seed: int = 0,
    n_workers: int = 4,
    sims: int = 100,
    c_puct: float = 1.5,
    device: str = "cpu",
) -> MatchResult:
    """Play `n_games` of the model (loaded from `checkpoint_path`) vs the
    baseline described by `opp_spec`, in parallel via multiprocessing.Pool.

    Result from the model's perspective (wins = model won).

    Aggregate W/L/D counts are equivalent to the sequential `play_match_pickers`
    in expectation, but per-game RNG is independent per worker rather than
    shared across the sequence. That's a correctness-preserving change (each
    game is independent anyway), not a bug.
    """
    import multiprocessing as mp

    if n_workers < 2:
        raise ValueError(f"play_match_parallel needs n_workers >= 2, got {n_workers}")
    init_args = (checkpoint_path, sims, c_puct, device, opp_spec)
    game_args = [(g_idx, seed + g_idx + 1) for g_idx in range(n_games)]
    # spawn (not fork) on macOS by default; spawn re-imports the module in
    # each worker, which is what we want for clean state.
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_pool_init, initargs=init_args) as pool:
        outcomes = pool.map(_pool_play_one, game_args)
    wins = sum(1 for o in outcomes if o == "win")
    losses = sum(1 for o in outcomes if o == "loss")
    draws = sum(1 for o in outcomes if o == "draw")
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
