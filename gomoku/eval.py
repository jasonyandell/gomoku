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
    # Per-color breakdown from the model's (player A's) perspective. A game is
    # counted under "black" if the model played black that game, else "white".
    # These are split tallies of the same outcomes the aggregates count, so by
    # construction black_w + white_w == wins (likewise losses/draws). They
    # default to 0 so existing MatchResult(...) call sites stay valid and the
    # aggregate fields remain the canonical strength signal; the split lets the
    # loss-tail analysis see whether losses cluster on white (second player).
    black_w: int = 0
    black_l: int = 0
    black_d: int = 0
    white_w: int = 0
    white_l: int = 0
    white_d: int = 0

    @property
    def win_rate(self) -> float:
        # Draws count as half-wins, standard chess scoring.
        return (self.wins + 0.5 * self.draws) / max(self.n_games, 1)


def _random_pick(state: GameState, rng: np.random.Generator) -> int:
    legal = state.legal_actions()
    return int(rng.choice(legal))


def vcf_overlay_picker(
    base: Picker,
    *,
    max_nodes: int,
    max_depth: int = 0,
) -> Picker:
    """Wrap a picker with an eval-time root VCF overlay.

    Before delegating to ``base``, run a bounded ``solve_vcf`` from the current
    root. If a forced continuous-four win is proven for the side-to-move, play
    the solver's recommended move; otherwise fall through to ``base`` unchanged.

    When ``max_nodes <= 0`` the overlay is OFF — this function returns ``base``
    UNCHANGED (same object, no wrapping), so OFF is byte-identical to the
    pre-lever path by construction.

    The solver is bounded by ``max_nodes`` (and ``max_depth`` when > 0; else
    ``vcf.DEFAULT_MAX_DEPTH``). Hitting the cap returns ``has_forced_win=False``
    with ``hit_cap=True`` — the overlay simply falls through to ``base``, so a
    tight cap can never block the picker indefinitely.

    This is EVAL-ONLY. The self-play / generation path uses ``--vcf-teacher`` on
    a separate seam (see ``selfplay_worker.py``); the two never collide.
    """
    if max_nodes <= 0:
        return base

    # Lazy import — keeps mcts_picker callers that pass eval_vcf_nodes=0
    # (the default) from paying the import cost.
    from gomoku.vcf import DEFAULT_MAX_DEPTH, solve_vcf

    depth = max_depth if max_depth > 0 else DEFAULT_MAX_DEPTH

    def pick(state: GameState, rng: np.random.Generator) -> int:
        # solve_vcf treats plane 0 as attacker (side-to-move) and plane 1 as
        # defender — which is exactly GameState's canonical board layout.
        res = solve_vcf(state.board, max_depth=depth, max_nodes=max_nodes)
        if res.has_forced_win and res.winning_move is not None:
            return int(res.winning_move)
        return base(state, rng)

    return pick


def mcts_picker(
    evaluator: Evaluator,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.5,
    eval_vcf_nodes: int = 0,
    eval_vcf_depth: int = 0,
) -> Picker:
    """Wrap a leaf evaluator into a one-shot picker for use in matches.

    Builds a fresh MCTS tree per call (no cross-move reuse — matches don't need
    it and this keeps the wrapper simple).

    ``eval_vcf_nodes`` (default 0 = OFF, byte-identical to the pre-lever path)
    enables an eval-time root VCF overlay: before MCTS picks a move, run a
    bounded ``solve_vcf`` from the current root; if a forced four-in-a-row win
    is proven, play that move; else fall through to the unchanged MCTS choice.
    ``eval_vcf_depth`` (default 0 → ``vcf.DEFAULT_MAX_DEPTH``) caps depth. The
    overlay is EVAL-ONLY — never used by self-play / generation / training.
    """

    def pick(state: GameState, rng: np.random.Generator) -> int:
        g = MCTSGame(state, c_puct=c_puct, rng=rng)
        run_batched_mcts([g], evaluator, n_simulations=n_simulations, add_root_noise=False)
        pi = policy_from_visits(g.root, temperature=0.0)
        return int(np.argmax(pi))

    return vcf_overlay_picker(pick, max_nodes=eval_vcf_nodes, max_depth=eval_vcf_depth)


def _match_result_from_outcomes(
    outcomes: list[tuple[bool, str]], n_games: int
) -> MatchResult:
    """Aggregate per-game (a_is_black, outcome) pairs into a MatchResult.

    `outcome` is "win"/"loss"/"draw" from the model's (player A's) perspective;
    `a_is_black` is True when the model played black that game. Produces both the
    aggregate W/L/D and the per-color split, with the invariant that
    black_<x> + white_<x> == aggregate_<x> for each of win/loss/draw.
    """
    wins = losses = draws = 0
    black_w = black_l = black_d = 0
    white_w = white_l = white_d = 0
    for a_is_black, outcome in outcomes:
        if outcome == "win":
            wins += 1
            if a_is_black:
                black_w += 1
            else:
                white_w += 1
        elif outcome == "loss":
            losses += 1
            if a_is_black:
                black_l += 1
            else:
                white_l += 1
        elif outcome == "draw":
            draws += 1
            if a_is_black:
                black_d += 1
            else:
                white_d += 1
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown outcome {outcome!r}")
    return MatchResult(
        n_games=n_games,
        wins=wins,
        losses=losses,
        draws=draws,
        black_w=black_w,
        black_l=black_l,
        black_d=black_d,
        white_w=white_w,
        white_l=white_l,
        white_d=white_d,
    )


def play_match_pickers(
    picker_a: Picker,
    picker_b: Picker,
    *,
    n_games: int,
    seed: int = 0,
) -> MatchResult:
    """Play A vs B, alternating colors. Returns result from A's perspective.

    Draws count as half-wins in `win_rate`. The result also carries a per-color
    split (see `MatchResult`) so analyses can see whether losses cluster on the
    color the model played.
    """
    rng = np.random.default_rng(seed)
    outcomes: list[tuple[bool, str]] = []
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
            outcomes.append((a_is_black, "draw"))
        else:
            a_side = 0 if a_is_black else 1
            outcomes.append((a_is_black, "win" if winner_side == a_side else "loss"))

    return _match_result_from_outcomes(outcomes, n_games)


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
               device: str, opp_spec_str: str,
               eval_vcf_nodes: int = 0, eval_vcf_depth: int = 0) -> None:
    """Run once in each worker on Pool startup. Loads the model + opp picker.

    Each worker holds its own copy of the model (forked-then-loaded, so the
    parent's model isn't copied — workers re-load from disk). Acceptable
    overhead at eval-cycle granularity since the pool only spins up once
    per cycle per baseline.

    ``eval_vcf_nodes`` (default 0 = OFF, byte-identical) threads the eval-time
    root VCF overlay through to the model picker in each worker.
    """
    global _WORKER_MODEL_PICKER, _WORKER_OPP_PICKER

    # Lazy imports so the worker process doesn't have to ship pytorch around
    # in the fork.
    from gomoku.match import build_player, parse_spec
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint

    if checkpoint_path:
        model, _ = load_checkpoint(checkpoint_path, device=device)
        model = fuse_model_for_inference(model)
        evaluator = make_torch_evaluator(model, device)
        _WORKER_MODEL_PICKER = mcts_picker(
            evaluator, n_simulations=sims, c_puct=c_puct,
            eval_vcf_nodes=eval_vcf_nodes, eval_vcf_depth=eval_vcf_depth,
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
    eval_vcf_nodes: int = 0,
    eval_vcf_depth: int = 0,
) -> MatchResult:
    """Play `n_games` of the model (loaded from `checkpoint_path`) vs the
    baseline described by `opp_spec`, in parallel via multiprocessing.Pool.

    Result from the model's perspective (wins = model won).

    Aggregate W/L/D counts are equivalent to the sequential `play_match_pickers`
    in expectation, but per-game RNG is independent per worker rather than
    shared across the sequence. That's a correctness-preserving change (each
    game is independent anyway), not a bug.

    ``eval_vcf_nodes`` (default 0 = OFF, byte-identical) threads the eval-time
    root VCF overlay through into each worker's model picker.
    """
    import multiprocessing as mp

    if n_workers < 2:
        raise ValueError(f"play_match_parallel needs n_workers >= 2, got {n_workers}")
    init_args = (checkpoint_path, sims, c_puct, device, opp_spec,
                 eval_vcf_nodes, eval_vcf_depth)
    game_args = [(g_idx, seed + g_idx + 1) for g_idx in range(n_games)]
    # spawn (not fork) on macOS by default; spawn re-imports the module in
    # each worker, which is what we want for clean state.
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_pool_init, initargs=init_args) as pool:
        outcomes = pool.map(_pool_play_one, game_args)
    # Re-derive each game's model color from its index (same rule the worker
    # used: a_is_black = g_idx % 2 == 0). pool.map preserves input order, so
    # outcomes[i] corresponds to game_args[i].
    color_outcomes = [
        ((g_idx % 2 == 0), outcome)
        for (g_idx, _seed), outcome in zip(game_args, outcomes)
    ]
    return _match_result_from_outcomes(color_outcomes, n_games)


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
