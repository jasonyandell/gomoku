"""Microbench + profiler for the lookahead_player alpha-beta eval path.

Generates a deterministic set of realistic midgame positions (heuristic vs
heuristic, varied ply depth) and times lookahead_player(depth=d) per move.
This is the perf-lab baseline harness for the lookahead-eval lane.

Usage:
    python bench_lookahead.py            # timing summary, depths 2 and 4
    python bench_lookahead.py --profile  # cProfile breakdown at depth 4
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/jason/code/gomoku")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gomoku.baselines import heuristic_player, lookahead_player
from gomoku.game import GameState


def gen_positions(n_positions: int, seed: int = 0) -> list[GameState]:
    """Play heuristic-vs-heuristic games, snapshotting non-terminal midgame
    positions at a spread of ply depths. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    positions: list[GameState] = []
    # Snapshot at these ply counts to span opening->midgame->late shapes.
    snap_plies = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24]
    games_needed = (n_positions // len(snap_plies)) + 1
    for g in range(games_needed):
        state = GameState.initial()
        ply = 0
        snaps_this_game = set(snap_plies)
        while True:
            done, _ = state.is_terminal()
            if done:
                break
            if ply in snaps_this_game:
                positions.append(state)
                snaps_this_game.discard(ply)
            if not snaps_this_game or ply > max(snap_plies):
                break
            a = heuristic_player(state, rng)
            state = state.apply(int(a))
            ply += 1
    return positions[:n_positions]


def time_depth(positions: list[GameState], depth: int, seed: int = 1) -> dict:
    player = lookahead_player(depth=depth)
    rng = np.random.default_rng(seed)
    per_move = []
    t_all0 = time.perf_counter()
    for st in positions:
        t0 = time.perf_counter()
        player(st, rng)
        per_move.append(time.perf_counter() - t0)
    total = time.perf_counter() - t_all0
    pm = np.array(per_move)
    return {
        "depth": depth,
        "n": len(positions),
        "total_s": total,
        "mean_ms": pm.mean() * 1e3,
        "median_ms": np.median(pm) * 1e3,
        "p95_ms": np.percentile(pm, 95) * 1e3,
        "max_ms": pm.max() * 1e3,
        "moves_per_s": len(positions) / total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-positions", type=int, default=60)
    ap.add_argument("--depths", type=str, default="2,4")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    positions = gen_positions(args.n_positions, seed=args.seed)
    print(f"# generated {len(positions)} midgame positions (seed={args.seed})")

    if args.profile:
        player = lookahead_player(depth=args.profile_depth)
        rng = np.random.default_rng(1)
        # warm
        player(positions[0], rng)
        pr = cProfile.Profile()
        pr.enable()
        for st in positions:
            player(st, rng)
        pr.disable()
        st = pstats.Stats(pr).sort_stats("tottime")
        print(f"\n# cProfile (depth={args.profile_depth}, {len(positions)} moves), top 25 by tottime:")
        st.print_stats(25)
        return

    depths = [int(d) for d in args.depths.split(",")]
    # warmup each depth once (JIT-free but warms caches / numpy)
    for d in depths:
        lookahead_player(depth=d)(positions[0], np.random.default_rng(99))
    print(f"\n{'depth':>5} {'n':>4} {'total_s':>9} {'mean_ms':>9} {'median_ms':>10} {'p95_ms':>9} {'max_ms':>9} {'moves/s':>9}")
    for d in depths:
        r = time_depth(positions, d, seed=1)
        print(f"{r['depth']:>5} {r['n']:>4} {r['total_s']:>9.3f} {r['mean_ms']:>9.2f} "
              f"{r['median_ms']:>10.2f} {r['p95_ms']:>9.2f} {r['max_ms']:>9.2f} {r['moves_per_s']:>9.1f}")


if __name__ == "__main__":
    main()
