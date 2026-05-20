"""Calibrate baseline-vs-baseline Elos for the anchor table in gomoku.rating.

Runs a round-robin tournament between the fixed baselines (random, heuristic,
lookahead:depth=2, ...) and fits Elo ratings by maximum likelihood, with
`random` anchored at 0. Prints a Python dict suitable for pasting into
`gomoku/rating.py`'s `ANCHOR_ELOS`.

Anchoring `random` at 0 means every other baseline's rating is its
"Elo over random." That gives a stable scale: if the model's implied Elo
is 1500, the model would beat random about as often as a 1500-vs-0 match
predicts (~99.99% — i.e. crushes random, modest vs lookahead2 ≈ 1200, etc.).

Usage::

    python scripts/calibrate_anchor_elos.py            # default 200 games each pair
    python scripts/calibrate_anchor_elos.py --n 50     # quick rough calibration

Costs scale O(P^2 * n_games). Default P=4 baselines * 6 pairs * 200 games
= 1200 games. Lookahead4 vs lookahead2 dominates wall-clock.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from gomoku.eval import play_match_pickers
from gomoku.match import build_player, parse_spec
from gomoku.rating import expected_score


DEFAULT_BASELINES = [
    "random",
    "heuristic",
    "lookahead:depth=2",
    "lookahead:depth=4",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baselines", type=str, default=",".join(DEFAULT_BASELINES))
    p.add_argument("--n", type=int, default=200,
                   help="Games per directional pair. n=200 gives ~4pp 95%% CI.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--anchor", type=str, default="random",
                   help="Which baseline gets rating=0. Others are relative.")
    return p.parse_args()


def play_pair(picker_a, picker_b, n_games: int, seed: int):
    """Play picker_a vs picker_b with alternating colors. Returns (score_a, n)."""
    res = play_match_pickers(picker_a, picker_b, n_games=n_games, seed=seed)
    score_a = res.wins + 0.5 * res.draws
    return score_a, res.n_games, res


def fit_elos(score_matrix: np.ndarray, n_matrix: np.ndarray, anchor_idx: int,
             *, iters: int = 200, lr: float = 0.5) -> np.ndarray:
    """MLE Bradley-Terry / Elo from a directional score matrix.

    score_matrix[i, j] = total score player i earned against player j (across
    all games where i played).
    n_matrix[i, j] = number of games between i and j (= n_matrix[j, i]).
    anchor_idx: index whose Elo stays at 0.

    Gradient ascent on log-likelihood. Each step nudges every rating toward
    closing its expected-vs-actual gap.
    """
    P = score_matrix.shape[0]
    elos = np.zeros(P, dtype=np.float64)
    for it in range(iters):
        for i in range(P):
            if i == anchor_idx:
                continue
            expected = sum(
                n_matrix[i, j] * expected_score(elos[i], elos[j])
                for j in range(P) if j != i
            )
            actual = sum(score_matrix[i, j] for j in range(P) if j != i)
            n_total = sum(n_matrix[i, j] for j in range(P) if j != i)
            if n_total == 0:
                continue
            # Gradient: actual - expected, scaled by 400/ln(10) for Elo units.
            elos[i] += lr * (actual - expected) * (400.0 / math.log(10)) / n_total
    return elos


def main() -> None:
    args = parse_args()
    specs = [parse_spec(s.strip()) for s in args.baselines.split(",") if s.strip()]
    if not specs:
        raise SystemExit("no baselines")
    if any(s.kind == "model" for s in specs):
        raise SystemExit("model specs not allowed in calibration")
    names = [s.label() for s in specs]
    pickers = [build_player(s) for s in specs]
    if args.anchor not in names:
        raise SystemExit(f"--anchor {args.anchor!r} not in baselines: {names}")
    anchor_idx = names.index(args.anchor)

    P = len(specs)
    print(f"=== Calibrating {P} baselines: {names} ===")
    print(f"    {args.n} games per directional pair, anchor={args.anchor}")

    score_mat = np.zeros((P, P), dtype=np.float64)
    n_mat = np.zeros((P, P), dtype=np.float64)
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            # We only need each unordered pair once, since play_match_pickers
            # already alternates colors. Skip (i, j) where i > j.
            if i > j:
                continue
            score_a, n_games, res = play_pair(pickers[i], pickers[j], args.n,
                                              seed=args.seed + i * 1000 + j)
            score_mat[i, j] = score_a
            score_mat[j, i] = n_games - score_a   # symmetric: opponent's score
            n_mat[i, j] = n_games
            n_mat[j, i] = n_games
            wr = score_a / n_games
            print(f"  {names[i]:<22} vs {names[j]:<22} : "
                  f"{res.wins}W-{res.losses}L-{res.draws}D ({wr:.0%})")

    elos = fit_elos(score_mat, n_mat, anchor_idx)
    print()
    print("=== Calibrated Elos ===")
    for n, e in zip(names, elos):
        print(f"  {n:<22} {e:>7.1f}")

    print()
    print("Python literal for ANCHOR_ELOS in gomoku/rating.py:")
    print("ANCHOR_ELOS: dict[str, float] = {")
    for n, e in zip(names, elos):
        # convert to the canonical key used in rating.ANCHOR_ELOS
        if n.startswith("lookahead("):
            depth = n.split("=")[1].rstrip(")")
            key = f"lookahead{depth}"
        else:
            key = n.split("(")[0]
        print(f'    "{key}": {e:.1f},')
    print("}")


if __name__ == "__main__":
    main()
