"""Elo rating for gomoku eval.

Two intended uses:

1. **Live diagnostic** (called from `eval_worker`). Each eval cycle plays the
   current model against a set of anchor baselines (random, heuristic,
   lookahead:depth=2). The set of `(opponent_rating, score, n_games)` triples
   for that cycle is fed to `implied_elo(...)` which returns the maximum-
   likelihood model Elo given the observed match scores and fixed anchor
   ratings. That number is stateless per cycle — plot it over time to see
   the strength trajectory without the path-dependence of online updates.

2. **Per-game online updates** via `EloRating.update(...)`. Matches the
   michaelnny/alpha_zero implementation (USCF K-factor). Useful when you
   want to run a head-to-head tournament across many checkpoints and let
   their ratings settle through repeated pairings.

Anchor Elos (`ANCHOR_ELOS`) are seeded with hand-picked spacings calibrated
to the typical winrates we observe in our own eval (random loses ~99% to
heuristic ≈ 800 Elo gap, etc.). Run `python -m gomoku.calibrate_elo` to
recompute them from actual baseline-vs-baseline matches.
"""
from __future__ import annotations

import math


# Reference Elos for fixed gomoku baselines. The only "true" anchor is
# random_player = 0 — everything else is calibrated by playing matches.
# These defaults are educated guesses; the model Elo trajectory is informative
# regardless because the offsets are consistent across eval cycles.
ANCHOR_ELOS: dict[str, float] = {
    "random": 0.0,
    "heuristic": 800.0,
    "lookahead2": 1200.0,
    "lookahead4": 1500.0,
    "lookahead6": 1700.0,
}


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for A vs B under standard Elo (logistic, slope 400)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_diff_from_winrate(winrate: float) -> float:
    """Inverse of expected_score: Elo gap implied by an observed winrate.

    winrate=0.5 -> 0; winrate=0.75 -> ~191; winrate=0.99 -> ~798. Clamped to
    [-2000, 2000] to avoid infinities at 0/100% — those bounds correspond
    roughly to <0.005 / >0.995 winrate.
    """
    w = max(min(winrate, 0.999), 0.001)
    return -400.0 * math.log10(1.0 / w - 1.0)


def uscf_k_factor(rating: float) -> float:
    """USCF K-factor by rating band (same as michaelnny/alpha_zero)."""
    if rating < 2100:
        return 32.0
    if rating < 2400:
        return 24.0
    return 16.0


class EloRating:
    """Online Elo with USCF K-factor. Mutable rating field.

    Use for round-robin tournaments where every player's rating moves.
    For "model vs fixed anchors" diagnostic eval, prefer `implied_elo`
    which is stateless and doesn't depend on ordering.
    """

    def __init__(self, rating: float = 1500.0):
        self.rating = float(rating)

    def expected_score(self, opponent_rating: float) -> float:
        return expected_score(self.rating, opponent_rating)

    def update(self, opponent_rating: float, actual_score: float, *,
               k: float | None = None) -> float:
        """Apply one game's result. actual_score: 1=win, 0.5=draw, 0=loss.

        Returns the delta applied. If `k` is None uses USCF K-factor for
        self.rating.
        """
        if not 0.0 <= actual_score <= 1.0:
            raise ValueError(f"actual_score must be in [0, 1], got {actual_score}")
        e = self.expected_score(opponent_rating)
        if k is None:
            k = uscf_k_factor(self.rating)
        delta = k * (actual_score - e)
        self.rating += delta
        return delta


def score_from_match(wins: int, losses: int, draws: int) -> tuple[float, int]:
    """Aggregate match result to (score_sum, n_games). Draws count 0.5."""
    n = wins + losses + draws
    return wins + 0.5 * draws, n


def implied_elo(
    matches: list[tuple[float, float, int]],
    *,
    bounds: tuple[float, float] = (-3000.0, 6000.0),
    tol: float = 0.5,
) -> float:
    """Maximum-likelihood model Elo from a set of matches vs fixed-rating opponents.

    `matches`: list of (opponent_elo, score_sum, n_games) tuples.
        score_sum = wins + 0.5 * draws (i.e. the result of score_from_match).
        n_games = total games in that pairing.

    Solves for the rating R that satisfies
        sum_i n_i * expected_score(R, opp_i) = sum_i score_sum_i
    via bisection on [bounds[0], bounds[1]]. Returns R to ±`tol` Elo.

    Boundary behavior: if the model scored 100% across all matches, returns
    bounds[1]; if 0%, returns bounds[0]. Useful "this model is off the
    scale" signal vs silently fitting a noisy answer.
    """
    if not matches:
        raise ValueError("matches is empty")

    total_score = sum(s for _, s, _ in matches)
    total_games = sum(n for _, _, n in matches)
    if total_games == 0:
        raise ValueError("zero total games across matches")

    # Boundary cases — bisection can't extrapolate beyond 0%/100%.
    if total_score <= 0:
        return bounds[0]
    if total_score >= total_games:
        return bounds[1]

    def deficit(r: float) -> float:
        # Expected total minus actual total. We want zero.
        exp_total = sum(n * expected_score(r, opp) for opp, _, n in matches)
        return exp_total - total_score

    lo, hi = bounds
    if deficit(lo) > 0:  # even at lowest rating we overshoot — return floor
        return lo
    if deficit(hi) < 0:  # even at highest we undershoot — return ceiling
        return hi
    # Bisect to tol Elo.
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if deficit(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def baseline_spec_to_anchor_key(spec) -> str | None:
    """Map a parsed player spec to its key in ANCHOR_ELOS, or None if unknown.

    `spec` is a PlayerSpec from gomoku.match.parse_spec. We treat
    `lookahead:depth=N` as `lookaheadN` to match ANCHOR_ELOS' naming.
    """
    if spec.kind == "lookahead":
        depth = spec.kwargs.get("depth")
        if depth is None:
            return None
        return f"lookahead{depth}"
    if spec.kind in ANCHOR_ELOS:
        return spec.kind
    return None
