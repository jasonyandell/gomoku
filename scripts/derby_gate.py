#!/usr/bin/env python3
"""The derby gate: a CI-aware VERDICT over a field of training-lane peaks.

`scripts/round_robin.py` already plays every unordered pair head-to-head (paired
random openings, parallel workers; via `delta_e_harness.head_to_head_eval`),
records a per-pair Δelo + Wilson-derived CI half-width, and emits a mean-centered
ranking. But it ALWAYS emits a ranking — even when every CI swamps every Δelo.
That's how the Derby v4 four-way "tie" got reported as a crowned winner: the #1
margin was buried inside the combined sampling noise.

This module wraps that machinery in `verdict()`, which GATES on the CI:

  * Crown the leader ONLY if its margin over #2 exceeds the combined CI
    half-width hypot(ci_leader, ci_runnerup).
  * If the top pair OVERLAPS (margin inside the combined CI), re-run just the
    overlapping pair(s) at a larger game count and recompute.
  * If still overlapping after escalation: NO verdict — crowned=None,
    escalate=True, "hold the incumbent".

The eval step (the actual H2H games) is isolated in ONE function,
`run_round_robin`, so tests can monkeypatch it and exercise the GATE LOGIC
without real checkpoints or minutes-slow MPS games. The pure gate math
(`mean_centered_ratings`, `_overlaps`, `decide`) is torch-free and unit-tested.

USAGE
  python scripts/derby_gate.py \\
      --peaks vcf=sweep_runs/derby_v8/_peaks/vcf/peak.pt \\
      --peaks control=sweep_runs/derby_v8/_peaks/control/peak.pt \\
      --games-per-pair 200 --escalate-to 400 --sims 100 --workers 8

  (prints the verdict JSON to stdout)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# The eval step — the ONE mockable function. Everything below it is pure.
#
# Returns the raw per-pair results in the same shape round_robin.py records:
#   {"a": name, "b": name, "delta_elo_a_vs_b": float, "ci_half": float,
#    "a_wins": int, "draws": int, "b_wins": int, "n_games": int}
# `delta_elo_a_vs_b` is A's Δelo relative to B (so B-vs-A is its negation);
# `ci_half` is the Wilson-derived half-width of THAT pair's Δelo CI.
# ---------------------------------------------------------------------------


def run_round_robin(
    peaks: dict[str, str | Path],
    *,
    games_per_pair: int,
    sims: int,
    n_workers: int,
    device: str,
    c_puct: float = 1.5,
    opening_plies: int = 4,
    seed: int = 0,
    pairs: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Play the pairwise round-robin and return per-pair Δelo + CI results.

    Reuses `delta_e_harness.head_to_head_eval` (NO re-implementation of game
    play). When `pairs` is given, only those unordered pairs are played (used by
    escalation to re-run just the overlapping pair at a higher game count);
    otherwise every unordered pair in `peaks` is played.

    This is the SOLE function the tests monkeypatch — keep it a thin shell over
    the real H2H eval so the gate logic above it stays test-covered.
    """
    from scripts.delta_e_harness import head_to_head_eval

    names = list(peaks)
    if pairs is None:
        pairs = list(itertools.combinations(names, 2))

    out: list[dict] = []
    for a, b in pairs:
        res = head_to_head_eval(
            str(peaks[a]), str(peaks[b]),
            recipe_label=f"{a}-vs-{b}", window_epochs=0, wall_secs=None,
            n_games=games_per_pair, sims=sims, c_puct=c_puct,
            seed=seed, device=device, n_workers=n_workers,
            opening_plies=opening_plies,
        )
        out.append({
            "a": a, "b": b,
            "delta_elo_a_vs_b": float(res.delta_elo),
            "ci_half": float(res.delta_ci_half),
            "a_wins": res.wins, "draws": res.draws, "b_wins": res.losses,
            "n_games": res.n_games,
        })
    return out


# ---------------------------------------------------------------------------
# Pure gate math (torch-free, unit-tested).
# ---------------------------------------------------------------------------


def mean_centered_ratings(
    names: list[str], pairs: list[dict]
) -> list[tuple[str, float, float]]:
    """Mean-centered round-robin ratings + a per-lane CI half-width.

    rating[a] = mean over opponents b of (a's Δelo vs b), exactly as
    round_robin.py fits it (already mean-centered: the sum of all signed
    pairwise deltas is zero, so the per-lane means already sum to ~0).

    Each pairwise Δelo carries an independent Wilson CI half-width. The CI on a
    lane's mean rating is those half-widths added in quadrature and divided by
    the opponent count (variance of a mean of independent samples):
        ci_half(rating[a]) = sqrt(sum_b ci_half[a][b]^2) / (n_opponents)

    Returns [(name, rating, ci_half), ...] sorted best-first.
    """
    # delta[a][b] = a's Δelo vs b; ci[a][b] = that pair's CI half-width (symmetric).
    delta: dict[str, dict[str, float]] = {a: {} for a in names}
    ci: dict[str, dict[str, float]] = {a: {} for a in names}
    for p in pairs:
        a, b = p["a"], p["b"]
        d = p["delta_elo_a_vs_b"]
        h = p["ci_half"]
        delta[a][b] = d
        delta[b][a] = -d
        ci[a][b] = h
        ci[b][a] = h

    out: list[tuple[str, float, float]] = []
    for a in names:
        opps = [b for b in names if b != a]
        n = len(opps)
        if n == 0:
            out.append((a, 0.0, 0.0))
            continue
        rating = sum(delta[a][b] for b in opps) / n
        ci_half = math.sqrt(sum(ci[a][b] ** 2 for b in opps)) / n
        out.append((a, rating, ci_half))

    out.sort(key=lambda t: t[1], reverse=True)
    return out


def combined_ci(ci_leader: float, ci_runnerup: float) -> float:
    """Combined CI half-width on the margin between two lanes (quadrature)."""
    return math.hypot(ci_leader, ci_runnerup)


def _overlaps(leader: tuple[str, float, float], runnerup: tuple[str, float, float]) -> bool:
    """True if the leader's margin over the runner-up is inside the combined CI
    (i.e. NOT a trustworthy separation) — the same `|Δ| <= ci` rule as
    delta_e_harness.inside_noise, applied to the top-of-table gap."""
    margin = leader[1] - runnerup[1]
    return margin <= combined_ci(leader[2], runnerup[2])


def decide(ranking: list[tuple[str, float, float]]) -> tuple[str | None, bool, str]:
    """Apply the crown rule to a (already-ranked) field.

    Returns (crowned, escalate, reason). `crowned` is the leader's name iff its
    margin over #2 clears the combined CI; otherwise crowned=None. `escalate` is
    advisory here (the orchestration in `verdict()` flips it on after a failed
    escalation); `decide` reports whether the CURRENT table separates.
    """
    if len(ranking) < 2:
        only = ranking[0][0] if ranking else None
        return only, False, ("single lane — nothing to compare against"
                             if only else "no lanes")

    leader, runnerup = ranking[0], ranking[1]
    margin = leader[1] - runnerup[1]
    cci = combined_ci(leader[2], runnerup[2])
    if margin > cci:
        return (leader[0], False,
                f"clear: {leader[0]} leads {runnerup[0]} by {margin:.1f} elo, "
                f"outside combined CI {cci:.1f}")
    return (None, True,
            f"overlap: {leader[0]} leads {runnerup[0]} by only {margin:.1f} elo, "
            f"inside combined CI {cci:.1f}")


def _top_overlapping_pairs(
    ranking: list[tuple[str, float, float]]
) -> list[tuple[str, str]]:
    """Unordered pairs at the top of the table whose separation is inside the
    combined CI — the pairs worth re-running at a higher game count. Walks down
    from the leader and collects every consecutive pair that overlaps so a
    tightly-bunched cluster all gets escalated, not just rank-1-vs-2."""
    pairs: list[tuple[str, str]] = []
    for i in range(len(ranking) - 1):
        hi, lo = ranking[i], ranking[i + 1]
        if _overlaps(hi, lo):
            pairs.append((hi[0], lo[0]))
        else:
            # Once a gap is clean, lanes below it can't dethrone the leader.
            break
    return pairs


def _merge_escalated(pairs: list[dict], escalated: list[dict]) -> list[dict]:
    """Replace re-run pairs (matched by unordered {a,b}) with their escalated
    results, keeping all others unchanged."""
    rerun_keys = {frozenset((p["a"], p["b"])) for p in escalated}
    kept = [p for p in pairs if frozenset((p["a"], p["b"])) not in rerun_keys]
    return kept + escalated


# ---------------------------------------------------------------------------
# The verdict orchestration (impure: drives run_round_robin, which is mockable).
# ---------------------------------------------------------------------------


def verdict(
    peaks: dict[str, str | Path],
    games_per_pair: int = 200,
    escalate_to: int = 400,
    sims: int = 100,
    n_workers: int = 8,
    device: str = "cpu",
    *,
    c_puct: float = 1.5,
    opening_plies: int = 4,
    seed: int = 0,
) -> dict:
    """CI-gated verdict over a field of lane peaks.

    1. Round-robin every pair at `games_per_pair` (off-GPU; device default cpu).
    2. Rank by mean pairwise Δelo (mean-centered).
    3. Crown #1 ONLY if its margin over #2 exceeds hypot(ci_1, ci_2).
    4. If the top pair(s) overlap, re-run just those at `escalate_to` games and
       recompute.
    5. Still overlapping after escalation -> crowned=None, escalate=True
       ("hold incumbent"). Clear -> crowned=<name>, escalate=False.

    Returns {"ranking": [(name, mean_delo, mean_ci_half), ...],  # best first
             "crowned": name | None,
             "escalate": bool,
             "reason": str,
             "pairs": [...]}  # raw per-pair results (post-escalation) for logging
    """
    names = list(peaks)
    if len(names) < 2:
        raise ValueError("verdict needs >= 2 lanes")

    pairs = run_round_robin(
        peaks, games_per_pair=games_per_pair, sims=sims, n_workers=n_workers,
        device=device, c_puct=c_puct, opening_plies=opening_plies, seed=seed,
    )
    ranking = mean_centered_ratings(names, pairs)
    crowned, _escalate, reason = decide(ranking)

    if crowned is not None:
        return {"ranking": ranking, "crowned": crowned, "escalate": False,
                "reason": reason, "pairs": pairs}

    # Overlap at the top — escalate just the overlapping pair(s) at escalate_to.
    rerun = _top_overlapping_pairs(ranking)
    if rerun and escalate_to > games_per_pair:
        escalated = run_round_robin(
            peaks, games_per_pair=escalate_to, sims=sims, n_workers=n_workers,
            device=device, c_puct=c_puct, opening_plies=opening_plies, seed=seed,
            pairs=rerun,
        )
        pairs = _merge_escalated(pairs, escalated)
        ranking = mean_centered_ratings(names, pairs)
        crowned, _escalate, reason = decide(ranking)
        if crowned is not None:
            return {"ranking": ranking, "crowned": crowned, "escalate": False,
                    "reason": f"{reason} (after escalation to {escalate_to} games)",
                    "pairs": pairs}
        # Still overlapping after escalation — no verdict.
        leader, runnerup = ranking[0], ranking[1]
        margin = leader[1] - runnerup[1]
        cci = combined_ci(leader[2], runnerup[2])
        return {
            "ranking": ranking, "crowned": None, "escalate": True,
            "reason": (f"no verdict — leader {leader[0]} margin {margin:.1f} inside "
                       f"combined CI {cci:.1f} after escalation to {escalate_to} "
                       f"games; hold incumbent"),
            "pairs": pairs,
        }

    # Overlap, but escalation is disabled (escalate_to <= games_per_pair) or no
    # pair to re-run — surface as no-verdict.
    leader, runnerup = ranking[0], ranking[1]
    margin = leader[1] - runnerup[1]
    cci = combined_ci(leader[2], runnerup[2])
    return {
        "ranking": ranking, "crowned": None, "escalate": True,
        "reason": (f"no verdict — leader {leader[0]} margin {margin:.1f} inside "
                   f"combined CI {cci:.1f} (no escalation); hold incumbent"),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _verdict_to_json(v: dict) -> dict:
    """Make the verdict JSON-serializable (ranking tuples -> lists)."""
    return {
        "ranking": [[n, round(r, 1), round(c, 1)] for (n, r, c) in v["ranking"]],
        "crowned": v["crowned"],
        "escalate": v["escalate"],
        "reason": v["reason"],
        "pairs": v["pairs"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peaks", action="append", default=[], metavar="name=path",
                    help="A lane as name=peak.pt (repeatable; need >= 2).")
    ap.add_argument("--games-per-pair", type=int, default=200)
    ap.add_argument("--escalate-to", type=int, default=400,
                    help="Game count for re-running overlapping pairs.")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                    help="Also write the verdict JSON to this path.")
    args = ap.parse_args(argv)

    peaks: dict[str, str] = {}
    for m in args.peaks:
        name, _, path = m.partition("=")
        if not path:
            print(f"bad --peaks {m!r} (need name=path)", file=sys.stderr)
            return 2
        peaks[name] = path
    if len(peaks) < 2:
        print("need >= 2 --peaks entries", file=sys.stderr)
        return 2

    v = verdict(
        peaks, games_per_pair=args.games_per_pair, escalate_to=args.escalate_to,
        sims=args.sims, n_workers=args.workers, device=args.device,
        c_puct=args.c_puct, opening_plies=args.opening_plies, seed=args.seed,
    )
    payload = _verdict_to_json(v)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text)
        print(f"\nwrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
