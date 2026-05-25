#!/usr/bin/env python3
"""Round-robin head-to-head among a set of checkpoints, reusing the delta-e
harness's model-vs-model match (paired random openings, parallel workers).

Each unordered pair plays --games games; we record pairwise Δelo (A relative to
B), then fit a simple least-squares rating (mean-centered) so the whole field is
ranked on ONE scale with no anchor ceiling. Two near-equal models meet at ~50%
(the sensitive region of the logistic), so this separates a saturated cluster
that anchored elo can't.

Usage:
  python scripts/round_robin.py \
    --model vcf=sweep_runs/derby_v4/_peaks/vcf/peak.pt \
    --model control=sweep_runs/derby_v4/_peaks/control/peak.pt \
    --games 100 --sims 100 --workers 8
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.delta_e_harness import head_to_head_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], metavar="name=path",
                    help="A contender as name=checkpoint_path (repeatable).")
    ap.add_argument("--games", type=int, default=100, help="games per pair")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="sweep_runs/derby_v4/round_robin.json")
    args = ap.parse_args()

    models = {}
    for m in args.model:
        name, _, path = m.partition("=")
        if not path:
            print(f"bad --model {m!r} (need name=path)"); return 2
        models[name] = path
    names = list(models)
    if len(names) < 2:
        print("need >= 2 --model entries"); return 2

    print(f"Round-robin: {names}  ({args.games} games/pair, sims={args.sims}, "
          f"workers={args.workers})\n")

    # pairwise: delta[a][b] = a's Δelo vs b (and delta[b][a] = -that)
    delta = {a: {} for a in names}
    pairs = []
    for a, b in itertools.combinations(names, 2):
        res = head_to_head_eval(
            models[a], models[b],
            recipe_label=f"{a}-vs-{b}", window_epochs=0, wall_secs=None,
            n_games=args.games, sims=args.sims, c_puct=args.c_puct,
            seed=args.seed, device=args.device, n_workers=args.workers,
            opening_plies=args.opening_plies,
        )
        delta[a][b] = res.delta_elo
        delta[b][a] = -res.delta_elo
        pairs.append({"a": a, "b": b, "a_wins": res.wins, "draws": res.draws,
                      "b_wins": res.losses, "a_winrate": round(res.win_rate, 4),
                      "delta_elo_a_vs_b": round(res.delta_elo, 1),
                      "ci_half": round(res.delta_ci_half, 1)})

    # rating = mean of a-vs-everyone Δelo (mean-centered round-robin estimate)
    rating = {a: sum(delta[a].values()) / (len(names) - 1) for a in names}
    ranking = sorted(names, key=lambda n: rating[n], reverse=True)

    print("\n=== PAIRWISE Δelo (row vs column) ===")
    hdr = "".join(f"{n[:9]:>11}" for n in names)
    print(f"{'':<11}{hdr}")
    for a in names:
        cells = "".join((f"{delta[a][b]:>+11.0f}" if b != a else f"{'·':>11}") for b in names)
        print(f"{a:<11}{cells}")

    print("\n=== RANKING (round-robin rating, mean-centered) ===")
    for i, n in enumerate(ranking, 1):
        print(f"  {i}. {n:<11} {rating[n]:>+7.0f} elo")

    out = {"models": models, "games_per_pair": args.games, "sims": args.sims,
           "pairs": pairs, "rating": {n: round(rating[n], 1) for n in names},
           "ranking": ranking}
    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
