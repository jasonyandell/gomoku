#!/usr/bin/env python3
"""Report distance-to-100% from the color-split eval (derby-gi7), pooled across recent
eval snapshots to cut the 10-game-per-color noise.

The v9 "100% target" (Jason, 2026-05-27): a model is "done" vs the in-repo evals when it
ALWAYS WINS as BLACK and NEVER LOSES as WHITE across heuristic / lookahead2 / lookahead4.
Anchored elo SATURATES (~1700) because a draw scores 0.5, so it hides the real gap (the
champion DRAWS lookahead4 as black instead of winning). This reads the gap directly.

Read-only: parses sweep_runs/<lane>/checkpoints/eval_results.jsonl. No GPU, no derby touch.
Pooling the last K color-split rows gives ~K*10 effective games/color, which is what makes
"always-wins-black / never-loses-white" distinguishable from a lucky 10-game sample.

Usage:
  python scripts/report_100pct.py                 # all sweep_runs/*/ with color-split evals
  python scripts/report_100pct.py derby-v9-small derby-v9-medium derby-v9-large --pool 6
"""
import argparse, glob, json, os, sys

BASELINES = ["heuristic", "lookahead2", "lookahead4"]


def pooled_color_split(path, pool):
    """Pool the last `pool` color-split rows. Returns {baseline: dict of summed w/l/d per color}
    and the (min,max) model_elo over the pooled rows, plus how many rows were pooled."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if any(k.endswith("_black_l") for k in r):  # a color-split row
                rows.append(r)
    rows = rows[-pool:]
    if not rows:
        return None, None, 0
    agg = {b: {f"{c}_{o}": 0 for c in ("black", "white") for o in ("w", "l", "d")} for b in BASELINES}
    elos = [r["eval/model_elo"] for r in rows if "eval/model_elo" in r]
    for r in rows:
        for b in BASELINES:
            for c in ("black", "white"):
                for o in ("w", "l", "d"):
                    agg[b][f"{c}_{o}"] += r.get(f"eval/vs_{b}_{c}_{o}", 0)
    elo_rng = (min(elos), max(elos)) if elos else (None, None)
    return agg, elo_rng, len(rows)


def score(agg):
    """Per-baseline (black_win_rate, white_loss_rate) and an aggregate distance-to-100%
    (0.0 == perfect: all black wins, zero white losses)."""
    per = {}
    dist = 0.0
    for b in BASELINES:
        a = agg[b]
        bt = a["black_w"] + a["black_l"] + a["black_d"]
        wt = a["white_w"] + a["white_l"] + a["white_d"]
        bw = a["black_w"] / bt if bt else float("nan")
        wl = a["white_l"] / wt if wt else float("nan")
        per[b] = (bw, wl, bt, wt)
        if bt and wt:
            dist += (1.0 - bw) + wl  # 0 when black always wins AND white never loses
    return per, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lanes", nargs="*", help="lane names; default = all with color-split evals")
    ap.add_argument("--pool", type=int, default=5, help="pool the last N color-split eval rows")
    ap.add_argument("--root", default="sweep_runs")
    args = ap.parse_args()

    if args.lanes:
        paths = [os.path.join(args.root, ln, "checkpoints", "eval_results.jsonl") for ln in args.lanes]
    else:
        paths = sorted(glob.glob(os.path.join(args.root, "*", "checkpoints", "eval_results.jsonl")))

    print(f"distance-to-100% (pooled last {args.pool} color-split evals; 0.0 = win-all-black/lose-none-white)")
    print(f"{'lane':28s} {'elo(pooled)':>13s} {'rows':>4s}  " + "  ".join(f"{b[:9]:>17s}" for b in BASELINES) + "   DIST")
    print(f"{'':28s} {'':>13s} {'':>4s}  " + "  ".join(f"{'Bwin / Wloss':>17s}" for _ in BASELINES))
    results = []
    for p in paths:
        if not os.path.exists(p):
            continue
        lane = p.split(os.sep)[-3]
        agg, elo_rng, n = pooled_color_split(p, args.pool)
        if not agg:
            continue
        per, dist = score(agg)
        elo_s = f"{elo_rng[0]:.0f}-{elo_rng[1]:.0f}" if elo_rng[0] is not None else "?"
        cells = []
        for b in BASELINES:
            bw, wl, bt, wt = per[b]
            cells.append(f"{bw*100:4.0f}% /{wl*100:4.0f}%" if bt and wt else "   - /   -")
        print(f"{lane:28s} {elo_s:>13s} {n:>4d}  " + "  ".join(f"{c:>17s}" for c in cells) + f"   {dist:5.2f}")
        results.append((lane, dist))
    if results:
        best = min(results, key=lambda x: x[1])
        print(f"\nclosest to 100%: {best[0]} (distance {best[1]:.2f}). 0.0 = perfect; lower is better.")


if __name__ == "__main__":
    main()
