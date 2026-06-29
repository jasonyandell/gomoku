"""MAXD 32->64 study (issue #95 follow-on): are the capped boards FRAME-DEPTH-bound
(need >32 forcing frames) or NODE-bound (huge <=32-frame trees / just hard)? The
profile found the capped pool 93.7% unresolved even at 20000 nodes — if raising the
frame ceiling resolves materially more, the cap was a DEPTH ceiling, not a node one.

Run TWICE (the kernel bakes MAXD at import, so one process = one MAXD):
  GOMOKU_BOARD_SIZE=15 GOMOKU_VCT_MAXD=32 PYTHONPATH=. uv run python -u -m scripts.vct_metal.maxd_study --out ~/data/idx2_solve/sweep
  GOMOKU_BOARD_SIZE=15 GOMOKU_VCT_MAXD=64 PYTHONPATH=. uv run python -u -m scripts.vct_metal.maxd_study --out ~/data/idx2_solve/sweep
Then diff the two MAXD rows (analyze_maxd at the bottom prints it). Append-only,
resumable (skips (maxd,purpose,budget) rows already logged).

Two purposes per run:
  * yield   — capped sample over a budget ladder: resolved/wins/wall at this MAXD.
  * monotone — random sample at a fixed budget: the win-set at MAXD must be a
    SUPERSET of MAXD=32 (clean wins never disappear); stored for the cross-check.
"""
import argparse
import json
import os
import time

import numpy as np

from scripts.vct_metal.mega_vct_bb import MAXD, solve_vct_mega_bb
from scripts.vct_metal.positions import load_position_stack


def load_pool(out, name, n):
    cache = os.path.join(out, "pools", f"pool_{name}_30000.npz")
    if os.path.exists(cache):
        return np.load(cache, allow_pickle=True)["boards"][:n]
    if name == "random":
        return load_position_stack(n, seed=0, min_ply=6, max_ply=40)
    raise FileNotFoundError(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/data/idx2_solve/sweep"))
    ap.add_argument("--n", type=int, default=15000)
    args = ap.parse_args()
    path = os.path.join(args.out, "maxd_study.jsonl")
    win_path = os.path.join(args.out, f"maxd_winset_{MAXD}.npy")

    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["maxd"], r["purpose"], r.get("budget")))
                except Exception:
                    pass

    def log(row):
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n"); f.flush()
        print("  " + json.dumps(row), flush=True)

    print(f"MAXD={MAXD}  n={args.n}", flush=True)

    # -- yield: capped sample over a budget ladder --
    capped = load_pool(args.out, "capped", args.n)
    for budget in (250, 1000, 4000, 20000):
        if (MAXD, "yield", budget) in done:
            continue
        t = time.time()
        w, h = solve_vct_mega_bb(capped, max_nodes=budget)
        log({"maxd": MAXD, "purpose": "yield", "budget": budget, "N": int(capped.shape[0]),
             "resolved": int((~h).sum()), "wins": int(w.sum()), "capped": int(h.sum()),
             "wall_s": round(time.time() - t, 2),
             "evals_per_min": round(capped.shape[0] / (time.time() - t) * 60, 1),
             "ts": time.time()})

    # -- monotone: random sample, fixed budget; persist the win-set for the cross-check --
    if (MAXD, "monotone", 4000) not in done:
        rnd = load_pool(args.out, "random", args.n)
        t = time.time()
        w, h = solve_vct_mega_bb(rnd, max_nodes=4000)
        np.save(win_path, w)
        log({"maxd": MAXD, "purpose": "monotone", "budget": 4000, "N": int(rnd.shape[0]),
             "wins": int(w.sum()), "capped": int(h.sum()), "wall_s": round(time.time() - t, 2),
             "winset_file": win_path, "ts": time.time()})

    # -- cross-check if both MAXD win-sets exist: 64 must be a superset of 32 --
    p32, p64 = (os.path.join(args.out, f"maxd_winset_{m}.npy") for m in (32, 64))
    if os.path.exists(p32) and os.path.exists(p64):
        w32, w64 = np.load(p32), np.load(p64)
        if w32.shape == w64.shape:
            lost = int((w32 & ~w64).sum())          # win@32 but not win@64 -> MUST be 0
            gained = int((~w32 & w64).sum())         # new wins only 64 finds (deeper proofs)
            print(f"\nMONOTONE CHECK (random@4000): lost={lost} (must be 0), "
                  f"gained_by_64={gained}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
