"""Reconcile the deepening contradiction (#95): #94 measured base-kernel deepening
1.60× vs base@4000 on N=83814; the throughput sweep got 0.80× at N=20000. Hypothesis:
deepening wins only when the hard-survivor batch is dense enough to saturate the GPU,
so the ratio should IMPROVE with N. Reuses #94's EXACT pool (bench_pool_100k.npz) so
it's apples-to-apples, and cross-checks solve_vct_streaming against an independent
manual deepen to rule out a regression.

  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python -u -m scripts.vct_metal.n_sweep --out ~/data/idx2_solve/sweep
"""
import argparse
import json
import os
import time

import numpy as np

from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb, solve_vct_streaming

POOL = os.path.expanduser("~/data/idx2_solve/bench_pool_100k.npz")   # #94's exact pool


def manual_deepen(bd, budgets):
    """Independent base-kernel deepening (mirrors #94's stream_base) — a bug-check
    against the shipped solve_vct_streaming."""
    n = bd.shape[0]
    hit = np.ones(n, bool); active = np.arange(n)
    for b in budgets:
        if active.size == 0:
            break
        w, h = solve_vct_mega_bb(bd[active], max_nodes=int(b))
        hit[active[~h]] = False        # ~h = resolved this round -> latch
        active = active[h]             # keep the CAPPED ones for the next deeper round
    return ~hit


def timeit(fn):
    fn()                                  # warm
    t = time.time(); r = fn(); return time.time() - t, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/data/idx2_solve/sweep"))
    args = ap.parse_args()
    if not os.path.exists(POOL):
        print(f"!! {POOL} absent (built by the earlier #94 bench); aborting"); return
    boards = np.load(POOL, allow_pickle=True)["boards"]
    path = os.path.join(args.out, "n_sweep.jsonl")
    LADDER = (250, 1000, 4000)
    CEIL = LADDER[-1]
    print(f"pool={boards.shape[0]} boards (#94's set); ladder={LADDER}", flush=True)

    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["N"])
                except Exception:
                    pass

    for N in (10000, 20000, 40000, 83814):
        N = min(N, boards.shape[0])
        if N in done:
            continue
        bd = boards[:N]
        bt, (_, hb) = timeit(lambda: solve_vct_mega_bb(bd, max_nodes=CEIL))
        dt, (_, hd) = timeit(lambda: solve_vct_streaming(bd, budgets=LADDER, work_steal=False))
        # cross-check the shipped streamer vs an independent manual deepen at this N
        mt, hm = timeit(lambda: manual_deepen(bd, LADDER))
        row = {"N": int(N), "base_wall": round(bt, 2), "deepen_wall": round(dt, 2),
               "manual_wall": round(mt, 2),
               "deepen_speedup": round(bt / dt, 3), "manual_speedup": round(bt / mt, 3),
               "base_resolved": int((~hb).sum()), "deepen_resolved": int((~hd).sum()),
               "agree_streamer_vs_manual": bool(np.array_equal(~hd, hm)),
               "ts": time.time()}
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n"); f.flush()
        print(f"  N={N:6}: base@{CEIL} {bt:6.1f}s  deepen {dt:6.1f}s = {bt/dt:.2f}x  "
              f"(manual {mt:.1f}s={bt/mt:.2f}x, agree={row['agree_streamer_vs_manual']})  "
              f"resolved {int((~hb).sum())}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
