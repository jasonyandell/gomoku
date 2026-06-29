"""Characterize mega-VCT solver runtime properties: base (one dispatch) vs base
chunked (the old wave pattern) vs work_steal, plus iterative-deepening (Option B)
vs a single deep dispatch. Solver-only — NO Rapfi.

Findings (84k idx-2 run-a boards, 24% capped@250, M-series GPU, #93/#94):
  * work_steal NEVER beats base on a single in-memory pool: 0.93-0.97x at best
    (resident=16384); the GPU already backfills one dispatch, the cursor is
    overhead. Smaller resident underutilizes (0.34x @ 4096, budget 250).
  * base chunked@16384 (frontier's old wave pattern) is 0.66-0.69x — a dispatch
    tail per chunk; one big dispatch is the lever.
  * Iterative deepening on the BASE kernel is 1.60x vs a single budget-4000
    dispatch (identical verdicts) — the deep budget only touches the survivor
    tail. Deepening on work_steal is 0.87x (work_steal's round-0 handicap eats
    the win). Coarse ladder (250,1000,4000)=1.60x beats fine (+500,+2000)=1.10x.

Usage:
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python -u -m scripts.vct_metal.bench_throughput
  # options: --n 80000  --pool {idx2,random}  --deep 4000
"""
import argparse
import json
import os
import random
import time

import numpy as np

from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb, solve_vct_streaming
from scripts.vct_metal.positions import load_position_stack

IDX2_NODES = os.path.expanduser("~/data/idx2_solve/run-a/nodes.jsonl")
IDX2_TOTAL = 11_514_718


def idx2_pool(n: int):
    """Reconstruct n real idx-2 frontier boards from stored move-sequences (pure
    replay, NO Rapfi); cached to npz. Returns (boards, capped_mask) or None if the
    run-a dump is absent on this machine."""
    if not os.path.exists(IDX2_NODES):
        return None
    from scripts.idx2_vct.frontier import state_from_moves
    cache = os.path.expanduser(f"~/data/idx2_solve/bench_pool_{n//1000}k.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["boards"], (d["verdict"] == "capped")
    rng = random.Random(12345)
    keep = set(rng.sample(range(IDX2_TOTAL), n))
    moves, verd = [], []
    with open(IDX2_NODES) as f:
        for i, line in enumerate(f):
            if i in keep:
                r = json.loads(line)
                if "moves" in r:                  # skip `done` resume-markers
                    moves.append(r["moves"]); verd.append(r["verdict"])
    boards = np.zeros((len(moves), 2, 15, 15), dtype=bool)
    for j, mv in enumerate(moves):
        boards[j] = np.asarray(state_from_moves(mv).board, dtype=bool)
    verd = np.asarray(verd)
    np.savez(cache, boards=boards, verdict=verd)
    return boards, (verd == "capped")


def timeit(fn, reps=2):
    r = fn()                                       # warm-up (compile + buffers)
    ts = []
    for _ in range(reps):
        t = time.time(); r = fn(); ts.append(time.time() - t)
    return float(np.median(ts)), r


def base_chunked(bd, budget, chunk=16384):
    for i in range(0, bd.shape[0], chunk):
        solve_vct_mega_bb(bd[i:i + chunk], max_nodes=budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80000)
    ap.add_argument("--pool", choices=["idx2", "random"], default="idx2")
    ap.add_argument("--deep", type=int, default=4000)
    args = ap.parse_args()

    pool = idx2_pool(args.n) if args.pool == "idx2" else None
    if pool is None:
        boards = load_position_stack(args.n, seed=0, min_ply=6, max_ply=40)
        capped = solve_vct_mega_bb(boards, max_nodes=250)[1]
        src = "random game positions"
    else:
        boards, capped = pool
        src = "idx-2 run-a frontier boards (no Rapfi)"
    B = boards.shape[0]
    print(f"pool: {B} {src}  capped@250={int(capped.sum())} ({100*capped.mean():.1f}%)")

    print("\n=== throughput: base vs chunked vs work_steal ===")
    for budget in (250, 500, 1000):
        base, _ = timeit(lambda b=budget: solve_vct_mega_bb(boards, max_nodes=b))
        print(f"\n--- budget={budget} ---")
        print(f"  base (1 dispatch)      {base:6.2f}s  {B/base:8.0f} boards/s  1.00x")
        ch, _ = timeit(lambda b=budget: base_chunked(boards, b))
        print(f"  base chunked@16384     {ch:6.2f}s  {B/ch:8.0f} boards/s  {base/ch:.2f}x")
        for R in (4096, 16384):
            w, _ = timeit(lambda b=budget, R=R: solve_vct_mega_bb(
                boards, max_nodes=b, work_steal=True, resident=R))
            print(f"  work_steal r={R:<6}     {w:6.2f}s  {B/w:8.0f} boards/s  {base/w:.2f}x")

    print(f"\n=== deepening (Option B) vs single dispatch @ {args.deep} ===")
    bd, (_, hb) = timeit(lambda: solve_vct_mega_bb(boards, max_nodes=args.deep), reps=1)
    print(f"  base @ {args.deep}                       {bd:7.2f}s  resolved={int((~hb).sum())}/{B}")
    for bg in [(250, 1000, args.deep), (250, 500, 1000, 2000, args.deep)]:
        s, (_, hs) = timeit(lambda g=bg: solve_vct_streaming(boards, budgets=g), reps=1)
        print(f"  deepen(base) {str(bg):<26} {s:7.2f}s  resolved={int((~hs).sum())}/{B}  {bd/s:.2f}x")
    s, (_, hs) = timeit(lambda: solve_vct_streaming(
        boards, budgets=(250, 1000, args.deep), work_steal=True), reps=1)
    print(f"  deepen(work_steal) (250,1000,{args.deep})  {s:7.2f}s  resolved={int((~hs).sum())}/{B}  {bd/s:.2f}x")


if __name__ == "__main__":
    main()
