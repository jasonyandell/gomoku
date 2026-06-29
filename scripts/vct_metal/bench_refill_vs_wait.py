"""bench_refill_vs_wait.py — issue #96.

Spotlight the long tail that exists because the board SUPPLY ran dry (forced waves),
NOT because of one slow proof. Three contestants on one mixed pool, fixed cap:

  wait     — process the pool in W separate dispatches, drain each fully before the
             next (pays W tails; the naive forced-wave approach).
  refill   — one work_steal cursor over the whole pool; `resident` lanes refill across
             what would have been wave boundaries (pays 1 tail).
  oneshot  — single dispatch over everything (reference upper bound = wait with W=1).

This is the matchup the single-pool work_steal test (#93/#94) never ran: there base and
work_steal drained the SAME pool and ran dry together, so refill had nothing extra to
pull. Here `wait` is FORCED to drain per wave; the question is whether `refill` recovers
the gap — i.e. whether keeping the queue topped up across waves beats relaunch-per-wave.

Caveat for the writeup: refill only helps to the extent the next wave's boards are
already in hand. A gated frontier (wave K+1 depends on wave K's answers) can't be
refilled — there the run-dry tail is genuinely unavoidable.

Append-only JSONL; resolution parity asserted across all three (same boards, same budget,
deterministic solver => identical resolved/win counts).
"""
import argparse
import json
import os
import time

import numpy as np

from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb

POOL = os.path.expanduser("~/data/idx2_solve/bench_pool_100k.npz")
OUT = os.path.expanduser("~/data/idx2_solve/sweep/refill_vs_wait.jsonl")


def load_pool(n):
    z = np.load(POOL)
    bd = z["boards"][:n]
    return np.ascontiguousarray(bd)


def score(win, hit):
    """resolved = did NOT hit the cap; wins = proven VCT. (hit==1 => unresolved/capped)"""
    win = np.asarray(win).astype(bool)
    hit = np.asarray(hit).astype(bool)
    return int((~hit).sum()), int(win.sum())


def run_wait(pool, budget, wave):
    """W separate dispatches, drained one at a time; wall = sum of per-wave walls."""
    N = len(pool)
    wins = np.zeros(N, bool)
    hits = np.zeros(N, bool)
    t0 = time.time()
    for i in range(0, N, wave):
        w, h = solve_vct_mega_bb(pool[i:i + wave], max_nodes=budget)
        wins[i:i + wave] = w
        hits[i:i + wave] = h
    wall = time.time() - t0
    return wall, score(wins, hits)


def run_refill(pool, budget, resident):
    t0 = time.time()
    w, h = solve_vct_mega_bb(pool, max_nodes=budget, work_steal=True, resident=resident)
    wall = time.time() - t0
    return wall, score(w, h)


def run_oneshot(pool, budget):
    t0 = time.time()
    w, h = solve_vct_mega_bb(pool, max_nodes=budget)
    wall = time.time() - t0
    return wall, score(w, h)


def emit(row):
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--waves", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--resident", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.budget, args.waves, args.resident = 2048, 500, [1, 4], [4096]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pool = load_pool(args.n)
    N = len(pool)
    print(f"pool N={N}  budget={args.budget}  waves={args.waves}  resident={args.resident}",
          flush=True)

    ref = None  # resolution parity reference (resolved, wins)

    def check(label, res):
        nonlocal ref
        if ref is None:
            ref = res
        elif res != ref:
            print(f"  !! PARITY FAIL {label}: {res} != {ref}", flush=True)

    # oneshot (reference upper bound)
    wall, res = run_oneshot(pool, args.budget)
    check("oneshot", res)
    bps = N / wall
    print(f"  oneshot           {wall:7.2f}s  {bps:8.0f} boards/s  resolved={res[0]} wins={res[1]}",
          flush=True)
    emit({"kind": "oneshot", "n": N, "budget": args.budget, "wall_s": round(wall, 3),
          "boards_per_s": round(bps, 1), "resolved": res[0], "wins": res[1]})
    base_wall = wall

    # wait — one number per wave count
    for W in args.waves:
        wave = (N + W - 1) // W
        wall, res = run_wait(pool, args.budget, wave)
        check(f"wait W={W}", res)
        bps = N / wall
        spd = base_wall / wall
        print(f"  wait W={W:<3d} (sz {wave:5d}) {wall:7.2f}s  {bps:8.0f} boards/s  "
              f"{spd:.2f}x oneshot  resolved={res[0]}", flush=True)
        emit({"kind": "wait", "waves": W, "wave_size": wave, "n": N, "budget": args.budget,
              "wall_s": round(wall, 3), "boards_per_s": round(bps, 1),
              "x_oneshot": round(spd, 3), "resolved": res[0], "wins": res[1]})

    # refill — one number per resident
    for R in args.resident:
        wall, res = run_refill(pool, args.budget, R)
        check(f"refill R={R}", res)
        bps = N / wall
        spd = base_wall / wall
        print(f"  refill R={R:<6d}    {wall:7.2f}s  {bps:8.0f} boards/s  "
              f"{spd:.2f}x oneshot  resolved={res[0]}", flush=True)
        emit({"kind": "refill", "resident": R, "n": N, "budget": args.budget,
              "wall_s": round(wall, 3), "boards_per_s": round(bps, 1),
              "x_oneshot": round(spd, 3), "resolved": res[0], "wins": res[1]})

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
