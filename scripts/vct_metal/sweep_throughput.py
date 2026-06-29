"""Characterize mega-VCT throughput across strategy × budget × board-population
(issue #95). North star: board-evals/min over the WHOLE mixed population, which
decomposes into differently-shaped subproblems (easies / long-tails / deeps).

Design (Jason's reducer-over-a-log aesthetic): append-only JSONL, one row per
config, RESUMABLE (skip configs already in the log), each config wrapped so a
single failure can't kill the overnight run, flushed after every row so a kill
loses nothing. Pools cached to npz. NO Rapfi anywhere.

  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python -u -m scripts.vct_metal.sweep_throughput \
      --out ~/data/idx2_solve/sweep --n 20000 --profile-n 50000

Sub-commands of one run: (1) build/cache pools, (2) per-pool resolution PROFILE
(each board's min-resolving budget -> hardness histogram -> oracle bound), (3) the
strategy×budget throughput sweep. Re-run to resume / extend with more configs.
"""
import argparse
import hashlib
import json
import os
import random
import time

import numpy as np

from scripts.vct_metal.mega_vct_bb import (
    MAXD, solve_vct_mega_bb, solve_vct_streaming)
from scripts.vct_metal.positions import load_position_stack

IDX2_NODES = os.path.expanduser("~/data/idx2_solve/run-a/nodes.jsonl")
IDX2_TOTAL = 11_514_718


# --------------------------------------------------------------------------- #
# Pools — all Rapfi-free. Cached to npz so re-runs are instant.
# --------------------------------------------------------------------------- #
def _reconstruct(moves_list, pool_dir, name):
    from scripts.idx2_vct.frontier import state_from_moves
    boards = np.zeros((len(moves_list), 2, 15, 15), dtype=bool)
    for j, mv in enumerate(moves_list):
        boards[j] = np.asarray(state_from_moves(mv).board, dtype=bool)
    return boards


def build_pool(name, n, pool_dir):
    cache = os.path.join(pool_dir, f"pool_{name}_{n}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["boards"]
    print(f"  building pool '{name}' (n={n}) ...", flush=True)
    t = time.time()
    if name == "random":
        boards = load_position_stack(n, seed=0, min_ply=6, max_ply=40)
    elif name == "deep":
        boards = load_position_stack(n, seed=1, min_ply=20, max_ply=60)
    elif name in ("quiet", "capped"):
        if not os.path.exists(IDX2_NODES):
            print(f"  !! {IDX2_NODES} absent; skipping pool '{name}'", flush=True)
            return None
        want_capped = (name == "capped")
        rng = random.Random(20260628)
        moves = []                              # reservoir sample of matching records
        seen = 0
        with open(IDX2_NODES) as f:
            for line in f:
                if '"moves"' not in line:        # skip `done` resume-markers fast
                    continue
                if want_capped and '"verdict": "capped"' not in line:
                    continue                     # capped pool: capped-only; quiet pool: natural mix
                seen += 1
                if len(moves) < n:
                    moves.append(json.loads(line)["moves"])
                else:
                    j = rng.randint(0, seen - 1)
                    if j < n:
                        moves[j] = json.loads(line)["moves"]
        boards = _reconstruct(moves, pool_dir, name)
    else:
        raise ValueError(name)
    np.savez(cache, boards=boards)
    print(f"  pool '{name}': {boards.shape[0]} boards in {time.time()-t:.0f}s -> {cache}",
          flush=True)
    return boards


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def run_base(bd, budget):
    return solve_vct_mega_bb(bd, max_nodes=budget)


def run_chunked(bd, budget, chunk=16384):
    wins = np.zeros(bd.shape[0], bool); hits = np.zeros(bd.shape[0], bool)
    for i in range(0, bd.shape[0], chunk):
        w, h = solve_vct_mega_bb(bd[i:i + chunk], max_nodes=budget)
        wins[i:i + chunk] = w; hits[i:i + chunk] = h
    return wins, hits


def run_worksteal(bd, budget, resident):
    return solve_vct_mega_bb(bd, max_nodes=budget, work_steal=True, resident=resident)


def run_deepen(bd, ladder, work_steal=False):
    return solve_vct_streaming(bd, budgets=tuple(ladder), work_steal=work_steal)


# --------------------------------------------------------------------------- #
# Resolution profile: each board's MIN resolving budget (latching ascending
# ladder) -> the hardness distribution that bounds the achievable optimum.
# --------------------------------------------------------------------------- #
def resolution_profile(bd, ladder):
    n = bd.shape[0]
    min_budget = np.full(n, -1, dtype=np.int64)     # -1 = still capped at top rung
    active = np.arange(n)
    per_rung = {}
    for bud in ladder:
        if active.size == 0:
            break
        t = time.time()
        w, h = solve_vct_mega_bb(bd[active], max_nodes=int(bud))
        clean = ~h
        min_budget[active[clean]] = bud
        per_rung[bud] = {"in": int(active.size), "resolved": int(clean.sum()),
                         "wins": int((w & clean).sum()), "wall_s": round(time.time() - t, 2)}
        active = active[~clean]
    return min_budget, per_rung


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def cfg_id(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def metrics(bd, wins, hits, wall):
    n = bd.shape[0]
    resolved = int((~hits).sum())
    return {"N": n, "wall_s": round(wall, 3),
            "evals_per_min": round(n / wall * 60, 1),         # boards SCREENED/min
            "resolved_per_min": round(resolved / wall * 60, 1),  # boards given a VERDICT/min
            "resolved": resolved, "wins": int(wins.sum()),
            "capped": int(hits.sum())}


def enumerate_configs(pools, budgets, ladders, ws_residents):
    cfgs = []
    for pool in pools:
        for b in budgets:
            cfgs.append({"pool": pool, "strategy": "base", "budget": b})
        for b in (250, 1000):
            cfgs.append({"pool": pool, "strategy": "chunked", "budget": b})
        for r in ws_residents:
            for b in (250, 1000, 4000):
                cfgs.append({"pool": pool, "strategy": "worksteal", "budget": b, "resident": r})
        for name, lad in ladders.items():
            cfgs.append({"pool": pool, "strategy": "deepen", "ladder": lad, "ladder_name": name})
        cfgs.append({"pool": pool, "strategy": "deepen_ws", "ladder": ladders["coarse"],
                     "ladder_name": "coarse"})
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/data/idx2_solve/sweep"))
    ap.add_argument("--n", type=int, default=20000)            # throughput pool size
    ap.add_argument("--profile-n", type=int, default=50000)    # profile pool size
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end validation (every code path, ~1-2 min)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pool_dir = os.path.join(args.out, "pools"); os.makedirs(pool_dir, exist_ok=True)
    results_path = os.path.join(args.out, "results.jsonl")
    profile_path = os.path.join(args.out, "profiles.jsonl")

    POOLS = ["quiet", "capped", "random", "deep"]
    BUDGETS = [10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 20000]
    LADDERS = {
        "low":      [10, 25, 50, 100, 250],
        "coarse":   [250, 1000, 4000],
        "wide":     [10, 50, 250, 1000, 4000, 20000],
        "full_low": [10, 25, 50, 100, 250, 500, 1000, 2000, 4000],
    }
    WS_RESIDENTS = [4096, 8192, 16384]
    PROFILE_LADDER = [10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000, 20000]
    if args.smoke:                              # validate every code path, fast
        args.n, args.profile_n = 300, 500
        POOLS = ["quiet", "capped", "random"]
        BUDGETS = [10, 250, 1000]
        LADDERS = {"low": [10, 25, 250], "coarse": [250, 1000]}
        WS_RESIDENTS = [8192]
        PROFILE_LADDER = [10, 250, 1000]

    print(f"MAXD={MAXD}  out={args.out}", flush=True)

    # -- resume: which config_ids / profile pools already done --
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["config_id"])
                except Exception:
                    pass
    profiled = set()
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            for line in f:
                try:
                    profiled.add(json.loads(line)["pool"])
                except Exception:
                    pass

    # ---- build pools once ----
    pools = {}
    for name in POOLS:
        bd = build_pool(name, max(args.n, args.profile_n), pool_dir)
        if bd is not None:
            pools[name] = bd

    # ---- PHASE 1: resolution profiles (the hardness map / oracle input) ----
    for name, bd in pools.items():
        if name in profiled:
            print(f"profile[{name}] already done, skip", flush=True)
            continue
        sub = bd[:args.profile_n]
        print(f"\n=== profile pool={name} N={sub.shape[0]} ladder={PROFILE_LADDER} ===", flush=True)
        t = time.time()
        try:
            min_budget, per_rung = resolution_profile(sub, PROFILE_LADDER)
            hist = {int(b): int((min_budget == b).sum()) for b in PROFILE_LADDER}
            still = int((min_budget == -1).sum())
            row = {"pool": name, "N": int(sub.shape[0]), "ladder": PROFILE_LADDER,
                   "maxd": MAXD, "hist_min_budget": hist, "still_capped": still,
                   "per_rung": per_rung, "wall_s": round(time.time() - t, 2),
                   "ts": time.time()}
            with open(profile_path, "a") as f:
                f.write(json.dumps(row) + "\n"); f.flush()
            print(f"  profile[{name}] hist={hist} still_capped={still} "
                  f"({time.time()-t:.0f}s)", flush=True)
        except Exception as e:
            print(f"  profile[{name}] FAILED: {type(e).__name__}: {e}", flush=True)

    # ---- PHASE 2: throughput sweep ----
    cfgs = enumerate_configs(POOLS, BUDGETS, LADDERS, WS_RESIDENTS)
    print(f"\n=== sweep: {len(cfgs)} configs ({len(done)} already done) ===", flush=True)
    for cfg in cfgs:
        if cfg["pool"] not in pools:
            continue
        cid = cfg_id(cfg)
        if cid in done:
            continue
        bd = pools[cfg["pool"]][:args.n]
        t = time.time()
        try:
            s = cfg["strategy"]
            if s == "base":
                w, h = run_base(bd, cfg["budget"])
            elif s == "chunked":
                w, h = run_chunked(bd, cfg["budget"])
            elif s == "worksteal":
                w, h = run_worksteal(bd, cfg["budget"], cfg["resident"])
            elif s == "deepen":
                w, h = run_deepen(bd, cfg["ladder"], work_steal=False)
            elif s == "deepen_ws":
                w, h = run_deepen(bd, cfg["ladder"], work_steal=True)
            else:
                raise ValueError(s)
            row = {"config_id": cid, "maxd": MAXD, **cfg, **metrics(bd, w, h, time.time() - t),
                   "ts": time.time()}
            label = f"{cfg['pool']:7} {s:10} b={cfg.get('budget','-')!s:6} " \
                    f"{cfg.get('ladder_name','')!s:8} r={cfg.get('resident','-')}"
            print(f"  {label}  {row['wall_s']:7.2f}s  {row['evals_per_min']:>9.0f} ev/min  "
                  f"resolved={row['resolved']}/{row['N']}", flush=True)
        except Exception as e:
            row = {"config_id": cid, "maxd": MAXD, **cfg, "error": f"{type(e).__name__}: {e}",
                   "ts": time.time()}
            print(f"  {cfg['pool']} {cfg['strategy']} FAILED: {row['error']}", flush=True)
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n"); f.flush()
        done.add(cid)

    print("\nSWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
