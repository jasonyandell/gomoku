"""Capture real merged-veto solver batches from a live gen run and bench the
lanes=K multi-thread-per-board kernel against base on them (issue #114).

Day-1 (#114) captured 22 real 13x13 batches in a throwaway scratchpad that did
not survive the session; this is the committed, reproducible version of that
methodology. Two modes:

  # 1) capture: run a short live-semantics gen (vct-terminus cap50 + full-breadth
  #    veto) and record every (boards, max_nodes) the oracle actually dispatched:
  GOMOKU_BOARD_SIZE=13 uv run python scripts/vct_metal/bench_lanes13.py \
      --capture /tmp/b13.npz --ckpt checkpoints_13x13_medium/latest.pt \
      --games 48 --concurrent 32

  # 2) bench: replay the captured batches serially (as gen does) per variant,
  #    asserting (win, hit, move) equality on every batch:
  GOMOKU_BOARD_SIZE=13 uv run python scripts/vct_metal/bench_lanes13.py \
      --bench /tmp/b13.npz --lanes 2,4,8,16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())


def capture(args) -> None:
    import torch
    from gomoku import self_play as sp
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint
    from gomoku.self_play import generate_games

    device = torch.device(args.device)
    model, _ = load_checkpoint(args.ckpt, device=device)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, device)

    sp.configure_vct_terminus(enabled=True, budget=args.budget)
    sp.configure_oracle_veto(enabled=True, max_cands=0)
    # serial solves: the capture IS the workload, don't hide it under search
    sp.configure_oracle_overlap(enabled=False)
    sp._warm_mega_solver()

    real = sp._load_mega_solver()
    batches: list[tuple[np.ndarray, int]] = []

    def recording_solver(boards, **kw):
        batches.append((np.asarray(boards).copy(), int(kw.get("max_nodes", 0))))
        return real(boards, **kw)

    sp._vct_terminus_solver = recording_solver
    t0 = time.perf_counter()
    generate_games(args.games, evaluator, n_simulations=args.sims,
                   wave_size=args.wave_size,
                   rng=np.random.default_rng(args.seed),
                   concurrent_games=args.concurrent)
    wall = time.perf_counter() - t0

    out = {"n_batches": np.int64(len(batches)),
           "budgets": np.array([b for _, b in batches], dtype=np.int64)}
    for i, (bd, _) in enumerate(batches):
        out[f"batch_{i}"] = bd
    np.savez_compressed(args.capture, **out)
    sizes = [b.shape[0] for b, _ in batches]
    print(json.dumps({
        "captured": len(batches), "boards_total": int(np.sum(sizes)),
        "boards_min": int(np.min(sizes)), "boards_max": int(np.max(sizes)),
        "gen_wall_s": round(wall, 1), "out": args.capture}))


def bench(args) -> None:
    from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb

    z = np.load(args.bench)
    nb = int(z["n_batches"])
    budgets = z["budgets"]
    batches = [z[f"batch_{i}"] for i in range(nb)]
    lanes = [int(x) for x in args.lanes.split(",")]
    print(f"{nb} batches, {sum(b.shape[0] for b in batches)} boards, "
          f"budget[0]={int(budgets[0])}")

    # reference pass (also warms every kernel out of the timing)
    ref = [solve_vct_mega_bb(b, max_nodes=int(m), return_move=True)
           for b, m in zip(batches, budgets)]
    for k in lanes:
        solve_vct_mega_bb(batches[0][:8], max_nodes=10, return_move=True, lanes=k)

    results = {}
    for k in [1] + lanes:
        kw = {} if k == 1 else {"lanes": k}
        best = None
        for _ in range(args.reps):
            t0 = time.perf_counter()
            outs = [solve_vct_mega_bb(b, max_nodes=int(m), return_move=True, **kw)
                    for b, m in zip(batches, budgets)]
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        for o, r in zip(outs, ref):
            for a, b_ in zip(o, r):
                assert (np.asarray(a) == np.asarray(b_)).all(), \
                    f"verdict mismatch at lanes={k}"
        results[k] = best
        base = results[1]
        print(f"lanes={k:>2}  total={best:6.2f}s  vs base {base / best:.2f}x  "
              f"(verdicts identical)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", help="output npz path (capture mode)")
    ap.add_argument("--bench", help="input npz path (bench mode)")
    ap.add_argument("--ckpt")
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--concurrent", type=int, default=32)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--wave-size", type=int, default=32)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--lanes", default="2,4,8,16")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    if args.capture:
        assert args.ckpt, "--capture needs --ckpt"
        capture(args)
    elif args.bench:
        bench(args)
    else:
        ap.error("pick --capture or --bench")


if __name__ == "__main__":
    main()
