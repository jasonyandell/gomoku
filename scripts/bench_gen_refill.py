"""Bench the continuous-refill consolidated generator (issue #112).

Measures self-play generation throughput at the live sound-world config
(vct-terminus cap50 + oracle-veto full breadth + oracle-overlap) for a given
(games, concurrent) shape, printing games/min, positions/s, and the profile
split (oracle vs evaluator vs tree). Run it once per shape:

  # today's per-worker shape (one lockstep batch of 8):
  uv run python scripts/bench_gen_refill.py --games 16 --concurrent 0 --ckpt <p>
  # consolidated continuous-refill worker:
  uv run python scripts/bench_gen_refill.py --games 64 --concurrent 32 --ckpt <p>

For the true 4-process aggregate baseline, launch 4 of the first shape
simultaneously (see --tag to label output lines).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())  # `scripts.vct_metal` import needs the repo root

from gomoku import self_play as sp
from gomoku.mcts import make_torch_evaluator
from gomoku.model import fuse_model_for_inference, load_checkpoint
from gomoku.self_play import generate_games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--concurrent", type=int, default=0)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--wave-size", type=int, default=32)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-oracle", action="store_true",
                    help="bench the bare generator (no terminus/veto/overlap)")
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, _payload = load_checkpoint(args.ckpt, device=device)
    model = fuse_model_for_inference(model)
    if args.fp16:
        model = model.half()
    evaluator = make_torch_evaluator(model, device, fp16=args.fp16)

    if not args.no_oracle:
        sp.configure_vct_terminus(enabled=True, budget=args.budget)
        sp.configure_oracle_veto(enabled=True, max_cands=0)
        sp.configure_oracle_overlap(enabled=not args.no_overlap)
        sp._warm_mega_solver()

    # Warm the MPS path (compile + first-dispatch costs stay out of the timing).
    generate_games(1, evaluator, n_simulations=8, wave_size=args.wave_size,
                   rng=np.random.default_rng(0), augment_symmetries=False)

    profile: dict[str, float] = {}
    t0 = time.perf_counter()
    records = generate_games(
        args.games,
        evaluator,
        n_simulations=args.sims,
        wave_size=args.wave_size,
        rng=np.random.default_rng(args.seed),
        concurrent_games=args.concurrent,
        profile=profile,
    )
    wall = time.perf_counter() - t0

    plies = [r.plies for r in records]
    n_examples = sum(len(r.examples) for r in records)
    out = {
        "tag": args.tag,
        "games": args.games,
        "concurrent": args.concurrent,
        "wall_s": round(wall, 2),
        "games_per_min": round(60.0 * len(records) / wall, 2),
        "aug_pos_per_s": round(n_examples / wall, 1),
        "plies_mean": round(float(np.mean(plies)), 1),
        "plies_max": int(np.max(plies)),
        "oracle_s": round(profile.get("oracle_solve_s", 0.0)
                          + profile.get("oracle_build_s", 0.0)
                          + profile.get("oracle_planes_s", 0.0), 2),
        "oracle_solve_s": round(profile.get("oracle_solve_s", 0.0), 2),
        "oracle_build_s": round(profile.get("oracle_build_s", 0.0), 2),
        "oracle_planes_s": round(profile.get("oracle_planes_s", 0.0), 2),
        "oracle_calls": int(profile.get("oracle_solve_calls", 0)),
        "oracle_boards": int(profile.get("oracle_boards", 0)),
        "oracle_join_stall_s": round(profile.get("oracle_join_stall_s", 0.0), 2),
        "search_s": round(profile.get("native_search_batch_s", 0.0), 2),
        "evaluator_s": round(profile.get("evaluator_s", 0.0), 2),
        "evaluator_positions": int(profile.get("evaluator_positions", 0)),
        "evaluator_calls": int(profile.get("evaluator_calls", 0)),
        "record_build_s": round(profile.get("record_build_s", 0.0), 2),
        "terminus_fired": int(profile.get("vct_terminus_fired", 0)),
        "veto_all_lose": int(profile.get("oracle_veto_all_lose", 0)),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
