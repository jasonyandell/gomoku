"""Small MCTS generation benchmark for Activity Monitor-oriented tuning.

This is intentionally production-shaped rather than profiler-perfect: it builds
or loads a model, runs bounded self-play through the existing MCTS generator,
and reports wall-clock throughput. Use it to compare worker/wave/config ideas
before touching a long W&B run. Do not treat GPU percent as the score.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gomoku.mcts import make_torch_evaluator
from gomoku.model import build_model, fuse_model_for_inference, load_checkpoint, n_params
from gomoku import native_mcts, state_ops
from gomoku.self_play import generate_games
from gomoku.util import pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional checkpoint to benchmark. Default builds a fresh model.")
    p.add_argument("--size", type=str, default="small",
                   help="Model size when --checkpoint is not set.")
    p.add_argument("--stem-padding", type=int, default=1,
                   help="Stem padding when --checkpoint is not set. WL1/Z use 1.")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device override. Default is MPS when available.")
    p.add_argument("--games", type=int, default=8,
                   help="Concurrent self-play games in the bench batch.")
    p.add_argument("--n-simulations", type=int, default=100,
                   help="MCTS simulations per move.")
    p.add_argument("--wave-size", type=int, default=64,
                   help="Wave-batched MCTS leaves per game per evaluator call.")
    p.add_argument("--max-plies", type=int, default=12,
                   help="Stop each game after this many plies so the bench is bounded.")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1,
                   help="Unmeasured repeats before timing.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp16", action="store_true",
                   help="Pass fp16=True to the evaluator. Historically slower on MPS.")
    p.add_argument("--no-fuse-eval", action="store_true",
                   help="Skip eval-only Conv+BatchNorm fusion.")
    return p.parse_args()


def sync_device(device: torch.device) -> None:
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def run_once(args: argparse.Namespace, evaluator, rng: np.random.Generator) -> dict[str, float]:
    sync_device(args.device)
    start = time.perf_counter()
    records = generate_games(
        args.games,
        evaluator,
        n_simulations=args.n_simulations,
        wave_size=args.wave_size,
        max_plies=args.max_plies,
        rng=rng,
        augment_symmetries=False,
    )
    sync_device(args.device)
    elapsed = time.perf_counter() - start
    raw_positions = sum(r.plies for r in records)
    return {
        "seconds": elapsed,
        "games": float(len(records)),
        "plies_mean": float(np.mean([r.plies for r in records])) if records else 0.0,
        "raw_positions": float(raw_positions),
        "aug_positions_equiv": float(raw_positions * 8),
    }


def main() -> None:
    args = parse_args()
    args.device = pick_device(args.device)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.checkpoint:
        model, payload = load_checkpoint(args.checkpoint, device=args.device)
        source = f"checkpoint={args.checkpoint} epoch={payload.get('epoch', '?')}"
    else:
        model = build_model(args.size, stem_padding=args.stem_padding).to(args.device)
        source = f"fresh size={args.size} stem_padding={args.stem_padding}"
    model.eval()
    param_count = n_params(model)
    if not args.no_fuse_eval:
        model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, args.device, fp16=args.fp16)

    print(
        f"device={args.device} params={param_count:,} "
        f"fused_eval={not args.no_fuse_eval} {source}"
    )
    print(
        f"native_mcts={native_mcts.USING_NATIVE_MCTS} "
        f"native_state_ops={state_ops.USING_NATIVE}"
    )
    print(
        f"bench games={args.games} sims={args.n_simulations} wave={args.wave_size} "
        f"max_plies={args.max_plies} repeats={args.repeats}"
    )
    print("watch Activity Monitor, but score wall seconds and positions/sec, not GPU percent")

    for _ in range(args.warmup):
        run_once(args, evaluator, rng)

    samples = []
    for i in range(args.repeats):
        result = run_once(args, evaluator, rng)
        samples.append(result)
        gps = result["games"] / result["seconds"] if result["seconds"] else 0.0
        pps = result["aug_positions_equiv"] / result["seconds"] if result["seconds"] else 0.0
        print(
            f"trial {i + 1}: {result['seconds']:.3f}s "
            f"games/s={gps:.2f} aug_pos/s={pps:.0f} "
            f"plies_mean={result['plies_mean']:.1f}"
        )

    secs = [s["seconds"] for s in samples]
    games = [s["games"] for s in samples]
    aug_positions = [s["aug_positions_equiv"] for s in samples]
    med_s = statistics.median(secs)
    total_games = statistics.median(games)
    total_aug_positions = statistics.median(aug_positions)
    print(
        "median: "
        f"{med_s:.3f}s games/s={total_games / med_s:.2f} "
        f"aug_pos/s={total_aug_positions / med_s:.0f}"
    )


if __name__ == "__main__":
    main()
