"""Build a retain-all PositionArchive to replay from.

Produces an on-disk `gomoku.curated_buffer.PositionArchive` (a directory of
sharded .pt files + manifest.json). The archive is the "arbitrary pasts" store
from wiki/topics/curated-buffer-and-curriculum-design.md — retain everything,
train from a curated slice.

Three seed paths (pick the easiest real one available):

  --from-selfplay         Fresh self-play with a checkpoint (or fresh model).
                          Self-contained; the smoke path. Tags positions with a
                          monotonically increasing pseudo-model_version per batch
                          so recency curators have something to discriminate.

  --from-checkpoint-buffer  Reuse a checkpoint's saved replay_buffer snapshot
                          (e.g. sweep_runs/*/checkpoints/latest.pt). Uses the
                          buffer's own weight_version / side / ply tags. The
                          cheapest "real data" path — no generation needed.

  --from-validation-archive  Reuse a mine_validation_archive.py archive
                          (planes/pi_mcts/z/side/ply). Tags source=ARCHIVE.

Usage examples:
    # fresh self-play, a few hundred positions, no checkpoint:
    python scripts/build_archive.py --out archives/smoke --from-selfplay \\
        --games 8 --n-simulations 12 --wave-size 8 --batches 3

    # from an existing checkpoint's replay buffer:
    python scripts/build_archive.py --out archives/wl5_replay \\
        --from-checkpoint-buffer sweep_runs/WL5-diagnostics-archive-start/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gomoku.curated_buffer import (  # noqa: E402
    SOURCE_ARCHIVE,
    SOURCE_SELFPLAY,
    PositionArchive,
)
from gomoku.game import N_INPUT_PLANES  # noqa: E402
from gomoku.mcts import make_torch_evaluator  # noqa: E402
from gomoku.model import build_model, load_checkpoint  # noqa: E402
from gomoku.self_play import generate_games  # noqa: E402
from gomoku.util import pick_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output archive directory.")
    # source selection (exactly one)
    p.add_argument("--from-selfplay", action="store_true",
                   help="Seed via fresh self-play.")
    p.add_argument("--from-checkpoint-buffer", type=str, default=None,
                   help="Seed from a checkpoint's saved replay_buffer.")
    p.add_argument("--from-validation-archive", type=str, default=None,
                   help="Seed from a mine_validation_archive.py archive.")
    # self-play knobs
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint for the self-play model (default: fresh tiny model).")
    p.add_argument("--size", type=str, default="tiny")
    p.add_argument("--games", type=int, default=8, help="Games per self-play batch.")
    p.add_argument("--batches", type=int, default=4,
                   help="Self-play batches; each tagged with an incrementing version.")
    p.add_argument("--n-simulations", type=int, default=16)
    p.add_argument("--wave-size", type=int, default=8)
    p.add_argument("--base-version", type=int, default=0,
                   help="Starting model_version tag for the first batch.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_from_selfplay(archive: PositionArchive, args: argparse.Namespace) -> int:
    device = pick_device(args.device)
    print(f"device = {device}")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.resume:
        model, _ = load_checkpoint(args.resume, device=device)
        print(f"self-play model: resumed from {args.resume}")
    else:
        model = build_model(args.size).to(device)
        print(f"self-play model: fresh {args.size}")
    model.eval()
    evaluator = make_torch_evaluator(model, device)

    total = 0
    for b in range(args.batches):
        version = args.base_version + b
        records = generate_games(
            args.games, evaluator,
            n_simulations=args.n_simulations,
            wave_size=args.wave_size,
            rng=rng,
        )
        # One shard per self-play batch (not per game), tagged with this batch's
        # version. Per-position `plies` carries the per-position ply-at-capture
        # (e.ply) so a curator can ply-balance on real game lengths.
        all_examples = [e for r in records for e in r.examples]
        archive.append_examples(
            all_examples, model_version=version, plies=None, source=SOURCE_SELFPLAY,
        )
        n_batch = len(all_examples)
        total += n_batch
        print(f"  batch {b + 1}/{args.batches} version={version}: "
              f"{len(records)} games, {n_batch} positions (total {total})", flush=True)
    return total


def build_from_checkpoint_buffer(archive: PositionArchive, path: str) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    buf = payload.get("replay_buffer")
    if buf is None:
        raise SystemExit(f"{path} has no replay_buffer snapshot")
    planes = buf["planes"].to(torch.float32)
    pi = buf["pi"].to(torch.float32)
    z = buf["z"].to(torch.float32)
    n = planes.shape[0]
    if planes.shape[1] != N_INPUT_PLANES:
        raise SystemExit(f"buffer plane count {planes.shape[1]} != {N_INPUT_PLANES}")
    version = buf.get("weight_version")
    side = buf.get("side")
    ply = buf.get("ply")
    version = version.to(torch.int32) if version is not None else torch.zeros(n, dtype=torch.int32)
    color = side.to(torch.int8) if side is not None else torch.zeros(n, dtype=torch.int8)
    plies = ply.to(torch.int16) if ply is not None else torch.zeros(n, dtype=torch.int16)
    # One append; the archive shards it as a single block.
    archive.append(
        planes, pi, z,
        model_version=version, plies=plies, color=color, source=SOURCE_SELFPLAY,
    )
    print(f"ingested {n} positions from checkpoint buffer {path}")
    return n


def build_from_validation_archive(archive: PositionArchive, path: str) -> int:
    va = torch.load(path, map_location="cpu", weights_only=False)
    planes = va["planes"].to(torch.float32)
    pi = va["pi_mcts"].to(torch.float32)
    z = va["z"].to(torch.float32)
    n = planes.shape[0]
    side = va.get("side")
    ply = va.get("ply")
    color = side.to(torch.int8) if side is not None else torch.zeros(n, dtype=torch.int8)
    plies = ply.to(torch.int16) if ply is not None else torch.zeros(n, dtype=torch.int16)
    archive.append(
        planes, pi, z,
        model_version=0, plies=plies, color=color, source=SOURCE_ARCHIVE,
    )
    print(f"ingested {n} positions from validation archive {path}")
    return n


def main() -> None:
    args = parse_args()
    n_sources = sum(bool(x) for x in (
        args.from_selfplay, args.from_checkpoint_buffer, args.from_validation_archive))
    if n_sources != 1:
        raise SystemExit("specify exactly one of --from-selfplay / "
                         "--from-checkpoint-buffer / --from-validation-archive")

    archive = PositionArchive(args.out, readonly=False)
    if args.from_selfplay:
        total = build_from_selfplay(archive, args)
    elif args.from_checkpoint_buffer:
        total = build_from_checkpoint_buffer(archive, args.from_checkpoint_buffer)
    else:
        total = build_from_validation_archive(archive, args.from_validation_archive)

    print(f"\narchive written to {args.out}: {len(archive)} positions "
          f"({len(archive._shards)} shards)")
    # Quick metadata sanity print.
    meta = archive.read_metadata()
    if len(meta["model_version"]):
        mv = meta["model_version"]
        print(f"  versions: {mv.min()}..{mv.max()}  "
              f"colors: black={int((meta['color'] == 0).sum())} white={int((meta['color'] == 1).sum())}  "
              f"z: +{int((meta['z'] > 0).sum())}/-{int((meta['z'] < 0).sum())}/={int((meta['z'] == 0).sum())}")


if __name__ == "__main__":
    main()
