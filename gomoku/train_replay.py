"""Replay-training mode: train from a FROZEN archive at a FIXED SGD budget.

This is the trainer half of "Layer 1" of
wiki/topics/curated-buffer-and-curriculum-design.md. It deliberately does
NOT do self-play: it loads a frozen on-disk `PositionArchive`, has a
`CuratedBuffer` curate an in-RAM slice, and runs a fixed number of SGD steps
per epoch (`--sgd-steps-per-epoch`). Because there is no generation in the loop
and the step count is fixed, every epoch costs ~the same — by construction the
LF1 runaway (wiki/topics/perf-bench-vs-real-training-cost.md) cannot happen.

It reuses train.py's model/SGD/EMA/eval/validation plumbing wholesale:
`train_step`, `ema_update`, `ema_l2_distance`, `_score_validation_archive`,
`_baseline_log_key`, `_parse_specs`. Only the data path (archive+curator instead
of self-play+FIFO) and the step schedule (fixed instead of sgd_per_position)
differ.

CLI CONTRACT (a sibling harness depends on these EXACT flags):
    --resume C                  checkpoint to resume the model from
    --archive-path PATH         frozen PositionArchive directory to replay
    --buffer-recipe NAME        curator name (default: lru)
    --sgd-steps-per-epoch N     FIXED optimizer microbatches per epoch
    --epochs N
    --size
    --validation-archive-path
    --eval-* (the existing eval flags)
    --wandb / --no-wandb

NOTE (live-gen hook): replay-from-frozen-archive is the immediate use case, so
there is no worker pool here. Lazy/pull-based live generation would hook in at
CuratedBuffer.refill() (signal "need N more from source X", block on workers,
then gather) — see the comment there. The trainer loop below would be unchanged.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from gomoku.curated_buffer import CuratedBuffer, PositionArchive, list_curators
from gomoku.eval import mcts_picker, play_match_pickers
from gomoku.match import build_player, parse_spec
from gomoku.mcts import make_torch_evaluator
from gomoku.model import build_model, load_checkpoint, n_params, save_checkpoint
from gomoku.train import (
    _baseline_log_key,
    _parse_specs,
    _score_validation_archive,
    ema_l2_distance,
    ema_update,
    train_step,
)
from gomoku.util import load_wandb_key_from_keychain, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # --- CLI CONTRACT (exact names — sibling harness depends on them) -------
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint to resume the model (+optimizer/EMA) from.")
    p.add_argument("--archive-path", type=str, required=True,
                   help="Frozen PositionArchive directory to replay from.")
    p.add_argument("--buffer-recipe", type=str, default="lru",
                   help=f"Curator recipe. Options: {list_curators()}")
    p.add_argument("--sgd-steps-per-epoch", type=int, default=400,
                   help="FIXED optimizer microbatches per epoch (decoupled from "
                        "any generation rate — this is what kills the runaway).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--size", type=str, default="small", help="tiny / small / medium / large")
    p.add_argument("--validation-archive-path", type=str, default=None,
                   help="Frozen validation archive (mine_validation_archive.py format).")
    # --- eval flags (mirror train.py exactly) -------------------------------
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-sims", type=int, default=50)
    p.add_argument("--eval-baselines", type=str, default="random,heuristic")
    p.add_argument("--eval-baseline-games", type=int, default=4)
    p.add_argument("--eval-baselines-slow", type=str, default="")
    p.add_argument("--eval-slow-every", type=int, default=4)
    p.add_argument("--eval-slow-games", type=int, default=6)
    p.add_argument("--no-eval", dest="eval_in_trainer", action="store_false")
    p.set_defaults(eval_in_trainer=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.set_defaults(wandb=False)
    p.add_argument("--wandb-project", type=str, default="gomoku")
    p.add_argument("--run-name", type=str, default=None)
    # --- slice / SGD knobs --------------------------------------------------
    p.add_argument("--slice-size", type=int, default=200_000,
                   help="Curated in-RAM sample-pool width. Selection matters "
                        "more than width (see project-buffer-curation).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--ema-tau", type=float, default=0.0,
                   help="EMA self-play weights (WL2 lever #1). 0 disables.")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--stem-padding", type=int, default=None)
    # --- bookkeeping --------------------------------------------------------
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--composition-tsv", type=str, default=None,
                   help="Append per-refill slice composition here when --wandb "
                        "is off (defaults to <checkpoint-dir>/composition.tsv).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device = {device}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- model / optimizer (resume reuses train.py's checkpoint format) -----
    start_epoch = 0
    total_games = 0
    wandb_run_id = None
    if args.resume:
        model, payload = load_checkpoint(args.resume, device=device)
        start_epoch = int(payload.get("epoch", 0))
        total_games = int(payload.get("total_games", 0))
        wandb_run_id = payload.get("wandb_run_id")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        print(f"resumed from {args.resume} @ epoch {start_epoch}, total_games={total_games}")
    else:
        model = build_model(args.size, stem_padding=args.stem_padding).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        payload = {}
        print(f"fresh {args.size} model: {n_params(model):,} params")

    # --- EMA (WL2 lever #1) — identical to train.py -------------------------
    ema_model = None
    if args.ema_tau > 0.0:
        if not (0.0 < args.ema_tau < 1.0):
            raise SystemExit(f"--ema-tau must be in (0, 1); got {args.ema_tau}")
        from gomoku.model import GomokuNet
        ema_model = GomokuNet(model.cfg).to(device)
        if args.resume and "ema_model_state_dict" in payload:
            ema_model.load_state_dict(payload["ema_model_state_dict"])
            print(f"ema model restored from checkpoint (tau={args.ema_tau})")
        else:
            ema_model.load_state_dict(model.state_dict())
            print(f"ema model initialized from current weights (tau={args.ema_tau})")
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()

    # --- validation archive (mine_validation_archive.py format) -------------
    validation_archive = None
    if args.validation_archive_path:
        validation_archive = torch.load(
            args.validation_archive_path, map_location="cpu", weights_only=False,
        )
        print(f"validation archive loaded: {validation_archive['planes'].shape[0]} positions "
              f"from {args.validation_archive_path}")

    # --- frozen archive + curated buffer (THE replay data path) -------------
    archive = PositionArchive(args.archive_path, readonly=True)
    if len(archive) == 0:
        raise SystemExit(f"archive {args.archive_path} is empty — build one with scripts/build_archive.py")
    buffer = CuratedBuffer(
        archive,
        slice_size=args.slice_size,
        curator=args.buffer_recipe,
        device=device,
        seed=args.seed,
    )
    print(f"replay archive: {len(archive)} positions, curator={args.buffer_recipe!r}, "
          f"slice_size={args.slice_size}")

    # --- wandb (mirror train.py) --------------------------------------------
    run = None
    if args.wandb:
        key = load_wandb_key_from_keychain()
        if not key:
            print("warning: --wandb set but no WANDB_API_KEY in keychain or env")
        else:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                id=wandb_run_id,
                resume="allow" if wandb_run_id else None,
                config=vars(args),
            )
            wandb_run_id = run.id

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    composition_tsv = None
    if run is None:
        composition_tsv = args.composition_tsv or os.path.join(args.checkpoint_dir, "composition.tsv")

    # --- eval setup (mirror train.py) ---------------------------------------
    fast_specs = _parse_specs(args.eval_baselines)
    slow_specs = _parse_specs(args.eval_baselines_slow)
    fast_pickers = [(s, build_player(s)) for s in fast_specs]
    slow_pickers = [(s, build_player(s)) for s in slow_specs]
    eval_counter = 0

    accum_n = max(1, int(args.grad_accum_steps))

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()

        # --- refill the curated slice at the EPOCH boundary (not per-batch) --
        refill_start = time.time()
        slice_n = buffer.refill(cycle=epoch)
        refill_time = time.time() - refill_start
        comp = buffer.log_composition(epoch, wandb_run=run, tsv_path=composition_tsv)

        # --- training: FIXED SGD budget, no generation ----------------------
        train_metrics_acc: dict[str, list[float]] = {}
        train_start = time.time()
        optimizer_steps_this_epoch = 0
        steps_this_epoch = int(args.sgd_steps_per_epoch)
        if buffer.size >= args.batch_size:
            for i in range(steps_this_epoch):
                planes, pi, z, side, ply = buffer.sample(args.batch_size)
                is_first_in_window = (i % accum_n == 0)
                is_last_in_window = ((i + 1) % accum_n == 0) or (i + 1 == steps_this_epoch)
                m = train_step(
                    model, optimizer, planes, pi, z,
                    value_weight=args.value_weight,
                    l2_weight=args.l2,
                    accum_scale=1.0 / accum_n,
                    do_optimizer_step=is_last_in_window,
                    zero_grad=is_first_in_window,
                    side=side,
                    ply=ply,
                )
                for k, v in m.items():
                    train_metrics_acc.setdefault(k, []).append(v)
                if is_last_in_window:
                    optimizer_steps_this_epoch += 1
                    if ema_model is not None:
                        ema_update(ema_model, model, args.ema_tau)
        else:
            print(f"WARNING: slice ({buffer.size}) < batch_size ({args.batch_size}); no SGD this epoch")
        train_time = time.time() - train_start
        train_metrics = {k: float(np.mean(v)) for k, v in train_metrics_acc.items()}

        log = {
            "epoch": epoch + 1,
            "total_games": total_games,
            "replay/slice_n": slice_n,
            "train/steps_this_epoch": steps_this_epoch,
            "train/optimizer_steps_this_epoch": optimizer_steps_this_epoch,
            "time/refill_s": refill_time,
            "time/train_s": train_time,
            **train_metrics,
            **comp,
        }
        if ema_model is not None:
            log["train/ema_l2_distance"] = ema_l2_distance(ema_model, model)

        # --- validation archive scoring (reused from train.py) --------------
        if validation_archive is not None and (epoch + 1) % args.eval_every == 0:
            va_start = time.time()
            log.update(_score_validation_archive(model, validation_archive, device))
            log["time/val_archive_s"] = time.time() - va_start

        # --- in-trainer eval (reused pattern from train.py) -----------------
        if args.eval_in_trainer and (epoch + 1) % args.eval_every == 0:
            eval_counter += 1
            eval_start = time.time()
            eval_evaluator = make_torch_evaluator(model, device)
            model_picker = mcts_picker(eval_evaluator, n_simulations=args.eval_sims)
            run_slow = bool(slow_pickers) and (eval_counter % args.eval_slow_every == 0)
            batches = [("fast", fast_pickers, args.eval_baseline_games)]
            if run_slow:
                batches.append(("slow", slow_pickers, args.eval_slow_games))
            for batch_label, pickers, n_games in batches:
                for spec_idx, (spec, baseline_picker) in enumerate(pickers):
                    m_start = time.time()
                    res = play_match_pickers(
                        model_picker, baseline_picker,
                        n_games=n_games,
                        seed=args.seed + epoch * 1000 + spec_idx,
                    )
                    key = _baseline_log_key(spec)
                    log[f"eval/{key}_winrate"] = res.win_rate
                    log[f"eval/{key}_wins"] = res.wins
                    log[f"eval/{key}_losses"] = res.losses
                    log[f"eval/{key}_draws"] = res.draws
                    log[f"time/eval_{key}_s"] = time.time() - m_start
            log["time/eval_s"] = time.time() - eval_start

        epoch_time = time.time() - epoch_start
        log["time/epoch_s"] = epoch_time

        msg = (
            f"epoch {epoch + 1}/{start_epoch + args.epochs} "
            f"slice={slice_n} steps={steps_this_epoch} "
            f"pl={train_metrics.get('loss/policy', float('nan')):.3f} "
            f"vl={train_metrics.get('loss/value', float('nan')):.3f} "
            f"vmean={comp.get('curator/version_mean', 0):.1f} "
            f"({epoch_time:.1f}s: refill={refill_time:.1f}s train={train_time:.1f}s)"
        )
        wr_bits = [
            f"{key[3:]}={log[f'eval/{key}_winrate']:.0%}"
            for spec in fast_specs + slow_specs
            for key in [_baseline_log_key(spec)]
            if f"eval/{key}_winrate" in log
        ]
        if wr_bits:
            msg += " wr[" + " ".join(wr_bits) + "]"
        print(msg, flush=True)

        if run is not None:
            run.log(log)

        # --- save (weights + optimizer + EMA; NO buffer — archive is frozen) -
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch{epoch + 1:04d}.pt")
            ckpt_extra = None
            if ema_model is not None:
                ckpt_extra = {"ema_model_state_dict": ema_model.state_dict()}
            save_checkpoint(
                ckpt_path, model, optimizer,
                epoch=epoch + 1, total_games=total_games,
                wandb_run_id=wandb_run_id, extra=ckpt_extra,
            )

    # Always write latest.pt with the final model, regardless of --save-every,
    # so downstream consumers (the delta-e harness reads <checkpoint-dir>/latest.pt)
    # always find the fork's result — even short windows that never hit save-every.
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest_extra = (
        {"ema_model_state_dict": ema_model.state_dict()}
        if ema_model is not None else None
    )
    save_checkpoint(
        os.path.join(args.checkpoint_dir, "latest.pt"), model, optimizer,
        epoch=epoch + 1, total_games=total_games,
        wandb_run_id=wandb_run_id, extra=latest_extra,
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
