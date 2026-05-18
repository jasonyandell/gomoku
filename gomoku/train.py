"""AlphaZero-style training loop: self-play -> replay buffer -> SGD -> repeat."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gomoku.eval import play_vs_random
from gomoku.mcts import make_torch_evaluator
from gomoku.model import build_model, load_checkpoint, n_params, save_checkpoint
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import generate_games
from gomoku.util import load_wandb_key_from_keychain, pick_device


def policy_loss(logits: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft targets. pi may have zeros on illegal moves; those contribute 0."""
    logp = F.log_softmax(logits, dim=-1)
    return -(pi * logp).sum(dim=-1).mean()


def value_loss(v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v, z)


def train_step(
    model,
    optimizer,
    planes: torch.Tensor,
    pi: torch.Tensor,
    z: torch.Tensor,
    *,
    value_weight: float = 1.0,
    l2_weight: float = 1e-4,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, v = model(planes)
    pl = policy_loss(logits, pi)
    vl = value_loss(v, z)
    loss = pl + value_weight * vl
    if l2_weight > 0:
        l2 = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
        loss = loss + l2_weight * l2
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        # Accuracy: fraction where argmax(logits over legal entries of pi) matches argmax(pi).
        pred = logits.argmax(dim=-1)
        target = pi.argmax(dim=-1)
        acc = (pred == target).float().mean()
    return {
        "loss/total": float(loss.detach()),
        "loss/policy": float(pl.detach()),
        "loss/value": float(vl.detach()),
        "train/policy_acc": float(acc),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--size", type=str, default="small", help="tiny / small / medium / large")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--games-per-epoch", type=int, default=64)
    p.add_argument("--n-simulations", type=int, default=100)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temperature-moves", type=int, default=8)
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--replay-buffer-size", type=int, default=50000)
    p.add_argument("--training-steps", type=int, default=400, help="SGD steps per epoch")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-sims", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default=None, help="torch device override (e.g. cpu, mps)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.set_defaults(wandb=False)
    p.add_argument("--wandb-project", type=str, default="gomoku")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device = {device}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Build / load model
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
        model = build_model(args.size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print(f"fresh {args.size} model: {n_params(model):,} params")

    buffer = ReplayBuffer(args.replay_buffer_size, device=device)
    if args.resume:
        # Optional: restore buffer if it was saved.
        payload_buf = payload.get("replay_buffer")
        if payload_buf is not None:
            buffer.load_state_dict(payload_buf)
            print(f"replay buffer restored: {buffer.size} examples")

    # wandb setup
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

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        evaluator = make_torch_evaluator(model, device)

        # --- self-play ---
        gen_start = time.time()
        records = generate_games(
            args.games_per_epoch,
            evaluator,
            n_simulations=args.n_simulations,
            c_puct=args.c_puct,
            temperature_moves=args.temperature_moves,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
            rng=rng,
        )
        gen_time = time.time() - gen_start

        n_examples_new = sum(len(r.examples) for r in records)
        for r in records:
            buffer.add(r.examples)

        plies_mean = float(np.mean([r.plies for r in records])) if records else 0.0
        outcomes = [r.outcome for r in records]
        wins_black = sum(1 for o in outcomes if o > 0)
        wins_white = sum(1 for o in outcomes if o < 0)
        draws = sum(1 for o in outcomes if o == 0)

        # --- training ---
        train_metrics_acc: dict[str, list[float]] = {}
        train_start = time.time()
        if buffer.size >= args.batch_size:
            for _ in range(args.training_steps):
                planes, pi, z = buffer.sample(args.batch_size)
                m = train_step(model, optimizer, planes, pi, z,
                               value_weight=args.value_weight, l2_weight=args.l2)
                for k, v in m.items():
                    train_metrics_acc.setdefault(k, []).append(v)
        train_time = time.time() - train_start
        train_metrics = {k: float(np.mean(v)) for k, v in train_metrics_acc.items()}

        total_games += args.games_per_epoch

        log = {
            "epoch": epoch + 1,
            "total_games": total_games,
            "buffer_size": buffer.size,
            "selfplay/new_examples": n_examples_new,
            "selfplay/plies_mean": plies_mean,
            "selfplay/black_wins": wins_black,
            "selfplay/white_wins": wins_white,
            "selfplay/draws": draws,
            "time/gen_s": gen_time,
            "time/train_s": train_time,
            **train_metrics,
        }

        # --- eval ---
        if (epoch + 1) % args.eval_every == 0:
            eval_start = time.time()
            eval_evaluator = make_torch_evaluator(model, device)
            res = play_vs_random(
                eval_evaluator,
                n_games=args.eval_games,
                n_simulations=args.eval_sims,
                c_puct=args.c_puct,
                seed=args.seed + epoch,
            )
            log["eval/vs_random_winrate"] = res.win_rate
            log["eval/vs_random_wins"] = res.wins
            log["eval/vs_random_losses"] = res.losses
            log["eval/vs_random_draws"] = res.draws
            log["time/eval_s"] = time.time() - eval_start

        epoch_time = time.time() - epoch_start
        log["time/epoch_s"] = epoch_time

        # Print
        msg = (
            f"epoch {epoch + 1}/{start_epoch + args.epochs} "
            f"games={total_games} buf={buffer.size} "
            f"pl={train_metrics.get('loss/policy', float('nan')):.3f} "
            f"vl={train_metrics.get('loss/value', float('nan')):.3f} "
            f"plies={plies_mean:.1f} "
            f"({epoch_time:.1f}s: gen={gen_time:.1f}s train={train_time:.1f}s)"
        )
        if "eval/vs_random_winrate" in log:
            msg += f" wr={log['eval/vs_random_winrate']:.1%}"
        print(msg, flush=True)

        if run is not None:
            run.log(log)

        # --- save ---
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch{epoch + 1:04d}.pt")
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                epoch=epoch + 1,
                total_games=total_games,
                wandb_run_id=wandb_run_id,
                extra={"replay_buffer": buffer.state_dict()},
            )
            latest = os.path.join(args.checkpoint_dir, "latest.pt")
            # Symlink overwrite
            try:
                if os.path.islink(latest) or os.path.exists(latest):
                    os.remove(latest)
                os.symlink(os.path.basename(ckpt_path), latest)
            except OSError:
                pass


if __name__ == "__main__":
    main()
