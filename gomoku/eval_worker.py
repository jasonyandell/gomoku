"""Eval worker: polls a checkpoint file and logs strength baselines to wandb.

Decouples eval from the training loop so generation+training stays at full
speed. The trainer publishes `worker_weights.pt` after every save cycle (with
the wandb run id embedded); this worker mtime-polls that file, runs n=N games
vs each baseline, and posts results to the same wandb run.

Defaults to CPU so it doesn't fight the trainer for MPS. With CPU + n=20 games
at 100 sims, a full pass (random + heuristic + lookahead:depth=2) typically
finishes in well under a minute, depending on how aggressive the model is.

Usage::

    python -m gomoku.eval_worker \\
        --checkpoint-path checkpoints_az_mini_9x9_fresh/worker_weights.pt \\
        --baselines random,heuristic,lookahead:depth=2 \\
        --n-games 20 --sims 100 --device cpu

If --wandb-run-id is not given, it is read from the checkpoint's payload (the
trainer embeds its run id in `worker_weights.pt`).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from gomoku.eval import mcts_picker, play_match_pickers
from gomoku.match import build_player, parse_spec
from gomoku.mcts import make_torch_evaluator
from gomoku.model import load_checkpoint
from gomoku.util import load_wandb_key_from_keychain, pick_device


def _baseline_log_key(spec) -> str:
    if spec.kind == "lookahead":
        depth = spec.kwargs.get("depth", "")
        return f"vs_lookahead{depth}"
    return f"vs_{spec.kind}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", type=str, required=True,
                   help="Path to mtime-poll. The trainer publishes worker_weights.pt "
                        "here after every save cycle (state_dict + wandb_run_id).")
    p.add_argument("--baselines", type=str, default="random,heuristic,lookahead:depth=2",
                   help="Comma-separated player specs. No model:... allowed.")
    p.add_argument("--n-games", type=int, default=20,
                   help="Games per matchup. n=20 gives ~12pp 95%% CI vs ±50pp at n=4.")
    p.add_argument("--sims", type=int, default=100,
                   help="MCTS sims for the model side of each eval game.")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--random-opening-moves", type=int, default=0,
                   help="Random plies before MCTS, to diversify deterministic games. "
                        "Each baseline gets the same RNG seed per-cell so results stay "
                        "comparable across polls.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Default cpu so the eval doesn't fight training for MPS.")
    p.add_argument("--wandb-project", type=str, default="gomoku")
    p.add_argument("--wandb-run-id", type=str, default=None,
                   help="Resume this run id. If omitted, read from the checkpoint payload.")
    p.add_argument("--poll-sec", type=float, default=2.0,
                   help="Sleep this long between mtime polls.")
    p.add_argument("--max-cycles", type=int, default=0,
                   help="Stop after this many evals (0 = run forever).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _parse_specs(s: str) -> list:
    out = []
    for raw in s.split(","):
        raw = raw.strip()
        if not raw:
            continue
        spec = parse_spec(raw)
        if spec.kind == "model":
            raise SystemExit(f"--baselines must not include model specs: {raw!r}")
        out.append(spec)
    return out


def _wait_for_checkpoint(path: str, poll_sec: float) -> None:
    while not os.path.exists(path):
        print(f"[eval] waiting for checkpoint at {path}...", flush=True)
        time.sleep(max(poll_sec, 1.0))


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    specs = _parse_specs(args.baselines)
    if not specs:
        raise SystemExit("no baselines configured")
    baseline_pickers = [(s, build_player(s)) for s in specs]
    print(f"[eval] device={device} specs={[s.label() for s in specs]} "
          f"n_games={args.n_games} sims={args.sims}", flush=True)

    # Bootstrap: load once to discover the wandb run id, then init wandb.
    _wait_for_checkpoint(args.checkpoint_path, args.poll_sec)
    model, payload = load_checkpoint(args.checkpoint_path, device=device)
    model.eval()
    last_mtime = os.path.getmtime(args.checkpoint_path)

    run_id = args.wandb_run_id or payload.get("wandb_run_id")
    if not run_id:
        raise SystemExit(
            "no wandb_run_id available — pass --wandb-run-id or have the trainer "
            "publish weights with its run id embedded"
        )

    key = load_wandb_key_from_keychain()
    if not key:
        print("[eval] warning: no WANDB_API_KEY available", flush=True)
    import wandb
    run = wandb.init(
        project=args.wandb_project,
        id=run_id,
        resume="allow",
    )
    print(f"[eval] attached to wandb run id={run_id}", flush=True)

    cycle_n = 0
    while True:
        # Run a pass on the currently-loaded model.
        epoch_tag = int(payload.get("epoch", 0))
        evaluator = make_torch_evaluator(model, device)
        model_picker = mcts_picker(evaluator, n_simulations=args.sims, c_puct=args.c_puct)

        log: dict = {"eval_worker/epoch_evaluated": epoch_tag}
        t_pass_0 = time.perf_counter()
        for spec_idx, (spec, baseline_picker) in enumerate(baseline_pickers):
            t0 = time.perf_counter()
            res = play_match_pickers(
                model_picker, baseline_picker,
                n_games=args.n_games,
                seed=args.seed + epoch_tag * 1000 + spec_idx,
            )
            dt = time.perf_counter() - t0
            key_ = _baseline_log_key(spec)
            log[f"eval/{key_}_winrate"] = res.win_rate
            log[f"eval/{key_}_wins"] = res.wins
            log[f"eval/{key_}_losses"] = res.losses
            log[f"eval/{key_}_draws"] = res.draws
            log[f"time/eval_{key_}_s"] = dt
            print(
                f"[eval] e={epoch_tag} {spec.label():<20} "
                f"{res.wins}W-{res.losses}L-{res.draws}D ({res.win_rate:.0%}) "
                f"in {dt:.1f}s",
                flush=True,
            )
        log["time/eval_pass_s"] = time.perf_counter() - t_pass_0
        run.log(log)
        cycle_n += 1

        if args.max_cycles > 0 and cycle_n >= args.max_cycles:
            print(f"[eval] hit max-cycles={args.max_cycles}, exiting", flush=True)
            return

        # Wait for a newer checkpoint.
        while True:
            try:
                cur = os.path.getmtime(args.checkpoint_path)
            except OSError:
                cur = last_mtime
            if cur > last_mtime:
                try:
                    model, payload = load_checkpoint(args.checkpoint_path, device=device)
                    model.eval()
                    last_mtime = cur
                    break
                except Exception as e:
                    print(f"[eval] reload failed ({e}); retrying", flush=True)
            time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
