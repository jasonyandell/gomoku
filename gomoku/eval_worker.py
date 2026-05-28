"""Eval worker: polls a checkpoint file and writes baseline results to JSONL.

Decouples eval from the training loop so generation+training stays at full
speed. The trainer publishes `worker_weights.pt` after every save cycle; this
worker mtime-polls that file, runs n=N games vs each baseline, and appends
each pass to `<output-dir>/eval_results.jsonl`. The trainer tails that file
and forwards the metrics into its own wandb run, so wandb stays single-writer.

Defaults to CPU so it doesn't fight the trainer for MPS. With CPU + n=20 games
at 100 sims, a full pass (random + heuristic + lookahead:depth=2) typically
finishes in well under a minute, depending on how aggressive the model is.

Usage::

    python -m gomoku.eval_worker \\
        --checkpoint-path checkpoints/worker_weights.pt \\
        --output-dir   checkpoints/ \\
        --baselines    random,heuristic,lookahead:depth=2 \\
        --n-games 20 --sims 100 --device cpu

Why a JSONL handoff: two processes calling wandb.init(id=..., resume="allow")
for the same run id silently drops the second writer's metrics. The trainer
is the sole wandb writer; this worker just produces eval rows.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from gomoku.eval import mcts_picker, play_match_parallel, play_match_pickers
from gomoku.match import build_player, parse_spec
from gomoku.mcts import make_torch_evaluator
from gomoku.model import fuse_model_for_inference, load_checkpoint
from gomoku.rating import (
    ANCHOR_ELOS,
    baseline_spec_to_anchor_key,
    implied_elo,
    score_from_match,
)
from gomoku.util import pick_device


def _baseline_log_key(spec) -> str:
    if spec.kind == "lookahead":
        depth = spec.kwargs.get("depth", "")
        return f"vs_lookahead{depth}"
    return f"vs_{spec.kind}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", type=str, required=True,
                   help="Path to mtime-poll. The trainer publishes worker_weights.pt "
                        "here after every save cycle (state_dict + run metadata).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to append eval_results.jsonl. Defaults to the dir "
                        "containing --checkpoint-path so the trainer finds it.")
    p.add_argument("--baselines", type=str, default="random,heuristic,lookahead:depth=2",
                   help="Comma-separated player specs. No model:... allowed.")
    p.add_argument("--n-games", type=int, default=20,
                   help="Games per matchup. n=20 gives ~12pp 95%% CI vs ±50pp at n=4.")
    p.add_argument("--sims", type=int, default=100,
                   help="MCTS sims for the model side of each eval game.")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--device", type=str, default="cpu",
                   help="Default cpu so the eval doesn't fight training for MPS.")
    p.add_argument("--poll-sec", type=float, default=2.0,
                   help="Sleep this long between mtime polls.")
    p.add_argument("--max-cycles", type=int, default=0,
                   help="Stop after this many evals (0 = run forever).")
    p.add_argument("--n-workers", type=int, default=1,
                   help="If > 1, play eval games in parallel via multiprocessing. "
                        "Each worker process loads the model from --checkpoint-path "
                        "and plays games independently. Aggregate W/L/D counts are "
                        "equivalent to sequential; per-game RNG is independent per "
                        "worker. Speed scales ~linearly up to # of CPU cores. With "
                        "spawn-start pool overhead ~0.5-1s per cycle, worth it for "
                        "n_games >= ~8 or when adding higher lookahead depths.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-vcf-nodes", type=int, default=0,
                   help="Eval-time root VCF overlay node budget. Default 0 = OFF "
                        "(byte-identical to the pre-lever path: no solver call, "
                        "MCTS picks unchanged). When >0, before MCTS picks a move "
                        "the eval picker runs a bounded solve_vcf from the root; "
                        "if a forced four-in-a-row win is proven, it plays the "
                        "solver's move; else the MCTS choice runs as today. "
                        "EVAL-ONLY: self-play / generation / training are NOT "
                        "affected. Typical: 200-3200 closes the lookahead4-black "
                        "search-depth gap. Hitting the cap returns has_forced_win="
                        "False — the overlay never blocks indefinitely.")
    p.add_argument("--eval-vcf-depth", type=int, default=0,
                   help="Eval-time root VCF overlay depth cap (attacker plies). "
                        "0 = use vcf.DEFAULT_MAX_DEPTH (16). Only matters with "
                        "--eval-vcf-nodes > 0.")
    p.add_argument("--fpu-reduction-c", type=float, default=0.0,
                   help="Eval-time KataGo First-Play Urgency reduction. Default "
                        "0.0 = OFF (byte-identical to the pre-lever path: "
                        "unvisited children take Q=0 in PUCT). When >0, "
                        "unvisited children inherit "
                        "parent_V - c * sqrt(sum_visited_priors), making them "
                        "look pessimistic relative to the visited PV so MCTS "
                        "deepens the PV instead of scattering on siblings. "
                        "KataGo uses c≈0.45 (subtree) / 0.20 (root); LCZero "
                        "0.33. EVAL-ONLY: self-play / generation / training "
                        "are NOT affected.")
    p.add_argument("--reuse-tree", action="store_true", default=False,
                   help="Reuse the MCTS tree across plies at EVAL time (derby-jmi). "
                        "Default OFF = byte-identical fresh-tree-per-call legacy. "
                        "When set, the eval picker holds one MCTSGame across the "
                        "match and calls MCTSGame.advance_root after each ply pair, "
                        "inheriting the previously-explored subtree's visits / W / "
                        "priors. Effectively multiplies the sim budget by the tree-"
                        "reuse fraction at zero extra compute (standard AlphaZero / "
                        "KataGo / LCZero). EVAL-ONLY: self-play / generation / "
                        "training are NOT affected.")
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
    # Keep the raw spec string alongside the parsed spec — parallel mode needs
    # to ship the string to worker processes for picker reconstruction.
    raw_specs = [s.strip() for s in args.baselines.split(",") if s.strip()]
    specs = _parse_specs(args.baselines)
    if not specs:
        raise SystemExit("no baselines configured")
    baseline_pickers = [(raw, s, build_player(s)) for raw, s in zip(raw_specs, specs)]
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "eval_results.jsonl"
    print(f"[eval] device={device} specs={[s.label() for s in specs]} "
          f"n_games={args.n_games} sims={args.sims} jsonl={jsonl_path}", flush=True)

    _wait_for_checkpoint(args.checkpoint_path, args.poll_sec)
    model, payload = load_checkpoint(args.checkpoint_path, device=device)
    model = fuse_model_for_inference(model)
    last_mtime = os.path.getmtime(args.checkpoint_path)

    cycle_n = 0
    while True:
        epoch_tag = int(payload.get("epoch", 0))
        evaluator = make_torch_evaluator(model, device)
        model_picker = mcts_picker(
            evaluator, n_simulations=args.sims, c_puct=args.c_puct,
            eval_vcf_nodes=args.eval_vcf_nodes,
            eval_vcf_depth=args.eval_vcf_depth,
            fpu_reduction_c=args.fpu_reduction_c,
            reuse_tree=args.reuse_tree,
        )

        log: dict = {"eval_worker/epoch_evaluated": epoch_tag}
        t_pass_0 = time.perf_counter()
        # Per-cycle Elo evidence: list of (anchor_rating, score, n_games) for
        # the baselines we have anchor Elos for. Fed to implied_elo at the end
        # of the cycle.
        elo_matches: list[tuple[float, float, int]] = []
        for spec_idx, (raw_spec, spec, baseline_picker) in enumerate(baseline_pickers):
            t0 = time.perf_counter()
            match_seed = args.seed + epoch_tag * 1000 + spec_idx
            if args.n_workers > 1:
                # Parallel: spawn pool, each worker reloads model from disk
                # and reconstructs the opponent picker from `raw_spec`.
                res = play_match_parallel(
                    checkpoint_path=args.checkpoint_path,
                    opp_spec=raw_spec,
                    n_games=args.n_games,
                    seed=match_seed,
                    n_workers=args.n_workers,
                    sims=args.sims,
                    c_puct=args.c_puct,
                    device=str(device),
                    eval_vcf_nodes=args.eval_vcf_nodes,
                    eval_vcf_depth=args.eval_vcf_depth,
                    fpu_reduction_c=args.fpu_reduction_c,
                    reuse_tree=args.reuse_tree,
                )
            else:
                res = play_match_pickers(
                    model_picker, baseline_picker,
                    n_games=args.n_games,
                    seed=match_seed,
                )
            dt = time.perf_counter() - t0
            key_ = _baseline_log_key(spec)
            log[f"eval/{key_}_winrate"] = res.win_rate
            log[f"eval/{key_}_wins"] = res.wins
            log[f"eval/{key_}_losses"] = res.losses
            log[f"eval/{key_}_draws"] = res.draws
            # Per-color split (model's perspective): black_* = games the model
            # played black, white_* = games it played white. Aggregate fields
            # above are unchanged; these are additive so loss-tail analysis can
            # see whether losses cluster on white (second player).
            log[f"eval/{key_}_black_w"] = res.black_w
            log[f"eval/{key_}_black_l"] = res.black_l
            log[f"eval/{key_}_black_d"] = res.black_d
            log[f"eval/{key_}_white_w"] = res.white_w
            log[f"eval/{key_}_white_l"] = res.white_l
            log[f"eval/{key_}_white_d"] = res.white_d
            log[f"time/eval_{key_}_s"] = dt
            anchor_key = baseline_spec_to_anchor_key(spec)
            anchor_elo = ANCHOR_ELOS.get(anchor_key) if anchor_key else None
            anchor_str = f" anchor={anchor_elo:.0f}" if anchor_elo is not None else ""
            print(
                f"[eval] e={epoch_tag} {spec.label():<20} "
                f"{res.wins}W-{res.losses}L-{res.draws}D ({res.win_rate:.0%}) "
                f"in {dt:.1f}s{anchor_str}",
                flush=True,
            )
            if anchor_elo is not None:
                score, n_games = score_from_match(res.wins, res.losses, res.draws)
                elo_matches.append((anchor_elo, score, n_games))
                log[f"eval/{key_}_anchor_elo"] = anchor_elo
        log["time/eval_pass_s"] = time.perf_counter() - t_pass_0

        # Maximum-likelihood implied Elo from this cycle's evidence. Stateless
        # per cycle — plotted over time it gives a single-number strength
        # trajectory that composes across anchors instead of forcing you to
        # eyeball three winrates separately.
        if elo_matches:
            model_elo = implied_elo(elo_matches)
            log["eval/model_elo"] = model_elo
            print(
                f"[eval] e={epoch_tag} model_elo={model_elo:.0f} "
                f"(from {len(elo_matches)} anchors)",
                flush=True,
            )
        # Atomic-append to JSONL so a partial line never leaves the file.
        line = json.dumps({"ts": time.time(), **log}) + "\n"
        with open(jsonl_path, "a") as f:
            f.write(line)
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
                    model = fuse_model_for_inference(model)
                    last_mtime = cur
                    break
                except Exception as e:
                    print(f"[eval] reload failed ({e}); retrying", flush=True)
            time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
