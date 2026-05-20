"""Self-play worker process: generate game records, write to a shared dir.

Pattern modeled on `~/code/mk5-main/forge/zeb/worker/run.py`:

- Reads model weights from a single file path (`--weights-path`).
- Generates batches of self-play games via `generate_games(...)`.
- Writes each batch atomically as `<worker_id>_<ts>_<short>.pt`
  (write to `.tmp`, then `os.replace` to `.pt` so the trainer never sees a
  partial file).
- Polls the weights-path's mtime each cycle; reloads if newer than last seen.

The trainer publishes new weights to the same path (also via atomic rename)
and ingests `.pt` files from the output dir, deleting them after consuming.

This module is the worker side; the trainer side is in `gomoku.train`
(activated by passing `--worker-input-dir` and `--worker-weights-path`).

Usage::

    python -m gomoku.selfplay_worker \\
        --weights-path /Users/me/code/gomoku/checkpoints_az_mini_9x9/worker_weights.pt \\
        --output-dir   /Users/me/code/gomoku/checkpoints_az_mini_9x9/_records \\
        --worker-id w0 \\
        --device mps \\
        --games-per-batch 16 --n-simulations 800 --wave-size 32 \\
        --dirichlet-alpha 0.13 --temperature-moves 10

Run several with different `--worker-id` and `--seed` values; the trainer
will pick up all their files.
"""
from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from gomoku.match import build_player, parse_spec
from gomoku.mcts import make_torch_evaluator
from gomoku.model import load_checkpoint
from gomoku.self_play import generate_games, generate_games_vs_baseline
from gomoku.util import pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights-path", type=str, required=True,
                   help="Checkpoint file the worker (re)loads weights from. The "
                        "trainer publishes here via atomic rename.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Where to write game-record batches. Trainer reads + deletes.")
    p.add_argument("--worker-id", type=str, default="w0")
    p.add_argument("--device", type=str, default=None)

    # Generation knobs (should match the trainer's MCTS config).
    p.add_argument("--games-per-batch", type=int, default=16)
    p.add_argument("--n-simulations", type=int, default=800)
    p.add_argument("--wave-size", type=int, default=32)
    p.add_argument("--c-puct", type=float, default=1.25,
                   help="c_puct_init in the AGZ log-schedule PUCT formula.")
    p.add_argument("--c-puct-base", type=float, default=19652.0,
                   help="c_puct_base in the AGZ log-schedule PUCT formula.")
    p.add_argument("--temperature-moves", type=int, default=10)
    p.add_argument("--temperature-final", type=float, default=0.1,
                   help="Sampling temperature after the warm-up plies. 0.1 matches "
                        "michaelnny/alpha_zero — sharp but not greedy, keeps the policy "
                        "training target a real soft distribution.")
    p.add_argument("--dirichlet-alpha", type=float, default=0.13)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--random-opening-moves", type=int, default=0)

    # Opponent options (default self-play, but support vs-baseline like the trainer).
    p.add_argument("--opponent", type=str, default="self",
                   help="self / random / heuristic / defensive / pacifist / "
                        "lookahead:depth=N. (No model:... here — workers always "
                        "use the file weights as the MCTS-side model.)")
    p.add_argument("--opponent-mix-random", type=float, default=0.0)
    p.add_argument("--model-first-frac", type=float, default=0.5)

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0,
                   help="Stop after this many batches (0 = run forever).")
    p.add_argument("--weights-poll-sec", type=float, default=1.0,
                   help="Sleep this long if the weights file isn't ready yet on startup.")
    p.add_argument("--gen-once-per-publish", action="store_true",
                   help="Generate exactly one batch per weight publish. After "
                        "writing a batch, sleep on the weights mtime until it "
                        "advances. Makes the dist setup semantically equivalent "
                        "to single-process: each cycle's games are produced by "
                        "exactly one model version. Trades worker idle-time for "
                        "clean per-version data stratification.")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile to the loaded model (eval-only). "
                        "Measured ~1.3-1.5x forward-pass speedup at batch sizes "
                        ">= 32 on Apple MPS for the small model. Applied after "
                        "each (re)load; fallback to uncompiled if compile raises. "
                        "Do NOT set this on the trainer's model — would interfere "
                        "with backward + optimizer.step.")
    return p.parse_args()


def _maybe_compile(model: torch.nn.Module, enabled: bool, worker_id: str) -> torch.nn.Module:
    """Optionally torch.compile the model. Falls back to uncompiled on failure."""
    if not enabled:
        return model
    try:
        # Default compile mode is the most compatible on MPS; reduce-overhead
        # relies on CUDA graphs which don't exist on MPS. The first forward
        # call after compile is slow (graph capture); subsequent calls are
        # the ones that benefit.
        compiled = torch.compile(model)
        print(f"[{worker_id}] torch.compile enabled (default mode)", flush=True)
        return compiled
    except Exception as e:
        print(f"[{worker_id}] torch.compile failed ({e}); using uncompiled model", flush=True)
        return model


def _load_model(weights_path: str, device: torch.device) -> tuple[torch.nn.Module, float]:
    """Return (model, weights_mtime)."""
    while not os.path.exists(weights_path):
        time.sleep(1.0)
    model, _ = load_checkpoint(weights_path, device=device)
    model.eval()
    return model, os.path.getmtime(weights_path)


def _atomic_save(out_dir: Path, worker_id: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)  # ms so two batches in the same second don't collide
    short = uuid.uuid4().hex[:8]
    final = out_dir / f"{worker_id}_{ts}_{short}.pt"
    tmp = out_dir / f"{worker_id}_{ts}_{short}.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)
    return final


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"[{args.worker_id}] device={device}", flush=True)
    print(f"[{args.worker_id}] weights={args.weights_path} output={args.output_dir}", flush=True)

    model, weights_mtime = _load_model(args.weights_path, device)
    model = _maybe_compile(model, args.compile, args.worker_id)
    evaluator = make_torch_evaluator(model, device)
    print(f"[{args.worker_id}] initial weights mtime={weights_mtime:.0f}", flush=True)

    # Opponent picker (None for pure self-play, like train.py does).
    opp_picker = None
    if args.opponent != "self":
        opp_spec = parse_spec(args.opponent)
        if opp_spec.kind == "model":
            raise SystemExit("workers don't support --opponent model:...")
        base_picker = build_player(opp_spec)
        if args.opponent_mix_random > 0.0:
            from gomoku.baselines import random_player as _rand
            p_random = float(args.opponent_mix_random)

            def _mixed(state, rng, *, _base=base_picker, _p=p_random):
                return _rand(state, rng) if rng.random() < _p else _base(state, rng)
            opp_picker = _mixed
        else:
            opp_picker = base_picker

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    batch_n = 0
    # In gen-once-per-publish mode we track the mtime we LAST generated against,
    # so we only emit one batch per published weight version.
    last_gen_mtime = 0.0

    while True:
        # 1) reload weights if newer (cheap mtime check)
        try:
            cur_mtime = os.path.getmtime(args.weights_path)
        except OSError:
            cur_mtime = weights_mtime  # file might be mid-rename
        if cur_mtime > weights_mtime:
            try:
                model, _ = load_checkpoint(args.weights_path, device=device)
                model.eval()
                model = _maybe_compile(model, args.compile, args.worker_id)
                evaluator = make_torch_evaluator(model, device)
                weights_mtime = cur_mtime
                print(f"[{args.worker_id}] reloaded weights mtime={weights_mtime:.0f}", flush=True)
            except Exception as e:
                # Trainer might be mid-write; try again next loop.
                print(f"[{args.worker_id}] reload failed ({e}); retrying", flush=True)

        # In gen-once-per-publish mode, only generate when the weights mtime has
        # advanced since our last batch. Otherwise sleep and re-poll.
        if args.gen_once_per_publish and cur_mtime <= last_gen_mtime:
            time.sleep(args.weights_poll_sec)
            continue

        # 2) generate a batch of games
        t0 = time.perf_counter()
        if opp_picker is None:
            records = generate_games(
                args.games_per_batch, evaluator,
                n_simulations=args.n_simulations,
                c_puct=args.c_puct,
                c_puct_base=args.c_puct_base,
                temperature_moves=args.temperature_moves,
                temperature_final=args.temperature_final,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps,
                rng=rng,
                wave_size=args.wave_size,
                random_opening_moves=args.random_opening_moves,
            )
        else:
            records = generate_games_vs_baseline(
                args.games_per_batch, evaluator, opp_picker,
                n_simulations=args.n_simulations,
                c_puct=args.c_puct,
                c_puct_base=args.c_puct_base,
                temperature_moves=args.temperature_moves,
                temperature_final=args.temperature_final,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps,
                rng=rng,
                wave_size=args.wave_size,
                model_first_frac=args.model_first_frac,
                random_opening_moves=args.random_opening_moves,
            )
        dt = time.perf_counter() - t0

        # 3) atomic write
        n_games = len(records)
        n_examples = sum(len(r.examples) for r in records)
        path = _atomic_save(out_dir, args.worker_id, {
            "records": records,
            "worker_id": args.worker_id,
            "weights_mtime": weights_mtime,
            "n_games": n_games,
            "n_examples": n_examples,
            "gen_s": dt,
        })
        batch_n += 1
        last_gen_mtime = weights_mtime
        print(
            f"[{args.worker_id}] batch {batch_n}: {n_games}g {n_examples}ex "
            f"{dt:.1f}s ({dt/n_games*1000:.0f} ms/game) -> {path.name}",
            flush=True,
        )

        if args.max_batches > 0 and batch_n >= args.max_batches:
            print(f"[{args.worker_id}] hit max-batches={args.max_batches}, exiting", flush=True)
            return


if __name__ == "__main__":
    main()
