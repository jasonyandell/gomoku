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
    p.add_argument("--wave-mode", action="store_true",
                   help="Wave-lockstep greedy-fill mode. Write one file per game "
                        "under output-dir/v{version}/worker{worker_id}/game{seq}.pt, "
                        "generate --games-per-batch games for each loaded version, "
                        "then keep filling extra games on that version until newer "
                        "published weights are available. Supersedes "
                        "--gen-once-per-publish when set.")
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


def _load_model(weights_path: str, device: torch.device) -> tuple[torch.nn.Module, float, int]:
    """Return (model, weights_mtime, version)."""
    while not os.path.exists(weights_path):
        time.sleep(1.0)
    model, payload = load_checkpoint(weights_path, device=device)
    model.eval()
    return model, os.path.getmtime(weights_path), int(payload.get("epoch", 0))


def _atomic_save(out_dir: Path, worker_id: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)  # ms so two batches in the same second don't collide
    short = uuid.uuid4().hex[:8]
    final = out_dir / f"{worker_id}_{ts}_{short}.pt"
    tmp = out_dir / f"{worker_id}_{ts}_{short}.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)
    return final


def _worker_dir_name(worker_id: str) -> str:
    return f"worker{worker_id}"


def _next_wave_seq(out_dir: Path, version: int, worker_id: str) -> int:
    worker_dir = out_dir / f"v{version}" / _worker_dir_name(worker_id)
    max_seq = 0
    for path in worker_dir.glob("game*.pt"):
        stem = path.stem
        seq_s = stem.removeprefix("game")
        if seq_s.isdigit():
            max_seq = max(max_seq, int(seq_s))
    return max_seq + 1


def _atomic_save_wave_game(
    out_dir: Path,
    worker_id: str,
    version: int,
    seq: int,
    payload: dict,
) -> Path | None:
    """Persist one greedy-fill game. Returns None if the trainer cleaned up
    `v{version}/` between mkdir and write — the game is dropped and the outer
    loop should reload fresh weights on its next iteration."""
    worker_dir = out_dir / f"v{version}" / _worker_dir_name(worker_id)
    try:
        worker_dir.mkdir(parents=True, exist_ok=True)
        final = worker_dir / f"game{seq:06d}.pt"
        tmp = worker_dir / f"game{seq:06d}.pt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, final)
    except (FileNotFoundError, OSError, RuntimeError) as e:
        print(
            f"[{worker_id}] drop wave game v{version} seq{seq}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return None
    return final


def _generate_records(args: argparse.Namespace, evaluator, opp_picker, rng, n_games: int):
    if opp_picker is None:
        return generate_games(
            n_games, evaluator,
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
    return generate_games_vs_baseline(
        n_games, evaluator, opp_picker,
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


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"[{args.worker_id}] device={device}", flush=True)
    print(f"[{args.worker_id}] weights={args.weights_path} output={args.output_dir}", flush=True)

    model, weights_mtime, model_version = _load_model(args.weights_path, device)
    model = _maybe_compile(model, args.compile, args.worker_id)
    evaluator = make_torch_evaluator(model, device)
    print(
        f"[{args.worker_id}] initial weights version={model_version} "
        f"mtime={weights_mtime:.0f}",
        flush=True,
    )

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

    if args.wave_mode:
        games_on_version = 0
        next_seq = _next_wave_seq(out_dir, model_version, args.worker_id)
        print(
            f"[{args.worker_id}] wave-mode G={args.games_per_batch} "
            f"version={model_version} next_seq={next_seq}",
            flush=True,
        )

        while True:
            # Wave mode reloads only at generation boundaries. Before the
            # worker has produced its G-game tile, keep filling the current
            # version even if a newer file appears unexpectedly.
            try:
                cur_mtime = os.path.getmtime(args.weights_path)
            except OSError:
                cur_mtime = weights_mtime
            if games_on_version >= args.games_per_batch and cur_mtime > weights_mtime:
                try:
                    model, payload = load_checkpoint(args.weights_path, device=device)
                    model.eval()
                    model = _maybe_compile(model, args.compile, args.worker_id)
                    evaluator = make_torch_evaluator(model, device)
                    weights_mtime = cur_mtime
                    model_version = int(payload.get("epoch", model_version + 1))
                    games_on_version = 0
                    next_seq = _next_wave_seq(out_dir, model_version, args.worker_id)
                    print(
                        f"[{args.worker_id}] wave reload version={model_version} "
                        f"mtime={weights_mtime:.0f} next_seq={next_seq}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[{args.worker_id}] wave reload failed ({e}); retrying", flush=True)
                    time.sleep(args.weights_poll_sec)
                    continue

            target_games = max(args.games_per_batch - games_on_version, 1)
            t0 = time.perf_counter()
            records = _generate_records(args, evaluator, opp_picker, rng, target_games)
            dt = time.perf_counter() - t0
            batch_n += 1

            paths: list[Path] = []
            saved_examples = 0
            for record in records:
                seq = next_seq
                next_seq += 1
                payload = {
                    "records": [record],
                    "worker_id": args.worker_id,
                    "weights_mtime": weights_mtime,
                    "model_version": model_version,
                    "version": model_version,
                    "game_seq": seq,
                    "n_games": 1,
                    "n_examples": len(record.examples),
                    "gen_s": dt / max(len(records), 1),
                    "batch_gen_s": dt,
                }
                saved = _atomic_save_wave_game(
                    out_dir, args.worker_id, model_version, seq, payload
                )
                if saved is None:
                    # Trainer ingested v{model_version} and cleaned the dir
                    # between mkdir and write — drop the game. Outer loop
                    # will pick up new weights on the next iteration.
                    break
                paths.append(saved)
                games_on_version += 1
                saved_examples += len(record.examples)

            print(
                f"[{args.worker_id}] wave batch {batch_n}: v{model_version} "
                f"{len(paths)}g {saved_examples}ex {dt:.1f}s "
                f"seq {paths[0].stem if paths else '-'}..{paths[-1].stem if paths else '-'} "
                f"tile_count={games_on_version}",
                flush=True,
            )

            if args.max_batches > 0 and batch_n >= args.max_batches:
                print(f"[{args.worker_id}] hit max-batches={args.max_batches}, exiting", flush=True)
                return

    while True:
        # 1) reload weights if newer (cheap mtime check)
        try:
            cur_mtime = os.path.getmtime(args.weights_path)
        except OSError:
            cur_mtime = weights_mtime  # file might be mid-rename
        if cur_mtime > weights_mtime:
            try:
                model, payload = load_checkpoint(args.weights_path, device=device)
                model.eval()
                model = _maybe_compile(model, args.compile, args.worker_id)
                evaluator = make_torch_evaluator(model, device)
                weights_mtime = cur_mtime
                model_version = int(payload.get("epoch", model_version + 1))
                print(
                    f"[{args.worker_id}] reloaded weights version={model_version} "
                    f"mtime={weights_mtime:.0f}",
                    flush=True,
                )
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
        records = _generate_records(args, evaluator, opp_picker, rng, args.games_per_batch)
        dt = time.perf_counter() - t0

        # 3) atomic write
        n_games = len(records)
        n_examples = sum(len(r.examples) for r in records)
        path = _atomic_save(out_dir, args.worker_id, {
            "records": records,
            "worker_id": args.worker_id,
            "weights_mtime": weights_mtime,
            "model_version": model_version,
            "version": model_version,
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
