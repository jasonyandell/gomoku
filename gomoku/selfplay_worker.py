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

WL2 levers implemented here (see `wiki/topics/wl2-scale-emulation-design.md`):

- Lever #2 (past-checkpoint opponent mix): at each wave start a worker rolls
  dice; with `--opponent-mix-recent` probability it loads a random checkpoint
  from the last `--opponent-mix-recent-window` snapshots, with
  `--opponent-mix-history` it loads from anywhere in the run, otherwise it
  uses the current published `worker_weights.pt`. The past-checkpoint games
  are written to the *current* model-version tile so the trainer ingests
  them normally.
- Lever #3 (worker poll jitter): each worker samples its own poll interval
  once at startup from `Uniform(--weights-poll-min-sec, --weights-poll-max-sec)`
  so the 8-worker pool picks up new weights with natural async-publish skew
  instead of all reloading on the same tick.

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
import json
import platform
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from gomoku.match import build_player, parse_spec
from gomoku.mcts import make_torch_evaluator
from gomoku.model import fuse_model_for_inference, load_checkpoint
from gomoku.self_play import generate_games, generate_games_vs_baseline
from gomoku.util import pick_device

import tempfile


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

    # Playout-Cap Randomization (KataGo, Wu 2019). Opt-in; defaults are inert and
    # preserve the byte-identical production self-play path. When frac < 1.0,
    # each move is full-search+recorded with probability `frac`, else fast (at
    # `--playout-cap-fast-sims`) and NOT recorded as a training target.
    p.add_argument("--playout-cap-frac", type=float, default=1.0,
                   help="Playout-Cap Randomization full-search fraction. 1.0 "
                        "(default) = every move is full-search and recorded "
                        "(current behavior). When <1.0, the probability a given "
                        "move runs the full --n-simulations budget AND is "
                        "recorded as a training target; the rest run "
                        "--playout-cap-fast-sims and are not recorded.")
    p.add_argument("--playout-cap-fast-sims", type=int, default=0,
                   help="Reduced sim budget for non-recorded (fast) PCR moves. "
                        "0 (default) = same as --n-simulations, which makes the "
                        "lever inert. Only used when --playout-cap-frac < 1.0.")
    # Gumbel AlphaZero root selection + Sequential Halving (Danihelka et al. 2022).
    # Derby v3 lever. Default OFF = byte-identical current PUCT+Dirichlet behavior.
    # When ON: root uses Gumbel-top-k candidate sampling + Sequential Halving over
    # the sim budget; the policy training target is the completed-policy (softmax of
    # logits + sigma(q_hat)), NOT visit counts. Internal nodes keep PUCT. Runs on
    # the pure-Python MCTS tree (native C engine does not implement Gumbel).
    p.add_argument("--gumbel-root", action="store_true", default=False,
                   help="Enable Gumbel AlphaZero root selection + Sequential "
                        "Halving. Default OFF = unchanged PUCT+Dirichlet self-play.")
    p.add_argument("--gumbel-m", type=int, default=16,
                   help="Number of root actions sampled via Gumbel-top-k "
                        "(clamped to #legal moves). Only used with --gumbel-root.")
    p.add_argument("--gumbel-c-visit", type=float, default=50.0,
                   help="Gumbel sigma(q) c_visit constant (paper default 50.0).")
    p.add_argument("--gumbel-c-scale", type=float, default=1.0,
                   help="Gumbel sigma(q) c_scale constant (paper default 1.0).")
    p.add_argument("--max-plies", type=int, default=None,
                   help="Optional bounded-worker cap for profiling/smoke runs. "
                        "Default None preserves full-game production behavior.")
    p.add_argument("--profile-output", type=str, default=None,
                   help="Write a JSON timing profile for bounded worker runs. "
                        "The profile separates native_search_batch, evaluator, "
                        "post-search Python, D4 augmentation, and file handoff.")

    # WL5 archive-start lever (wiki/topics/wl5-diagnostics-archive-start-design.md).
    # Per-game roll: with probability `--archive-start-frac` the game starts from
    # a uniform-random position drawn from the archive instead of the empty board.
    # Defaults disable the feature.
    p.add_argument("--archive-start-path", type=str, default=None,
                   help="Path to a .pt archive (torch.load dict with `planes`, "
                        "`side`, `ply` tensors). Loaded once at worker startup. "
                        "Enables WL5 archive-start when paired with "
                        "--archive-start-frac > 0.")
    p.add_argument("--archive-start-frac", type=float, default=0.0,
                   help="Per-game probability of seeding from an archive position "
                        "instead of the canonical empty board. Requires "
                        "--archive-start-path. 0.0 disables.")

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
                   help="Sleep this long if the weights file isn't ready yet on startup. "
                        "Also used as the base poll interval unless --weights-poll-min-sec "
                        "or --weights-poll-max-sec is set (in which case the worker draws "
                        "its own interval once at startup from Uniform(min, max)).")
    p.add_argument("--weights-poll-min-sec", type=float, default=None,
                   help="WL2 lever #3 (poll jitter): lower bound of the per-worker poll "
                        "interval, sampled once at startup. If unset, defaults to "
                        "--weights-poll-sec (no jitter).")
    p.add_argument("--weights-poll-max-sec", type=float, default=None,
                   help="WL2 lever #3 (poll jitter): upper bound of the per-worker poll "
                        "interval, sampled once at startup. If unset, defaults to "
                        "--weights-poll-sec (no jitter).")

    # WL2 lever #2: past-checkpoint opponent mix. Defaults disable the
    # feature; recommended starting values per the design doc are 0.4 / 0.1.
    p.add_argument("--opponent-mix-recent", type=float, default=0.0,
                   help="Per-wave probability of running full self-play against a "
                        "random checkpoint drawn from the last "
                        "--opponent-mix-recent-window snapshots. Games are still "
                        "written to the current model-version tile.")
    p.add_argument("--opponent-mix-history", type=float, default=0.0,
                   help="Per-wave probability of running full self-play against a "
                        "random checkpoint drawn from anywhere in the run history.")
    p.add_argument("--opponent-mix-recent-window", type=int, default=100,
                   help="Number of most-recent checkpoints eligible for the "
                        "--opponent-mix-recent draw.")
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
    p.add_argument("--fp16-eval", action="store_true",
                   help="Cast the worker's eval model to torch.float16 and feed "
                        "fp16 inputs to make_torch_evaluator. Forward outputs are "
                        "cast back to float32 inside the evaluator before host "
                        "transfer, so MCTS and game-record payloads are unchanged. "
                        "L06 perf-lab lane (see wiki/ops/perf-queue.md). Eval-only "
                        "— DO NOT set on the trainer's model.")
    p.add_argument("--evaluator", type=str, default="torch",
                   choices=["torch", "coreml"],
                   help="Inference backend. 'torch' = PyTorch on --device (default). "
                        "'coreml' = export the loaded model to a fresh .mlpackage "
                        "and use Core ML for leaf evaluation. Re-export on every "
                        "weight reload. Used by the L09 ANE-offload lane to free "
                        "the GPU from inference (R-TRAIN-ANE in perf-lab-charter).")
    p.add_argument("--coreml-compute-units", type=str, default="CPU_AND_NE",
                   help="Core ML compute-units selection when --evaluator coreml: "
                        "CPU_ONLY / CPU_AND_NE / CPU_AND_GPU / ALL. Default "
                        "CPU_AND_NE asks Core ML to schedule on the ANE when it "
                        "can. Use CPU_ONLY to isolate the Core ML path from any "
                        "GPU contention.")
    args = p.parse_args()
    # --fp16-eval is structurally incompatible AND semantically redundant
    # with --evaluator coreml: Core ML already exports at
    # compute_precision=FLOAT16, and torch.jit.trace inside export_model_to_coreml
    # passes a fp32 dummy input — casting model.half() first causes
    # "Input type (float) and bias type (Half) should be the same" at the
    # first conv. L09b discovered this at runtime. Force fp16_eval off for
    # the Core ML path; print a one-line note so the choice is auditable.
    if args.evaluator == "coreml" and args.fp16_eval:
        print(
            f"[{args.worker_id}] --fp16-eval ignored on --evaluator coreml "
            f"(Core ML is already FLOAT16 internally)",
            flush=True,
        )
        args.fp16_eval = False
    return args


def _build_evaluator(args: argparse.Namespace, model: torch.nn.Module, device: torch.device):
    """Construct an evaluator from a loaded model per --evaluator.

    'torch' returns the PyTorch evaluator on --device (the existing path).
    'coreml' exports the model to a fresh .mlpackage in a per-worker
    tempdir and wraps it as a CoreMLEvaluator. Re-export runs on every
    weight reload because export_model_to_coreml requires a concrete
    PyTorch module. Per the L09 charter, this is the ANE-offload path.

    --fp16-eval applies to the 'torch' path only (Core ML already exports
    at compute_precision=FLOAT16 above)."""
    if args.evaluator == "torch":
        return make_torch_evaluator(model, device, fp16=bool(args.fp16_eval))
    if args.evaluator == "coreml":
        from gomoku.coreml_evaluator import (
            export_model_to_coreml,
            make_coreml_evaluator,
        )
        tmp_root = Path(tempfile.gettempdir()) / f"gomoku_worker_{args.worker_id}_coreml"
        tmp_root.mkdir(parents=True, exist_ok=True)
        ml_path = tmp_root / "eval.mlpackage"
        # Cap Core ML's export-side max_batch at wave_size * games_per_batch
        # * 2 so the model can accept the largest leaf-batch the wave engine
        # might hand it. Keep it modest because Core ML compile-time scales
        # with the upper bound.
        max_batch = max(1, int(args.wave_size) * int(args.games_per_batch) * 2)
        # On CPU, the original PyTorch model must NOT be the same instance
        # we hand to JIT trace (jit moves it to CPU); pass the already-cpu
        # model so trainer-MPS gradients aren't disturbed.
        export_model_to_coreml(
            model.cpu(),
            ml_path,
            max_batch_size=max_batch,
            compute_precision="FLOAT16",
        )
        # Move back to original device if the caller still wants a torch
        # reference (we don't here, but be a good citizen).
        if device.type != "cpu":
            model.to(device)
        print(
            f"[{args.worker_id}] exported Core ML model -> {ml_path} "
            f"(compute_units={args.coreml_compute_units}, max_batch={max_batch})",
            flush=True,
        )
        return make_coreml_evaluator(
            ml_path,
            compute_units=args.coreml_compute_units,
            max_batch_size=max_batch,
        )
    raise ValueError(f"unknown --evaluator {args.evaluator!r}")


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


def _maybe_half(model: torch.nn.Module, enabled: bool, worker_id: str) -> torch.nn.Module:
    """Optionally cast the eval model to torch.float16 in-place.

    The L06 perf-lab lane: fp16 forward on MPS / CPU at eval time. Inputs
    are cast to .half() inside make_torch_evaluator and outputs cast back
    to float32 there before host transfer, so MCTS and game-record
    payloads stay fp32. Eval-only — DO NOT call on the trainer's model.
    """
    if not enabled:
        return model
    try:
        model = model.half()
        # Belt-and-suspenders: every parameter and buffer is now fp16.
        # A mismatched buffer (e.g. running_mean fused into the conv) would
        # raise at forward time on MPS; surface it now if it ever happens.
        print(f"[{worker_id}] fp16-eval enabled (model cast to torch.float16)", flush=True)
        return model
    except Exception as e:
        print(f"[{worker_id}] fp16-eval cast failed ({e}); using fp32 model", flush=True)
        return model


def _load_model(weights_path: str, device: torch.device) -> tuple[torch.nn.Module, float, int]:
    """Return (model, weights_mtime, version)."""
    while not os.path.exists(weights_path):
        time.sleep(1.0)
    model, payload = load_checkpoint(weights_path, device=device)
    model = fuse_model_for_inference(model)
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


def _list_past_checkpoints(weights_path: str) -> list[tuple[int, Path]]:
    """Scan the checkpoint dir (parent of `weights_path`) for `epochNNNN.pt`.

    Returns a list of (epoch_num, path) sorted ascending by epoch. The
    `latest.pt` symlink and `worker_weights.pt` are excluded; if a file
    happens to be a symlink whose realpath matches an `epoch*.pt` already
    in the list, it is deduped.
    """
    ckpt_dir = Path(weights_path).parent
    out: dict[int, Path] = {}
    for p in ckpt_dir.glob("epoch*.pt"):
        stem = p.stem  # epoch0089
        rest = stem.removeprefix("epoch")
        if not rest.isdigit():
            continue
        # Dedupe by realpath so symlinks pointing at the same blob collapse.
        try:
            real = p.resolve()
        except OSError:
            real = p
        epoch = int(rest)
        # Prefer the non-symlink (concrete) path when both exist.
        existing = out.get(epoch)
        if existing is None or (existing.is_symlink() and not p.is_symlink()):
            out[epoch] = p
        # If we've already got a concrete file, ignore symlink duplicates.
        _ = real
    return sorted(out.items(), key=lambda kv: kv[0])


def _pick_wave_mix_source(
    rng: np.random.Generator,
    p_recent: float,
    p_history: float,
    recent_window: int,
    weights_path: str,
) -> tuple[str, Path | None]:
    """Roll dice for what weights this wave should use.

    Returns (mix_source, checkpoint_path) where mix_source is one of
    "self" | "recent" | "history". For "self", checkpoint_path is None
    (caller keeps using the currently loaded weights). For "recent" and
    "history", checkpoint_path is the chosen `epochNNNN.pt`.

    If the requested bucket happens to be empty (e.g. early in a run with
    fewer checkpoints than recent_window), the roll falls back to "self"
    rather than crashing — the worker just generates one more wave with
    current weights, which is the safest no-op.
    """
    p_recent = max(0.0, float(p_recent))
    p_history = max(0.0, float(p_history))
    if p_recent + p_history <= 0.0:
        return "self", None
    if p_recent + p_history > 1.0:
        # Normalize defensively; "self" share would be negative otherwise.
        scale = p_recent + p_history
        p_recent /= scale
        p_history /= scale
    roll = float(rng.random())
    if roll < p_recent:
        bucket = "recent"
    elif roll < p_recent + p_history:
        bucket = "history"
    else:
        return "self", None

    ckpts = _list_past_checkpoints(weights_path)
    if not ckpts:
        return "self", None
    if bucket == "recent":
        window = max(1, int(recent_window))
        pool = ckpts[-window:]
    else:
        pool = ckpts
    idx = int(rng.integers(0, len(pool)))
    return bucket, pool[idx][1]


def _roll_wave_mix(
    rng: np.random.Generator,
    args: argparse.Namespace,
    *,
    current_evaluator,
    device: torch.device,
    model_version: int,
    worker_id: str,
):
    """Roll the WL2 lever #2 mix dice for one wave.

    Returns (wave_evaluator, wave_mix_source, wave_mix_label) where
    wave_mix_source is "self" | "recent" | "history" and wave_mix_label is
    a short string for logs/payloads ("current" or e.g. "epoch0089"). If
    the chosen bucket is empty or the load fails, falls back to "self" so
    the worker never blocks on a bad past checkpoint.

    Logs the choice with the standard `[wN] wave v{model_version}
    weights=... (mix=...)` line.
    """
    mix_source, ckpt_path = _pick_wave_mix_source(
        rng,
        p_recent=args.opponent_mix_recent,
        p_history=args.opponent_mix_history,
        recent_window=args.opponent_mix_recent_window,
        weights_path=args.weights_path,
    )
    if mix_source == "self" or ckpt_path is None:
        print(
            f"[{worker_id}] wave v{model_version} weights=current (mix=self)",
            flush=True,
        )
        return current_evaluator, "self", "current"

    try:
        past_model, _payload = load_checkpoint(str(ckpt_path), device=device)
        past_model = fuse_model_for_inference(past_model)
        past_model = _maybe_half(past_model, args.fp16_eval, worker_id)
        past_model = _maybe_compile(past_model, args.compile, worker_id)
        past_evaluator = make_torch_evaluator(
            past_model, device, fp16=bool(args.fp16_eval)
        )
    except Exception as e:
        print(
            f"[{worker_id}] wave v{model_version} past-ckpt load failed "
            f"({ckpt_path.name}: {e}); falling back to current",
            flush=True,
        )
        return current_evaluator, "self", "current"

    label = ckpt_path.stem  # e.g. "epoch0089"
    print(
        f"[{worker_id}] wave v{model_version} weights={label} (mix={mix_source})",
        flush=True,
    )
    return past_evaluator, mix_source, label


def _draw_poll_interval(
    rng: np.random.Generator,
    base_sec: float,
    min_sec: float | None,
    max_sec: float | None,
) -> float:
    """Resolve the per-worker poll interval (WL2 lever #3).

    If neither min nor max is set, returns `base_sec` (no jitter). If only
    one of min/max is set, the other defaults to `base_sec`. The interval
    is sampled exactly once per worker lifetime.
    """
    lo = min_sec if min_sec is not None else base_sec
    hi = max_sec if max_sec is not None else base_sec
    if hi < lo:
        lo, hi = hi, lo
    if hi <= lo:
        return float(lo)
    return float(rng.uniform(lo, hi))


def _generate_records(args: argparse.Namespace, evaluator, opp_picker, rng, n_games: int,
                      archive: dict | None = None, profile: dict[str, float] | None = None):
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
            max_plies=args.max_plies,
            rng=rng,
            wave_size=args.wave_size,
            random_opening_moves=args.random_opening_moves,
            archive=archive,
            archive_start_frac=args.archive_start_frac,
            playout_cap_frac=args.playout_cap_frac,
            playout_cap_fast_sims=args.playout_cap_fast_sims,
            profile=profile,
            gumbel_root=args.gumbel_root,
            gumbel_m=args.gumbel_m,
            gumbel_c_visit=args.gumbel_c_visit,
            gumbel_c_scale=args.gumbel_c_scale,
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
        max_plies=args.max_plies,
        rng=rng,
        wave_size=args.wave_size,
        model_first_frac=args.model_first_frac,
        random_opening_moves=args.random_opening_moves,
    )


def _load_archive(path: str | None, worker_id: str) -> dict | None:
    if not path:
        return None
    archive = torch.load(path, map_location="cpu")
    n = int(archive["planes"].shape[0])
    print(
        f"[{worker_id}] loaded archive {path} ({n} positions)",
        flush=True,
    )
    return archive


def _profile_add(profile: dict[str, float] | None, key: str, value: float) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + float(value)


def _write_profile(
    path: str | None,
    *,
    profile: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    model_version: int,
    batch_n: int,
    n_games: int,
    n_examples: int,
    plies: list[int],
    output_paths: list[str],
) -> None:
    if not path:
        return
    wall_s = float(profile.get("batch_wall_s", 0.0))
    search_s = float(profile.get("native_search_batch_s", 0.0))
    eval_s = float(profile.get("evaluator_s", 0.0))
    profile["native_search_excluding_evaluator_s"] = max(search_s - eval_s, 0.0)
    post_search_keys = [
        "post_search_loop_s",
        "record_build_s",
        "file_handoff_s",
        "active_list_s",
        "mtime_check_s",
        "worker_payload_s",
    ]
    profile["post_search_python_s"] = float(sum(profile.get(k, 0.0) for k in post_search_keys))
    shares = {
        k: (float(v) / wall_s if wall_s > 0 and isinstance(v, (int, float)) else 0.0)
        for k, v in profile.items()
        if k.endswith("_s")
    }
    payload = {
        "schema": "gomoku.outer_loop_profile.v1",
        "created_at_unix": time.time(),
        "hardware": platform.platform(),
        "python": platform.python_version(),
        "device": str(device),
        "worker_id": args.worker_id,
        "model_version": model_version,
        "batch_n": batch_n,
        "config": {
            "games_per_batch": args.games_per_batch,
            "n_simulations": args.n_simulations,
            "wave_size": args.wave_size,
            "max_plies": args.max_plies,
            "temperature_moves": args.temperature_moves,
            "temperature_final": args.temperature_final,
            "dirichlet_alpha": args.dirichlet_alpha,
            "dirichlet_eps": args.dirichlet_eps,
            "opponent": args.opponent,
            "wave_mode": bool(args.wave_mode),
        },
        "totals": {
            "games": n_games,
            "examples": n_examples,
            "plies": int(sum(plies)),
            "plies_mean": float(np.mean(plies)) if plies else 0.0,
            "output_paths": output_paths,
        },
        "timings_s": profile,
        "shares_of_batch_wall": shares,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"[{args.worker_id}] device={device}", flush=True)
    print(f"[{args.worker_id}] weights={args.weights_path} output={args.output_dir}", flush=True)

    archive = _load_archive(args.archive_start_path, args.worker_id)
    if archive is not None and args.archive_start_frac > 0:
        print(
            f"[{args.worker_id}] archive-start enabled: frac={args.archive_start_frac}",
            flush=True,
        )

    model, weights_mtime, model_version = _load_model(args.weights_path, device)
    model = _maybe_half(model, args.fp16_eval, args.worker_id)
    model = _maybe_compile(model, args.compile, args.worker_id)
    evaluator = _build_evaluator(args, model, device)
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

    # WL2 lever #3: each worker draws its own poll interval ONCE here.
    poll_sec = _draw_poll_interval(
        rng,
        base_sec=args.weights_poll_sec,
        min_sec=args.weights_poll_min_sec,
        max_sec=args.weights_poll_max_sec,
    )
    print(
        f"[{args.worker_id}] weights poll interval = {poll_sec:.1f}s",
        flush=True,
    )

    if args.wave_mode:
        games_on_version = 0
        next_seq = _next_wave_seq(out_dir, model_version, args.worker_id)
        print(
            f"[{args.worker_id}] wave-mode G={args.games_per_batch} "
            f"version={model_version} next_seq={next_seq}",
            flush=True,
        )

        # WL2 lever #2 wave-local state. `wave_evaluator` is what we generate
        # against this wave; it's either the current evaluator or a temporary
        # one wrapping a past checkpoint. `wave_mix_source` ("self"|"recent"
        # |"history") and `wave_mix_label` (e.g. "current" or "epoch0089") are
        # recorded in every game's payload so the trainer can aggregate
        # per-wave mix composition. All three are (re)rolled at every wave
        # boundary including this initial one before the loop starts.
        wave_evaluator, wave_mix_source, wave_mix_label = _roll_wave_mix(
            rng,
            args,
            current_evaluator=evaluator,
            device=device,
            model_version=model_version,
            worker_id=args.worker_id,
        )

        while True:
            profile = {} if args.profile_output else None
            batch_wall_t0 = time.perf_counter()
            # Wave mode reloads only at generation boundaries. Before the
            # worker has produced its G-game tile, keep filling the current
            # version even if a newer file appears unexpectedly.
            mtime_t0 = time.perf_counter()
            try:
                cur_mtime = os.path.getmtime(args.weights_path)
            except OSError:
                cur_mtime = weights_mtime
            _profile_add(profile, "mtime_check_s", time.perf_counter() - mtime_t0)
            if games_on_version >= args.games_per_batch and cur_mtime > weights_mtime:
                try:
                    model, payload = load_checkpoint(args.weights_path, device=device)
                    model = fuse_model_for_inference(model)
                    model = _maybe_half(model, args.fp16_eval, args.worker_id)
                    model = _maybe_compile(model, args.compile, args.worker_id)
                    evaluator = _build_evaluator(args, model, device)
                    weights_mtime = cur_mtime
                    model_version = int(payload.get("epoch", model_version + 1))
                    games_on_version = 0
                    next_seq = _next_wave_seq(out_dir, model_version, args.worker_id)
                    print(
                        f"[{args.worker_id}] wave reload version={model_version} "
                        f"mtime={weights_mtime:.0f} next_seq={next_seq}",
                        flush=True,
                    )
                    # WL2 lever #2: new wave, new roll. Even if "self" is
                    # chosen, this drops any past-checkpoint evaluator from
                    # the prior wave so we don't leak old weights.
                    wave_evaluator, wave_mix_source, wave_mix_label = _roll_wave_mix(
                        rng,
                        args,
                        current_evaluator=evaluator,
                        device=device,
                        model_version=model_version,
                        worker_id=args.worker_id,
                    )
                except Exception as e:
                    print(f"[{args.worker_id}] wave reload failed ({e}); retrying", flush=True)
                    time.sleep(poll_sec)
                    continue

            target_games = max(args.games_per_batch - games_on_version, 1)
            t0 = time.perf_counter()
            records = _generate_records(
                args, wave_evaluator, opp_picker, rng, target_games, archive=archive,
                profile=profile,
            )
            dt = time.perf_counter() - t0
            _profile_add(profile, "generate_records_s", dt)
            batch_n += 1
            archive_started = sum(1 for r in records if getattr(r, "archive_start", False))

            paths: list[Path] = []
            saved_examples = 0
            for record in records:
                seq = next_seq
                next_seq += 1
                payload_t0 = time.perf_counter()
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
                    # WL2 lever #2 accounting. mix_source is "self" | "recent"
                    # | "history"; mix_weights is the checkpoint label for
                    # debugging (e.g. "epoch0089" or "current").
                    "mix_source": wave_mix_source,
                    "mix_weights": wave_mix_label,
                }
                _profile_add(profile, "worker_payload_s", time.perf_counter() - payload_t0)
                save_t0 = time.perf_counter()
                saved = _atomic_save_wave_game(
                    out_dir, args.worker_id, model_version, seq, payload
                )
                _profile_add(profile, "file_handoff_s", time.perf_counter() - save_t0)
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
                f"tile_count={games_on_version} "
                f"mix={wave_mix_source}({wave_mix_label}) "
                f"archive_started={archive_started}",
                flush=True,
            )
            _profile_add(profile, "batch_wall_s", time.perf_counter() - batch_wall_t0)
            _write_profile(
                args.profile_output,
                profile=profile or {},
                args=args,
                device=device,
                model_version=model_version,
                batch_n=batch_n,
                n_games=len(paths),
                n_examples=saved_examples,
                plies=[int(r.plies) for r in records[:len(paths)]],
                output_paths=[str(p) for p in paths],
            )

            if args.max_batches > 0 and batch_n >= args.max_batches:
                print(f"[{args.worker_id}] hit max-batches={args.max_batches}, exiting", flush=True)
                return

    while True:
        profile = {} if args.profile_output else None
        batch_wall_t0 = time.perf_counter()
        # 1) reload weights if newer (cheap mtime check)
        mtime_t0 = time.perf_counter()
        try:
            cur_mtime = os.path.getmtime(args.weights_path)
        except OSError:
            cur_mtime = weights_mtime  # file might be mid-rename
        _profile_add(profile, "mtime_check_s", time.perf_counter() - mtime_t0)
        if cur_mtime > weights_mtime:
            try:
                model, payload = load_checkpoint(args.weights_path, device=device)
                model = fuse_model_for_inference(model)
                model = _maybe_half(model, args.fp16_eval, args.worker_id)
                model = _maybe_compile(model, args.compile, args.worker_id)
                evaluator = _build_evaluator(args, model, device)
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
            time.sleep(poll_sec)
            continue

        # 2) generate a batch of games
        t0 = time.perf_counter()
        records = _generate_records(
            args, evaluator, opp_picker, rng, args.games_per_batch, archive=archive,
            profile=profile,
        )
        dt = time.perf_counter() - t0
        _profile_add(profile, "generate_records_s", dt)
        archive_started = sum(1 for r in records if getattr(r, "archive_start", False))

        # 3) atomic write
        n_games = len(records)
        n_examples = sum(len(r.examples) for r in records)
        payload_t0 = time.perf_counter()
        payload = {
            "records": records,
            "worker_id": args.worker_id,
            "weights_mtime": weights_mtime,
            "model_version": model_version,
            "version": model_version,
            "n_games": n_games,
            "n_examples": n_examples,
            "gen_s": dt,
        }
        _profile_add(profile, "worker_payload_s", time.perf_counter() - payload_t0)
        save_t0 = time.perf_counter()
        path = _atomic_save(out_dir, args.worker_id, payload)
        _profile_add(profile, "file_handoff_s", time.perf_counter() - save_t0)
        batch_n += 1
        last_gen_mtime = weights_mtime
        print(
            f"[{args.worker_id}] batch {batch_n}: {n_games}g {n_examples}ex "
            f"{dt:.1f}s ({dt/n_games*1000:.0f} ms/game) "
            f"archive_started={archive_started} -> {path.name}",
            flush=True,
        )
        _profile_add(profile, "batch_wall_s", time.perf_counter() - batch_wall_t0)
        _write_profile(
            args.profile_output,
            profile=profile or {},
            args=args,
            device=device,
            model_version=model_version,
            batch_n=batch_n,
            n_games=n_games,
            n_examples=n_examples,
            plies=[int(r.plies) for r in records],
            output_paths=[str(path)],
        )

        if args.max_batches > 0 and batch_n >= args.max_batches:
            print(f"[{args.worker_id}] hit max-batches={args.max_batches}, exiting", flush=True)
            return


if __name__ == "__main__":
    main()
