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
        past_model.eval()
        past_model = _maybe_compile(past_model, args.compile, worker_id)
        past_evaluator = make_torch_evaluator(past_model, device)
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
                      archive: dict | None = None):
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
            archive=archive,
            archive_start_frac=args.archive_start_frac,
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
                args, wave_evaluator, opp_picker, rng, target_games, archive=archive
            )
            dt = time.perf_counter() - t0
            batch_n += 1
            archive_started = sum(1 for r in records if getattr(r, "archive_start", False))

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
                    # WL2 lever #2 accounting. mix_source is "self" | "recent"
                    # | "history"; mix_weights is the checkpoint label for
                    # debugging (e.g. "epoch0089" or "current").
                    "mix_source": wave_mix_source,
                    "mix_weights": wave_mix_label,
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
                f"tile_count={games_on_version} "
                f"mix={wave_mix_source}({wave_mix_label}) "
                f"archive_started={archive_started}",
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
            time.sleep(poll_sec)
            continue

        # 2) generate a batch of games
        t0 = time.perf_counter()
        records = _generate_records(
            args, evaluator, opp_picker, rng, args.games_per_batch, archive=archive
        )
        dt = time.perf_counter() - t0
        archive_started = sum(1 for r in records if getattr(r, "archive_start", False))

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
            f"{dt:.1f}s ({dt/n_games*1000:.0f} ms/game) "
            f"archive_started={archive_started} -> {path.name}",
            flush=True,
        )

        if args.max_batches > 0 and batch_n >= args.max_batches:
            print(f"[{args.worker_id}] hit max-batches={args.max_batches}, exiting", flush=True)
            return


if __name__ == "__main__":
    main()
