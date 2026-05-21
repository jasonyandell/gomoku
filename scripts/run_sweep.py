"""Multi-run-safe launcher for the K x buffer-size sweep.

Each cell defines a sweep point; the launcher generates unique directory paths
from the cell name so multiple cells can run in parallel without colliding on
checkpoint dirs, worker_records dirs, or wandb run names. Each cell launches:

  - 1 trainer (gomoku.train --no-eval)
  - N self-play workers (gomoku.selfplay_worker)
  - 1 eval worker (gomoku.eval_worker)

Usage::

    # See all defined cells.
    python scripts/run_sweep.py --list

    # Run one cell in the foreground (logs to stdout).
    python scripts/run_sweep.py --cell C --foreground

    # Default: spawn a cell in the background, return immediately.
    python scripts/run_sweep.py --cell C

    # Wipe a cell's persistent state (checkpoint dir + records dir).
    python scripts/run_sweep.py --cell C --clean

Per-process logs land in `sweep_logs/<cell>/{trainer,w0..wN,eval}.log`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
# Reuse whichever interpreter launched us — works from main repo OR a worktree
# that doesn't have its own .venv.
PYTHON = sys.executable


@dataclass
class Cell:
    """One sweep point. dirs/run_name are derived from `name`."""
    name: str
    sgd_per_game: float
    buffer_size: int
    games_per_epoch: int = 32              # also the worker_min_games default
    n_simulations: int = 800
    wave_size: int = 32
    batch_size: int = 512
    lr: float = 1e-3
    size: str = "medium"
    c_puct: float = 1.25
    c_puct_base: float = 19652.0
    dirichlet_alpha: float = 0.13
    dirichlet_eps: float = 0.25
    temperature_moves: int = 10
    temperature_final: float = 0.1
    n_workers: int = 4
    epochs: int = 100
    save_every: int = 1
    save_buffer_every: int = 20
    keep_last_n: int = 3
    stem_padding: int | None = None  # None = use train.py default (3, AGZ edge-fix)
    # Constant-age ingest: when set, ingest by positions instead of games. Pairs
    # with sgd_per_position so SGD steps also scale with positions. Together
    # they keep buffer turnover + training intensity stable regardless of
    # self-play game length (which swings between attack and defense regimes).
    worker_min_positions: int = 0
    sgd_per_position: float | None = None
    # Per-worker games-per-batch. Bigger batches give MCTS more leaves per
    # evaluator call, but the 2026-05-20 production read showed that one big
    # worker leaves MPS idle during Python tree work. At this model size, more
    # small workers plus wave batching beats a single wide worker.
    games_per_batch: int = 8
    # Wave-lockstep mode is a per-version tile barrier: workers generate a
    # tile against one published model, the trainer ingests that tile, then
    # publishes the next model. Only cells that opt in should receive the flag.
    wave_mode: bool = False
    # Apply torch.compile to worker models (eval-only). ~1.3-1.5x forward
    # speedup at batch>=32 for the small model on MPS.
    compile_workers: bool = False
    # WL2 levers (wiki/topics/wl2-scale-emulation-design.md). All default-off
    # so existing cells are unchanged. WL2 turns all four on.
    ema_tau: float = 0.0                          # lever #1: EMA self-play weights
    grad_accum_steps: int = 1                     # lever #4: gradient accumulation
    opponent_mix_recent: float = 0.0              # lever #2: past-checkpoint mix (recent)
    opponent_mix_history: float = 0.0             # lever #2: past-checkpoint mix (history)
    opponent_mix_recent_window: int = 100         # lever #2: recent window size
    weights_poll_min_sec: float | None = None     # lever #3: poll-interval jitter min
    weights_poll_max_sec: float | None = None     # lever #3: poll-interval jitter max
    # WL3 lever: K plies of uniform-random legal moves at game start. Training
    # examples are NOT recorded for those K plies — MCTS picks up at K+1 and
    # the model trains on post-random positions. Breaks opening monoculture.
    random_opening_moves: int = 0
    extra_train_args: list[str] = field(default_factory=list)
    extra_worker_args: list[str] = field(default_factory=list)


# Sweep matrix: K = sgd-per-game, buffer-size axes.
# Cell E ≈ what we just ran (control). C is the highest-contrast first try.
CELLS: dict[str, Cell] = {
    "A": Cell("A-K1-buf50k",   sgd_per_game=1.0, buffer_size=50_000),
    "B": Cell("B-K2-buf50k",   sgd_per_game=2.0, buffer_size=50_000),
    "C": Cell("C-K4-buf50k",   sgd_per_game=4.0, buffer_size=50_000),
    "D": Cell("D-K1-buf500k",  sgd_per_game=1.0, buffer_size=500_000),
    "E": Cell("E-K2-buf500k",  sgd_per_game=2.0, buffer_size=500_000),
    "F": Cell("F-K4-buf500k",  sgd_per_game=4.0, buffer_size=500_000),
    # AZ-recipe long run, tuned for laptop wall-clock: small model + padding=1
    # + sims=400. Trades the stem-padding edge-fix and some sim quality for ~4x
    # the per-cycle throughput, so a 160k-step run fits in a multi-day budget
    # rather than 2+ days. K=1, 1.5M buffer, tau_final=0.1, AGZ log-PUCT all
    # stay from the recipe imports.
    #
    # Production-proven config: 4 workers × 8 games × wave=64. The 4-worker
    # setup keeps MPS busy via OS-scheduled kernel interleaving across
    # processes — single-worker × big-batch hurts GPU utilization because
    # the GPU sits idle during Python tree-traversal between forward calls
    # (subagent's "MPS serializes processes" claim turned out wrong in
    # practice). wave=64 is the genuine win from the bench (1.71× per-worker
    # vs wave=32 with no plies regression). torch.compile dropped — gains
    # don't survive the per-cycle weight-reload recompiles.
    "Z": Cell("az-recipe-160k", sgd_per_game=1.0, buffer_size=1_500_000,
              size="small", stem_padding=1, n_simulations=400,
              lr=5e-4, epochs=5000,
              n_workers=8, wave_size=64, games_per_batch=8,
              compile_workers=False),
    # AZ-recipe long run with constant-age ingest. Same recipe as Z but uses
    # positions-based ingest + SGD so buffer turnover stays stable when
    # self-play game length swings (which Z's run showed it does — plies
    # dropped from ~60 to ~38 mid-run and broke training-loss stability).
    # Target: 32 games * 50 mean plies * 8 D4 aug = 12800 positions/cycle,
    # 32 SGD steps/cycle = 0.0025 SGD/position. Matches Z's effective rate
    # at the 50-plies baseline.
    "Zc": Cell("az-recipe-160k-constage", sgd_per_game=1.0,
               buffer_size=1_500_000, size="small", stem_padding=1,
               n_simulations=400, lr=5e-4, epochs=5000,
               worker_min_positions=12800, sgd_per_position=0.0025),
    # WL1: wave-lockstep at the proven 1.5M replay-buffer scale (matches
    # az-recipe-160k and avoids changing the buffer-size axis). First test of
    # the per-version uniformity hypothesis from
    # wiki/topics/wave-of-lockstep-design.md: 8 workers each produce an 8-game
    # tile against one model version, then the trainer steps and publishes the
    # next version. Existing sgd_per_position convention is K=1 per game at
    # ~50 plies with D4 augmentation:
    #   64 games * 50 plies * 8 aug = 25,600 positions/tile
    #   64 SGD steps / 25,600 positions = 0.0025
    "WL1": Cell("WL1-wave-lockstep-1p5M-buffer", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100),
    # WL2: WL1 + the four scale-emulation levers from
    # wiki/topics/wl2-scale-emulation-design.md, designed in response to WL1's
    # high-frequency strength oscillation (elo bounced 620-1281 across single
    # evals after e500, la4 regressed 52% -> 5%). Each lever emulates one
    # AZ-at-scale property our 8-worker laptop setup lacks:
    #   #1 EMA self-play weights (tau=0.99): decouples the brain that plays
    #      from the brain that learns -> emulates AZ's publish-to-workers lag.
    #   #2 past-checkpoint opponent mix (recent=0.4, history=0.1): some waves
    #      run an older snapshot on both sides -> emulates the multi-snapshot
    #      generation that AZ has by default from async publishing.
    #   #3 worker poll jitter (2-8s): workers pick up new weights at slightly
    #      different times -> emulates per-worker async-publish skew.
    #   #4 gradient accumulation 4x: cuts per-step gradient noise -> emulates
    #      AZ's batch=4096-against-broad-buffer stability.
    # Everything else identical to WL1 for an apples-to-apples test.
    "WL2": Cell("WL2-scale-emulation", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100,
                ema_tau=0.99,
                grad_accum_steps=4,
                opponent_mix_recent=0.4,
                opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0,
                weights_poll_max_sec=8.0),
    # WL3: WL2 + K=2 random opening plies. WL2 reached higher peaks than WL1
    # (la4=62% vs 52%) but exhibited the same arc-then-regression failure
    # mode. Working theory: the four scale-emulation levers raised the
    # ceiling by stabilizing the training feedback loop, but didn't address
    # opening monoculture — even past-checkpoint mix delivers diverse brains
    # that all share the same opening lineage. WL3 adds K=2 uniform-random
    # legal moves at game start; training examples are NOT recorded for the
    # random plies (per train.py:165-170), so the model just sees more
    # diverse mid-game starting positions without learning broken-move signal.
    "WL3": Cell("WL3-random-openings", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100,
                ema_tau=0.99,
                grad_accum_steps=4,
                opponent_mix_recent=0.4,
                opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0,
                weights_poll_max_sec=8.0,
                random_opening_moves=2),
    # WL3.1: identical to WL3, restart after WL3 died at e825 from a NaN-
    # in-policy crash. The native MCTS engine occasionally emits NaN visit-
    # policies; commits c5049be + 0557671 add NaN guards at the play path
    # (gomoku/self_play.py::_sample_action) and the data-recording path
    # (trajectory pi sanitization). With those guards in place, an
    # occasional NaN no longer kills workers or poisons the buffer — it
    # falls back to argmax / uniform target. A parallel investigator is
    # looking at the underlying MCTS NaN root cause.
    "WL3.1": Cell("WL3.1-random-openings-nanfix", sgd_per_game=1.0,
                  buffer_size=1_500_000, games_per_epoch=64,
                  size="small", stem_padding=1, n_simulations=400,
                  n_workers=8, games_per_batch=8, wave_mode=True,
                  c_puct=1.25, c_puct_base=19652.0,
                  dirichlet_alpha=0.13, dirichlet_eps=0.25,
                  temperature_moves=30, temperature_final=0.1,
                  sgd_per_position=0.0025,
                  save_buffer_every=100,
                  ema_tau=0.99,
                  grad_accum_steps=4,
                  opponent_mix_recent=0.4,
                  opponent_mix_history=0.1,
                  opponent_mix_recent_window=100,
                  weights_poll_min_sec=2.0,
                  weights_poll_max_sec=8.0,
                  random_opening_moves=2),
    # WL4: WL3.1 with random openings TURNED OFF. Resumes from WL3.1 e1536
    # (latest.pt snapshotted as $CLAUDE_JOB_DIR/wl3.1_e1536_latest.pt) which
    # reached the "established model" trigger Jason proposed:
    # eval/vs_heuristic hit 100% sustained (e1123-1157), la4 sustained 60-95%
    # across 5+ evals, plies hit 27 (single-eval). Hypothesis: with diversity
    # baked in, canonical-opening depth becomes the bottleneck. Pulling
    # random openings should either (a) unlock further breakthrough as the
    # model can finally compound on canonical lines, or (b) cause rapid
    # collapse — showing diversity is permanent training infrastructure at
    # this model size. Both outcomes informative.
    "WL4": Cell("WL4-no-random-openings", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100,
                ema_tau=0.99,
                grad_accum_steps=4,
                opponent_mix_recent=0.4,
                opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0,
                weights_poll_max_sec=8.0,
                random_opening_moves=0),  # ← the experimental change
}


def cell_dirs(cell: Cell) -> dict:
    base = REPO_ROOT / "sweep_runs" / cell.name
    return {
        "base": base,
        "checkpoint_dir": base / "checkpoints",
        "records_dir": base / "checkpoints" / "_records",
        "worker_weights": base / "checkpoints" / "worker_weights.pt",
        "log_dir": REPO_ROOT / "sweep_logs" / cell.name,
    }


def trainer_cmd(cell: Cell, dirs: dict) -> list[str]:
    cmd = [
        PYTHON, "-u", "-m", "gomoku.train",
        "--size", cell.size,
        "--epochs", str(cell.epochs),
        "--games-per-epoch", str(cell.games_per_epoch),
        "--n-simulations", str(cell.n_simulations),
        "--wave-size", str(cell.wave_size),
        "--sgd-per-game", str(cell.sgd_per_game),
        "--min-training-steps", "16",
        "--batch-size", str(cell.batch_size),
        "--lr", str(cell.lr),
        "--replay-buffer-size", str(cell.buffer_size),
        "--c-puct", str(cell.c_puct),
        "--c-puct-base", str(cell.c_puct_base),
        "--dirichlet-alpha", str(cell.dirichlet_alpha),
        "--dirichlet-eps", str(cell.dirichlet_eps),
        "--temperature-moves", str(cell.temperature_moves),
        "--temperature-final", str(cell.temperature_final),
        "--worker-input-dir", str(dirs["records_dir"]),
        "--worker-weights-path", str(dirs["worker_weights"]),
        "--worker-min-games", str(cell.games_per_epoch),
        "--checkpoint-dir", str(dirs["checkpoint_dir"]),
        "--save-every", str(cell.save_every),
        "--save-buffer-every", str(cell.save_buffer_every),
        "--keep-last-n", str(cell.keep_last_n),
        "--no-eval",
        "--wandb", "--wandb-project", "gomoku",
        "--run-name", f"9x9-sweep-{cell.name}",
        *cell.extra_train_args,
    ]
    if cell.stem_padding is not None:
        cmd += ["--stem-padding", str(cell.stem_padding)]
    if cell.worker_min_positions > 0:
        cmd += ["--worker-min-positions", str(cell.worker_min_positions)]
    if cell.sgd_per_position is not None:
        cmd += ["--sgd-per-position", str(cell.sgd_per_position)]
    if cell.wave_mode:
        cmd += [
            "--wave-mode",
            "--wave-workers", str(cell.n_workers),
            "--wave-games-per-worker", str(cell.games_per_batch),
        ]
    if cell.ema_tau > 0:
        cmd += ["--ema-tau", str(cell.ema_tau)]
    if cell.grad_accum_steps > 1:
        cmd += ["--grad-accum-steps", str(cell.grad_accum_steps)]
    if cell.random_opening_moves > 0:
        cmd += ["--random-opening-moves", str(cell.random_opening_moves)]
    return cmd


def worker_cmd(cell: Cell, dirs: dict, worker_id: str, seed: int) -> list[str]:
    cmd = [
        PYTHON, "-u", "-m", "gomoku.selfplay_worker",
        "--weights-path", str(dirs["worker_weights"]),
        "--output-dir", str(dirs["records_dir"]),
        "--worker-id", worker_id,
        "--games-per-batch", str(cell.games_per_batch),
        "--n-simulations", str(cell.n_simulations),
        "--wave-size", str(cell.wave_size),
        "--c-puct", str(cell.c_puct),
        "--c-puct-base", str(cell.c_puct_base),
        "--temperature-moves", str(cell.temperature_moves),
        "--temperature-final", str(cell.temperature_final),
        "--dirichlet-alpha", str(cell.dirichlet_alpha),
        "--dirichlet-eps", str(cell.dirichlet_eps),
        "--seed", str(seed),
        *cell.extra_worker_args,
    ]
    if cell.wave_mode:
        cmd += ["--wave-mode"]
    if cell.compile_workers:
        cmd += ["--compile"]
    if cell.opponent_mix_recent > 0:
        cmd += ["--opponent-mix-recent", str(cell.opponent_mix_recent)]
    if cell.opponent_mix_history > 0:
        cmd += ["--opponent-mix-history", str(cell.opponent_mix_history)]
    if cell.opponent_mix_recent > 0 or cell.opponent_mix_history > 0:
        cmd += ["--opponent-mix-recent-window", str(cell.opponent_mix_recent_window)]
    if cell.weights_poll_min_sec is not None:
        cmd += ["--weights-poll-min-sec", str(cell.weights_poll_min_sec)]
    if cell.weights_poll_max_sec is not None:
        cmd += ["--weights-poll-max-sec", str(cell.weights_poll_max_sec)]
    if cell.random_opening_moves > 0:
        cmd += ["--random-opening-moves", str(cell.random_opening_moves)]
    return cmd


def eval_cmd(cell: Cell, dirs: dict) -> list[str]:
    # Lookahead:depth=4 is included as a 4th anchor: with depth=4 anchored at
    # 1500 Elo the implied_elo MLE doesn't saturate against lookahead2 (1200),
    # so the model_elo trajectory shows real strength changes instead of
    # ceiling-clamping when the model crushes the lower-rated baselines.
    # n_workers=4 keeps the multi-baseline eval cycle under ~3min via parallel
    # game-playing (see gomoku/eval.py:play_match_parallel).
    return [
        PYTHON, "-u", "-m", "gomoku.eval_worker",
        "--checkpoint-path", str(dirs["worker_weights"]),
        "--baselines", "random,heuristic,lookahead:depth=2,lookahead:depth=4",
        "--n-games", "20",
        "--sims", "100",
        "--device", "cpu",
        "--n-workers", "4",
        "--poll-sec", "2.0",
    ]


def list_cells() -> None:
    print(f"{'cell':<6} {'name':<20} {'K':<6} {'buffer':<10} {'sims':<6} {'workers':<8} {'epochs':<8}")
    for k, c in CELLS.items():
        print(f"{k:<6} {c.name:<20} {c.sgd_per_game:<6} {c.buffer_size:<10} "
              f"{c.n_simulations:<6} {c.n_workers:<8} {c.epochs:<8}")


def clean_cell(cell: Cell) -> None:
    dirs = cell_dirs(cell)
    for key in ("checkpoint_dir", "log_dir"):
        p = dirs[key]
        if p.exists():
            shutil.rmtree(p)
            print(f"removed {p}")
    base = dirs["base"]
    if base.exists() and not any(base.iterdir()):
        base.rmdir()


def launch_cell(cell: Cell, foreground: bool, resume_path: str | None = None) -> None:
    dirs = cell_dirs(cell)
    dirs["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    dirs["records_dir"].mkdir(parents=True, exist_ok=True)
    dirs["log_dir"].mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    procs: list[tuple[str, subprocess.Popen]] = []

    def spawn(label: str, cmd: list[str], env_override: dict | None = None):
        log_path = dirs["log_dir"] / f"{label}.log"
        log_f = open(log_path, "a")
        e = dict(env)
        if env_override:
            e.update(env_override)
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=e, cwd=str(REPO_ROOT))
        procs.append((label, p))
        print(f"  spawned {label} pid={p.pid} -> {log_path}")

    print(f"=== launching cell {cell.name} ===")
    t_cmd = trainer_cmd(cell, dirs)
    if resume_path:
        t_cmd += ["--resume", str(resume_path)]
        print(f"  trainer resuming from {resume_path}")
    spawn("trainer", t_cmd)
    # Give the trainer a head start so workers find the initial weights file.
    time.sleep(2.0)
    for i in range(cell.n_workers):
        spawn(f"w{i}", worker_cmd(cell, dirs, f"w{i}", seed=1000 + i))
    spawn("eval", eval_cmd(cell, dirs), env_override={"GOMOKU_DEVICE": "cpu"})

    if not foreground:
        print(f"backgrounded {len(procs)} processes for cell {cell.name}")
        print(f"  tail logs:  tail -F {dirs['log_dir']}/trainer.log")
        print(f"  stop cell:  pkill -f 'sweep-{cell.name}|sweep_runs/{cell.name}/'")
        return

    print(f"foregrounded — Ctrl-C to stop all {len(procs)} processes")
    try:
        while True:
            time.sleep(2.0)
            for label, p in procs:
                rc = p.poll()
                if rc is not None:
                    print(f"  {label} exited with rc={rc}; stopping others")
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        for label, p in procs:
            if p.poll() is None:
                p.terminate()
        print("stopped all")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="list defined cells and exit")
    p.add_argument("--cell", type=str, help="cell key from CELLS")
    p.add_argument("--foreground", action="store_true",
                   help="run in foreground (Ctrl-C stops all). Default backgrounds.")
    p.add_argument("--clean", action="store_true",
                   help="delete the cell's checkpoint+log dirs (does not stop a running cell)")
    p.add_argument("--epochs", type=int, default=None, help="override cell.epochs")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint (.pt) to resume the trainer from. "
                        "Passes through as --resume to gomoku.train. wandb run id "
                        "embedded in the checkpoint will continue the same wandb "
                        "timeline. Useful for swapping perf configs mid-run.")
    args = p.parse_args()

    if args.list or not args.cell:
        list_cells()
        return

    if args.cell not in CELLS:
        raise SystemExit(f"unknown cell {args.cell!r}; one of {sorted(CELLS)}")
    cell = CELLS[args.cell]
    if args.epochs:
        cell.epochs = args.epochs

    if args.clean:
        clean_cell(cell)
        return

    launch_cell(cell, foreground=args.foreground, resume_path=args.resume)


if __name__ == "__main__":
    main()
