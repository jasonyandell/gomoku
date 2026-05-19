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
    dirichlet_alpha: float = 0.13
    dirichlet_eps: float = 0.25
    temperature_moves: int = 10
    n_workers: int = 4
    epochs: int = 100
    save_every: int = 1
    save_buffer_every: int = 20
    keep_last_n: int = 3
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
    return [
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
        "--dirichlet-alpha", str(cell.dirichlet_alpha),
        "--dirichlet-eps", str(cell.dirichlet_eps),
        "--temperature-moves", str(cell.temperature_moves),
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


def worker_cmd(cell: Cell, dirs: dict, worker_id: str, seed: int) -> list[str]:
    return [
        PYTHON, "-u", "-m", "gomoku.selfplay_worker",
        "--weights-path", str(dirs["worker_weights"]),
        "--output-dir", str(dirs["records_dir"]),
        "--worker-id", worker_id,
        "--games-per-batch", "8",
        "--n-simulations", str(cell.n_simulations),
        "--wave-size", str(cell.wave_size),
        "--temperature-moves", str(cell.temperature_moves),
        "--dirichlet-alpha", str(cell.dirichlet_alpha),
        "--dirichlet-eps", str(cell.dirichlet_eps),
        "--seed", str(seed),
        *cell.extra_worker_args,
    ]


def eval_cmd(cell: Cell, dirs: dict) -> list[str]:
    return [
        PYTHON, "-u", "-m", "gomoku.eval_worker",
        "--checkpoint-path", str(dirs["worker_weights"]),
        "--baselines", "random,heuristic,lookahead:depth=2",
        "--n-games", "20",
        "--sims", "100",
        "--device", "cpu",
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


def launch_cell(cell: Cell, foreground: bool) -> None:
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
    spawn("trainer", trainer_cmd(cell, dirs))
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

    launch_cell(cell, foreground=args.foreground)


if __name__ == "__main__":
    main()
