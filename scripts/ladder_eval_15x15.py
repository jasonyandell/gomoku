#!/usr/bin/env python3
"""Paced EXTERNAL-LADDER eval driver for the live 15x15 training campaign.

WHY: the during-training anchored ladder (random / heuristic / lookahead2 /
lookahead4) is 9x9-calibrated and SATURATES instantly at 15x15 — heuristic and
both lookahead tiers sit at 0.0 win-rate from epoch 0, so they give NO usable
strength signal for the 15x15 run. The only honest yardstick past that ceiling
is the rated external engine Rapfi (Gomocup freestyle Elo ~2625). This driver
periodically scores the LIVE run's current checkpoint vs Rapfi at 15x15 and
appends one strength row per tier, so the campaign loop can read a real
Delta-elo signal.

This is a thin wrapper around ``scripts/eval_vs_rapfi.py``'s ``run_rapfi_eval``
core (NO duplicated match / model-load / JSONL logic). It adds:
  - forces / requires GOMOKU_BOARD_SIZE=15 BEFORE any gomoku import (board size
    is a process-level constant fixed at import time — see board_config.py);
  - a checkpoint resolver: the live run writes ``epoch00NN.pt`` (no ``latest.pt``
    symlink), so a missing checkpoint resolves to the newest ``epoch*.pt`` in the
    same directory (else ``worker_weights.pt``);
  - small, cheap defaults (few games, low-difficulty tiers) so it does not
    compete with the GPU-owning live run — Rapfi is CPU; our model eval is a
    brief MPS (or CPU) forward.

Safe to call repeatedly: every row is APPENDED to the JSONL out file with a
timestamp + resolved checkpoint + per-tier win-rate + config provenance.

EVAL-ONLY. Rapfi never touches self-play / training.

Campaign-loop usage (one eval tick):
    GOMOKU_BOARD_SIZE=15 python scripts/ladder_eval_15x15.py \
        --checkpoint sweep_runs/G15-seed-v8recipe-board15/checkpoints/latest.pt \
        --rapfi engines/rapfi/pbrain-rapfi \
        --n-games 10 --timeouts 200 1000 \
        --out sweep_runs/G15-seed-v8recipe-board15/rapfi_ladder_15x15.jsonl
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# Board size is a PROCESS-LEVEL constant resolved once at import time by
# gomoku.board_config (precedence: --board-size flag > GOMOKU_BOARD_SIZE env >
# default 9). The external-engine coordinate mapping derives from it, so we MUST
# pin it to 15 BEFORE importing eval_vs_rapfi (which transitively imports
# gomoku.*). Do this at module top, before the heavy import below.
_REQUIRED_BOARD_SIZE = 15


def _pin_board_size() -> None:
    """Force/require GOMOKU_BOARD_SIZE=15 before any gomoku import.

    If the env var is already set to something other than 15, that is a hard
    error (the operator asked for a 15x15 ladder but the process would resolve a
    different size). Otherwise we set it to 15 and propagate to children.
    """
    cur = os.environ.get("GOMOKU_BOARD_SIZE")
    if cur is not None and cur.strip() and int(cur) != _REQUIRED_BOARD_SIZE:
        raise SystemExit(
            f"GOMOKU_BOARD_SIZE={cur} but this 15x15 ladder driver requires "
            f"{_REQUIRED_BOARD_SIZE}. Unset it or set it to {_REQUIRED_BOARD_SIZE}."
        )
    os.environ["GOMOKU_BOARD_SIZE"] = str(_REQUIRED_BOARD_SIZE)


def _pin_cpu_default() -> None:
    """Default the model evaluator to CPU so a paced ladder eval never competes
    with the GPU-owning live training run.

    This driver is meant to be called periodically WHILE a G15-seed training run
    owns the GPU (8 self-play workers + trainer on MPS). Rapfi is CPU-only; our
    model's MCTS forwards are the only thing that could touch MPS. A handful of
    eval games on CPU is plenty fast and keeps the GPU uncontended. The operator
    can override for a free GPU by exporting GOMOKU_DEVICE=mps before launch.
    """
    if not os.environ.get("GOMOKU_DEVICE", "").strip():
        os.environ["GOMOKU_DEVICE"] = "cpu"


DEFAULT_CHECKPOINT = (
    "sweep_runs/G15-seed-v8recipe-board15/checkpoints/latest.pt"
)
DEFAULT_RAPFI = "engines/rapfi/pbrain-rapfi"
WARMSTART_SEED = "sweep_runs/g15_warmstart_seed.pt"

_EPOCH_RE = re.compile(r"epoch0*?(\d+)\.pt$")


def resolve_checkpoint(path: str) -> str:
    """Resolve the checkpoint to score, tolerating the live run's layout.

    The live run does NOT write a ``latest.pt`` symlink — it writes
    ``epoch00NN.pt`` files plus a rolling ``worker_weights.pt``. So:
      - if ``path`` exists, use it as-is;
      - else look in its directory for the newest ``epoch*.pt`` (by epoch
        number), then fall back to ``worker_weights.pt``;
      - else fall back to the warm-start seed if it exists;
      - else error.
    Returns an absolute path.
    """
    if os.path.exists(path):
        return os.path.abspath(path)

    ckpt_dir = os.path.dirname(os.path.abspath(path)) or "."
    epoch_files = glob.glob(os.path.join(ckpt_dir, "epoch*.pt"))
    if epoch_files:
        def _epoch_num(p: str) -> int:
            m = _EPOCH_RE.search(os.path.basename(p))
            return int(m.group(1)) if m else -1

        newest = max(epoch_files, key=_epoch_num)
        return os.path.abspath(newest)

    worker = os.path.join(ckpt_dir, "worker_weights.pt")
    if os.path.exists(worker):
        return os.path.abspath(worker)

    if os.path.exists(WARMSTART_SEED):
        return os.path.abspath(WARMSTART_SEED)

    raise SystemExit(
        f"no checkpoint found: {path!r} does not exist, and no epoch*.pt / "
        f"worker_weights.pt in {ckpt_dir!r}, and warm-start seed "
        f"{WARMSTART_SEED!r} is missing."
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help="Checkpoint to score (default: the live 15x15 run's "
                         "latest.pt; a missing path resolves to the newest "
                         "epoch*.pt / worker_weights.pt in its directory).")
    ap.add_argument("--rapfi", default=DEFAULT_RAPFI,
                    help="Path to the pbrain-rapfi binary (CPU engine).")
    ap.add_argument("--n-games", type=int, default=10,
                    help="Games per tier, color-alternated (keep small/cheap; "
                         "the GPU is owned by the live run). small-n = noisy hint.")
    ap.add_argument("--timeouts", type=int, nargs="+", default=[200, 1000],
                    help="Rapfi per-move timeout tiers in ms (difficulty tiers).")
    ap.add_argument("--sims", type=int, default=100,
                    help="Model MCTS sims per move.")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", type=int, default=0, help="0 = freestyle.")
    # Eval-config levers (the 9x9 champion's 100%-ladder-sweep recipe). Default
    # OFF = byte-identical to the plain ladder eval. Testing whether these
    # transfer to 15x15 is a free strength experiment (no retraining).
    ap.add_argument("--fpu-reduction-c", type=float, default=0.0,
                    help="KataGo FPU reduction (0.0=OFF). 0.45 = verified 9x9 derby value.")
    ap.add_argument("--reuse-tree", action="store_true",
                    help="Persistent MCTS tree across moves (derby-jmi lever).")
    ap.add_argument("--proven-prop", action="store_true",
                    help="Proven win/loss propagation (derby-b3n lever).")
    ap.add_argument("--out", default=None,
                    help="JSONL output path (APPEND). Default: a "
                         "rapfi_ladder_15x15.jsonl alongside the checkpoint dir.")
    args = ap.parse_args(argv)

    # Heavy import is deferred until after the board size is pinned.
    from eval_vs_rapfi import run_rapfi_eval  # noqa: E402

    ckpt = resolve_checkpoint(args.checkpoint)

    out = args.out
    if out is None:
        # Sibling of the checkpoints dir by default, so all ladder rows for a
        # run live next to it.
        ckpt_dir = os.path.dirname(ckpt)
        run_dir = os.path.dirname(ckpt_dir) if os.path.basename(ckpt_dir) == "checkpoints" else ckpt_dir
        out = os.path.join(run_dir, "rapfi_ladder_15x15.jsonl")

    print(
        f"[ladder-15x15] checkpoint={ckpt}\n"
        f"[ladder-15x15] rapfi={args.rapfi} board_size={_REQUIRED_BOARD_SIZE} "
        f"n_games={args.n_games} timeouts={args.timeouts} sims={args.sims}\n"
        f"[ladder-15x15] eval_config: fpu={args.fpu_reduction_c} "
        f"reuse_tree={args.reuse_tree} proven_prop={args.proven_prop}\n"
        f"[ladder-15x15] out={out}",
        flush=True,
    )

    records = run_rapfi_eval(
        checkpoint=ckpt,
        rapfi=args.rapfi,
        timeouts=args.timeouts,
        sims=args.sims,
        c_puct=args.c_puct,
        n_games=args.n_games,
        seed=args.seed,
        rule=args.rule,
        size=_REQUIRED_BOARD_SIZE,
        out=out,
        fpu_reduction_c=args.fpu_reduction_c,
        reuse_tree=args.reuse_tree,
        proven_prop=args.proven_prop,
        extra_fields={"ladder": "rapfi_15x15"},
    )
    print(f"[ladder-15x15] wrote {len(records)} row(s) to {out}", flush=True)


# Pin the board size at import time, before this module's heavy import path and
# before argparse runs, so `from eval_vs_rapfi import ...` resolves BOARD_SIZE=15.
_pin_board_size()
_pin_cpu_default()

# Make sibling scripts importable (eval_vs_rapfi lives in the same scripts/ dir).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
