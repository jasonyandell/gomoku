"""Eval GRADIENT for the idx-2 warm-start net — progress against a strength ladder.

`eval_idx2` only measures vs max-strength Rapfi-NNUE, which is a wall: a climbing
net reads 0/48 for many hours before it dents the top. To SEE progress we play the
net (from idx-2 only, white split) against a ladder of opponents spanning weak →
strong, so improvement shows up as the net clearing successive rungs:

    random  <  heuristic  <  lookahead-d2  <  lookahead-d4  <  rapfi@100ms  <  rapfi@1000ms

All rungs are the project's STANDARD opponents (gomoku.baselines via parse_spec,
and the native Rapfi pool graded by per-move time) — no new engines invented.
Rapfi strength is graded by `timeout_ms` (think-time per move); the baselines span
the bottom where the early climb actually happens (run_sweep's --internal-eval
ladder, same idea). White is reported SEPARATELY per rung (the known-hard side).

    GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.eval_gradient \\
        --checkpoint sweep_runs/.../checkpoints/epoch0150.pt --n-games 24 --sims 160
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from gomoku.board_config import BOARD_SIZE
from gomoku.eval_panel import (
    IDX2_OPENING, EvaluatorCache, Ruler, eval_vs_ruler, fixed_opening_state,
    result_to_row,
)
from gomoku.rapfi_pool import RapfiPool, rapfi_available


# (label, opponent_spec, rapfi_timeout_ms | None) — weak → strong. Non-rapfi rungs
# are gomoku.baselines specs; rapfi rungs lease a warm engine at the given think-time.
# At epoch ~145 the warm-start net already CRUSHES random/heuristic/lookahead-d2
# (100% both colors) while reading 0% vs max Rapfi — so the informative band is
# lookahead-d4 → graded-Rapfi. Rapfi think-time (timeout_ms) is the strength dial
# that fills the gap between "beats classical search" and "touches max Rapfi".
DEFAULT_RUNGS = [
    ("heuristic",    "heuristic",         None),  # cheap canary (must stay ~1.0)
    ("lookahead-d4", "lookahead:depth=4", None),  # strongest classical rung
    ("rapfi@50ms",   "rapfi",             50),
    ("rapfi@200ms",  "rapfi",             200),
    ("rapfi@1000ms", "rapfi",             1000),   # ~the max-strength bar
]


def _epoch_of(path: str) -> int:
    m = re.search(r"epoch0*([0-9]+)\.pt", path)
    return int(m.group(1)) if m else -1


def run_gradient(checkpoint: str, *, n_games: int, sims: int, seed: int,
                 device=None, rungs=DEFAULT_RUNGS):
    start = fixed_opening_state(IDX2_OPENING)
    cache = EvaluatorCache(device=device)
    out = []
    for label, opponent, tmo in rungs:
        ruler = Ruler(label=label, opponent=opponent, n_games=n_games, sims=sims)
        if opponent == "rapfi":
            with RapfiPool(size=1, timeout_ms=tmo, board_size=BOARD_SIZE) as pool:
                res = eval_vs_ruler(checkpoint, ruler, cache=cache, pool=pool,
                                    start_state=start, seed=seed)
        else:
            res = eval_vs_ruler(checkpoint, ruler, cache=cache, pool=None,
                                start_state=start, seed=seed)
        row = result_to_row(label, res)
        out.append({
            "rung": label,
            "score": row.get("score"),
            "black": row.get("black_score"),
            "white": row.get("white_score"),
            "white_loss_rate": row.get("white_loss_rate"),
            "n_games": row.get("n_games"),
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-games", type=int, default=24)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    if BOARD_SIZE != 15:
        print("WARNING: GOMOKU_BOARD_SIZE != 15", file=sys.stderr)
    if not rapfi_available():
        print("ERROR: native Rapfi not available", file=sys.stderr)
        return 2

    ep = _epoch_of(args.checkpoint)
    rows = run_gradient(args.checkpoint, n_games=args.n_games, sims=args.sims,
                        seed=args.seed, device=args.device)
    print(json.dumps({"checkpoint": args.checkpoint, "epoch": ep, "rungs": rows},
                     indent=2))
    # One compact gradient line (overall|white per rung) — the curve at a glance.
    parts = [f"{r['rung']}={r['score']:.2f}/w{r['white']:.2f}" for r in rows]
    print(f"\nGRADIENT epoch={ep} (n={args.n_games} sims={args.sims}): "
          + "  ".join(parts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
