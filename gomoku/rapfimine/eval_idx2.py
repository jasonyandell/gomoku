"""The idx-2 gate: H2H of a checkpoint vs native Rapfi-NNUE, from the idx-2
opening only, white reported SEPARATELY (the real bar — #34/#18).

This is the verdict instrument for the Bruce-Lee experiment: run it on the
PRETRAINED net (cheap sanity — can supervised distillation alone hold its own?)
and again on the WARM-STARTED self-play net. Reuses gomoku.eval_panel entirely.

    GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.eval_idx2 \\
        --checkpoint checkpoints/idx2_pretrain.pt --n-games 48 --sims 160
"""
from __future__ import annotations

import argparse
import json
import sys

from gomoku.board_config import BOARD_SIZE
from gomoku.eval_panel import (
    IDX2_OPENING, EvaluatorCache, Ruler, eval_vs_ruler, fixed_opening_state,
    result_to_row,
)
from gomoku.rapfi_pool import RapfiPool, rapfi_available


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-games", type=int, default=48)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--rapfi-timeout-ms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    if BOARD_SIZE != 15:
        print("WARNING: GOMOKU_BOARD_SIZE != 15", file=sys.stderr)
    if not rapfi_available():
        print("ERROR: native Rapfi not available", file=sys.stderr)
        return 2

    start = fixed_opening_state(IDX2_OPENING)
    cache = EvaluatorCache(device=args.device)
    ruler = Ruler(label="rapfi@idx2", opponent="rapfi", n_games=args.n_games,
                  sims=args.sims)
    with RapfiPool(size=1, timeout_ms=args.rapfi_timeout_ms, board_size=15) as pool:
        res = eval_vs_ruler(args.checkpoint, ruler, cache=cache, pool=pool,
                            start_state=start, seed=args.seed)
    row = result_to_row("rapfi@idx2", res)
    print(json.dumps(row, indent=2))
    print(f"\nidx-2 vs Rapfi (n={row['n_games']}): "
          f"overall={row.get('score')}  "
          f"BLACK={row.get('black_score')}  WHITE={row.get('white_score')}  "
          f"white_loss_rate={row.get('white_loss_rate')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
