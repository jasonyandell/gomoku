"""What does the max_nodes=250 cap HIDE? Re-solve a sample of `capped` nodes at
escalating budgets and measure how many flip to a decided verdict.

A `capped` node is inconclusive: the VCT search hit the node budget without
proving a win OR exhausting the tree. So the harvested win/loss counts are LOWER
BOUNDS — some capped nodes are really wins (or real no-wins) we just couldn't
afford to prove at 250. This reservoir-samples capped nodes per depth from the
append-only log and re-solves them at higher budgets to estimate the flip rates.

Interpretation by parity: a capped node at an ODD depth is black-to-move (flip to
win = a real black VCT we missed); at an EVEN depth it is white-to-move (flip to
win = a real white VCT = black-fumble loss we missed).

Run (from worktree root, GPU; only when no live tenant is using it):
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python -m scripts.idx2_vct.probe_capped \
      --log ~/data/idx2_solve/run-a/nodes.jsonl --depths 7 8 9 10 11 --k 3000 \
      --budgets 250 1000 4000
"""
from __future__ import annotations

import os

os.environ.setdefault("GOMOKU_BOARD_SIZE", "15")

import argparse
import json
import random
from pathlib import Path

import numpy as np

from gomoku.eval_panel import IDX2_OPENING, fixed_opening_state
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb


def board_from_moves(moves) -> np.ndarray:
    s = fixed_opening_state(IDX2_OPENING)
    for a in moves:
        s = s.apply(int(a))
    return s.board.astype(bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[7, 8, 9, 10, 11])
    ap.add_argument("--k", type=int, default=3000, help="reservoir sample per depth")
    ap.add_argument("--budgets", type=int, nargs="+", default=[250, 1000, 4000])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log = Path(os.path.expanduser(args.log))
    depths = set(args.depths)
    rng = random.Random(args.seed)
    reservoir: dict[int, list] = {d: [] for d in depths}
    seen: dict[int, int] = {d: 0 for d in depths}

    # Stream the log; substring-prefilter before json.loads (most lines aren't capped).
    with log.open() as f:
        for line in f:
            if '"verdict": "capped"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = r.get("depth")
            if d not in depths:
                continue
            seen[d] += 1
            res = reservoir[d]
            if len(res) < args.k:
                res.append(r["moves"])
            else:
                j = rng.randint(0, seen[d] - 1)
                if j < args.k:
                    res[j] = r["moves"]

    print(f"\ncapped re-solve probe  (sample k={args.k}/depth)\n")
    hdr = "depth  stm   capped_total  sample  " + "  ".join(
        f"b={b}:win/noVCT/cap" for b in args.budgets)
    print(hdr)
    for d in sorted(depths):
        moves_list = reservoir[d]
        if not moves_list:
            print(f"{d:>5}  (no capped sampled)")
            continue
        boards = np.stack([board_from_moves(m) for m in moves_list])
        stm = "white" if d % 2 == 0 else "black"
        cells = []
        for b in args.budgets:
            win, hit = solve_vct_mega_bb(boards, max_nodes=b)
            n = len(win)
            w = int(win.sum())
            cap = int((hit & ~win).sum())
            novct = n - w - cap
            cells.append(f"{100*w/n:4.1f}/{100*novct/n:4.1f}/{100*cap/n:4.1f}")
        print(f"{d:>5}  {stm:>5}  {seen[d]:>12,}  {len(moves_list):>6}  "
              + "   ".join(cells))
    print("\n(percentages of the sampled capped set: win = real VCT for the "
          "side to move; noVCT = decided no-win; cap = still inconclusive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
