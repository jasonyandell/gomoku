"""Attach a winning move to every proven-win puzzle from the puzzle miner.

The miner (``mine_puzzles.py``) records a verdict-only trit per ply; the
``win=True, cap=False`` rows are proven VCT wins but have no move. This pass
batches those boards back through ``solve_vct_mega_bb(return_move=True)`` — which
now emits a VALID forced-win first move per board (sound: the kernel never reports
a false win, so the root move of the line it proves always starts a real forced
win; not necessarily the shortest mate). Any such move is a true VCT solution,
which is all the trainer needs as a label.

Boards are already SIDE-TO-MOVE-RELATIVE in the miner output (board[0]=attacker),
exactly the kernel's input frame — no plane swap. ``move`` is a flat row-major
cell index in the same frame.

Resumable: output rows are written in the same order as the (win,cap=False) input
rows, so a restart counts existing output rows and skips that many inputs.

Run:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.solve_puzzles \
      --puzzles ~/data/puzzle_miner/puzzles.jsonl.gz \
      --out ~/data/puzzle_miner/solutions.jsonl.gz

Output row schema (solutions.jsonl.gz):
  {"shard","idx","ply","stm","winner","atk":[...],"dfd":[...],"move":<flat idx>}
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb

NN = N * N
GPU_BATCH = 16384


def _board(atk, dfd):
    b = np.zeros((2, N, N), dtype=bool)
    for a in atk:
        b[0, a // N, a % N] = True
    for d in dfd:
        b[1, d // N, d % N] = True
    return b


def _count_lines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with gzip.open(path, "rt") as f:
        for _ in f:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", default=os.path.expanduser("~/data/puzzle_miner/puzzles.jsonl.gz"))
    ap.add_argument("--out", default=os.path.expanduser("~/data/puzzle_miner/solutions.jsonl.gz"))
    ap.add_argument("--max-nodes", type=int, default=1000,
                    help="search budget; wins were proven at 500 so this reproduces them")
    args = ap.parse_args()

    already = _count_lines(args.out)
    print(f"resume: {already} solution rows already written; skipping that many win rows",
          flush=True)

    fout = gzip.open(args.out, "at")  # append
    buf_rows, buf_boards = [], []
    seen_wins = 0          # win=True,cap=False rows encountered
    written = already
    solved = capflip = 0
    t0 = time.time()

    def flush():
        nonlocal solved, capflip, written
        if not buf_boards:
            return
        boards = np.stack(buf_boards)
        w, h, mv = solve_vct_mega_bb(boards, max_nodes=args.max_nodes, return_move=True)
        for r, won, cap, m in zip(buf_rows, w, h, mv):
            if won and not cap:
                r["move"] = int(m)
                fout.write(json.dumps(r, separators=(",", ":")) + "\n")
                solved += 1
                written += 1
            else:
                # a row that was win at 500 but not reproduced here (shouldn't happen
                # at >=500 budget); record so nothing is silently dropped.
                capflip += 1
        fout.flush()
        buf_rows.clear()
        buf_boards.clear()

    with gzip.open(args.puzzles, "rt") as fin:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            if not (r["win"] and not r["cap"]):
                continue
            seen_wins += 1
            if seen_wins <= already:        # resume skip
                continue
            buf_rows.append({k: r[k] for k in ("shard", "idx", "ply", "stm", "winner", "atk", "dfd")})
            buf_boards.append(_board(r["atk"], r["dfd"]))
            if len(buf_boards) >= GPU_BATCH:
                flush()
                if solved % (GPU_BATCH * 20) == 0:
                    dt = time.time() - t0
                    print(f"  {written} written  ({solved/max(dt,1e-9):.0f} solved/s)", flush=True)
    flush()
    fout.close()
    dt = time.time() - t0
    print(f"DONE: wrote {written} total ({solved} this run) in {dt:.0f}s; "
          f"non-reproduced={capflip}", flush=True)


if __name__ == "__main__":
    main()
