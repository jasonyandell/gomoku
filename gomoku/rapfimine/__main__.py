"""CLI for the idx-2 Rapfi mine.

    GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine run \\
        --out mined/idx2_15x15 --total 1000000 --workers 30

``status`` prints the on-disk example count for a mine dir.
"""
from __future__ import annotations

import argparse
import os
import sys

from gomoku.board_config import BOARD_SIZE
from gomoku.eval_panel import IDX2_OPENING, fixed_opening_state
from gomoku.rapfi_pool import default_rapfi_cmd
from gomoku.rapfimine.coordinator import run_mine
from gomoku.rapfimine.store import count_examples


def _start_state(opening: str):
    if opening == "idx2":
        return fixed_opening_state(IDX2_OPENING)
    raise SystemExit(f"unknown opening {opening!r} (only 'idx2' supported — "
                     "this is the Bruce-Lee-one-position mine)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="mine until --total canonical examples exist")
    r.add_argument("--out", required=True, help="mine output dir (sharded npz)")
    r.add_argument("--total", type=int, default=1_000_000)
    r.add_argument("--workers", type=int, default=24,
                   help="worker processes = independent Rapfi engines (tune to max CPU)")
    r.add_argument("--expand-k", type=int, default=8, help="BFS children per board")
    r.add_argument("--max-pv", type=int, default=24, help="scored support per board")
    r.add_argument("--max-node", type=int, default=20_000)
    r.add_argument("--timeout-ms", type=int, default=1000)
    r.add_argument("--shard-size", type=int, default=50_000)
    r.add_argument("--opening", default="idx2")
    r.add_argument("--monitor-every-s", type=float, default=5.0)

    s = sub.add_parser("status", help="print on-disk example count for a mine dir")
    s.add_argument("--out", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "status":
        print(f"{args.out}: {count_examples(args.out)} examples on disk")
        return 0

    if BOARD_SIZE != 15:
        print("WARNING: GOMOKU_BOARD_SIZE != 15 — Bruce is 15x15. Set "
              "GOMOKU_BOARD_SIZE=15.", file=sys.stderr)

    cmd = default_rapfi_cmd()
    print(f"[mine] opening={args.opening} board={BOARD_SIZE} workers={args.workers} "
          f"expand_k={args.expand_k} max_pv={args.max_pv} -> {args.out}", flush=True)
    final = run_mine(
        start_state=_start_state(args.opening), out_dir=args.out, total=args.total,
        workers=args.workers, cmd=cmd, board_size=BOARD_SIZE, max_node=args.max_node,
        max_pv=args.max_pv, expand_k=args.expand_k, timeout_ms=args.timeout_ms,
        shard_size=args.shard_size, monitor_every_s=args.monitor_every_s)
    return 0 if final >= args.total else 1


if __name__ == "__main__":
    sys.exit(main())
