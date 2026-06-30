"""Supervised throughput sweep — boards/s at different WIDTHS for one node DEPTH.

NOT the labeler. This is the empirical characterization step: pick a node budget,
try a list of batch widths, and for each width run dispatches for up to
``--secs-per-width`` seconds, recording every dispatch's throughput to ``perf/``
(tagged ``sweep``). Bounded so a single invocation is a ~1-minute observation you
can read, then move the width/depth and go again.

Boards come from any position parquet (``--positions``); the same pool is reused
across widths (re-served from the front each width), so the only variable is the
width. Verdicts here are throwaway (characterization, not the ledger) — use the
cascade for durable labeling once the knees are known.

Run (one ~1-min observation):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_cascade.sweep \
      --positions '~/data/raphi_vct/positions/*.parquet' --cap 50 \
      --widths 8192,65536,262144,1048576,4194304 --secs-per-width 10 \
      --out ~/data/raphi_vct
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.vct_cascade.common import boards_from_lists, write_parquet_atomic
from scripts.vct_cascade.cascade import _PERF_SCHEMA, _list_array, _solve


def _load_pool(pos_glob: str, cap_boards: int) -> tuple[pa.Array, pa.Array, int]:
    """Load up to ``cap_boards`` (id-less) atk/dfd columns into memory once."""
    files = sorted(glob.glob(os.path.expanduser(pos_glob)))
    if not files:
        raise SystemExit(f"no positions at {pos_glob}")
    atks, dfds, n = [], [], 0
    for f in files:
        t = pq.read_table(f, columns=["atk", "dfd"])
        atks.append(_list_array(t.column("atk")))
        dfds.append(_list_array(t.column("dfd")))
        n += t.num_rows
        if n >= cap_boards:
            break
    atk = pa.concat_arrays(atks)
    dfd = pa.concat_arrays(dfds)
    return atk, dfd, len(atk)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", required=True, help="position parquet glob")
    ap.add_argument("--out", required=True, help="root; writes perf/ here")
    ap.add_argument("--cap", type=int, required=True, help="node budget (depth)")
    ap.add_argument("--widths", required=True, help="comma list of batch widths")
    ap.add_argument("--secs-per-width", type=float, default=10.0)
    ap.add_argument("--pool", type=int, default=0,
                    help="max boards to load (0 = max width)")
    args = ap.parse_args()

    widths = [int(x) for x in args.widths.split(",")]
    pool_cap = args.pool or max(widths)
    atk_all, dfd_all, npool = _load_pool(args.positions, pool_cap)
    perf_dir = os.path.join(os.path.expanduser(args.out), "perf")
    print(f"[sweep] cap={args.cap} · pool={npool:,} boards · "
          f"widths={widths} · {args.secs_per_width}s/width")

    for W in widths:
        if W > npool:
            print(f"[sweep] cap={args.cap} W={W:,} > pool {npool:,} — skip")
            continue
        atk = atk_all.slice(0, W)
        dfd = dfd_all.slice(0, W)
        boards = boards_from_lists(atk, dfd, W)
        # warm-up (compile / first-touch) is excluded from the recorded rate
        t_end = time.time() + args.secs_per_width
        rates, first = [], True
        while time.time() < t_end:
            t0 = time.time()
            win, hit, move, support, carriers, w = _solve(boards, args.cap)
            wall = time.time() - t0
            rate = W / max(wall, 1e-9)
            n_win = int(win.sum()); n_cap = int((hit & ~win).sum())
            perf = pa.table({
                "cap": [args.cap], "batch_size": [W], "n_boards": [W],
                "wall_s": [wall], "boards_per_s": [rate],
                "n_win": [n_win], "n_cap": [n_cap], "n_nowin": [W - n_win - n_cap],
                "ramp": [True], "ts": [time.time()],
            }, schema=_PERF_SCHEMA)
            write_parquet_atomic(
                perf, os.path.join(perf_dir, f"sweep-cap{args.cap}-W{W}-{int(t0*1e3)}.parquet"))
            if not first:
                rates.append(rate)
            first = False
        best = max(rates) if rates else rate
        avg = sum(rates) / len(rates) if rates else rate
        print(f"[sweep] cap={args.cap:>5} W={W:>9,} · {len(rates)+1} disp · "
              f"avg {avg:>12,.0f} b/s · best {best:>12,.0f} b/s · {wall:.2f}s/disp",
              flush=True)


if __name__ == "__main__":
    main()
