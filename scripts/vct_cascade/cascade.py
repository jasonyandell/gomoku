"""Stage 2 — durable iterative-deepening VCT cascade (issue #97).

The disk-spilled twin of ``solve_vct_streaming``: a ladder of node budgets, where
each rung solves the previous rung's still-capped survivors at one ``max_nodes``,
writes a FULL verdict ledger row per board (win / no_win / cap), and spills the
still-capped tail as the next rung's input. Every result is recorded explicitly
(no "absence = state"), so a single DuckDB query yields "X% solved by N nodes,
Z% still cap at the ceiling".

Properties:
  * Append-only + trivially resumable. The ledger IS the state: resume = count
    rows already in ``results/cap<N>/`` and skip that many input boards (input is
    read in a deterministic order). No global in-RAM id set -> scales to billions.
  * Crash-loss <= one dispatch (the kernel writes nothing until it returns; we
    write results LAST, so an orphaned survivor shard is overwritten on redo).
  * Per-rung width auto-tuner: ramp the batch (x2 from --ramp-start) while
    boards/s keeps climbing; lock the knee. Every dispatch's throughput is logged
    to ``perf/`` so the knees are a query, not a guess.

Outputs ON: move + support + carriers + w (proof extraction, no extra search).
complete=OFF (greedy first proof; no full winning-move enumeration).

Run:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_cascade.cascade \
      --out ~/data/raphi_vct --ladder 50,100,250,500,1000,2000,4000,10000
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
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb

_RES_SCHEMA = pa.schema([
    ("id", pa.binary(16)),
    ("verdict", pa.string()),
    ("win_move", pa.int16()),
    ("support", pa.list_(pa.uint64(), 4)),
    ("carriers", pa.list_(pa.uint64(), 4)),
    ("w", pa.list_(pa.uint64(), 4)),
    ("cap", pa.int32()),
])
_PERF_SCHEMA = pa.schema([
    ("cap", pa.int32()), ("batch_size", pa.int32()), ("n_boards", pa.int32()),
    ("wall_s", pa.float64()), ("boards_per_s", pa.float64()),
    ("n_win", pa.int32()), ("n_cap", pa.int32()), ("n_nowin", pa.int32()),
    ("ramp", pa.bool_()), ("ts", pa.float64()),
])


def _total_rows(files: list[str]) -> int:
    return sum(pq.read_metadata(f).num_rows for f in files)


def _list_array(col) -> pa.Array:
    """Table column (ChunkedArray) -> single contiguous ListArray."""
    return col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col


def _fsl(words: np.ndarray) -> pa.Array:
    """(B,4) uint64 -> Arrow fixed-size-list<uint64,4>."""
    return pa.FixedSizeListArray.from_arrays(
        pa.array(words.reshape(-1).astype(np.uint64)), 4)


class Batcher:
    """Streams (id, atk, dfd) rows from a deterministic file order, skipping the
    first ``skip`` rows (resume), serving contiguous batches of <= W rows."""

    def __init__(self, files: list[str], skip: int):
        self.files = files
        self.fi = 0
        self.skip = skip
        self.buf: pa.Table | None = None

    def _next_table(self) -> pa.Table | None:
        while self.fi < len(self.files):
            t = pq.read_table(self.files[self.fi], columns=["id", "atk", "dfd"])
            self.fi += 1
            if self.skip:
                if self.skip >= t.num_rows:
                    self.skip -= t.num_rows
                    continue
                t = t.slice(self.skip)
                self.skip = 0
            if t.num_rows:
                return t
        return None

    def take(self, W: int) -> pa.Table | None:
        parts = [] if self.buf is None else [self.buf]
        self.buf = None
        have = sum(p.num_rows for p in parts)
        while have < W:
            t = self._next_table()
            if t is None:
                break
            parts.append(t)
            have += t.num_rows
        if not parts or have == 0:
            return None
        full = pa.concat_tables(parts).combine_chunks()
        if full.num_rows > W:
            self.buf = full.slice(W)
            return full.slice(0, W)
        return full


def _best_width_from_perf(perf_dir: str, cap: int, default: int) -> int | None:
    files = sorted(glob.glob(os.path.join(perf_dir, f"cap{cap}-*.parquet")))
    if not files:
        return None
    t = pa.concat_tables([pq.read_table(f) for f in files])
    rates = t.column("boards_per_s").to_numpy()
    sizes = t.column("batch_size").to_numpy()
    return int(sizes[int(np.argmax(rates))])


def _solve(boards: np.ndarray, cap: int):
    return solve_vct_mega_bb(
        boards, max_nodes=cap,
        return_move=True, return_support=True,
        return_carriers=True, return_w=True)


def run_rung(cap, in_files, res_dir, surv_dir, perf_dir, args) -> None:
    n_in = _total_rows(in_files)
    if n_in == 0:
        print(f"[cap{cap}] no input — rung empty")
        return
    done = _total_rows(sorted(glob.glob(os.path.join(res_dir, "*.parquet"))))
    if done >= n_in:
        print(f"[cap{cap}] already complete ({done:,}/{n_in:,})")
        return
    watchdog = args.watchdog_low if cap <= args.watchdog_split else args.watchdog_high
    # resume mid-rung: skip the ramp, reuse the best measured width
    resumed_W = _best_width_from_perf(perf_dir, cap, args.ramp_start) if done else None
    locked = resumed_W is not None
    W = resumed_W if locked else args.ramp_start
    print(f"[cap{cap}] {n_in:,} in · resume@{done:,} · "
          f"{'locked W=%d' % W if locked else 'ramp from %d' % W} · watchdog={watchdog}s")

    batcher = Batcher(in_files, skip=done)
    offset = done
    best_rate, best_W, prev_rate = 0.0, W, 0.0
    t_rung = time.time()
    while True:
        start = offset
        tbl = batcher.take(W)
        if tbl is None:
            break
        B = tbl.num_rows
        atk = _list_array(tbl.column("atk"))
        dfd = _list_array(tbl.column("dfd"))
        boards = boards_from_lists(atk, dfd, B)

        t0 = time.time()
        win, hit, move, support, carriers, w = _solve(boards, cap)
        wall = time.time() - t0
        rate = B / max(wall, 1e-9)

        verdict = np.where(win, "win", np.where(hit, "cap", "no_win"))
        n_win = int(win.sum())
        n_cap = int((hit & ~win).sum())
        n_nowin = B - n_win - n_cap

        # --- write results LAST so it is the sole done-marker -------------------
        cap_mask = hit & ~win
        if cap_mask.any():
            sidx = np.flatnonzero(cap_mask)
            surv = pa.table({
                "id": tbl.column("id").take(pa.array(sidx)),
                "atk": atk.take(pa.array(sidx)),
                "dfd": dfd.take(pa.array(sidx)),
            })
            write_parquet_atomic(surv, os.path.join(surv_dir, f"part-{start:012d}.parquet"))
        perf = pa.table({
            "cap": [cap], "batch_size": [W], "n_boards": [B], "wall_s": [wall],
            "boards_per_s": [rate], "n_win": [n_win], "n_cap": [n_cap],
            "n_nowin": [n_nowin], "ramp": [not locked], "ts": [time.time()],
        }, schema=_PERF_SCHEMA)
        write_parquet_atomic(perf, os.path.join(perf_dir, f"cap{cap}-{start:012d}.parquet"))
        res = pa.table({
            "id": tbl.column("id"),
            "verdict": pa.array(verdict, pa.string()),
            "win_move": pa.array(move.astype(np.int16)),
            "support": _fsl(support), "carriers": _fsl(carriers), "w": _fsl(w),
            "cap": pa.array(np.full(B, cap, np.int32)),
        }, schema=_RES_SCHEMA)
        write_parquet_atomic(res, os.path.join(res_dir, f"part-{start:012d}.parquet"))

        offset += B
        frac = (offset - done) / max(n_in - done, 1)
        eta = (time.time() - t_rung) / max(frac, 1e-9) * (1 - frac)
        print(f"[cap{cap}] {offset:,}/{n_in:,} ({100*offset/n_in:4.1f}%) · W={W:,} · "
              f"{rate:,.0f} b/s · +{n_win} win +{n_cap} cap +{n_nowin} nowin · "
              f"{wall:.1f}s · ETA {eta/60:.0f}m", flush=True)

        if not locked:
            if rate > best_rate:
                best_rate, best_W = rate, W
            grow = rate > prev_rate * 1.05 and wall < watchdog and W * 2 <= args.max_width
            prev_rate = rate
            if grow:
                W *= 2
            else:
                locked, W = True, best_W
                print(f"[cap{cap}] >> locked width {W:,} ({best_rate:,.0f} b/s)", flush=True)
    print(f"[cap{cap}] rung done in {(time.time()-t_rung)/60:.1f}m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="root with positions/; writes results/ survivors/ perf/")
    ap.add_argument("--positions", default=None, help="override positions glob (default <out>/positions)")
    ap.add_argument("--ladder", default="50,100,250,500,1000,2000,4000,10000")
    ap.add_argument("--ramp-start", type=int, default=2048)
    ap.add_argument("--max-width", type=int, default=8_388_608)
    ap.add_argument("--watchdog-low", type=float, default=60.0)
    ap.add_argument("--watchdog-high", type=float, default=180.0)
    ap.add_argument("--watchdog-split", type=int, default=1000)
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    ladder = [int(x) for x in args.ladder.split(",")]
    pos_glob = (os.path.expanduser(args.positions) if args.positions
                else os.path.join(out, "positions", "*.parquet"))
    in_files = sorted(glob.glob(pos_glob))
    if not in_files:
        raise SystemExit(f"no positions at {pos_glob} — run extract first")
    perf_dir = os.path.join(out, "perf")

    for i, cap in enumerate(ladder):
        res_dir = os.path.join(out, "results", f"cap{cap}")
        surv_dir = os.path.join(out, "survivors", f"cap{cap}")
        run_rung(cap, in_files, res_dir, surv_dir, perf_dir, args)
        in_files = sorted(glob.glob(os.path.join(surv_dir, "*.parquet")))
        if not in_files:
            print(f"[cascade] no survivors after cap{cap} — all positions resolved")
            break
    print("[cascade] complete")


if __name__ == "__main__":
    main()
