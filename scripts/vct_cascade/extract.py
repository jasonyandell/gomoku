"""Stage 1 — extract every ply of every game into a content-addressed position set.

Reads the append-only game shards (``games_w*_*.jsonl[.gz]``), replays each game,
and emits one row per ply: ``(id, atk, dfd, shard, game_idx, ply)`` where ``id`` is
the D4-canonical blake2b-16 of the (attacker-relative) board. Output is sharded
Parquet, one file per game-shard — so the run is trivially resumable (a finished
file is skipped) and embarrassingly parallel (one worker per shard, no contention).

Per-shard files still contain cross-shard duplicates; ``dedup`` collapses them into
the canonical ``positions/`` set with DuckDB (one row per id). That deduped set is
the corpus-agnostic input the cascade solves.

Run:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_cascade.extract \
      --games-dir ~/data/games_raphi --out ~/data/raphi_vct --workers 12
  # then collapse duplicates:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_cascade.extract dedup \
      --out ~/data/raphi_vct
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa

from scripts.threat_shapes.read_games import iter_shard_paths, iter_games
from scripts.threat_shapes.mine_first_vct import all_boards
from scripts.vct_cascade.common import canonical_id, occupied, write_parquet_atomic

_SCHEMA = pa.schema([
    ("id", pa.binary(16)),
    ("atk", pa.list_(pa.uint8())),
    ("dfd", pa.list_(pa.uint8())),
    ("shard", pa.string()),
    ("game_idx", pa.int32()),
    ("ply", pa.int16()),
])


def _process_shard(args) -> tuple[str, int, int]:
    path, raw_dir, max_games = args
    base = os.path.basename(path).split(".jsonl")[0]
    out_path = os.path.join(raw_dir, base + ".parquet")
    if os.path.exists(out_path):
        return base, -1, -1  # already done
    ids, atks, dfds, shards, gidxs, plies = [], [], [], [], [], []
    seen: set[bytes] = set()
    n_ply = 0
    for gi, g in enumerate(iter_games(None, [path])):
        if max_games and gi >= max_games:
            break
        boards = all_boards(g["moves"])
        if boards is None:
            continue
        for ply, b in enumerate(boards):
            n_ply += 1
            cid = canonical_id(b)
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            atks.append(occupied(b[0]))
            dfds.append(occupied(b[1]))
            shards.append(base)
            gidxs.append(gi)
            plies.append(ply)
    tbl = pa.table({
        "id": pa.array(ids, pa.binary(16)),
        "atk": pa.array(atks, pa.list_(pa.uint8())),
        "dfd": pa.array(dfds, pa.list_(pa.uint8())),
        "shard": pa.array(shards, pa.string()),
        "game_idx": pa.array(gidxs, pa.int32()),
        "ply": pa.array(plies, pa.int16()),
    }, schema=_SCHEMA)
    write_parquet_atomic(tbl, out_path)
    return base, n_ply, len(ids)


def cmd_extract(args) -> None:
    raw_dir = os.path.join(os.path.expanduser(args.out), "positions_raw")
    os.makedirs(raw_dir, exist_ok=True)
    shards = iter_shard_paths(os.path.expanduser(args.games_dir))
    if args.max_shards:
        shards = shards[: args.max_shards]
    todo = [s for s in shards
            if not os.path.exists(os.path.join(
                raw_dir, os.path.basename(s).split(".jsonl")[0] + ".parquet"))]
    print(f"[extract] {len(shards)} shards, {len(todo)} to do, {args.workers} workers")
    t0 = time.time()
    work = [(s, raw_dir, args.max_games) for s in todo]
    done = ply_tot = uniq_tot = 0
    with Pool(args.workers) as pool:
        for base, n_ply, n_uniq in pool.imap_unordered(_process_shard, work):
            done += 1
            if n_ply >= 0:
                ply_tot += n_ply
                uniq_tot += n_uniq
            if done % 100 == 0 or done == len(todo):
                dt = time.time() - t0
                print(f"[extract] {done}/{len(todo)} shards · {ply_tot:,} plies · "
                      f"{uniq_tot:,} local-uniq · {ply_tot/max(dt,1e-9):,.0f} plies/s · {dt:.0f}s")
    print(f"[extract] done: {ply_tot:,} plies, {uniq_tot:,} pre-global-dedup rows in {raw_dir}")


def cmd_dedup(args) -> None:
    import duckdb
    out = os.path.expanduser(args.out)
    raw_glob = os.path.join(out, "positions_raw", "*.parquet")
    pos_dir = os.path.join(out, "positions")
    os.makedirs(pos_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.workers}")
    n_raw = con.execute(f"SELECT count(*) FROM read_parquet('{raw_glob}')").fetchone()[0]
    print(f"[dedup] {n_raw:,} raw rows -> collapsing by canonical id ...")
    # one row per id (atk/dfd identical within an id by construction; keep first prov)
    con.execute(f"""
        COPY (
            SELECT id, any_value(atk) AS atk, any_value(dfd) AS dfd,
                   any_value(shard) AS shard, any_value(game_idx) AS game_idx,
                   any_value(ply) AS ply
            FROM read_parquet('{raw_glob}')
            GROUP BY id
        ) TO '{pos_dir}' (FORMAT parquet, COMPRESSION zstd,
                          PER_THREAD_OUTPUT true, OVERWRITE_OR_IGNORE true);
    """)
    n_uniq = con.execute(
        f"SELECT count(*) FROM read_parquet('{os.path.join(pos_dir, '*.parquet')}')"
    ).fetchone()[0]
    print(f"[dedup] {n_raw:,} -> {n_uniq:,} unique positions "
          f"({100*n_uniq/max(n_raw,1):.1f}%) in {pos_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="replay games -> per-shard position parquet")
    ex.add_argument("--games-dir", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--workers", type=int, default=12)
    ex.add_argument("--max-shards", type=int, default=0, help="0=all (smoke cap)")
    ex.add_argument("--max-games", type=int, default=0, help="per shard; 0=all (smoke cap)")
    dd = sub.add_parser("dedup", help="collapse per-shard parquet -> unique positions/")
    dd.add_argument("--out", required=True)
    dd.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.cmd == "dedup":
        cmd_dedup(args)
    else:
        cmd_extract(args)


if __name__ == "__main__":
    main()
