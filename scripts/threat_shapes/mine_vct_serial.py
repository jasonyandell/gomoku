"""Dead-simple SERIAL VCT-backward miner — one game at a time.

No multiprocessing, no shard-flush latency, no cleverness. Streams ONE log line
per game so you can watch it work. Jason's framing: each game starts from a WON
position (forced fours ahead), so the backward walk to the enabling shape is
usually 0-2 steps -> most games should be FAST; the per-game timings prove it.

For each decisive game it calls the (already-correctness-proven) ``mine_game``
from mine_vct_backward.py, which walks the winner's to-move positions backward
while forced and returns the enabling shape (or None). Emitted shapes are
appended immediately to an append-only gzipped JSONL (crash-survivable, compact:
boards stored as occupied-cell index lists). A manifest of processed input
shards makes restarts resume where they left off.

Run (overnight, until killed):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_vct_serial \
      --games-dir <games_rapfi> --out <vct_serial> --max-nodes 4000
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter

import numpy as np

from scripts.threat_shapes.read_games import iter_shard_paths, iter_games
from scripts.threat_shapes.mine_vct_backward import mine_game


def _occupied(plane: np.ndarray) -> list[int]:
    """Flat indices of set cells in a (N,N) bool plane -- compact storage."""
    return [int(i) for i in np.flatnonzero(plane.reshape(-1))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-run", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--max-nodes", type=int, default=4000)
    ap.add_argument("--sleep", type=float, default=60.0,
                    help="seconds to sleep when no new shards (then rescan)")
    ap.add_argument("--once", action="store_true",
                    help="single pass over current shards, then exit")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "serial.log")
    manifest_path = os.path.join(args.out, "serial_manifest.txt")
    rec_path = os.path.join(args.out, "enable_serial.jsonl.gz")

    def log(msg: str) -> None:
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    processed: set[str] = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            processed = {ln.strip() for ln in f if ln.strip()}

    stats: Counter = Counter()
    gi = emitted = 0
    t_start = time.time()
    log(f"[serial] start pid={os.getpid()} pgid={os.getpgrp()} "
        f"cap={args.max_nodes} depth={args.max_depth} min_run={args.min_run} "
        f"(already processed {len(processed)} shards)")

    while True:
        shards = [p for p in iter_shard_paths(args.games_dir)
                  if os.path.basename(p) not in processed]
        if not shards:
            if args.once:
                break
            el = time.time() - t_start
            gps = gi / el if el else 0
            log(f"[serial] idle: no new shards. games={gi} emitted={emitted} "
                f"({gps:.1f} g/s avg) sleeping {args.sleep:.0f}s")
            time.sleep(args.sleep)
            continue

        for sp in shards:
            base = os.path.basename(sp)
            s_emit = s_games = 0
            for g in iter_games(args.games_dir, [sp]):
                gi += 1
                s_games += 1
                w = g.get("winner", -1)
                if w == -1:
                    stats["draw"] += 1
                    log(f"g#{gi} {base} len={len(g['moves'])} draw")
                    continue
                t0 = time.time()
                rec = mine_game(g["moves"], int(w), min_run=args.min_run,
                                max_depth=args.max_depth,
                                max_nodes=args.max_nodes, stats=stats)
                dt = time.time() - t0
                if rec is not None:
                    emitted += 1
                    s_emit += 1
                    board = rec["board"]
                    row = {
                        "atk": _occupied(board[0]), "dfd": _occupied(board[1]),
                        "move": rec["move"], "md": rec["md"],
                        "run": rec["run_plies"], "winner": rec["winner"],
                        "game_len": rec["game_len"], "ply": rec["ply"],
                    }
                    with gzip.open(rec_path, "at") as rf:
                        rf.write(json.dumps(row) + "\n")
                log(f"g#{gi} {base} len={len(g['moves'])} "
                    f"run={rec['run_plies'] if rec else '-'} "
                    f"emit={1 if rec else 0} {dt:.2f}s")
            processed.add(base)
            with open(manifest_path, "a") as mf:
                mf.write(base + "\n")
            log(f"[shard done] {base} games={s_games} emitted={s_emit} "
                f"| total games={gi} emitted={emitted}")

        if args.once:
            break

    el = time.time() - t_start
    log(f"[serial] EXIT games={gi} emitted={emitted} elapsed={el:.0f}s")


if __name__ == "__main__":
    main()
