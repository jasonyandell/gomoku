#!/usr/bin/env python
"""Reference consumer for the append-only game shards.

Reads COMPLETED shards (`games_w*_*.jsonl[.gz]`) from a collection dir while a
producer keeps appending new shards. Safe to run concurrently with
`collect_games.py`: shards appear atomically (temp + os.replace), so a shard a
reader sees is always whole. `.tmp` files (in-flight) are ignored.

Usage:
  # one-shot tally of everything written so far
  python scripts/threat_shapes/read_games.py --dir <games_dir> --stats

  # stream games as dicts (e.g. for the stage-2 VCT-backward pass):
  #   for g in iter_games(<dir>): g["moves"], g["winner"], g["bs"]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os


def iter_shard_paths(games_dir):
    paths = glob.glob(os.path.join(games_dir, "games_w*_*.jsonl")) + \
            glob.glob(os.path.join(games_dir, "games_w*_*.jsonl.gz"))
    return sorted(p for p in paths if not p.endswith(".tmp"))


def _open(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def iter_games(games_dir, shard_paths=None):
    for path in (shard_paths or iter_shard_paths(games_dir)):
        try:
            with _open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        except (OSError, EOFError, ValueError):
            # a shard being replaced right now; skip, it'll be read next pass
            continue


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    shards = iter_shard_paths(args.dir)
    n = npos = decisive = 0
    wins = {0: 0, 1: 0, -1: 0}
    for g in iter_games(args.dir, shards):
        n += 1
        npos += len(g["moves"])
        w = g["winner"]
        wins[w] = wins.get(w, 0) + 1
        if w != -1:
            decisive += 1
        if args.limit and n >= args.limit:
            break
    if args.stats or True:
        print(f"shards={len(shards)} games={n} positions={npos}")
        if n:
            print(f"  decisive={decisive} ({decisive / n:.3f})  "
                  f"black_wins={wins.get(0,0)} white_wins={wins.get(1,0)} "
                  f"draws={wins.get(-1,0)}")
            print(f"  avg_plies={npos / n:.1f}")


if __name__ == "__main__":
    main()
