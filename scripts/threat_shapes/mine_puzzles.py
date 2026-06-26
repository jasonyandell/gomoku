"""Puzzle miner — for EVERY game, EVERY ply: does the side-to-move have a forced mate?

Not minimization, not first/last, not the move — just a verdict per position, to *see the
space*. Each kept row is a **puzzle**: "X/O to move, there is a forced VCT — where is it?"
(we don't answer; we collect the boards). The GPU solver gives only `(win, hit_cap)` — a
verdict and "ran out of time" — and that's all we record.

Batch HARD + low budget (per the CALL-COST LAW, gpu-vct-feasibility.md): flatten every ply of
a chunk of games into one ~16k-board call at a low `max_nodes` (default 500). Most positions
come back null (no win) — fine, they're simply not written. Three outcomes per ply:
  win=True              -> a PUZZLE (written, with the board + whose turn).
  win=False, cap=True   -> UNKNOWN at this budget -> written too, a candidate for a later
                           DEEPER run (filter cap==true and re-mine at a higher --max-nodes).
  win=False, cap=False  -> proven no-win -> NOT written (the bulk; nulls are absent by design).

Append-only + trivially resumable: a game shard is solved in full, its non-null rows appended
to `puzzles.jsonl.gz`, then its basename recorded in `manifest.txt`; a restart skips done
shards. Inputs are recorded per row (shard/idx/ply + the board), so every puzzle is
self-contained and reproducible.

Row schema (`puzzles.jsonl.gz`):
  {"shard","idx","ply","stm","win","cap","winner","atk":[idx...],"dfd":[idx...]}
  atk/dfd = occupied cells of board[0]/board[1] (0..N*N-1 row-major); board[0] is the
  SIDE-TO-MOVE (attacker) frame at this ply (side-to-move-relative, no swap). stm = ply%2
  (0 = first player). winner = the game's winner (-1 = draw), for context only.

Run (full sweep over all games):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_puzzles \
      --games-dir ~/data/games_raphi --out ~/data/puzzle_miner --max-nodes 500

Deeper re-mine of the unknowns (later):
  filter `cap==true` rows out of puzzles.jsonl.gz, then re-solve those boards at a higher cap.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from scripts.threat_shapes.read_games import iter_shard_paths, iter_games
from scripts.threat_shapes.mine_first_vct import all_boards, _occupied
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb

GPU_BATCH = 16384  # one saturating call; the tail (one hardest board) is amortized over all of it


def _process(games, args, out_path, log, stats):
    """games: list of [shard, idx, moves, winner]. Flatten EVERY ply, solve once (sub-batched),
    append every non-null (win or cap) row."""
    t0 = time.time()
    flat, index, boards_of = [], [], {}
    for gi, (_sh, _ix, moves, _w) in enumerate(games):
        bs = all_boards(moves)
        if bs is None:
            stats["too_short"] += 1
            continue
        boards_of[gi] = bs
        for p, b in enumerate(bs):
            flat.append(b)
            index.append((gi, p))
    if not flat:
        return
    boards = np.stack(flat)
    M = len(boards)
    win = np.empty(M, dtype=bool)
    cap = np.empty(M, dtype=bool)
    for s in range(0, M, GPU_BATCH):
        e = min(s + GPU_BATCH, M)
        w, h = solve_vct_mega_bb(boards[s:e], max_nodes=args.max_nodes, tg=args.tg)
        win[s:e] = np.asarray(w, dtype=bool)
        cap[s:e] = np.asarray(h, dtype=bool)
    t_gpu = time.time() - t0

    rows = []
    for (gi, p), wv, cv in zip(index, win, cap):
        stats["positions"] += 1
        if not wv and not cv:
            stats["null"] += 1
            continue                      # proven no-win -> not written (the bulk)
        b = boards_of[gi][p]
        shard, idx, _moves, winner = games[gi]
        rows.append({
            "shard": shard, "idx": int(idx), "ply": int(p), "stm": int(p % 2),
            "win": bool(wv), "cap": bool(cv), "winner": int(winner),
            "atk": _occupied(b[0]), "dfd": _occupied(b[1]),
        })
        stats["win" if wv else "cap"] += 1

    if rows:
        with gzip.open(out_path, "at") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    dt = time.time() - t0
    log(f"[puzzle] {len(games)}g {M}pos gpu={t_gpu:.1f}s total={dt:.1f}s "
        f"win={stats['win']} cap={stats['cap']} null={stats['null']} "
        f"({M/max(dt,1e-9):.0f} pos/s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-nodes", type=int, default=500, help="low by design; CAP rows re-mined deeper later")
    ap.add_argument("--tg", type=int, default=32)
    ap.add_argument("--games-per-flush", type=int, default=600, help="~16k+ positions per GPU call")
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all (smoke cap)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "puzzles.jsonl.gz")
    mpath = os.path.join(args.out, "manifest.txt")
    log_path = os.path.join(args.out, "puzzles.log")

    def log(m):
        with open(log_path, "a") as f:
            f.write(m + "\n")
        print(m, flush=True)

    processed = set()
    if os.path.exists(mpath):
        processed = {l.strip() for l in open(mpath) if l.strip()}
    shards = [p for p in iter_shard_paths(args.games_dir)
              if os.path.basename(p) not in processed]
    if args.max_shards:
        shards = shards[:args.max_shards]

    stats: Counter = Counter()
    log(f"[puzzle] start pid={os.getpid()} N={N} mn={args.max_nodes} "
        f"gpf={args.games_per_flush} shards={len(shards)} (skip {len(processed)} done)")

    # Accumulate WHOLE shards into one big batch (~16k positions/call — the CALL-COST LAW: the
    # tail is the same for 4k or 16k boards, so fill it). A shard is marked done only after the
    # flush that consumed it, so resume never skips or splits a shard.
    pend, pend_shards = [], []

    def flush():
        if pend:
            _process(pend, args, out_path, log, stats)
        with open(mpath, "a") as mf:
            for b in pend_shards:
                mf.write(b + "\n")
        stats["shards"] += len(pend_shards)
        pend.clear()
        pend_shards.clear()

    for sp in shards:
        base = os.path.basename(sp)
        for ix, g in enumerate(iter_games(args.games_dir, [sp])):
            pend.append([base, ix, g["moves"], g.get("winner", -1)])
        pend_shards.append(base)
        if len(pend) >= args.games_per_flush:
            flush()
    flush()

    log(f"[puzzle] EXIT shards={stats['shards']} positions={stats['positions']} "
        f"win={stats['win']} cap={stats['cap']} null={stats['null']}")


if __name__ == "__main__":
    main()
