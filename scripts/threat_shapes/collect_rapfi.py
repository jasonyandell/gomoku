#!/usr/bin/env python
"""Strong Rapfi-vs-Rapfi full-game collector — persistent warm engine pool,
all cores, append-only sharded logs. Overnight-robust.

WHY THIS IS THE RIGHT TOOL
--------------------------
`gomoku.rapfi_pool.RapfiPool` pre-spawns `size` PERSISTENT `pbrain-rapfi`
processes once and lends them out warm — `size` engines == `size` true-parallel
OS processes (the engine releases the GIL during the blocking pipe read). So
"persistent engines" + "multi-process core saturation" are already built in;
pool `size` IS the parallelism lever. Each engine is configured
`default_thread_num = 1`, so `size ≈ cores` lights up the box.

MEASURED FACTS (this box, M5 Max, 15x15):
  * Per-move protocol overhead floor (RESTART + BOARD re-dump + pipe) at
    timeout_ms=1 is ~3 ms/move — RESTART does NOT reload the NNUE. So the engine
    is THINK-TIME BOUND, not restart-bound: an incremental-TURN rewrite would
    save <~3% at 100 ms. The pool's stateless BOARD mode is the right tool.
  * Throughput is therefore ~ size / (plies_per_game * timeout_ms). Strong play
    (good shapes) costs games/s; we report 50/100/200 ms so you pick the trade.

ARCHITECTURE
------------
`threads` independent game-driver threads (default = pool size) each loop:
random 2..6-ply opening -> Rapfi plays BOTH sides to a terminal (pool.pick per
move; the same evaluator labels whoever is to move) -> record the full game.
Independent threads (vs a synchronous label_states wave) avoid a per-tick
barrier stall, so every engine stays busy. Each thread owns its OWN shard
stream, so there is zero write contention.

OUTPUT: append-only sharded JSONL (atomic temp + os.replace per shard, the
store.py contract). One line per game:
    {"moves":[<flat cell idx, black first>...], "winner":0|1|-1, "bs":15}
A separate stage-2 consumer reads COMPLETED shards while we append; a SIGKILL
mid-shard never exposes a torn shard (loses only the in-memory buffer).
Read with scripts/threat_shapes/read_games.py (shared with collect_games.py).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from gomoku.game import GameState
from gomoku.rapfi_pool import RapfiPool, rapfi_available

# Reuse the atomic JSONL shard writer + the opening generator already written.
from scripts.threat_shapes.collect_games import ShardWriter
from scripts.threat_shapes.harvest_forced import _random_opening


def play_game(pool, rng, *, max_plies, min_open, max_open):
    """One Rapfi-vs-Rapfi game from a random opening.
    Returns (moves, winner) with winner in {0,1} or (moves,-1) for draw/cap."""
    n_open = int(rng.integers(min_open, max_open + 1))
    moves = _random_opening(rng, n_open)
    s = GameState.initial()
    for m in moves:
        s = s.apply(m)
        done, _ = s.is_terminal()
        if done:  # opening accidentally made five (rare) — discard
            return None
    while len(moves) < max_plies:
        a = int(pool.pick(s, rng))
        s = s.apply(a)
        moves.append(a)
        done, val = s.is_terminal()
        if done:
            if val < 0:                      # the player who just moved won
                return moves, (len(moves) - 1) % 2
            return moves, -1                 # draw (full board)
    return moves, -1                         # hit ply cap, undecided


class _Stop:
    def __init__(self):
        self.flag = False

    def __call__(self, *a):
        self.flag = True


def generate(args, *, bench=False):
    out_dir = args.out
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    threads = args.threads or args.size
    stop = _Stop()
    if not bench:
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
    deadline = (time.time() + args.duration) if args.duration > 0 else None

    counts = [0] * threads          # games per thread
    plies = [0] * threads           # plies per thread
    decisive = [0] * threads
    t0 = time.time()
    progress_path = os.path.join(out_dir, "progress") if out_dir else None

    with RapfiPool(size=args.size, timeout_ms=args.timeout_ms, board_size=N) as pool:

        def worker(wid):
            rng = np.random.default_rng(args.seed + wid * 100003)
            w = ShardWriter(out_dir, wid, args.shard_size,
                            gzip_shards=not args.no_gzip) if out_dir else None
            last_report = time.time()
            try:
                while not stop.flag:
                    if deadline is not None and time.time() >= deadline:
                        break
                    try:
                        res = play_game(pool, rng, max_plies=args.max_plies,
                                        min_open=args.min_open, max_open=args.max_open)
                    except Exception:
                        continue           # transient engine death already self-healed
                    if res is None:
                        continue
                    moves, winner = res
                    counts[wid] += 1
                    plies[wid] += len(moves)
                    if winner != -1:
                        decisive[wid] += 1
                    if w is not None:
                        w.add(moves, winner)
                    now = time.time()
                    if progress_path and now - last_report >= 5.0:
                        last_report = now
                        _write_progress(progress_path, wid, counts[wid],
                                        plies[wid], decisive[wid], now - t0)
            finally:
                if w is not None:
                    w.close()
                    _write_progress(progress_path, wid, counts[wid],
                                    plies[wid], decisive[wid], time.time() - t0)

        ths = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(threads)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()

    elapsed = time.time() - t0
    g = sum(counts)
    p = sum(plies)
    d = sum(decisive)
    return {
        "games": g, "positions": p, "decisive": d,
        "elapsed_s": round(elapsed, 1),
        "games_per_s": round(g / elapsed, 2) if elapsed else 0,
        "positions_per_s": round(p / elapsed, 1) if elapsed else 0,
        "decisive_frac": round(d / g, 3) if g else 0,
        "avg_plies": round(p / g, 1) if g else 0,
        "size": args.size, "threads": threads, "timeout_ms": args.timeout_ms,
    }


def _write_progress(progress_path, wid, games, plies, decisive, elapsed):
    tmp = f"{progress_path}.w{wid}.tmp"
    final = f"{progress_path}.w{wid}"
    try:
        with open(tmp, "w") as f:
            json.dump({"wid": wid, "games": games, "plies": plies,
                       "decisive": decisive, "elapsed": elapsed}, f)
        os.replace(tmp, final)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="output dir for shards (omit with --bench)")
    ap.add_argument("--size", type=int, default=16, help="pool size == busy cores")
    ap.add_argument("--threads", type=int, default=0,
                    help="game-driver threads (0 = pool size)")
    ap.add_argument("--timeout-ms", type=int, default=100, help="Rapfi think ms/move")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds; 0 = until killed (SIGTERM/SIGINT)")
    ap.add_argument("--shard-size", type=int, default=500, help="games per shard")
    ap.add_argument("--min-open", type=int, default=2)
    ap.add_argument("--max-open", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-gzip", action="store_true")
    ap.add_argument("--bench", action="store_true",
                    help="sweep timeout-ms over 50/100/200 for --bench-secs each")
    ap.add_argument("--bench-secs", type=float, default=30.0)
    args = ap.parse_args()

    if not rapfi_available():
        raise SystemExit("Rapfi not available (engines/rapfi build or HF cache missing)")
    print(f"[board {N}x{N}] rapfi pool collector")

    if args.bench:
        for tmo in (50, 100, 200):
            args.timeout_ms = tmo
            args.duration = args.bench_secs
            stats = generate(args, bench=True)
            print(f"  timeout={tmo:4d}ms -> {json.dumps(stats)}")
        return

    if not args.out:
        ap.error("--out required unless --bench")
    print(f"launching size={args.size} threads={args.threads or args.size} "
          f"timeout={args.timeout_ms}ms -> {args.out} "
          f"(duration={args.duration or 'until-killed'})")
    stats = generate(args)
    print("DONE:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
