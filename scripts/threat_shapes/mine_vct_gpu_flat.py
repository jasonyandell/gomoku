"""SLOPPY flat-batch GPU VCT-backward miner (Jason's idea).

The level-synchronized walk (mine_vct_gpu.py) early-stops cleverly but SERIALIZES
into ~one GPU call per depth-step (~25 calls, gated by the deepest game). The
megakernel is TAIL-BOUND — 16k positions cost ~the same ~16s as 1 — so the clever
early-stop is a pessimization. Sloppy wins: throw EVERY winner-to-move position of
a chunk of games into ONE big GPU batch, solve all at once, then per game read off
the contiguous VCT-won suffix from the end -> the enabling shape. One ~16s call
replaces the whole walk; first flush in ~20s instead of ~8min.

We "waste" GPU on early-game positions that are obviously lost — but they ride free
in the tail-bound batch. Per emitted shape we still do one CHEAP CPU solve_vct to
recover the catalyst move + mate distance (the GPU gives only a win/no-win verdict).

Run: GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_vct_gpu_flat \
       --games-dir <games_rapfi> --out <vct_gpu_flat> --once
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
from scripts.threat_shapes.mine_vct_gpu import reconstruct, _occupied
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb
from gomoku.vcf import solve_vct

GPU_BATCH = 16384  # megakernel saturates here; sub-batch larger flattenings


def _process(games, args, rec_path, log, stats):
    t0 = time.time()
    flat, index = [], []
    for gi, g in enumerate(games):
        for k, b in enumerate(g.wboards):
            flat.append(b)
            index.append((gi, k))
    boards = np.stack(flat)
    M = len(boards)
    wins = np.empty(M, dtype=bool)
    for s in range(0, M, GPU_BATCH):
        e = min(s + GPU_BATCH, M)
        w, _hit = solve_vct_mega_bb(boards[s:e], max_nodes=args.max_nodes, tg=args.tg)
        wins[s:e] = np.asarray(w, dtype=bool)
    t_gpu = time.time() - t0

    per_game = [[False] * len(g.wboards) for g in games]
    for (gi, k), wv in zip(index, wins):
        per_game[gi][k] = bool(wv)

    rows = []
    for gi, g in enumerate(games):
        wv = per_game[gi]
        if not wv or not wv[0]:            # near-win position not even proven won
            stats["no_forced_at_end"] += 1
            continue
        k_max = 0                          # contiguous won suffix from the end (k=0)
        while k_max + 1 < len(wv) and wv[k_max + 1]:
            k_max += 1
        forced_run = k_max + 1
        stats[f"run_{forced_run}"] += 1
        if forced_run < args.min_run:
            stats["below_min_run"] += 1
            continue
        board = g.wboards[k_max]
        res = solve_vct(board, max_depth=args.extract_depth,
                        max_nodes=args.extract_nodes)
        if not res.has_forced_win:
            stats["extract_miss"] += 1
        move = int(res.winning_move) if res.winning_move is not None else -1
        md = int(res.mate_distance) if res.mate_distance is not None else -1
        stats[f"md_{md}"] += 1
        rows.append({
            "atk": _occupied(board[0]), "dfd": _occupied(board[1]),
            "move": move, "md": md, "run": int(forced_run),
            "winner": int(g.winner), "game_len": int(g.T),
            "ply": int(g.L - 2 * k_max),
        })
        stats["emitted"] += 1

    if rows:
        with gzip.open(rec_path, "at") as rf:
            for r in rows:
                rf.write(json.dumps(r) + "\n")
    dt = time.time() - t0
    log(f"[flat] {len(games)}g {M}pos gpu={t_gpu:.1f}s total={dt:.1f}s "
        f"emit={len(rows)} miss={stats.get('extract_miss', 0)} "
        f"cum_emit={stats['emitted']} ({len(games)/dt:.1f} g/s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-run", type=int, default=3)
    ap.add_argument("--max-nodes", type=int, default=1500)
    ap.add_argument("--tg", type=int, default=32)
    ap.add_argument("--extract-depth", type=int, default=10)
    ap.add_argument("--extract-nodes", type=int, default=3000)
    ap.add_argument("--games-per-flush", type=int, default=700)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--sleep", type=float, default=60.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rec_path = os.path.join(args.out, "enable_serial.jsonl.gz")
    manifest_path = os.path.join(args.out, "manifest.txt")
    log_path = os.path.join(args.out, "gpu_flat.log")

    def log(m):
        with open(log_path, "a") as f:
            f.write(m + "\n")
        print(m, flush=True)

    processed = set()
    if os.path.exists(manifest_path):
        processed = {l.strip() for l in open(manifest_path) if l.strip()}

    stats: Counter = Counter()
    log(f"[flat] start pid={os.getpid()} mn={args.max_nodes} "
        f"extract={args.extract_depth}/{args.extract_nodes} "
        f"gpf={args.games_per_flush} (processed {len(processed)} shards)")

    def flush_shards(shard_names):
        for b in shard_names:
            with open(manifest_path, "a") as mf:
                mf.write(b + "\n")
            processed.add(b)

    while True:
        shards = [p for p in iter_shard_paths(args.games_dir)
                  if os.path.basename(p) not in processed]
        if not shards:
            if args.once:
                break
            log(f"[flat] idle: no new shards (cum_emit={stats['emitted']}); "
                f"sleeping {args.sleep:.0f}s")
            time.sleep(args.sleep)
            continue

        pend_games, pend_shards = [], []
        for sp in shards:
            base = os.path.basename(sp)
            for g in iter_games(args.games_dir, [sp]):
                stats["games"] += 1
                w = g.get("winner", -1)
                if w == -1:
                    stats["draw"] += 1
                    continue
                stats["decisive"] += 1
                gm = reconstruct(g["moves"], int(w), stats)
                if gm is not None and gm.wboards:
                    pend_games.append(gm)
            pend_shards.append(base)
            if len(pend_games) >= args.games_per_flush:
                _process(pend_games, args, rec_path, log, stats)
                flush_shards(pend_shards)
                pend_games, pend_shards = [], []
        if pend_games:
            _process(pend_games, args, rec_path, log, stats)
            flush_shards(pend_shards)
        if args.once:
            break

    log(f"[flat] EXIT games={stats['games']} emitted={stats['emitted']}")


if __name__ == "__main__":
    main()
