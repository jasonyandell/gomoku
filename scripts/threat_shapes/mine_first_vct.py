"""Forward-scan FIRST-VCT miner -- the guaranteed-first VCT in each real game.

The backward miner (``mine_vct_gpu_flat.py``) anchored at the END and read the
contiguous won-suffix -- an engineering shortcut from when proving "no VCT yet" on
every position was unaffordable. It only ever sees the LAST VCT window the game rode
to victory. But VCT-existence is NOT monotone across plies: a forced win can open at
ply k, close at k+2 (the player didn't take it, the opponent's reply blocked it),
and reopen later. Every earlier window is invisible to the backward walk.

With the on-device megakernel (``solve_vct_mega_bb``, tail-bound, 0 FP / 0 FN) we can
afford the honest thing: scan EVERY position from move 0 and take the FIRST ply where
the side-to-move has a forced VCT -- the moment the game was first theoretically over
by continuous threats, whoever was to move, whether or not anyone noticed.

SCOPE: any VCT by anyone. ``GameState.board`` is side-to-move-relative, so at EVERY
ply ``boards_before[p][0]`` is already the to-move/attacker frame (no swap). We scan
all plies, both colors. Decisive games AND draws (a draw with a missed VCT is a real
shape). ``stm != winner`` on an emitted row => a missed win (the L2 fog set).

THE TRIT (why "first" is trustworthy). Per position the kernel gives (win, hit_cap),
which we read as a 3-way verdict: WIN / NO-WIN-proven / CAP (budget exhausted, unknown).
"First" at ply w is only certifiable if EVERY ply < w is *proven* NO-WIN. A CAP in the
prefix might itself be an even-earlier VCT we couldn't prove -- so we cannot trust w.

  clean NO-WIN prefix then WIN  -> emit the first-VCT (extract catalyst move p, anchor).
  CAP anywhere before first WIN -> DEFER the whole game: append it to the defer queue and
                                   come back with a larger budget later. Never a wrong
                                   "first"; we only postpone the hard games. Fail-safe.
  no WIN, no CAP (all NO-WIN)   -> the game has no VCT at this budget (rare for decisive).

APPEND-ONLY / RESUMABLE. Three artifacts in ``--out``, all append-only so a re-run
trivially resumes:
  first_vct.jsonl.gz  -- one JSON row per certified first-VCT (the corpus).
  defer.jsonl.gz      -- games set aside on a prefix CAP (retry at higher --max-nodes).
  manifest.txt        -- basenames of game shards fully processed (resume marker).
Row schema:
  {atk:[idx], dfd:[idx], move, md, ply, game_len, stm, winner, shard, idx}
  atk/dfd = occupied cells of board[0]/board[1] (0..N*N-1 row-major); move/md = the
  catalyst VCT move + mate distance (anchor for minimization), -1 if extraction missed.

Run (small smoke / a few games):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_first_vct \
      --games-dir <games_rapfi> --out <vct_first> --once --max-shards 1 --max-games 20

Resolve the deferred pile later at a fatter budget (separate pass):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_first_vct \
      --resolve-defers <vct_first>/defer.jsonl.gz --out <vct_first2> --max-nodes 20000
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
from gomoku.game import GameState
from gomoku.vcf import solve_vct
from scripts.threat_shapes.read_games import iter_shard_paths, iter_games
from scripts.vct_metal.mega_vct_bb import MAXD, solve_vct_mega_bb

GPU_BATCH = 16384  # megakernel saturates here; sub-batch any larger flattening


def _occupied(plane: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(plane.reshape(-1))]


def all_boards(moves: list[int]) -> list[np.ndarray] | None:
    """Side-to-move-relative board BEFORE each ply (board[0] = side to move). One
    entry per move; the entry at index p is the position the player at ply p faces."""
    if len(moves) < 2:
        return None
    s = GameState.initial()
    out: list[np.ndarray] = []
    for m in moves:
        out.append(s.board.astype(bool).copy())
        s = s.apply(m)
    return out


def _extract(board: np.ndarray, args) -> tuple[int, int]:
    """One CPU positive proof to recover the catalyst move + mate distance (the GPU
    gives only a verdict). Fast at a first-VCT (a real win exists)."""
    res = solve_vct(board, max_depth=args.extract_depth, max_nodes=args.extract_nodes)
    move = int(res.winning_move) if res.winning_move is not None else -1
    md = int(res.mate_distance) if res.mate_distance is not None else -1
    return move, md


def _solve_flat(boards: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    """Solve a (M,2,N,N) stack in GPU_BATCH sub-batches. Returns (win, cap) bool (M,)."""
    M = len(boards)
    win = np.empty(M, dtype=bool)
    cap = np.empty(M, dtype=bool)
    for s in range(0, M, GPU_BATCH):
        e = min(s + GPU_BATCH, M)
        w, h = solve_vct_mega_bb(boards[s:e], max_nodes=args.max_nodes, tg=args.tg)
        win[s:e] = np.asarray(w, dtype=bool)
        cap[s:e] = np.asarray(h, dtype=bool)
    return win, cap


def _classify(win_row: np.ndarray, cap_row: np.ndarray) -> tuple[str, int, list[int]]:
    """Forward scan one game's per-ply verdicts. Returns (verdict, first_win_ply,
    cap_plies_in_prefix). verdict in {'emit','defer','novct'}."""
    cap_plies = []
    for p in range(len(win_row)):
        if win_row[p]:
            # first WIN: certifiable iff no CAP strictly before it
            return ("defer" if cap_plies else "emit", p, cap_plies)
        if cap_row[p]:
            cap_plies.append(p)
    # no WIN at all
    return ("defer" if cap_plies else "novct", -1, cap_plies)


def _process(games, args, out, log, stats):
    """games: list of [shard, idx, moves, winner]. Flatten every ply, GPU-solve once,
    then per game apply the trit rule and append rows."""
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
    win, cap = _solve_flat(boards, args)
    t_gpu = time.time() - t0

    per_win, per_cap = {}, {}
    for (gi, p), w, c in zip(index, win, cap):
        per_win.setdefault(gi, {})[p] = bool(w)
        per_cap.setdefault(gi, {})[p] = bool(c)

    first_rows, defer_rows = [], []
    for gi, (shard, idx, moves, winner) in enumerate(games):
        if gi not in boards_of:
            continue
        npl = len(boards_of[gi])
        wr = np.array([per_win[gi][p] for p in range(npl)], dtype=bool)
        cr = np.array([per_cap[gi][p] for p in range(npl)], dtype=bool)
        verdict, w, cap_plies = _classify(wr, cr)
        if verdict == "emit":
            board = boards_of[gi][w]
            move, md = _extract(board, args)
            if move < 0:
                stats["extract_miss"] += 1
            stats[f"stm_{w % 2}"] += 1
            stats[f"md_{md}"] += 1
            first_rows.append({
                "atk": _occupied(board[0]), "dfd": _occupied(board[1]),
                "move": move, "md": md, "ply": int(w), "game_len": int(npl),
                "stm": int(w % 2), "winner": int(winner),
                "shard": shard, "idx": int(idx),
            })
            stats["emitted"] += 1
        elif verdict == "defer":
            defer_rows.append({
                "shard": shard, "idx": int(idx), "moves": moves, "winner": int(winner),
                "budget": int(args.max_nodes), "first_win_ply": int(w),
                "cap_plies": [int(p) for p in cap_plies],
            })
            stats["deferred"] += 1
        else:
            stats["novct"] += 1

    if first_rows:
        with gzip.open(out["first"], "at") as f:
            for r in first_rows:
                f.write(json.dumps(r) + "\n")
    if defer_rows:
        with gzip.open(out["defer"], "at") as f:
            for r in defer_rows:
                f.write(json.dumps(r) + "\n")
    dt = time.time() - t0
    log(f"[first] {len(games)}g {len(flat)}pos gpu={t_gpu:.1f}s total={dt:.1f}s "
        f"emit={len(first_rows)} defer={len(defer_rows)} "
        f"cum_emit={stats['emitted']} cum_defer={stats['deferred']} "
        f"({len(games)/max(dt,1e-9):.1f} g/s)")


def _flush(pend, args, out, log, stats):
    if pend:
        _process(pend, args, out, log, stats)
        stats["seen"] += len(pend)
        pend.clear()


def _capped(args, stats) -> bool:
    return bool(args.max_games) and stats["seen"] >= args.max_games


def _run_shards(args, out, log, stats, mpath):
    """Process unprocessed game shards, flush in chunks, mark a shard done only
    when fully consumed (so resume never re-mines or skips a partial shard)."""
    processed = set()
    if os.path.exists(mpath):
        processed = {l.strip() for l in open(mpath) if l.strip()}
    shards = [p for p in iter_shard_paths(args.games_dir)
              if os.path.basename(p) not in processed]
    if args.max_shards:
        shards = shards[:args.max_shards]
    if not shards:
        return True  # nothing left
    for sp in shards:
        if _capped(args, stats):
            return True
        base = os.path.basename(sp)
        pend, interrupted = [], False
        for ix, g in enumerate(iter_games(args.games_dir, [sp])):
            if _capped(args, stats):
                interrupted = True
                break
            pend.append([base, ix, g["moves"], g.get("winner", -1)])
            if len(pend) >= args.games_per_flush:
                _flush(pend, args, out, log, stats)
        _flush(pend, args, out, log, stats)
        if not interrupted:                       # only mark a fully-consumed shard
            with open(mpath, "a") as mf:
                mf.write(base + "\n")
    return False


def _run_defers(args, out, log, stats):
    """Re-mine the deferred pile (one stream, no manifest) at the current budget."""
    pend = []
    with gzip.open(args.resolve_defers, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pend.append([d["shard"], d["idx"], d["moves"], d.get("winner", -1)])
            if len(pend) >= args.games_per_flush:
                _flush(pend, args, out, log, stats)
    _flush(pend, args, out, log, stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-dir")
    ap.add_argument("--resolve-defers", help="defer.jsonl.gz to re-mine at this budget")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-nodes", type=int, default=4000,
                    help="megakernel node budget; fatter = truer 'first', more cost")
    ap.add_argument("--tg", type=int, default=32)
    ap.add_argument("--extract-depth", type=int, default=MAXD - 2)
    ap.add_argument("--extract-nodes", type=int, default=4000)
    ap.add_argument("--games-per-flush", type=int, default=400)
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all")
    ap.add_argument("--max-games", type=int, default=0, help="0 = all (smoke cap)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--sleep", type=float, default=120.0)
    args = ap.parse_args()
    if not args.games_dir and not args.resolve_defers:
        ap.error("need --games-dir or --resolve-defers")

    os.makedirs(args.out, exist_ok=True)
    out = {"first": os.path.join(args.out, "first_vct.jsonl.gz"),
           "defer": os.path.join(args.out, "defer.jsonl.gz")}
    log_path = os.path.join(args.out, "first_vct.log")
    mpath = os.path.join(args.out, "manifest.txt")

    def log(m):
        with open(log_path, "a") as f:
            f.write(m + "\n")
        print(m, flush=True)

    stats: Counter = Counter()
    log(f"[first] start pid={os.getpid()} N={N} mn={args.max_nodes} "
        f"extract={args.extract_depth}/{args.extract_nodes} "
        f"gpf={args.games_per_flush} resolve={bool(args.resolve_defers)}")

    if args.resolve_defers:
        _run_defers(args, out, log, stats)
    else:
        while True:
            empty = _run_shards(args, out, log, stats, mpath)
            if args.once or _capped(args, stats):
                break
            if empty:
                log(f"[first] idle (cum_emit={stats['emitted']}); sleeping {args.sleep:.0f}s")
                time.sleep(args.sleep)

    log(f"[first] EXIT seen={stats['seen']} emitted={stats['emitted']} "
        f"deferred={stats['deferred']} novct={stats['novct']} "
        f"extract_miss={stats['extract_miss']}")


if __name__ == "__main__":
    main()
