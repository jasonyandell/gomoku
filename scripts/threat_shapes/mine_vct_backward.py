"""Stage-2 VCT-backward enabling-shape miner (the REAL solve_vct version).

Jason's two-stage plan, stage 2. Stage 1 (the 16-engine Rapfi collector) appends
decisive strong-vs-strong games as JSONL shards. This stage walks each decisive
game BACKWARD from the win and finds the ENABLING SHAPE: the earliest position
(contiguous to the end) at which the winner already had a PROVEN forced kill.

Algorithm, per decisive game (winner W != draw):
  1. Reconstruct the game move-by-move from ``GameState.initial()``. Because the
     canonical board is side-to-move-relative (plane 0 = side to move; VERIFIED,
     see ``verify_planes`` / the test at the bottom), at any W-to-move position
     ``GameState.board[0]`` is ALREADY the winner's stones == the attacker frame
     ``solve_vct`` wants. NO plane swap is ever needed.
  2. Consider only the W-to-move positions (plies p with ``p % 2 == W``; the
     winner makes the last move, so the final position p == L is W-to-move and is
     trivially a forced win -- the immediate five). Walk BACKWARD over them
     (p = L, L-2, L-4, ...). At each, run the REAL ``solve_vct`` with plane 0 = W.
     Keep walking back while ``has_forced_win`` stays True. STOP at the first
     W-to-move position (going back) that is NO LONGER a proven forced win. The
     earliest still-forced position is the ENABLING SHAPE -- the moment the
     forced kill became available.
  3. MIN FORCED-RUN FILTER: emit only if the contiguous forced suffix is
     ``>= --min-run`` (default 3) W-moves deep. Drops trivial endgame mates.
  4. Emit: the enabling board (plane 0 = winner/attacker, directly comparable to
     the existing VCT corpus), the catalyst move (``solve_vct(...).winning_move``
     at the enabling position), the mate distance, the forced-run length, the
     enabling ply index, the total game length, and the winner.

This is INHERENTLY INCOMPLETE and that is expected: ``solve_vct`` caps (depth /
nodes) yield FALSE-NEGATIVES (a real forced win it could not prove inside budget
-> we simply stop the walk-back early, mining a later / no enabling shape). It
NEVER yields a false positive: ``solve_vct`` returns ``has_forced_win=True`` only
when it constructed an explicit forcing line to five. So every emitted enabling
shape is a GENUINE proven-forced-win position for the winner.

Output (mirrors ``harvest_forced.py`` so the downstream bio scripts consume it
unchanged): atomic ``.npz`` shards (temp + os.replace), one per input shard, with
  boards (bool, plane0 = winner/attacker frame), moves (int32 catalyst),
  tags ("vct_enable"), mds (int16 mate_distance), run_plies (int16 forced-run
  length in W-moves), winner (int8), game_len (int16), ply (int32 enabling ply),
  keys (uint8 D4-canonical dedup key).
``--reduce`` merges all shards, D4-dedups, and writes ``corpus.npz`` in the
build_corpus format (boards/moves/tags/mds + the extras above).

Engineering: be a POLITE TENANT. The priority tenant is the 16-engine Rapfi
collector pegging ~16 cores. Default to ``--workers 3`` (CPU only; the GPU is
left free). Incremental + append-only + resumable: a processed-shards manifest in
the output dir means re-runs and the sustained loop never redo a shard. ``--once``
does a single pass; ``--loop`` rescans for new completed shards forever (until
killed). Per-shard work is naturally bounded -- the backward walk stops at the
first non-forced position, so a non-tactical game costs ~1-2 solves.

Run (smoke test, single pass):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_vct_backward \
      --games-dir <games_rapfi> --out <vct_enable> --once --workers 3
Sustained overnight loop:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_vct_backward \
      --games-dir <games_rapfi> --out <vct_enable> --loop --workers 3
Reduce to a corpus:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_vct_backward \
      --out <vct_enable> --reduce
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from gomoku.game import GameState, _sym_board
from gomoku.vcf import DEFAULT_VCT_MAX_DEPTH, DEFAULT_VCT_MAX_NODES, solve_vct
from scripts.threat_shapes.read_games import iter_games, iter_shard_paths

TAG = "vct_enable"
MANIFEST = "processed_shards.txt"


# --------------------------------------------------------------------------
# D4-canonical board key (colour-aware: winner stays plane 0), same as
# harvest_forced.canon_board_key so the two corpora share a dedup namespace.
# --------------------------------------------------------------------------


def canon_board_key(board: np.ndarray) -> bytes:
    best: bytes | None = None
    for s in range(8):
        b = np.stack([_sym_board(board[0], s), _sym_board(board[1], s)])
        by = np.ascontiguousarray(b).tobytes()
        if best is None or by < best:
            best = by
    assert best is not None
    return hashlib.blake2b(best, digest_size=16).digest()


# --------------------------------------------------------------------------
# The backward walk for one game.
# --------------------------------------------------------------------------


def mine_game(moves: list[int], winner: int, *, min_run: int,
              max_depth: int, max_nodes: int, stats: Counter) -> dict | None:
    """Walk one decisive game backward to its enabling shape. Returns a record
    dict or None (no qualifying enabling shape). ``stats`` is mutated with
    diagnostics (cap hits, forced-run length, etc.)."""
    T = len(moves)
    if T < 2:
        return None
    L = T - 1
    # The winner makes the last move, so the side to move at position L has
    # parity L % 2; that MUST equal the winner. (A clean invariant of the data.)
    if (L % 2) != winner:
        stats["bad_parity"] += 1
        return None

    # Reconstruct, snapshotting the canonical board BEFORE each move. board[0] at
    # a W-to-move position == the winner's stones (verified), so no swap needed.
    s = GameState.initial()
    boards_before: list[np.ndarray] = []
    for m in moves:
        boards_before.append(s.board.copy())
        s = s.apply(m)
    boards_before.append(s.board.copy())  # terminal (unused, kept for symmetry)

    # Walk backward over W-to-move positions p = L, L-2, ... while forced.
    enable_p = None
    enable_res = None
    p = L
    while p >= 0:
        board = boards_before[p]            # plane 0 = winner/attacker frame
        res = solve_vct(board, max_depth=max_depth, max_nodes=max_nodes)
        if not res.has_forced_win:
            if res.hit_cap:
                # An "unproven" boundary -> a possible false-negative cut. Count
                # it so we can report what fraction of walks ended on a cap vs a
                # genuine non-forced position.
                stats["walk_end_cap"] += 1
            else:
                stats["walk_end_clean"] += 1
            break
        enable_p = p
        enable_res = res
        p -= 2

    if enable_p is None:
        # Even the final winning position was not provable as forced (e.g. the
        # terminal move is not a clean five, or an immediate cap). Skip.
        stats["no_forced_at_end"] += 1
        return None

    forced_run = (L - enable_p) // 2 + 1     # W-moves deep in the forced suffix
    stats[f"run_{forced_run}"] += 1
    if forced_run < min_run:
        stats["below_min_run"] += 1
        return None

    board = boards_before[enable_p]
    md = enable_res.mate_distance
    stats[f"md_{int(md) if md is not None else -1}"] += 1
    return {
        "board": board.astype(bool),         # plane0 = winner/attacker
        "move": int(enable_res.winning_move) if enable_res.winning_move is not None else -1,
        "tag": TAG,
        "md": int(md) if md is not None else -1,
        "run_plies": int(forced_run),
        "winner": int(winner),
        "game_len": int(T),
        "ply": int(enable_p),
        "key": canon_board_key(board),
    }


# --------------------------------------------------------------------------
# Per-shard worker (pure function: read shard -> records + stats). Runs in a
# child process; the parent does all the output IO + manifest writes.
# --------------------------------------------------------------------------


def process_shard(args_tuple) -> tuple[str, list[dict], dict]:
    shard_path, min_run, max_depth, max_nodes = args_tuple
    stats: Counter = Counter()
    records: list[dict] = []
    for g in iter_games(os.path.dirname(shard_path), [shard_path]):
        stats["games"] += 1
        w = g.get("winner", -1)
        if w == -1:
            continue
        stats["decisive"] += 1
        rec = mine_game(g["moves"], int(w), min_run=min_run,
                        max_depth=max_depth, max_nodes=max_nodes, stats=stats)
        if rec is not None:
            records.append(rec)
            stats["emitted"] += 1
    return os.path.basename(shard_path), records, dict(stats)


# --------------------------------------------------------------------------
# Atomic shard store (temp + os.replace), keyed off the input shard basename so
# re-processing a shard overwrites idempotently.
# --------------------------------------------------------------------------


def _out_name(in_basename: str) -> str:
    stem = in_basename
    for ext in (".jsonl.gz", ".jsonl"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return f"vct_enable_{stem}.npz"


def flush_shard(out_dir: str, in_basename: str, rows: list[dict]) -> None:
    boards = np.stack([r["board"] for r in rows]).astype(bool)
    moves = np.asarray([r["move"] for r in rows], np.int32)
    tags = np.asarray([r["tag"] for r in rows], dtype=f"<U{len(TAG)}")
    mds = np.asarray([r["md"] for r in rows], np.int16)
    run_plies = np.asarray([r["run_plies"] for r in rows], np.int16)
    winner = np.asarray([r["winner"] for r in rows], np.int8)
    game_len = np.asarray([r["game_len"] for r in rows], np.int16)
    ply = np.asarray([r["ply"] for r in rows], np.int32)
    keys = np.stack([np.frombuffer(r["key"], np.uint8) for r in rows])
    path = os.path.join(out_dir, _out_name(in_basename))
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, boards=boards, moves=moves, tags=tags, mds=mds,
                 run_plies=run_plies, winner=winner, game_len=game_len,
                 ply=ply, keys=keys)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Manifest (append-only set of processed input-shard basenames).
# --------------------------------------------------------------------------


def load_manifest(out_dir: str) -> set[str]:
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def append_manifest(out_dir: str, basename: str) -> None:
    with open(os.path.join(out_dir, MANIFEST), "a") as f:
        f.write(basename + "\n")
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------
# One pass over all unprocessed completed shards (parallel across workers).
# --------------------------------------------------------------------------


def run_pass(args, agg: Counter) -> int:
    os.makedirs(args.out, exist_ok=True)
    processed = load_manifest(args.out)
    all_shards = iter_shard_paths(args.games_dir)
    todo = [p for p in all_shards if os.path.basename(p) not in processed]
    if args.max_shards:
        todo = todo[: args.max_shards]
    if not todo:
        return 0

    t0 = time.time()
    work = [(p, args.min_run, args.max_depth, args.max_nodes) for p in todo]
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, w): w[0] for w in work}
        for fut in as_completed(futs):
            basename, records, stats = fut.result()
            if records:
                flush_shard(args.out, basename, records)
            append_manifest(args.out, basename)
            for k, v in stats.items():
                agg[k] += v
            n_done += 1
            if n_done % 20 == 0:
                el = time.time() - t0
                gps = agg["games"] / el if el else 0
                print(f"  [{n_done}/{len(todo)} shards] games={agg['games']} "
                      f"emitted={agg['emitted']} ({gps:.0f} games/s)", flush=True)
    el = time.time() - t0
    agg["_pass_elapsed_s"] = round(el, 1)
    return n_done


def _print_summary(agg: Counter, label: str) -> None:
    el = agg.get("_pass_elapsed_s", 0) or 0
    games = agg.get("games", 0)
    md_hist = {int(k[3:]): v for k, v in agg.items() if k.startswith("md_")}
    run_hist = {int(k[4:]): v for k, v in agg.items() if k.startswith("run_")}
    summary = {
        "label": label,
        "games_scanned": games,
        "decisive": agg.get("decisive", 0),
        "emitted": agg.get("emitted", 0),
        "emit_rate_per_decisive": round(agg.get("emitted", 0) / agg["decisive"], 4)
        if agg.get("decisive") else 0.0,
        "below_min_run": agg.get("below_min_run", 0),
        "no_forced_at_end": agg.get("no_forced_at_end", 0),
        "bad_parity": agg.get("bad_parity", 0),
        "walk_end_clean": agg.get("walk_end_clean", 0),
        "walk_end_cap": agg.get("walk_end_cap", 0),
        "walk_end_cap_frac": round(
            agg.get("walk_end_cap", 0)
            / max(1, agg.get("walk_end_cap", 0) + agg.get("walk_end_clean", 0)), 4),
        "forced_run_hist": {k: run_hist[k] for k in sorted(run_hist)},
        "mate_distance_hist": {k: md_hist[k] for k in sorted(md_hist)},
        "elapsed_s": el,
        "games_per_s": round(games / el, 1) if el else 0,
    }
    print(json.dumps(summary, indent=2), flush=True)


# --------------------------------------------------------------------------
# Reduce: merge shards, D4-dedup, write corpus.npz (build_corpus format).
# --------------------------------------------------------------------------


def reduce(args) -> None:
    paths = sorted(glob.glob(os.path.join(args.out, "vct_enable_*.npz")))
    seen: set[bytes] = set()
    B, MV, TG, MD, RP, WN, GL, PLY = [], [], [], [], [], [], [], []
    n_raw = 0
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            keys = z["keys"]
            for i in range(len(z["boards"])):
                n_raw += 1
                k = keys[i].tobytes()
                if k in seen:
                    continue
                seen.add(k)
                B.append(z["boards"][i]); MV.append(z["moves"][i])
                TG.append(str(z["tags"][i])); MD.append(z["mds"][i])
                RP.append(z["run_plies"][i]); WN.append(z["winner"][i])
                GL.append(z["game_len"][i]); PLY.append(z["ply"][i])
    if not B:
        print(json.dumps({"n_raw": 0, "n_unique_d4": 0, "corpus": None}))
        return
    boards = np.stack(B).astype(bool)
    moves = np.asarray(MV, np.int64)
    tags = np.asarray(TG, dtype=f"<U{len(TAG)}")
    mds = np.asarray(MD, np.int64)
    run_plies = np.asarray(RP, np.int64)
    winner = np.asarray(WN, np.int8)
    game_len = np.asarray(GL, np.int64)
    ply = np.asarray(PLY, np.int64)
    corpus = os.path.join(args.out, "corpus.npz")
    tmp = corpus + ".tmp"
    with open(tmp, "wb") as fh:   # file handle => no ".npz" auto-append surprise
        np.savez_compressed(fh, boards=boards, moves=moves, tags=tags, mds=mds,
                            run_plies=run_plies, winner=winner, game_len=game_len,
                            ply=ply)
    os.replace(tmp, corpus)

    run_hist = {int(k): int(v) for k, v in zip(*np.unique(run_plies, return_counts=True))}
    md_hist = {int(k): int(v) for k, v in zip(*np.unique(mds, return_counts=True))}
    stats = {
        "n_raw": n_raw, "n_unique_d4": int(len(boards)),
        "n_dropped_d4": n_raw - int(len(boards)),
        "winner_black": int((winner == 0).sum()),
        "winner_white": int((winner == 1).sum()),
        "mate_distance_hist": md_hist,
        "forced_run_hist": run_hist,
        "mate_distance_mean": round(float(mds[mds >= 0].mean()), 2) if (mds >= 0).any() else None,
        "forced_run_mean": round(float(run_plies.mean()), 2),
        "game_len_mean": round(float(game_len.mean()), 1),
        "corpus": corpus, "n_shards": len(paths),
    }
    print(json.dumps(stats, indent=2))
    with open(os.path.join(args.out, "reduce_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games-dir",
                    help="input dir of append-only Rapfi game shards")
    ap.add_argument("--out", required=True, help="output dir (shards + manifest)")
    ap.add_argument("--reduce", action="store_true",
                    help="merge + D4-dedup shards -> corpus.npz, then exit")
    ap.add_argument("--once", action="store_true",
                    help="single pass over unprocessed shards, then exit")
    ap.add_argument("--loop", action="store_true",
                    help="sustained: process, sleep, rescan for new shards forever")
    ap.add_argument("--workers", type=int, default=3,
                    help="worker processes (default 3 -- polite to the collector)")
    ap.add_argument("--min-run", type=int, default=3,
                    help="emit only enabling shapes whose forced-run length "
                         "(winner-moves deep) >= this")
    ap.add_argument("--max-depth", type=int, default=DEFAULT_VCT_MAX_DEPTH,
                    help="solve_vct max attacker-move depth")
    ap.add_argument("--max-nodes", type=int, default=DEFAULT_VCT_MAX_NODES,
                    help="solve_vct global node cap")
    ap.add_argument("--sleep", type=float, default=45.0,
                    help="--loop: seconds to sleep between rescans")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="cap shards processed PER PASS (0 = unlimited); bounds a "
                         "smoke test and keeps each loop pass short")
    args = ap.parse_args()

    if args.reduce:
        reduce(args)
        return

    if not args.games_dir:
        ap.error("--games-dir is required unless --reduce")

    if args.loop:
        # Become a session/process-group leader so the whole worker tree can be
        # stopped with a single `kill -TERM -<PGID>` (macOS has no `setsid`).
        try:
            os.setsid()
        except OSError:
            pass  # already a group leader (e.g. launched under a job-control shell)
        print(f"[mine_vct_backward] LOOP start pid={os.getpid()} pgid={os.getpgrp()} "
              f"workers={args.workers} min_run={args.min_run}", flush=True)
        lifetime: Counter = Counter()
        while True:
            agg: Counter = Counter()
            n = run_pass(args, agg)
            if n:
                for k, v in agg.items():
                    lifetime[k] += v
                _print_summary(agg, label=f"pass(+{n} shards)")
                print(f"[loop] lifetime emitted={lifetime['emitted']} "
                      f"games={lifetime['games']}", flush=True)
            else:
                print(f"[loop] no new shards; sleeping {args.sleep:.0f}s "
                      f"(t={time.strftime('%H:%M:%S')})", flush=True)
            time.sleep(args.sleep)
    else:
        # single pass (the default / --once)
        agg = Counter()
        run_pass(args, agg)
        _print_summary(agg, label="once")


if __name__ == "__main__":
    main()
