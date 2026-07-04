#!/usr/bin/env python
"""Max-throughput 15x15 free-style gomoku game generator (CPU-only, all cores).

GOAL: mine as many games/sec as possible while keeping REAL THREAT STRUCTURE —
the players make and respond to fives/fours/threes, so decisive games end in
genuine forcing sequences (four-chains / double threats). Strength is NOT a
goal; throughput + threat structure are.

WHY THIS IS FAST
----------------
We drop the network AND the engine process AND the package's GameState (which
carries 17 history planes per ply). Instead each game runs a self-contained,
fully-incremental "line counter" in pure Python:

  * Precompute every length-5 window ("line") on the board and, per cell, the
    list of lines through it.
  * Maintain per-line stone counts for each player. A move touches only the
    ~<=20 lines through its cell -> O(1)-ish per move, no board rescans.
  * Immediate wins (a line at 4-of-mine, 0-of-yours -> its empty cell) and the
    blocks against them are maintained incrementally as `fours[player]` sets.
  * Move policy: (1) complete a five if I can; (2) else block the opponent's
    five; (3) else epsilon-greedy over a cheap attack/defense line score that
    rewards making fours/open-threes and blocking the opponent's threes.

This yields forcing games (every five is completed, every single four is
blocked) at hundreds of games/sec PER CORE, fanned out across all cores.

OUTPUT: append-only sharded JSONL. Each worker owns its own shard stream
(`games_w{wid}_{seq}.jsonl`) written whole via temp-file + atomic os.replace,
so a separate consumer can read COMPLETED shards while we keep appending and a
SIGKILL mid-shard never leaves a torn shard a reader can see (it loses at most
the games buffered in the current shard). One line per game:

    {"moves": [<flat cell idx in play order, black first>, ...],
     "winner": 0|1|-1,   # 0=black(first), 1=white, -1=draw
     "bs": 15}

Stage 2 (the VCT-backward pass) replays `moves` to reconstruct every position.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time

# Board size: process-level, from GOMOKU_BOARD_SIZE (default 15 here).
N = int(os.environ.get("GOMOKU_BOARD_SIZE", "15"))
WIN = 5
NCELL = N * N

# ---------------------------------------------------------------------------
# Precompute lines (length-5 windows) and per-cell line membership. Done once.
# ---------------------------------------------------------------------------
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _build_lines():
    lines = []  # each: tuple of 5 cell indices
    for r in range(N):
        for c in range(N):
            for dr, dc in _DIRS:
                er, ec = r + 4 * dr, c + 4 * dc
                if 0 <= er < N and 0 <= ec < N:
                    lines.append(tuple((r + k * dr) * N + (c + k * dc) for k in range(5)))
    lines_at = [[] for _ in range(NCELL)]
    for lid, cells in enumerate(lines):
        for cell in cells:
            lines_at[cell].append(lid)
    # freeze to tuples for speed
    lines_at = [tuple(x) for x in lines_at]
    return lines, lines_at


LINES, LINES_AT = _build_lines()
N_LINES = len(LINES)

# Neighbors within Chebyshev distance 1 (frontier expansion).
_NEIGH = [[] for _ in range(NCELL)]
for r in range(N):
    for c in range(N):
        cell = r * N + c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    _NEIGH[cell].append(nr * N + nc)
_NEIGH = [tuple(x) for x in _NEIGH]

# Scoring tables. ATT indexed by my resulting count (1..4); DEF indexed by the
# opponent's count in a line I'd neutralize (1..3). Tuned so: make-a-four >>
# make-an-open-three > extend, and blocking an opponent three is worth more
# than making my own pair. Opponent fours are handled explicitly (must-block),
# not via DEF.
ATT = (0, 1, 9, 90, 1500)        # new count 0..4
DEF = (0, 1, 7, 130)             # opp count 0..3
_CENTER = (N // 2) * N + (N // 2)


def play_game(rng_random, rng_randint, rng_choice, eps, max_moves=NCELL):
    """Play one game. Returns (moves_list, winner) winner in {0,1,-1}."""
    cnt0 = [0] * N_LINES
    cnt1 = [0] * N_LINES
    cnt = (cnt0, cnt1)
    occ = bytearray(b"\xff" * NCELL)  # 255 == empty, else player 0/1
    fours0 = set()
    fours1 = set()
    fours = (fours0, fours1)
    frontier = set()
    moves = []
    p = 0

    # Diversified opening: black plays a random cell near center.
    rad = 3
    cr = N // 2 + rng_randint(-rad, rad)
    cc = N // 2 + rng_randint(-rad, rad)
    cr = 0 if cr < 0 else (N - 1 if cr >= N else cr)
    cc = 0 if cc < 0 else (N - 1 if cc >= N else cc)
    first = cr * N + cc

    move = first
    while True:
        # ---- place `move` for player p ----
        occ[move] = p
        moves.append(move)
        cp = cnt[p]
        copp = cnt[1 - p]
        fp = fours[p]
        fopp = fours[1 - p]
        won = False
        for lid in LINES_AT[move]:
            v = cp[lid] + 1
            cp[lid] = v
            if copp[lid] == 0:
                if v == 5:
                    won = True
                elif v == 4:
                    fp.add(lid)
            # any line I just touched is dead for the opponent
            if lid in fopp:
                fopp.discard(lid)
        # frontier update
        frontier.discard(move)
        for nb in _NEIGH[move]:
            if occ[nb] == 255:
                frontier.add(nb)

        if won:
            return moves, p
        if len(moves) >= max_moves:
            return moves, -1

        # ---- choose next player's move ----
        p = 1 - p
        cp = cnt[p]
        copp = cnt[1 - p]
        myfours = fours[p]
        oppfours = fours[1 - p]

        if myfours:
            # complete a five: pick the empty cell of one of my four-lines
            lid = next(iter(myfours))
            cells = LINES[lid]
            move = next(cell for cell in cells if occ[cell] == 255)
            continue
        if oppfours:
            # must block: empty cell of an opponent four-line
            lid = next(iter(oppfours))
            cells = LINES[lid]
            move = next(cell for cell in cells if occ[cell] == 255)
            continue

        # epsilon: random frontier move for diversity
        if not frontier:
            # no stones adjacency yet (shouldn't happen after move 1) -> center
            move = _CENTER if occ[_CENTER] == 255 else next(
                (i for i in range(NCELL) if occ[i] == 255), None)
            if move is None:
                return moves, -1
            continue

        if rng_random() < eps:
            # cheap random choice from frontier
            move = rng_choice(tuple(frontier))
            continue

        # greedy: argmax cheap attack/defense score, random tie-break
        best_score = -1.0
        best_move = -1
        for e in frontier:
            s = 0
            for lid in LINES_AT[e]:
                co = copp[lid]
                cm = cp[lid]
                if co == 0:
                    s += ATT[cm + 1]
                elif cm == 0:
                    s += DEF[co]
            # tiny noise to break ties and add diversity
            sf = s + rng_random()
            if sf > best_score:
                best_score = sf
                best_move = e
        move = best_move


# ---------------------------------------------------------------------------
# Sharded append-only JSONL writer (atomic per shard; one stream per worker).
# ---------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, out_dir, worker_id, shard_size, gzip_shards=True):
        self.out_dir = out_dir
        self.worker_id = worker_id
        self.shard_size = shard_size
        self.gzip = gzip_shards
        self.ext = "jsonl.gz" if gzip_shards else "jsonl"
        os.makedirs(out_dir, exist_ok=True)
        self._seq = self._next_seq()
        self._buf = []
        self.n_games = 0

    def _next_seq(self):
        import glob
        existing = glob.glob(os.path.join(self.out_dir, f"games_w{self.worker_id}_*.{self.ext}"))
        seqs = []
        for p in existing:
            try:
                seqs.append(int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]))
            except (ValueError, IndexError):
                pass
        return (max(seqs) + 1) if seqs else 0

    def add(self, moves, winner):
        self._buf.append((moves, winner))
        if len(self._buf) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        path = os.path.join(self.out_dir, f"games_w{self.worker_id}_{self._seq}.{self.ext}")
        tmp = path + ".tmp"
        lines = []
        for moves, winner in self._buf:
            lines.append('{"moves":[%s],"winner":%d,"bs":%d}\n'
                         % (",".join(map(str, moves)), winner, N))
        data = "".join(lines).encode()
        if self.gzip:
            import gzip
            with open(tmp, "wb") as f:
                with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=4, mtime=0) as gz:
                    gz.write(data)
        else:
            with open(tmp, "wb") as f:
                f.write(data)
        os.replace(tmp, path)  # atomic: a reader never sees a partial shard
        self.n_games += len(self._buf)
        self._seq += 1
        self._buf = []

    def close(self):
        self.flush()


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------
_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True


def worker(worker_id, out_dir, duration, shard_size, eps, seed, progress_path,
           gzip_shards=True):
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    rng = random.Random(seed + worker_id * 7919)
    rnd = rng.random
    rint = rng.randint
    rchoice = rng.choice
    w = ShardWriter(out_dir, worker_id, shard_size, gzip_shards=gzip_shards)
    t0 = time.time()
    n_games = 0
    n_pos = 0
    last_report = t0
    deadline = (t0 + duration) if duration > 0 else None
    try:
        while not _STOP:
            moves, winner = play_game(rnd, rint, rchoice, eps)
            w.add(moves, winner)
            n_games += 1
            n_pos += len(moves)
            if (n_games & 1023) == 0:
                now = time.time()
                if deadline is not None and now >= deadline:
                    break
                if now - last_report >= 5.0 and progress_path:
                    last_report = now
                    _write_progress(progress_path, worker_id, n_games, n_pos, now - t0)
    finally:
        w.close()
        if progress_path:
            _write_progress(progress_path, worker_id, n_games, n_pos, time.time() - t0)
    return n_games, n_pos


def _write_progress(progress_path, worker_id, n_games, n_pos, elapsed):
    tmp = f"{progress_path}.w{worker_id}.tmp"
    final = f"{progress_path}.w{worker_id}"
    try:
        with open(tmp, "w") as f:
            json.dump({"wid": worker_id, "games": n_games, "pos": n_pos,
                       "elapsed": elapsed}, f)
        os.replace(tmp, final)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Benchmark + spot-check (single process), and a parallel launcher.
# ---------------------------------------------------------------------------
def spot_check(n=2000, eps=0.15, seed=0):
    """Generate n games single-threaded; report games/s, pos/s, threat stats."""
    rng = random.Random(seed)
    rnd, rint, rchoice = rng.random, rng.randint, rng.choice
    t0 = time.time()
    n_pos = 0
    decisive = 0
    forcing_end = 0
    lengths = []
    for _ in range(n):
        moves, winner = play_game(rnd, rint, rchoice, eps)
        n_pos += len(moves)
        lengths.append(len(moves))
        if winner != -1:
            decisive += 1
            if _ends_in_forcing_chain(moves, winner):
                forcing_end += 1
    dt = time.time() - t0
    lengths.sort()
    return {
        "games": n, "dt": dt, "games_per_s": n / dt, "pos_per_s": n_pos / dt,
        "decisive_frac": decisive / n,
        "forcing_end_frac_of_decisive": (forcing_end / decisive) if decisive else 0.0,
        "median_len": lengths[len(lengths) // 2],
        "avg_len": n_pos / n,
    }


def _ends_in_forcing_chain(moves, winner):
    """Heuristic verifier: replay; check the winner's final move completed a 5
    AND at least the prior move by the winner created a four (i.e. the win
    arrived via a forcing four). Returns True if the last >=2 winner moves were
    a four->five forcing finish."""
    # Replay with the line counter; record, for each winner move, whether it
    # created a four (a forcing threat) at the time it was played.
    cnt0 = [0] * N_LINES
    cnt1 = [0] * N_LINES
    cnt = (cnt0, cnt1)
    made_four_seq = []  # winner's moves: did this move create a >=four threat?
    p = 0
    for move in moves:
        cp = cnt[p]
        copp = cnt[1 - p]
        made_four = False
        for lid in LINES_AT[move]:
            v = cp[lid] + 1
            cp[lid] = v
            if copp[lid] == 0 and v >= 4:
                made_four = True
        if p == winner:
            made_four_seq.append(made_four)
        p = 1 - p
    # last winner move made the five (four->five). Require the immediately
    # preceding winner move ALSO made a four (a forcing chain, not a lone four).
    return len(made_four_seq) >= 2 and made_four_seq[-1] and made_four_seq[-2]


def _parallel_worker_entry(args):
    return worker(*args)


def run_parallel(out_dir, workers, duration, shard_size, eps, seed, progress_path,
                 gzip_shards=True):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    procs = []
    for wid in range(workers):
        a = (wid, out_dir, duration, shard_size, eps, seed, progress_path, gzip_shards)
        pr = ctx.Process(target=worker, args=a, daemon=False)
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()


def _aggregate_progress(progress_path, workers):
    g = pos = 0
    el = 0.0
    for wid in range(workers):
        f = f"{progress_path}.w{wid}"
        try:
            with open(f) as fh:
                d = json.load(fh)
            g += d["games"]
            pos += d["pos"]
            el = max(el, d["elapsed"])
        except (OSError, ValueError, KeyError):
            pass
    return g, pos, el


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=False, help="output dir for shards")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to run; 0 = until killed (SIGTERM/SIGINT)")
    ap.add_argument("--shard-size", type=int, default=2000,
                    help="games per shard file")
    ap.add_argument("--eps", type=float, default=0.15,
                    help="epsilon for random (diversity) moves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gzip", action="store_true",
                    help="write plain .jsonl shards instead of gzipped")
    ap.add_argument("--bench", action="store_true",
                    help="run single-core benchmark + threat spot-check and exit")
    ap.add_argument("--bench-n", type=int, default=2000)
    args = ap.parse_args()

    print(f"[board {N}x{N}, {N_LINES} lines]")
    if args.bench:
        stats = spot_check(n=args.bench_n, eps=args.eps, seed=args.seed)
        print("SINGLE-CORE BENCHMARK + THREAT SPOT-CHECK:")
        for k, v in stats.items():
            print(f"  {k:30s} {v}")
        return

    if not args.out:
        ap.error("--out required unless --bench")
    out_dir = args.out
    progress_path = os.path.join(out_dir, "progress")
    os.makedirs(out_dir, exist_ok=True)
    print(f"launching {args.workers} workers -> {out_dir} "
          f"(duration={args.duration or 'until-killed'}, shard={args.shard_size}, eps={args.eps})")
    t0 = time.time()
    run_parallel(out_dir, args.workers, args.duration, args.shard_size,
                 args.eps, args.seed, progress_path, gzip_shards=not args.no_gzip)
    g, pos, el = _aggregate_progress(progress_path, args.workers)
    wall = time.time() - t0
    print(f"DONE: {g} games, {pos} positions in {wall:.1f}s wall "
          f"({g / wall:.0f} games/s, {pos / wall:.0f} pos/s across {args.workers} workers)")


if __name__ == "__main__":
    main()
