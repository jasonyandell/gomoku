"""Rapfi-vs-Rapfi "enabling-shape" harvester (NO VCT solve).

Jason's idea: generate strong-vs-strong games cheaply, walk BACKWARD from the
win through the *realized forced run*, and mine the board shape at the MOMENT the
forced kill-sequence began -- the ENABLING SHAPE. A forcing/VCT line is
line-bound by construction (the solver scans only the 4 axes), but the board ONE
MOVE BEFORE the forced kill begins is the SETUP and need not be line-bound. That
setup is the non-line "molecule" candidate. We mine the setup, not the kill.

Pipeline (all CPU; Rapfi is CPU NNUE -- leaves the GPU free):

  1. GAME GENERATION -- Rapfi vs Rapfi via ``gomoku.rapfi_pool.RapfiPool``, with
     MANDATORY opening diversity (Rapfi-vs-Rapfi from a fixed start is near
     deterministic -> you'd mine the same game forever). Each game gets a random
     opening of 2..6 plies sampled from near-centre legal cells; Rapfi then plays
     both sides to a terminal or the ply cap. We record the full move sequence +
     the winner.

  2. FORCED-RUN WALK-BACK (the cheap core, NO recursion) -- from the winning move
     (makes >=5) walk backward classifying each ply FORCED vs FREE with a one-ply
     four proxy (``gomoku.vcf._five_completions``): the alternating chain
     {winner makes a four -> loser forced-blocks a completion square -> ... ->
     winner makes five}. Stop at the winner's FIRST four of the chain. The board
     just BEFORE that first four = the ENABLING SHAPE. We capture the enabling
     board (winner = plane 0 = attacker frame, directly comparable to the VCT
     control corpus), the catalyst move, the realized forced-run length, the
     winner, and a cheap offensive-setup vs defensive-blunder tag.

  3. HARVEST CORPUS -- atomic .npz shards (store.py pattern) with D4-canonical
     dedup keys; ``--reduce`` merges + dedups -> corpus.npz in the build_corpus
     format (boards/moves/tags/mds) so the bio scripts consume it unchanged.

Run (generate, time-capped):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.harvest_forced \
      --out <dir> --secs 240 --engines 16 --timeout-ms 80
Then reduce:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.harvest_forced \
      --out <dir> --reduce
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from gomoku.game import GameState, _sym_board
from gomoku.rapfi_pool import RapfiPool
from gomoku.vcf import _five_completions

# --------------------------------------------------------------------------
# Cheap one-ply four primitives (NO recursion, NO VCF/VCT solve). Everything
# operates on a pair of absolute-colour (N,N) bool planes (black, white).
# --------------------------------------------------------------------------


def _makes_four(mine: np.ndarray, opp: np.ndarray, m: int) -> bool:
    """True iff placing my stone at flat cell ``m`` gives me an immediate-five
    (i.e. the move makes a four/open-four -- an immediate five-threat)."""
    r, c = divmod(m, N)
    if mine[r, c] or opp[r, c]:
        return False
    mine = mine.copy()
    mine[r, c] = True
    empty = ~(mine | opp)
    return len(_five_completions(mine, empty)) > 0


def _is_forced_block(mine: np.ndarray, opp: np.ndarray, m: int) -> bool:
    """True iff the side to move (planes ``mine``) is facing an opponent four (an
    immediate five-threat) AND ``m`` occupies one of its completion squares -- the
    cheap 'this move was a forced block' proxy."""
    empty = ~(mine | opp)
    return m in set(_five_completions(opp, empty))


# --------------------------------------------------------------------------
# D4-canonical board key (colour-aware: winner stays plane 0). Two enabling
# shapes that are rotations/reflections of each other are ONE shape.
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
# Game generation with opening diversity.
# --------------------------------------------------------------------------


def _random_opening(rng: np.random.Generator, n_plies: int) -> list[int]:
    """A diverse opening: ``n_plies`` distinct cells drawn near the board centre.

    First stone uniformly within the central (2c+1)^2 box; the rest are drawn from
    empties within Chebyshev distance 3 of the existing stones (keeps the opening
    in contact / sensible) -- this scatters the game into a different basin every
    time so strong-vs-strong does not collapse onto one deterministic line.
    """
    c = N // 2
    box = 3
    cells: list[int] = []
    occ = np.zeros((N, N), bool)
    # first stone near centre
    r = c + int(rng.integers(-box, box + 1))
    col = c + int(rng.integers(-box, box + 1))
    cells.append(r * N + col)
    occ[r, col] = True
    for _ in range(n_plies - 1):
        rows, cols = np.nonzero(occ)
        near = np.zeros((N, N), bool)
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                rr, cc = rows + dr, cols + dc
                v = (rr >= 0) & (rr < N) & (cc >= 0) & (cc < N)
                near[rr[v], cc[v]] = True
        near &= ~occ
        choices = np.flatnonzero(near.reshape(-1))
        if len(choices) == 0:
            break
        m = int(rng.choice(choices))
        cells.append(m)
        occ[m // N, m % N] = True
    return cells


def play_game(pool: RapfiPool, rng: np.random.Generator, *,
              max_plies: int, min_open: int, max_open: int):
    """Play one Rapfi-vs-Rapfi game from a random opening. Returns
    ``(moves, winner)`` with winner in {0,1} (colour of the last mover) or None
    for a non-decisive game (draw / hit ply cap)."""
    n_open = int(rng.integers(min_open, max_open + 1))
    moves = _random_opening(rng, n_open)
    s = GameState.initial()
    for m in moves:
        s = s.apply(m)
        done, _ = s.is_terminal()
        if done:  # an opening accidentally made five (rare) -- discard
            return None, None
    while len(moves) < max_plies:
        a = pool.pick(s, rng)
        s = s.apply(a)
        moves.append(int(a))
        done, val = s.is_terminal()
        if done:
            # val == -1 from side-to-move perspective => the player who just moved
            # (plane 1 now) won. winner colour = parity of the last move index.
            if val < 0:
                return moves, (len(moves) - 1) % 2
            return None, None  # draw
    return None, None  # hit ply cap, undecided


# --------------------------------------------------------------------------
# The walk-back. Reconstruct absolute-colour planes ply by ply, then walk the
# realized forced chain back to the winner's first four.
# --------------------------------------------------------------------------


def walk_back(moves: list[int], winner: int) -> dict | None:
    """Walk back the realized forced run. Returns a record with the enabling
    board (winner=plane0) + metadata, or None if the terminal move is not a real
    five (defensive)."""
    T = len(moves)
    # board_before[i] = (black, white) planes just before move i is played.
    blk = np.zeros((N, N), bool)
    wht = np.zeros((N, N), bool)
    before: list[tuple[np.ndarray, np.ndarray]] = []
    for i, m in enumerate(moves):
        before.append((blk.copy(), wht.copy()))
        r, c = divmod(m, N)
        if i % 2 == 0:
            blk[r, c] = True
        else:
            wht[r, c] = True

    def planes_for(color: int, idx: int):
        b, w = before[idx]
        return (b, w) if color == 0 else (w, b)

    L = T - 1
    if (L % 2) != winner:
        return None
    # sanity: the winning move must actually make a five
    mine_L, opp_L = planes_for(winner, L)
    if not _makes_four(mine_L, opp_L, moves[L]):
        return None  # not a clean five-completion; skip

    t = L  # current winner-four index of the chain (L = the four that became five)
    while t - 2 >= 0:
        lc = (t - 1) % 2  # loser to move at t-1
        l_mine, l_opp = planes_for(lc, t - 1)
        if not _is_forced_block(l_mine, l_opp, moves[t - 1]):
            break
        w_mine, w_opp = planes_for(winner, t - 2)
        if not _makes_four(w_mine, w_opp, moves[t - 2]):
            break
        t -= 2

    catalyst = moves[t]
    winner_fours = (L - t) // 2 + 1
    run_plies = L - t + 1

    # enabling board in winner=attacker frame (plane0 = winner stones)
    b, w = before[t]
    enab = np.stack([b, w]) if winner == 0 else np.stack([w, b])

    # offensive-setup vs defensive-blunder (cheap heuristic): could the winner
    # have launched the IDENTICAL catalyst four two plies earlier, on its own
    # tempo (before the loser's intervening free move)? If yes -> the winner
    # constructed it (offensive setup). If the catalyst only became available
    # because of the loser's last move -> defensive blunder.
    if t - 2 >= 0:
        w_mine2, w_opp2 = planes_for(winner, t - 2)
        tag = "off" if _makes_four(w_mine2, w_opp2, catalyst) else "def"
    else:
        tag = "def"

    return {
        "board": enab.astype(bool),
        "move": int(catalyst),
        "tag": tag,
        "winner_fours": int(winner_fours),
        "run_plies": int(run_plies),
        "winner": int(winner),
        "game_len": int(T),
        "key": canon_board_key(enab),
    }


# --------------------------------------------------------------------------
# Sharded store (atomic temp+replace; store.py pattern).
# --------------------------------------------------------------------------


def _flush_shard(out_dir: str, seq: int, rows: list[dict]) -> None:
    boards = np.stack([r["board"] for r in rows]).astype(bool)
    moves = np.asarray([r["move"] for r in rows], np.int32)
    tags = np.asarray([r["tag"] for r in rows], dtype="<U3")
    mds = np.asarray([r["winner_fours"] for r in rows], np.int16)
    run_plies = np.asarray([r["run_plies"] for r in rows], np.int16)
    winner = np.asarray([r["winner"] for r in rows], np.int8)
    game_len = np.asarray([r["game_len"] for r in rows], np.int16)
    keys = np.stack([np.frombuffer(r["key"], np.uint8) for r in rows])
    path = os.path.join(out_dir, f"harvest_{seq}.npz")
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, boards=boards, moves=moves, tags=tags, mds=mds,
                 run_plies=run_plies, winner=winner, game_len=game_len, keys=keys)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Generate (time-capped, parallel games over the pool).
# --------------------------------------------------------------------------


def generate(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    deadline = time.time() + args.secs
    rows: list[dict] = []
    lock = threading.Lock()
    seq = [len(glob.glob(os.path.join(args.out, "harvest_*.npz")))]
    stats = {"games": 0, "decisive": 0, "harvested": 0, "discarded": 0,
             "below_min_run": 0}
    # FULL realized-run-length histogram over ALL decisive games (incl. the
    # filtered-out 1s/2s and the rare long tail) -- the substrate, for tuning.
    full_run_hist: dict[int, int] = {}
    seed_counter = [0]

    with RapfiPool(size=args.engines, timeout_ms=args.timeout_ms,
                   board_size=N) as pool:

        def worker():
            while time.time() < deadline:
                with lock:
                    sd = seed_counter[0]
                    seed_counter[0] += 1
                rng = np.random.default_rng(args.seed + sd)
                try:
                    moves, winner = play_game(
                        pool, rng, max_plies=args.max_plies,
                        min_open=args.min_open, max_open=args.max_open)
                except Exception:
                    continue
                rec = None
                with lock:
                    stats["games"] += 1
                if winner is None:
                    continue
                with lock:
                    stats["decisive"] += 1
                rec = walk_back(moves, winner)
                if rec is None:
                    with lock:
                        stats["discarded"] += 1
                    continue
                wf = rec["winner_fours"]
                with lock:
                    full_run_hist[wf] = full_run_hist.get(wf, 0) + 1
                    # GOLD-MINER FILTER: keep only multi-step forcing chains
                    # (>= min_run winner-fours). 1 == enabling shape IS the kill
                    # (no setup); 2 == one-block kill. Both trivial -> discard.
                    # The rare long chains are never capped (the nuggets).
                    if wf < args.min_run:
                        stats["below_min_run"] += 1
                        continue
                    rows.append(rec)
                    stats["harvested"] += 1
                    if len(rows) >= args.shard_size:
                        _flush_shard(args.out, seq[0], rows)
                        seq[0] += 1
                        rows.clear()

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.games) as ex:
            futs = [ex.submit(worker) for _ in range(args.games)]
            for f in futs:
                f.result()
        elapsed = time.time() - t0

    with lock:
        if rows:
            _flush_shard(args.out, seq[0], rows)
            rows.clear()

    gps = stats["games"] / elapsed if elapsed else 0.0
    out = {**stats, "elapsed_s": round(elapsed, 1),
           "games_per_s": round(gps, 1),
           "harvested_per_s": round(stats["harvested"] / elapsed, 1) if elapsed else 0,
           "min_run": args.min_run,
           "full_run_hist": {int(k): int(full_run_hist[k]) for k in sorted(full_run_hist)},
           "engines": args.engines, "concurrent_games": args.games,
           "timeout_ms": args.timeout_ms}
    print(json.dumps(out, indent=2))
    with open(os.path.join(args.out, "gen_stats.json"), "w") as f:
        json.dump(out, f, indent=2)


# --------------------------------------------------------------------------
# Reduce: merge shards, D4-dedup, write corpus.npz (build_corpus format).
# --------------------------------------------------------------------------


def reduce(args) -> None:
    paths = sorted(glob.glob(os.path.join(args.out, "harvest_*.npz")))
    seen: set[bytes] = set()
    B, MV, TG, MD, RP, WN, GL = [], [], [], [], [], [], []
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
                GL.append(z["game_len"][i])
    boards = np.stack(B).astype(bool)
    moves = np.asarray(MV, np.int64)
    tags = np.asarray(TG, dtype="<U3")
    mds = np.asarray(MD, np.int64)
    run_plies = np.asarray(RP, np.int64)
    winner = np.asarray(WN, np.int8)
    game_len = np.asarray(GL, np.int64)
    corpus = os.path.join(args.out, "corpus.npz")
    np.savez_compressed(corpus, boards=boards, moves=moves, tags=tags, mds=mds,
                        run_plies=run_plies, winner=winner, game_len=game_len)

    run_hist = {int(k): int(v) for k, v in
                zip(*np.unique(mds, return_counts=True))}
    plies_hist = {int(k): int(v) for k, v in
                  zip(*np.unique(run_plies, return_counts=True))}
    stats = {
        "n_raw": n_raw, "n_unique_d4": int(len(boards)),
        "n_dropped_d4": n_raw - int(len(boards)),
        "n_offensive_setup": int((tags == "off").sum()),
        "n_defensive_blunder": int((tags == "def").sum()),
        "winner_fours_hist": run_hist,
        "run_plies_hist": plies_hist,
        "winner_fours_mean": round(float(mds.mean()), 2),
        "game_len_mean": round(float(game_len.mean()), 1),
        "corpus": corpus, "n_shards": len(paths),
    }
    print(json.dumps(stats, indent=2))
    with open(os.path.join(args.out, "harvest_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--secs", type=float, default=180.0)
    ap.add_argument("--engines", type=int, default=16)
    ap.add_argument("--games", type=int, default=16, help="concurrent game threads")
    ap.add_argument("--timeout-ms", type=int, default=80)
    ap.add_argument("--min-run", type=int, default=3,
                    help="keep only enabling shapes whose realized forced-run "
                         "length (winner fours) >= this (gold-miner filter)")
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--min-open", type=int, default=2)
    ap.add_argument("--max-open", type=int, default=6)
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7000)
    args = ap.parse_args()
    if args.reduce:
        reduce(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
