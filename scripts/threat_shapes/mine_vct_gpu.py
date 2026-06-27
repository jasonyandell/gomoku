"""GPU-batched stage-2 VCT-backward enabling-shape miner.

Same job as ``mine_vct_serial.py`` -- walk each decisive game BACKWARD over the
winner's to-move positions while the position stays a PROVEN forced win, emit the
earliest still-forced board (the "enabling shape") -- but the per-position VCT
verdict comes from the on-device bitboard megakernel ``solve_vct_mega_bb`` instead
of the CPU ``gomoku.vcf.solve_vct``, and the walk is LEVEL-SYNCHRONIZED across a
whole batch of games so thousands of games ride each GPU call.

Why this beats the CPU serial miner: the serial miner is monster-bound -- a single
hard position (deep VCT search) costs tens of seconds and stalls the whole stream.
The megakernel runs every board in a batch under one node budget and is
*tail-bound by the single deepest board* (wall ~constant in B), so the monster
just sits in the shadow of the batch wall while thousands of easy games finish for
free. (See wiki/topics/gpu-vct-feasibility.md.)

The batched walk-back (level-synchronized):
  1. Reconstruct each decisive game; collect the winner's to-move boards
     ``p = L, L-2, L-4, ...`` (board[0] is already the winner frame at a
     W-to-move position -- side-to-move-relative canonical board, NO swap).
  2. Round r: batch every still-active game's board at its current step, GPU-solve.
     win  -> record this step as the game's enable_step, advance its step (walk
             one more position back).
     lose -> finalize the game at its last winning step (drop it from the batch).
     Each game needs ~run+1 GPU calls; the batch shrinks each round; monsters just
     ride the tail.
  3. min-run>=3 filter (winner-moves deep in the forced suffix).
  4. For each surviving enabling board, ONE CPU ``solve_vct`` extracts the catalyst
     ``winning_move`` + ``mate_distance`` for the record (a positive solve is fast).

CORRECTNESS: ``solve_vct_mega_bb`` is validated 0 FP / 0 FN vs ``gomoku.vcf.solve_vct``
over 320 real positions (it only returns win=True when it constructs a real forcing
proof), so every emitted enabling board is a GENUINE proven forced win -- no
false-positive shapes. A win=False with hit_cap=True is treated (like the CPU
miner's ``not has_forced_win``) as a STOP: conservative, never a false positive.

DEPTH NOTE: the megakernel's search depth is the compile-time ``MAXD`` in
``mega_vct_bb`` (its own harness validates against ``solve_vct(max_depth=MAXD-2)``).
The CPU serial miner runs ``--max-depth 10``. So the GPU path may PROVE forced wins
the depth-10 CPU miner cannot, walking back FURTHER -> runs >= the CPU miner's runs.
That is a strict, understood superset (deeper = more truths), not a disagreement;
``--validate`` quantifies it against both depth-10 and a depth-matched CPU walk.

Record format mirrors ``mine_vct_serial.py`` EXACTLY (so outputs are directly
comparable): one row per emitted shape appended to ``enable_serial.jsonl.gz``:
  {atk:[occupied idx], dfd:[occupied idx], move, md, run, winner, game_len, ply}
A manifest of processed input shards makes restarts resume where they left off.

Run (overnight, until killed):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_vct_gpu \
      --games-dir <games_rapfi> --out <vct_serial_gpu> --max-nodes 8000

Validate end-to-end vs the CPU miner on the same games:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_vct_gpu \
      --games-dir <games_rapfi> --validate --validate-games 200
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


def _occupied(plane: np.ndarray) -> list[int]:
    """Flat indices of set cells in a (N,N) bool plane -- compact storage."""
    return [int(i) for i in np.flatnonzero(plane.reshape(-1))]


class _Game:
    """One decisive game's walk-back state.

    ``wboards[k]`` is the winner-to-move board at ply ``p = L - 2k`` (board[0] =
    winner/attacker frame). ``step`` is the current walk index; ``enable_step`` is
    the deepest (largest k) step proven forced so far (-1 = none yet).
    """
    __slots__ = ("moves", "winner", "T", "L", "wboards", "step", "enable_step")

    def __init__(self, moves, winner, wboards, L):
        self.moves = moves
        self.winner = winner
        self.T = len(moves)
        self.L = L
        self.wboards = wboards          # list[(2,N,N) bool], index 0 == ply L
        self.step = 0
        self.enable_step = -1


def reconstruct(moves: list[int], winner: int, stats: Counter) -> _Game | None:
    """Build the winner-to-move board stack for one decisive game, or None if the
    game is too short / has the wrong parity (mirrors ``mine_game``'s guards)."""
    T = len(moves)
    if T < 2:
        return None
    L = T - 1
    # The winner makes the last move => side-to-move parity at ply L is L%2, which
    # must equal the winner. A clean invariant of the data.
    if (L % 2) != winner:
        stats["bad_parity"] += 1
        return None
    s = GameState.initial()
    boards_before: list[np.ndarray] = []
    for m in moves:
        boards_before.append(s.board.copy())
        s = s.apply(m)
    # W-to-move positions p = L, L-2, ... (board[0] is the winner at each: p%2==W).
    wboards: list[np.ndarray] = []
    p = L
    while p >= 0:
        wboards.append(boards_before[p].astype(bool))
        p -= 2
    return _Game(moves, winner, wboards, L)


def walk_batch(games: list[_Game], *, max_nodes: int, tg: int,
               stats: Counter) -> None:
    """Level-synchronized backward walk over a batch of games. Mutates each
    game's ``enable_step`` in place. Each round GPU-solves every still-active
    game's current board; winners advance one step back, losers drop."""
    active = [g for g in games if g.wboards]
    rounds = 0
    while active:
        rounds += 1
        batch = np.stack([g.wboards[g.step] for g in active])  # (B,2,N,N) bool
        win, hit = solve_vct_mega_bb(batch, max_nodes=max_nodes, tg=tg)
        stats["gpu_calls"] += 1
        stats["gpu_boards"] += len(active)
        stats["gpu_cap_hits"] += int(hit.sum())
        still: list[_Game] = []
        for i, g in enumerate(active):
            if win[i]:
                g.enable_step = g.step
                g.step += 1
                if g.step < len(g.wboards):
                    still.append(g)
                # else: forced all the way to the opening -> finalize at last step
            # else: not a proven forced win -> stop walking (finalize)
        active = still
    stats["walk_rounds_max"] = max(stats.get("walk_rounds_max", 0), rounds)


def finalize_and_emit(games: list[_Game], *, min_run: int, extract_depth: int,
                      extract_nodes: int, rec_path: str | None,
                      stats: Counter) -> list[dict]:
    """Apply the min-run filter, extract catalyst move + mate distance on CPU for
    survivors, append rows to ``rec_path`` (if given). Returns the emitted rows."""
    rows: list[dict] = []
    for g in games:
        if g.enable_step < 0:
            stats["no_forced_at_end"] += 1
            continue
        forced_run = g.enable_step + 1          # step 0 (ply L) == run 1
        stats[f"run_{forced_run}"] += 1
        if forced_run < min_run:
            stats["below_min_run"] += 1
            continue
        board = g.wboards[g.enable_step]
        # ONE CPU solve to recover the catalyst move + mate distance. The GPU
        # already proved the win, so this is a fast positive solve; use a depth
        # >= the GPU's effective depth so it re-proves the same win and yields a
        # move (a depth-10 extract could miss a deeper GPU proof).
        res = solve_vct(board, max_depth=extract_depth, max_nodes=extract_nodes)
        if not res.has_forced_win:
            stats["extract_miss"] += 1      # CPU couldn't re-prove within budget
        move = int(res.winning_move) if res.winning_move is not None else -1
        md = int(res.mate_distance) if res.mate_distance is not None else -1
        stats[f"md_{md}"] += 1
        ply = g.L - 2 * g.enable_step
        row = {
            "atk": _occupied(board[0]), "dfd": _occupied(board[1]),
            "move": move, "md": md, "run": int(forced_run),
            "winner": int(g.winner), "game_len": int(g.T), "ply": int(ply),
        }
        rows.append(row)
        stats["emitted"] += 1
    if rec_path and rows:
        with gzip.open(rec_path, "at") as rf:
            for row in rows:
                rf.write(json.dumps(row) + "\n")
    return rows


# --------------------------------------------------------------------------
# Main mining loop: accumulate games across shards into big chunks, GPU-walk
# each chunk, write rows, mark shards processed. Resumable at shard granularity.
# --------------------------------------------------------------------------


def run(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "gpu.log")
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
    log(f"[gpu] start pid={os.getpid()} MAXD={MAXD} cap={args.max_nodes} "
        f"tg={args.tg} chunk_shards={args.chunk_shards} min_run={args.min_run} "
        f"(already processed {len(processed)} shards)")

    while True:
        shards = [p for p in iter_shard_paths(args.games_dir)
                  if os.path.basename(p) not in processed]
        if not shards:
            if args.once:
                break
            el = time.time() - t_start
            log(f"[gpu] idle: no new shards. games={gi} emitted={emitted} "
                f"({gi / el if el else 0:.1f} g/s) sleeping {args.sleep:.0f}s")
            time.sleep(args.sleep)
            continue

        # Process shards in chunks so each GPU walk batches many games at once.
        for ci in range(0, len(shards), args.chunk_shards):
            chunk_shards = shards[ci:ci + args.chunk_shards]
            t0 = time.time()
            games: list[_Game] = []
            c_games = c_draw = 0
            for sp in chunk_shards:
                for g in iter_games(args.games_dir, [sp]):
                    c_games += 1
                    gi += 1
                    w = g.get("winner", -1)
                    if w == -1:
                        c_draw += 1
                        continue
                    rec = reconstruct(g["moves"], int(w), stats)
                    if rec is not None:
                        games.append(rec)
            t_recon = time.time() - t0

            t1 = time.time()
            if games:
                walk_batch(games, max_nodes=args.max_nodes, tg=args.tg, stats=stats)
            t_walk = time.time() - t1

            t2 = time.time()
            rows = finalize_and_emit(
                games, min_run=args.min_run, extract_depth=args.extract_depth,
                extract_nodes=args.extract_nodes, rec_path=rec_path, stats=stats)
            t_extract = time.time() - t2
            emitted += len(rows)

            for sp in chunk_shards:
                processed.add(os.path.basename(sp))
            with open(manifest_path, "a") as mf:
                for sp in chunk_shards:
                    mf.write(os.path.basename(sp) + "\n")

            el = time.time() - t_start
            log(f"[chunk {ci // args.chunk_shards}] shards={len(chunk_shards)} "
                f"games={c_games} (draw={c_draw}) batch={len(games)} "
                f"emitted={len(rows)} | recon={t_recon:.1f}s walk={t_walk:.1f}s "
                f"extract={t_extract:.1f}s | total games={gi} emitted={emitted} "
                f"({gi / el if el else 0:.0f} g/s)")

        if args.once:
            break

    el = time.time() - t_start
    log(f"[gpu] EXIT games={gi} emitted={emitted} elapsed={el:.0f}s "
        f"({gi / el if el else 0:.1f} g/s) "
        f"gpu_calls={stats['gpu_calls']} gpu_boards={stats['gpu_boards']}")
    # Run-length histogram for the report.
    run_hist = {int(k[4:]): v for k, v in stats.items() if k.startswith("run_")}
    log("[gpu] forced_run_hist=" + json.dumps({k: run_hist[k] for k in sorted(run_hist)}))


# --------------------------------------------------------------------------
# End-to-end validation vs the CPU miner (mine_game) on the same games.
# --------------------------------------------------------------------------


def validate(args) -> None:
    from scripts.threat_shapes.mine_vct_backward import mine_game

    shards = iter_shard_paths(args.games_dir)
    # Collect decisive games up to the requested count.
    raw: list[dict] = []
    for g in iter_games(args.games_dir, shards):
        if g.get("winner", -1) != -1:
            raw.append(g)
        if len(raw) >= args.validate_games:
            break
    print(f"[validate] {len(raw)} decisive games; GPU MAXD={MAXD} "
          f"(harness oracle depth={MAXD - 2}); CPU miner depth=10/nodes=100000", flush=True)

    # --- GPU walk on all games (one batched chunk) ---
    stats_g: Counter = Counter()
    t0 = time.time()
    ggames = [reconstruct(g["moves"], int(g["winner"]), stats_g) for g in raw]
    pairs = [(g, gg) for g, gg in zip(raw, ggames) if gg is not None]
    walk_batch([gg for _, gg in pairs], max_nodes=args.max_nodes, tg=args.tg, stats=stats_g)
    gpu_t = time.time() - t0
    # GPU per-game result: (emit?, enable_ply, run) using the SAME min_run.
    gpu_res: dict[int, tuple] = {}
    for idx, (g, gg) in enumerate(pairs):
        if gg.enable_step < 0:
            gpu_res[idx] = (False, None, 0)
        else:
            run_len = gg.enable_step + 1
            ply = gg.L - 2 * gg.enable_step
            gpu_res[idx] = (run_len >= args.min_run, ply, run_len)

    # --- CPU mine_game at depth=10 (production) AND a depth-matched walk ---
    def cpu_walk(depth, nodes):
        out: dict[int, tuple] = {}
        st: Counter = Counter()
        t = time.time()
        for idx, (g, _) in enumerate(pairs):
            rec = mine_game(g["moves"], int(g["winner"]), min_run=args.min_run,
                            max_depth=depth, max_nodes=nodes, stats=st)
            if rec is None:
                out[idx] = (False, None, 0)
            else:
                out[idx] = (True, rec["ply"], rec["run_plies"])
        return out, time.time() - t

    cpu10, cpu10_t = cpu_walk(10, 100_000)
    cpu_deep, cpu_deep_t = cpu_walk(args.extract_depth, args.extract_nodes)

    def compare(cpu, label, cpu_t):
        emit_agree = enable_agree = run_agree = 0
        gpu_only = cpu_only = run_gt = run_lt = 0
        n = len(pairs)
        for idx in range(n):
            ge, gp, gr = gpu_res[idx]
            ce, cp, cr = cpu[idx]
            if ge == ce:
                emit_agree += 1
            if ge and ce:
                if gp == cp:
                    enable_agree += 1
                if gr == cr:
                    run_agree += 1
                elif gr > cr:
                    run_gt += 1
                else:
                    run_lt += 1
            elif ge and not ce:
                gpu_only += 1
            elif ce and not ge:
                cpu_only += 1
        print(f"\n[validate vs CPU {label}] n={n}  cpu_wall={cpu_t:.1f}s", flush=True)
        print(f"  emit agreement:   {emit_agree}/{n} ({emit_agree/n:.3f})", flush=True)
        print(f"  GPU-emit-only:    {gpu_only}   CPU-emit-only: {cpu_only}", flush=True)
        both = sum(1 for idx in range(n) if gpu_res[idx][0] and cpu[idx][0])
        if both:
            print(f"  among {both} both-emit: enable_ply match={enable_agree} "
                  f"run match={run_agree} (GPU run>CPU: {run_gt}, GPU run<CPU: {run_lt})",
                  flush=True)

    print(f"\n[timing] GPU walk: {gpu_t:.2f}s for {len(pairs)} games "
          f"({len(pairs)/gpu_t:.1f} g/s)  |  CPU@10: {cpu10_t:.1f}s "
          f"({len(pairs)/cpu10_t:.2f} g/s)  |  CPU@deep: {cpu_deep_t:.1f}s", flush=True)
    compare(cpu10, "depth=10 (production serial miner)", cpu10_t)
    compare(cpu_deep, f"depth={args.extract_depth} (depth-matched)", cpu_deep_t)
    print(f"\n[validate] GPU gpu_calls={stats_g['gpu_calls']} "
          f"cap_hits={stats_g['gpu_cap_hits']}/{stats_g['gpu_boards']} boards", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--out", default=None, help="output dir (required unless --validate)")
    ap.add_argument("--min-run", type=int, default=3)
    ap.add_argument("--max-nodes", type=int, default=8000,
                    help="GPU megakernel per-solve node cap (batched -> cheap)")
    ap.add_argument("--tg", type=int, default=32, help="GPU threadgroup width")
    ap.add_argument("--chunk-shards", type=int, default=64,
                    help="shards accumulated into one GPU walk (bigger = fatter batch)")
    ap.add_argument("--extract-depth", type=int, default=16,
                    help="CPU solve_vct depth to recover catalyst move + mate dist")
    ap.add_argument("--extract-nodes", type=int, default=200_000)
    ap.add_argument("--sleep", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="compare GPU walk vs CPU mine_game on the same games, then exit")
    ap.add_argument("--validate-games", type=int, default=200)
    args = ap.parse_args()

    if args.validate:
        validate(args)
        return
    if not args.out:
        ap.error("--out is required unless --validate")
    run(args)


if __name__ == "__main__":
    main()
