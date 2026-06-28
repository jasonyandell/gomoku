"""idx-2 VCT frontier expander — append-only, resumable, content-addressed.

THE EXPERIMENT (Jason, 2026-06-28). "Solve" the Bruce-Lee / idx-2 board for
BLACK = find an unstoppable path to a VCT. This is a deliberately MASSIVE
approximation, not a sound proof:

  * Root = idx-2 (15x15, 3-stone fair opening, WHITE to move).
  * The frontier is generated ENTIRELY by the warm Rapfi pool: at every node we
    take Rapfi's top-K moves for the side to move (K=8). Both sides. No legal-move
    enumeration, no minimax, no backed-up AND/OR proof.
  * The mega GPU VCT solver is the ONLY oracle, run on EVERY node (both colors):
      - black-to-move has a forced VCT  -> black_win   (terminus, harvest)
      - white-to-move has a forced VCT  -> black_loss   (black fumbled; terminus, prune)
      - solver hit its node cap          -> capped       (inconclusive; keep expanding)
      - otherwise                        -> quiet        (expand: Rapfi top-K)
    So we hunt a chain of Rapfi-top-8 black moves that Rapfi-top-8 white cannot
    refute, with a VCT (either colour) ending the branch.

WHAT A GREEN FRONTIER WOULD *NOT* MEAN: because white's defences are restricted
to Rapfi's top-8 (the AND nodes are not exhaustively enumerated), even a fully
black-won frontier proves only "black wins vs Rapfi-top-8 white", never a true
solve. The records keep full parent pointers + verdicts so a later pass could
harden any line by exhaustive defence expansion.

DESIGN (reducer-over-a-log, Jason's preferred shape; mirrors
scripts/threat_shapes/md_minimize_stream.py):
  * nodes.jsonl is the ONLY durable state. Each node is content-addressed
    (`id = sha1(D4-canonical board)[:16]`, collapsing transpositions AND the 8
    board symmetries) and written EXACTLY ONCE with its verdict. After a node's
    children are all written we append a tiny `{"id":..,"done":1}` marker.
  * Resume = read the log: `seen` = every id (dedup), frontier = quiet/capped ids
    with no `done` marker. Reconstruct each frontier GameState by replaying its
    stored move list from idx-2. Idempotent: re-expanding a partially-expanded
    parent re-writes only its missing children (the rest are already in `seen`).
  * Caps are RECORDED, never dropped: solver cap -> verdict "capped"; a depth/wall
    cap simply leaves the unexpanded quiet nodes on disk = the frontier.

The two heavy phases are bulk-synchronous (the call-cost law): one parallel Rapfi
sweep per chunk, one <=16384 solver call per chunk. CPU (Rapfi) and GPU (solver)
do not contend.

Run (from worktree root):
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python -m scripts.idx2_vct.frontier \
      --run-dir ~/data/idx2_solve/run-a --max-wall-secs 1800
Resume / extend: re-run the identical command (raise --max-depth / --max-wall-secs).
"""
from __future__ import annotations

import os

os.environ.setdefault("GOMOKU_BOARD_SIZE", "15")

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from gomoku.game import GameState, _sym_board
from gomoku.eval_panel import IDX2_OPENING, fixed_opening_state
from gomoku.rapfi_pool import RapfiPool
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb

OTHER = {"b": "w", "w": "b"}


def light_state(s: GameState) -> GameState:
    """Drop the 8-ply history: Rapfi reads board stones only (full-board replay,
    external_engine.py) and the VCT solver reads `board` only, so history is dead
    weight — and at million-node frontiers it is the whole RAM budget."""
    return GameState(s.board, s.move_count, ())


# --------------------------------------------------------------------------- #
# Content address: D4-canonical, order-independent id for a side-to-move board.
# --------------------------------------------------------------------------- #
def canon_id(board: np.ndarray) -> str:
    """sha1 of the lexicographically-smallest of the 8 D4 images of `board`.

    `board` is (2, N, N) bool, side-to-move-relative (plane 0 = side to move).
    Two positions reached by different move orders, or related by any board
    symmetry, collapse to the same id (side-to-move is implied by stone parity).
    """
    best: bytes | None = None
    for s in range(8):
        key = _sym_board(board, s).tobytes()
        if best is None or key < best:
            best = key
    return hashlib.sha1(best).hexdigest()[:16]


class Node:
    __slots__ = ("id", "parent", "depth", "stm", "move", "moves", "state")

    def __init__(self, id, parent, depth, stm, move, moves, state):
        self.id = id
        self.parent = parent
        self.depth = depth
        self.stm = stm            # side to move at THIS node: 'b' or 'w'
        self.move = move          # action that created this node (parent's move)
        self.moves = moves        # full action list from idx-2 (for reconstruct)
        self.state = state        # transient GameState (not serialized)


def state_from_moves(moves: list[int]) -> GameState:
    s = fixed_opening_state(IDX2_OPENING)   # 15x15, white to move
    for a in moves:
        s = s.apply(int(a))
    return light_state(s)


def make_root() -> Node:
    st = light_state(fixed_opening_state(IDX2_OPENING))
    return Node(canon_id(st.board), None, 0, "w", None, [], st)


def node_record(n: Node, verdict: str, vct_move: int) -> dict:
    return {
        "id": n.id,
        "parent": n.parent,
        "depth": n.depth,
        "stm": n.stm,
        "move": n.move,
        "moves": n.moves,
        "n": int(n.state.board.sum()),
        "verdict": verdict,
        "vct_move": int(vct_move),
    }


# --------------------------------------------------------------------------- #
# Solver: bulk VCT over a board batch, chunked to <= `batch` (call-cost law).
# --------------------------------------------------------------------------- #
def solve_boards(boards: np.ndarray, max_nodes: int, batch: int):
    M = boards.shape[0]
    win = np.empty(M, bool)
    hit = np.empty(M, bool)
    mv = np.empty(M, np.int32)
    for i in range(0, M, batch):
        j = min(i + batch, M)
        w, h, m = solve_vct_mega_bb(boards[i:j], max_nodes=max_nodes, return_move=True)
        win[i:j], hit[i:j], mv[i:j] = w, h, m
    return win, hit, mv


# --------------------------------------------------------------------------- #
# Resume: rebuild `seen` and the live frontier from the append-only log.
# --------------------------------------------------------------------------- #
def read_log(path: Path):
    seen: set[str] = set()
    done: set[str] = set()
    pending: dict[str, list[int]] = {}     # id -> moves, for quiet/capped nodes
    counts: Counter = Counter()
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                # torn final line — ignore
                if r.get("done"):
                    done.add(r["id"])
                    continue
                seen.add(r["id"])
                counts[r["verdict"]] += 1
                if r["verdict"] in ("quiet", "capped"):
                    pending[r["id"]] = r["moves"]
    frontier = [
        Node(i, None, len(mv), "w" if len(mv) % 2 == 0 else "b", None, mv,
             state_from_moves(mv))
        for i, mv in pending.items() if i not in done
    ]
    return seen, done, frontier, counts


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="output dir (nodes.jsonl + run.json); out of git")
    ap.add_argument("--k", type=int, default=8, help="Rapfi top-K per node")
    ap.add_argument("--rapfi-max-node", type=int, default=2000,
                    help="Rapfi cost knob (binds wall-time; 2000 = peak children/s)")
    ap.add_argument("--timeout-ms", type=int, default=1000,
                    help="Rapfi per-move timeout (generous; max_node binds first)")
    ap.add_argument("--pool", type=int, default=12, help="Rapfi pool size")
    ap.add_argument("--max-nodes", type=int, default=250, help="VCT solver node cap")
    ap.add_argument("--batch", type=int, default=16384, help="solver batch cap")
    ap.add_argument("--pool-chunk", type=int, default=2048,
                    help="parents analysed per Rapfi sweep (<= batch/K children)")
    ap.add_argument("--max-depth", type=int, default=40, help="frontier depth cap")
    ap.add_argument("--max-wall-secs", type=float, default=1800.0,
                    help="soft wall-clock cap (stoppable/resumable any time)")
    ap.add_argument("--max-frontier", type=int, default=20_000_000,
                    help="stop if a level exceeds this many nodes")
    args = ap.parse_args()

    run_dir = Path(os.path.expanduser(args.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "nodes.jsonl"
    cfg = {k: getattr(args, k) for k in vars(args)}
    cfg["board_size"] = N
    cfg["idx2_opening"] = IDX2_OPENING
    cfg["launched"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (run_dir / "run.json").write_text(json.dumps(cfg, indent=2))

    seen, done, frontier, counts = read_log(log_path)
    t_start = time.time()
    deadline = t_start + args.max_wall_secs

    with RapfiPool(size=args.pool, timeout_ms=args.timeout_ms, board_size=N) as pool, \
            log_path.open("a") as fh:

        def emit(rec: dict):
            fh.write(json.dumps(rec) + "\n")

        # Fresh start: solve + record the root, seed the frontier with it.
        if not seen and not frontier:
            root = make_root()
            w, h, mv = solve_boards(root.state.board.astype(bool)[None], args.max_nodes,
                                    args.batch)
            verdict = ("black_loss" if w[0] else "capped" if h[0] else "quiet")
            emit(node_record(root, verdict, mv[0] if w[0] else -1))
            seen.add(root.id)
            counts[verdict] += 1
            frontier = [root] if verdict in ("quiet", "capped") else []
            fh.flush()
            os.fsync(fh.fileno())

        print(f"[idx2] resume: {len(seen)} nodes seen, frontier={len(frontier)}, "
              f"verdicts={dict(counts)}", flush=True)

        level = frontier
        next_level: list[Node] = []
        stop = ""
        while level:
            depth = level[0].depth
            if depth > args.max_depth:
                stop = f"max-depth {args.max_depth}"
                break
            if len(level) > args.max_frontier:
                stop = f"max-frontier {args.max_frontier} (level={len(level)})"
                break
            t_lvl = time.time()
            lvl_counts: Counter = Counter()
            created = skipped = 0
            next_level: list[Node] = []

            for chunk in chunks(level, args.pool_chunk):
                maps = pool.analyze_states([n.state for n in chunk],
                                           max_node=args.rapfi_max_node, max_pv=args.k)
                children: list[Node] = []
                for n, m in zip(chunk, maps):
                    topk = sorted(m, key=m.get, reverse=True)[:args.k]
                    for a in topk:
                        cs = light_state(n.state.apply(int(a)))
                        cid = canon_id(cs.board)
                        if cid in seen:
                            skipped += 1
                            continue
                        seen.add(cid)
                        children.append(Node(cid, n.id, n.depth + 1, OTHER[n.stm],
                                             int(a), n.moves + [int(a)], cs))
                created += len(children)

                # Immediate 5-in-a-row (the mover = parent's stm just won) -> terminus.
                term, tosolve = [], []
                for c in children:
                    if c.state.is_terminal()[0]:
                        term.append(c)
                    else:
                        tosolve.append(c)

                if tosolve:
                    boards = np.stack([c.state.board.astype(bool) for c in tosolve])
                    win, hit, mvz = solve_boards(boards, args.max_nodes, args.batch)
                else:
                    win = hit = mvz = np.empty(0)

                for c in term:
                    mover = OTHER[c.stm]            # parent's stm made the 5
                    verdict = "black_win" if mover == "b" else "black_loss"
                    emit(node_record(c, verdict, c.move))
                    lvl_counts[verdict] += 1

                for c, w, h, m in zip(tosolve, win, hit, mvz):
                    if w:
                        verdict = "black_win" if c.stm == "b" else "black_loss"
                        vm = int(m)
                    elif h:
                        verdict, vm = "capped", -1
                    else:
                        verdict, vm = "quiet", -1
                    emit(node_record(c, verdict, vm))
                    lvl_counts[verdict] += 1
                    if verdict in ("quiet", "capped"):
                        next_level.append(c)

                for n in chunk:                    # mark expanded parents done
                    emit({"id": n.id, "done": 1})
                fh.flush()
                os.fsync(fh.fileno())

                if time.time() > deadline:
                    stop = f"max-wall-secs {args.max_wall_secs:.0f}"
                    break

            for k, v in lvl_counts.items():
                counts[k] += v
            dt = time.time() - t_lvl
            print(f"[depth {depth:>2} -> {depth+1}] expanded={len(level)} created={created} "
                  f"dedup_skip={skipped} | win={lvl_counts['black_win']} "
                  f"loss={lvl_counts['black_loss']} cap={lvl_counts['capped']} "
                  f"quiet={lvl_counts['quiet']} | next={len(next_level)} "
                  f"{dt:.1f}s ({created/max(dt,1e-9):.0f} child/s)", flush=True)

            if stop:
                break
            level = next_level

        elapsed = time.time() - t_start
        if not level and not stop:
            stop = "frontier exhausted"
        print(f"[idx2] STOP: {stop} | total nodes={len(seen)} verdicts={dict(counts)} "
              f"elapsed={elapsed:.0f}s "
              f"(frontier recoverable from {log_path})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
