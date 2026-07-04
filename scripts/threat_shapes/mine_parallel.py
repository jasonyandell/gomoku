"""Parallel CPU VCT-mining harness -- fan out across the Mac's cores to mine
VCT-positive positions (with their replayed threat structure) at throughput.

WHY THIS EXISTS
---------------
``build_corpus.py`` is the proven SINGLE-core miner (GPU VCF funnel + CPU
``solve_vct`` confirm). The first pipeline run showed the INTERESTING cases are
RARE -- ~7 pure three-only VCT mates, ~17 non-collinear 2-D forks, residual
corpus only 235. To cross those into statistical significance we need 50-100x
more positions. We don't need one fast solver; we need ~12 cores grinding
continuously. This module parallelizes the mine into a sharded, stateless,
crash-robust multiprocess harness.

ARCHITECTURE (stateless / sharded -- no shared mutable state)
-------------------------------------------------------------
``--workers N`` independent processes. Each worker, forever until its deadline:
  1. GENERATE dense midgame boards with ``build_corpus.gen_gravity`` (its own
     seed stream: ``seed0 + wid*1_000_000 + chunk`` -> no two workers do the same
     work).
  2. CHEAP CPU PRE-FILTER ``has_forcing_candidate`` -- reject boards where the
     side-to-move has no four-making and no forcing-three move. SOUND: every VCT
     win's first move is a forcing move, so this never drops a real win. Kills
     the dense-but-quiet majority before the expensive solve.
  3. CONFIRM with ``gomoku.vcf.solve_vct`` (a strict superset of VCF: finds both
     four-mates and continuous-threes mates in one call).
  4. REPLAY the win with ``threats.replay_vct`` -> (gain,cost,rest) threat list +
     dependency structure, and tag the valuable buckets (below).
  5. WRITE to its OWN shard pair (``shard_w{wid}_{seq}.npz`` + a sidecar
     ``.replay.pkl``) via temp-file + atomic ``os.replace`` -- zero cross-process
     contention, and a SIGKILL mid-write never leaves a torn shard a reader sees.
Any worker is killable/restartable; progress == the accumulated shards (loops are
cadence, not load-bearing).

MPS vs CPU-ONLY -- THE DECISION
-------------------------------
**CPU-ONLY.** 12 processes each hammering ``solve_vcf_batch`` on the single MPS
device would serialize on the GPU and thrash -- the opposite of throughput. The
mine is embarrassingly parallel on the CPU: ``solve_vct`` is pure-NumPy/Python and
returns fast on the pre-filtered majority. So every worker stays on the CPU, the
GPU is left entirely free for training / the derby / residual.py's value head.
Cost: we give up the GPU's ~130k/s bulk VCF pre-filter, but ``has_forcing_candidate``
is a cheap CPU stand-in and the cores more than make up for it. (See the harness
docstring in the module header and the deliverable notes.)

VALUABLE BUCKETS (tagged per position so the reducer can report them)
---------------------------------------------------------------------
  (i)   vct            -- every VCT-positive, with its full replayed threat structure.
  (ii)  three_only     -- mate whose WHOLE forcing line is threes (kinds in
                          {three, three_fork}); the rare pure continuous-threes win.
  (iii) noncollinear   -- a fork (double_four / three_fork) whose completion squares
                          span >= 2 axes through the gain == genuine 2-D combination
                          geometry (reuses finalize.fork_geometry's axis test).
  (iv)  residual_cand  -- OPTIONAL (--residual): forcing-rich boards where solve_vct
                          hit its node/depth cap WITHOUT a proof -> "won-but-no-
                          forcing-proof" frontier candidates. Label is APPROXIMATE
                          (cap = unproven, not proven-won); the SOUND residual
                          corpus stays with residual.py (GPU value head). Flagged.

REDUCE (--reduce)
-----------------
Merge all shards, D4-dedup via the canonical board key (reuses ``game._sym_board``,
the same D4 group the trainer augments with), emit:
  * ``corpus.npz``  (boards, moves, tags, mds) -- the build_corpus format, so
    ``run.py`` / ``finalize.py`` consume it unchanged.
  * ``replay.pkl``  (list of (board, tag, ReplayResult)) -- so ``run.py
    --reuse-replay`` and ``finalize.py`` skip re-replaying.
  * ``mine_stats.json`` -- bucket counts + per-minute / per-hour extrapolations.

USAGE
-----
  # small polite smoke test (build + verify):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_parallel \
      --out <dir> --workers 4 --duration 60

  # the full mine -- light up the cores and walk away:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_parallel \
      --out <dir> --workers 12 --duration 3600 [--residual]

  # fold the shards into a corpus (stateless; re-runnable anytime, even mid-mine):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.mine_parallel \
      --reduce --out <dir>
"""

from __future__ import annotations

# BOARD_SIZE is read at gomoku import; pin it before anything imports gomoku so
# spawned children (macOS default start method) resolve 15x15 too.
import os
os.environ.setdefault("GOMOKU_BOARD_SIZE", "15")

import argparse
import glob
import hashlib
import json
import pickle
import signal
import time
from multiprocessing import Process

import numpy as np

from gomoku.game import _sym_board
from gomoku.vcf import (
    _candidate_cells_from_planes,
    _collinear_empties,
    _completions_through,
    _empties_from_plane,
    _open_four_threats,
    solve_vct,
)
from scripts.threat_shapes.build_corpus import N, gen_gravity
from scripts.threat_shapes.threats import rc, replay_vct


# ---------------------------------------------------------------------------
# Cheap CPU pre-filter (SOUND: a VCT win's first move is always forcing).
# ---------------------------------------------------------------------------
def has_forcing_candidate(board: np.ndarray) -> bool:
    """True iff the side-to-move has a move that makes a four OR a forcing three.

    Mirrors ``build_corpus.has_forcing_three`` but ACCEPTS fours too (so it gates
    VCF-style four-mates as well as three-mates). Never mutates ``board``.
    """
    atk = board[0].copy()
    dfd = board[1]
    occ = atk | dfd
    empty = ~occ
    for m in _candidate_cells_from_planes(atk, dfd, _empties_from_plane(empty)):
        mr, mc = int(m) // N, int(m) % N
        atk[mr, mc] = True
        occ2 = atk | dfd
        if _completions_through(atk, int(m), occ2):     # makes a four
            atk[mr, mc] = False
            return True
        ne = ~occ2
        th = _open_four_threats(atk, dfd, ne, _collinear_empties(int(m), ne))
        atk[mr, mc] = False
        if th:                                            # makes a forcing three
            return True
    return False


# ---------------------------------------------------------------------------
# Bucket classification from a replayed forcing line.
# ---------------------------------------------------------------------------
_THREE_KINDS = frozenset({"three", "three_fork"})
_FORK_KINDS = frozenset({"double_four", "three_fork"})


def _fork_is_noncollinear(threat) -> bool:
    """A fork whose completion (cost) squares span >= 2 axes through the gain."""
    if threat.kind not in _FORK_KINDS:
        return False
    axes = set()
    gr, gc = rc(threat.gain)
    for c in threat.cost:
        cr, cc = rc(c)
        dr, dc = cr - gr, cc - gc
        if dr == 0 and dc == 0:
            continue
        g = np.gcd(abs(dr), abs(dc)) or 1
        axes.add((dr // g, dc // g) if (dr, dc) >= (0, 0) else (-dr // g, -dc // g))
    return len(axes) >= 2


def classify(replay) -> dict:
    """Per-position bucket flags + threat-kind tallies from a ReplayResult."""
    ts = replay.threats
    n_fours = sum(t.kind in ("four", "double_four") for t in ts)
    n_threes = sum(t.kind in _THREE_KINDS for t in ts)
    three_only = bool(ts) and all(t.kind in _THREE_KINDS for t in ts)
    noncollinear = any(_fork_is_noncollinear(t) for t in ts)
    return {"three_only": three_only, "noncollinear": noncollinear,
            "n_fours": int(n_fours), "n_threes": int(n_threes),
            "n_threats": len(ts)}


# ---------------------------------------------------------------------------
# D4-canonical board key (reuses the trainer's D4 group via _sym_board).
# ---------------------------------------------------------------------------
def canonical_board_key(board: np.ndarray) -> bytes:
    """16-byte dedup key for ``board``'s D4 class: blake2b of the min-bytes image."""
    best = None
    for sym in range(8):
        b = np.ascontiguousarray(_sym_board(board, sym)).tobytes()
        if best is None or b < best:
            best = b
    return hashlib.blake2b(best, digest_size=16).digest()


# ---------------------------------------------------------------------------
# Shard writer (atomic; npz primary + sidecar replay pkl).
# ---------------------------------------------------------------------------
def _next_seq(out_dir: str, wid: int) -> int:
    seqs = []
    for p in glob.glob(os.path.join(out_dir, f"shard_w{wid}_*.npz")):
        try:
            seqs.append(int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]))
        except (ValueError, IndexError):
            pass
    return (max(seqs) + 1) if seqs else 0


def _atomic_npz(path: str, **arrays) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:           # file handle -> exact path (no .npz suffixing)
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def _atomic_pickle(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def flush_shard(out_dir: str, wid: int, rows: list) -> int:
    """Write one shard pair from buffered rows. ``rows`` = list of dicts.

    npz holds the queryable corpus (boards/moves/tags/mds/keys + bucket flags);
    the sidecar ``.replay.pkl`` holds the aligned ReplayResult list so the shape /
    motif stages reuse it without re-replaying.
    """
    if not rows:
        return 0
    seq = _next_seq(out_dir, wid)
    boards = np.stack([r["board"] for r in rows]).astype(bool)
    moves = np.array([r["move"] for r in rows], dtype=np.int64)
    tags = np.array([r["tag"] for r in rows])
    mds = np.array([r["md"] if r["md"] is not None else -1 for r in rows], dtype=np.int64)
    keys = np.stack([np.frombuffer(r["key"], dtype=np.uint8) for r in rows])
    three_only = np.array([r["three_only"] for r in rows], dtype=bool)
    noncollinear = np.array([r["noncollinear"] for r in rows], dtype=bool)
    n_fours = np.array([r["n_fours"] for r in rows], dtype=np.int32)
    n_threes = np.array([r["n_threes"] for r in rows], dtype=np.int32)

    base = os.path.join(out_dir, f"shard_w{wid}_{seq}")
    _atomic_pickle(base + ".replay.pkl",
                   [(r["board"], r["tag"], r["replay"]) for r in rows])
    _atomic_npz(base + ".npz", boards=boards, moves=moves, tags=tags, mds=mds,
                keys=keys, three_only=three_only, noncollinear=noncollinear,
                n_fours=n_fours, n_threes=n_threes)
    return len(rows)


# ---------------------------------------------------------------------------
# Worker.
# ---------------------------------------------------------------------------
def worker(wid: int, out_dir: str, deadline: float, seed0: int, chunk: int,
           vct_depth: int, vct_nodes: int, residual: bool,
           flush_every: int) -> None:
    # Stateless graceful stop: SIGTERM just flips the deadline to now; the loop
    # flushes its buffer and exits. (A killed worker loses only an unflushed
    # buffer -- never a written shard.)
    stop = {"deadline": deadline}

    def _term(_sig, _frm):
        stop["deadline"] = 0.0
    signal.signal(signal.SIGTERM, _term)

    rng_seed_base = seed0 + wid * 1_000_000
    chunk_i = 0
    rows: list = []
    last_flush = time.time()
    while time.time() < stop["deadline"]:
        boards = gen_gravity(chunk, rng_seed_base + chunk_i)
        chunk_i += 1
        for b in boards:
            if time.time() >= stop["deadline"]:  # per-board check: prompt, clean exit
                break
            if not has_forcing_candidate(b):
                continue
            r = solve_vct(b, max_depth=vct_depth, max_nodes=vct_nodes)
            if r.has_forced_win and r.winning_move is not None:
                rep = replay_vct(b, solver="vct", max_plies=8,
                                 vct_depth=vct_depth, vct_nodes=vct_nodes)
                buckets = classify(rep)
                rows.append({"board": np.asarray(b, dtype=bool),
                             "move": int(r.winning_move), "md": r.mate_distance,
                             "tag": "vct", "key": canonical_board_key(b),
                             "replay": rep, **buckets})
            elif residual and r.hit_cap:
                # forcing-rich but unproven within budget -> approximate residual
                # candidate (won-but-no-forcing-proof frontier; flagged below).
                rows.append({"board": np.asarray(b, dtype=bool),
                             "move": -1, "md": -1, "tag": "residual_cand",
                             "key": canonical_board_key(b), "replay": None,
                             "three_only": False, "noncollinear": False,
                             "n_fours": 0, "n_threes": 0})
            if len(rows) >= flush_every or (rows and time.time() - last_flush > 30):
                flush_shard(out_dir, wid, rows)
                rows = []
                last_flush = time.time()
    flush_shard(out_dir, wid, rows)


# ---------------------------------------------------------------------------
# Live progress (reads shards -- the only source of truth).
# ---------------------------------------------------------------------------
def _scan_counts(out_dir: str) -> dict:
    tot = vct = three = noncoll = resid = 0
    for p in glob.glob(os.path.join(out_dir, "shard_w*_*.npz")):
        try:
            with np.load(p, allow_pickle=False) as z:
                tags = z["tags"]
                tot += len(tags)
                is_vct = tags == "vct"
                vct += int(is_vct.sum())
                three += int((z["three_only"] & is_vct).sum())
                noncoll += int((z["noncollinear"] & is_vct).sum())
                resid += int((tags == "residual_cand").sum())
        except (OSError, KeyError, ValueError):
            continue
    return {"rows": tot, "vct": vct, "three_only": three,
            "noncollinear": noncoll, "residual_cand": resid}


# ---------------------------------------------------------------------------
# Reducer (stateless merge over shards).
# ---------------------------------------------------------------------------
def reduce(out_dir: str) -> dict:
    seen: set[bytes] = set()
    boards, moves, tags, mds = [], [], [], []
    replays = []
    three_only = noncollinear = resid = dup = 0
    md_hist: dict[int, int] = {}
    npz_paths = sorted(glob.glob(os.path.join(out_dir, "shard_w*_*.npz")))
    for p in npz_paths:
        try:
            z = np.load(p, allow_pickle=False)
        except (OSError, ValueError):
            continue
        pkl_path = p[:-4] + ".replay.pkl"
        try:
            with open(pkl_path, "rb") as f:
                rep_list = pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError):
            rep_list = [(b, t, None) for b, t in zip(z["boards"], z["tags"])]
        keys = z["keys"]
        for i in range(len(z["tags"])):
            k = keys[i].tobytes()
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            tag = str(z["tags"][i])
            boards.append(z["boards"][i])
            moves.append(int(z["moves"][i]))
            tags.append(tag)
            md = int(z["mds"][i])
            mds.append(md)
            replays.append(rep_list[i])
            if tag == "vct":
                md_hist[md] = md_hist.get(md, 0) + 1
                if bool(z["three_only"][i]):
                    three_only += 1
                if bool(z["noncollinear"][i]):
                    noncollinear += 1
            elif tag == "residual_cand":
                resid += 1
        z.close()

    if boards:
        np.savez_compressed(os.path.join(out_dir, "corpus.npz"),
                            boards=np.stack(boards).astype(bool),
                            moves=np.array(moves, dtype=np.int64),
                            tags=np.array(tags),
                            mds=np.array(mds, dtype=np.int64))
        with open(os.path.join(out_dir, "replay.pkl"), "wb") as f:
            pickle.dump(replays, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_vct = sum(t == "vct" for t in tags)
    stats = {
        "n_unique_total": len(boards),
        "n_vct": int(n_vct),
        "n_three_only_mates": int(three_only),
        "n_noncollinear_forks": int(noncollinear),
        "n_residual_candidates": int(resid),
        "residual_label": "APPROXIMATE -- solve_vct hit_cap (unproven, not proven-won)",
        "n_duplicates_dropped_d4": int(dup),
        "n_shards": len(npz_paths),
        "md_hist": {int(k): int(v) for k, v in sorted(md_hist.items())},
        "corpus": os.path.join(out_dir, "corpus.npz"),
        "replay": os.path.join(out_dir, "replay.pkl"),
    }
    with open(os.path.join(out_dir, "mine_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="shard + corpus directory")
    ap.add_argument("--reduce", action="store_true",
                    help="merge shards -> corpus.npz + replay.pkl + mine_stats.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--duration", type=float, default=600, help="seconds to mine")
    ap.add_argument("--seed0", type=int, default=10_000)
    ap.add_argument("--chunk", type=int, default=400, help="boards generated per gen call")
    ap.add_argument("--vct-depth", type=int, default=7)
    ap.add_argument("--vct-nodes", type=int, default=12_000)
    ap.add_argument("--flush-every", type=int, default=64,
                    help="buffer this many hits before writing a shard")
    ap.add_argument("--residual", action="store_true",
                    help="also store forcing-rich hit_cap boards as residual candidates")
    ap.add_argument("--report-every", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.reduce:
        reduce(args.out)
        return

    start = time.time()
    deadline = start + args.duration
    base0 = _scan_counts(args.out)        # pre-existing shards (resume-friendly)
    procs = []
    for wid in range(args.workers):
        p = Process(target=worker, args=(
            wid, args.out, deadline, args.seed0, args.chunk,
            args.vct_depth, args.vct_nodes, args.residual, args.flush_every))
        p.start()
        procs.append(p)
    print(f"[mine] {args.workers} workers, {args.duration:.0f}s, out={args.out}",
          flush=True)

    last = 0.0
    while any(p.is_alive() for p in procs):
        time.sleep(1.0)
        now = time.time()
        if now - last >= args.report_every:
            last = now
            c = _scan_counts(args.out)
            el = now - start
            dv = c["vct"] - base0["vct"]
            rate = dv / el * 60 if el > 0 else 0
            print(f"[mine] t={el:5.0f}s  vct={c['vct']:5d} (+{dv}) "
                  f"three_only={c['three_only']} noncoll={c['noncollinear']} "
                  f"resid={c['residual_cand']}  rate={rate:.1f} vct/min", flush=True)
    for p in procs:
        p.join()

    el = time.time() - start
    c = _scan_counts(args.out)
    dv = c["vct"] - base0["vct"]
    print(f"\n[mine] DONE {el:.0f}s on {args.workers} workers", flush=True)
    print(f"  VCT mined this run: {dv}  ({dv/el*60:.1f} vct/min, "
          f"{dv/el*3600:.0f} vct/hr extrapolated)", flush=True)
    print(f"  totals on disk: vct={c['vct']} three_only={c['three_only']} "
          f"noncoll={c['noncollinear']} residual_cand={c['residual_cand']}", flush=True)
    print(f"  reduce with: uv run python -m scripts.threat_shapes.mine_parallel "
          f"--reduce --out {args.out}", flush=True)


if __name__ == "__main__":
    main()
