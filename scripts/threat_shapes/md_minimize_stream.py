"""Append-only, trivially-resumable streaming md-invariant stencil minimizer.

The batch `md_minimize.py` loads top-N, minimizes, writes one dump. That doesn't
scale to n=1000+ deep boards (the no-window ablation is slow on deep VCTs — #92) and
isn't resumable: a crash loses everything. This wraps the SAME `minimize()` core in a
**reducer-over-a-log** design (Jason's preferred shape):

  * The output JSONL is the ONLY state. Each board is content-addressed
    (`id = sha1(corpus|atk|dfd)`) and written EXACTLY ONCE with a status
    (`ok` = minimized stencil / `capped` = budget hit, no stencil / `dead` = clean
    no-win). No in-process durable state; the loop is cadence, not load-bearing.
  * Resume = read the log, skip every id already present, keep going. Idempotent:
    re-running with the same config is a no-op once the target is met. A crash
    mid-chunk loses only the in-flight chunk (those ids simply aren't in the log yet).
  * Unbounded-n later = raise --target (or drop it) and re-run; the stream picks up
    where the log left off and appends. `capped` ids are recorded so they are NOT
    retried — re-attempt them deliberately at a higher budget via a fresh --out-name.

Boards are processed deepest-first (corpus sort key) in bulk chunks (the flat-in-B
call-cost law), appending + fsync'ing after each chunk so the log is always a valid
prefix.

Run (from worktree root, GPU):
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python -m scripts.threat_shapes.md_minimize_stream \
      --corpus enable --target 1000 --max-nodes 3000 --hi 18 --chunk 256
Resume: re-run the identical command. Analyse: scripts.threat_shapes.analyze_vocab_stream
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from scripts.threat_shapes.md_minimize import CORPORA, minimize
from scripts.vct_metal.mega_vct_bb import (
    cells_from_words, solve_md_min)


def board_id(corpus: str, atk, dfd) -> str:
    """Content-addressed, order-independent board id (stable across runs/corpora)."""
    key = f"{corpus}|{','.join(map(str, sorted(atk)))}|{','.join(map(str, sorted(dfd)))}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def load_rows(corpus: str, order: str = "shuffled", seed: int = 0):
    """Deterministic row order (so resume processes the same boards in the same
    sequence). `deep` = corpus deepest-setup-first (hardest, low md0-resolve yield);
    `natural` = file order; `shuffled` = seeded shuffle (representative sample)."""
    path, sortkey = CORPORA[corpus]
    rows = []
    with gzip.open(os.path.expanduser(path), "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    if order == "deep":
        rows.sort(key=lambda r: -int(r.get(sortkey, 0)))
    elif order == "shuffled":
        import random
        random.Random(seed).shuffle(rows)
    return rows


def read_done(logpath: Path):
    """Return (done_ids set, n_ok) from the append-only log. Tolerates a torn
    final line (a crash mid-write) by skipping unparseable lines."""
    done, n_ok = set(), 0
    if not logpath.exists():
        return done, n_ok
    with logpath.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                       # torn last line — ignore
            done.add(r["id"])
            if r.get("status") == "ok":
                n_ok += 1
    return done, n_ok


def boards_from(rows):
    B = len(rows)
    boards = np.zeros((B, 2, N, N), bool)
    for i, r in enumerate(rows):
        boards[i, 0].reshape(-1)[np.asarray(r["atk"], np.int64)] = True
        boards[i, 1].reshape(-1)[np.asarray(r["dfd"], np.int64)] = True
    return boards


def process_chunk(corpus, chunk_rows, chunk_ids, max_nodes, hi):
    """Minimize one bulk chunk; yield one record dict per board (ok/capped/dead)."""
    boards = boards_from(chunk_rows)
    B = boards.shape[0]
    n_orig = boards.reshape(B, -1).sum(1).astype(int)
    md0, capped = solve_md_min(boards, max_nodes=max_nodes, hi=hi)
    alive = (md0 >= 1) & (~capped)

    recs = [None] * B
    for b in range(B):
        if capped[b]:
            recs[b] = dict(id=chunk_ids[b], corpus=corpus, status="capped",
                           n_orig=int(n_orig[b]))
        elif not alive[b]:
            recs[b] = dict(id=chunk_ids[b], corpus=corpus, status="dead",
                           n_orig=int(n_orig[b]))

    aidx = np.where(alive)[0]
    if aidx.size:
        res, _ = minimize(boards[aidx], md0[aidx], max_nodes)
        own, opp, supp = res["own"], res["opp"], res["supp"]
        carr, wch, move = res["carr"], res["wch"], res["move"]
        for j, b in enumerate(aidx):
            B_cells = [int(c) for c in np.flatnonzero(own[j])]
            W_cells = [int(c) for c in np.flatnonzero(opp[j])]
            s_cells = cells_from_words(supp[j])
            anchor = B_cells + W_cells + s_cells
            if not anchor:                      # degenerate (shouldn't happen) -> mark dead
                recs[b] = dict(id=chunk_ids[b], corpus=corpus, status="dead",
                               n_orig=int(n_orig[b]))
                continue
            r0 = min(c // N for c in anchor)
            c0 = min(c % N for c in anchor)
            mv = int(move[j])
            recs[b] = dict(
                id=chunk_ids[b], corpus=corpus, status="ok",
                md0=int(md0[b]), n_orig=int(n_orig[b]),
                move_rel=[mv // N - r0, mv % N - c0],
                B_rel=sorted([c // N - r0, c % N - c0] for c in B_cells),
                W_rel=sorted([c // N - r0, c % N - c0] for c in W_cells),
                support_rel=sorted([c // N - r0, c % N - c0] for c in s_cells),
                w_over_n=len(cells_from_words(wch[j])),
                carr_n=len(cells_from_words(carr[j])),
            )
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(CORPORA), default="enable")
    ap.add_argument("--target", type=int, default=1000,
                    help="stop once this many status=ok stencils exist (0 = exhaust input)")
    ap.add_argument("--max-nodes", type=int, default=3000)
    ap.add_argument("--hi", type=int, default=18)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--order", choices=["deep", "natural", "shuffled"],
                    default="shuffled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="~/data/md_stencils")
    ap.add_argument("--out-name", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    logpath = out_dir / (args.out_name or f"stream_{args.corpus}.jsonl")

    t0 = time.time()
    rows = load_rows(args.corpus, order=args.order, seed=args.seed)
    done, n_ok = read_done(logpath)
    print(f"[init] corpus={args.corpus} order={args.order} rows={len(rows)}  "
          f"log={logpath.name} already-done={len(done)} (ok={n_ok})  "
          f"target={args.target}", flush=True)

    fh = logpath.open("a")
    buf_rows, buf_ids = [], []
    n_seen = n_cap = n_dead = 0
    target = args.target

    def flush_chunk():
        nonlocal n_ok, n_cap, n_dead
        if not buf_rows:
            return
        t = time.time()
        recs = process_chunk(args.corpus, buf_rows, buf_ids, args.max_nodes, args.hi)
        for r in recs:
            fh.write(json.dumps(r) + "\n")
            if r["status"] == "ok":
                n_ok += 1
            elif r["status"] == "capped":
                n_cap += 1
            else:
                n_dead += 1
        fh.flush()
        os.fsync(fh.fileno())
        nok = sum(r["status"] == "ok" for r in recs)
        print(f"[chunk] +{len(recs)} ({nok} ok) in {time.time()-t:.0f}s | "
              f"cum ok={n_ok} cap={n_cap} dead={n_dead} | {time.time()-t0:.0f}s elapsed",
              flush=True)
        buf_rows.clear()
        buf_ids.clear()

    for r in rows:
        if target and n_ok >= target:
            break
        bid = board_id(args.corpus, r["atk"], r["dfd"])
        if bid in done:
            continue
        done.add(bid)
        buf_rows.append(r)
        buf_ids.append(bid)
        n_seen += 1
        if len(buf_rows) >= args.chunk:
            flush_chunk()
            if target and n_ok >= target:
                break
    flush_chunk()
    fh.close()

    reached = "TARGET MET" if (target and n_ok >= target) else "input exhausted"
    print(f"[done] {reached}: ok={n_ok} (+{n_seen} newly processed this run: "
          f"{n_cap} capped, {n_dead} dead) in {time.time()-t0:.0f}s -> {logpath}",
          flush=True)


if __name__ == "__main__":
    main()
