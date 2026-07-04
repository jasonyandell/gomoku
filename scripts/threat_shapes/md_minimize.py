"""md-INVARIANT stencil minimizer (the L1 reduction unblocked by md-extraction).

The shape-library §3 program: reduce a proven-VCT board to the minimal typed
stencil that still guarantees *this* VCT, by **mate-distance-invariant context
ablation** (NOT "still a win" — `shape-library-engine.md` §3 correction #2). A
stone is *load-bearing* iff removing it changes the mate distance md0; a stone
whose removal preserves md0 is a don't-care.

This is now possible because `solve_vct_mega_bb(max_depth=)` (issue #91) yields the
order-independent md_min on GPU. The directional single-cap tests (exploiting
freestyle monotonicity — removing OWN can only preserve/lengthen md, removing OPP
can only preserve/shorten):

  * OWN stone c  — probe R−c at cap **md0**:   clean win => md0 preserved => DROP
                                               clean nowin => load-bearing `B`, KEEP
  * OPP stone c  — probe R−c at cap **md0−1**: clean win => a shorter win opened
                                               => load-bearing `W`, KEEP
                                               clean nowin => irrelevant => DROP
  * hit_cap on any probe => KEEP (fail-safe; md never silently wrong, only withheld)

Cumulative greedy in lockstep (one bulk call per ablation step, all boards march
their own candidate + own per-board max_depth — the flat-in-B call-cost law). Each
DROP provably preserves md_min(R)==md0 by monotonicity, so the invariant holds
under cumulative dropping. No windowing: every occupied stone is tested
individually (sound; sidesteps the found-line-vs-shortest-line windowing risk).

Run (from worktree root, GPU):
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python -m scripts.threat_shapes.md_minimize \
      --corpus molecule --n 16384 --max-nodes 2000
  ... --corpus enable --n 16384 ...
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.vct_metal.mega_vct_bb import (
    solve_vct_mega_bb, solve_md_min, cells_from_words)
from gomoku.board_config import BOARD_SIZE as N

CORPORA = {
    "molecule": ("~/data/molecule_gold/gold.jsonl.gz", "dist"),
    "enable": ("~/data/vct_shapes/enable_serial.jsonl.gz", "run"),
}


def canon(cells) -> tuple:
    """Translation-free relative cell set (bbox-min anchored). Matches
    mine_support_shapes.canon for cross-comparable vocabulary stats."""
    if not cells:
        return ()
    rc = [(c // N, c % N) for c in cells]
    r0 = min(r for r, _ in rc)
    c0 = min(c for _, c in rc)
    return tuple(sorted((r - r0, c - c0) for r, c in rc))


def load_corpus(name, n):
    path, sortkey = CORPORA[name]
    rows = []
    with gzip.open(os.path.expanduser(path), "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    total = len(rows)
    rows.sort(key=lambda r: -int(r.get(sortkey, 0)))   # deepest setups first
    rows = rows[:n]
    B = len(rows)
    boards = np.zeros((B, 2, N, N), bool)
    for i, r in enumerate(rows):
        boards[i, 0].reshape(-1)[np.asarray(r["atk"], np.int64)] = True
        boards[i, 1].reshape(-1)[np.asarray(r["dfd"], np.int64)] = True
    return boards, rows, total


def cheb(c, mv):
    return max(abs(c // N - mv // N), abs(c % N - mv % N))


def minimize(boards, md0, max_nodes):
    """Cumulative lockstep md-invariant ablation. Returns dict of per-board arrays:
    own/opp (B,N*N) bool surviving masks (B-channel / W-channel), plus the seed
    support/carriers/w/move from the full-board found line."""
    B = boards.shape[0]
    own = boards[:, 0].reshape(B, -1).copy()
    opp = boards[:, 1].reshape(B, -1).copy()
    win, hit, move, supp, carr, wch = solve_vct_mega_bb(
        boards, max_nodes=max_nodes, return_move=True,
        return_support=True, return_carriers=True, return_w=True)

    # per-board candidate order = all occupied cells, farthest-from-move first
    cand = []
    for b in range(B):
        mv = int(move[b]) if move[b] >= 0 else 0
        occ = list(np.flatnonzero(own[b] | opp[b]))
        occ.sort(key=lambda c: -cheb(int(c), mv))
        cand.append(occ)
    maxlen = max((len(c) for c in cand), default=0)
    lens = np.array([len(c) for c in cand])
    ptr = np.zeros(B, int)
    n_drop_own = n_keep_own = n_drop_opp = n_keep_w = n_capkeep = 0

    for step in range(maxlen):
        cap = md0.astype(np.int32).copy()
        cell = np.full(B, -1, int)
        test_own = own.copy()
        test_opp = opp.copy()
        act = ptr < lens
        for b in np.where(act)[0]:
            c = cand[b][ptr[b]]
            cell[b] = c
            if own[b, c]:
                test_own[b, c] = False
                cap[b] = int(md0[b])
            else:  # opp
                test_opp[b, c] = False
                cap[b] = max(int(md0[b]) - 1, 0)
        tb = np.stack([test_own.reshape(B, N, N), test_opp.reshape(B, N, N)], 1)
        wd, hd = solve_vct_mega_bb(tb, max_nodes=max_nodes, max_depth=cap)
        for b in np.where(act)[0]:
            c = cell[b]
            clean_win = bool(wd[b] and not hd[b])
            capped = bool(hd[b])
            if own[b, c]:
                if capped:
                    n_capkeep += 1                 # KEEP (fail-safe)
                elif clean_win:
                    own[b, c] = False               # redundant -> DROP
                    n_drop_own += 1
                else:
                    n_keep_own += 1                 # load-bearing B -> KEEP
            else:  # opp
                if capped:
                    n_capkeep += 1                 # KEEP (fail-safe) -> stays in opp = W
                    n_keep_w += 1
                elif clean_win:
                    n_keep_w += 1                   # shorter win opened -> load-bearing W KEEP
                else:
                    opp[b, c] = False               # irrelevant -> DROP
                    n_drop_opp += 1
            ptr[b] += 1

    stats = dict(drop_own=n_drop_own, keep_own_B=n_keep_own, drop_opp=n_drop_opp,
                 keep_W=n_keep_w, capkeep=n_capkeep)
    return dict(own=own, opp=opp, supp=supp, carr=carr, wch=wch, move=move,
                seed_win=win, seed_hit=hit), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(CORPORA), default="molecule")
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--max-nodes", type=int, default=2000)
    ap.add_argument("--hi", type=int, default=30)
    ap.add_argument("--out", type=str, default="~/data/md_stencils")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    boards, rows, total = load_corpus(args.corpus, args.n)
    B = boards.shape[0]
    nown0 = boards[:, 0].reshape(B, -1).sum(1)
    nopp0 = boards[:, 1].reshape(B, -1).sum(1)
    print(f"[load] {args.corpus}: {B}/{total} boards  own-stones mean {nown0.mean():.1f} "
          f"opp-stones mean {nopp0.mean():.1f}")

    t = time.time()
    md0, capped = solve_md_min(boards, max_nodes=args.max_nodes, hi=args.hi)
    print(f"[md0] {time.time()-t:.1f}s  alive {int((md0>=1).sum())}/{B}  "
          f"capped {int(capped.sum())}")
    alive = md0 >= 1
    bidx = np.where(alive)[0]
    bd = boards[bidx]
    md0a = md0[bidx]

    t = time.time()
    res, stats = minimize(bd, md0a, args.max_nodes)
    dt_ab = time.time() - t
    print(f"[ablate] {dt_ab:.1f}s  {stats}")

    # --- per-board minimal stencils + measurements ---
    own, opp, supp, carr, wch, move = (res["own"], res["opp"], res["supp"],
                                       res["carr"], res["wch"], res["move"])
    A = len(bidx)
    rec = []
    for i in range(A):
        B_cells = [int(c) for c in np.flatnonzero(own[i])]
        W_cells = [int(c) for c in np.flatnonzero(opp[i])]
        s_cells = cells_from_words(supp[i])
        carr_cells = cells_from_words(carr[i])
        w_over = cells_from_words(wch[i])
        rec.append(dict(
            md0=int(md0a[i]), move=int(move[i]),
            B=B_cells, W=W_cells, support=s_cells,
            carriers=carr_cells, w_over=w_over,
            own0=int(nown0[bidx[i]]), opp0=int(nopp0[bidx[i]])))

    def arr(key, f):
        return np.array([f(r) for r in rec])

    # stone counts: baseline (today's support∪carriers, NO W) vs ablated (B + W)
    base_stones = arr("c", lambda r: len(r["carriers"]))
    abl_stones = arr("a", lambda r: len(r["B"]) + len(r["W"]))
    base_full = arr("bf", lambda r: len(r["carriers"]) + len(r["support"]))
    abl_full = arr("af", lambda r: len(r["B"]) + len(r["W"]) + len(r["support"]))
    nW = arr("w", lambda r: len(r["W"]))
    nB = arr("b", lambda r: len(r["B"]))
    orig_stones = arr("o", lambda r: r["own0"] + r["opp0"])
    has_W = nW > 0
    w_over_n = arr("wo", lambda r: len(r["w_over"]))

    # W-rate by md0
    md_vals = sorted(set(md0a.tolist()))
    wrate_rows = []
    for d in md_vals:
        m = md0a == d
        if m.sum() == 0:
            continue
        wrate_rows.append((d, int(m.sum()), float(has_W[m].mean()),
                           float(nW[m].mean()), float(abl_full[m].mean()),
                           float(base_full[m].mean())))

    # saturation: distinct canon(B ∪ W ∪ support) vs cumulative boards
    seen = set()
    sat = []
    for i, r in enumerate(rec):
        st = canon(r["B"] + r["W"] + r["support"])
        seen.add(st)
        if (i + 1) % max(1, A // 20) == 0 or i == A - 1:
            sat.append((i + 1, len(seen)))
    distinct_full = len(seen)
    # baseline distinct (support∪carriers) for comparison
    seen_b = set(canon(r["carriers"] + r["support"]) for r in rec)

    # --- report ---
    L = [f"# md-invariant minimizer — corpus={args.corpus} n={A} "
         f"max_nodes={args.max_nodes}\n",
         f"Minimal typed stencils via md0-invariant context ablation (issue #91 "
         f"md-extraction). md0 = shortest-mate FRAME distance from "
         f"`solve_md_min`.\n",
         "## Yield / timing",
         f"- boards (alive VCT): **{A}** (of {B} loaded; {int(capped.sum())} capped at md0)",
         f"- ablation wall: **{dt_ab:.1f}s**  ({stats})",
         f"- md0 histogram: " + ", ".join(
             f"{d}:{int((md0a==d).sum())}" for d in md_vals),
         "",
         "## Reduction (stones)",
         f"- original stones (atk+dfd): mean **{orig_stones.mean():.1f}** "
         f"(p50 {int(np.median(orig_stones))})",
         f"- baseline support∪carriers stones (own only, NO W): mean "
         f"**{base_stones.mean():.2f}**",
         f"- ablated B+W stones: mean **{abl_stones.mean():.2f}**  "
         f"(B mean {nB.mean():.2f}, W mean {nW.mean():.2f})",
         f"- ablated reduces orig stones by **{100*(1-abl_stones.mean()/max(orig_stones.mean(),1e-9)):.0f}%**",
         "",
         "## Load-bearing W (the phenomenon md-ablation exists to find)",
         f"- stencils with >=1 load-bearing W: **{int(has_W.sum())}/{A} "
         f"({100*has_W.mean():.1f}%)**",
         f"- over-inclusive `w` channel had >=1 stone on: "
         f"{int((w_over_n>0).sum())}/{A} ({100*(w_over_n>0).mean():.1f}%)  "
         f"(ablation distills {nW.sum()} load-bearing W from {w_over_n.sum()} `w`-channel stones)",
         "",
         "### W-rate / size by md0",
         "| md0 | n | %with-W | mean-W | ablated-full | base-full |",
         "|----|----|----|----|----|----|"]
    for d, n_, wr, mw, af, bf in wrate_rows:
        L.append(f"| {d} | {n_} | {100*wr:.0f}% | {mw:.2f} | {af:.2f} | {bf:.2f} |")
    L += ["",
          "## Vocabulary",
          f"- distinct ablated stencils canon(B∪W∪support): **{distinct_full}** "
          f"(of {A}; {100*distinct_full/A:.0f}% distinct)",
          f"- distinct baseline canon(carriers∪support): **{len(seen_b)}**",
          f"- saturation curve (boards, distinct): " +
          " ".join(f"({a},{b})" for a, b in sat),
          ""]
    report = out / f"report_{args.corpus}_n{A}_mn{args.max_nodes}.md"
    report.write_text("\n".join(L))

    dump = out / f"stencils_{args.corpus}_n{A}_mn{args.max_nodes}.jsonl.gz"
    with gzip.open(dump, "wt") as fh:
        for r in rec:
            mv = r["move"]
            anchor = r["B"] + r["W"] + r["support"]
            if not anchor:
                continue
            r0 = min(c // N for c in anchor)
            c0 = min(c % N for c in anchor)
            fh.write(json.dumps({
                "md0": r["md0"], "move_rel": [mv // N - r0, mv % N - c0],
                "B_rel": sorted([c // N - r0, c % N - c0] for c in r["B"]),
                "W_rel": sorted([c // N - r0, c % N - c0] for c in r["W"]),
                "support_rel": sorted([c // N - r0, c % N - c0] for c in r["support"]),
            }) + "\n")

    print("\n".join(L[:40]))
    print(f"\n[done] {time.time()-t0:.1f}s total  report -> {report}\n  dump -> {dump}")


if __name__ == "__main__":
    main()
