"""Scalable vocabulary / diversity analysis over a streaming-minimizer append-log.

The n=105 IoU analysis was O(n^2) — it does not survive n=1000+. This reads the
append-only log (`stream_<corpus>.jsonl`), keeps `status=ok` rows, and reports:

  * exact-set distinct  and  D4-folded distinct   — O(n) via hashing (ALL n)
  * mean stencil size, md0 histogram, load-bearing-W rate by md0, `w`-channel ratio
  * IoU>=t clustering on a CAPPED RANDOM SAMPLE (default 400) — the size-robust
    diversity check, kept tractable so the whole thing is O(n) + O(sample^2).

The second-pass lesson (2026-06-28): on over-inclusive stencils exact-distinct
over-counts and IoU under-counts, both size-dominated — this prints both so the
artifact is visible rather than hidden behind one number.

Run:
  python -m scripts.threat_shapes.analyze_vocab_stream --corpus enable [--sample 400]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_ok(logpath):
    out = []
    with open(logpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                out.append(r)
    return out


def cells_of(r):
    return frozenset(tuple(c) for c in r["B_rel"] + r["W_rel"] + r["support_rel"])


def canon(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def d4_key(cells):
    keys = []
    for flip in (False, True):
        cs = [(r, -c) for r, c in cells] if flip else list(cells)
        for _ in range(4):
            keys.append(tuple(sorted(canon(cs))))
            cs = [(c, -r) for r, c in cs]
    return min(keys)


def best_iou(a, b):
    best, seen = 0.0, set()
    for (ar, ac) in a:
        for (br, bc) in b:
            s = (br - ar, bc - ac)
            if s in seen:
                continue
            seen.add(s)
            sh = frozenset((r + s[0], c + s[1]) for r, c in a)
            inter = len(sh & b)
            if inter:
                iou = inter / len(sh | b)
                if iou > best:
                    best = iou
                    if best == 1.0:
                        return 1.0
    return best


def iou_clusters(stencils, thresh):
    n = len(stencils)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) != find(j) and best_iou(stencils[i], stencils[j]) >= thresh:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="enable")
    ap.add_argument("--log", default=None)
    ap.add_argument("--dir", default="~/data/md_stencils")
    ap.add_argument("--sample", type=int, default=400, help="IoU sample cap (0=skip)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logpath = Path(args.log).expanduser() if args.log else \
        Path(args.dir).expanduser() / f"stream_{args.corpus}.jsonl"
    rows = load_ok(logpath)
    n = len(rows)
    if n == 0:
        print(f"no status=ok rows in {logpath}")
        return
    cells = [cells_of(r) for r in rows]
    sizes = [len(c) for c in cells]
    md0 = [r["md0"] for r in rows]
    nW = [len(r["W_rel"]) for r in rows]
    nB = [len(r["B_rel"]) for r in rows]
    w_over = [r.get("w_over_n", 0) for r in rows]

    exact = len({canon(c) for c in cells})
    d4 = len({d4_key(c) for c in cells})

    print(f"# vocab analysis  {logpath.name}  n_ok={n}")
    print(f"mean_cells={sum(sizes)/n:.1f}  mean_B={sum(nB)/n:.2f}  mean_W={sum(nW)/n:.2f}")
    print(f"exact-distinct = {exact:5d} ({100*exact/n:.0f}%)   "
          f"D4-distinct = {d4:5d} ({100*d4/n:.0f}%)")

    # md0 histogram + W-rate
    hist = {}
    for d, w in zip(md0, nW):
        hist.setdefault(d, [0, 0])
        hist[d][0] += 1
        hist[d][1] += (w > 0)
    print("md0 | n | %with-W")
    for d in sorted(hist):
        cnt, wc = hist[d]
        print(f"  {d:2d} | {cnt:5d} | {100*wc/cnt:3.0f}%")
    tot_W = sum(nW)
    tot_wover = sum(w_over)
    if tot_wover:
        print(f"w-channel ratio: {tot_wover} over-inclusive -> {tot_W} load-bearing "
              f"(w is {tot_wover/max(tot_W,1):.1f}x the single-removal set)")

    if args.sample:
        import random
        rng = random.Random(args.seed)
        idx = list(range(n))
        rng.shuffle(idx)
        samp = [cells[i] for i in idx[:min(args.sample, n)]]
        m = len(samp)
        se = len({canon(c) for c in samp})
        for t in (0.5, 0.3):
            c = iou_clusters(samp, t)
            print(f"IoU>={t} clusters on sample({m}) = {c:4d} ({100*c/m:.0f}%)   "
                  f"[exact on same sample = {se} ({100*se/m:.0f}%)]")


if __name__ == "__main__":
    main()
