"""Mine *relative* (translation-free) VCT stencils from the GPU solver.

The ask (Jason, 2026-06-27): run known-win / known-loss positions through
``solve_vct_mega_bb`` and mine the proof shape — NOT specific positions, but a
*non-minimal stencil* free to translate anywhere it fits. Then analyse the
vocabulary: which shapes overlap, which are wildly different.

The solver exposes the proof as two COMPLEMENTARY masks (issue #88):
  * ``support``  — the required-OPENINGS: empty cells the forcing line plays into
                   (the ``.``/``p`` channel; ⊆ empty-at-root).
  * ``carriers`` — the load-bearing OWN stones the proof's five-lines run through
                   (the ``B`` channel; ⊆ occupied-own-at-root). [NEW: #88]
A complete typed stencil is the UNION ``support ∪ carriers`` — e.g. the literal
``.BBBB.`` comes out as carriers={the four B} + support={the two ends}.

We analyse two views per vocabulary:
  * OPENINGS  = canon(support)            — the original (incomplete) shape.
  * FULL      = canon(support ∪ carriers) — the complete typed shape.

Vocabularies (one bulk pool, bulk-synchronous per the call-cost law):
  * WIN  — side-to-move has a forced VCT (offense).
  * LOSS — side-to-move has NO VCT but the opponent has a VCT-in-hand (planes
           flipped): the shape that beats you (danger proxy, not a proven loss).
  * VCF vs MOL — within WIN, split by the VCF twin: also-clean-VCF = trivial
           four-chain; VCT-but-not-VCF = combinational "molecule".

Translation-only canonicalisation (D4 deliberately NOT folded). Run from root:
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python \
      scripts/threat_shapes/mine_support_shapes.py --n 16384 --max-nodes 500
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
from pathlib import Path

import numpy as np

from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb, cells_from_words
from scripts.vct_metal.mega_vcf_bb import solve_vcf_mega_bb
from scripts.vct_metal.positions import load_position_stack

N = 15


# --------------------------------------------------------------------------- #
# Stencil = translation-free relative cell set (bbox-min anchored).
# --------------------------------------------------------------------------- #
def canon(cells) -> tuple:
    rc = [(c // N, c % N) for c in cells]
    r0 = min(r for r, _ in rc)
    c0 = min(c for _, c in rc)
    return tuple(sorted((r - r0, c - c0) for r, c in rc))


def bbox_dims(stencil) -> tuple:
    rs = [r for r, _ in stencil]
    cs = [c for _, c in stencil]
    return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)


def render_typed(support_cells, carrier_cells, move) -> str:
    """ASCII grid over the UNION bbox: 'B' carrier stone, '*' first move,
    '#' other support (reserved opening), '.' empty/don't-care."""
    cells = list(support_cells) + list(carrier_cells)
    r0 = min(c // N for c in cells)
    c0 = min(c % N for c in cells)
    h = max(c // N for c in cells) - r0 + 1
    w = max(c % N for c in cells) - c0 + 1
    grid = [["." for _ in range(w)] for _ in range(h)]
    for c in support_cells:
        grid[c // N - r0][c % N - c0] = "#"
    for c in carrier_cells:
        grid[c // N - r0][c % N - c0] = "B"
    if move in set(support_cells):
        grid[move // N - r0][move % N - c0] = "*"
    return "\n".join(" ".join(row) for row in grid)


def best_iou(a: frozenset, b: frozenset) -> float:
    """Max Jaccard over all integer translations aligning a cell of a onto b."""
    if not a or not b:
        return 0.0
    best = 0.0
    seen = set()
    for (ar, ac) in a:
        for (br, bc) in b:
            s = (br - ar, bc - ac)
            if s in seen:
                continue
            seen.add(s)
            shifted = frozenset((r + s[0], c + s[1]) for r, c in a)
            inter = len(shifted & b)
            if not inter:
                continue
            iou = inter / len(shifted | b)
            if iou > best:
                best = iou
                if best == 1.0:
                    return 1.0
    return best


# --------------------------------------------------------------------------- #
# Mining: one pool -> bulk solves (now WITH carriers) -> typed stencils.
# A record = (support_cells, carrier_cells, move) in ABSOLUTE flat indices.
# --------------------------------------------------------------------------- #
def mine(n, seed, max_nodes, min_ply, max_ply):
    print(f"[mine] loading {n} positions (seed={seed}, ply {min_ply}-{max_ply})")
    pool = load_position_stack(n, seed=seed, min_ply=min_ply, max_ply=max_ply)

    print(f"[mine] attacker VCT (support+move+carriers) @max_nodes={max_nodes} ...")
    win, hit, move, supp, carr = solve_vct_mega_bb(
        pool, max_nodes=max_nodes, return_move=True,
        return_support=True, return_carriers=True)
    print("[mine] VCF twin (classify trivial vs molecule) ...")
    vcf_win, vcf_hit = solve_vcf_mega_bb(pool, max_nodes=max_nodes)
    print("[mine] flipped VCT (opponent support+move+carriers) ...")
    flip = pool[:, [1, 0]].copy()
    winf, hitf, movef, suppf, carrf = solve_vct_mega_bb(
        flip, max_nodes=max_nodes, return_move=True,
        return_support=True, return_carriers=True)

    clean = ~hit
    own_win = win & clean
    is_vcf = own_win & vcf_win & ~vcf_hit
    is_mol = own_win & ~(vcf_win & ~vcf_hit)
    loss = (~win & clean) & (winf & ~hitf)

    def recs(mask, supp_arr, carr_arr, move_arr):
        out = []
        for b in np.flatnonzero(mask):
            s = cells_from_words(supp_arr[b])
            if not s:
                continue
            out.append((s, cells_from_words(carr_arr[b]), int(move_arr[b])))
        return out

    vocab = {
        "win": recs(own_win, supp, carr, move),
        "vcf": recs(is_vcf, supp, carr, move),
        "mol": recs(is_mol, supp, carr, move),
        "loss": recs(loss, suppf, carrf, movef),
    }
    counts = {
        "pool": n,
        "attacker_clean_win": int(own_win.sum()),
        "attacker_hit_cap": int(hit.sum()),
        "vcf_trivial": int(is_vcf.sum()),
        "molecule": int(is_mol.sum()),
        "loss_proxy": int(loss.sum()),
        "win_with_carriers": int(sum(1 for r in vocab["win"] if r[1])),
        "mol_mean_carriers": round(
            float(np.mean([len(r[1]) for r in vocab["mol"]])) if vocab["mol"] else 0.0, 2),
    }
    return vocab, counts


# --------------------------------------------------------------------------- #
# Analysis helpers.
# --------------------------------------------------------------------------- #
def opening_stencil(rec):
    return canon(rec[0])


def full_stencil(rec):
    return canon(rec[0] + rec[1])


def freq(recs, key):
    return collections.Counter(key(r) for r in recs)


def diversity(f: collections.Counter) -> dict:
    total = sum(f.values())
    distinct = len(f)
    if total == 0:
        return dict(total=0, distinct=0, singleton_frac=0.0, top1=0.0,
                    top5=0.0, norm_entropy=0.0)
    counts = np.array(sorted(f.values(), reverse=True), dtype=float)
    p = counts / total
    ent = -(p * np.log(p)).sum()
    return dict(
        total=total, distinct=distinct,
        singleton_frac=float((counts == 1).sum()) / distinct,
        top1=counts[0] / total, top5=counts[:5].sum() / total,
        norm_entropy=ent / math.log(distinct) if distinct > 1 else 0.0,
    )


def cluster_topk(f: collections.Counter, k, thresh):
    top = [st for st, _ in f.most_common(k)]
    sets = [frozenset(st) for st in top]
    m = len(top)
    M = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            M[i, j] = M[j, i] = best_iou(sets[i], sets[j])
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(m):
        for j in range(i + 1, m):
            if M[i, j] >= thresh:
                parent[find(i)] = find(j)
    clusters = collections.defaultdict(list)
    for i in range(m):
        clusters[find(i)].append(i)
    return top, M, list(clusters.values())


def vocab_jaccard(a: collections.Counter, b: collections.Counter) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def rep_record(recs, key):
    """One representative record per stencil (for typed rendering)."""
    rep = {}
    for r in recs:
        st = key(r)
        if st not in rep:
            rep[st] = r
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=500)
    ap.add_argument("--min-ply", type=int, default=6)
    ap.add_argument("--max-ply", type=int, default=60)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--cluster-thresh", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="~/data/support_shapes")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    vocab, counts = mine(args.n, args.seed, args.max_nodes, args.min_ply, args.max_ply)

    f_open = {k: freq(v, opening_stencil) for k, v in vocab.items()}
    f_full = {k: freq(v, full_stencil) for k, v in vocab.items()}
    rep_full = {k: rep_record(v, full_stencil) for k, v in vocab.items()}

    # dump full typed stencils
    dump = out / f"stencils_full_n{args.n}_s{args.seed}_mn{args.max_nodes}.jsonl.gz"
    with gzip.open(dump, "wt") as fh:
        for name, recs in vocab.items():
            seen = collections.Counter(full_stencil(r) for r in recs)
            for st, c in seen.items():
                r = rep_full[name][st]
                anchor_r = min(x // N for x in r[0] + r[1])
                anchor_c = min(x % N for x in r[0] + r[1])
                fh.write(json.dumps({
                    "vocab": name, "count": c, "bbox": list(bbox_dims(st)),
                    "size": len(st), "n_carriers": len(r[1]),
                    "carriers_rel": sorted([x // N - anchor_r, x % N - anchor_c] for x in r[1]),
                    "support_rel": sorted([x // N - anchor_r, x % N - anchor_c] for x in r[0]),
                }) + "\n")

    L = [f"# VCT stencil mining (with carriers) — n={args.n} seed={args.seed} "
         f"max_nodes={args.max_nodes}\n",
         "Translation-free stencils from `solve_vct_mega_bb` with the NEW "
         "`return_carriers` (#88). OPENINGS = support (`./p`); FULL = "
         "support ∪ carriers (adds the `B` substrate). D4 not folded.\n",
         "## Pool / yield\n"]
    for k, v in counts.items():
        L.append(f"- **{k}**: {v}")

    L.append("\n## Diversity: OPENINGS view vs FULL (support ∪ carriers) view\n")
    L.append("| vocab | view | supports | distinct | singleton% | top5% | norm-entropy |")
    L.append("|---|---|---|---|---|---|---|")
    for name in ("win", "vcf", "mol", "loss"):
        for view, fd in (("openings", f_open), ("full", f_full)):
            d = diversity(fd[name])
            L.append(f"| {name} | {view} | {d['total']} | {d['distinct']} | "
                     f"{d['singleton_frac']*100:.0f}% | {d['top5']*100:.0f}% | "
                     f"{d['norm_entropy']:.3f} |")
    L.append("\n*Adding the B substrate can only split shape-classes apart → "
             "distinct↑, entropy↑. The gap openings→full = how much the carrier "
             "stones distinguish shapes that shared an opening footprint.*\n")

    L.append("## Cross-vocabulary overlap — FULL stencils (exact-set Jaccard)\n")
    L.append(f"- win ∩ loss: {vocab_jaccard(f_full['win'], f_full['loss']):.3f}")
    L.append(f"- vcf ∩ mol : {vocab_jaccard(f_full['vcf'], f_full['mol']):.3f}\n")

    # typed exemplars: most frequent FULL molecules / win shapes / loss shapes
    for name in ("mol", "win", "loss"):
        recs = vocab[name]
        if not recs:
            continue
        L.append(f"## {name.upper()} — most frequent FULL typed stencils "
                 "(`B`=carrier stone, `*`=move, `#`=reserved opening)\n")
        for st, c in f_full[name].most_common(5):
            r = rep_full[name][st]
            h, w = bbox_dims(st)
            L.append(f"**count={c}, size={len(st)} ({len(r[1])} B + "
                     f"{len(r[0])} opening), bbox={h}×{w}**\n```")
            L.append(render_typed(r[0], r[1], r[2]))
            L.append("```\n")

    report = out / f"report_carriers_n{args.n}_s{args.seed}_mn{args.max_nodes}.md"
    report.write_text("\n".join(L))
    print(f"\n[done] report -> {report}")
    print(f"[done] dump   -> {dump}")

    try:
        _figures(out, args, vocab, f_full, rep_full)
    except Exception as e:  # noqa: BLE001
        print(f"[figures] skipped: {e}")

    print("\n=== counts ===")
    for k, v in counts.items():
        print(f"  {k:24s} {v}")
    print("\n=== diversity openings -> full ===")
    for name in ("win", "vcf", "mol", "loss"):
        do, df = diversity(f_open[name]), diversity(f_full[name])
        print(f"  {name:5s} distinct {do['distinct']:4d}->{df['distinct']:4d}  "
              f"entropy {do['norm_entropy']:.3f}->{df['norm_entropy']:.3f}")


def _figures(out, args, vocab, f_full, rep_full):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # typed exemplars: B stones (black) vs openings (grey) vs move (red star)
    for name in ("mol", "win"):
        recs = vocab[name]
        if not recs:
            continue
        top = [st for st, _ in f_full[name].most_common(12)]
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for ax, st in zip(axes.flat, top):
            r = rep_full[name][st]
            cells = r[0] + r[1]
            r0 = min(c // N for c in cells); c0 = min(c % N for c in cells)
            h = max(c // N for c in cells) - r0 + 1
            w = max(c % N for c in cells) - c0 + 1
            img = np.zeros((h, w))
            for c in r[0]:
                img[c // N - r0, c % N - c0] = 0.5      # opening
            for c in r[1]:
                img[c // N - r0, c % N - c0] = 1.0      # carrier B
            ax.imshow(img, cmap="Greys", vmin=0, vmax=1)
            if r[2] in set(r[0]):
                ax.scatter([r[2] % N - c0], [r[2] // N - r0], c="red", s=90, marker="*")
            ax.set_title(f"×{f_full[name][st]}  {len(r[1])}B+{len(r[0])}o", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes.flat[len(top):]:
            ax.axis("off")
        fig.suptitle(f"{name.upper()} — 12 most frequent FULL typed stencils "
                     "(black=B carrier, grey=opening, ★=move)")
        fig.tight_layout()
        fig.savefig(out / f"typed_{name}_n{args.n}.png", dpi=110)
        plt.close(fig)
    print(f"[figures] wrote typed PNGs to {out}")


if __name__ == "__main__":
    main()
