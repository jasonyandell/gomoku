"""Finalize pass over the cached replay: fork collinearity + a motif figure.

Answers the load-bearing 'junction' question directly from the replayed threats
(no re-solving): is a double-four fork COLLINEAR (an open/straight four -- one
line, 2 completions on the same axis) or NON-COLLINEAR (a genuine 2-D fork --
two fours on different axes sharing the gain stone)? The non-collinear forks are
where 2-D combination geometry lives even though each constituent four is a line.

Also renders the reproducible dependency-motif census as a labelled bar chart.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

from scripts.threat_shapes import shapes as S
from scripts.threat_shapes import motifs as M
from scripts.threat_shapes.threats import rc

N = 15


def fork_geometry(results):
    """Tally double-four / three-fork threats by collinear vs non-collinear."""
    coll, noncoll = 0, 0
    noncoll_patches, coll_patches = [], []
    for board, tag, r in results:
        for t in r.threats:
            if t.kind not in ("double_four", "three_fork"):
                continue
            # cost squares are the completion squares; are they all on ONE axis
            # through the gain, or spread across >= 2 axes (a 2-D fork)?
            axes = set()
            gr, gc = rc(t.gain)
            for c in t.cost:
                cr, cc = rc(c)
                dr, dc = cr - gr, cc - gc
                if dr == 0 and dc == 0:
                    continue
                g = np.gcd(abs(dr), abs(dc)) or 1
                axes.add((dr // g, dc // g) if (dr, dc) >= (0, 0) else (-dr // g, -dc // g))
            patch = S.crop_patch(board, t.gain, 4)
            if len(axes) >= 2:
                noncoll += 1
                noncoll_patches.append(patch)
            else:
                coll += 1
                coll_patches.append(patch)
    return coll, noncoll, np.array(coll_patches), np.array(noncoll_patches)


def motif_bar(stats_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in stats_rows if r["obs"] >= 20][:10]
    labels = [r["motif"].replace("|", "\n") for r in rows]
    z = [r["z"] for r in rows]
    repro = [r["reproducible_sig"] for r in rows]
    colors = ["#2a7" if rp else "#aaa" for rp in repro]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(z)), z, color=colors)
    ax.set_xticks(range(len(z)))
    ax.set_xticklabels(labels, fontsize=6, rotation=0)
    ax.axhline(2, ls="--", c="r", lw=0.8)
    ax.axhline(-2, ls="--", c="r", lw=0.8)
    ax.set_ylabel("z vs degree/color-matched null")
    ax.set_title("Dependency-motif census (green = significant & reproducible in both halves)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    out = sys.argv[1]
    with open(os.path.join(out, "replay.pkl"), "rb") as f:
        results = pickle.load(f)

    coll, noncoll, cpatch, ncpatch = fork_geometry(results)
    print(f"FORK GEOMETRY: collinear(open/straight four)={coll}  "
          f"non-collinear(2-D fork)={noncoll}  "
          f"non-collinear frac={noncoll/max(1,coll+noncoll):.3f}")

    summary = {"fork_collinear": coll, "fork_noncollinear": noncoll,
               "noncollinear_fraction": round(noncoll / max(1, coll + noncoll), 3)}

    # cluster the NON-COLLINEAR forks (the genuine 2-D combination geometry)
    if len(ncpatch) >= 40:
        cen, lab, al, inert, counts = S.d4_kmeans(ncpatch, 6, iters=20, restarts=3)
        # two-half repro
        h = np.arange(len(ncpatch)) % 2
        pa, pb = ncpatch[h == 0], ncpatch[h == 1]
        rep = None
        if len(pa) >= 30 and len(pb) >= 30:
            ca = S.d4_kmeans(pa, 6, iters=12, restarts=2)[0]
            cb = S.d4_kmeans(pb, 6, iters=12, restarts=2)[0]
            md, _ = S.match_centroids(ca, cb)
            rng = np.random.default_rng(0)
            rnd = ncpatch[rng.choice(len(ncpatch), len(pb), replace=False)]
            mdr, _ = S.match_centroids(ca, S.d4_kmeans(rnd, 6, iters=8, restarts=1)[0])
            rep = {"match": round(md, 3), "null": round(mdr, 3),
                   "reproducible": bool(md < mdr)}
        from scripts.threat_shapes.run import render_logo_grid
        render_logo_grid(cen, counts, os.path.join(out, "logo_noncollinear_fork.png"),
                         f"non-collinear forks (n={len(ncpatch)})")
        summary["noncollinear_fork_clusters"] = {"k": 6, "sizes": counts.tolist(),
                                                  "reproducibility": rep,
                                                  "figure": os.path.join(out, "logo_noncollinear_fork.png")}

    # motif figure from report.json
    rep_path = os.path.join(out, "report.json")
    if os.path.exists(rep_path):
        with open(rep_path) as f:
            report = json.load(f)
        if report.get("motifs"):
            motif_bar(report["motifs"]["rows"], os.path.join(out, "motif_census.png"))
            summary["motif_figure"] = os.path.join(out, "motif_census.png")

    with open(os.path.join(out, "finalize_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
