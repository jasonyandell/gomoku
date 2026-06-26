"""Driver: replay the corpus -> threat/junction shapes + dependency-motif census.

Stages (all artifacts -> --out):
  1. Replay every corpus position (solver chosen by tag) -> per-position threat
     lists + junction squares. Cache to replay.pkl.
  2. Crop D4-canonical patches: THREAT patches (centred on each threat's gain,
     split by kind) and JUNCTION patches (centred on shared squares).
  3. D4-aware k-means class averaging (cryo-EM) on each set; render
     information-weighted "threat-logo" grids. Two-corpus-half reproducibility.
  4. Colored 3-node dependency-motif census vs a degree-matched, color-preserving
     null, with two-half significance.

Run:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.run \
      --corpus <dir>/corpus.npz --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import Counter

import numpy as np

from scripts.threat_shapes.threats import replay_vct
from scripts.threat_shapes import shapes as S
from scripts.threat_shapes import motifs as M


def stage_replay(corpus_path, out_dir, max_positions=None):
    d = np.load(corpus_path)
    boards, tags = d["boards"], d["tags"]
    if max_positions:
        boards, tags = boards[:max_positions], tags[:max_positions]
    results = []
    t0 = time.time()
    for i in range(len(boards)):
        solver = "vcf" if tags[i] == "vcf" else "vct"
        r = replay_vct(boards[i], solver=solver, max_plies=6,
                       vcf_nodes=8000, vcf_depth=10)
        results.append((boards[i], tags[i], r))
        if (i + 1) % 200 == 0:
            print(f"  replay {i+1}/{len(boards)}  {(i+1)/(time.time()-t0):.1f}/s",
                  flush=True)
    with open(os.path.join(out_dir, "replay.pkl"), "wb") as f:
        pickle.dump(results, f)
    print(f"replayed {len(results)} in {time.time()-t0:.1f}s", flush=True)
    return results


def collect_patches(results, radius=4):
    """Return dict: 'threat:<kind>' -> (P,C,H,W), 'junction' -> (P,C,H,W),
    plus the per-position threat graphs (for the motif census)."""
    threat_patches = {}
    junction = []
    graphs = []
    halves = []   # 0/1 corpus-half tag per position (by index parity)
    for idx, (board, tag, r) in enumerate(results):
        graphs.append((M.build_graph(r.threats), idx % 2))
        for t in r.threats:
            if t.kind == "five":
                continue
            key = f"threat:{t.kind}"
            threat_patches.setdefault(key, []).append(
                (S.crop_patch(board, t.gain, radius), idx % 2))
        for j in r.junctions:
            junction.append((S.crop_patch(board, j, radius), idx % 2))
    return threat_patches, junction, graphs


def _arr_half(items):
    patches = np.stack([p for p, _ in items])
    half = np.array([h for _, h in items])
    return patches, half


def render_logo_grid(centroids, counts, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(centroids)
    cols = min(8, k)
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows),
                             squeeze=False)
    order = np.argsort(-counts)
    for ax in axes.ravel():
        ax.axis("off")
    for slot, c in enumerate(order):
        ax = axes[slot // cols][slot % cols]
        cen = centroids[c]                      # (3,H,W): atk,def,empty
        info = S.cell_information(cen)
        opacity = np.clip(info / np.log2(3), 0, 1)
        H, W = info.shape
        # RGB: attacker -> dark, defender -> red, empty -> white
        atk, dfd = cen[0], cen[1]
        img = np.ones((H, W, 3))
        # mix toward black for attacker, toward red for defender, weighted by who dominates
        img[..., 0] = 1 - opacity * (atk * 1.0)             # atk lowers R,G,B (black)
        img[..., 1] = 1 - opacity * (atk * 1.0 + dfd * 1.0)
        img[..., 2] = 1 - opacity * (atk * 1.0 + dfd * 1.0)
        # defender adds red tint: raise R back up where defender dominates
        img[..., 0] = np.clip(img[..., 0] + opacity * dfd * 0.8, 0, 1)
        ax.imshow(img, interpolation="nearest")
        # mark center
        ax.plot(W // 2, H // 2, "+", color="#2c7", ms=8, mew=2)
        ax.set_title(f"#{c}  n={counts[c]}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cluster_and_report(name, patches, half, out_dir, k=8, min_n=40, cap=2500):
    n_total = len(patches)
    if n_total < min_n:
        return {"name": name, "n": int(n_total), "skipped": "too few"}
    if n_total > cap:                       # subsample for tractable clustering
        sub = np.random.default_rng(0).choice(n_total, cap, replace=False)
        patches, half = patches[sub], half[sub]
    k = min(k, max(2, len(patches) // 20))
    cen, labels, aligned, inertia, counts = S.d4_kmeans(patches, k, iters=20,
                                                        restarts=3)
    # two-half reproducibility: cluster each half, match
    pa, pb = patches[half == 0], patches[half == 1]
    repro = None
    if len(pa) >= min_n and len(pb) >= min_n:
        ka = min(k, max(2, len(pa) // 20))
        cen_a = S.d4_kmeans(pa, ka, iters=15, restarts=2)[0]
        cen_b = S.d4_kmeans(pb, ka, iters=15, restarts=2)[0]
        md, dists = S.match_centroids(cen_a, cen_b)
        # null: distance between A-centroids and SHUFFLED random patches of same size
        rng = np.random.default_rng(0)
        rand = patches[rng.choice(len(patches), size=len(pb), replace=False)]
        md_rand, _ = S.match_centroids(cen_a, S.d4_kmeans(rand, ka, iters=8,
                                                          restarts=1)[0])
        repro = {"mean_match_dist": round(md, 3),
                 "null_shuffle_match_dist": round(md_rand, 3),
                 "reproducible": bool(md < md_rand)}
    path = os.path.join(out_dir, f"logo_{name}.png")
    render_logo_grid(cen, counts, path, f"{name}  (n={len(patches)}, k={k})")
    return {"name": name, "n": int(n_total), "n_clustered": int(len(patches)),
            "k": int(k), "cluster_sizes": counts.tolist(), "figure": path,
            "reproducibility": repro}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--reuse-replay", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rp = os.path.join(args.out, "replay.pkl")
    if args.reuse_replay and os.path.exists(rp):
        with open(rp, "rb") as f:
            results = pickle.load(f)
        print(f"reused {len(results)} replays", flush=True)
    else:
        results = stage_replay(args.corpus, args.out, args.max_positions)

    threat_patches, junction, graphs = collect_patches(results, args.radius)
    report = {"corpus": args.corpus, "n_positions": len(results),
              "threat_kind_counts": {k: len(v) for k, v in threat_patches.items()},
              "n_junction_patches": len(junction),
              "clusters": [], "motifs": None}

    # ---- shape clustering ----
    for key, items in sorted(threat_patches.items(), key=lambda kv: -len(kv[1])):
        patches, half = _arr_half(items)
        report["clusters"].append(cluster_and_report(
            key.replace(":", "_"), patches, half, args.out, k=8))
    if len(junction) >= 40:
        patches, half = _arr_half(junction)
        report["clusters"].append(cluster_and_report(
            "junction", patches, half, args.out, k=8))

    # ---- dependency-motif census ----
    g_all = [g for g, _ in graphs if g["n"] >= 3]
    g_h0 = [g for g, h in graphs if h == 0 and g["n"] >= 3]
    g_h1 = [g for g, h in graphs if h == 1 and g["n"] >= 3]
    print(f"motif census over {len(g_all)} graphs with >=3 threats", flush=True)
    if len(g_all) >= 20:
        stats_all, obs_all = M.motif_significance(g_all, n_null=200, seed=1)
        stats0, _ = M.motif_significance(g_h0, n_null=200, seed=2) if len(g_h0) >= 10 else ({}, None)
        stats1, _ = M.motif_significance(g_h1, n_null=200, seed=3) if len(g_h1) >= 10 else ({}, None)
        rows = []
        for motif, st in sorted(stats_all.items(), key=lambda kv: -kv[1]["z"]):
            if st["obs"] < 5:
                continue
            z0 = stats0.get(motif, {}).get("z", float("nan"))
            z1 = stats1.get(motif, {}).get("z", float("nan"))
            rows.append({"motif": motif, **{k: round(v, 2) if isinstance(v, float) else v
                                            for k, v in st.items()},
                         "z_half0": round(z0, 2), "z_half1": round(z1, 2),
                         "reproducible_sig": bool(abs(z0) > 2 and abs(z1) > 2
                                                  and np.sign(z0) == np.sign(z1))})
        report["motifs"] = {"n_graphs": len(g_all), "rows": rows[:40]}

    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "motifs"},
                     indent=2, default=str), flush=True)
    if report["motifs"]:
        print("TOP MOTIFS (z-sorted):", flush=True)
        for r in report["motifs"]["rows"][:15]:
            print(f"  z={r['z']:+.1f} (h0={r['z_half0']:+.1f} h1={r['z_half1']:+.1f}) "
                  f"obs={r['obs']} repro={r['reproducible_sig']}  {r['motif']}",
                  flush=True)


if __name__ == "__main__":
    main()
