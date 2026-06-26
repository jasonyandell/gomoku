"""Bio-discovery on the enabling-shape corpus + line-bound VCT control.

THE SCIENTIFIC QUESTION: do the ENABLING shapes (the board ONE move before a
realized forced kill begins) carry NON-LINE structure that the line-bound VCT
shapes do not? A VCF/VCT line is collinear by construction; the SETUP that
launches it need not be.

Three probes (all reuse the existing threat-shape toolkit):

  (a) CONNECTED-CORRELATION DCA BOND MAP with a PROPER density/marginal-preserving
      null. For attacker-stone offset (dr,dc) we measure the connected correlation
      C(dr,dc) = <x_i x_{i+off}> - <x_i><x_{i+off}> averaged over cells, and z it
      against an INDEPENDENT-BERNOULLI null drawn from the exact per-cell marginals
      (so the null preserves stone density everywhere and only destroys pairwise
      coupling). NON-COLLINEAR offsets (knight-move etc.) with z above the null are
      genuine 2-D structure -- the disentangled test. Harvest vs control, and per
      run-length bucket (do LONGER chains give richer non-line setups?).

  (b) D4-AWARE CLASS-AVERAGING (cryo-EM) on catalyst-centred patches -> the shape
      vocabulary, rendered as information-weighted heatmaps. Harvest vs control.

  (c) (cross-check) the original ``residual.mi_bond_map`` count-preserving null.

Run:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.harvest_science \
      --harvest <dir>/corpus.npz --control <mine>/corpus.npz --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from scripts.threat_shapes import shapes as S
from scripts.threat_shapes.run import render_logo_grid
from scripts.threat_shapes.residual import mi_bond_map


def _offsets(max_off: int):
    return [(dr, dc) for dr in range(-max_off, max_off + 1)
            for dc in range(0, max_off + 1)
            if (dr, dc) != (0, 0) and not (dr < 0 and dc == 0)]


def _is_collinear(dr: int, dc: int) -> bool:
    return dr == 0 or dc == 0 or abs(dr) == abs(dc)


def connected_corr(boards, *, max_off=4, n_perm=150, seed=0, max_b=2500):
    """Connected-correlation bond map vs an independent-Bernoulli (marginal-
    preserving) null. Returns offset -> {obs, z, collinear}."""
    rng = np.random.default_rng(seed)
    atk = boards[:, 0].astype(np.float32)
    if len(atk) > max_b:
        atk = atk[rng.choice(len(atk), max_b, replace=False)]
    B = len(atk)
    marg = atk.mean(0)                      # (N,N) per-cell P(stone)
    offs = _offsets(max_off)

    def conn(a):
        out = {}
        for (dr, dc) in offs:
            r0, r1 = max(0, -dr), min(N, N - dr)
            c0, c1 = max(0, -dc), min(N, N - dc)
            ai = a[:, r0:r1, c0:c1]
            aj = a[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            cooc = (ai * aj).mean(0)
            mi, mj = marg[r0:r1, c0:c1], marg[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            out[(dr, dc)] = float((cooc - mi * mj).mean())
        return out

    obs = conn(atk)
    null = {o: [] for o in offs}
    for _ in range(n_perm):
        a2 = (rng.random((B, N, N)) < marg).astype(np.float32)
        c = conn(a2)
        for o in offs:
            null[o].append(c[o])
    res = {}
    for o in offs:
        arr = np.asarray(null[o])
        mu, sd = float(arr.mean()), float(arr.std())
        z = (obs[o] - mu) / sd if sd > 1e-12 else 0.0
        res[o] = {"obs": obs[o], "z": float(z),
                  "collinear": _is_collinear(*o)}
    return res


def top_noncollinear(res, k=8):
    items = [(o, d) for o, d in res.items() if not d["collinear"]]
    items.sort(key=lambda kv: -kv[1]["z"])
    return [{"offset": list(o), "z": round(d["z"], 2), "obs": round(d["obs"], 5)}
            for o, d in items[:k]]


def top_collinear(res, k=5):
    items = [(o, d) for o, d in res.items() if d["collinear"]]
    items.sort(key=lambda kv: -kv[1]["z"])
    return [{"offset": list(o), "z": round(d["z"], 2)} for o, d in items[:k]]


def z_heatmap(res, max_off, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.full((2 * max_off + 1, 2 * max_off + 1), np.nan)
    coll = np.zeros_like(grid, bool)
    for (dr, dc), d in res.items():
        for (r, c) in ((dr, dc), (-dr, -dc)):  # symmetric
            grid[r + max_off, c + max_off] = d["z"]
            coll[r + max_off, c + max_off] = d["collinear"]
    fig, ax = plt.subplots(figsize=(5, 4.4))
    vmax = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[-max_off - .5, max_off + .5, max_off + .5, -max_off - .5])
    for r in range(2 * max_off + 1):
        for c in range(2 * max_off + 1):
            if not np.isnan(grid[r, c]):
                mark = "o" if coll[r, c] else "x"
                ax.plot(c - max_off, r - max_off, mark, ms=4,
                        color="k" if abs(grid[r, c]) < vmax * .5 else "w",
                        alpha=.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("dc"); ax.set_ylabel("dr")
    fig.colorbar(im, ax=ax, label="connected-corr z")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def cluster_logos(boards, moves, out_path, title, k=8, radius=5, cap=2500):
    if len(boards) < 40:
        return None
    rng = np.random.default_rng(0)
    idx = (rng.choice(len(boards), cap, replace=False)
           if len(boards) > cap else np.arange(len(boards)))
    patches = np.stack([S.crop_patch(boards[i], int(moves[i]), radius)
                        for i in idx])
    cen, labels, _aligned, _inertia, counts = S.d4_kmeans(
        patches, k, iters=20, restarts=3)
    render_logo_grid(cen, counts, out_path, title)
    return {"figure": out_path, "n": int(len(patches)),
            "cluster_sizes": counts.tolist()}


def run_length_fig(mds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vals, cnts = np.unique(mds, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(vals, cnts, color="#357")
    ax.set_xlabel("realized forced-run length (winner fours)")
    ax.set_ylabel("# enabling shapes (kept)")
    ax.set_title("Kept enabling-shape forced-run-length distribution")
    for v, c in zip(vals, cnts):
        ax.text(v, c, str(int(c)), ha="center", va="bottom", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-off", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=150)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    H = np.load(args.harvest)
    C = np.load(args.control)
    hb, hm, hmd, htags = H["boards"], H["moves"], H["mds"], H["tags"]
    cb, cm = C["boards"], C["moves"]
    report = {"n_harvest": int(len(hb)), "n_control": int(len(cb)),
              "harvest_offensive": int((htags == "off").sum()),
              "harvest_defensive": int((htags == "def").sum())}

    # ---- run-length figure ----
    report["fig_run_length"] = run_length_fig(
        hmd, os.path.join(args.out, "run_length_hist.png"))

    # ---- (a) connected-correlation DCA, harvest vs control ----
    nfair = min(len(hb), len(cb))
    rng = np.random.default_rng(1)
    hb_f = hb[rng.choice(len(hb), nfair, replace=False)] if len(hb) > nfair else hb
    cb_f = cb[rng.choice(len(cb), nfair, replace=False)] if len(cb) > nfair else cb
    dca_h = connected_corr(hb_f, max_off=args.max_off, n_perm=args.n_perm, seed=10)
    dca_c = connected_corr(cb_f, max_off=args.max_off, n_perm=args.n_perm, seed=11)
    report["dca_fair_n"] = int(nfair)
    report["dca_harvest_top_noncollinear"] = top_noncollinear(dca_h)
    report["dca_harvest_top_collinear"] = top_collinear(dca_h)
    report["dca_control_top_noncollinear"] = top_noncollinear(dca_c)
    report["dca_control_top_collinear"] = top_collinear(dca_c)
    report["fig_dca_harvest"] = z_heatmap(
        dca_h, args.max_off, os.path.join(args.out, "dca_harvest.png"),
        f"Enabling shapes: connected-corr z (n={nfair})")
    report["fig_dca_control"] = z_heatmap(
        dca_c, args.max_off, os.path.join(args.out, "dca_control.png"),
        f"VCT control (line-bound): connected-corr z (n={nfair})")

    def max_nc_z(res):
        return round(max(d["z"] for o, d in res.items() if not d["collinear"]), 2)
    report["max_noncollinear_z_harvest"] = max_nc_z(dca_h)
    report["max_noncollinear_z_control"] = max_nc_z(dca_c)

    # ---- (a') per run-length bucket on harvest ----
    # CRITICAL: connected-corr z scales with sqrt(#boards), so buckets of
    # different size are NOT z-comparable. We subsample every bucket to a COMMON
    # n (the smallest qualifying bucket) for a fair z, and ALSO report the
    # n-independent effect size (max non-collinear obs connected-corr).
    buckets = {"r3": (hmd == 3), "r4-5": (hmd >= 4) & (hmd <= 5),
               "r6plus": (hmd >= 6)}
    qual = {k: int(m.sum()) for k, m in buckets.items() if int(m.sum()) >= 120}
    n_fair_b = min(qual.values()) if qual else 0
    report["bucket_fair_n"] = n_fair_b

    def max_nc_obs(res):
        return round(max(d["obs"] for o, d in res.items() if not d["collinear"]), 5)

    rngb = np.random.default_rng(99)
    report["buckets"] = {}
    for name, mask in buckets.items():
        n = int(mask.sum())
        entry = {"n": n}
        if n >= 120 and n_fair_b:
            bb = hb[mask]
            sub = bb[rngb.choice(n, n_fair_b, replace=False)] if n > n_fair_b else bb
            d = connected_corr(sub, max_off=args.max_off,
                               n_perm=args.n_perm, seed=20)
            entry["max_noncollinear_z_fair_n"] = max_nc_z(d)
            entry["max_noncollinear_obs"] = max_nc_obs(d)   # effect size, n-free
            entry["top_noncollinear"] = top_noncollinear(d, k=4)
            entry["fig"] = z_heatmap(
                d, args.max_off, os.path.join(args.out, f"dca_{name}.png"),
                f"Enabling shapes [{name}] (n={n_fair_b} of {n})")
        report["buckets"][name] = entry

    # ---- (b) class-averaging vocabularies ----
    report["logos_harvest"] = cluster_logos(
        hb, hm, os.path.join(args.out, "logos_harvest.png"),
        f"Enabling-shape vocabulary (catalyst-centred, n={len(hb)})")
    report["logos_control"] = cluster_logos(
        cb, cm, os.path.join(args.out, "logos_control.png"),
        f"VCT line-bound vocabulary (n={len(cb)})")
    # 6+ bucket vocabulary (the deepest gold), if enough
    if int(buckets["r6plus"].sum()) >= 40:
        report["logos_r6plus"] = cluster_logos(
            hb[buckets["r6plus"]], hm[buckets["r6plus"]],
            os.path.join(args.out, "logos_r6plus.png"),
            f"Enabling vocab, run>=6 (n={int(buckets['r6plus'].sum())})", k=6)

    # ---- (c) cross-check: count-preserving MI null ----
    mi_h = mi_bond_map(hb_f[:1500], max_off=args.max_off, n_perm=120, seed=30)
    mi_c = mi_bond_map(cb_f[:1500], max_off=args.max_off, n_perm=120, seed=31)

    def mi_top_nc(mi):
        items = [(o, d) for o, d in mi.items() if not d["collinear"]]
        items.sort(key=lambda kv: -kv[1]["z"])
        return [{"offset": list(o), "z": round(d["z"], 2)} for o, d in items[:6]]
    report["mi_countnull_harvest_top_nc"] = mi_top_nc(mi_h)
    report["mi_countnull_control_top_nc"] = mi_top_nc(mi_c)

    # ---- verdict ----
    zh, zc = report["max_noncollinear_z_harvest"], report["max_noncollinear_z_control"]
    # effect sizes (n-independent) for the fair contrast
    obs_h = round(max(d["obs"] for o, d in dca_h.items() if not d["collinear"]), 5)
    obs_c = round(max(d["obs"] for o, d in dca_c.items() if not d["collinear"]), 5)
    coll_h = report["dca_harvest_top_collinear"][0]["z"]
    coll_c = report["dca_control_top_collinear"][0]["z"]
    bucket_obs_trend = [report["buckets"][b].get("max_noncollinear_obs")
                        for b in ("r3", "r4-5", "r6plus")]
    bucket_zfair_trend = [report["buckets"][b].get("max_noncollinear_z_fair_n")
                          for b in ("r3", "r4-5", "r6plus")]
    report["verdict"] = {
        "max_noncollinear_z_harvest_fairn": zh,
        "max_noncollinear_z_control_fairn": zc,
        "max_noncollinear_obs_harvest": obs_h,
        "max_noncollinear_obs_control": obs_c,
        "top_collinear_z_harvest": coll_h,
        "top_collinear_z_control": coll_c,
        "harvest_noncollinear_above_null": bool(zh > 4.0),
        "harvest_noncollinear_stronger_than_control": bool(obs_h > obs_c),
        "noncoll_over_coll_ratio_harvest": round(zh / coll_h, 2),
        "noncoll_over_coll_ratio_control": round(zc / coll_c, 2),
        "bucket_noncollinear_OBS_trend_r3_r45_r6": bucket_obs_trend,
        "bucket_noncollinear_Zfairn_trend_r3_r45_r6": bucket_zfair_trend,
        "note": "z is connected-correlation vs a per-cell-marginal-preserving "
                "independent-Bernoulli null (z scales ~sqrt(n) -> only compare z at "
                "equal n; obs is the n-free effect size). collinear=line structure; "
                "non-collinear z>~4 = genuine 2-D (non-line) coupling. The "
                "non-coll/coll RATIO tests whether setups are DISPROPORTIONATELY "
                "non-line vs merely more-structured overall.",
    }

    with open(os.path.join(args.out, "science_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
