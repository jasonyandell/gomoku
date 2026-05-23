"""Render the canonical sweep contour from summary.tsv.

Reads <sweep-dir>/summary.tsv and writes:
  - contour.png     (the wall chart: throughput per cell, faceted by model)
  - axes.png        (single-axis sweeps: workers, sims, wave-size, games/worker)

Filters out cell_status=failed rows. Run after the sweep completes.

Usage:
  python scripts/plot_canonical_sweep.py --sweep-dir sweep_logs/canonical-sweep-<TS>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row.get("cell_status", "ok") == "failed":
                continue
            for k in ("workers", "games_per_batch", "n_simulations", "wave_size",
                     "total_games", "total_aug_examples", "total_raw_plies"):
                if row.get(k):
                    row[k] = int(row[k])
            for k in ("wall_secs", "aug_pos_per_sec", "games_per_sec", "plies_mean"):
                if row.get(k):
                    row[k] = float(row[k])
            rows.append(row)
    return rows


def plot_axes(rows: list[dict], out_path: Path) -> None:
    """One subplot per single-axis sweep, holding the default constant."""
    DEFAULT = dict(model="small", workers=8, games_per_batch=8, n_simulations=400, wave_size=64)

    def held(row, axis):
        for k, v in DEFAULT.items():
            if k == axis:
                continue
            if row.get(k) != v:
                return False
        return True

    axes_specs = [
        ("workers", "workers", "small / G=8 / S=400 / V=64"),
        ("n_simulations", "n-sims", "small / W=8 / G=8 / V=64"),
        ("wave_size", "wave-size", "small / W=8 / G=8 / S=400"),
        ("games_per_batch", "games/worker", "small / W=8 / S=400 / V=64"),
    ]
    fig, axarr = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (axis, label, sub) in zip(axarr.flat, axes_specs):
        pts = [(r[axis], r["aug_pos_per_sec"]) for r in rows if held(r, axis)]
        pts.sort()
        if not pts:
            ax.set_title(f"{label}\n(no data)")
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", color="C0")
        ax.set_xlabel(label)
        ax.set_ylabel("aug pos/sec")
        ax.set_title(f"{label} sweep\n{sub}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Canonical sweep: single-axis pivots (held = default)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_model_compare(rows: list[dict], out_path: Path) -> None:
    """Bar chart per model at the canonical default cell."""
    canon = [r for r in rows
             if r["workers"] == 8 and r["games_per_batch"] == 8
             and r["n_simulations"] == 400 and r["wave_size"] == 64]
    canon.sort(key=lambda r: ("tiny", "small", "medium").index(r["model"]))
    if not canon:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r["model"] for r in canon]
    ys = [r["aug_pos_per_sec"] for r in canon]
    bars = ax.bar(xs, ys, color=["C2", "C0", "C3"])
    ax.set_ylabel("aug pos/sec")
    ax.set_title("Model size at canonical default (W8 G8 S400 V64)")
    ax.grid(True, axis="y", alpha=0.3)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.0f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_contour(rows: list[dict], out_path: Path) -> None:
    """The wall chart: aug-pos/sec heatmap over workers x games-per-worker,
    one panel per model. Uses cells where sims=400 / wave=64."""
    models = sorted({r["model"] for r in rows}, key=lambda m: ("tiny", "small", "medium").index(m))
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.5), squeeze=False)
    for ax, model in zip(axes[0], models):
        cells = [r for r in rows
                 if r["model"] == model and r["n_simulations"] == 400 and r["wave_size"] == 64]
        if not cells:
            ax.set_title(f"{model}\n(no data at S=400 V=64)")
            continue
        ws = sorted({r["workers"] for r in cells})
        gs = sorted({r["games_per_batch"] for r in cells})
        grid = np.full((len(gs), len(ws)), np.nan)
        for r in cells:
            i = gs.index(r["games_per_batch"])
            j = ws.index(r["workers"])
            grid[i, j] = r["aug_pos_per_sec"]
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(ws)))
        ax.set_xticklabels(ws)
        ax.set_yticks(range(len(gs)))
        ax.set_yticklabels(gs)
        ax.set_xlabel("workers")
        ax.set_ylabel("games/worker")
        ax.set_title(f"{model} (sims=400, wave=64)")
        for i in range(len(gs)):
            for j in range(len(ws)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                            color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="aug pos/sec")
    fig.suptitle("Canonical sweep contour: workers x games-per-worker per model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", required=True, type=Path)
    args = p.parse_args()

    summary = args.sweep_dir / "summary.tsv"
    if not summary.exists():
        raise SystemExit(f"no summary.tsv at {summary}")
    rows = load_summary(summary)
    if not rows:
        raise SystemExit("summary.tsv has no successful rows")

    plot_axes(rows, args.sweep_dir / "axes.png")
    plot_model_compare(rows, args.sweep_dir / "model_compare.png")
    plot_contour(rows, args.sweep_dir / "contour.png")
    print(f"wrote {args.sweep_dir / 'axes.png'}")
    print(f"wrote {args.sweep_dir / 'model_compare.png'}")
    print(f"wrote {args.sweep_dir / 'contour.png'}")


if __name__ == "__main__":
    main()
