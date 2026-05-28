#!/usr/bin/env python3
"""Eval-sims × eval-VCF sweep driver — the one-command orchestrator for the
RESUME PLAYBOOK step 1 probe (derby-5xs).

The matured v8 champion 100%s heuristic + lookahead2; the SOLE binding gap to
the v9 "100% target" (Jason, 2026-05-27) is lookahead4-as-BLACK ~50% wins
(rest DRAWS, not losses). The H2 finding (code-walk, 2026-05-28): the ~50%
ceiling is SEARCH-DEPTH at eval, not structural. Two cheap eval-config levers
attack it directly without any training change:

  (a) raise eval --sims (100 -> {200, 400, 800}: more total MCTS depth)
  (b) eval-VCF root overlay (--eval-vcf-nodes {0, 200, 800}: forced-fours found
      deterministically; same tactical class lookahead4 defends)

This driver enumerates the {sims} × {eval-vcf-nodes} grid against a single
matured checkpoint vs lookahead:depth=4 (the binding opponent), one cell at a
time (cheapest first), reports the per-cell color-split W/L/D, and computes
each cell's distance-to-100% via the EXISTING formula in
scripts/report_100pct.py (imported, NOT duplicated).

Output:
  - JSONL: one row per cell to ``--output`` (default
    ``probe_100pct_<timestamp>.jsonl``).
  - stdout: a human-readable per-cell table + a final 4x3 markdown grid of
    distances with the best cell tagged.

NO training, NO new lane, NO derby cell. Eval-only on a frozen checkpoint.

Wall-time guidance (rule-of-thumb, CPU eval — the orchestrator's runtime is
GPU-and-checkpoint dependent):

  cell wall ~  n_games * sims_per_move * avg_plies * (1 / moves_per_sec)
            +  vcf_overhead (bounded by --eval-vcf-nodes per move)

Order-of-magnitude (eval_worker.py header says ~<1 min for the n=20 default
random+heuristic+lookahead2 pass at 100 sims, CPU). Each probe cell here is
~40 games (single baseline), so:

  ~     sims=100  vcf=0    -> ~1-2 minutes  (cheapest)
  ~     sims=800  vcf=800  -> ~10-25 minutes (most expensive)

The 12-cell sweep should fit in ~1-3 hours of serial GPU time. The
``--dry-run`` flag lists the grid without running anything.

Usage:

    python scripts/probe_100pct.py \\
        --checkpoint sweep_runs/derby_v8/_peaks/champ/peak.pt \\
        --baseline lookahead:depth=4 \\
        --games-per-cell 40 \\
        --sims-grid 100,200,400,800 \\
        --vcf-nodes-grid 0,200,800 \\
        --output probe_100pct_results.jsonl

If the Δelo Derby is currently running, the driver refuses to proceed unless
``--i-know-derby-is-running`` is passed (same defensive pattern as
``scripts/reclaim_worktrees.py``). The probe is eval-only and does NOT touch
the derby's checkpoints or wandb, but the GPU lane is single-tenant.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Import the EXISTING distance formula so we don't duplicate it.
# scripts/report_100pct.py exposes ``score(agg)`` which returns
# (per_baseline_dict, distance_float). We adapt it to a single-baseline aggregate
# in ``cell_distance_to_100`` below.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_100pct import score as report_100pct_score  # noqa: E402


# ---------------------------------------------------------------------------
# Cell grid


@dataclass(frozen=True)
class Cell:
    """A single (sims, eval_vcf_nodes) probe point."""

    sims: int
    eval_vcf_nodes: int

    def label(self) -> str:
        return f"sims={self.sims},vcf={self.eval_vcf_nodes}"


def enumerate_cells(sims_grid: list[int], vcf_nodes_grid: list[int]) -> list[Cell]:
    """Cartesian product of the two grids, cheapest first.

    Order: outer = sims (ascending), inner = vcf_nodes (ascending) — so the
    fastest cell runs first and the most expensive cell last. Failing fast on
    the cheapest cell is the right default when GPU time is finite.
    """
    cells: list[Cell] = []
    for sims in sorted(set(sims_grid)):
        for vcf in sorted(set(vcf_nodes_grid)):
            cells.append(Cell(sims=sims, eval_vcf_nodes=vcf))
    return cells


# ---------------------------------------------------------------------------
# Distance-to-100% — single-baseline adaptation


def cell_distance_to_100(black_w: int, black_l: int, black_d: int,
                         white_w: int, white_l: int, white_d: int,
                         baseline_label: str) -> tuple[float, float, float]:
    """Compute a SINGLE-baseline distance-to-100% by pooling the cell's
    color-split totals into the same aggregate shape ``report_100pct.score``
    expects, then running its formula.

    Returns ``(distance, black_win_rate, white_loss_rate)`` where ``distance``
    is ``(1 - black_win_rate) + white_loss_rate`` — exactly the per-baseline
    contribution to ``report_100pct.score``'s sum. We DELIBERATELY call into
    ``report_100pct_score`` rather than re-implementing the formula, so the
    single source of truth stays in ``report_100pct.py``.
    """
    # report_100pct.score iterates over ``BASELINES`` (heuristic, lookahead2,
    # lookahead4). Inject this cell's counts under one of those keys and put
    # zero-game stub rows under the other two; the ``if bt and wt:`` guard in
    # ``score`` skips zero-game baselines (they don't contribute to dist).
    # Using "lookahead4" as the key works for the default --baseline, but the
    # caller may use heuristic/lookahead2 for sanity tests — we accept that the
    # contribution formula ``(1-bw) + wl`` is identical regardless of which
    # baseline key we slot into.
    from report_100pct import BASELINES
    agg = {b: {f"{c}_{o}": 0 for c in ("black", "white") for o in ("w", "l", "d")}
           for b in BASELINES}
    # Pick a key from BASELINES so the formula's iteration includes our row.
    # Prefer matching the actual baseline label suffix when possible.
    key = BASELINES[-1]  # default: lookahead4
    for b in BASELINES:
        if b in baseline_label or baseline_label.endswith(b):
            key = b
            break
    agg[key]["black_w"] = black_w
    agg[key]["black_l"] = black_l
    agg[key]["black_d"] = black_d
    agg[key]["white_w"] = white_w
    agg[key]["white_l"] = white_l
    agg[key]["white_d"] = white_d
    per, dist = report_100pct_score(agg)
    bw, wl, _bt, _wt = per[key]
    return dist, bw, wl


# ---------------------------------------------------------------------------
# Eval entrypoint — the single mockable seam


@dataclass
class CellResult:
    sims: int
    eval_vcf_nodes: int
    n_games: int
    baseline: str
    black_w: int = 0
    black_l: int = 0
    black_d: int = 0
    white_w: int = 0
    white_l: int = 0
    white_d: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    distance: float = 0.0
    black_win_rate: float = 0.0
    white_loss_rate: float = 0.0
    wall_secs: float = 0.0
    error: str | None = None


def run_cell_eval(*, checkpoint: str, baseline: str, n_games: int,
                  sims: int, eval_vcf_nodes: int, c_puct: float = 1.5,
                  device: str = "cpu", seed: int = 0,
                  n_workers: int = 1) -> dict:
    """Run one cell's eval and return a dict of color-split tallies.

    This is the GPU/torch-touching seam — tests mock THIS function. Keeping it
    a single thin wrapper around the existing eval harness means the probe
    driver itself stays pure-Python / no-torch at import time, and the test
    suite can drop in a deterministic stub without touching torch or MPS.

    Returns a dict with keys
        n_games, wins, losses, draws,
        black_w, black_l, black_d, white_w, white_l, white_d.
    """
    # Lazy imports so importing this module costs nothing (tests don't pay it).
    from gomoku.eval import mcts_picker, play_match_parallel, play_match_pickers
    from gomoku.match import build_player, parse_spec
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint

    opp_spec = parse_spec(baseline)
    if opp_spec.kind == "model":
        raise SystemExit(f"--baseline must not be a model spec: {baseline!r}")
    opp_picker = build_player(opp_spec)

    if n_workers > 1:
        res = play_match_parallel(
            checkpoint_path=checkpoint,
            opp_spec=baseline,
            n_games=n_games,
            seed=seed,
            n_workers=n_workers,
            sims=sims,
            c_puct=c_puct,
            device=device,
            eval_vcf_nodes=eval_vcf_nodes,
            eval_vcf_depth=0,
        )
    else:
        model, _payload = load_checkpoint(checkpoint, device=device)
        model = fuse_model_for_inference(model)
        evaluator = make_torch_evaluator(model, device)
        model_picker = mcts_picker(
            evaluator,
            n_simulations=sims,
            c_puct=c_puct,
            eval_vcf_nodes=eval_vcf_nodes,
            eval_vcf_depth=0,
        )
        res = play_match_pickers(model_picker, opp_picker,
                                 n_games=n_games, seed=seed)

    return {
        "n_games": res.n_games,
        "wins": res.wins,
        "losses": res.losses,
        "draws": res.draws,
        "black_w": res.black_w,
        "black_l": res.black_l,
        "black_d": res.black_d,
        "white_w": res.white_w,
        "white_l": res.white_l,
        "white_d": res.white_d,
    }


# ---------------------------------------------------------------------------
# Driver


def is_derby_running() -> bool:
    """True iff a delo_derby.py process is running. Same defensive pattern as
    scripts/derby_watchdog.sh — pgrep -f."""
    try:
        rc = subprocess.run(
            ["pgrep", "-f", "delo_derby.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        return rc == 0
    except (FileNotFoundError, OSError):
        return False


def run_probe(*, checkpoint: str, baseline: str, cells: list[Cell],
              n_games: int, seed: int, c_puct: float, device: str,
              n_workers: int, eval_fn=None) -> list[CellResult]:
    """Iterate the cell grid serially, calling ``eval_fn`` for each cell.

    ``eval_fn`` defaults to ``run_cell_eval`` (the GPU-touching path); tests
    pass a deterministic stub. The function signature is kept identical so the
    driver doesn't need to know which one is wired in.
    """
    if eval_fn is None:
        eval_fn = run_cell_eval

    results: list[CellResult] = []
    for cell in cells:
        t0 = time.perf_counter()
        try:
            tally = eval_fn(
                checkpoint=checkpoint,
                baseline=baseline,
                n_games=n_games,
                sims=cell.sims,
                eval_vcf_nodes=cell.eval_vcf_nodes,
                c_puct=c_puct,
                device=device,
                seed=seed,
                n_workers=n_workers,
            )
            dist, bw, wl = cell_distance_to_100(
                black_w=tally["black_w"], black_l=tally["black_l"],
                black_d=tally["black_d"],
                white_w=tally["white_w"], white_l=tally["white_l"],
                white_d=tally["white_d"],
                baseline_label=baseline,
            )
            cr = CellResult(
                sims=cell.sims,
                eval_vcf_nodes=cell.eval_vcf_nodes,
                n_games=tally["n_games"],
                baseline=baseline,
                wins=tally["wins"],
                losses=tally["losses"],
                draws=tally["draws"],
                black_w=tally["black_w"], black_l=tally["black_l"],
                black_d=tally["black_d"],
                white_w=tally["white_w"], white_l=tally["white_l"],
                white_d=tally["white_d"],
                distance=dist,
                black_win_rate=bw,
                white_loss_rate=wl,
                wall_secs=time.perf_counter() - t0,
            )
        except Exception as e:  # pragma: no cover - defensive in driver
            cr = CellResult(
                sims=cell.sims,
                eval_vcf_nodes=cell.eval_vcf_nodes,
                n_games=n_games,
                baseline=baseline,
                wall_secs=time.perf_counter() - t0,
                error=f"{type(e).__name__}: {e}",
            )
        results.append(cr)
    return results


# ---------------------------------------------------------------------------
# Output


def format_per_cell_table(results: list[CellResult]) -> str:
    """A per-cell row table (the long-form view)."""
    lines = []
    header = (f"{'sims':>5s} {'vcf':>5s} {'n':>4s}  "
              f"{'BlackW/L/D':>14s}  {'WhiteW/L/D':>14s}  "
              f"{'Bwin':>6s} {'Wloss':>6s}  {'dist':>6s}  {'secs':>6s}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        if r.error:
            lines.append(f"{r.sims:>5d} {r.eval_vcf_nodes:>5d} {r.n_games:>4d}  "
                         f"{'ERROR':>14s}  {r.error}")
            continue
        lines.append(
            f"{r.sims:>5d} {r.eval_vcf_nodes:>5d} {r.n_games:>4d}  "
            f"{r.black_w:>3d}/{r.black_l:>3d}/{r.black_d:>3d}  ".rjust(16)
            + f"{r.white_w:>3d}/{r.white_l:>3d}/{r.white_d:>3d}  ".rjust(16)
            + f"{r.black_win_rate*100:5.0f}% {r.white_loss_rate*100:5.0f}%  "
            f"{r.distance:6.3f}  {r.wall_secs:6.1f}"
        )
    return "\n".join(lines)


def format_distance_grid(results: list[CellResult],
                         sims_grid: list[int],
                         vcf_grid: list[int]) -> str:
    """A 2D markdown table of distance-to-100% with the best cell starred.

    Rows = sims (ascending), Columns = vcf_nodes (ascending). Cells without a
    successful result render as "-". The best (lowest) cell gets a "*" suffix.
    """
    sims_sorted = sorted(set(sims_grid))
    vcf_sorted = sorted(set(vcf_grid))
    lookup: dict[tuple[int, int], CellResult] = {
        (r.sims, r.eval_vcf_nodes): r for r in results if r.error is None
    }
    # Find best cell (lowest distance).
    successful = [r for r in results if r.error is None]
    best_key: tuple[int, int] | None = None
    if successful:
        best = min(successful, key=lambda r: r.distance)
        best_key = (best.sims, best.eval_vcf_nodes)

    # Column headers: vcf values.
    col_w = 10
    lines = []
    lines.append(f"distance-to-100% grid (lower is better; "
                 f"0.0 = win-all-black/lose-none-white vs the baseline)")
    lines.append("")
    header = f"| {'sims \\ vcf':<{col_w}} | " + " | ".join(
        f"{v:>{col_w}d}" for v in vcf_sorted) + " |"
    sep = "|" + "-" * (col_w + 2) + ("|" + "-" * (col_w + 2)) * len(vcf_sorted) + "|"
    lines.append(header)
    lines.append(sep)
    for s in sims_sorted:
        row_cells = [f"{s:<{col_w}d}"]
        for v in vcf_sorted:
            r = lookup.get((s, v))
            if r is None:
                cell = "-"
            else:
                tag = "*" if best_key == (s, v) else " "
                cell = f"{r.distance:6.3f}{tag}"
            row_cells.append(f"{cell:>{col_w}}")
        lines.append("| " + " | ".join(row_cells) + " |")
    if best_key is not None:
        bw_best = lookup[best_key].black_win_rate
        lines.append("")
        lines.append(f"best cell: sims={best_key[0]} vcf={best_key[1]} "
                     f"dist={lookup[best_key].distance:.3f} "
                     f"L4-black-winrate={bw_best*100:.0f}%  (*)")
    return "\n".join(lines)


def write_jsonl(results: list[CellResult], path: Path, meta: dict) -> None:
    """One JSON object per line. First line is a meta header
    ({"meta": {...}}); subsequent lines are CellResult dicts."""
    with open(path, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


# ---------------------------------------------------------------------------
# CLI


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="eval-sims × eval-VCF sweep driver vs lookahead:depth=4 "
                    "(RESUME PLAYBOOK step 1; H2 search-depth probe).",
    )
    p.add_argument("--checkpoint", type=str,
                   default="sweep_runs/derby_v8/_peaks/champ/peak.pt",
                   help="Path to the matured checkpoint to evaluate. Default "
                        "points at the v8 champion peak.")
    p.add_argument("--baseline", type=str, default="lookahead:depth=4",
                   help="Opponent player spec (see gomoku/match.py). Default "
                        "is the BINDING one (lookahead4); --baseline heuristic "
                        "or lookahead:depth=2 is allowed for sanity checks.")
    p.add_argument("--games-per-cell", type=int, default=40,
                   help="Games per cell. H2 finding: 10 is too noisy; ~40-100 "
                        "gives ~16pp 95%% CI on each color. 40 is the floor.")
    p.add_argument("--sims-grid", type=str, default="100,200,400,800",
                   help="Comma-separated MCTS sims values. Default 4 values "
                        "× 3 vcf values = 12 cells.")
    p.add_argument("--vcf-nodes-grid", type=str, default="0,200,800",
                   help="Comma-separated eval-vcf-nodes values "
                        "(0 disables the overlay).")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path. Default "
                        "probe_100pct_<UTC-timestamp>.jsonl in CWD.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--device", type=str, default="cpu",
                   help="Eval device. Default cpu so we don't fight a live "
                        "trainer for MPS.")
    p.add_argument("--n-workers", type=int, default=1,
                   help="If >1, parallelise the n_games per cell via "
                        "play_match_parallel.")
    p.add_argument("--i-know-derby-is-running", action="store_true",
                   help="Acknowledge a live delo_derby.py and proceed anyway. "
                        "The probe is eval-only — it does not touch the "
                        "derby's checkpoints — but the GPU lane is "
                        "single-tenant, so this gate prevents accidental "
                        "contention.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the grid and exit without running any eval.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None,
         *, eval_fn=None,
         derby_check=is_derby_running) -> int:
    args = parse_args(argv)
    sims_grid = _parse_int_list(args.sims_grid)
    vcf_grid = _parse_int_list(args.vcf_nodes_grid)
    if not sims_grid or not vcf_grid:
        print("error: empty --sims-grid or --vcf-nodes-grid", file=sys.stderr)
        return 2

    cells = enumerate_cells(sims_grid, vcf_grid)

    if args.dry_run:
        print(f"[dry-run] would evaluate {len(cells)} cells against "
              f"{args.baseline} with {args.games_per_cell} games/cell")
        for c in cells:
            print(f"  - {c.label()}")
        return 0

    # Defensive: refuse to compete with a live derby unless explicitly
    # acknowledged. Same pattern as scripts/reclaim_worktrees.py.
    if derby_check() and not args.i_know_derby_is_running:
        print("error: delo_derby.py appears to be running. The probe is "
              "eval-only (does not touch derby checkpoints / wandb) but the "
              "GPU lane is single-tenant. Pass "
              "--i-know-derby-is-running to proceed anyway.",
              file=sys.stderr)
        return 3

    if args.output is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        args.output = f"probe_100pct_{ts}.jsonl"
    out_path = Path(args.output)

    print(f"[probe] checkpoint={args.checkpoint}")
    print(f"[probe] baseline={args.baseline}  games/cell={args.games_per_cell}")
    print(f"[probe] grid: sims={sims_grid}  vcf_nodes={vcf_grid}  "
          f"({len(cells)} cells)")
    print(f"[probe] output={out_path}")

    results = run_probe(
        checkpoint=args.checkpoint,
        baseline=args.baseline,
        cells=cells,
        n_games=args.games_per_cell,
        seed=args.seed,
        c_puct=args.c_puct,
        device=args.device,
        n_workers=args.n_workers,
        eval_fn=eval_fn,
    )

    meta = {
        "checkpoint": args.checkpoint,
        "baseline": args.baseline,
        "games_per_cell": args.games_per_cell,
        "sims_grid": sims_grid,
        "vcf_nodes_grid": vcf_grid,
        "seed": args.seed,
        "c_puct": args.c_puct,
        "device": args.device,
        "n_workers": args.n_workers,
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_jsonl(results, out_path, meta)

    print()
    print(format_per_cell_table(results))
    print()
    print(format_distance_grid(results, sims_grid, vcf_grid))
    print()
    print(f"[probe] wrote {len(results)} cells to {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
