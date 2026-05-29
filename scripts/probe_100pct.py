#!/usr/bin/env python3
"""Eval-config sweep driver — the one-command orchestrator for the
RESUME PLAYBOOK step 1 probe (derby-5xs, extended in derby-u8d, operability
gates retrofitted in derby-cec).

OPERABILITY PROPERTIES (derby-cec, 2026-05-28):
  - Streaming output: per-cell JSON rows are flushed to ``--output`` as each
    cell completes (append mode), so a mid-run crash leaves all completed
    cells safe on disk. The meta header is written once at run start.
  - ``--resume`` (default ON): if the ``--output`` file already exists, the
    driver parses any prior cell rows, builds a set of completed cell-keys
    (tuple of all axis values), and SKIPS those cells on the new run. Pass
    ``--no-resume`` to ignore the file (still appends).
  - Honest timing self-report: at run start the driver prints a plan line
    with cell count, estimated per-cell wall, and total ETA; at end it
    prints the actual/estimate ratio so the operator can calibrate future
    estimates (and the driver suggests filing a perf-meta bead when the
    ratio exceeds 2x).
  - Grids RESPECT single values: ``--fpu-c-grid 0.45`` runs EXACTLY ONE
    cell at fpu=0.45 (no silent expansion to {0, 0.45}). To explicitly
    ablate a non-zero value against OFF, pass ``--fpu-c-grid 0,0.45``.
    This REVERSED the derby-u8d behaviour that silently injected OFF.

The matured v8 champion 100%s heuristic + lookahead2; the SOLE binding gap to
the v9 "100% target" (Jason, 2026-05-27) is lookahead4-as-BLACK ~50% wins
(rest DRAWS, not losses). The H2 finding (code-walk, 2026-05-28): the ~50%
ceiling is SEARCH-DEPTH at eval, not structural. Cheap eval-config levers
attack it directly without any training change.

Five eval-config axes are swept here (the legacy two + three new search seams
from derby-3w0 / derby-jmi / derby-b3n):

  (1) ``--sims-grid``                MCTS sims at eval.
  (2) ``--vcf-nodes-grid``           Root VCF overlay (forced-fours).
  (3) ``--fpu-c-grid``               FPU reduction c (derby-3w0).
  (4) ``--reuse-tree-grid``          Cross-ply tree reuse {0,1} (derby-jmi).
  (5) ``--proven-prop-grid``         Proven-win/loss propagation {0,1}
                                     (derby-b3n).
      ``--proven-vcf-leaf-nodes-grid``  Bounded leaf-VCF nodes (derby-b3n,
                                        requires proven-prop=1).

Default behaviour is BACKWARD-COMPATIBLE: each new axis defaults to a single
OFF value, so the default invocation is the same 4×3=12-cell sweep as
derby-5xs. Specify any new --*-grid to expand the matrix.

This driver enumerates the Cartesian product against a single matured
checkpoint vs lookahead:depth=4 (the binding opponent), one cell at a time
(cheapest first by default), reports the per-cell color-split W/L/D, and
computes each cell's distance-to-100% via the EXISTING formula in
scripts/report_100pct.py (imported, NOT duplicated).

Cheap-first ordering (``--order cheap-first``, the default): runs the all-OFF
baseline first, then each lever isolated at its smallest non-OFF value, then
pair-combos, then everything together — so when an early cell hits the target
or the matrix is plainly worse than baseline, we drop the rest without paying
the expensive cells. ``--order full`` falls back to the legacy Cartesian
enumeration (outer-to-inner by axis order); ``--order custom-list`` is
reserved (currently maps to ``full``).

Early-stop knobs (both default OFF):
  - ``--early-stop-on-target DIST`` — first cell with distance ≤ DIST wins;
    skip remaining.
  - ``--worse-than-baseline-margin MARGIN`` — once the all-OFF baseline has
    been measured, drop any subsequent cell whose distance is ≥
    ``baseline + MARGIN`` (it's clearly worse, no point running its expensive
    siblings either — we mark it ``early-stop:worse``).

Output:
  - JSONL: one row per cell to ``--output`` (default
    ``probe_100pct_<timestamp>.jsonl``).
  - stdout: a human-readable per-cell table + (when only 1-2 axes vary) a 2D
    markdown grid of distances; when ≥3 axes vary, a long-form best-by-axis
    summary instead.

NO training, NO new lane, NO derby cell. Eval-only on a frozen checkpoint.

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


# Axis order is the canonical iteration / label order. Keep stable for tests.
AXIS_NAMES = (
    "sims",
    "eval_vcf_nodes",
    "fpu_reduction_c",
    "reuse_tree",
    "proven_prop",
    "proven_vcf_leaf_nodes",
)


# Per-axis OFF value (the byte-identical legacy baseline). MUST match the
# defaults in gomoku/eval.py:mcts_picker / play_match_pickers /
# play_match_parallel so a default-grid invocation reproduces derby-5xs exactly.
OFF_VALUES: dict[str, float | int] = {
    "sims": 100,          # legacy "cheapest" sims setting (the existing default)
    "eval_vcf_nodes": 0,
    "fpu_reduction_c": 0.0,
    "reuse_tree": 0,
    "proven_prop": 0,
    "proven_vcf_leaf_nodes": 0,
}


@dataclass(frozen=True)
class Cell:
    """A single eval-config probe point — one value per axis.

    The two legacy axes (``sims``, ``eval_vcf_nodes``) keep their original
    names for backward compatibility (the existing JSONL + tests assume them).
    The three new axes are named after their CLI flag stems.
    """

    sims: int
    eval_vcf_nodes: int
    fpu_reduction_c: float = 0.0
    reuse_tree: int = 0
    proven_prop: int = 0
    proven_vcf_leaf_nodes: int = 0

    def label(self) -> str:
        return (
            f"sims={self.sims},vcf={self.eval_vcf_nodes}"
            f",fpu={self.fpu_reduction_c:g},reuse={self.reuse_tree}"
            f",pprop={self.proven_prop},pvcf={self.proven_vcf_leaf_nodes}"
        )

    def axis_value(self, axis: str) -> float | int:
        return getattr(self, axis)

    def is_off(self, axis: str) -> bool:
        return self.axis_value(axis) == OFF_VALUES[axis]

    def num_active_levers(self) -> int:
        """Count of axes that are NOT at their OFF value. Cheap = small count."""
        return sum(0 if self.is_off(a) else 1 for a in AXIS_NAMES)


def _normalize_grid(values, axis: str) -> list:
    """Dedupe + sort a per-axis grid; RESPECT what the caller passed.

    Derby-cec (2026-05-28) reversed the prior derby-u8d behaviour of silently
    injecting OFF into every grid. The new contract: **what you pass is what
    you get** — a single-value grid is a single cell, full stop. To ablate a
    non-OFF value against OFF, pass an explicit two-value grid (e.g.
    ``--fpu-c-grid 0,0.45``).

    Rationale: the auto-injection produced a silent 2x cell-count expansion
    that bit derby-cec's 100g re-eval (8 cells silently instead of the 1
    requested). The principle of least surprise wins; cheap-first ordering
    survives because callers who *want* the baseline still pass it
    explicitly.
    """
    out = list(dict.fromkeys(values))  # dedupe, preserve order
    # Sort ascending so iteration is predictable & smallest value comes first.
    return sorted(out)


def _grids_for_cells(*, sims_grid, vcf_nodes_grid, fpu_c_grid,
                     reuse_tree_grid, proven_prop_grid,
                     proven_vcf_leaf_nodes_grid):
    """Build the per-axis normalized grid dict, keyed by axis name."""
    return {
        "sims": _normalize_grid(sims_grid, "sims"),
        "eval_vcf_nodes": _normalize_grid(vcf_nodes_grid, "eval_vcf_nodes"),
        "fpu_reduction_c": _normalize_grid(fpu_c_grid, "fpu_reduction_c"),
        "reuse_tree": _normalize_grid(reuse_tree_grid, "reuse_tree"),
        "proven_prop": _normalize_grid(proven_prop_grid, "proven_prop"),
        "proven_vcf_leaf_nodes": _normalize_grid(
            proven_vcf_leaf_nodes_grid, "proven_vcf_leaf_nodes"
        ),
    }


def enumerate_cells(
    sims_grid: list[int],
    vcf_nodes_grid: list[int],
    fpu_c_grid: list[float] | None = None,
    reuse_tree_grid: list[int] | None = None,
    proven_prop_grid: list[int] | None = None,
    proven_vcf_leaf_nodes_grid: list[int] | None = None,
) -> list[Cell]:
    """Cartesian product of all axes in canonical (full) order.

    Outer-to-inner axis order follows ``AXIS_NAMES``. Within each axis, values
    are sorted ascending and OFF is guaranteed present (see ``_normalize_grid``).

    The two legacy axes default to the caller-supplied grids (required); the
    three new axes default to a single-OFF grid when ``None`` is passed, which
    keeps backward-compatible behaviour for callers (and the legacy CLI default
    produces the same 4×3=12 cells as derby-5xs).
    """
    grids = _grids_for_cells(
        sims_grid=sims_grid,
        vcf_nodes_grid=vcf_nodes_grid,
        fpu_c_grid=fpu_c_grid if fpu_c_grid is not None else [OFF_VALUES["fpu_reduction_c"]],
        reuse_tree_grid=reuse_tree_grid if reuse_tree_grid is not None else [OFF_VALUES["reuse_tree"]],
        proven_prop_grid=proven_prop_grid if proven_prop_grid is not None else [OFF_VALUES["proven_prop"]],
        proven_vcf_leaf_nodes_grid=(
            proven_vcf_leaf_nodes_grid
            if proven_vcf_leaf_nodes_grid is not None
            else [OFF_VALUES["proven_vcf_leaf_nodes"]]
        ),
    )
    cells: list[Cell] = []
    for sims in grids["sims"]:
        for vcf in grids["eval_vcf_nodes"]:
            for fpu in grids["fpu_reduction_c"]:
                for reuse in grids["reuse_tree"]:
                    for pprop in grids["proven_prop"]:
                        for pvcf in grids["proven_vcf_leaf_nodes"]:
                            cells.append(Cell(
                                sims=sims,
                                eval_vcf_nodes=vcf,
                                fpu_reduction_c=fpu,
                                reuse_tree=reuse,
                                proven_prop=pprop,
                                proven_vcf_leaf_nodes=pvcf,
                            ))
    return cells


def order_cells_cheap_first(cells: list[Cell]) -> list[Cell]:
    """Reorder cells so the cheapest configs run first.

    Heuristic: rank each cell by

      (num_active_levers,
       sims, eval_vcf_nodes, fpu_reduction_c, reuse_tree,
       proven_prop, proven_vcf_leaf_nodes)

    Effect:
      1. The all-OFF baseline (0 active levers) is first.
      2. Single-lever cells (1 active lever) follow, sorted by axis & value
         — each lever isolated at its smallest non-OFF value before any
         pair-combos run.
      3. Pair-combos, triples, etc. follow, ordered by total active-lever count.

    The lexicographic secondary sort means within each "tier" the smaller-sims
    / smaller-overlay / smaller-fpu cells run first, so the very cheapest one
    per tier wins the slot. Early-stop on the cheap cells therefore drops the
    most expensive siblings — the whole point.

    Note: this is a stable sort, so cells that compare equal stay in their
    enumeration order (canonical Cartesian).
    """
    def key(c: Cell):
        return (
            c.num_active_levers(),
            c.sims,
            c.eval_vcf_nodes,
            c.fpu_reduction_c,
            c.reuse_tree,
            c.proven_prop,
            c.proven_vcf_leaf_nodes,
        )
    return sorted(cells, key=key)


def order_cells(cells: list[Cell], order: str) -> list[Cell]:
    """Dispatch on the ``--order`` knob. Unknown values fall through to full
    Cartesian (the canonical enumerate order)."""
    if order == "cheap-first":
        return order_cells_cheap_first(cells)
    # 'full' and 'custom-list' (reserved) preserve the canonical Cartesian
    # order, which sorts ascending on every axis (cheapest legacy cell first).
    return list(cells)


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
    from report_100pct import BASELINES
    agg = {b: {f"{c}_{o}": 0 for c in ("black", "white") for o in ("w", "l", "d")}
           for b in BASELINES}
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
    fpu_reduction_c: float = 0.0
    reuse_tree: int = 0
    proven_prop: int = 0
    proven_vcf_leaf_nodes: int = 0
    n_games: int = 0
    baseline: str = ""
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
    # Set to "target-hit" / "worse" when this run terminates the sweep early,
    # or "skipped:<reason>" when an early-stop dropped this cell entirely.
    early_stop: str | None = None


def run_cell_eval(*, checkpoint: str, baseline: str, n_games: int,
                  sims: int, eval_vcf_nodes: int,
                  fpu_reduction_c: float = 0.0,
                  reuse_tree: int = 0,
                  proven_prop: int = 0,
                  proven_vcf_leaf_nodes: int = 0,
                  c_puct: float = 1.5,
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

    use_reuse = bool(reuse_tree)
    use_pprop = bool(proven_prop)

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
            fpu_reduction_c=fpu_reduction_c,
            reuse_tree=use_reuse,
            proven_prop=use_pprop,
            proven_vcf_leaf_nodes=proven_vcf_leaf_nodes,
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
            fpu_reduction_c=fpu_reduction_c,
            reuse_tree=use_reuse,
            proven_prop=use_pprop,
            proven_vcf_leaf_nodes=proven_vcf_leaf_nodes,
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


def _is_all_off(cell: Cell) -> bool:
    return all(cell.is_off(a) for a in AXIS_NAMES)


def run_probe(*, checkpoint: str, baseline: str, cells: list[Cell],
              n_games: int, seed: int, c_puct: float, device: str,
              n_workers: int, eval_fn=None,
              early_stop_on_target: float | None = None,
              worse_than_baseline_margin: float | None = None,
              on_cell=None,
              skip_keys: set | None = None) -> list[CellResult]:
    """Iterate the cell grid serially, calling ``eval_fn`` for each cell.

    ``eval_fn`` defaults to ``run_cell_eval`` (the GPU-touching path); tests
    pass a deterministic stub.

    Early-stop semantics
    --------------------
    ``early_stop_on_target`` (default ``None`` = OFF): the FIRST cell whose
    ``distance`` is ≤ the threshold ends the sweep. That cell's
    ``early_stop="target-hit"``; remaining cells are appended with
    ``error=None``, ``early_stop="skipped:target-hit"`` and otherwise zeroed
    out — so the JSONL/table still has one row per requested cell.

    ``worse_than_baseline_margin`` (default ``None`` = OFF): once the all-OFF
    baseline cell has been measured (in cheap-first order, it runs first), any
    later cell whose distance is ≥ ``baseline_distance + margin`` gets
    ``early_stop="worse"``. The sweep DOES continue past a "worse" cell — we
    only mark it; combining "worse" with "skip its siblings" would be more
    aggressive than the bead's spec demands and would risk dropping a slightly
    worse single-lever cell that later combos with another into a winner. The
    flag's value is purely informational on each cell so the analyst can spot
    losing levers in the JSONL.

    Both knobs are independent and may be combined.

    Streaming + resume hooks (derby-cec):
        - ``on_cell``: optional callable invoked as ``on_cell(cell_result)``
          immediately after each cell completes (or is recorded as skipped).
          Used by ``main()`` to stream rows to disk + flush.
        - ``skip_keys``: optional set of cell-key tuples (in ``cell_key()``
          form) — cells whose key is in the set are skipped entirely (no
          eval call, no result row appended). Resume-friendly: callers that
          want the old "produce a row per cell" behaviour should pre-filter
          before passing ``cells`` instead of using ``skip_keys``.
    """
    if eval_fn is None:
        eval_fn = run_cell_eval

    results: list[CellResult] = []
    target_hit = False
    baseline_distance: float | None = None

    for cell in cells:
        if skip_keys is not None and cell_key(cell) in skip_keys:
            # Resume case: this cell already has a row on disk; do not re-eval
            # and do not double-write.
            continue
        if target_hit:
            cr = CellResult(
                sims=cell.sims,
                eval_vcf_nodes=cell.eval_vcf_nodes,
                fpu_reduction_c=cell.fpu_reduction_c,
                reuse_tree=cell.reuse_tree,
                proven_prop=cell.proven_prop,
                proven_vcf_leaf_nodes=cell.proven_vcf_leaf_nodes,
                n_games=n_games,
                baseline=baseline,
                early_stop="skipped:target-hit",
            )
            results.append(cr)
            if on_cell is not None:
                on_cell(cr)
            continue
        t0 = time.perf_counter()
        try:
            tally = eval_fn(
                checkpoint=checkpoint,
                baseline=baseline,
                n_games=n_games,
                sims=cell.sims,
                eval_vcf_nodes=cell.eval_vcf_nodes,
                fpu_reduction_c=cell.fpu_reduction_c,
                reuse_tree=cell.reuse_tree,
                proven_prop=cell.proven_prop,
                proven_vcf_leaf_nodes=cell.proven_vcf_leaf_nodes,
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
                fpu_reduction_c=cell.fpu_reduction_c,
                reuse_tree=cell.reuse_tree,
                proven_prop=cell.proven_prop,
                proven_vcf_leaf_nodes=cell.proven_vcf_leaf_nodes,
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
            # Record the baseline (all-OFF) distance the first time we see it.
            # In cheap-first order it's cell 0; in full order it's also the
            # first cell because OFF sorts to position 0 of every axis.
            if baseline_distance is None and _is_all_off(cell):
                baseline_distance = dist
            # Early-stop logic.
            if early_stop_on_target is not None and dist <= early_stop_on_target:
                cr.early_stop = "target-hit"
                target_hit = True
            elif (
                worse_than_baseline_margin is not None
                and baseline_distance is not None
                and not _is_all_off(cell)
                and dist >= baseline_distance + worse_than_baseline_margin
            ):
                cr.early_stop = "worse"
        except Exception as e:  # pragma: no cover - defensive in driver
            cr = CellResult(
                sims=cell.sims,
                eval_vcf_nodes=cell.eval_vcf_nodes,
                fpu_reduction_c=cell.fpu_reduction_c,
                reuse_tree=cell.reuse_tree,
                proven_prop=cell.proven_prop,
                proven_vcf_leaf_nodes=cell.proven_vcf_leaf_nodes,
                n_games=n_games,
                baseline=baseline,
                wall_secs=time.perf_counter() - t0,
                error=f"{type(e).__name__}: {e}",
            )
        results.append(cr)
        if on_cell is not None:
            on_cell(cr)
    return results


# ---------------------------------------------------------------------------
# Output


def format_per_cell_table(results: list[CellResult]) -> str:
    """A per-cell row table (the long-form view)."""
    lines = []
    header = (f"{'sims':>5s} {'vcf':>5s} {'fpu':>5s} {'reu':>4s} "
              f"{'pp':>3s} {'pvcf':>5s} {'n':>4s}  "
              f"{'BlackW/L/D':>14s}  {'WhiteW/L/D':>14s}  "
              f"{'Bwin':>6s} {'Wloss':>6s}  {'dist':>6s}  {'secs':>6s}  {'stop':>10s}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        stop = r.early_stop or ""
        if r.error:
            lines.append(
                f"{r.sims:>5d} {r.eval_vcf_nodes:>5d} "
                f"{r.fpu_reduction_c:>5.2f} {r.reuse_tree:>4d} "
                f"{r.proven_prop:>3d} {r.proven_vcf_leaf_nodes:>5d} "
                f"{r.n_games:>4d}  "
                f"{'ERROR':>14s}  {r.error}"
            )
            continue
        if stop == "skipped:target-hit":
            lines.append(
                f"{r.sims:>5d} {r.eval_vcf_nodes:>5d} "
                f"{r.fpu_reduction_c:>5.2f} {r.reuse_tree:>4d} "
                f"{r.proven_prop:>3d} {r.proven_vcf_leaf_nodes:>5d} "
                f"{r.n_games:>4d}  {'(skipped)':>14s}  "
                f"{'':>14s}  {'':>6s} {'':>6s}  {'':>6s}  {'':>6s}  {stop:>10s}"
            )
            continue
        lines.append(
            f"{r.sims:>5d} {r.eval_vcf_nodes:>5d} "
            f"{r.fpu_reduction_c:>5.2f} {r.reuse_tree:>4d} "
            f"{r.proven_prop:>3d} {r.proven_vcf_leaf_nodes:>5d} "
            f"{r.n_games:>4d}  "
            f"{r.black_w:>3d}/{r.black_l:>3d}/{r.black_d:>3d}  ".rjust(16)
            + f"{r.white_w:>3d}/{r.white_l:>3d}/{r.white_d:>3d}  ".rjust(16)
            + f"{r.black_win_rate*100:5.0f}% {r.white_loss_rate*100:5.0f}%  "
            f"{r.distance:6.3f}  {r.wall_secs:6.1f}  {stop:>10s}"
        )
    return "\n".join(lines)


def _varying_axes(grids: dict[str, list]) -> list[str]:
    """Return axis names whose grid has >1 distinct value."""
    return [a for a in AXIS_NAMES if len(set(grids[a])) > 1]


def format_distance_grid(results: list[CellResult],
                         sims_grid: list[int],
                         vcf_grid: list[int],
                         *,
                         fpu_c_grid: list[float] | None = None,
                         reuse_tree_grid: list[int] | None = None,
                         proven_prop_grid: list[int] | None = None,
                         proven_vcf_leaf_nodes_grid: list[int] | None = None) -> str:
    """A markdown distance table.

    When ≤2 axes vary, render the classic 2D markdown grid (rows = the
    first-varying axis, columns = the second-varying axis), pivoted by
    holding all other axes at their OFF value (a stable slice).

    When ≥3 axes vary, the 2D pivot would be ambiguous, so we render a
    long-form summary instead:
      - top-N cells by distance,
      - the "other axes held at: ..." line for any pivoted-out axes,
      - per-lever best/worst summary.

    The best (lowest-distance) successful cell always gets a ``*`` tag and a
    final "best cell" line.
    """
    grids = _grids_for_cells(
        sims_grid=sims_grid,
        vcf_nodes_grid=vcf_grid,
        fpu_c_grid=fpu_c_grid if fpu_c_grid is not None else [OFF_VALUES["fpu_reduction_c"]],
        reuse_tree_grid=reuse_tree_grid if reuse_tree_grid is not None else [OFF_VALUES["reuse_tree"]],
        proven_prop_grid=proven_prop_grid if proven_prop_grid is not None else [OFF_VALUES["proven_prop"]],
        proven_vcf_leaf_nodes_grid=(
            proven_vcf_leaf_nodes_grid
            if proven_vcf_leaf_nodes_grid is not None
            else [OFF_VALUES["proven_vcf_leaf_nodes"]]
        ),
    )

    successful = [r for r in results if r.error is None and r.early_stop != "skipped:target-hit"]
    best = min(successful, key=lambda r: r.distance) if successful else None
    best_axes = (
        (best.sims, best.eval_vcf_nodes, best.fpu_reduction_c,
         best.reuse_tree, best.proven_prop, best.proven_vcf_leaf_nodes)
        if best else None
    )

    varying = _varying_axes(grids)
    lines: list[str] = []
    lines.append(
        "distance-to-100% grid (lower is better; "
        "0.0 = win-all-black/lose-none-white vs the baseline)"
    )
    lines.append("")

    if len(varying) <= 2:
        # Classic 2D pivot. Pick the two varying axes (or pad with the legacy
        # sims/vcf axes for backward-compat when only the legacy axes vary).
        if len(varying) == 0:
            varying_to_use = ["sims", "eval_vcf_nodes"]
        elif len(varying) == 1:
            v = varying[0]
            other = "sims" if v != "sims" else "eval_vcf_nodes"
            varying_to_use = [v, other]
        else:
            varying_to_use = list(varying)
        row_axis, col_axis = varying_to_use[0], varying_to_use[1]
        row_vals = grids[row_axis]
        col_vals = grids[col_axis]

        # Slice: hold every other axis at its OFF value.
        held = {a: OFF_VALUES[a] for a in AXIS_NAMES if a not in (row_axis, col_axis)}
        slice_results: dict[tuple, CellResult] = {}
        for r in successful:
            if all(getattr(r, a) == v for a, v in held.items()):
                slice_results[(getattr(r, row_axis), getattr(r, col_axis))] = r

        col_w = 10
        header = (f"| {row_axis + ' \\\\ ' + col_axis:<{col_w}} | "
                  + " | ".join(_fmt_axis_value(col_axis, v, col_w) for v in col_vals)
                  + " |")
        sep = "|" + "-" * (col_w + 2) + ("|" + "-" * (col_w + 2)) * len(col_vals) + "|"
        lines.append(header)
        lines.append(sep)
        for rv in row_vals:
            cells_out = [_fmt_axis_value(row_axis, rv, col_w)]
            for cv in col_vals:
                r = slice_results.get((rv, cv))
                if r is None:
                    cell = "-"
                else:
                    is_best = (
                        best is not None
                        and getattr(r, "sims") == best.sims
                        and getattr(r, "eval_vcf_nodes") == best.eval_vcf_nodes
                        and getattr(r, "fpu_reduction_c") == best.fpu_reduction_c
                        and getattr(r, "reuse_tree") == best.reuse_tree
                        and getattr(r, "proven_prop") == best.proven_prop
                        and getattr(r, "proven_vcf_leaf_nodes") == best.proven_vcf_leaf_nodes
                    )
                    tag = "*" if is_best else " "
                    cell = f"{r.distance:6.3f}{tag}"
                cells_out.append(f"{cell:>{col_w}}")
            lines.append("| " + " | ".join(cells_out) + " |")
        if held and any(len(set(grids[a])) > 1 for a in held):
            held_str = ", ".join(
                f"{a}={_axis_repr(a, held[a])}" for a in sorted(held)
            )
            lines.append("")
            lines.append(f"other axes held at OFF: {held_str}")
    else:
        # ≥3 axes vary — long-form summary.
        lines.append(f"(≥3 axes vary: {', '.join(varying)} — long-form view)")
        lines.append("")
        top_n = min(10, len(successful))
        ranked = sorted(successful, key=lambda r: r.distance)
        lines.append(f"top {top_n} cells by distance:")
        for i, r in enumerate(ranked[:top_n]):
            tag = " *" if (best is not None and r is best) else "  "
            lines.append(
                f"{tag}{i+1:2d}. dist={r.distance:6.3f}  "
                f"sims={r.sims} vcf={r.eval_vcf_nodes} "
                f"fpu={r.fpu_reduction_c:g} reuse={r.reuse_tree} "
                f"pprop={r.proven_prop} pvcf={r.proven_vcf_leaf_nodes}  "
                f"L4-Bwin={r.black_win_rate*100:.0f}%"
            )

    if best is not None:
        assert best_axes is not None
        lines.append("")
        lines.append(
            f"best cell: sims={best_axes[0]} vcf={best_axes[1]} "
            f"fpu={best_axes[2]:g} reuse={best_axes[3]} "
            f"pprop={best_axes[4]} pvcf={best_axes[5]} "
            f"dist={best.distance:.3f} "
            f"L4-black-winrate={best.black_win_rate*100:.0f}%  (*)"
        )
    return "\n".join(lines)


def _fmt_axis_value(axis: str, v, width: int) -> str:
    if axis == "fpu_reduction_c":
        return f"{v:>{width}.2f}"
    return f"{v:>{width}}"


def _axis_repr(axis: str, v) -> str:
    if axis == "fpu_reduction_c":
        return f"{v:g}"
    return f"{v}"


def write_jsonl(results: list[CellResult], path: Path, meta: dict) -> None:
    """One JSON object per line. First line is a meta header
    ({"meta": {...}}); subsequent lines are CellResult dicts.

    LEGACY (pre-derby-cec) batch writer — kept for callers that hold full
    results in memory and want to dump them as a single atomic file. The
    driver's ``main()`` no longer uses this; it streams per-cell via
    ``append_meta_header`` + ``append_cell_row`` so a mid-run crash leaves
    completed cells safe on disk.
    """
    with open(path, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


# ---------------------------------------------------------------------------
# Streaming output + --resume helpers (derby-cec)


def cell_key(cell: Cell) -> tuple:
    """Stable hashable key for a cell — tuple of all axis values in
    canonical order. Used by ``--resume`` to skip completed cells."""
    return tuple(cell.axis_value(a) for a in AXIS_NAMES)


def result_key(row: dict) -> tuple:
    """Build a cell-key tuple from a parsed JSONL row dict. Mirrors
    ``cell_key`` but reads from the row's recorded axis values.

    Skipped cells (target-hit) still count as 'completed' for resume
    purposes — re-running shouldn't re-evaluate a cell the prior run
    decided not to evaluate.
    """
    return tuple(row[a] for a in AXIS_NAMES)


def parse_completed_keys(output_path: Path) -> tuple[set, int]:
    """Read an existing --output file and return (set of cell-keys already
    on disk, count of those keys).

    Robust to a partial last line (a mid-write crash). Skips the meta header
    row. Missing file → empty set.
    """
    if not output_path.exists():
        return set(), 0
    keys: set = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Partial last line from a SIGKILL mid-write — ignore.
                continue
            if "meta" in row and len(row) == 1:
                continue
            try:
                keys.add(result_key(row))
            except KeyError:
                # Older-format row missing an axis — best-effort skip.
                continue
    return keys, len(keys)


def append_meta_header(output_path: Path, meta: dict) -> None:
    """Append a meta-header row to the output file (append-mode + flush).

    Idempotent-ish: callers normally only invoke this when the file does
    NOT already contain a meta header. On --resume we DO NOT re-write the
    meta header (the original run's meta is the canonical one).
    """
    with open(output_path, "a") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass  # tmpfs / read-only mount during tests — best-effort.


def append_cell_row(output_path: Path, cr: CellResult) -> None:
    """Append a single cell-result row and flush immediately. The point of
    streaming: a SIGKILL after this returns leaves the row safe on disk."""
    with open(output_path, "a") as f:
        f.write(json.dumps(asdict(cr)) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Honest timing self-report (derby-cec)


# Per-100-sims baseline wall (seconds). The 100g re-eval observed ~14 min /
# cell at sims=100, n_games=100, single-worker CPU — so ~14*60/100 = 8.4
# seconds-per-game-at-sims=100. Use 9 as a round number; the actual/estimate
# ratio reported at end will calibrate it.
SECS_PER_GAME_AT_SIMS_100_BASELINE = 9.0


def estimate_cell_wall_secs(cell: Cell, n_games: int, n_workers: int = 1) -> float:
    """Rough per-cell wall-clock estimate, in seconds.

    Scaling factors (best-effort; the actual/estimate ratio printed at end
    is the calibration loop):
      - linear in n_games / n_workers
      - linear in sims / 100
      - ~+15% per extra non-OFF lever (vcf nodes, fpu, reuse-tree, proven-*)
    """
    per_game = SECS_PER_GAME_AT_SIMS_100_BASELINE * (cell.sims / 100.0)
    levers = cell.num_active_levers()
    # Count the sims-axis lever as part of the per-game cost above, so the
    # lever-overhead bump applies only to NON-sims active levers.
    non_sims_levers = max(0, levers - (1 if cell.sims != OFF_VALUES["sims"] else 0))
    per_game *= (1.0 + 0.15 * non_sims_levers)
    total = per_game * n_games / max(1, n_workers)
    return total


def estimate_total_wall_secs(cells: list[Cell], n_games: int,
                             n_workers: int = 1) -> float:
    return sum(estimate_cell_wall_secs(c, n_games, n_workers) for c in cells)


def format_duration(secs: float) -> str:
    """Concise duration: '12.3s', '4.5min', '2.7h'."""
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        return f"{secs/60:.1f}min"
    return f"{secs/3600:.2f}h"


# ---------------------------------------------------------------------------
# CLI


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="eval-config sweep driver vs lookahead:depth=4 "
                    "(RESUME PLAYBOOK step 1; H2 search-depth probe). "
                    "Sweeps the full sims × vcf × fpu × reuse-tree × proven-prop "
                    "matrix with cheap-first ordering and optional early-stop.",
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

    # Legacy axes (derby-5xs).
    p.add_argument("--sims-grid", type=str, default="100,200,400,800",
                   help="Comma-separated MCTS sims values.")
    p.add_argument("--vcf-nodes-grid", type=str, default="0,200,800",
                   help="Comma-separated eval-vcf-nodes values "
                        "(0 disables the overlay).")

    # New axes (derby-u8d). Per derby-cec: grids RESPECT what you pass —
    # a single value runs exactly one cell on that axis. To ablate a
    # non-zero value against OFF, pass an explicit two-value grid
    # (e.g. ``--fpu-c-grid 0,0.45``).
    p.add_argument("--fpu-c-grid", type=str, default="0.0",
                   help="Comma-separated FPU-reduction c values (derby-3w0). "
                        "Single value = one cell at that value (no auto-injection "
                        "of OFF). To ablate against 0, pass '0,VAL'. "
                        "Default '0.0' (single = OFF).")
    p.add_argument("--reuse-tree-grid", type=str, default="0",
                   help="Comma-separated 0/1 values for the cross-ply tree-reuse "
                        "lever (derby-jmi). Single value = one cell at that "
                        "value (no auto-injection of OFF). To ablate against 0, "
                        "pass '0,1'. Default '0' (single = OFF).")
    p.add_argument("--proven-prop-grid", type=str, default="0",
                   help="Comma-separated 0/1 values for the proven-win/loss "
                        "propagation lever (derby-b3n). Single value = one cell at "
                        "that value (no auto-injection of OFF). To ablate against "
                        "0, pass '0,1'. Default '0' (single = OFF).")
    p.add_argument("--proven-vcf-leaf-nodes-grid", type=str, default="0",
                   help="Comma-separated leaf-VCF bounded-nodes values "
                        "(derby-b3n; requires --proven-prop=1 to take effect). "
                        "Single value = one cell at that value (no auto-injection "
                        "of OFF). To ablate against 0, pass '0,VAL'. "
                        "Default '0' (single = OFF).")

    # Ordering + early-stop knobs.
    p.add_argument("--order", type=str, default="cheap-first",
                   choices=["cheap-first", "full", "custom-list"],
                   help="Cell iteration order. 'cheap-first' (default) puts the "
                        "all-OFF baseline first, then single-lever isolations, "
                        "then pair-combos, etc. — so early-stop drops the "
                        "expensive cells first. 'full' = canonical Cartesian "
                        "(legacy derby-5xs order). 'custom-list' is reserved.")
    p.add_argument("--early-stop-on-target", type=float, default=None,
                   help="If set, stop the sweep as soon as a cell's distance "
                        "is ≤ this threshold (e.g. 0.1).")
    p.add_argument("--worse-than-baseline-margin", type=float, default=None,
                   help="If set, mark any cell whose distance is ≥ "
                        "(baseline_distance + margin) with early_stop='worse'. "
                        "Informational only — the sweep continues. Default off.")

    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path (append mode + per-cell flush). "
                        "Default probe_100pct_<UTC-timestamp>.jsonl in CWD.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   default=True,
                   help="Disable --resume. By default, if --output already "
                        "exists the driver parses completed cell-keys and skips "
                        "them; --no-resume re-evaluates every cell (still "
                        "appends to the file).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--device", type=str, default="cpu",
                   help="Eval device. Default cpu so we don't fight a live "
                        "trainer for MPS.")
    p.add_argument("--n-workers", type=int, default=1,
                   help="If >1, parallelise the n_games per cell via "
                        "play_match_parallel.")
    p.add_argument("--i-know-derby-is-running", action="store_true",
                   help="Acknowledge a live delo_derby.py and proceed anyway.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the grid and exit without running any eval.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None,
         *, eval_fn=None,
         derby_check=is_derby_running) -> int:
    args = parse_args(argv)
    sims_grid = _parse_int_list(args.sims_grid)
    vcf_grid = _parse_int_list(args.vcf_nodes_grid)
    fpu_c_grid = _parse_float_list(args.fpu_c_grid)
    reuse_tree_grid = _parse_int_list(args.reuse_tree_grid)
    proven_prop_grid = _parse_int_list(args.proven_prop_grid)
    proven_vcf_leaf_grid = _parse_int_list(args.proven_vcf_leaf_nodes_grid)
    if not sims_grid or not vcf_grid:
        print("error: empty --sims-grid or --vcf-nodes-grid", file=sys.stderr)
        return 2

    cells = enumerate_cells(
        sims_grid=sims_grid,
        vcf_nodes_grid=vcf_grid,
        fpu_c_grid=fpu_c_grid,
        reuse_tree_grid=reuse_tree_grid,
        proven_prop_grid=proven_prop_grid,
        proven_vcf_leaf_nodes_grid=proven_vcf_leaf_grid,
    )
    cells = order_cells(cells, args.order)

    if args.dry_run:
        print(f"[dry-run] would evaluate {len(cells)} cells against "
              f"{args.baseline} with {args.games_per_cell} games/cell "
              f"(order={args.order})")
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
    out_path = Path(args.output).resolve()

    # --resume: parse the existing --output (if any) and build the skip set.
    # If --no-resume was passed, skip the pre-pass entirely.
    skip_keys: set = set()
    file_exists = out_path.exists()
    if args.resume and file_exists:
        skip_keys, n_done = parse_completed_keys(out_path)
        print(f"[probe] resuming: {n_done}/{len(cells)} cells already "
              f"complete in {out_path}")
    elif file_exists and not args.resume:
        print(f"[probe] --no-resume: ignoring prior cells in {out_path} "
              f"(still appending)")

    # Honest timing self-report: upfront plan + ETA, plus absolute paths so
    # the operator can confirm cwd/checkpoint are what they think.
    remaining_cells = [c for c in cells if cell_key(c) not in skip_keys]
    est_total_secs = estimate_total_wall_secs(
        remaining_cells, args.games_per_cell, args.n_workers,
    )
    est_per_cell_secs = (
        est_total_secs / len(remaining_cells) if remaining_cells else 0.0
    )

    print(f"[probe] checkpoint={Path(args.checkpoint).resolve()}")
    print(f"[probe] baseline={args.baseline}  games/cell={args.games_per_cell}")
    print(f"[probe] grid: sims={sims_grid}  vcf_nodes={vcf_grid}  "
          f"fpu_c={fpu_c_grid}  reuse_tree={reuse_tree_grid}  "
          f"proven_prop={proven_prop_grid}  "
          f"proven_vcf_leaf_nodes={proven_vcf_leaf_grid}  "
          f"({len(cells)} cells, order={args.order})")
    if args.early_stop_on_target is not None:
        print(f"[probe] early-stop on target distance ≤ "
              f"{args.early_stop_on_target}")
    if args.worse_than_baseline_margin is not None:
        print(f"[probe] mark cells worse-than-baseline by ≥ "
              f"{args.worse_than_baseline_margin}")
    print(f"[probe] cwd={Path.cwd()}")
    print(f"[probe] output={out_path}")
    print(f"[probe] plan: {len(remaining_cells)} cells to run "
          f"(of {len(cells)} total), estimated "
          f"~{format_duration(est_per_cell_secs)}/cell, "
          f"total ~{format_duration(est_total_secs)} "
          f"(scales linearly with sims, n_games, n_workers; "
          f"calibration in ratio printed at end).")

    meta = {
        "checkpoint": args.checkpoint,
        "baseline": args.baseline,
        "games_per_cell": args.games_per_cell,
        "sims_grid": sims_grid,
        "vcf_nodes_grid": vcf_grid,
        "fpu_c_grid": fpu_c_grid,
        "reuse_tree_grid": reuse_tree_grid,
        "proven_prop_grid": proven_prop_grid,
        "proven_vcf_leaf_nodes_grid": proven_vcf_leaf_grid,
        "order": args.order,
        "early_stop_on_target": args.early_stop_on_target,
        "worse_than_baseline_margin": args.worse_than_baseline_margin,
        "seed": args.seed,
        "c_puct": args.c_puct,
        "device": args.device,
        "n_workers": args.n_workers,
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resume": args.resume,
    }

    # Streaming output: write the meta header ONCE (only when the file does
    # NOT already have one). Then stream cells via on_cell.
    if not file_exists:
        append_meta_header(out_path, meta)

    def stream_row(cr: CellResult) -> None:
        append_cell_row(out_path, cr)

    t_run_start = time.perf_counter()
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
        early_stop_on_target=args.early_stop_on_target,
        worse_than_baseline_margin=args.worse_than_baseline_margin,
        on_cell=stream_row,
        skip_keys=skip_keys,
    )
    actual_secs = time.perf_counter() - t_run_start

    print()
    print(format_per_cell_table(results))
    print()
    # The grid renderer needs the full set of results (skipped cells from a
    # prior run live in the file but aren't in ``results``); for the table we
    # only render what this run produced. Callers that want a unified
    # post-resume view should re-parse the file.
    print(format_distance_grid(
        results, sims_grid, vcf_grid,
        fpu_c_grid=fpu_c_grid,
        reuse_tree_grid=reuse_tree_grid,
        proven_prop_grid=proven_prop_grid,
        proven_vcf_leaf_nodes_grid=proven_vcf_leaf_grid,
    ))
    print()
    print(f"[probe] wrote {len(results)} cells to {out_path}")

    # Honest timing self-report — end-of-run actual/estimate ratio.
    ratio = (actual_secs / est_total_secs) if est_total_secs > 0 else 0.0
    print(
        f"[probe] actual/estimate ratio: {len(results)} cells in "
        f"{format_duration(actual_secs)} "
        f"({format_duration(est_total_secs)} estimated; "
        f"ratio {ratio:.2f}x)"
    )
    if ratio > 2.0:
        print(
            f"[probe] WARNING: actual/estimate ratio {ratio:.2f}x > 2.0x — "
            f"suspect contention or a missed cost factor; consider filing a "
            f"perf-meta bead to update SECS_PER_GAME_AT_SIMS_100_BASELINE "
            f"or the scaling factors in estimate_cell_wall_secs."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
