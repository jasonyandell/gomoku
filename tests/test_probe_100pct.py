"""Tests for scripts/probe_100pct.py — the eval-config sweep driver.

Originally derby-5xs (sims × eval-vcf only); extended in derby-u8d to cover
three more axes (fpu_reduction_c, reuse_tree, proven_prop +
proven_vcf_leaf_nodes), cheap-first ordering, and early-stop on target.

CPU-only, NO GPU, NO actual eval/model loads. The single ``run_cell_eval``
seam is mocked with a deterministic stub that returns fixed color-split
tallies per cell.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable as a regular package.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import probe_100pct as probe  # noqa: E402
import report_100pct  # noqa: E402  — used for the "imports, doesn't duplicate" assertion


# ---------------------------------------------------------------------------
# Cell enumeration


def test_enumerate_cells_is_cartesian_product_sorted_cheapest_first():
    """Legacy (sims × vcf only) shape with the three new axes defaulted to a
    single OFF value. Reproduces the derby-5xs 4×3=12-cell shape, in canonical
    (full) order — outer ascending sims, inner ascending vcf."""
    cells = probe.enumerate_cells([400, 100, 200, 800], [800, 0, 200])
    # 4 sims × 3 vcf × 1×1×1×1 = 12 cells.
    assert len(cells) == 12
    sims_order = [c.sims for c in cells]
    assert sims_order == sorted(sims_order)
    # Inner per-sims group: ascending vcf, all OFF on the new axes.
    for i in range(0, len(cells), 3):
        group = cells[i:i + 3]
        assert [c.eval_vcf_nodes for c in group] == [0, 200, 800]
        for c in group:
            assert c.fpu_reduction_c == 0.0
            assert c.reuse_tree == 0
            assert c.proven_prop == 0
            assert c.proven_vcf_leaf_nodes == 0
    # Cheapest cell first.
    assert cells[0] == probe.Cell(sims=100, eval_vcf_nodes=0)
    # Most expensive cell last.
    assert cells[-1] == probe.Cell(sims=800, eval_vcf_nodes=800)


def test_default_cli_args_produce_4_by_3_grid():
    """Bead-spec acceptance (b): defaults still yield 12 cells (backward-
    compatible with derby-5xs)."""
    args = probe.parse_args(["--checkpoint", "irrelevant.pt"])
    sims = probe._parse_int_list(args.sims_grid)
    vcf = probe._parse_int_list(args.vcf_nodes_grid)
    fpu_c = probe._parse_float_list(args.fpu_c_grid)
    reuse = probe._parse_int_list(args.reuse_tree_grid)
    pprop = probe._parse_int_list(args.proven_prop_grid)
    pvcf = probe._parse_int_list(args.proven_vcf_leaf_nodes_grid)
    assert sims == [100, 200, 400, 800]
    assert vcf == [0, 200, 800]
    # New axes default to a single OFF value.
    assert fpu_c == [0.0]
    assert reuse == [0]
    assert pprop == [0]
    assert pvcf == [0]
    cells = probe.enumerate_cells(
        sims_grid=sims, vcf_nodes_grid=vcf,
        fpu_c_grid=fpu_c, reuse_tree_grid=reuse,
        proven_prop_grid=pprop, proven_vcf_leaf_nodes_grid=pvcf,
    )
    assert len(cells) == 12


def test_default_cli_args_use_cheap_first_ordering():
    """The new default order is cheap-first; the all-OFF baseline (sims=100,
    vcf=0, fpu=0, reuse=0, pprop=0, pvcf=0) runs first. The legacy axes still
    contribute active-lever count (sims!=100 or vcf!=0 are each counted as a
    flipped lever), so cheap-first reorders the legacy 12-cell sweep into:
        cell 0: all-OFF baseline
        next:    single-lever isolations (one axis flipped)
        then:    pairs, etc.
    """
    args = probe.parse_args(["--checkpoint", "irrelevant.pt"])
    assert args.order == "cheap-first"
    sims = probe._parse_int_list(args.sims_grid)
    vcf = probe._parse_int_list(args.vcf_nodes_grid)
    cells = probe.enumerate_cells(sims, vcf)
    cheap = probe.order_cells_cheap_first(cells)
    # The all-OFF baseline is cell 0.
    assert cheap[0] == probe.Cell(sims=100, eval_vcf_nodes=0)
    assert cheap[0].num_active_levers() == 0
    # The active-lever count sequence is monotone non-decreasing.
    active_seq = [c.num_active_levers() for c in cheap]
    assert active_seq == sorted(active_seq)
    # Single-lever isolations follow immediately: sims∈{200,400,800} with
    # vcf=0, plus vcf∈{200,800} with sims=100. That's 5 single-lever cells.
    single = [c for c in cheap if c.num_active_levers() == 1]
    assert len(single) == 5


# ---------------------------------------------------------------------------
# Bead acceptance (a) — Cartesian product of all five axes


def test_extended_grid_is_cartesian_product_of_all_axes():
    """Bead-spec acceptance (a): the extended grid is the Cartesian product
    of all 5 axes (or 6 counting proven-vcf-leaf-nodes as a separate axis).
    With 2 sims × 2 vcf × 2 fpu × 2 reuse × 2 pprop × 2 pvcf = 64 cells."""
    cells = probe.enumerate_cells(
        sims_grid=[100, 200],
        vcf_nodes_grid=[0, 200],
        fpu_c_grid=[0.0, 0.20],
        reuse_tree_grid=[0, 1],
        proven_prop_grid=[0, 1],
        proven_vcf_leaf_nodes_grid=[0, 200],
    )
    assert len(cells) == 64
    # Every cell is unique.
    assert len({(c.sims, c.eval_vcf_nodes, c.fpu_reduction_c,
                 c.reuse_tree, c.proven_prop, c.proven_vcf_leaf_nodes)
                for c in cells}) == 64
    # All 2^6 = 64 combos are represented.
    by_axis_values = {
        "sims": {c.sims for c in cells},
        "vcf": {c.eval_vcf_nodes for c in cells},
        "fpu": {c.fpu_reduction_c for c in cells},
        "reuse": {c.reuse_tree for c in cells},
        "pprop": {c.proven_prop for c in cells},
        "pvcf": {c.proven_vcf_leaf_nodes for c in cells},
    }
    assert by_axis_values["sims"] == {100, 200}
    assert by_axis_values["vcf"] == {0, 200}
    assert by_axis_values["fpu"] == {0.0, 0.20}
    assert by_axis_values["reuse"] == {0, 1}
    assert by_axis_values["pprop"] == {0, 1}
    assert by_axis_values["pvcf"] == {0, 200}


def test_normalize_grid_injects_off_if_missing():
    """The normalizer guarantees OFF is present in each grid. Cheap-first
    relies on the all-OFF baseline existing in the Cartesian product."""
    # Caller forgets to include 0 in vcf grid → driver injects it silently.
    cells = probe.enumerate_cells(
        sims_grid=[100],
        vcf_nodes_grid=[200, 800],
    )
    vcfs = sorted({c.eval_vcf_nodes for c in cells})
    assert 0 in vcfs


# ---------------------------------------------------------------------------
# Bead acceptance (c) — cheap-first ordering


def test_cheap_first_yields_all_off_baseline_as_cell_zero():
    """Bead-spec acceptance (c): cheap-first order yields the all-OFF baseline
    as cell 0, then individual-lever isolations next."""
    cells = probe.enumerate_cells(
        sims_grid=[100, 200],
        vcf_nodes_grid=[0, 200],
        fpu_c_grid=[0.0, 0.20],
        reuse_tree_grid=[0, 1],
        proven_prop_grid=[0, 1],
        proven_vcf_leaf_nodes_grid=[0, 200],
    )
    cheap = probe.order_cells_cheap_first(cells)
    # Cell 0 is the all-OFF baseline.
    baseline = cheap[0]
    assert baseline == probe.Cell(
        sims=100, eval_vcf_nodes=0,
        fpu_reduction_c=0.0, reuse_tree=0,
        proven_prop=0, proven_vcf_leaf_nodes=0,
    )
    assert baseline.num_active_levers() == 0
    # The next cells are single-lever isolations (active_levers == 1).
    # Count how many cells share num_active_levers=1 in this 6-axis 2-value
    # matrix — exactly 6 (one per axis flipped).
    single_lever = [c for c in cheap if c.num_active_levers() == 1]
    assert len(single_lever) == 6
    # Those single-lever cells come BEFORE any pair-combo (active=2) cell.
    first_pair_idx = next(i for i, c in enumerate(cheap)
                          if c.num_active_levers() == 2)
    last_single_idx = max(i for i, c in enumerate(cheap)
                          if c.num_active_levers() == 1)
    assert last_single_idx < first_pair_idx
    # Active-lever counts are monotone non-decreasing.
    active_seq = [c.num_active_levers() for c in cheap]
    assert active_seq == sorted(active_seq)


def test_full_order_preserves_canonical_cartesian():
    """``--order full`` is the legacy Cartesian enumeration — no cheap-first
    reshuffling."""
    cells = probe.enumerate_cells(
        sims_grid=[100, 200],
        vcf_nodes_grid=[0, 200],
        fpu_c_grid=[0.0, 0.20],
    )
    full = probe.order_cells(cells, "full")
    assert full == cells


# ---------------------------------------------------------------------------
# Distance-to-100% — imports report_100pct's formula, doesn't duplicate it


def test_cell_distance_uses_report_100pct_formula_perfect_cell():
    dist, bw, wl = probe.cell_distance_to_100(
        black_w=20, black_l=0, black_d=0,
        white_w=10, white_l=0, white_d=10,
        baseline_label="lookahead:depth=4",
    )
    assert dist == pytest.approx(0.0)
    assert bw == pytest.approx(1.0)
    assert wl == pytest.approx(0.0)


def test_cell_distance_uses_report_100pct_formula_known_value():
    dist, bw, wl = probe.cell_distance_to_100(
        black_w=10, black_l=0, black_d=10,
        white_w=18, white_l=2, white_d=0,
        baseline_label="lookahead:depth=4",
    )
    assert bw == pytest.approx(0.5)
    assert wl == pytest.approx(0.1)
    assert dist == pytest.approx(0.6)


def test_cell_distance_delegates_to_report_100pct(monkeypatch):
    calls: list[dict] = []

    def fake_score(agg):
        calls.append(agg)
        return {b: (0.0, 0.0, 0, 0) for b in report_100pct.BASELINES}, 1234.0

    monkeypatch.setattr(probe, "report_100pct_score", fake_score)
    dist, _bw, _wl = probe.cell_distance_to_100(
        black_w=1, black_l=0, black_d=0, white_w=0, white_l=1, white_d=0,
        baseline_label="lookahead:depth=4",
    )
    assert dist == 1234.0
    assert calls, "expected probe to call report_100pct_score"


# ---------------------------------------------------------------------------
# CLI parser exposes each new axis


def test_argparser_exposes_all_new_axes():
    args = probe.parse_args([
        "--checkpoint", "x.pt",
        "--fpu-c-grid", "0.0,0.45",
        "--reuse-tree-grid", "0,1",
        "--proven-prop-grid", "0,1",
        "--proven-vcf-leaf-nodes-grid", "0,200,400",
        "--order", "cheap-first",
        "--early-stop-on-target", "0.1",
    ])
    assert args.fpu_c_grid == "0.0,0.45"
    assert args.reuse_tree_grid == "0,1"
    assert args.proven_prop_grid == "0,1"
    assert args.proven_vcf_leaf_nodes_grid == "0,200,400"
    assert args.order == "cheap-first"
    assert args.early_stop_on_target == 0.1


def test_argparser_order_choices():
    """--order accepts cheap-first, full, custom-list."""
    for opt in ("cheap-first", "full", "custom-list"):
        ns = probe.parse_args(["--checkpoint", "x.pt", "--order", opt])
        assert ns.order == opt


# ---------------------------------------------------------------------------
# Driver: iterates grid, calls eval with the right kwargs per cell


def _stub_eval_factory(distance_for_cell=None):
    """Build a stub eval_fn that records every kwargs invocation and returns
    a deterministic tally derived from the cell's axes.

    ``distance_for_cell`` is an optional callable
    ``(sims, eval_vcf_nodes, fpu_reduction_c, reuse_tree, proven_prop,
       proven_vcf_leaf_nodes) -> float in [0, 2]`` — if supplied, the stub
    tunes black_w / black_d so the cell's distance lands close to the
    requested value (used by early-stop tests).
    """
    calls: list[dict] = []

    def stub(*, checkpoint, baseline, n_games, sims, eval_vcf_nodes,
             fpu_reduction_c=0.0, reuse_tree=0, proven_prop=0,
             proven_vcf_leaf_nodes=0,
             c_puct, device, seed, n_workers):
        calls.append(dict(
            checkpoint=checkpoint, baseline=baseline, n_games=n_games,
            sims=sims, eval_vcf_nodes=eval_vcf_nodes,
            fpu_reduction_c=fpu_reduction_c, reuse_tree=reuse_tree,
            proven_prop=proven_prop,
            proven_vcf_leaf_nodes=proven_vcf_leaf_nodes,
            c_puct=c_puct, device=device, seed=seed, n_workers=n_workers,
        ))
        half = n_games // 2
        if distance_for_cell is not None:
            target = distance_for_cell(sims, eval_vcf_nodes, fpu_reduction_c,
                                       reuse_tree, proven_prop,
                                       proven_vcf_leaf_nodes)
            # distance = (1 - bw) + wl. We'll fix wl=0 (no white losses) and
            # tune bw via black_w/half. Required: bw = max(0, 1 - target).
            bw_target = max(0.0, min(1.0, 1.0 - target))
            black_w = int(round(bw_target * half))
            black_d = half - black_w
            return {
                "n_games": n_games,
                "wins": black_w,
                "losses": 0,
                "draws": black_d + half,
                "black_w": black_w, "black_l": 0, "black_d": black_d,
                "white_w": 0, "white_l": 0, "white_d": half,
            }
        # Default deterministic tally (the derby-5xs behaviour, kept for
        # backward compat with the legacy tests below).
        bw_bonus = min(half, sims // 200 + eval_vcf_nodes // 200)
        black_w = min(half, bw_bonus)
        black_d = half - black_w
        return {
            "n_games": n_games,
            "wins": black_w,
            "losses": 0,
            "draws": black_d + half,
            "black_w": black_w, "black_l": 0, "black_d": black_d,
            "white_w": 0, "white_l": 0, "white_d": half,
        }

    return stub, calls


def test_run_probe_iterates_full_grid_and_passes_per_cell_kwargs():
    cells = probe.enumerate_cells([100, 200], [0, 200])
    stub, calls = _stub_eval_factory()
    results = probe.run_probe(
        checkpoint="fake.pt", baseline="lookahead:depth=4",
        cells=cells, n_games=40, seed=0, c_puct=1.5,
        device="cpu", n_workers=1, eval_fn=stub,
    )
    assert len(calls) == len(cells) == 4
    for cell, call in zip(cells, calls):
        assert call["sims"] == cell.sims
        assert call["eval_vcf_nodes"] == cell.eval_vcf_nodes
        # New axes threaded through with their cell value (all OFF here).
        assert call["fpu_reduction_c"] == cell.fpu_reduction_c
        assert call["reuse_tree"] == cell.reuse_tree
        assert call["proven_prop"] == cell.proven_prop
        assert call["proven_vcf_leaf_nodes"] == cell.proven_vcf_leaf_nodes
        assert call["n_games"] == 40
        assert call["baseline"] == "lookahead:depth=4"
        assert call["checkpoint"] == "fake.pt"
    for r in results:
        assert r.error is None
        assert r.n_games == 40
        assert r.black_w + r.black_l + r.black_d == 20
        assert r.white_w + r.white_l + r.white_d == 20
        expected_dist, _bw, _wl = probe.cell_distance_to_100(
            black_w=r.black_w, black_l=r.black_l, black_d=r.black_d,
            white_w=r.white_w, white_l=r.white_l, white_d=r.white_d,
            baseline_label=r.baseline,
        )
        assert r.distance == pytest.approx(expected_dist)


def test_run_probe_threads_new_axes_per_cell():
    """Each cell's new-axis value reaches the eval seam unchanged."""
    cells = probe.enumerate_cells(
        sims_grid=[100], vcf_nodes_grid=[0],
        fpu_c_grid=[0.0, 0.45],
        reuse_tree_grid=[0, 1],
        proven_prop_grid=[0, 1],
        proven_vcf_leaf_nodes_grid=[0, 200],
    )
    stub, calls = _stub_eval_factory()
    probe.run_probe(
        checkpoint="fake.pt", baseline="lookahead:depth=4",
        cells=cells, n_games=10, seed=0, c_puct=1.5,
        device="cpu", n_workers=1, eval_fn=stub,
    )
    assert len(calls) == len(cells)
    for cell, call in zip(cells, calls):
        assert call["fpu_reduction_c"] == cell.fpu_reduction_c
        assert call["reuse_tree"] == cell.reuse_tree
        assert call["proven_prop"] == cell.proven_prop
        assert call["proven_vcf_leaf_nodes"] == cell.proven_vcf_leaf_nodes


# ---------------------------------------------------------------------------
# Bead acceptance (d) — --early-stop-on-target stops the sweep


def test_early_stop_on_target_stops_sweep_at_first_hit():
    """Bead-spec acceptance (d): when a cell's distance is ≤ the threshold,
    the sweep stops; remaining cells are recorded as 'skipped:target-hit'."""
    # Build a 4-cell grid; tune the stub so cell index 1 hits the target,
    # rest would be worse.
    cells = [
        probe.Cell(sims=100, eval_vcf_nodes=0),
        probe.Cell(sims=200, eval_vcf_nodes=0),
        probe.Cell(sims=400, eval_vcf_nodes=0),
        probe.Cell(sims=800, eval_vcf_nodes=0),
    ]
    # Distance progression: 0.50, 0.05 (hit), 0.30, 0.40.
    def dist_for(sims, *_):
        return {100: 0.50, 200: 0.05, 400: 0.30, 800: 0.40}[sims]

    stub, calls = _stub_eval_factory(distance_for_cell=dist_for)
    results = probe.run_probe(
        checkpoint="x.pt", baseline="lookahead:depth=4",
        cells=cells, n_games=40, seed=0, c_puct=1.5,
        device="cpu", n_workers=1, eval_fn=stub,
        early_stop_on_target=0.1,
    )
    # Only 2 cells were actually evaluated (the baseline + the target-hit).
    assert len(calls) == 2
    assert calls[0]["sims"] == 100
    assert calls[1]["sims"] == 200
    # Result list still has one row per requested cell, but cells 2/3 are
    # marked 'skipped:target-hit'.
    assert len(results) == 4
    assert results[0].early_stop is None
    assert results[1].early_stop == "target-hit"
    assert results[2].early_stop == "skipped:target-hit"
    assert results[3].early_stop == "skipped:target-hit"
    # The target-hit cell carries its real distance.
    assert results[1].distance == pytest.approx(0.05, abs=0.05)


def test_early_stop_off_runs_full_sweep():
    """Default early_stop_on_target=None → all cells run."""
    cells = probe.enumerate_cells([100, 200], [0, 200])
    stub, calls = _stub_eval_factory()
    results = probe.run_probe(
        checkpoint="x.pt", baseline="lookahead:depth=4",
        cells=cells, n_games=10, seed=0, c_puct=1.5,
        device="cpu", n_workers=1, eval_fn=stub,
    )
    assert len(calls) == 4
    assert all(r.early_stop is None for r in results)


def test_worse_than_baseline_margin_flags_clearly_worse_cells():
    """The 'clearly worse' early-stop knob flags cells whose distance is
    ≥ baseline_distance + margin. Sweep continues (informational only)."""
    cells = [
        probe.Cell(sims=100, eval_vcf_nodes=0),  # baseline, dist=0.40
        probe.Cell(sims=100, eval_vcf_nodes=200),  # dist=0.80 (worse)
        probe.Cell(sims=200, eval_vcf_nodes=0),  # dist=0.30 (better)
    ]
    def dist_for(sims, vcf, *_):
        return {(100, 0): 0.40, (100, 200): 0.80, (200, 0): 0.30}[(sims, vcf)]

    stub, calls = _stub_eval_factory(distance_for_cell=dist_for)
    results = probe.run_probe(
        checkpoint="x.pt", baseline="lookahead:depth=4",
        cells=cells, n_games=10, seed=0, c_puct=1.5,
        device="cpu", n_workers=1, eval_fn=stub,
        worse_than_baseline_margin=0.20,
    )
    # All cells run; second is flagged worse, third is not.
    assert len(calls) == 3
    assert results[0].early_stop is None  # baseline
    assert results[1].early_stop == "worse"
    assert results[2].early_stop is None


# ---------------------------------------------------------------------------
# JSONL output shape


def test_write_jsonl_meta_header_and_rows(tmp_path):
    cells = probe.enumerate_cells([100, 200], [0])
    stub, _calls = _stub_eval_factory()
    results = probe.run_probe(
        checkpoint="fake.pt", baseline="lookahead:depth=4",
        cells=cells, n_games=10, seed=0, c_puct=1.5,
        device="cpu", n_workers=1, eval_fn=stub,
    )
    out = tmp_path / "probe.jsonl"
    meta = {"checkpoint": "fake.pt", "sims_grid": [100, 200],
            "vcf_nodes_grid": [0], "baseline": "lookahead:depth=4"}
    probe.write_jsonl(results, out, meta)

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1 + len(results)
    head = json.loads(lines[0])
    assert "meta" in head
    assert head["meta"]["checkpoint"] == "fake.pt"
    row = json.loads(lines[1])
    expected_keys = {
        "sims", "eval_vcf_nodes",
        "fpu_reduction_c", "reuse_tree", "proven_prop", "proven_vcf_leaf_nodes",
        "n_games", "baseline",
        "black_w", "black_l", "black_d",
        "white_w", "white_l", "white_d",
        "wins", "losses", "draws",
        "distance", "black_win_rate", "white_loss_rate",
        "wall_secs", "error", "early_stop",
    }
    assert expected_keys.issubset(row.keys()), \
        f"missing keys: {expected_keys - row.keys()}"


def test_main_writes_jsonl_and_does_not_call_eval_in_dry_run(tmp_path, capsys):
    sentinel_called = {"n": 0}

    def explosive(**_):
        sentinel_called["n"] += 1
        raise AssertionError("eval should not run in --dry-run")

    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100,200",
        "--vcf-nodes-grid", "0,200",
        "--dry-run",
    ], eval_fn=explosive, derby_check=lambda: False)
    out = capsys.readouterr().out
    assert rc == 0
    assert sentinel_called["n"] == 0
    assert "dry-run" in out
    assert "would evaluate 4 cells" in out


def test_main_end_to_end_with_stub(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    stub, calls = _stub_eval_factory()
    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100,200",
        "--vcf-nodes-grid", "0,200",
        "--games-per-cell", "10",
        "--output", "out.jsonl",
        "--device", "cpu",
    ], eval_fn=stub, derby_check=lambda: False)
    assert rc == 0
    assert len(calls) == 4
    out_path = tmp_path / "out.jsonl"
    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 5
    captured = capsys.readouterr().out
    assert "distance-to-100% grid" in captured
    assert "best cell" in captured


def test_main_end_to_end_with_extended_axes(tmp_path, monkeypatch, capsys):
    """End-to-end with the new axes activated; each cell's new-axis value
    reaches the stub."""
    monkeypatch.chdir(tmp_path)
    stub, calls = _stub_eval_factory()
    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100",
        "--vcf-nodes-grid", "0",
        "--fpu-c-grid", "0.0,0.45",
        "--reuse-tree-grid", "0,1",
        "--games-per-cell", "10",
        "--output", "out.jsonl",
        "--device", "cpu",
    ], eval_fn=stub, derby_check=lambda: False)
    assert rc == 0
    # 1 × 1 × 2 × 2 × 1 × 1 = 4 cells.
    assert len(calls) == 4
    # Stub saw both fpu and reuse-tree values.
    fpu_vals = {c["fpu_reduction_c"] for c in calls}
    reuse_vals = {c["reuse_tree"] for c in calls}
    assert fpu_vals == {0.0, 0.45}
    assert reuse_vals == {0, 1}


# ---------------------------------------------------------------------------
# Derby-running gate


def test_derby_check_blocks_unless_acknowledged(capsys):
    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100",
        "--vcf-nodes-grid", "0",
    ], eval_fn=lambda **kw: (_ for _ in ()).throw(AssertionError("must not run")),
       derby_check=lambda: True)
    err = capsys.readouterr().err
    assert rc != 0
    assert "delo_derby.py" in err
    assert "--i-know-derby-is-running" in err


def test_derby_check_pass_when_ack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stub, calls = _stub_eval_factory()
    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100",
        "--vcf-nodes-grid", "0",
        "--games-per-cell", "4",
        "--output", "ok.jsonl",
        "--i-know-derby-is-running",
    ], eval_fn=stub, derby_check=lambda: True)
    assert rc == 0
    assert len(calls) == 1


def test_is_derby_running_is_callable():
    result = probe.is_derby_running()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Bead acceptance (e) — markdown table renders >2 varying axes


def test_distance_grid_renders_shape_and_marks_best_cell():
    """Classic 2-axis case (legacy derby-5xs shape)."""
    results = [
        probe.CellResult(sims=100, eval_vcf_nodes=0, n_games=10,
                         baseline="lookahead:depth=4", distance=0.50,
                         black_win_rate=0.5, white_loss_rate=0.0),
        probe.CellResult(sims=100, eval_vcf_nodes=200, n_games=10,
                         baseline="lookahead:depth=4", distance=0.30,
                         black_win_rate=0.7, white_loss_rate=0.0),
        probe.CellResult(sims=200, eval_vcf_nodes=0, n_games=10,
                         baseline="lookahead:depth=4", distance=0.40,
                         black_win_rate=0.6, white_loss_rate=0.0),
        probe.CellResult(sims=200, eval_vcf_nodes=200, n_games=10,
                         baseline="lookahead:depth=4", distance=0.10,
                         black_win_rate=0.9, white_loss_rate=0.0),  # best
    ]
    grid = probe.format_distance_grid(results, [100, 200], [0, 200])
    assert "distance-to-100% grid" in grid
    assert "0" in grid and "200" in grid
    assert "100" in grid and "200" in grid
    star_lines = [ln for ln in grid.splitlines() if "*" in ln]
    assert any("0.100*" in ln for ln in star_lines)
    assert "best cell" in grid
    assert "sims=200" in grid and "vcf=200" in grid


def test_distance_grid_handles_three_or_more_varying_axes():
    """Bead-spec acceptance (e): when ≥3 axes vary the formatter renders the
    long-form summary view (top-N + per-cell axis values) instead of a 2D
    pivot that would mis-represent the matrix."""
    # 3 varying axes: sims × vcf × fpu. Build a 2×2×2 = 8-result fixture.
    results = []
    for sims in (100, 200):
        for vcf in (0, 200):
            for fpu in (0.0, 0.45):
                # Choose distances so the all-OFF baseline is the best.
                dist = 0.1 if (sims == 100 and vcf == 0 and fpu == 0.0) else 0.5
                results.append(probe.CellResult(
                    sims=sims, eval_vcf_nodes=vcf, fpu_reduction_c=fpu,
                    n_games=10, baseline="lookahead:depth=4",
                    distance=dist, black_win_rate=1.0 - dist,
                    white_loss_rate=0.0,
                ))
    grid = probe.format_distance_grid(
        results, sims_grid=[100, 200], vcf_grid=[0, 200],
        fpu_c_grid=[0.0, 0.45],
    )
    # The long-form summary is rendered (≥3 axes vary).
    assert "≥3 axes vary" in grid
    assert "sims" in grid and "eval_vcf_nodes" in grid and "fpu_reduction_c" in grid
    assert "top" in grid and "cells by distance" in grid
    # Best cell is rendered.
    assert "best cell" in grid
    assert "fpu=0" in grid  # the best is fpu=0 (the all-OFF baseline)
