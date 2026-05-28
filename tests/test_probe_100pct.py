"""Tests for scripts/probe_100pct.py — the eval-sims × eval-VCF sweep driver
(derby-5xs).

CPU-only, NO GPU, NO actual eval/model loads. The single ``run_cell_eval``
seam is mocked with a deterministic stub that returns fixed color-split
tallies per cell; we assert (a) the driver iterates the full Cartesian grid,
(b) each cell calls eval with the right --sims / --eval-vcf-nodes values,
(c) distance-to-100% is computed by importing scripts/report_100pct.py's
formula (not duplicated), (d) the output JSON has the expected shape, and (e)
``--dry-run`` skips eval entirely.

A separate test mocks ``is_derby_running`` and verifies the
``--i-know-derby-is-running`` gate.
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
    cells = probe.enumerate_cells([400, 100, 200, 800], [800, 0, 200])
    # 4 sims × 3 vcf = 12 cells.
    assert len(cells) == 12
    # Outer order: ascending sims.
    sims_order = [c.sims for c in cells]
    assert sims_order == sorted(sims_order)
    # Inner order: per-sims-group ascending vcf.
    for i in range(0, len(cells), 3):
        group = cells[i:i + 3]
        assert [c.eval_vcf_nodes for c in group] == [0, 200, 800]
    # Cheapest cell first.
    assert cells[0] == probe.Cell(sims=100, eval_vcf_nodes=0)
    # Most expensive cell last.
    assert cells[-1] == probe.Cell(sims=800, eval_vcf_nodes=800)


def test_default_cli_args_produce_4_by_3_grid():
    args = probe.parse_args(["--checkpoint", "irrelevant.pt"])
    sims = probe._parse_int_list(args.sims_grid)
    vcf = probe._parse_int_list(args.vcf_nodes_grid)
    assert sims == [100, 200, 400, 800]
    assert vcf == [0, 200, 800]
    cells = probe.enumerate_cells(sims, vcf)
    assert len(cells) == 12


# ---------------------------------------------------------------------------
# Distance-to-100% — imports report_100pct's formula, doesn't duplicate it


def test_cell_distance_uses_report_100pct_formula_perfect_cell():
    # Perfect cell: wins all 20 black games, never loses any of 20 white.
    dist, bw, wl = probe.cell_distance_to_100(
        black_w=20, black_l=0, black_d=0,
        white_w=10, white_l=0, white_d=10,
        baseline_label="lookahead:depth=4",
    )
    assert dist == pytest.approx(0.0)
    assert bw == pytest.approx(1.0)
    assert wl == pytest.approx(0.0)


def test_cell_distance_uses_report_100pct_formula_known_value():
    # 10/0/10 black means 50% black win rate; 0/2/18 white means 10% white
    # loss rate; report_100pct.score's per-baseline contribution is
    # (1 - 0.5) + 0.1 == 0.6.
    dist, bw, wl = probe.cell_distance_to_100(
        black_w=10, black_l=0, black_d=10,
        white_w=18, white_l=2, white_d=0,
        baseline_label="lookahead:depth=4",
    )
    assert bw == pytest.approx(0.5)
    assert wl == pytest.approx(0.1)
    assert dist == pytest.approx(0.6)


def test_cell_distance_delegates_to_report_100pct(monkeypatch):
    """If we monkey-patch report_100pct.score the probe should see the new
    value — proving the formula is imported, not re-implemented inline."""
    calls: list[dict] = []

    def fake_score(agg):
        calls.append(agg)
        return {b: (0.0, 0.0, 0, 0) for b in report_100pct.BASELINES}, 1234.0

    monkeypatch.setattr(probe, "report_100pct_score", fake_score)
    dist, bw, wl = probe.cell_distance_to_100(
        black_w=1, black_l=0, black_d=0, white_w=0, white_l=1, white_d=0,
        baseline_label="lookahead:depth=4",
    )
    assert dist == 1234.0
    assert calls, "expected probe to call report_100pct_score"


# ---------------------------------------------------------------------------
# Driver: iterates grid, calls eval with the right kwargs per cell


def _stub_eval_factory():
    """Build a stub eval_fn that records every kwargs invocation and returns
    a deterministic tally derived from the cell's (sims, vcf_nodes) values."""
    calls: list[dict] = []

    def stub(*, checkpoint, baseline, n_games, sims, eval_vcf_nodes,
             c_puct, device, seed, n_workers):
        calls.append(dict(
            checkpoint=checkpoint, baseline=baseline, n_games=n_games,
            sims=sims, eval_vcf_nodes=eval_vcf_nodes, c_puct=c_puct,
            device=device, seed=seed, n_workers=n_workers,
        ))
        # Deterministic: more sims and more vcf push black wins up linearly.
        half = n_games // 2
        bw_bonus = min(half, sims // 200 + eval_vcf_nodes // 200)
        black_w = min(half, bw_bonus)
        black_d = half - black_w
        # White: never lose.
        white_w = 0
        white_l = 0
        white_d = half
        return {
            "n_games": n_games,
            "wins": black_w + white_w,
            "losses": 0,
            "draws": black_d + white_d,
            "black_w": black_w, "black_l": 0, "black_d": black_d,
            "white_w": white_w, "white_l": white_l, "white_d": white_d,
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
    # One call per cell, in the same order as cells.
    assert len(calls) == len(cells) == 4
    for cell, call in zip(cells, calls):
        assert call["sims"] == cell.sims
        assert call["eval_vcf_nodes"] == cell.eval_vcf_nodes
        assert call["n_games"] == 40
        assert call["baseline"] == "lookahead:depth=4"
        assert call["checkpoint"] == "fake.pt"
    # CellResult shape:
    for r in results:
        assert r.error is None
        assert r.n_games == 40
        assert r.black_w + r.black_l + r.black_d == 20
        assert r.white_w + r.white_l + r.white_d == 20
        # Distance should match report_100pct's formula on this cell's tally.
        expected_dist, _bw, _wl = probe.cell_distance_to_100(
            black_w=r.black_w, black_l=r.black_l, black_d=r.black_d,
            white_w=r.white_w, white_l=r.white_l, white_d=r.white_d,
            baseline_label=r.baseline,
        )
        assert r.distance == pytest.approx(expected_dist)


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
    assert len(lines) == 1 + len(results)  # meta header + rows.
    head = json.loads(lines[0])
    assert "meta" in head
    assert head["meta"]["checkpoint"] == "fake.pt"
    row = json.loads(lines[1])
    expected_keys = {
        "sims", "eval_vcf_nodes", "n_games", "baseline",
        "black_w", "black_l", "black_d",
        "white_w", "white_l", "white_d",
        "wins", "losses", "draws",
        "distance", "black_win_rate", "white_loss_rate",
        "wall_secs", "error",
    }
    assert expected_keys.issubset(row.keys()), \
        f"missing keys: {expected_keys - row.keys()}"


def test_main_writes_jsonl_and_does_not_call_eval_in_dry_run(tmp_path, capsys):
    # --dry-run path: no eval, no derby-check needed, no jsonl required.
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
    ], eval_fn=stub, derby_check=lambda: False)
    assert rc == 0
    assert len(calls) == 4
    out_path = tmp_path / "out.jsonl"
    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    # meta + 4 cells
    assert len(lines) == 5
    captured = capsys.readouterr().out
    # The table + grid should appear in stdout.
    assert "distance-to-100% grid" in captured
    assert "best cell" in captured


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
    # Smoke: should return a bool without raising, regardless of system state.
    result = probe.is_derby_running()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Distance grid formatting (best cell marker + shape)


def test_distance_grid_renders_shape_and_marks_best_cell():
    # Hand-build 2x2 results: one clear winner.
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
    # Header row mentions both vcf values.
    assert "0" in grid and "200" in grid
    # Both sims values appear as row labels.
    assert "100" in grid and "200" in grid
    # Best cell tagged once with *.
    star_lines = [ln for ln in grid.splitlines() if "*" in ln]
    # One row + one summary line.
    assert any("0.100*" in ln for ln in star_lines)
    assert "best cell" in grid
    assert "sims=200" in grid and "vcf=200" in grid
