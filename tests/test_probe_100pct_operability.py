"""Operability-gate tests for scripts/probe_100pct.py (derby-cec, 2026-05-28).

Four properties retrofitted to a script that was already in production:

  (a) STREAMING output — per-cell rows are flushed to ``--output`` as each
      cell completes (append mode). A SIGKILL after cell 2 leaves
      ``meta + cell1 + cell2`` safely on disk.
  (b) ``--resume`` — re-running with the same ``--output`` parses existing
      cell rows and SKIPS already-completed cells. Default ON; ``--no-resume``
      bypasses.
  (c) HONEST TIMING self-report — upfront ``[probe] plan: ...`` line with a
      per-cell estimate + total ETA; end-of-run ``[probe] actual/estimate
      ratio: ...`` line for calibration.
  (d) GRID respects single values — ``--fpu-c-grid 0.45`` runs EXACTLY ONE
      cell, not 8 (no silent {0, 0.45} auto-expansion).

CPU-only, NO GPU. The single ``run_cell_eval`` seam is stubbed with a
deterministic function returning fixed numbers (lifted from
test_probe_100pct.py).
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

# Make scripts/ importable as a regular package.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import probe_100pct as probe  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic stub eval (cell -> fixed tally)


def _stub_eval(**kwargs):
    """Return a fixed-shape tally that varies a little with sims so the
    distance landscape is non-degenerate (different cells get different
    distances, the cheap-first ordering matters, etc.). NO torch, NO MPS."""
    n_games = kwargs["n_games"]
    sims = kwargs["sims"]
    vcf = kwargs["eval_vcf_nodes"]
    half = n_games // 2
    bw_bonus = min(half, sims // 200 + vcf // 200)
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


# ---------------------------------------------------------------------------
# (d) Grid: single value runs exactly one cell


def test_grid_respects_single_value_no_auto_injection(tmp_path, monkeypatch, capsys):
    """Derby-cec acceptance (d): ``--fpu-c-grid 0.45`` runs EXACTLY ONE cell
    (not 8). The derby-u8d behaviour silently expanded to {0, 0.45}; this
    test pins the reversal."""
    monkeypatch.chdir(tmp_path)
    calls: list[dict] = []

    def stub(**kw):
        calls.append(kw)
        return _stub_eval(**kw)

    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100",
        "--vcf-nodes-grid", "0",
        "--fpu-c-grid", "0.45",  # single value
        "--reuse-tree-grid", "0",
        "--proven-prop-grid", "0",
        "--proven-vcf-leaf-nodes-grid", "0",
        "--games-per-cell", "4",
        "--output", "out.jsonl",
        "--device", "cpu",
    ], eval_fn=stub, derby_check=lambda: False)
    assert rc == 0, capsys.readouterr().out
    # The bug: without the fix this would be 2 calls (fpu auto-expanded to
    # {0, 0.45}); the multi-axis combination would actually produce 8 cells
    # if you also single-valued the other axes.
    assert len(calls) == 1, (
        f"single-value --fpu-c-grid 0.45 ran {len(calls)} cells, "
        f"not 1 — derby-cec grid bug regressed"
    )
    assert calls[0]["fpu_reduction_c"] == 0.45
    # Confirm the would-be-injected OFF (0.0) was NOT also run.
    fpu_vals = {c["fpu_reduction_c"] for c in calls}
    assert fpu_vals == {0.45}


def test_grid_respects_single_value_across_all_new_axes(tmp_path, monkeypatch):
    """Pin the same property for each derby-u8d axis individually. A user
    who specifies a single non-OFF value on EVERY axis gets ONE cell."""
    monkeypatch.chdir(tmp_path)
    calls: list[dict] = []

    def stub(**kw):
        calls.append(kw)
        return _stub_eval(**kw)

    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "200",                # single non-OFF
        "--vcf-nodes-grid", "200",           # single non-OFF
        "--fpu-c-grid", "0.45",              # single non-OFF
        "--reuse-tree-grid", "1",            # single non-OFF
        "--proven-prop-grid", "1",           # single non-OFF
        "--proven-vcf-leaf-nodes-grid", "200",  # single non-OFF
        "--games-per-cell", "4",
        "--output", "out.jsonl",
        "--device", "cpu",
    ], eval_fn=stub, derby_check=lambda: False)
    assert rc == 0
    # OLD (derby-u8d) bug: each single-non-OFF axis would auto-inject OFF,
    # so 2^6 = 64 cells. NEW (derby-cec): 1.
    assert len(calls) == 1, (
        f"all-single-value grids ran {len(calls)} cells, not 1 — "
        f"derby-cec respect-the-grid contract violated"
    )


def test_grid_two_value_still_runs_full_cartesian(tmp_path, monkeypatch):
    """Backward-compat: explicit two-value grids still produce the full
    Cartesian product. The user opted in to the ablation."""
    monkeypatch.chdir(tmp_path)
    calls: list[dict] = []

    def stub(**kw):
        calls.append(kw)
        return _stub_eval(**kw)

    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100",
        "--vcf-nodes-grid", "0",
        "--fpu-c-grid", "0,0.45",   # explicit two-value (the documented ablation)
        "--reuse-tree-grid", "0",
        "--proven-prop-grid", "0",
        "--proven-vcf-leaf-nodes-grid", "0",
        "--games-per-cell", "4",
        "--output", "out.jsonl",
        "--device", "cpu",
    ], eval_fn=stub, derby_check=lambda: False)
    assert rc == 0
    assert len(calls) == 2
    assert {c["fpu_reduction_c"] for c in calls} == {0.0, 0.45}


# ---------------------------------------------------------------------------
# (a) Streaming output


def test_streaming_writes_rows_per_cell_during_execution(tmp_path, monkeypatch):
    """Derby-cec acceptance (a): with N=3 cells, the output file holds
    meta + 3 rows AFTER each completed cell, not only at the end.

    We assert per-cell streaming by checking the file's row count INSIDE
    the eval stub (i.e. mid-run) — the prior cell's row must be on disk
    before the next cell's stub is called."""
    monkeypatch.chdir(tmp_path)
    out_path = tmp_path / "stream.jsonl"

    observed_row_counts: list[int] = []

    def counting_stub(**kw):
        # When the K'th cell's eval starts, the file should already have
        # meta + (K-1) rows on disk from the previously-streamed cells.
        if out_path.exists():
            observed_row_counts.append(
                len(out_path.read_text().strip().splitlines())
            )
        else:
            observed_row_counts.append(0)
        return _stub_eval(**kw)

    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100,200,400",
        "--vcf-nodes-grid", "0",
        "--games-per-cell", "4",
        "--output", str(out_path),
        "--device", "cpu",
    ], eval_fn=counting_stub, derby_check=lambda: False)
    assert rc == 0
    # 3 cells: pre-eval observations expected to be:
    #   cell 1: 1 (meta header only — written before the loop)
    #   cell 2: 2 (meta + cell1 row)
    #   cell 3: 3 (meta + cell1 + cell2 rows)
    assert observed_row_counts == [1, 2, 3], (
        f"streaming row-count sequence was {observed_row_counts}; expected "
        f"[1, 2, 3] (meta written first, then 1 row per completed cell)."
    )
    final_lines = out_path.read_text().strip().splitlines()
    assert len(final_lines) == 1 + 3  # meta + 3 cells
    meta = json.loads(final_lines[0])
    assert "meta" in meta


def test_streaming_survives_sigkill_after_second_cell(tmp_path, monkeypatch):
    """Derby-cec acceptance (a) crash-resistance: a SIGKILL after the second
    cell leaves meta + 2 cells on disk (the streaming + per-cell flush
    contract). Simulate by raising SystemExit from inside the stub on cell 3."""
    monkeypatch.chdir(tmp_path)
    out_path = tmp_path / "killed.jsonl"

    counter = {"n": 0}

    def killing_stub(**kw):
        counter["n"] += 1
        if counter["n"] == 3:
            # Simulate SIGKILL (no atexit, no batch end-write). SystemExit
            # is harsher than a normal exception — it bypasses run_probe's
            # try/except (which only catches Exception) and propagates out.
            raise SystemExit("simulated SIGKILL after cell 2")
        return _stub_eval(**kw)

    with pytest.raises(SystemExit):
        probe.main([
            "--checkpoint", "fake.pt",
            "--baseline", "lookahead:depth=4",
            "--sims-grid", "100,200,400",
            "--vcf-nodes-grid", "0",
            "--games-per-cell", "4",
            "--output", str(out_path),
            "--device", "cpu",
        ], eval_fn=killing_stub, derby_check=lambda: False)

    # 2 cells completed before the kill — meta + 2 cell rows must be on disk.
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1 + 2, (
        f"expected meta + 2 cell rows after SIGKILL-on-cell-3, got {len(lines)}: "
        f"{lines}"
    )
    meta = json.loads(lines[0])
    assert "meta" in meta
    row1 = json.loads(lines[1])
    row2 = json.loads(lines[2])
    assert row1["sims"] == 100
    assert row2["sims"] == 200


# ---------------------------------------------------------------------------
# (b) --resume


def test_resume_skips_completed_cells_and_appends_remaining(tmp_path, monkeypatch, capsys):
    """Derby-cec acceptance (b): pre-populate the output file with 2 cell
    rows; re-running with the same ``--output`` skips those and runs only
    cell 3, ending at meta + 3 cells total."""
    monkeypatch.chdir(tmp_path)
    out_path = tmp_path / "resume.jsonl"

    # First run: a SIGKILL-style mid-run termination (use the same trick as
    # the streaming test — kill the stub on cell 3).
    counter = {"n": 0}

    def kill_after_2(**kw):
        counter["n"] += 1
        if counter["n"] == 3:
            raise SystemExit("simulated mid-run kill")
        return _stub_eval(**kw)

    with pytest.raises(SystemExit):
        probe.main([
            "--checkpoint", "fake.pt",
            "--baseline", "lookahead:depth=4",
            "--sims-grid", "100,200,400",
            "--vcf-nodes-grid", "0",
            "--games-per-cell", "4",
            "--output", str(out_path),
            "--device", "cpu",
        ], eval_fn=kill_after_2, derby_check=lambda: False)
    capsys.readouterr()  # drain
    pre_lines = out_path.read_text().strip().splitlines()
    assert len(pre_lines) == 3  # meta + 2 cells

    # Second run: same --output. The driver should report 2/3 cells already
    # complete, re-eval ONLY cell 3, append it, end at meta + 3 cells.
    calls: list[dict] = []

    def fresh_stub(**kw):
        calls.append(kw)
        return _stub_eval(**kw)

    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100,200,400",
        "--vcf-nodes-grid", "0",
        "--games-per-cell", "4",
        "--output", str(out_path),
        "--device", "cpu",
    ], eval_fn=fresh_stub, derby_check=lambda: False)
    out = capsys.readouterr().out
    assert rc == 0
    # Only cell 3 (sims=400) was re-evaluated.
    assert len(calls) == 1
    assert calls[0]["sims"] == 400
    # Resume banner printed.
    assert "resuming" in out
    assert "2/3" in out or "2/3 cells" in out
    # File state: meta + 3 cells = 4 lines.
    post_lines = out_path.read_text().strip().splitlines()
    assert len(post_lines) == 4, (
        f"expected meta + 3 cell rows post-resume, got {len(post_lines)}"
    )
    rows = [json.loads(l) for l in post_lines[1:]]
    sims_run = sorted(r["sims"] for r in rows)
    assert sims_run == [100, 200, 400]


def test_no_resume_re_evaluates_every_cell(tmp_path, monkeypatch, capsys):
    """``--no-resume`` ignores the existing --output and re-runs everything
    (still appending). After a 2-cell pre-fill, a --no-resume re-run with a
    3-cell grid runs 3 evals; file ends with meta + 2 (old) + 3 (new) = 6
    lines (we still append; we just don't skip)."""
    monkeypatch.chdir(tmp_path)
    out_path = tmp_path / "no_resume.jsonl"

    # First run: full 3-cell sweep.
    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100,200,400",
        "--vcf-nodes-grid", "0",
        "--games-per-cell", "4",
        "--output", str(out_path),
        "--device", "cpu",
    ], eval_fn=_stub_eval, derby_check=lambda: False)
    assert rc == 0
    capsys.readouterr()

    # Second run with --no-resume: should re-eval all 3 cells.
    calls: list[dict] = []

    def stub(**kw):
        calls.append(kw)
        return _stub_eval(**kw)

    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100,200,400",
        "--vcf-nodes-grid", "0",
        "--games-per-cell", "4",
        "--output", str(out_path),
        "--device", "cpu",
        "--no-resume",
    ], eval_fn=stub, derby_check=lambda: False)
    out = capsys.readouterr().out
    assert rc == 0
    assert len(calls) == 3
    assert "no-resume" in out or "--no-resume" in out


# ---------------------------------------------------------------------------
# (c) Honest timing self-report


def test_timing_prints_upfront_estimate_and_end_ratio(tmp_path, monkeypatch, capsys):
    """Derby-cec acceptance (c): the run prints an upfront ``[probe] plan:``
    line with a per-cell estimate + total ETA, and an end-of-run
    ``[probe] actual/estimate ratio:`` line."""
    monkeypatch.chdir(tmp_path)
    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100,200",
        "--vcf-nodes-grid", "0",
        "--games-per-cell", "4",
        "--output", "timing.jsonl",
        "--device", "cpu",
    ], eval_fn=_stub_eval, derby_check=lambda: False)
    assert rc == 0
    out = capsys.readouterr().out
    # Upfront plan + estimate.
    assert "[probe] plan:" in out
    assert "/cell" in out
    assert "cwd=" in out
    assert "checkpoint=" in out
    assert "output=" in out
    # End-of-run actual/estimate ratio.
    assert "actual/estimate ratio:" in out
    assert "ratio" in out


def test_timing_warns_when_ratio_exceeds_2x(tmp_path, monkeypatch, capsys):
    """When the actual wall is >2x the estimate, the driver prints a
    perf-meta-bead suggestion. We force the condition by making the stub
    sleep long enough that the actual >> estimate."""
    monkeypatch.chdir(tmp_path)

    # Shrink the baseline so the estimate is small; the stub sleeps to push
    # actual well past 2x.
    monkeypatch.setattr(probe, "SECS_PER_GAME_AT_SIMS_100_BASELINE", 0.001)

    def slow_stub(**kw):
        time.sleep(0.05)  # 50ms per cell, vs ~4ms estimated => ratio > 2x
        return _stub_eval(**kw)

    rc = probe.main([
        "--checkpoint", "fake.pt",
        "--baseline", "lookahead:depth=4",
        "--sims-grid", "100",
        "--vcf-nodes-grid", "0",
        "--games-per-cell", "4",
        "--output", "slow.jsonl",
        "--device", "cpu",
    ], eval_fn=slow_stub, derby_check=lambda: False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "actual/estimate ratio:" in out
    assert "WARNING" in out or "perf-meta" in out, (
        f"expected a perf-meta warning when ratio >2x; full stdout:\n{out}"
    )


# ---------------------------------------------------------------------------
# Helper-level coverage (cheap regressions for the streaming + resume seams)


def test_parse_completed_keys_handles_missing_file(tmp_path):
    keys, n = probe.parse_completed_keys(tmp_path / "nope.jsonl")
    assert keys == set()
    assert n == 0


def test_parse_completed_keys_skips_meta_and_partial_lines(tmp_path):
    p = tmp_path / "partial.jsonl"
    rows = [
        '{"meta": {"foo": 1}}',
        json.dumps({
            "sims": 100, "eval_vcf_nodes": 0, "fpu_reduction_c": 0.0,
            "reuse_tree": 0, "proven_prop": 0, "proven_vcf_leaf_nodes": 0,
            "n_games": 4,
        }),
        # Partial last line (no newline, truncated JSON).
        '{"sims": 200, "eval_vcf_nod',
    ]
    p.write_text("\n".join(rows) + "\n")
    keys, n = probe.parse_completed_keys(p)
    assert n == 1
    assert (100, 0, 0.0, 0, 0, 0) in keys


def test_estimate_cell_wall_secs_scales_with_sims_and_games():
    cell_100 = probe.Cell(sims=100, eval_vcf_nodes=0)
    cell_400 = probe.Cell(sims=400, eval_vcf_nodes=0)
    e100 = probe.estimate_cell_wall_secs(cell_100, n_games=10)
    e400 = probe.estimate_cell_wall_secs(cell_400, n_games=10)
    # 4x sims should give ~4x estimate (within rounding).
    assert e400 == pytest.approx(4 * e100, rel=0.01)
    # Linear in n_games.
    e100_x4 = probe.estimate_cell_wall_secs(cell_100, n_games=40)
    assert e100_x4 == pytest.approx(4 * e100, rel=0.01)


def test_format_duration_units():
    assert probe.format_duration(12.3) == "12.3s"
    assert probe.format_duration(75.0) == "1.2min"
    assert probe.format_duration(7200.0) == "2.00h"


def test_cell_key_round_trips_with_result_key():
    cell = probe.Cell(
        sims=200, eval_vcf_nodes=400, fpu_reduction_c=0.45,
        reuse_tree=1, proven_prop=1, proven_vcf_leaf_nodes=200,
    )
    k = probe.cell_key(cell)
    # A serialized result row should produce the same key.
    row = {
        "sims": 200, "eval_vcf_nodes": 400, "fpu_reduction_c": 0.45,
        "reuse_tree": 1, "proven_prop": 1, "proven_vcf_leaf_nodes": 200,
    }
    assert probe.result_key(row) == k
