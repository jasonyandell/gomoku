"""Tests for scripts/lab_status.py — the one-glance cockpit view."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import derby_pool  # noqa: E402
import lab_status  # noqa: E402


def test_snapshot_degrades_gracefully(tmp_path):
    # nothing exists yet: no broker state, fresh queue, empty pool
    snap = lab_status.snapshot(tmp_path / "derby", tmp_path / "q", tmp_path / "pool")
    assert set(snap) == {"daemon", "broker", "pool", "needs_you"}
    assert snap["broker"] == {}
    assert snap["needs_you"] is None
    assert snap["daemon"]["alive"] is False
    # render must not crash on an empty lab
    text = lab_status.render(snap)
    assert "NEEDS YOU: —" in text
    assert "DAEMON: DOWN" in text


def test_snapshot_reflects_broker_and_pool(tmp_path):
    derby = tmp_path / "derby"
    derby.mkdir()
    (derby / "broker_state.json").write_text(json.dumps({
        "tick": 7, "champion": "stack", "needs_you": None, "current_job": None,
        "lanes": {"control": {"status": "running", "chunks_done": 3,
                              "wall_secs_total": 900.0, "peak_elo": 1240.0,
                              "climb_rate": 18.0, "chunks_since_peak": 1}},
        "last_verdict": None,
    }))
    pool = tmp_path / "pool"
    derby_pool.register("vct", "cellV", "vct teacher", pool_dir=pool)
    snap = lab_status.snapshot(derby, tmp_path / "q", pool)
    assert snap["broker"]["champion"] == "stack"
    assert "vct" in snap["pool"]["available"]
    text = lab_status.render(snap)
    assert "control" in text
    assert "champion=stack" in text
    assert "vct" in text


def test_render_surfaces_needs_you(tmp_path):
    derby = tmp_path / "derby"
    derby.mkdir()
    (derby / "broker_state.json").write_text(json.dumps({
        "tick": 1, "champion": None, "current_job": None, "lanes": {},
        "needs_you": "derby verdict inconclusive: top two inside CI after escalation",
        "last_verdict": None,
    }))
    snap = lab_status.snapshot(derby, tmp_path / "q", tmp_path / "pool")
    assert snap["needs_you"]
    text = lab_status.render(snap)
    assert "NEEDS YOU: derby verdict inconclusive" in text
