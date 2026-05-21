"""Unit tests for the WL2 worker-side levers (#2 past-checkpoint mix and
#3 poll jitter) implemented in `gomoku.selfplay_worker`.

See `wiki/topics/wl2-scale-emulation-design.md` for the design rationale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gomoku.selfplay_worker import (
    _draw_poll_interval,
    _list_past_checkpoints,
    _pick_wave_mix_source,
)


def _make_ckpt_dir(tmp_path: Path, n: int) -> Path:
    """Create `n` empty `epochNNNN.pt` files + a `latest.pt` symlink + a
    `worker_weights.pt` decoy in `tmp_path`. Returns the path to
    `worker_weights.pt` (the conventional `--weights-path`)."""
    for i in range(1, n + 1):
        (tmp_path / f"epoch{i:04d}.pt").write_bytes(b"")
    worker_path = tmp_path / "worker_weights.pt"
    worker_path.write_bytes(b"")
    if n > 0:
        latest = tmp_path / "latest.pt"
        try:
            latest.symlink_to(tmp_path / f"epoch{n:04d}.pt")
        except OSError:
            # Some filesystems disallow symlinks; just write a plain file
            # so the dedupe path still gets exercised on those systems.
            latest.write_bytes(b"")
    return worker_path


def test_list_past_checkpoints_finds_and_sorts(tmp_path):
    worker_path = _make_ckpt_dir(tmp_path, n=5)
    ckpts = _list_past_checkpoints(str(worker_path))
    assert [e for e, _ in ckpts] == [1, 2, 3, 4, 5]
    for epoch, path in ckpts:
        assert path.name == f"epoch{epoch:04d}.pt"


def test_list_past_checkpoints_excludes_latest_and_worker_weights(tmp_path):
    worker_path = _make_ckpt_dir(tmp_path, n=3)
    names = {p.name for _, p in _list_past_checkpoints(str(worker_path))}
    assert "latest.pt" not in names
    assert "worker_weights.pt" not in names
    assert names == {"epoch0001.pt", "epoch0002.pt", "epoch0003.pt"}


def test_list_past_checkpoints_empty_dir(tmp_path):
    # Just the worker_weights.pt and nothing else.
    (tmp_path / "worker_weights.pt").write_bytes(b"")
    assert _list_past_checkpoints(str(tmp_path / "worker_weights.pt")) == []


def test_pick_wave_mix_source_distribution_within_bounds(tmp_path):
    """With p_recent=0.4 + p_history=0.1, 1000 trials should land each
    bucket close to its expected fraction."""
    worker_path = _make_ckpt_dir(tmp_path, n=200)
    rng = np.random.default_rng(seed=12345)
    counts = {"self": 0, "recent": 0, "history": 0}
    n = 1000
    for _ in range(n):
        src, _ = _pick_wave_mix_source(
            rng,
            p_recent=0.4,
            p_history=0.1,
            recent_window=100,
            weights_path=str(worker_path),
        )
        counts[src] += 1

    # Expected: self=500, recent=400, history=100. Allow generous tolerance
    # (Chebyshev says +/- 3 sigma is ~99%; use 5% absolute bands to keep
    # this robust against the test seed).
    assert 450 <= counts["self"] <= 550, counts
    assert 350 <= counts["recent"] <= 450, counts
    assert 60 <= counts["history"] <= 140, counts


def test_pick_wave_mix_source_respects_recent_window(tmp_path):
    """With recent_window=10, the "recent" pool must only be the last 10
    epochs even though 200 are available on disk."""
    worker_path = _make_ckpt_dir(tmp_path, n=200)
    rng = np.random.default_rng(seed=42)
    eligible = set()
    # Force the "recent" branch every roll by p_recent=1.0.
    for _ in range(500):
        src, path = _pick_wave_mix_source(
            rng,
            p_recent=1.0,
            p_history=0.0,
            recent_window=10,
            weights_path=str(worker_path),
        )
        assert src == "recent"
        assert path is not None
        eligible.add(path.name)
    expected = {f"epoch{e:04d}.pt" for e in range(191, 201)}
    assert eligible == expected, eligible


def test_pick_wave_mix_source_history_can_pick_old_checkpoints(tmp_path):
    worker_path = _make_ckpt_dir(tmp_path, n=200)
    rng = np.random.default_rng(seed=7)
    seen_old = False
    for _ in range(500):
        src, path = _pick_wave_mix_source(
            rng,
            p_recent=0.0,
            p_history=1.0,
            recent_window=100,
            weights_path=str(worker_path),
        )
        assert src == "history"
        assert path is not None
        # An "old" pick is one outside the last-100 window.
        ep = int(path.stem.removeprefix("epoch"))
        if ep <= 100:
            seen_old = True
            break
    assert seen_old, "history bucket should be able to pick checkpoints older than recent_window"


def test_pick_wave_mix_source_defaults_disable(tmp_path):
    worker_path = _make_ckpt_dir(tmp_path, n=5)
    rng = np.random.default_rng(seed=0)
    for _ in range(50):
        src, path = _pick_wave_mix_source(
            rng,
            p_recent=0.0,
            p_history=0.0,
            recent_window=100,
            weights_path=str(worker_path),
        )
        assert src == "self"
        assert path is None


def test_pick_wave_mix_source_falls_back_when_dir_empty(tmp_path):
    """With p_recent=1.0 but no past checkpoints, fall back to "self" rather
    than crash. Early in a run, the worker may roll for "recent" before any
    epoch checkpoints have been written."""
    (tmp_path / "worker_weights.pt").write_bytes(b"")
    rng = np.random.default_rng(seed=0)
    src, path = _pick_wave_mix_source(
        rng,
        p_recent=1.0,
        p_history=0.0,
        recent_window=100,
        weights_path=str(tmp_path / "worker_weights.pt"),
    )
    assert src == "self"
    assert path is None


def test_draw_poll_interval_default_no_jitter():
    rng = np.random.default_rng(seed=0)
    # No min/max set -> always returns base.
    for _ in range(100):
        v = _draw_poll_interval(rng, base_sec=2.0, min_sec=None, max_sec=None)
        assert v == pytest.approx(2.0)


def test_draw_poll_interval_range_respected():
    """Per-worker poll interval samples must all live in [min, max] and
    look roughly uniform across 100 worker startups."""
    rng = np.random.default_rng(seed=2026)
    values = []
    for _ in range(100):
        v = _draw_poll_interval(rng, base_sec=2.0, min_sec=2.0, max_sec=8.0)
        assert 2.0 <= v <= 8.0
        values.append(v)
    arr = np.array(values)
    # Mean of Uniform(2, 8) is 5; allow generous slack for n=100.
    assert 4.0 <= arr.mean() <= 6.0
    # Make sure we actually got spread (not a constant).
    assert arr.std() > 0.5


def test_draw_poll_interval_swaps_when_min_above_max():
    rng = np.random.default_rng(seed=1)
    # Defensively handle a flipped pair -> still in [2, 8].
    v = _draw_poll_interval(rng, base_sec=2.0, min_sec=8.0, max_sec=2.0)
    assert 2.0 <= v <= 8.0


def test_draw_poll_interval_only_min_set():
    """If only --weights-poll-min-sec is provided, --weights-poll-sec acts
    as the implicit max so the worker still gets a real range."""
    rng = np.random.default_rng(seed=3)
    values = [
        _draw_poll_interval(rng, base_sec=8.0, min_sec=2.0, max_sec=None)
        for _ in range(50)
    ]
    assert all(2.0 <= v <= 8.0 for v in values)
