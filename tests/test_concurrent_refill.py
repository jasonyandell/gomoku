"""Continuous-refill self-play generation (issue #112).

`concurrent_games > 0` caps the ACTIVE set and seeds a replacement game the
moment one completes, so the per-ply merged oracle solve and the search wave
always run at full width. These tests exercise the native path with a uniform
CPU evaluator (no MLX/Metal); the oracle interplay is covered by monkeypatched
partition tests below and by scripts/gen_poison_check.py end-to-end.
"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku import self_play as sp
from gomoku.game import HISTORY_PLY, N_ACTIONS
from gomoku.self_play import generate_games


def _uniform_planes_evaluator(planes: np.ndarray):
    planes = np.asarray(planes)
    batch = planes.shape[0]
    priors = np.zeros((batch, N_ACTIONS), dtype=np.float32)
    values = np.zeros(batch, dtype=np.float32)
    occupied = (planes[:, 0] > 0.5) | (planes[:, HISTORY_PLY] > 0.5)
    legal = ~occupied.reshape(batch, N_ACTIONS)
    for i in range(batch):
        if legal[i].any():
            priors[i, legal[i]] = 1.0 / float(legal[i].sum())
    return priors, values


class _Evaluator:
    """Exposes `evaluate_planes` so generate_games routes to the native path."""

    evaluate_planes = staticmethod(_uniform_planes_evaluator)


@pytest.mark.parametrize("n_games,concurrent", [(6, 2), (5, 3), (4, 4), (3, 8)])
def test_refill_produces_all_games(n_games, concurrent):
    records = generate_games(
        n_games,
        _Evaluator(),
        n_simulations=8,
        wave_size=4,
        max_plies=30,
        rng=np.random.default_rng(7),
        augment_symmetries=False,
        concurrent_games=concurrent,
    )
    assert len(records) == n_games
    for rec in records:
        assert rec.plies > 0
        assert len(rec.examples) > 0
        assert rec.outcome in (-1.0, 0.0, 1.0)
        for ex in rec.examples:
            assert np.isfinite(ex.pi).all()
            assert abs(ex.pi.sum() - 1.0) < 1e-4


def test_refill_alternating_sides_and_plies_consistent():
    """Per-game plies replaced the global lockstep counter: sides recorded in
    each trajectory must still alternate from the game's own start, and the
    record's plies must equal its example count (no teachers, no openings,
    max_plies ends nothing early at 81)."""
    records = generate_games(
        4,
        _Evaluator(),
        n_simulations=8,
        wave_size=4,
        rng=np.random.default_rng(11),
        augment_symmetries=False,
        concurrent_games=2,
    )
    for rec in records:
        sides = [ex.side for ex in rec.examples]
        assert sides == [k % 2 for k in range(len(sides))]
        assert rec.plies == len(rec.examples)


def test_refill_max_plies_draw():
    """Games cut off at max_plies retire as draws with plies == max_plies."""
    records = generate_games(
        3,
        _Evaluator(),
        n_simulations=4,
        wave_size=4,
        max_plies=6,
        rng=np.random.default_rng(3),
        augment_symmetries=False,
        concurrent_games=2,
    )
    assert len(records) == 3
    for rec in records:
        assert rec.plies <= 6
        if rec.plies == 6 and rec.outcome == 0.0:
            assert len(rec.examples) == 6


def test_lockstep_zero_matches_full_width():
    """concurrent_games=0 and concurrent_games>=n_games both take the legacy
    lockstep path (width == n_games, no refill) — same-seed identical output."""
    kw = dict(
        n_simulations=8,
        wave_size=4,
        max_plies=20,
        augment_symmetries=False,
    )
    a = generate_games(3, _Evaluator(), rng=np.random.default_rng(5),
                       concurrent_games=0, **kw)
    b = generate_games(3, _Evaluator(), rng=np.random.default_rng(5),
                       concurrent_games=3, **kw)
    c = generate_games(3, _Evaluator(), rng=np.random.default_rng(5),
                       concurrent_games=99, **kw)
    for other in (b, c):
        assert len(a) == len(other)
        for ra, rb in zip(a, other):
            assert ra.plies == rb.plies
            assert ra.outcome == rb.outcome
            assert len(ra.examples) == len(rb.examples)
            for ea, eb in zip(ra.examples, rb.examples):
                assert np.array_equal(ea.planes, eb.planes)
                assert np.array_equal(ea.pi, eb.pi)
                assert ea.z == eb.z


def test_streaming_flush_delivers_every_game_once():
    """flush_records mode: all games arrive through the callback in chunks of
    flush_games (final chunk may be short), the call returns [], and every
    delivered record is complete."""
    chunks: list[list] = []
    ret = generate_games(
        10,
        _Evaluator(),
        n_simulations=8,
        wave_size=4,
        max_plies=25,
        rng=np.random.default_rng(21),
        augment_symmetries=False,
        concurrent_games=3,
        flush_records=chunks.append,
        flush_games=4,
    )
    assert ret == []
    delivered = [rec for chunk in chunks for rec in chunk]
    assert len(delivered) == 10
    for chunk in chunks[:-1]:
        assert len(chunk) >= 4
    for rec in delivered:
        assert rec.plies > 0
        assert len(rec.examples) > 0
        sides = [ex.side for ex in rec.examples]
        assert sides == [k % 2 for k in range(len(sides))]


def test_streaming_refresh_evaluator_hot_swaps():
    """refresh_evaluator is polled every round; returning a new evaluator makes
    subsequent leaf evals use it (games in flight keep their trees)."""
    calls = {"refresh": 0, "second_ev": 0}

    class _CountingEvaluator:
        @staticmethod
        def evaluate_planes(planes):
            calls["second_ev"] += 1
            return _uniform_planes_evaluator(planes)

    def refresh():
        calls["refresh"] += 1
        if calls["refresh"] == 3:
            return _CountingEvaluator()
        return None

    records = generate_games(
        6,
        _Evaluator(),
        n_simulations=8,
        wave_size=4,
        max_plies=20,
        rng=np.random.default_rng(2),
        augment_symmetries=False,
        concurrent_games=2,
        refresh_evaluator=refresh,
    )
    assert len(records) == 6
    assert calls["refresh"] >= 3
    assert calls["second_ev"] > 0     # the swapped-in evaluator served evals


def test_refill_with_terminus_partition(monkeypatch):
    """Oracle partitions fire mid-stream with per-game plies: force the
    terminus to end every game at its 3rd ply and check sides/plies are
    computed from the GAME-LOCAL ply, not a global round counter."""
    sp.configure_vct_terminus(enabled=True, budget=50)
    try:
        def fake_ply_solve(planes_list, *, want_terminus, want_defense,
                           defense_max_cands=0, profile=None):
            B = len(planes_list)
            win = np.zeros(B, dtype=bool)
            move = np.full(B, -1, dtype=np.int64)
            for s, p in enumerate(planes_list):
                stones = int(np.asarray(p)[0].sum() + np.asarray(p)[HISTORY_PLY].sum())
                if stones >= 3:          # game-local ply 3 -> terminus fires
                    win[s] = True
                    occ = (np.asarray(p)[0] > 0.5) | (np.asarray(p)[HISTORY_PLY] > 0.5)
                    move[s] = int(np.flatnonzero(~occ.reshape(-1))[0])
            return win, move, None, None

        monkeypatch.setattr(sp, "_oracle_ply_solve", fake_ply_solve)
        records = generate_games(
            5,
            _Evaluator(),
            n_simulations=4,
            wave_size=4,
            rng=np.random.default_rng(9),
            augment_symmetries=False,
            concurrent_games=2,
        )
        assert len(records) == 5
        for rec in records:
            # terminus at game-local ply 3: side to move = 3 % 2 = 1 (white)
            # -> outcome_for_black = -1.0, plies = 4, last pi is one-hot.
            assert rec.plies == 4
            assert rec.outcome == -1.0
            last_pi = rec.examples[-1].pi
            assert (last_pi > 0.999).sum() == 1
    finally:
        sp.configure_vct_terminus(enabled=False)
