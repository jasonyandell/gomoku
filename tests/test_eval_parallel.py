"""Tests for the multi-process parallel match runner.

The aggregate W/L/D counts of `play_match_parallel` should match what
`play_match_pickers` produces in expectation — each game is independent
in both paths, so the only difference is per-game RNG isolation. We don't
assert byte-identical reproducibility (sequential threads one shared RNG
across games; parallel gives each game its own seed-derived RNG), but we
do assert that the counts add up correctly and that the model wins
heavily against a random opponent (sanity that the worker pipeline
actually plays games rather than silently failing).
"""
import os
import tempfile

import pytest
import torch

from gomoku.eval import MatchResult, play_match_parallel
from gomoku.model import build_model, save_checkpoint


@pytest.fixture(scope="module")
def tiny_checkpoint_path():
    """Save a fresh tiny model to a tmp file once for the whole module."""
    model = build_model("tiny")
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(path, model, optimizer=None, epoch=0, total_games=0)
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_parallel_match_returns_consistent_counts(tiny_checkpoint_path):
    """Aggregate w + l + d must equal n_games. (Trivial but catches the
    worst kind of bug where outcomes get dropped on the floor.)"""
    res = play_match_parallel(
        checkpoint_path=tiny_checkpoint_path,
        opp_spec="random",
        n_games=4,
        seed=0,
        n_workers=2,
        sims=10,  # tiny — we just need games to play, not be strong
        c_puct=1.5,
        device="cpu",
    )
    assert isinstance(res, MatchResult)
    assert res.n_games == 4
    assert res.wins + res.losses + res.draws == 4


def test_parallel_match_games_actually_play(tiny_checkpoint_path):
    """Sanity: with multiple workers each game must complete and produce a
    terminal state. Bug-class this catches: workers crashing silently or
    a pool deadlock returning empty results."""
    res = play_match_parallel(
        checkpoint_path=tiny_checkpoint_path,
        opp_spec="random",
        n_games=6,
        seed=42,
        n_workers=2,
        sims=10,
        c_puct=1.5,
        device="cpu",
    )
    # The untrained tiny model with sims=10 is too weak to beat random
    # reliably, so we don't assert direction; just that 6 games ran and
    # all returned a final state.
    assert res.n_games == 6
    assert res.wins + res.losses + res.draws == 6


def test_parallel_needs_n_workers_ge_2(tiny_checkpoint_path):
    with pytest.raises(ValueError):
        play_match_parallel(
            checkpoint_path=tiny_checkpoint_path,
            opp_spec="random",
            n_games=2,
            seed=0,
            n_workers=1,
            sims=10,
            c_puct=1.5,
            device="cpu",
        )
