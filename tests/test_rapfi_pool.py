"""Warm Rapfi pool: lifecycle invariants (fake engine, always-on) + the real
native engine (gated, like test_rapfi_native).

Run the real-engine tests with::

    GOMOKU_BOARD_SIZE=15 pytest tests/test_rapfi_pool.py
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from gomoku.board_config import BOARD_SIZE
from gomoku.external_engine import ExternalEngineError
from gomoku.game import GameState
from gomoku.rapfi_pool import RapfiPool, rapfi_available

requires_rapfi = pytest.mark.skipif(
    not rapfi_available() or BOARD_SIZE != 15,
    reason="native Rapfi artifact absent or board_size != 15",
)

# A cmd whose first token is a real file so RapfiPool's existence check passes;
# ExternalEnginePlayer is monkeypatched out so no real subprocess is spawned.
_FAKE_CMD = f"{sys.executable} --fake"


# --------------------------------------------------------------------------
# Lifecycle invariants with a fake engine (no Rapfi needed)
# --------------------------------------------------------------------------
def _install_fake(monkeypatch, *, registry, n_die_first=0):
    """Fake engine factory: the first ``n_die_first`` engines created die once
    on their first call (then recover); later engines (respawns) are healthy."""
    count = {"n": 0}

    class Fake:
        def __init__(self, cfg):
            self.closed = 0
            self._first = count["n"] < n_die_first
            count["n"] += 1
            registry.append(self)

        def __call__(self, state, rng):
            if self._first:
                self._first = False
                raise ExternalEngineError("simulated death")
            return int(state.legal_actions()[0])

        def close(self):
            self.closed += 1

    monkeypatch.setattr("gomoku.rapfi_pool.ExternalEnginePlayer", Fake)
    return Fake


def test_close_is_idempotent_and_closes_each_engine_once(monkeypatch):
    created: list = []
    _install_fake(monkeypatch, registry=created)
    pool = RapfiPool(size=3, cmd=_FAKE_CMD, board_size=BOARD_SIZE)
    assert len(created) == 3
    pool.close()
    pool.close()  # idempotent — a second close must not double-close or raise
    assert all(f.closed == 1 for f in created)


def test_size_preserved_and_no_leak_after_self_heal(monkeypatch):
    created: list = []
    # One of the two initial engines dies once; pick() self-heals (retry) and a
    # healthy replacement is spawned, so every pick still returns a legal move.
    _install_fake(monkeypatch, registry=created, n_die_first=1)
    pool = RapfiPool(size=2, cmd=_FAKE_CMD, board_size=BOARD_SIZE)
    s = GameState.initial()
    legal = set(int(a) for a in s.legal_actions())
    for _ in range(6):
        assert pool.pick(s) in legal
    assert pool._pool.qsize() == 2  # capacity preserved across the death
    pool.close()
    # No subprocess leaked: every engine ever created is closed exactly once
    # (the retired corpse via _retire, the rest via close()).
    assert all(f.closed == 1 for f in created)


def test_label_states_preserves_order(monkeypatch):
    created: list = []
    _install_fake(monkeypatch, registry=created)
    pool = RapfiPool(size=3, cmd=_FAKE_CMD, board_size=BOARD_SIZE)
    states = [GameState.initial() for _ in range(7)]
    moves = pool.label_states(states)
    # The fake returns the first legal action; identical states -> identical moves.
    assert len(moves) == 7
    assert len(set(moves)) == 1
    pool.close()


# --------------------------------------------------------------------------
# Real native Rapfi (gated)
# --------------------------------------------------------------------------
def _opening() -> GameState:
    s = GameState.initial()
    for a in [7 * 15 + 7, 7 * 15 + 8, 8 * 15 + 7]:
        s = s.apply(a)
    return s


@requires_rapfi
def test_pool_pick_returns_legal_move():
    with RapfiPool(size=2, timeout_ms=600, board_size=15) as pool:
        s = _opening()
        legal = set(int(a) for a in s.legal_actions())
        for _ in range(3):
            assert pool.pick(s) in legal


@requires_rapfi
def test_pool_label_states_parallel():
    with RapfiPool(size=3, timeout_ms=600, board_size=15) as pool:
        states = [_opening() for _ in range(6)]
        moves = pool.label_states(states)
        assert len(moves) == 6
        legal = set(int(a) for a in states[0].legal_actions())
        assert all(m in legal for m in moves)


@requires_rapfi
def test_pool_preserves_size_across_leases():
    with RapfiPool(size=2, timeout_ms=600, board_size=15) as pool:
        s = _opening()
        for _ in range(5):
            pool.pick(s)
        assert pool._pool.qsize() == 2


@requires_rapfi
def test_pool_plays_legal_block():
    with RapfiPool(size=1, timeout_ms=800, board_size=15) as pool:
        s = GameState.initial()
        for a in [7 * 15 + 7, 0, 7 * 15 + 8, 1, 7 * 15 + 9]:
            s = s.apply(a)
        move = pool.pick(s)
        assert move in set(int(a) for a in s.legal_actions())
