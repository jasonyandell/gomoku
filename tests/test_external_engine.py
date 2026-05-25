"""Tests for the Piskvork/Gomocup external-engine wrapper.

These do NOT require the real Rapfi binary. We ship a tiny fake-protocol
engine (`tests/fake_piskvork_engine.py`) that speaks just enough of the
Gomocup protocol to exercise the wrapper: START/INFO handshake, BOARD/DONE
position input, and an X,Y move reply (lowest-index empty cell). Behaviour
is selectable via argv so we can also exercise the failure paths
(unsupported board size, illegal move reply).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from gomoku.eval import play_match_pickers
from gomoku.external_engine import (
    ExternalEngineConfig,
    ExternalEngineError,
    ExternalEnginePlayer,
    _coord_to_xy,
    _xy_to_action,
)
from gomoku.game import BOARD_SIZE, GameState
from gomoku.match import build_player, parse_spec

_STUB = os.path.join(os.path.dirname(__file__), "fake_piskvork_engine.py")


def _stub_cmd(mode: str = "lowest") -> str:
    return f"{sys.executable} {_STUB} {mode}"


def test_coord_roundtrip():
    for action in range(BOARD_SIZE * BOARD_SIZE):
        x, y = _coord_to_xy(action)
        assert 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE
        assert _xy_to_action(x, y) == action


def test_xy_mapping_is_x_col_y_row():
    # action for row=2, col=3 should map to X=3, Y=2
    action = 2 * BOARD_SIZE + 3
    assert _coord_to_xy(action) == (3, 2)
    assert _xy_to_action(3, 2) == action


def test_handshake_and_first_move():
    p = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_stub_cmd("lowest"), timeout_ms=200, label="stub")
    )
    try:
        state = GameState.initial()
        rng = np.random.default_rng(0)
        action = p(state, rng)
        # "lowest" stub plays the lowest-index empty cell -> action 0 on empty board.
        assert action == 0
        assert action in set(int(a) for a in state.legal_actions())
    finally:
        p.close()


def test_field_encoding_skips_occupied_and_returns_legal():
    """After applying a move, the engine must return a legal (empty) cell."""
    p = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_stub_cmd("lowest"), timeout_ms=200, label="stub")
    )
    try:
        state = GameState.initial()
        rng = np.random.default_rng(0)
        a0 = p(state, rng)          # picks 0
        state = state.apply(a0)     # now cell 0 occupied, perspective flipped
        a1 = p(state, rng)          # stub sees one occupied cell, picks lowest empty
        legal = set(int(a) for a in state.legal_actions())
        assert a1 in legal
        assert a1 != 0  # cell 0 is occupied now
    finally:
        p.close()


def test_unsupported_board_size_is_hard_error():
    """A stub that ERRORs on START must surface as ExternalEngineError."""
    with pytest.raises(ExternalEngineError):
        ExternalEnginePlayer(
            ExternalEngineConfig(cmd=_stub_cmd("reject_start"), timeout_ms=200)
        )


def test_illegal_move_reply_is_rejected():
    """If the engine returns an occupied cell, we must raise. The 'illegal'
    stub always replies 0,0; we occupy action 0 first so that reply collides
    with an existing stone."""
    p = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_stub_cmd("illegal"), timeout_ms=200)
    )
    try:
        # Occupy cell 0,0 (action 0) so the stub's "0,0" reply is illegal.
        state = GameState.initial().apply(0)
        rng = np.random.default_rng(0)
        with pytest.raises(ExternalEngineError):
            p(state, rng)
    finally:
        p.close()


def test_chatter_lines_are_skipped():
    """MESSAGE/DEBUG lines before the move must not break parsing."""
    p = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_stub_cmd("chatter"), timeout_ms=200)
    )
    try:
        state = GameState.initial()
        rng = np.random.default_rng(0)
        action = p(state, rng)
        assert action in set(int(a) for a in state.legal_actions())
    finally:
        p.close()


def test_full_game_via_play_match_pickers():
    """The wrapper plugs into the generic match engine and a full game runs
    to a terminal state. Stub vs stub (both play lowest-index empty cell) ->
    deterministic, terminates, counts add up."""
    a = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_stub_cmd("lowest"), timeout_ms=200, label="A")
    )
    b = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_stub_cmd("lowest"), timeout_ms=200, label="B")
    )
    try:
        res = play_match_pickers(a, b, n_games=2, seed=0)
        assert res.n_games == 2
        assert res.wins + res.losses + res.draws == 2
    finally:
        a.close()
        b.close()


def test_match_spec_builds_external_player():
    spec = parse_spec(f"external:cmd={_stub_cmd('lowest')},timeout_ms=150,label=stub")
    assert spec.kind == "external"
    p = build_player(spec)
    try:
        assert isinstance(p, ExternalEnginePlayer)
        prov = p.provenance()
        assert prov["engine"] == "stub"
        assert prov["timeout_ms"] == 150
        assert prov["board_size"] == BOARD_SIZE
        assert prov["rule"] == 0
    finally:
        p.close()
