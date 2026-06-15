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
    _parse_coord,
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


def test_parse_coord_takes_only_bare_coordinates():
    """The move reply is identified POSITIVELY: only a leading X,Y is a move."""
    assert _parse_coord("7,7") == (7, 7)
    assert _parse_coord("7,7 1") == (7, 7)  # swap-variant trailing tokens ok
    # Chatter that must NOT be mistaken for a move:
    assert _parse_coord("ERROR my move [7,7]") is None  # bracketed, not bare
    assert _parse_coord("MESSAGE Speed=2175") is None
    assert _parse_coord("DEBUG depth 50, nodes 0") is None
    assert _parse_coord("DATABASE hit") is None
    assert _parse_coord("?") is None
    assert _parse_coord("OK") is None
    assert _parse_coord("UNKNOWN command '7,7,2'") is None  # leading token != coord


def test_nonfatal_error_chatter_is_skipped():
    """Engines (Yixin/Pela/Eulring) emit 'ERROR my move [..]' diagnostics BEFORE
    the real move. The OLD reader hard-failed on any ERROR line; the move reader
    must skip them and read the following coordinate."""
    p = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_stub_cmd("err_chatter"), timeout_ms=200)
    )
    try:
        state = GameState.initial().apply(0)  # one stone, so we send a BOARD
        rng = np.random.default_rng(0)
        action = p(state, rng)
        assert action in set(int(a) for a in state.legal_actions())
    finally:
        p.close()


def test_boardonce_engine_survives_multi_move_via_restart():
    """Zetor2017 class: BOARD is one-shot; a second BOARD without an intervening
    RESTART desyncs. Our wrapper sends RESTART before every BOARD, so a full
    multi-move sequence must work against the 'boardonce' stub."""
    p = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_stub_cmd("boardonce"), timeout_ms=200)
    )
    try:
        rng = np.random.default_rng(0)
        state = GameState.initial().apply(0)  # non-empty -> BOARD path
        for _ in range(4):  # several moves; each re-sends a full BOARD
            action = p(state, rng)
            assert action in set(int(a) for a in state.legal_actions())
            state = state.apply(action)
            # apply an opponent move too so the board keeps growing
            opp_legal = list(int(a) for a in state.legal_actions())
            state = state.apply(opp_legal[0])
    finally:
        p.close()


def test_resign_on_empty_board_is_handled_via_begin():
    """Zetor2017 resigns on an empty BOARD/DONE but answers BEGIN. On a truly
    empty board the wrapper must send BEGIN (not an empty BOARD) and get a move."""
    p = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_stub_cmd("resign_empty"), timeout_ms=200)
    )
    try:
        state = GameState.initial()  # empty board -> BEGIN path
        rng = np.random.default_rng(0)
        action = p(state, rng)
        assert action == 0  # lowest empty cell on an empty board
        assert action in set(int(a) for a in state.legal_actions())
    finally:
        p.close()


def test_read_deadline_scales_with_timeout_budget():
    """A long per-move budget must lift the read deadline above the static floor,
    so a slow engine under a generous timeout_turn is not cut off early."""
    cfg = ExternalEngineConfig(
        cmd=_stub_cmd("lowest"), timeout_ms=20000, read_timeout_s=30.0,
        read_timeout_slack=3.0,
    )
    p = ExternalEnginePlayer(cfg)
    try:
        # 20000ms * 3 = 60s > 30s floor.
        assert p._read_deadline_s() == pytest.approx(60.0)
    finally:
        p.close()
    # Short budget keeps the floor.
    cfg2 = ExternalEngineConfig(
        cmd=_stub_cmd("lowest"), timeout_ms=200, read_timeout_s=30.0,
        read_timeout_slack=3.0,
    )
    p2 = ExternalEnginePlayer(cfg2)
    try:
        assert p2._read_deadline_s() == pytest.approx(30.0)
    finally:
        p2.close()


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
