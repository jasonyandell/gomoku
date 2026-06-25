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
        assert prov["incremental"] is False  # default: classic BOARD-replay mode
    finally:
        p.close()


def test_incremental_flag_parsed_from_spec():
    spec = parse_spec(f"external:cmd={_stub_cmd('lowest')},timeout_ms=150,incremental=1")
    p = build_player(spec)
    try:
        assert p.config.incremental is True
        assert p.provenance()["incremental"] is True
    finally:
        p.close()
    # Absent / falsey values keep the robust default (BOARD-replay every move).
    for falsey in ("0", "false", "no"):
        s = parse_spec(f"external:cmd={_stub_cmd('lowest')},incremental={falsey}")
        q = build_player(s)
        try:
            assert q.config.incremental is False
        finally:
            q.close()


def test_can_turn_incremental_diff_logic():
    """`_can_turn` gates the incremental TURN path: only a clean continuation
    (our stones unchanged, opponent +1 stone) qualifies; everything else falls
    back to a full BOARD resync."""
    cfg = ExternalEngineConfig(cmd=_stub_cmd("lowest"), timeout_ms=200, incremental=True)
    p = ExternalEnginePlayer(cfg)
    try:
        # No prior reply yet -> must BOARD-sync first.
        assert p._can_turn({(0, 0)}, set()) is False
        p._prev_own = {(0, 0)}
        p._prev_opp = set()
        assert p._can_turn({(0, 0)}, {(5, 5)}) is True  # opponent +1, ours same
        assert p._can_turn({(0, 0), (1, 1)}, {(5, 5)}) is False  # our stones changed
        assert p._can_turn({(0, 0)}, {(5, 5), (6, 6)}) is False  # opponent +2
        assert p._can_turn({(0, 0)}, set()) is False  # not a superset (new game)
    finally:
        p.close()


def _mate_stream(*, n_pv: int, n_rounds: int) -> list[str]:
    """A synthetic multiPV iterative-deepening stream: ``n_rounds`` deepening
    rounds, each re-emitting all ``n_pv`` PV blocks (the real shape — a mate
    position deepens far and re-prints every PV per depth), then the bare-coord
    bestmove terminator. ~6 lines/PV/round, so n_pv=24, n_rounds=20 -> ~2880
    lines, which OVERFLOWS the old fixed 2000-line cap (the crash) but is well
    inside the pv-scaled cap. The deepest round's winrate is what _read_analysis
    must keep per PV root."""
    lines: list[str] = []
    for d in range(2, 2 + n_rounds):
        for pv in range(1, n_pv + 1):
            # Distinct on-board cell per PV (board-size-agnostic: BOARD_SIZE>=9
            # holds n_pv<=24 distinct cells).
            x, y = pv % BOARD_SIZE, pv // BOARD_SIZE
            wr = round(0.01 * pv, 4)  # distinct per PV; constant across rounds
            lines += [
                f"INFO PV {pv}",
                f"INFO NUMPV {n_pv}",
                f"INFO DEPTH {d}",
                f"INFO WINRATE {wr}",
                f"INFO BESTLINE {x},{y}",
                "INFO PV DONE",
            ]
    lines.append("7,7")  # the bare-coord bestmove terminator
    return lines


def test_read_analysis_survives_mate_stream_past_old_2000_cap(monkeypatch):
    """REGRESSION (#86 BFS-miner crash): a forced-mate position makes Rapfi's
    iterative deepening re-emit all PV blocks to a deep DEPTH, legitimately
    streaming >2000 lines before the bestmove terminator (measured: an 8-stone
    EVAL=-M4 board ran to DEPTH 26 -> 2302 lines). The old fixed
    _MAX_CHATTER_LINES=2000 cap tripped on this and raised
    'no bestmove terminator', crashing the mine. _read_analysis now uses a
    pv-scaled cap so the full stream parses."""
    p = ExternalEnginePlayer(ExternalEngineConfig(cmd=_stub_cmd("lowest"), timeout_ms=200))
    try:
        stream = iter(_mate_stream(n_pv=24, n_rounds=20))  # ~2880 lines > 2000
        assert len(_mate_stream(n_pv=24, n_rounds=20)) > 2000
        monkeypatch.setattr(p, "_read_line", lambda deadline: next(stream))
        out = p._read_analysis(pv_count=24)
        # All 24 PV roots scored, winrates in [0,1] from the kept (deepest) block.
        assert len(out) == 24
        for a, w in out.items():
            assert 0.0 <= w <= 1.0
    finally:
        p.close()


def test_read_analysis_still_raises_on_genuine_runaway(monkeypatch):
    """The pv-scaled cap must still backstop a truly non-terminating engine:
    an endless INFO stream (no terminator ever) still raises rather than
    looping forever."""
    def _endless(deadline):
        return "INFO PV 1"  # never a bare-coord terminator
    p = ExternalEnginePlayer(ExternalEngineConfig(cmd=_stub_cmd("lowest"), timeout_ms=200))
    try:
        monkeypatch.setattr(p, "_read_line", _endless)
        with pytest.raises(ExternalEngineError, match="without ending the multiPV"):
            p._read_analysis(pv_count=1)
    finally:
        p.close()
