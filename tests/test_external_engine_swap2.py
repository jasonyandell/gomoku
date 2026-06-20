"""Tests for the SWAP2BOARD swap2-negotiation path of the external-engine wrapper.

These do NOT require the real Rapfi binary. We reuse the shipped fake-protocol
engine (`tests/fake_piskvork_engine.py`), which now speaks the SWAP2BOARD block
too (swap2_* modes), launched via `ExternalEngineConfig(cmd=...)` exactly like
`test_external_engine.py`. They pin the wire grammar:

    0 stones in (engine OPENS)    -> three coords            -> OPEN_THREE
    3 stones in (engine RESPONDS) -> SWAP | 1 coord | 2 coords
    5 stones in (engine PICKS)    -> SWAP | 1 coord

plus arity validation, chatter-skipping, and that the `SWAP` literal is never
mis-parsed as a coordinate. Coordinate<->action mapping matches the move path
(action = row*BOARD_SIZE + col; X=col, Y=row).
"""

from __future__ import annotations

import os
import sys

import pytest

from gomoku.external_engine import (
    ExternalEngineConfig,
    ExternalEngineError,
    ExternalEnginePlayer,
    Swap2Option,
    Swap2Reply,
    _parse_coord,
    _parse_coord_list,
    _xy_to_action,
)
from gomoku.game import BOARD_SIZE

_STUB = os.path.join(os.path.dirname(__file__), "fake_piskvork_engine.py")


def _stub_cmd(mode: str) -> str:
    return f"{sys.executable} {_STUB} {mode}"


def _player(mode: str) -> ExternalEnginePlayer:
    return ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_stub_cmd(mode), timeout_ms=200, label="stub")
    )


# -- pure parsing layer -------------------------------------------------------

def test_swap_literal_is_not_a_coordinate():
    """`SWAP` has no comma, so the positive-match coord gate rejects it as a
    coord (it is handled separately by the swap2 reader)."""
    assert _parse_coord("SWAP") is None
    assert _parse_coord_list("SWAP") is None
    assert _parse_coord_list("DONE") is None


def test_parse_coord_list_accepts_n_coords_rejects_mixed():
    assert _parse_coord_list("7,7") == [(7, 7)]
    assert _parse_coord_list("7,7 8,7") == [(7, 7), (8, 7)]
    assert _parse_coord_list("7,7 8,7 9,8") == [(7, 7), (8, 7), (9, 8)]
    # One non-coord token disqualifies the whole line (never half-accepted).
    assert _parse_coord_list("7,7 SWAP") is None
    assert _parse_coord_list("MESSAGE Speed=2175") is None
    assert _parse_coord_list("") is None


def test_swap2reply_actions_match_xy_mapping():
    r = Swap2Reply(option=Swap2Option.OPEN_THREE, coords=((3, 2), (0, 0), (8, 8)))
    assert r.actions == (
        _xy_to_action(3, 2),
        _xy_to_action(0, 0),
        _xy_to_action(8, 8),
    )
    # row=2,col=3 -> action 2*N+3, X=col=3, Y=row=2
    assert r.actions[0] == 2 * BOARD_SIZE + 3
    assert not r.is_swap
    assert Swap2Reply(option=Swap2Option.SWAP).is_swap


# -- engine as OPENER (0 stones -> 3 coords) ----------------------------------

def test_engine_opener_returns_three_coords():
    p = _player("swap2_open")
    try:
        reply = p.swap2_open()
        assert reply.option is Swap2Option.OPEN_THREE
        assert len(reply.coords) == 3
        assert len(reply.actions) == 3
        # All distinct, in range, mapping back to valid actions.
        assert len(set(reply.actions)) == 3
        for a in reply.actions:
            assert 0 <= a < BOARD_SIZE * BOARD_SIZE
    finally:
        p.close()


# -- engine as RESPONDER (3 stones -> SWAP | 1 | 2) ---------------------------

_THREE_STONES = [(4, 4), (4, 5), (5, 5)]  # 2B+1W by order, board order


def test_engine_responder_swap():
    p = _player("swap2_swap")
    try:
        reply = p.swap2_respond(_THREE_STONES)
        assert reply.option is Swap2Option.SWAP
        assert reply.is_swap
        assert reply.coords == ()
        assert reply.actions == ()
    finally:
        p.close()


def test_engine_responder_one_coord():
    p = _player("swap2_one")
    try:
        reply = p.swap2_respond(_THREE_STONES)
        assert reply.option is Swap2Option.ONE_COORD
        assert len(reply.coords) == 1
        assert len(reply.actions) == 1
    finally:
        p.close()


def test_engine_responder_two_coords():
    p = _player("swap2_two")
    try:
        reply = p.swap2_respond(_THREE_STONES)
        assert reply.option is Swap2Option.TWO_COORDS
        assert len(reply.coords) == 2
        assert len(reply.actions) == 2
        assert reply.actions[0] != reply.actions[1]
    finally:
        p.close()


def test_engine_responder_requires_three_stones():
    p = _player("swap2_one")
    try:
        with pytest.raises(ExternalEngineError):
            p.swap2_respond([(0, 0), (1, 1)])  # wrong count
    finally:
        p.close()


# -- engine as PICKER (5 stones -> SWAP | 1) ----------------------------------

_FIVE_STONES = [(4, 4), (4, 5), (5, 5), (6, 6), (6, 7)]  # 3B+2W by order


def test_engine_picker_swap():
    p = _player("swap2_swap")
    try:
        reply = p.swap2_pick(_FIVE_STONES)
        assert reply.option is Swap2Option.SWAP
        assert reply.is_swap
    finally:
        p.close()


def test_engine_picker_one_coord():
    p = _player("swap2_one")
    try:
        reply = p.swap2_pick(_FIVE_STONES)
        assert reply.option is Swap2Option.ONE_COORD
        assert len(reply.coords) == 1
    finally:
        p.close()


def test_engine_picker_requires_five_stones():
    p = _player("swap2_one")
    try:
        with pytest.raises(ExternalEngineError):
            p.swap2_pick(_THREE_STONES)  # only 3, not 5
    finally:
        p.close()


# -- arity validation ---------------------------------------------------------

def test_opener_one_coord_is_arity_mismatch():
    """An opener (0 stones in) MUST return 3 coords; a single coord is an arity
    error, surfaced as ExternalEngineError."""
    p = _player("swap2_badarity")
    try:
        with pytest.raises(ExternalEngineError):
            p.swap2_open()
    finally:
        p.close()


def test_responder_two_coord_is_legal_but_picker_two_coord_is_arity_mismatch():
    """At 5 stones (picker) only 1 coord or SWAP is legal; 2 coords is an arity
    mismatch. The swap2_two stub always returns two coords."""
    p = _player("swap2_two")
    try:
        # responder (3 stones) -> 2 coords is legal
        ok = p.swap2_respond(_THREE_STONES)
        assert ok.option is Swap2Option.TWO_COORDS
        # picker (5 stones) -> 2 coords is NOT legal
        with pytest.raises(ExternalEngineError):
            p.swap2_pick(_FIVE_STONES)
    finally:
        p.close()


def test_swap_literal_rejected_when_opener():
    """The `SWAP` literal is only legal at 3 / 5 stones; at 0 stones (opener) it
    must raise rather than be accepted."""
    p = _player("swap2_swap")
    try:
        with pytest.raises(ExternalEngineError):
            p.swap2_open()
    finally:
        p.close()


def test_read_swap2_reply_unsupported_stone_count_raises():
    p = _player("swap2_one")
    try:
        with pytest.raises(ExternalEngineError):
            p._read_swap2_reply(4)  # 4 is not a valid SWAP2BOARD stone count
    finally:
        p.close()


# -- chatter skipping ---------------------------------------------------------

def test_chatter_before_swap2_reply_is_skipped():
    """MESSAGE/DEBUG/`?` lines before the reply must not break parsing — the
    first qualifying line (coords or SWAP) is the reply."""
    p = _player("swap2_chatter")
    try:
        reply = p.swap2_respond(_THREE_STONES)
        assert reply.option is Swap2Option.ONE_COORD
        assert len(reply.coords) == 1
    finally:
        p.close()


# -- a full negotiation against one engine, both roles ------------------------

def test_engine_drives_open_then_responder_query_in_one_session():
    """The same engine process can open AND answer a responder query — exercises
    that RESTART makes each SWAP2BOARD block a fresh re-entrant initialisation."""
    p = _player("swap2_open")
    try:
        opened = p.swap2_open()
        assert opened.option is Swap2Option.OPEN_THREE
        # now query it as a responder with our own 3 stones
        resp = p.swap2_respond(_THREE_STONES)
        # swap2_open mode replies a single coord when stones are present
        assert resp.option is Swap2Option.ONE_COORD
    finally:
        p.close()
