"""Tests for the brain-side Gomocup-protocol wrapper (`gomoku.gomocup_brain`).

Two layers:
- Protocol-level: drive the pure `GomocupBrain` line-handler with a stub picker
  (lowest-legal cell) and a list-appending `emit`. No torch, no subprocess —
  fast and deterministic. Mirrors `tests/test_external_engine.py` in spirit.
- Closed-loop: drive the REAL `python -m gomoku.gomocup_brain --stub-picker`
  subprocess WITH `external_engine.py` (our client half). This is the keystone:
  it proves both protocol halves interoperate over real pipes (incl. flushing),
  which is exactly the round-trip the panel derby (#30) relies on.
"""

from __future__ import annotations

import sys

from gomoku.eval import play_match_pickers
from gomoku.external_engine import ExternalEngineConfig, ExternalEnginePlayer
from gomoku.game import BOARD_SIZE
from gomoku.gomocup_brain import (
    GomocupBrain,
    _first_legal_pick,
    _parse_xy,
)


def _make_brain(pick=None, name="gomoku-az"):
    out: list[str] = []
    brain = GomocupBrain(
        pick or _first_legal_pick, board_size=BOARD_SIZE, name=name, emit=out.append
    )
    return brain, out


def _is_coord(line: str) -> bool:
    tok = line.split()[0] if line.split() else ""
    return _parse_xy(tok, BOARD_SIZE) is not None


# -- handshake -----------------------------------------------------------


def test_handshake_ok_and_info_is_silent():
    brain, out = _make_brain()
    assert brain.handle(f"START {BOARD_SIZE}") is True
    assert out == ["OK"]
    out.clear()
    brain.handle("INFO rule 0")
    brain.handle("INFO timeout_turn 5000")
    brain.handle("INFO timeout_match 0")
    assert out == []  # INFO never replies
    assert brain.info["rule"] == "0"
    assert brain.info["timeout_turn"] == "5000"


def test_start_size_mismatch_is_error():
    brain, out = _make_brain()
    brain.handle(f"START {BOARD_SIZE + 1}")
    assert len(out) == 1 and out[0].startswith("ERROR")


# -- move elicitation ----------------------------------------------------


def test_begin_opening_is_single_legal_coord():
    brain, out = _make_brain()
    brain.handle(f"START {BOARD_SIZE}")
    out.clear()
    brain.handle("BEGIN")
    assert len(out) == 1
    # stub plays lowest-index legal cell -> action 0 -> (0,0) on an empty board
    assert out[0] == "0,0"
    assert _is_coord(out[0])


def test_turn_sequence_never_replays_occupied_cell():
    brain, out = _make_brain()
    brain.handle(f"START {BOARD_SIZE}")
    out.clear()
    brain.handle("BEGIN")  # our move at (0,0)
    our0 = out[-1]
    brain.handle("TURN 5,5")  # opponent plays (5,5)
    our1 = out[-1]
    assert _is_coord(our1)
    assert our1 not in (our0, "5,5")  # not our own nor opponent's stone
    # one more ply stays legal
    brain.handle("TURN 6,6")
    assert _is_coord(out[-1])
    assert out[-1] not in (our0, our1, "5,5", "6,6")


def test_board_rebuild_is_reentrant_and_legal():
    brain, out = _make_brain()
    brain.handle(f"START {BOARD_SIZE}")
    out.clear()
    # field 1 = own (us), field 2 = opponent
    brain.handle("BOARD")
    brain.handle("0,0,2")  # opponent at (0,0)
    brain.handle("1,1,1")  # us at (1,1)
    brain.handle("DONE")
    assert len(out) == 1
    assert _is_coord(out[0])
    # (0,0) occupied by opp -> lowest legal is action 1 = (x=1,y=0) -> "1,0"
    assert out[0] == "1,0"

    # Re-BOARD WITHOUT a RESTART (the ai_match oracle's process-reuse path):
    out.clear()
    brain.handle("BOARD")
    brain.handle("7,7,2")
    brain.handle("DONE")
    assert len(out) == 1
    # only (7,7) occupied -> lowest legal is action 0 = (0,0) -> "0,0"
    assert out[0] == "0,0"


def test_board_skips_out_of_range_and_marker_fields():
    brain, out = _make_brain()
    brain.handle(f"START {BOARD_SIZE}")
    out.clear()
    brain.handle("BOARD")
    brain.handle(f"{BOARD_SIZE + 9},0,1")  # out of range -> skipped
    brain.handle("3,3,3")  # renju marker field -> not a stone, skipped
    brain.handle("0,0,2")  # real opponent stone
    brain.handle("DONE")
    assert len(out) == 1 and _is_coord(out[0])


# -- session lifecycle ---------------------------------------------------


def test_restart_clears_the_board():
    brain, out = _make_brain()
    brain.handle(f"START {BOARD_SIZE}")
    brain.handle("BEGIN")
    out.clear()
    brain.handle("RESTART")
    assert out == ["OK"]
    assert brain.state.move_count == 0
    out.clear()
    brain.handle("BEGIN")
    assert out[-1] == "0,0"  # fresh opening again


def test_about_identity_is_single_line():
    brain, out = _make_brain(name="gomoku-az")
    brain.handle("ABOUT")
    assert len(out) == 1
    assert 'name="gomoku-az"' in out[0]


def test_end_signals_exit():
    brain, _ = _make_brain()
    assert brain.handle("END") is False


# -- robustness ----------------------------------------------------------


def test_unknown_commands_never_emit_a_bare_coordinate():
    brain, out = _make_brain()
    brain.handle(f"START {BOARD_SIZE}")
    out.clear()
    brain.handle("FOOBAR baz")
    brain.handle("")  # empty line ignored
    brain.handle("RECTSTART 5,5")
    brain.handle("SWAP2BOARD")
    # A stray bare X,Y on stdout would be eaten as our move — must never happen.
    for line in out:
        assert not _is_coord(line)


def test_malformed_turn_is_safe_and_recoverable():
    brain, out = _make_brain()
    brain.handle(f"START {BOARD_SIZE}")
    out.clear()
    brain.handle("TURN garbage")
    for line in out:
        assert not _is_coord(line)  # no move on a malformed TURN
    out.clear()
    brain.handle("TURN 3,3")  # a valid TURN still works afterward
    assert len(out) == 1 and _is_coord(out[0])


def test_crlf_line_endings_tolerated():
    brain, out = _make_brain()
    brain.handle(f"START {BOARD_SIZE}\r\n")
    assert out == ["OK"]


def test_parse_xy_bounds():
    assert _parse_xy("3,2", BOARD_SIZE) == (3, 2)
    assert _parse_xy("0,0", BOARD_SIZE) == (0, 0)
    assert _parse_xy("nope", BOARD_SIZE) is None
    assert _parse_xy(f"{BOARD_SIZE},0", BOARD_SIZE) is None  # out of range
    assert _parse_xy("-1,0", BOARD_SIZE) is None


# -- closed loop (subprocess): our CLIENT drives our BRAIN ----------------


def _brain_cmd() -> str:
    # --stub-picker => no checkpoint/torch; pure protocol over real pipes.
    return f"{sys.executable} -m gomoku.gomocup_brain --stub-picker"


def test_closed_loop_client_drives_brain_subprocess():
    """`external_engine.py` (client) drives `gomocup_brain.py` (brain) over real
    stdin/stdout pipes for a full game. Proves both protocol halves interoperate
    AND that every reply is flushed (an unflushed move would hang the reader)."""
    a = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_brain_cmd(), timeout_ms=3000, label="brain-A")
    )
    b = ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_brain_cmd(), timeout_ms=3000, label="brain-B")
    )
    try:
        res = play_match_pickers(a, b, n_games=2, seed=0)
        assert res.n_games == 2
        assert res.wins + res.losses + res.draws == 2
    finally:
        a.close()
        b.close()
