"""Smoke test for the NATIVE (non-wine) Rapfi-NNUE engine (issues #40/#28/#35).

Unlike ``test_external_engine.py`` (which uses a fake-protocol stub), this test
drives the REAL ``engines/rapfi/pbrain-rapfi`` binary with its NNUE ``config.toml``
through our own ``ExternalEnginePlayer`` client — the exact path the engine panel
uses. It SKIPS cleanly when the local build artifact is absent (the binary +
NNUE weights are gitignored; build with ``engines/rapfi/build_rapfi.sh``), so CI
without the artifact stays green while a machine that has it gets real coverage.

What it guards:
- the native binary handshakes (`START 15` -> `OK`) and returns a LEGAL move
  through our client;
- coordinates are NOT mirrored — given an opponent four with one end already
  blocked, Rapfi plays the unique saving block at the correct cell (the #28
  config wires `coord_conversion_mode` that must round-trip consistently).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from gomoku.external_engine import ExternalEnginePlayer, ExternalEngineConfig
from gomoku.game import BOARD_SIZE, GameState

_REPO = os.environ.get("GOMOKU_REPO", os.path.expanduser("~/code/gomoku"))
_RAPFI_DIR = os.path.join(_REPO, "engines", "rapfi")
_BIN = os.path.join(_RAPFI_DIR, "pbrain-rapfi")
_CFG = os.path.join(_RAPFI_DIR, "config.toml")
_FREESTYLE_W = os.path.join(_RAPFI_DIR, "mix9svqfreestyle_bsmix.bin.lz4")

_have_rapfi = os.path.isfile(_BIN) and os.path.isfile(_CFG) and os.path.isfile(_FREESTYLE_W)
pytestmark = [
    pytest.mark.skipif(not _have_rapfi, reason="native Rapfi-NNUE artifact absent (build_rapfi.sh)"),
    pytest.mark.skipif(BOARD_SIZE != 15, reason="NNUE weights are 15x15; set GOMOKU_BOARD_SIZE=15"),
]

_CMD = f"{_BIN} --config {_CFG} gomocup"


def _player(timeout_ms: int = 800) -> ExternalEnginePlayer:
    return ExternalEnginePlayer(
        ExternalEngineConfig(cmd=_CMD, timeout_ms=timeout_ms, board_size=15, label="rapfi")
    )


def test_native_rapfi_handshake_and_legal_move():
    p = _player()
    try:
        st = GameState.initial()
        for a in [7 * 15 + 7, 7 * 15 + 8, 8 * 15 + 7, 6 * 15 + 8]:
            st = st.apply(a)
        action = p(st, np.random.default_rng(0))
        assert action in set(int(a) for a in st.legal_actions())
    finally:
        p.close()


def test_native_rapfi_plays_the_forced_block_unmirrored():
    """Drive the raw Piskvork protocol with a crafted position: opponent (field 2)
    has a four at row 4, cols 4-7, with the left end (col 3) already blocked by
    own (field 1). The unique saving move is the RIGHT end -> X=8, Y=4 -> "8,4".
    A mirrored-coord bug (`coord_conversion_mode` not round-tripping) would emit a
    different cell, so this pins orientation directly at the binary boundary."""
    import subprocess

    board = "\n".join([
        "START 15",
        "INFO timeout_turn 1000",
        "BOARD",
        "4,4,2", "5,4,2", "6,4,2", "7,4,2",  # opponent four
        "3,4,1", "3,10,1",                    # own stones (left end blocked)
        "DONE",
    ]) + "\n"
    out = subprocess.run(
        [_BIN, "--config", _CFG, "gomocup"],
        input=board, capture_output=True, text=True, timeout=30,
    ).stdout
    # last non-MESSAGE line is the chosen move "X,Y"
    move = [ln.strip() for ln in out.splitlines() if "," in ln and not ln.startswith("MESSAGE")][-1]
    assert move == "8,4", f"expected forced block 8,4, got {move!r} (coord mirror?)"
