"""Optional native MCTS engine boundary.

The native engine owns the arena-backed MCTS tree, PUCT selection, child
creation, virtual loss, backup, and input-plane materialization. Python still
owns the PyTorch model callback at wave boundaries and the outer self-play loop.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

from gomoku.board_config import BOARD_SIZE

# Native MCTS is compiled per board size: `_mcts_native` is the 9x9 build,
# `_mcts_native11/13/15` the larger-board builds for the 9->11->13->15
# curriculum (see setup.py). Any other size falls back to the pure-Python MCTS
# (same code path as GOMOKU_DISABLE_NATIVE_MCTS=1).
try:  # pragma: no cover - import path depends on local build artifacts.
    if os.environ.get("GOMOKU_DISABLE_NATIVE_MCTS"):
        raise ImportError("native MCTS disabled by environment")
    if BOARD_SIZE == 9:
        from gomoku import _mcts_native as _native
    elif BOARD_SIZE == 11:
        from gomoku import _mcts_native11 as _native
    elif BOARD_SIZE == 13:
        from gomoku import _mcts_native13 as _native
    elif BOARD_SIZE == 15:
        from gomoku import _mcts_native15 as _native
    else:
        warnings.warn(
            f"no native MCTS build for board size {BOARD_SIZE}; "
            "using pure-Python MCTS fallback",
            RuntimeWarning,
            stacklevel=2,
        )
        raise ImportError(f"no native MCTS for board size {BOARD_SIZE}")
except ImportError:  # pragma: no cover - fallback path on source-only installs.
    _native = None

USING_NATIVE_MCTS = _native is not None
NativeMCTSGame = None if _native is None else _native.NativeMCTSGame


def search_batch(*args: Any, **kwargs: Any) -> None:
    if _native is None:
        raise RuntimeError("native MCTS extension is not available")
    _native.search_batch(*args, **kwargs)


def gumbel_search_batch(*args: Any, **kwargs: Any) -> None:
    """Wave-batched native Gumbel root selection + Sequential Halving.

    Per-game Sequential Halving runs inside the lockstep wave (all games share
    the phase index; the per-game round-robin forced visits batch their leaf
    evaluations across games, exactly like ``search_batch``). After the call,
    each game exposes ``gumbel_selected_action()`` (the SH-chosen move) and
    ``gumbel_policy(c_visit, c_scale)`` (the completed-policy training target).
    """
    if _native is None:
        raise RuntimeError("native MCTS extension is not available")
    _native.gumbel_search_batch(*args, **kwargs)


def has_native_gumbel() -> bool:
    """True if the built extension implements the native Gumbel batch path."""
    return _native is not None and hasattr(_native, "gumbel_search_batch")


def backend() -> Any | None:
    return _native
