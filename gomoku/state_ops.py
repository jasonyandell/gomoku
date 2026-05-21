"""Hot-path state operations for Gomoku.

This module is the boundary for state operations that are small but called very
often during MCTS: legal masks, terminal checks, and apply-related board copies.
It is intentionally pure-Python/NumPy today, with an optional native import slot
so a future C/Cython/Rust helper can replace the internals without changing
`GameState` or MCTS call sites.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

BOARD_SIZE = 9
N_ACTIONS = BOARD_SIZE * BOARD_SIZE
WIN_LEN = 5
HISTORY_PLY = 8

try:  # pragma: no cover - exercised only when an optional extension exists.
    if os.environ.get("GOMOKU_DISABLE_NATIVE_STATE_OPS"):
        raise ImportError("native state ops disabled by environment")
    from gomoku import _state_ops_native as _native
except ImportError:  # pragma: no cover - normal path when extension is absent.
    _native = None

USING_NATIVE = _native is not None


def legal_mask(board: np.ndarray) -> np.ndarray:
    """Return a flat bool mask where True means the action is legal."""
    if _native is not None and hasattr(_native, "legal_mask"):
        return np.asarray(_native.legal_mask(board), dtype=bool)
    occupied = board[0] | board[1]
    return ~occupied.reshape(-1)


def terminal_value(board: np.ndarray, move_count: int) -> tuple[bool, float]:
    """Return terminal status and value from side-to-move perspective."""
    if _native is not None and hasattr(_native, "terminal_value"):
        done, value = _native.terminal_value(board, move_count)
        return bool(done), float(value)
    if has_five_in_a_row(board[1]):
        return True, -1.0
    if move_count >= N_ACTIONS:
        return True, 0.0
    return False, 0.0


def init_node_status(
    board: np.ndarray,
    move_count: int,
) -> tuple[bool, float, np.ndarray | None]:
    """Return terminal info plus the legal mask needed to initialize an MCTS node."""
    if _native is not None and hasattr(_native, "init_node_status"):
        done, value, mask = _native.init_node_status(board, move_count)
        mask_array = None if mask is None else np.asarray(mask, dtype=bool)
        return bool(done), float(value), mask_array

    done, value = terminal_value(board, move_count)
    if done:
        return done, value, None
    return False, 0.0, legal_mask(board)


def apply_move_arrays(
    board: np.ndarray,
    history: tuple[np.ndarray, ...],
    action: int,
    *,
    history_ply: int = HISTORY_PLY,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Apply one action to canonical arrays and return flipped board/history.

    The returned board is copied and perspective-flipped. The history receives a
    snapshot of the pre-move canonical board, matching `GameState.apply`.
    """
    if _native is not None and hasattr(_native, "apply_move_arrays"):
        new_board, new_history = _native.apply_move_arrays(
            board, history, int(action), history_ply=history_ply
        )
        return np.asarray(new_board, dtype=bool), tuple(
            np.asarray(item, dtype=bool) for item in new_history
        )

    row, col = divmod(int(action), BOARD_SIZE)
    if board[0, row, col] or board[1, row, col]:
        raise ValueError(f"illegal move {action} on occupied square")

    snapshot = board.copy()
    new_board = board.copy()
    new_board[0, row, col] = True
    new_board = new_board[::-1].copy()
    new_history = (snapshot,) + history[: history_ply - 1]
    return new_board, new_history


def has_five_in_a_row(plane: np.ndarray) -> bool:
    """Return True if a 9x9 bool plane has any 5-in-a-row."""
    if _native is not None and hasattr(_native, "has_five_in_a_row"):
        return bool(_native.has_five_in_a_row(plane))
    return _has_five_in_a_row_numpy(plane)


def _has_five_in_a_row_numpy(plane: np.ndarray) -> bool:
    p = plane
    h = p[:, :-4] & p[:, 1:-3] & p[:, 2:-2] & p[:, 3:-1] & p[:, 4:]
    if h.any():
        return True
    v = p[:-4, :] & p[1:-3, :] & p[2:-2, :] & p[3:-1, :] & p[4:, :]
    if v.any():
        return True
    d = p[:-4, :-4] & p[1:-3, 1:-3] & p[2:-2, 2:-2] & p[3:-1, 3:-1] & p[4:, 4:]
    if d.any():
        return True
    a = p[:-4, 4:] & p[1:-3, 3:-1] & p[2:-2, 2:-2] & p[3:-1, 1:-3] & p[4:, :-4]
    return bool(a.any())


def native_backend() -> Any | None:
    """Expose the optional native module for diagnostics without importing it elsewhere."""
    return _native
