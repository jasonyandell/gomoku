"""D4-canonical state hashing for the idx-2 mine (Bruce-Lee-one-position).

The mine stores ONE representative per D4-equivalence class and lets the trainer's
existing sample-time D4 augment (``gomoku.teacher.TeacherDataset``) recover the 8×
diversity for free. So two boards that are rotations/reflections of each other are
the SAME work item — analyzed once, stored once.

Correctness with history: the net input (``to_planes``) carries ``HISTORY_PLY``
past frames, so a canonical form must transform the board AND every history frame
under the SAME symmetry, or the recency planes end up in a different frame than
plane 0. ``transform_state`` does exactly that; ``canonical_state`` picks the
symmetry that minimizes the *current* board's bytes (a stable, content-only key),
and returns the fully-transformed state so a downstream ``analyze`` /
``to_planes`` is already in canonical frame.

Reuses ``gomoku.game._sym_board`` (the same D4 op the trainer augments with), so
the mine's canonicalization and the trainer's augmentation are guaranteed to be
the same group.
"""
from __future__ import annotations

import hashlib

import numpy as np

from gomoku.game import GameState, _sym_board


def transform_state(s: GameState, sym: int) -> GameState:
    """Apply one D4 symmetry (0..7) to a whole state: board + every history frame.

    The action frame rotates with the board, so a position reached by a move
    sequence maps to the symmetric position reached by the symmetric sequence —
    ``analyze`` on the result returns winrates already in this frame.
    """
    board = _sym_board(s.board, sym)
    history = tuple(_sym_board(h, sym) for h in s.history)
    return GameState(board=board, move_count=s.move_count, history=history)


def _board_bytes(board: np.ndarray) -> bytes:
    return np.ascontiguousarray(board).tobytes()


def canonical_state(s: GameState) -> tuple[GameState, int]:
    """Return ``(canonical_state, sym)`` — the D4 image whose *current board* has
    the lexicographically smallest bytes. Deterministic and content-only.
    """
    best_sym = 0
    best_bytes = _board_bytes(_sym_board(s.board, 0))
    for sym in range(1, 8):
        b = _board_bytes(_sym_board(s.board, sym))
        if b < best_bytes:
            best_bytes = b
            best_sym = sym
    return transform_state(s, best_sym), best_sym


def canonical_key(s: GameState) -> bytes:
    """A compact, collision-resistant dedup key for ``s``'s D4 class.

    Keyed on the canonical CURRENT board only (the analysis + soft target depend
    on the board, not the artificial BFS history) plus move_count to keep plies
    distinct. blake2b-16 → 16 bytes, fine for a set of tens of millions.
    """
    canon, _ = canonical_state(s)
    h = hashlib.blake2b(_board_bytes(canon.board), digest_size=16)
    h.update(canon.move_count.to_bytes(2, "little"))
    return h.digest()
