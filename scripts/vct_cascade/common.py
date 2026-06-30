"""Shared helpers for the VCT cascade labeler (issue #97).

Content-addressing, board (de)serialization, and atomic Parquet writes — the
primitives the extract + cascade stages share. Everything here is corpus-agnostic
and pure-ish; the only global is the board size (``GOMOKU_BOARD_SIZE``, must be 15
for the rapfi corpus).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gomoku.board_config import BOARD_SIZE as N
from gomoku.game import _sym_board

NN = N * N
_NSYM = 8  # D4: 4 rotations x 2 reflections


# --------------------------------------------------------------------------
# Content addressing — D4-canonical, color-fixed (board[0] is always the
# side-to-move/attacker, so two boards with the same stones AND same side-to-move
# collapse to one id regardless of which game/ply produced them).
# --------------------------------------------------------------------------
def canonical_id(board: np.ndarray) -> bytes:
    """blake2b-16 over the lexicographically-smallest D4 image of a (2,N,N) bool
    board. Move-count-free: the VCT verdict depends only on the stones + whose
    turn it is, both of which the (attacker-relative) two-plane board already
    encodes. Identical-under-symmetry positions => identical id."""
    best = min(_sym_board(board, k).tobytes() for k in range(_NSYM))
    return hashlib.blake2b(best, digest_size=16).digest()


def occupied(plane: np.ndarray) -> list[int]:
    """Flat row-major indices of the set cells of one (N,N) plane."""
    return [int(i) for i in np.flatnonzero(plane.reshape(-1))]


def boards_from_lists(atk, dfd, B: int) -> np.ndarray:
    """Vectorized rebuild of a (B,2,N,N) bool stack from two Arrow list<uint8>
    columns (attacker cells in plane 0, defender in plane 1) — the exact frame
    ``solve_vct_mega_bb`` eats (board[0]=attacker, no swap)."""
    flat = np.zeros((B, 2, NN), dtype=bool)
    for plane, col in ((0, atk), (1, dfd)):
        lengths = np.asarray(col.value_lengths(), dtype=np.int64)
        values = np.asarray(col.values.to_numpy(zero_copy_only=False), dtype=np.int64)
        board_ids = np.repeat(np.arange(B, dtype=np.int64), lengths)
        flat[board_ids, plane, values] = True
    return flat.reshape(B, 2, N, N)


# --------------------------------------------------------------------------
# Atomic Parquet write (temp + os.replace) — a reader never sees a torn shard,
# so the set of finished files is a sound durable log / resume marker.
# --------------------------------------------------------------------------
def write_parquet_atomic(table: pa.Table, path: str, compression: str = "zstd") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression=compression)
    os.replace(tmp, path)


def words_to_list(words: np.ndarray) -> list[list[int]]:
    """(B,4) uint64 proof-mask -> list of 4 python ints per row (Arrow-friendly)."""
    return [[int(x) for x in row] for row in words]
