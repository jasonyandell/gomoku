"""Bit-packing helpers for the binary occupancy planes stored in the replay buffer.

The AlphaZero input planes (``State.to_planes`` → shape ``(N_INPUT_PLANES, N, N)``)
are *binary*: every stone plane is 0/1 occupancy and the final plane is the
constant ``1.0``. Stored as float32 they cost ``4 * N_INPUT_PLANES * N * N``
bytes/position — the dominant term in the buffer (94% at 15x15). Since the data
is binary we can pack 8 cells into one ``uint8`` and unpack on the sample hot
path, cutting that term ~32x.

This module is pure-numpy and board-size agnostic (it takes the flat element
count). It is only exercised when the opt-in ``pack_planes`` storage mode is on;
the default float32 path never imports the round-trip.

Round-trip contract: ``unpack_planes(pack_planes(x, ...), ...) == x`` *exactly*
for any array whose values are all 0.0 or 1.0. Inputs are asserted binary so a
non-binary plane (which would silently lose information) fails loudly rather
than corrupting a training target.
"""

from __future__ import annotations

import numpy as np

__all__ = ["packed_bytes_per_position", "pack_planes", "unpack_planes"]


def packed_bytes_per_position(n_elems: int) -> int:
    """Bytes needed to bit-pack ``n_elems`` binary values (8 per byte)."""
    return (int(n_elems) + 7) // 8


def pack_planes(planes: np.ndarray, n_elems: int) -> np.ndarray:
    """Bit-pack a batch of binary planes.

    ``planes`` is ``(B, ...)`` (any trailing shape) whose flattened per-row
    length is ``n_elems``; values must all be 0 or 1. Returns ``uint8`` of shape
    ``(B, packed_bytes_per_position(n_elems))``. ``np.packbits`` pads the final
    byte with zero bits, which ``unpack_planes`` discards.
    """
    arr = np.ascontiguousarray(planes)
    flat = arr.reshape(arr.shape[0], -1)
    if flat.shape[1] != n_elems:
        raise ValueError(
            f"pack_planes: expected {n_elems} elements per row, got {flat.shape[1]}"
        )
    # Assert binary: a non-{0,1} plane cannot survive a bit-pack and must fail
    # loudly rather than silently corrupt a stored target.
    if not np.array_equal(flat, flat.astype(bool)):
        raise ValueError("pack_planes: planes must be binary (0/1) to bit-pack")
    bits = flat.astype(np.uint8)
    return np.packbits(bits, axis=1)


def unpack_planes(packed: np.ndarray, n_elems: int, shape: tuple[int, ...]) -> np.ndarray:
    """Inverse of :func:`pack_planes`.

    ``packed`` is ``(B, packed_bytes)`` ``uint8``; returns ``float32`` of shape
    ``(B, *shape)`` where ``prod(shape) == n_elems``. The pad bits added by
    ``np.packbits`` are dropped by slicing to ``n_elems``.
    """
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    bits = np.unpackbits(packed, axis=1)[:, :n_elems]
    out = bits.astype(np.float32)
    return out.reshape(packed.shape[0], *shape)
