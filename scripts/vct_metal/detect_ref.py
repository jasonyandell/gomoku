"""Batched, whole-board threat detection in numpy — the GPU-kernel SPEC.

Every function operates on a *batch* of boards with NO per-cell Python loop:
boards are ``(B, N, N)`` bool planes, all work is whole-board mask algebra
(shifts, logical ops, integer accumulation). This is deliberately the exact
shape an MSL kernel wants — so this module doubles as (a) the reference the
future Metal kernel is validated against and (b) the place we get the batched
formulation provably right against the scalar CPU oracle ``gomoku.vcf`` first.

Conventions match ``gomoku.vcf`` exactly:
  * ``own`` / ``opp`` are ``(B, N, N)`` bool; ``own`` is the side to move (attacker).
  * a *five* is ``>= WIN_LEN`` in a row (freestyle: overlines win).
  * a *four-move* ``m`` is an empty cell where placing ``own`` creates >=1
    five-completion ``c`` (``n_comp`` counts the distinct completions; ``>=2`` is
    a double/open four = win-next-move).
  * candidates are empty cells within Chebyshev distance 2 of any stone.
"""
from __future__ import annotations

import numpy as np

from gomoku import state_ops

N = state_ops.BOARD_SIZE
WIN = state_ops.WIN_LEN
# 4 axes; both ray directions are covered by scanning +/- along each.
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


# --------------------------------------------------------------------------- #
# Shift primitive: nb(x, dr, dc)[b, r, c] = x[b, r + dr, c + dc]  (0 off-board)
# --------------------------------------------------------------------------- #
def nb(x: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Neighbour-gather: the value of the cell offset by (+dr, +dc), 0 off-board.

    Works for any |dr|, |dc| < N. Batched over the leading axis.
    """
    if dr == 0 and dc == 0:
        return x
    y = np.roll(x, shift=(-dr, -dc), axis=(1, 2))
    if dr > 0:
        y[:, N - dr:, :] = 0
    elif dr < 0:
        y[:, :(-dr), :] = 0
    if dc > 0:
        y[:, :, N - dc:] = 0
    elif dc < 0:
        y[:, :, :(-dc)] = 0
    return y


def _runlen(plane: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """count[b, r, c] = # consecutive True in ``plane`` strictly in +(dr,dc) from (r,c).

    Fixed-point of  cnt = nb(plane) * (1 + nb(cnt)); N iterations is exact
    (a run cannot exceed N).
    """
    nb_plane = nb(plane, dr, dc).astype(np.int16)
    cnt = np.zeros_like(nb_plane)
    for _ in range(N):
        cnt = nb_plane * (1 + nb(cnt, dr, dc))
    return cnt


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def empties(own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    return ~(own | opp)


def five_completion_mask(own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """(B,N,N) bool: empty cells where placing ``own`` makes >= WIN_LEN in a row.

    Mirrors ``gomoku.vcf._five_completions`` (for the to-move side ``own``).
    """
    empty = empties(own, opp)
    five = np.zeros(own.shape, dtype=bool)
    for dr, dc in DIRS:
        total = _runlen(own, dr, dc) + _runlen(own, -dr, -dc) + 1
        five |= total >= WIN
    return five & empty


def four_structure(own: np.ndarray, opp: np.ndarray):
    """Per empty cell ``m``, the four-structure if ``own`` plays ``m``.

    Returns:
      n_comp  (B,N,N) int16  -- number of distinct five-completions a four at m
                                would have (0 = not a four, 1 = single/forced,
                                >=2 = double/open four = win-next).
      block   (B,N,N) int64  -- flat index of *a* completion cell when n_comp>=1
                                (the forced block for a single four), else -1.

    Mirrors counting ``len(gomoku.vcf._completions_through(own+m, m, occ))`` for
    every empty m at once. Each (direction, completion-offset delta) contributes
    at most one completion cell c = m + delta*dir, fired iff some length-5 window
    through {m, c} holds 3 ``own`` stones with m and c empty.
    """
    empty = empties(own, opp)
    n_comp = np.zeros(own.shape, dtype=np.int16)
    block = np.full(own.shape, -1, dtype=np.int64)
    idx = np.arange(N * N, dtype=np.int64).reshape(1, N, N)

    for dr, dc in DIRS:
        for delta in (-4, -3, -2, -1, 1, 2, 3, 4):
            ec = nb(empty, delta * dr, delta * dc)        # c = m + delta*dir empty
            jm_lo = max(0, -delta)
            jm_hi = min(WIN - 1, WIN - 1 - delta)
            if jm_lo > jm_hi:
                continue
            fired = np.zeros(own.shape, dtype=bool)
            for jm in range(jm_lo, jm_hi + 1):
                pat = empty & ec
                for i in range(WIN):
                    r = i - jm
                    if r == 0 or r == delta:              # skip m and c
                        continue
                    pat = pat & nb(own, r * dr, r * dc)   # own at m + r*dir
                fired |= pat
            n_comp += fired.astype(np.int16)
            cidx = np.broadcast_to(nb(idx, delta * dr, delta * dc), own.shape)
            block = np.where(fired & (block < 0), cidx, block)
    return n_comp, block


def candidate_mask(own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """(B,N,N) bool: empty cells within Chebyshev distance 2 of any stone.

    Mirrors ``gomoku.vcf._candidate_cells`` (batched). Boards with no stones
    yield an all-False mask.
    """
    occ = own | opp
    near = np.zeros(own.shape, dtype=bool)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            near |= nb(occ, dr, dc)
    return near & ~occ


def defender_can_four_or_five(own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """(B,) bool: can the DEFENDER (``opp``) make a four or five anywhere?

    The tempo guard (``gomoku.vcf._defender_has_four_or_five``), batched: swap
    roles and reuse five/four detection for the opponent.
    """
    five = five_completion_mask(opp, own).reshape(own.shape[0], -1).any(1)
    ncomp, _ = four_structure(opp, own)
    four = (ncomp >= 1).reshape(own.shape[0], -1).any(1)
    return five | four
