"""VCT megakernel (mega_vct_bb) works at the curriculum board sizes 11 and 13.

The moonshot gauntlet (issue #103) climbs 9 -> 11 -> 13 -> 15, so the GPU VCT
solver must run at every rung. It already does: the solver is fully board-size-
parametric -- every bitboard width, direction step, five/four mask, COLLIN /
king-dilation table and the 4x64-bit board packing derive from
``N = state_ops.BOARD_SIZE`` (resolved once at import from GOMOKU_BOARD_SIZE).
Nothing in scripts/vct_metal hardcodes 9 or 15; the 256-bit packing holds any
N*N <= 256 (N <= 16) and the specialised shr/shl256 cover shifts up to
4*(N+1) <= 64 in that range. This test proves it with hand-built forced wins at
the ACTIVE size.

Board size is a process-level constant fixed at import, so -- exactly like
tests/test_board15.py -- these cases are SKIPPED in a default (9x9) run and must
be driven with the size set before pytest imports the package:

    GOMOKU_BOARD_SIZE=11 uv run pytest tests/test_vct_board_sizes.py
    GOMOKU_BOARD_SIZE=13 uv run pytest tests/test_vct_board_sizes.py

Each case is a SINGLE board (one momentary Metal dispatch) -- deliberately
GPU-cheap, safe alongside a training run. It is the "does size N even work" gate,
NOT the deep golden-vs-oracle validation (that stays in validate_deep.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from gomoku import game

pytestmark = pytest.mark.skipif(
    game.BOARD_SIZE not in (11, 13),
    reason="curriculum-size VCT test needs GOMOKU_BOARD_SIZE in {11, 13} "
           "(board size is fixed per process)",
)

N = game.BOARD_SIZE

# The four freestyle line directions as (dr, dc): horizontal, vertical, and the
# two diagonals. Each has a distinct bit-step (1, N, N+1, N-1), so covering all
# four exercises every per-direction / per-N table in the kernel.
_DIRS = [("H", 0, 1), ("V", 1, 0), ("D1", 1, 1), ("D2", 1, -1)]


def _open_four(dr: int, dc: int):
    """A centered open four for the side to move along (dr, dc): four stones
    with both ends empty and on-board -> an immediate five-completion (a VCT
    win in one). Returns (board (1,2,N,N) bool, {the two winning end cells})."""
    mid = N // 2
    # Start so the 6-cell window (end, 4 stones, end) is centered on the board.
    r0 = mid - 2 * dr
    c0 = mid - 2 * dc
    stones = [(r0 + i * dr, c0 + i * dc) for i in range(4)]
    ends = [(r0 - dr, c0 - dc), (r0 + 4 * dr, c0 + 4 * dc)]
    for r, c in stones + ends:
        assert 0 <= r < N and 0 <= c < N, (dr, dc, r, c, N)
    b = np.zeros((1, 2, N, N), dtype=bool)
    for r, c in stones:
        b[0, 0, r, c] = True
    # a couple of inert defender stones far away in a corner
    b[0, 1, 0, 0] = b[0, 1, 0, 1] = True
    end_cells = {r * N + c for r, c in ends}
    return b, end_cells


@pytest.mark.parametrize("name,dr,dc", _DIRS)
def test_open_four_is_a_vct_win(name, dr, dc):
    from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb, N as SOLVER_N

    assert SOLVER_N == N, (SOLVER_N, N)
    b, ends = _open_four(dr, dc)
    win, hit, move = solve_vct_mega_bb(b, max_nodes=50, return_move=True)
    assert bool(win[0]), f"{name} open four not solved as a win at N={N}"
    assert not bool(hit[0]), f"{name} open four hit the node cap at N={N}"
    assert int(move[0]) in ends, (
        f"{name} winning move {int(move[0])} not an open end {sorted(ends)} at N={N}"
    )


def test_quiet_board_is_not_a_win():
    """A scattered, non-forcing position must NOT be reported as a VCT win
    (guards against the empty-high-word leak that issue #98 fixed for N<15)."""
    from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb

    b = np.zeros((1, 2, N, N), dtype=bool)
    mid = N // 2
    b[0, 0, mid, mid] = b[0, 0, mid + 1, mid + 1] = True
    b[0, 1, mid - 1, mid - 1] = b[0, 1, mid + 2, mid + 2] = True
    win, hit, _move = solve_vct_mega_bb(b, max_nodes=50, return_move=True)
    assert not (bool(win[0]) and not bool(hit[0])), (
        f"quiet board falsely solved as a clean VCT win at N={N}"
    )
