"""Tests for the stage-2 VCT-backward enabling-shape miner.

The #1 correctness gotcha is board ORIENTATION: ``solve_vct`` needs plane 0 = the
side to move (= the winner/attacker). These tests pin that invariant down for
BOTH a black winner and a WHITE winner (the swap-sensitive case), plus the
min-run filter and the parity guard.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from gomoku.board_config import BOARD_SIZE as N
from gomoku.game import GameState
from gomoku.vcf import solve_vct
from scripts.threat_shapes.mine_vct_backward import TAG, mine_game

pytestmark = pytest.mark.skipif(
    N != 15, reason="enabling-shape miner is a 15x15 pipeline (set GOMOKU_BOARD_SIZE=15)"
)


def _abs_planes_before(moves, ply):
    """Absolute (black, white) stone planes just before ``moves[ply]`` is played."""
    blk = np.zeros((N, N), bool)
    wht = np.zeros((N, N), bool)
    for i, m in enumerate(moves[:ply]):
        r, c = divmod(m, N)
        (blk if i % 2 == 0 else wht)[r, c] = True
    return blk, wht


def _white_win_game():
    """A hand-built decisive game: WHITE completes a five on the last move, built
    up as a forcing line so the backward walk has real depth. Black plays a
    harmless detached line that never makes a four."""
    # even idx = black, odd idx = white. White wins => last ply L is odd.
    rc = lambda r, c: r * N + c
    moves = [
        rc(0, 0), rc(5, 4),   # 0 B, 1 W
        rc(0, 1), rc(5, 5),   # 2 B, 3 W
        rc(0, 2), rc(5, 6),   # 4 B, 5 W (white open three)
        rc(0, 3), rc(5, 7),   # 6 B, 7 W (white open four)
        rc(1, 0), rc(5, 8),   # 8 B, 9 W -> white (5,4..5,8) = five, WIN
    ]
    return moves, 1


def test_white_winner_orientation_no_swap():
    """For a WHITE winner the emitted enabling board must have WHITE in plane 0
    and BLACK in plane 1 (board is side-to-move-relative; no manual swap)."""
    moves, winner = _white_win_game()
    stats: Counter = Counter()
    rec = mine_game(moves, winner, min_run=1, max_depth=7, max_nodes=20000,
                    stats=stats)
    assert rec is not None
    assert rec["winner"] == 1
    assert rec["tag"] == TAG

    # Reconstruct absolute stones at the enabling ply and confirm the orientation.
    blk, wht = _abs_planes_before(moves, rec["ply"])
    assert np.array_equal(rec["board"][0], wht), "plane 0 must be the WHITE (winner) stones"
    assert np.array_equal(rec["board"][1], blk), "plane 1 must be the BLACK (loser) stones"

    # The enabling position must genuinely re-solve to a forced win for the winner,
    # and the catalyst must be that winning move (no false positive).
    res = solve_vct(rec["board"])
    assert res.has_forced_win
    assert rec["move"] == res.winning_move
    # catalyst is a legal (empty) cell
    r, c = divmod(rec["move"], N)
    assert not (rec["board"][0][r, c] or rec["board"][1][r, c])


def test_black_winner_orientation():
    """Mirror game with a BLACK winner: plane 0 = black stones."""
    rc = lambda r, c: r * N + c
    # black completes the five on the last (even) move.
    moves = [
        rc(5, 4), rc(0, 0),   # 0 B, 1 W
        rc(5, 5), rc(0, 1),   # 2 B, 3 W
        rc(5, 6), rc(0, 2),   # 4 B (open three), 5 W
        rc(5, 7), rc(0, 3),   # 6 B (open four), 7 W
        rc(5, 8),             # 8 B -> black five, WIN  (L=8, even => winner 0)
    ]
    winner = 0
    stats: Counter = Counter()
    rec = mine_game(moves, winner, min_run=1, max_depth=7, max_nodes=20000,
                    stats=stats)
    assert rec is not None
    assert rec["winner"] == 0
    blk, wht = _abs_planes_before(moves, rec["ply"])
    assert np.array_equal(rec["board"][0], blk)
    assert np.array_equal(rec["board"][1], wht)
    assert solve_vct(rec["board"]).has_forced_win


def test_min_run_filter_drops_short_tails():
    """A high min-run threshold drops the (necessarily shallow) hand-built game."""
    moves, winner = _white_win_game()
    rec_lo = mine_game(moves, winner, min_run=1, max_depth=7, max_nodes=20000,
                       stats=Counter())
    assert rec_lo is not None
    forced_run = rec_lo["run_plies"]
    # Asking for a deeper forced run than this game realizes must yield nothing.
    rec_hi = mine_game(moves, winner, min_run=forced_run + 1, max_depth=7,
                       max_nodes=20000, stats=Counter())
    assert rec_hi is None


def test_parity_guard():
    """If the winner does not own the last move (corrupt parity) -> None, no emit."""
    moves, _ = _white_win_game()  # last move is white (winner should be 1)
    stats: Counter = Counter()
    rec = mine_game(moves, winner=0, min_run=1, max_depth=7, max_nodes=20000,
                    stats=stats)
    assert rec is None
    assert stats["bad_parity"] == 1


def test_board_is_side_to_move_relative():
    """Direct check of the convention the miner relies on: after reconstructing to
    a white-to-move position, GameState.board[0] holds white's stones."""
    moves, _ = _white_win_game()
    s = GameState.initial()
    for m in moves[:9]:   # position before ply 9 (white to move)
        s = s.apply(m)
    blk, wht = _abs_planes_before(moves, 9)
    assert np.array_equal(s.board[0], wht)
    assert np.array_equal(s.board[1], blk)
