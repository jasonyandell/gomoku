"""Gen-cost bound for the OFFENSIVE VCT teacher (bead derby-b6r).

The VCT (Victory-by-Continuous-Threes) solver's tree fans out on the defender
side, so an UNBOUNDED (or loosely bounded) per-move solve on the self-play
GENERATION hot path is ruinous: Derby v8 raced the `derby-x-vct` cell with the
general-purpose library defaults (``vcf.DEFAULT_VCT_MAX_DEPTH`` = 7 /
``DEFAULT_VCT_MAX_NODES`` = 20k) and produced ZERO games / buf=0 — a single
per-move solve never returned in time, fully starving generation.

The fix decouples the TEACHER hot-path caps from the library defaults and pins
them aggressively (``self_play._VCT_TEACHER_MAX_DEPTH`` = 4 /
``_VCT_TEACHER_MAX_NODES`` = 800), threaded from the ``--vct-max-depth`` /
``--vct-max-nodes`` flags. These tests prove, on CPU, that:

  1. an explosive wide-open position that blows up the loose cap now bails fast
     under the aggressive cap (node counter <= cap, returns hit_cap / no-win, no
     hang) — time-bounded;
  2. the cap is actually THREADED from the flag (a tiny cap does strictly less
     work than a larger one);
  3. short VCT wins are still proven within the aggressive cap (the teacher still
     gives signal on easy tactics);
  4. the teacher module defaults are in fact the aggressive (small) values, not
     the loose library defaults.

NOTE (honesty): these CPU tests prove the per-move solve is now fast/bounded.
They CANNOT reproduce the live generation-flooding regime (8 MPS workers feeding
a trainer); that is the derby runner's real 2-chunk smoke (buf grows, epoch ~3s).
"""

from __future__ import annotations

import time

import numpy as np

from gomoku import state_ops, vcf
from gomoku.game import HISTORY_PLY, N_INPUT_PLANES
from gomoku.self_play import (
    _VCT_TEACHER_MAX_DEPTH,
    _VCT_TEACHER_MAX_NODES,
    _apply_vct_teacher,
)

N = state_ops.BOARD_SIZE


def make_planes(me_cells, opp_cells):
    """(N_INPUT_PLANES, N, N) float32 plane stack: side-to-move at plane 0,
    opponent at plane HISTORY_PLY — the layout `_apply_vct_teacher` reads."""
    planes = np.zeros((N_INPUT_PLANES, N, N), dtype=np.float32)
    for r, c in me_cells:
        planes[0, r, c] = 1.0
    for r, c in opp_cells:
        planes[HISTORY_PLY, r, c] = 1.0
    return planes


# A wide-open position empirically found to blow the LOOSE library cap (seed-53
# fuzz, 8-14 random stones): under depth 7 / nodes 20k it explores ~5700 nodes /
# ~15s on the pure-Python path (the gen-starvation worst case). Under the
# aggressive teacher cap it must bail in a tiny fraction of that. Neither side
# has five and there is no forced win — the solver MUST explore (worst case).
_EXPLOSIVE_ME = [(0, 8), (5, 8), (6, 5), (4, 2), (6, 6), (5, 6)]
_EXPLOSIVE_OPP = [(0, 0), (5, 3), (2, 6), (3, 1), (7, 7), (1, 1)]


def _explosive_board():
    b = np.zeros((2, N, N), dtype=bool)
    for r, c in _EXPLOSIVE_ME:
        b[0, r, c] = True
    for r, c in _EXPLOSIVE_OPP:
        b[1, r, c] = True
    return b


def test_explosive_position_is_a_real_worst_case():
    # Guard the fixture: neither side already has five (a legal stm position), and
    # there is genuinely NO forced win for the side to move (so the solver must
    # actually search — otherwise the bound test would be vacuous).
    b = _explosive_board()
    assert not state_ops.has_five_in_a_row(b[0])
    assert not state_ops.has_five_in_a_row(b[1])
    assert vcf.solve_vct(b, max_depth=_VCT_TEACHER_MAX_DEPTH,
                         max_nodes=_VCT_TEACHER_MAX_NODES).has_forced_win is False


def test_aggressive_cap_bails_fast_and_respects_node_budget():
    # The headline gen-cost guard: on the explosive position the aggressive teacher
    # cap must (a) keep the node counter at/under the budget, (b) return promptly
    # (no hang), and (c) report hit_cap / no-forced-win rather than spinning.
    b = _explosive_board()
    t0 = time.perf_counter()
    res = vcf.solve_vct(b, max_depth=_VCT_TEACHER_MAX_DEPTH,
                        max_nodes=_VCT_TEACHER_MAX_NODES)
    dt = time.perf_counter() - t0
    assert res.nodes <= _VCT_TEACHER_MAX_NODES, (
        f"node cap blown: {res.nodes} > {_VCT_TEACHER_MAX_NODES}")
    assert res.has_forced_win is False  # no false positive on a non-win
    assert res.hit_cap is True          # bailed on the cap, did not exhaust the tree
    # Generous wall ceiling (pure-Python, native ext absent in the worktree). The
    # loose default takes ~15s here; the bounded solve is an order of magnitude
    # faster. This asserts "does not hang", not a precise speed.
    assert dt < 3.0, f"bounded solve took too long: {dt:.2f}s"


def test_cap_is_threaded_from_the_flag():
    # The cap must actually flow through to the solver: a TINY node cap explores
    # strictly fewer nodes than a larger one on the same explosive position. (If
    # the flag were ignored, the node counts would be equal.)
    b = _explosive_board()
    tiny = vcf.solve_vct(b, max_depth=_VCT_TEACHER_MAX_DEPTH, max_nodes=50)
    big = vcf.solve_vct(b, max_depth=_VCT_TEACHER_MAX_DEPTH, max_nodes=2000)
    assert tiny.nodes <= 50
    assert tiny.nodes < big.nodes, (
        f"cap not threaded: tiny={tiny.nodes} not < big={big.nodes}")
    # Also exercise the teacher seam itself with an explicit tiny cap: it must bail
    # without firing (no proven win under 50 nodes here) and never raise.
    planes = make_planes(_EXPLOSIVE_ME, _EXPLOSIVE_OPP)
    pi_in = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    new_pi, new_z, fired = _apply_vct_teacher(
        planes, pi_in.copy(), 0.0, side=0, max_depth=_VCT_TEACHER_MAX_DEPTH,
        max_nodes=50,
    )
    assert fired is False
    assert np.array_equal(new_pi, pi_in) and new_z == 0.0


def test_short_vct_win_still_found_within_aggressive_cap():
    # Signal preservation: the double-three fork (a threes-only VCT win in 2
    # attacker moves — VCF is blind to it) must STILL be proven under the
    # aggressive teacher cap, so the teacher keeps providing tactical signal.
    fork_me = [(4, 3), (4, 5), (3, 4), (5, 4)]
    fork_opp = [(0, 0)]
    planes = make_planes(fork_me, fork_opp)
    pi_in = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    new_pi, new_z, fired = _apply_vct_teacher(
        planes, pi_in, 0.0, side=0,
        max_depth=_VCT_TEACHER_MAX_DEPTH, max_nodes=_VCT_TEACHER_MAX_NODES,
    )
    assert fired is True
    assert int(np.argmax(new_pi)) == 4 * N + 4  # proven winning move (4,4)
    # And the immediate open-four mate (dist 1) is trivially within the cap too.
    of_planes = make_planes([(2, 2), (2, 3), (2, 4), (2, 5)], [(8, 8)])
    _, _, of_fired = _apply_vct_teacher(
        of_planes, pi_in.copy(), 0.5, side=0,
        max_depth=_VCT_TEACHER_MAX_DEPTH, max_nodes=_VCT_TEACHER_MAX_NODES,
    )
    assert of_fired is True


def test_teacher_defaults_are_aggressive_not_library_defaults():
    # The whole bug was the teacher using the LOOSE library defaults (depth 7 /
    # nodes 20k). Lock in that the teacher hot-path defaults are decoupled and
    # aggressive (small), and far below the library defaults.
    assert _VCT_TEACHER_MAX_DEPTH <= 5
    assert _VCT_TEACHER_MAX_NODES <= 2000
    assert _VCT_TEACHER_MAX_DEPTH < vcf.DEFAULT_VCT_MAX_DEPTH
    assert _VCT_TEACHER_MAX_NODES < vcf.DEFAULT_VCT_MAX_NODES
    # The teacher uses these as its None-default (configure_vct_teacher seeds them).
    from gomoku import self_play
    assert self_play._VCT_MAX_DEPTH == _VCT_TEACHER_MAX_DEPTH
    assert self_play._VCT_MAX_NODES == _VCT_TEACHER_MAX_NODES
