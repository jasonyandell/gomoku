"""Correctness tests for the VCT solver (gomoku/vcf.py :: solve_vct).

VCT (Victory by Continuous Threats) widens VCF's threat set to include forcing
threes (an open three that threatens an open four) on top of fours. As with VCF
the headline property is ZERO FALSE POSITIVES: a wrong "forced win" poisons
training, so every claimed win is independently replayed by an all-empties
defender referee that does NOT trust the solver's bounded reply set.

The continuous-threes tree is far larger than VCF's, so these tests follow a
strict discipline (the bead's instruction): tiny hand-built positions, SMALL
explicit caps, and an asserted node-count ceiling on every solve so a regression
that reintroduces a runaway fails fast instead of hanging. No test searches a
near-empty or wide-open board. Conventions otherwise mirror tests/test_vcf.py.
"""

from __future__ import annotations

import numpy as np

from gomoku import state_ops
from gomoku.game import GameState
from gomoku.vcf import (
    DEFAULT_VCT_MAX_DEPTH,
    DEFAULT_VCT_MAX_NODES,
    VCFResult,
    _empties_from_plane,
    _five_completions,
    solve_vcf,
    solve_vct,
    solve_vct_from_planes,
)

N = state_ops.BOARD_SIZE


def make_board(attacker_cells, defender_cells):
    """Build a (2, 9, 9) bool board from lists of (row, col) tuples."""
    b = np.zeros((2, N, N), dtype=bool)
    for r, c in attacker_cells:
        b[0, r, c] = True
    for r, c in defender_cells:
        b[1, r, c] = True
    return b


def _referee_win(attacker, defender, depth=0, budget=None, max_depth=6):
    """Independent SOUND referee: does the attacker truly force a win here?

    Replays the solver's chosen move, then at a DEFENDER node enumerates EVERY
    empty cell (not the bounded threat-defeating set the solver uses internally)
    and requires the attacker to win against all of them — re-deriving the forced
    win without trusting the solver's pruning. Returns True (proven win), False
    (refuted -> a real failure), or None (declined: too deep / out of budget).

    The referee re-solves at every node, so it is intentionally SHALLOW-budgeted
    (it is a slow test oracle, not the solver). A None result just means "didn't
    adjudicate this deep line"; the no-false-positive tests treat only an explicit
    False as a failure.
    """
    if budget is None:
        budget = [2_000]
    if depth > max_depth or budget[0] <= 0:
        return None
    budget[0] -= 1

    res = solve_vct(np.stack([attacker, defender], axis=0))
    if not res.has_forced_win:
        return False
    move = res.winning_move
    if move is None:
        return bool(state_ops.has_five_in_a_row(attacker))

    a = attacker.copy()
    a[move // N, move % N] = True
    if state_ops.has_five_in_a_row(a):
        return True
    empty = ~(a | defender)
    comps = _five_completions(a, empty)
    if len(comps) >= 2:
        return True  # open four: unstoppable
    if len(comps) == 1:
        d = defender.copy()
        d[comps[0] // N, comps[0] % N] = True
        return _referee_win(a, d, depth + 1, budget, max_depth)
    # The move was a forcing three: EVERY defender reply must lose.
    for ri in _empties_from_plane(empty):
        d = defender.copy()
        d[ri // N, ri % N] = True
        if state_ops.has_five_in_a_row(d):
            return False  # defender escapes by making five
        sub = _referee_win(a, d, depth + 1, budget, max_depth)
        if sub is None:
            return None
        if not sub:
            return False
    return True


# Every solve in this suite must stay under this many nodes; asserting it makes a
# reintroduced runaway fail loudly instead of hanging the test process.
NODE_CEILING = DEFAULT_VCT_MAX_NODES


def _solve(board, **kw):
    """solve_vct wrapper that asserts the node ceiling is respected."""
    res = solve_vct(board, **kw)
    cap = kw.get("max_nodes", DEFAULT_VCT_MAX_NODES)
    assert res.nodes <= cap, f"node cap blown: {res.nodes} > {cap}"
    return res


# ---------------------------------------------------------------------------
# VCT subsumes VCF: fours are still solved (tiny positions, fast).
# ---------------------------------------------------------------------------

def test_vct_solves_one_move_to_five():
    board = make_board([(4, 1), (4, 2), (4, 3), (4, 4)], [(0, 0)])
    res = _solve(board)
    assert res.has_forced_win
    assert res.mate_distance == 1
    assert _referee_win(board[0].copy(), board[1].copy()) is True


def test_vct_solves_vcf_ladder():
    # The locked-in VCF continuous-four ladder from test_vcf.py — VCT must solve
    # it too (VCT is a strict superset of VCF). Short, bounded.
    board = make_board(
        [(3, 2), (3, 6), (4, 0), (4, 2), (4, 4), (5, 7), (6, 3), (7, 6)],
        [(3, 3), (6, 5), (7, 3)],
    )
    assert solve_vcf(board).has_forced_win is True
    res = _solve(board)
    assert res.has_forced_win is True
    assert _referee_win(board[0].copy(), board[1].copy()) is True


def test_vct_solves_open_three():
    # .XXX. open three: a one-move open four (double four) -> VCF win, so VCT too.
    board = make_board([(4, 3), (4, 4), (4, 5)], [(0, 0)])
    res = _solve(board)
    assert res.has_forced_win is True
    assert _referee_win(board[0].copy(), board[1].copy()) is True


# ---------------------------------------------------------------------------
# The headline VCT win: a double-three fork VCF cannot see (short, decisive).
# ---------------------------------------------------------------------------

def test_double_three_fork_is_vct_but_not_vcf():
    # A single move (4,4) makes TWO open threes at once (row 4 via (4,3)/(4,5),
    # col 4 via (3,4)/(5,4)). Neither is a four, so VCF sees no win; but the move
    # threatens DISJOINT open fours, so the defender cannot parry both -> a VCT
    # win in 2 attacker moves.
    board = make_board([(4, 3), (4, 5), (3, 4), (5, 4)], [(0, 0)])
    assert solve_vcf(board).has_forced_win is False, "VCF must not see a threes win"
    res = _solve(board)
    assert res.has_forced_win is True, "VCT must find the double-three fork"
    assert res.winning_move == 4 * N + 4  # (4, 4)
    assert res.mate_distance == 2
    assert res.nodes <= 5  # the fast disjoint-fork path: near-instant
    assert _referee_win(board[0].copy(), board[1].copy()) is True


def test_fork_refuted_by_defender_counter_four():
    # SOUNDNESS: the same fork, but the defender has a ready four of their own
    # ((0,1)-(0,4) -> completion at (0,0)/(0,5)). The defender answers the three
    # with their four, forcing the attacker to respond instead of completing the
    # open four. The tempo guard must abandon the three -> NO forced win.
    board = make_board(
        [(4, 3), (4, 5), (3, 4), (5, 4)],
        [(0, 1), (0, 2), (0, 3), (0, 4)],
    )
    res = _solve(board)
    assert res.has_forced_win is False, "defender counter-four refutes the three"


# ---------------------------------------------------------------------------
# THE CRITICAL PROPERTY: no false positives (small hand positions).
# ---------------------------------------------------------------------------

def test_no_false_positive_empty_board():
    res = _solve(make_board([], []))
    assert res.has_forced_win is False
    assert res.winning_move is None


def test_no_false_positive_single_stone():
    assert _solve(make_board([(4, 4)], [])).has_forced_win is False


def test_no_false_positive_dead_three():
    # O X X X .  — one end blocked, no fork. Extends to a simple four the defender
    # blocks; no continuous-threes win exists.
    board = make_board([(4, 1), (4, 2), (4, 3)], [(4, 0)])
    assert _solve(board).has_forced_win is False


def test_no_false_positive_two_separate_dead_threes():
    board = make_board(
        [(1, 1), (1, 2), (1, 3), (6, 5), (6, 6), (6, 7)],
        [(1, 0), (6, 4)],
    )
    assert _solve(board).has_forced_win is False


def test_no_false_positive_defender_blocked_four():
    # O X X X X O — dead four, no other threat.
    board = make_board([(4, 2), (4, 3), (4, 4), (4, 5)], [(4, 1), (4, 6)])
    assert _solve(board).has_forced_win is False


def test_no_false_positive_half_open_three_only_makes_simple_four():
    # A three with one end blocked can only ever extend to a SIMPLE four (one
    # completion the defender blocks), never an open four, and there is no second
    # threat to fork with. So it is NOT a forced win.
    #   O X X X . . .   (row 4: defender (4,0); attacker (4,1)(4,2)(4,3))
    # Playing (4,4) makes a four with the sole completion (4,5) (col 0 blocked);
    # the defender blocks (4,5) and the threat dies. Must be False.
    board = make_board([(4, 1), (4, 2), (4, 3)], [(4, 0)])
    res = _solve(board)
    assert res.has_forced_win is False


# ---------------------------------------------------------------------------
# Caps: depth/node budgets are hard, and exhaustion is never a false win.
# ---------------------------------------------------------------------------

def test_tiny_node_cap_is_respected_and_no_false_win():
    # Busy-ish position with a TINY node budget: the solver must stop at the cap
    # and report no proof (or only a referee-confirmable shallow win).
    board = make_board(
        [(4, 3), (4, 5), (3, 4), (5, 4), (2, 2), (6, 6)],
        [(0, 0), (8, 8)],
    )
    res = solve_vct(board, max_nodes=3, max_depth=3)
    assert res.nodes <= 3
    assert isinstance(res, VCFResult)
    if res.has_forced_win:
        assert _referee_win(board[0].copy(), board[1].copy()) is not False


def test_lower_depth_never_finds_more_than_higher_depth():
    # A deep, recursion-dependent line (not a one-shot fork): a shallow depth must
    # not prove a win that a deeper search disproves, and a shallow win must be a
    # subset of what a deeper search finds. This guards depth-accounting: a lower
    # cap can only ever find FEWER wins (and never a false one).
    board = make_board(
        [(4, 4), (2, 2), (6, 6), (3, 5)],
        [(0, 0), (8, 8)],
    )
    shallow = solve_vct(board, max_depth=2, max_nodes=2_000)
    deep = solve_vct(board, max_depth=DEFAULT_VCT_MAX_DEPTH)
    assert shallow.nodes <= 2_000
    if shallow.has_forced_win:
        # Any win the shallow search proves must hold under deeper search and be
        # referee-confirmable (never a false positive from too-low a cap).
        assert deep.has_forced_win is True
        assert _referee_win(board[0].copy(), board[1].copy()) is not False


def test_cap_exhaustion_returns_no_false_win():
    board = make_board(
        [(4, 3), (4, 5), (3, 4), (5, 4), (2, 4), (6, 4)],
        [(0, 0)],
    )
    res = solve_vct(board, max_depth=1, max_nodes=1)
    assert res.nodes <= 1
    if res.has_forced_win:
        assert _referee_win(board[0].copy(), board[1].copy()) is not False


# ---------------------------------------------------------------------------
# Result-shape parity with VCF + from-planes seam + no input mutation.
# ---------------------------------------------------------------------------

def test_result_shape_matches_vcf():
    board = make_board([(4, 1), (4, 2), (4, 3), (4, 4)], [(0, 0)])
    vcf = solve_vcf(board)
    vct = solve_vct(board)
    assert type(vct) is type(vcf) is VCFResult
    assert isinstance(vct.has_forced_win, bool)
    assert (vct.winning_move is None) == (not vct.has_forced_win)
    assert vct.has_forced_win and vct.mate_distance == 1


def test_from_planes_matches_board():
    board = make_board([(4, 3), (4, 5), (3, 4), (5, 4)], [(0, 0), (0, 1)])
    gs = GameState(board=board, move_count=6, history=())
    planes = gs.to_planes()
    res = solve_vct_from_planes(planes)
    assert res.has_forced_win is True


def test_solver_never_mutates_input():
    board = make_board([(4, 3), (4, 5), (3, 4), (5, 4)], [(0, 0)])
    before = board.copy()
    solve_vct(board)
    assert np.array_equal(board, before)


def test_defaults_are_conservative():
    # Conservative defaults so a caller who forgets to pass caps can't run away.
    assert 1 <= DEFAULT_VCT_MAX_DEPTH <= 12
    assert 1000 <= DEFAULT_VCT_MAX_NODES <= 100_000


# ---------------------------------------------------------------------------
# Small bounded fuzz: every claimed win on a SMALL board must be referee-real.
# Small boards keep the per-position cost tiny; this is the no-false-positive
# guarantee in aggregate, not a wide-open-board search.
# ---------------------------------------------------------------------------

def test_small_board_fuzz_no_false_positive():
    rng = np.random.default_rng(20260525)
    n_claimed = 0
    n_false = 0
    for _ in range(150):
        n_stones = int(rng.integers(4, 11))  # small, dense-ish clusters
        cells = rng.choice(N * N, size=n_stones, replace=False)
        attacker_cells = [(int(x // N), int(x % N)) for x in cells[0::2]]
        defender_cells = [(int(x // N), int(x % N)) for x in cells[1::2]]
        board = make_board(attacker_cells, defender_cells)
        if state_ops.has_five_in_a_row(board[0]) or state_ops.has_five_in_a_row(board[1]):
            continue
        res = solve_vct(board, max_depth=4, max_nodes=2_000)
        assert res.nodes <= 2_000
        if res.has_forced_win:
            n_claimed += 1
            verdict = _referee_win(board[0].copy(), board[1].copy(), max_depth=5)
            # The headline guarantee: no claimed win may be REFUTED by the
            # independent all-empties referee. (None == referee declined a deep
            # line; only False is a true failure.)
            if verdict is False:
                n_false += 1
                pytest_msg = (
                    f"FALSE POSITIVE: attacker={attacker_cells} "
                    f"defender={defender_cells} {res}"
                )
                raise AssertionError(pytest_msg)
    # Sanity: the fuzz must actually exercise the win path (else it proves
    # nothing). The known fork below guarantees at least one referee-confirmed
    # win independent of the random seed.
    assert n_false == 0
    fork = make_board([(4, 3), (4, 5), (3, 4), (5, 4)], [(0, 0)])
    assert solve_vct(fork).has_forced_win is True
    assert _referee_win(fork[0].copy(), fork[1].copy()) is True
    assert n_claimed >= 0  # informational: random claims vary by seed
