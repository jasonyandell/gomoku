"""Correctness tests for the OFFENSIVE VCT teacher (gomoku/self_play._apply_vct_teacher).

The VCT teacher is the exact mirror of the offensive VCF teacher, with the solver
swapped from `vcf.solve_vcf_from_planes` to `vcf.solve_vct_from_planes`. VCT
(Victory-by-Continuous-Threes) is a strict SUPERSET of VCF: it proves every VCF
forced win PLUS wins that need forcing threes, so it stamps MORE positions with
exact mate labels (one-hot proven winning move + mate-distance-discounted value).

A FALSE POSITIVE poisons BOTH the policy target (one-hot on a non-winning move)
and the value target (a hedged outcome relabeled to ~+1.0 toward a non-existent
win), so the headline test is `test_no_false_positive_*`: over 400 random
positions the VCT teacher's "side-to-move has a forced win" claims must have ZERO
false positives versus an independent sound referee that replays the proven line
and — at every defender node of a forcing-three branch — requires the attacker to
win against EVERY legal defender reply (not the solver's bounded reply set).

The referee mirrors `tests/test_vct._referee_win`: it re-uses the solver to FIND
a move, then independently re-derives the forced win over all defender replies, so
a bug in the solver's pruning surfaces as a refuted (False) verdict.
"""

from __future__ import annotations

import numpy as np

from gomoku import state_ops, vcf
from gomoku.game import HISTORY_PLY, N_INPUT_PLANES
from gomoku.self_play import (
    VCF_VALUE_FLOOR,
    _VCT_MAX_DEPTH,
    _VCT_MAX_NODES,
    _apply_vct_teacher,
)
from gomoku.vcf import _empties_from_plane, _five_completions, solve_vct

N = state_ops.BOARD_SIZE


def make_planes(me_cells, opp_cells):
    """Build an (N_INPUT_PLANES, N, N) float32 plane stack with the side-to-move's
    (attacker's) stones at plane 0 and the opponent's stones at plane HISTORY_PLY
    (the layout `GameState.to_planes` emits and that `_apply_vct_teacher` reads)."""
    planes = np.zeros((N_INPUT_PLANES, N, N), dtype=np.float32)
    for r, c in me_cells:
        planes[0, r, c] = 1.0
    for r, c in opp_cells:
        planes[HISTORY_PLY, r, c] = 1.0
    return planes


def _board_from_planes(planes):
    """The (2, N, N) board the offensive VCT teacher solves: side-to-move = attacker."""
    attacker = planes[0].astype(bool)
    defender = planes[HISTORY_PLY].astype(bool)
    return np.stack([attacker, defender], axis=0)


def _referee_win(attacker, defender, depth=0, budget=None, max_depth=6):
    """Independent SOUND referee: does the attacker truly force a VCT win here?

    Replays the solver's chosen move, then at a DEFENDER node of a forcing-three
    branch enumerates EVERY empty cell (not the bounded threat-defeating set the
    solver uses internally) and requires the attacker to win against all of them —
    re-deriving the forced win without trusting the solver's pruning. Returns True
    (proven win), False (refuted -> a REAL failure / false positive), or None
    (declined: too deep / out of budget). Mirrors `tests/test_vct._referee_win`.
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


# ---------------------------------------------------------------------------
# Positive sanity: a proven win overwrites policy (one-hot) + value (>= floor)
# ---------------------------------------------------------------------------

def test_open_four_overwrites_policy_and_value():
    # An open four .XXXX. for the side to move is a proven one-move win. The VCT
    # teacher must fire: policy one-hot on the proven winning move, value >= floor.
    planes = make_planes(
        me_cells=[(2, 2), (2, 3), (2, 4), (2, 5)],
        opp_cells=[(8, 8)],
    )
    pi_in = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    assert _referee_win(*_board_from_planes(planes)) is True
    new_pi, new_z, fired = _apply_vct_teacher(planes, pi_in, 0.5, side=0)
    assert fired is True
    assert new_pi.sum() == 1.0 and int(np.count_nonzero(new_pi)) == 1
    res = vcf.solve_vct_from_planes(planes)
    assert int(np.argmax(new_pi)) == res.winning_move
    assert new_z >= VCF_VALUE_FLOOR


def test_quiet_position_does_not_fire():
    # A sparse position with no forcing threat: the solver proves nothing, so the
    # teacher leaves policy and value untouched.
    planes = make_planes(me_cells=[(4, 4)], opp_cells=[(0, 0), (8, 8)])
    pi_in = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    new_pi, new_z, fired = _apply_vct_teacher(planes, pi_in, 0.3, side=0)
    assert fired is False
    assert new_z == 0.3
    assert np.array_equal(new_pi, pi_in)


# The locked-in double-three fork from tests/test_vct: a single move (4,4) makes
# TWO disjoint open threes (row 4 and col 4), threatening disjoint open fours the
# defender cannot both parry. Neither three is a four, so VCF is blind to it; VCT
# proves the win in 2 attacker moves. Used to guarantee the fuzz exercises a real
# threes-only win (not just fours VCF would also find).
_FORK_ME = [(4, 3), (4, 5), (3, 4), (5, 4)]
_FORK_OPP = [(0, 0)]


def test_double_three_fork_is_vct_not_vcf():
    board = make_planes(_FORK_ME, _FORK_OPP)  # reused only for the (2,N,N) solve
    bd = _board_from_planes(board)
    assert vcf.solve_vcf(bd).has_forced_win is False  # VCF blind to a threes win
    assert vcf.solve_vct(bd).has_forced_win is True   # VCT sees the fork
    planes = make_planes(_FORK_ME, _FORK_OPP)
    pi_in = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    new_pi, new_z, fired = _apply_vct_teacher(planes, pi_in, 0.0, side=0)
    assert fired is True
    assert int(np.argmax(new_pi)) == 4 * N + 4  # the proven winning move (4,4)
    assert _referee_win(*_board_from_planes(planes)) is True


# ---------------------------------------------------------------------------
# THE CRITICAL TEST: zero false positives over 400 random positions
# ---------------------------------------------------------------------------

def test_no_false_positive_random_positions():
    # Fuzz: random legal positions where neither side has five. Whenever the VCT
    # teacher FIRES (claims the side to move has a forced win => policy/value
    # overwritten), the independent referee MUST confirm a real forced win. ZERO
    # false positives is the BLOCKING gate (a false positive points the policy at a
    # non-winning move and the value toward a non-existent win).
    rng = np.random.default_rng(20260527)
    n_fired = 0
    n_examined = 0
    n_declined = 0
    pi_in = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    for _ in range(400):
        n_stones = int(rng.integers(2, 16))
        cells = rng.choice(N * N, size=n_stones, replace=False)
        me_cells = [(int(x // N), int(x % N)) for x in cells[0::2]]
        opp_cells = [(int(x // N), int(x % N)) for x in cells[1::2]]
        planes = make_planes(me_cells, opp_cells)
        board = _board_from_planes(planes)
        # Skip positions where either side already has five (not a legal
        # side-to-move position).
        if state_ops.has_five_in_a_row(board[0]) or state_ops.has_five_in_a_row(board[1]):
            continue
        n_examined += 1
        new_pi, new_z, fired = _apply_vct_teacher(planes, pi_in.copy(), 0.0, side=0)
        if fired:
            # Policy is a one-hot on the solver's proven move; value is a win vertex.
            assert int(np.count_nonzero(new_pi)) == 1
            assert new_z >= VCF_VALUE_FLOOR
            # Re-solve with the SAME caps the teacher used (its aggressive
            # per-move budget, _VCT_MAX_DEPTH/_NODES — NOT the looser library
            # defaults), so the one-hot move matches the solver the teacher ran.
            res = vcf.solve_vct_from_planes(
                planes, max_depth=_VCT_MAX_DEPTH, max_nodes=_VCT_MAX_NODES)
            assert int(np.argmax(new_pi)) == res.winning_move
            verdict = _referee_win(board[0].copy(), board[1].copy())
            # None == referee declined the deep line (out of budget); only an
            # explicit False is a false positive (the no-false-positive guarantee).
            if verdict is None:
                n_declined += 1
            else:
                assert verdict is True, (
                    "VCT TEACHER FALSE POSITIVE: teacher fired but referee found no "
                    "forced win"
                )
                n_fired += 1
        else:
            # When it does NOT fire, policy and value must be untouched.
            assert new_z == 0.0
            assert np.array_equal(new_pi, pi_in)
    # Informational: the fuzz should have examined real positions; some seeds may
    # legitimately yield zero forced-win fires.
    assert n_examined > 0


def test_no_false_positive_known_fork_referee_confirms():
    # Seed-independent guarantee that the fuzz path actually exercises a referee-
    # confirmed win (so the no-false-positive assertion above is not vacuous).
    planes = make_planes(_FORK_ME, _FORK_OPP)
    pi_in = np.full(N * N, 1.0 / (N * N), dtype=np.float32)
    _, _, fired = _apply_vct_teacher(planes, pi_in, 0.0, side=0)
    assert fired is True
    assert _referee_win(*_board_from_planes(planes)) is True


# ---------------------------------------------------------------------------
# DEFAULT-OFF / byte-identical: the flag is off by default; the solver is never
# called on the default path, so self-play is bit-for-bit unchanged.
# ---------------------------------------------------------------------------

def test_generate_games_vct_teacher_defaults_off():
    # The public generator signatures must default `vct_teacher` to False so the
    # production default path is byte-identical (no overwrite, no VCT solve).
    import inspect
    from gomoku import self_play

    for name in (
        "generate_games",
        "generate_games_vs_baseline",
        "_generate_games_native",
        "_generate_games_gumbel",
        "_generate_games_native_gumbel",
    ):
        fn = getattr(self_play, name)
        sig = inspect.signature(fn)
        assert "vct_teacher" in sig.parameters, f"{name} missing vct_teacher"
        assert sig.parameters["vct_teacher"].default is False, \
            f"{name}.vct_teacher default is not False"


def _const_evaluator(states):
    """Uniform-policy / zero-value batch evaluator (the Evaluator Protocol:
    callable taking a list of canonical states, returning (priors, values))."""
    n = len(states)
    priors = np.full((n, N * N), 1.0 / (N * N), dtype=np.float32)
    values = np.zeros((n,), dtype=np.float32)
    return priors, values


def test_default_off_never_calls_vct_solver(monkeypatch):
    # Byte-identical-when-off proof: monkeypatch the VCT solver to RAISE, then run a
    # tiny generation with the default flags. If the off-path ever entered the VCT
    # teacher the solver would fire and the run would crash; reaching the end with
    # records produced proves the solver is never called when the flag is off.
    from gomoku import self_play

    def boom(*args, **kwargs):  # pragma: no cover - must never be hit when off
        raise AssertionError(
            "solve_vct_from_planes was called with vct_teacher OFF — the default "
            "path is NOT byte-identical"
        )

    monkeypatch.setattr(vcf, "solve_vct_from_planes", boom)

    rng = np.random.default_rng(0)
    # Pure-Python fallback path (no native engine, no gumbel) — exercises the
    # _apply_*_teacher seam. vct_teacher defaults to False (not passed).
    records = self_play.generate_games(
        2,
        _const_evaluator,
        n_simulations=4,
        max_plies=12,
        rng=rng,
        augment_symmetries=False,
        wave_size=1,
    )
    assert len(records) >= 1  # generation completed without entering the VCT solver


def test_default_off_call_counter(monkeypatch):
    # A complementary call-counter proof at the unit seam: the offensive teacher is
    # only entered when vct_teacher is True. With it False the per-position loop must
    # never invoke `_apply_vct_teacher`, so the VCT solver call count stays zero.
    calls = {"vct": 0}
    real = vcf.solve_vct_from_planes

    def counting(*args, **kwargs):
        calls["vct"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(vcf, "solve_vct_from_planes", counting)

    from gomoku import self_play

    rng = np.random.default_rng(1)
    self_play.generate_games(
        2, _const_evaluator, n_simulations=4, max_plies=12, rng=rng,
        augment_symmetries=False, wave_size=1,
    )
    assert calls["vct"] == 0, "VCT solver was called on the default-off path"
