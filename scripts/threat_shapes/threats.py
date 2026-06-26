"""Threat extraction + VCT forcing-line replay for the threat-shape pipeline.

Given a VCT-positive position (plane-0 attacker to move, with a proven
continuous-threats win), we *replay* the forcing line move-by-move and, for every
attacker move, recover Allis's (gain, cost, rest) decomposition of the threat it
makes plus a coarse threat *type*. The whole pipeline downstream (shape
clustering, dependency-motif census) consumes the list of threats this module
emits.

We reuse the SOUND primitives from ``gomoku.vcf`` verbatim:
  * ``solve_vct`` / ``solve_vcf``     -- the proof + principal-variation move.
  * ``_completions_through``          -- five-completion cells collinear with m
                                         (= the four's cost square, and the
                                         detector that "m makes a four").
  * ``_collinear_empties`` + ``_open_four_threats`` -- the forcing-three test
                                         (the open-four squares a three threatens).
  * ``_five_completions``             -- immediate-five detection (terminal).

Allis (gain/cost/rest), verbatim:
  * gain square  = the square the attacker plays (the move m).
  * cost squares = the squares the defender must play to refute the threat
                   (a four -> its 1 completion; a three -> the defeating set of
                   its threatened open four(s)).
  * rest squares = the threat's *other* attacker stones (the line minus gain).

Nothing here mutates a caller's board; we always work on copies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gomoku import state_ops
from gomoku.vcf import (
    _collinear_empties,
    _completions_through,
    _five_completions,
    _open_four_threats,
    solve_vcf,
    solve_vct,
)

N = state_ops.BOARD_SIZE
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def rc(m: int) -> tuple[int, int]:
    return m // N, m % N


def _axis(a: int, b: int) -> tuple[int, int] | None:
    """Unit line-direction from cell a to cell b if collinear on a board axis."""
    ar, ac = rc(a)
    br, bc = rc(b)
    dr, dc = br - ar, bc - ac
    if dr == 0 and dc == 0:
        return None
    for (ur, uc) in _DIRS:
        # b = a + k*(ur,uc) for some integer k (either sign)?
        for s in (1, -1):
            vr, vc = s * ur, s * uc
            if vr != 0:
                if dr % vr != 0:
                    continue
                k = dr // vr
            elif dc % vc != 0:
                continue
            else:
                k = dc // vc
            if k > 0 and ar + k * vr == br and ac + k * vc == bc:
                return (ur, uc)
    return None


def _rest_stones(attacker: np.ndarray, m: int, directions: list[tuple[int, int]]) -> set[int]:
    """Attacker stones collinear with m (within WIN_LEN-1) along the given axes.

    These are the threat's *rest* squares: the already-present attacker stones the
    threat leans on, the gain square m excepted.
    """
    rest: set[int] = set()
    mr, mc = rc(m)
    for (dr, dc) in directions:
        for s in (1, -1):
            for k in range(1, state_ops.WIN_LEN):
                rr, cc = mr + s * dr * k, mc + s * dc * k
                if not (0 <= rr < N and 0 <= cc < N):
                    break
                if attacker[rr, cc]:
                    rest.add(rr * N + cc)
                # keep walking through our own stones; stop at a gap of >1? We
                # walk the whole WIN_LEN-1 reach to catch gap-fours/threes.
    return rest


@dataclass
class Threat:
    gain: int                       # flat index of the attacker move
    cost: tuple[int, ...]           # flat indices defender must occupy
    rest: tuple[int, ...]           # flat indices of the threat's other own stones
    kind: str                       # 'five' | 'double_four' | 'four' | 'three_fork' | 'three'
    depth: int                      # ply within the forcing line (0 = root move)
    directions: tuple[tuple[int, int], ...] = ()
    n_threats: int = 0              # for threes: # distinct open-four threats


def classify_threat(attacker: np.ndarray, defender: np.ndarray, m: int) -> Threat:
    """Decompose the threat created by attacker move m (Allis gain/cost/rest).

    Assumes m is empty and legal. Returns a :class:`Threat`; ``depth`` is filled
    by the replayer. Never mutates the inputs.
    """
    atk = attacker.copy()
    occupied = attacker | defender
    empty_plane = ~occupied
    mr, mc = rc(m)

    # Immediate five? (m completes >=5)
    if m in _five_completions(attacker, empty_plane):
        dirs = []
        atk[mr, mc] = True
        # direction(s) along which five formed -- for rest stones
        for (dr, dc) in _DIRS:
            cnt = 1
            for s in (1, -1):
                rr, cc = mr + s * dr, mc + s * dc
                while 0 <= rr < N and 0 <= cc < N and atk[rr, cc]:
                    cnt += 1
                    rr += s * dr
                    cc += s * dc
            if cnt >= state_ops.WIN_LEN:
                dirs.append((dr, dc))
        rest = _rest_stones(atk, m, dirs)
        return Threat(m, (), tuple(sorted(rest)), "five", -1, tuple(dirs))

    # Place m, look for a four (>=1 collinear five-completion).
    atk[mr, mc] = True
    occ2 = atk | defender
    comps = _completions_through(atk, m, occ2)
    if comps:
        dirs = []
        for c in comps:
            ax = _axis(m, c)
            if ax is not None and ax not in dirs:
                dirs.append(ax)
        rest = _rest_stones(atk, m, dirs if dirs else list(_DIRS))
        kind = "double_four" if len(comps) >= 2 else "four"
        return Threat(m, tuple(sorted(comps)), tuple(sorted(rest)), kind, -1,
                      tuple(dirs), n_threats=len(comps))

    # Not a four -> forcing-three test: does m enable an open four next move?
    new_empty = ~occ2
    of_cands = _collinear_empties(m, new_empty)
    threats = _open_four_threats(atk, defender, new_empty, of_cands)
    if threats:
        # cost = the defeating set of the threatened open four(s): each threat
        # (f, comps) is stopped only by occupying f or one of comps.
        cost: set[int] = set()
        dirs: list[tuple[int, int]] = []
        for (f, fcomps) in threats:
            cost.add(f)
            cost.update(fcomps)
            ax = _axis(m, f)
            if ax is not None and ax not in dirs:
                dirs.append(ax)
        rest = _rest_stones(atk, m, dirs if dirs else list(_DIRS))
        # open three ~ a three threatening straight-four at BOTH ends => >=2
        # distinct open-four squares; broken/blocked three usually one.
        n_distinct_f = len({f for f, _ in threats})
        kind = "three_fork" if n_distinct_f >= 2 else "three"
        return Threat(m, tuple(sorted(cost)), tuple(sorted(rest)), kind, -1,
                      tuple(dirs), n_threats=len(threats))

    # Not obviously a threat (shouldn't happen on a principal VCT line).
    return Threat(m, (), (), "other", -1, ())


@dataclass
class ReplayResult:
    threats: list[Threat] = field(default_factory=list)
    junctions: list[int] = field(default_factory=list)   # squares where >=2 threats meet
    branch_points: int = 0                                # # defender AND-branch nodes
    mate_len: int = 0
    truncated: bool = False


def replay_vct(board: np.ndarray, *, max_plies: int = 12,
               vct_depth: int = 7, vct_nodes: int = 20_000,
               vcf_depth: int = 14, vcf_nodes: int = 50_000,
               solver: str = "vct") -> ReplayResult:
    """Replay the principal VCT forcing line, emitting one Threat per attacker move.

    Defender branching: a single four has a unique forced block; a three opens a
    real AND-node -- we follow the principal continuation (the defender reply that
    KEEPS the attacker's proven win, picked to maximise mate distance so we walk
    the full line) and count the branch point. A double/open four is unstoppable:
    we record it as the terminal fork and stop.

    ``solver='vcf'`` uses the (much faster) four-only solver for the principal
    move -- correct for VCF-tagged corpus positions (their whole line is fours)
    and ~10x cheaper than re-running VCT every ply.
    """
    attacker = np.ascontiguousarray(board[0], dtype=bool).copy()
    defender = np.ascontiguousarray(board[1], dtype=bool).copy()
    out = ReplayResult()

    def _solve(a, d):
        if solver == "vcf":
            return solve_vcf(np.stack([a, d]), max_depth=vcf_depth, max_nodes=vcf_nodes)
        return solve_vct(np.stack([a, d]), max_depth=vct_depth, max_nodes=vct_nodes)

    for depth in range(max_plies):
        res = _solve(attacker, defender)
        if not res.has_forced_win or res.winning_move is None:
            # winning_move is None only for already-five roots; otherwise stop.
            break
        m = res.winning_move
        t = classify_threat(attacker, defender, m)
        t.depth = depth
        out.threats.append(t)
        mr, mc = rc(m)
        attacker[mr, mc] = True

        if t.kind == "five":
            out.mate_len = depth + 1
            break
        if t.kind in ("double_four", "three_fork") and len(t.cost) >= 2:
            # Unstoppable fork: defender can't cover all completions. Terminal.
            # (three_fork with disjoint threats == the VCT double-threat win.)
            if t.kind == "double_four":
                out.mate_len = depth + 1
                break
        if t.kind == "four":
            # unique forced block
            block = t.cost[0]
            br, bc = rc(block)
            defender[br, bc] = True
            continue

        # three / three_fork: AND-node. Pick the principal defender reply.
        out.branch_points += 1
        best_d, best_dist = None, -1
        for d in t.cost:
            dr, dc = rc(d)
            if attacker[dr, dc] or defender[dr, dc]:
                continue
            defender[dr, dc] = True
            sub = _solve(attacker, defender)
            defender[dr, dc] = False
            if sub.has_forced_win:
                dist = sub.mate_distance or 0
                if dist > best_dist:
                    best_dist, best_d = dist, d
        if best_d is None:
            # No defender reply keeps it a win => attacker already winning here
            # (disjoint fork). Terminal.
            out.mate_len = depth + 1
            break
        bdr, bdc = rc(best_d)
        defender[bdr, bdc] = True
    else:
        out.truncated = True

    # Junctions: squares appearing in (gain|cost) of >= 2 distinct threats.
    from collections import Counter
    cnt: Counter = Counter()
    for t in out.threats:
        for s in {t.gain, *t.cost}:
            cnt[s] += 1
    out.junctions = sorted(s for s, c in cnt.items() if c >= 2)
    return out


# Re-export for convenience.
__all__ = ["Threat", "ReplayResult", "classify_threat", "replay_vct", "rc",
           "solve_vct", "solve_vcf"]
