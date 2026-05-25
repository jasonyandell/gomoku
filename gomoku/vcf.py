"""Exact VCF (Victory-by-Continuous-Fours) solver for 9x9 freestyle gomoku.

Freestyle rules: first to >=5-in-a-row wins; no overline restriction; no
opening restrictions. Under these rules a *four* (a line of attacker stones
with at least one empty cell that would complete five) is an absolutely
forcing threat: the defender must occupy the completion cell on the very next
ply or lose immediately. VCF is the search over attacker moves that each make
a four, with the defender's replies forced. If such a forcing sequence ends in
five-in-a-row, the position is an exact forced win and we have proved it with
no network estimate involved.

Why this matters for training: a 100-sim MCTS rollout only *estimates* the
value/policy on tactically forced positions, and our anchored elo saturates
around 1700 on exactly these tactics. A VCF proof turns a forced-win position
into a perfect label (winning move + value +1), attacking the ceiling at its
tactical root.

CORRECTNESS IS PARAMOUNT. A false "forced win" poisons training. The solver is
therefore conservative: it only ever returns ``has_forced_win=True`` when it
has *constructed* an explicit forcing line to five. The two soundness rules:

  1. A candidate attacker four is only forcing if the defender cannot, on the
     reply ply, make their OWN five somewhere (which would win first / escape).
     We check this before recursing.
  2. The defender's reply is taken as forced (the unique block) ONLY in the
     |completion|==1 case. If the attacker move makes a *double* four
     (|completion| >= 2) it is an immediate win provided the defender has no
     instant five available.

Board convention (matches ``gomoku.game.GameState``): a position is a
``(2, 9, 9)`` boolean array where plane 0 is the side-to-move's (attacker's)
stones and plane 1 is the opponent's (defender's) stones. The solver never
mutates the input.

The search is threat-only alpha-beta (really an AND/OR proof search: attacker
OR-nodes pick a four, defender AND-nodes are forced) bounded by a maximum
depth (in attacker moves) and a global node cap so it can never hang.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gomoku import state_ops

BOARD_SIZE = state_ops.BOARD_SIZE
WIN_LEN = state_ops.WIN_LEN

# Defaults chosen so a single position solve stays well under a millisecond-ish
# in the common (no-win / shallow-win) case while still proving the long
# continuous-four ladders that appear in 9x9 freestyle. Both are hard caps:
# hitting either returns "no proof found" (never a false positive).
DEFAULT_MAX_DEPTH = 16          # attacker moves deep (each = our four + their block)
DEFAULT_MAX_NODES = 200_000     # global node budget across the whole solve

# 4 line directions; the opposite directions are covered by scanning both ways
# from a placed stone, so these four cover all 8 ray directions.
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True)
class VCFResult:
    """Outcome of a VCF solve from a side-to-move position.

    ``has_forced_win`` is True only when an explicit forcing line to five was
    constructed. ``winning_move`` is the first attacker move of that line (flat
    action index in ``[0, 81)``), or ``None`` when no win was proved.
    ``mate_distance`` is the number of attacker moves to reach five along the
    found line (1 == the move makes five immediately), or ``None``.
    ``nodes`` and ``hit_cap`` are diagnostics; ``hit_cap`` True means the search
    exhausted its node/depth budget before completing (so a "no win" is
    "unproven", not "proven safe" — but it is still never a false positive).
    """

    has_forced_win: bool
    winning_move: int | None
    mate_distance: int | None
    nodes: int = 0
    hit_cap: bool = False


def _empties(board: np.ndarray) -> np.ndarray:
    """Flat indices of empty cells."""
    occupied = board[0] | board[1]
    return np.flatnonzero(~occupied.reshape(-1))


def _five_completions(stones: np.ndarray, empty: np.ndarray) -> list[int]:
    """Return the empty cells where placing a `stones` stone makes >=5-in-a-row.

    `stones` is a (9, 9) bool plane (one side's stones). `empty` is a (9, 9)
    bool plane of currently-empty cells (cells we are allowed to consider). A
    cell is a completion iff, looking along each of the 4 axes through it, the
    run of `stones` extending in the two opposite directions plus this cell
    totals >= WIN_LEN.

    Freestyle: an overline (6+) also counts as a win, so ">=" not "==".
    """
    completions: list[int] = []
    n = BOARD_SIZE
    flat_empty = np.flatnonzero(empty.reshape(-1))
    for idx in flat_empty:
        r, c = divmod(int(idx), n)
        for dr, dc in _DIRS:
            count = 1  # the cell we'd place at
            # forward
            rr, cc = r + dr, c + dc
            while 0 <= rr < n and 0 <= cc < n and stones[rr, cc]:
                count += 1
                rr += dr
                cc += dc
            # backward
            rr, cc = r - dr, c - dc
            while 0 <= rr < n and 0 <= cc < n and stones[rr, cc]:
                count += 1
                rr -= dr
                cc -= dc
            if count >= WIN_LEN:
                completions.append(int(idx))
                break
    return completions


def _candidate_cells(board: np.ndarray, empty_idx: np.ndarray) -> np.ndarray:
    """Restrict move generation to empty cells adjacent (incl. diagonal, within
    distance 2) to any existing stone. A four can only be made next to existing
    stones, so this prunes the branching factor hugely without affecting
    correctness (an isolated stone can never be part of a four-making move).

    Distance 2 (Chebyshev) is the right radius: a stone two cells away along a
    line can still be part of the four that a placement completes (e.g. the
    pattern X.X.X where the gap two away matters). Using 2 keeps us sound; a
    smaller radius could miss a gap-four.
    """
    n = BOARD_SIZE
    occupied = board[0] | board[1]
    if not occupied.any():
        return np.empty(0, dtype=np.int64)
    # Dilate the occupied mask by Chebyshev distance 2.
    near = np.zeros((n, n), dtype=bool)
    occ_rows, occ_cols = np.nonzero(occupied)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            rr = occ_rows + dr
            cc = occ_cols + dc
            valid = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
            near[rr[valid], cc[valid]] = True
    near &= ~occupied
    near_flat = near.reshape(-1)
    return empty_idx[near_flat[empty_idx]]


def _has_immediate_five(stones: np.ndarray, empty: np.ndarray) -> bool:
    """True if `stones` side has any single move that completes five."""
    return len(_five_completions(stones, empty)) > 0


def solve_vcf(
    board: np.ndarray,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> VCFResult:
    """Solve VCF for the side to move (plane 0 = attacker, plane 1 = defender).

    Returns a :class:`VCFResult`. ``has_forced_win`` is True only if an explicit
    continuous-four line to five was found. Never mutates ``board``.

    The position is assumed non-terminal (no side already has five). If the
    attacker already has an immediate five it is reported as a depth-1 win.
    """
    board = np.ascontiguousarray(board, dtype=bool)
    attacker = board[0].copy()
    defender = board[1].copy()

    # Defensive sanity: if the defender already has five, this isn't a position
    # the attacker is to-move into legally; report no win rather than guess.
    if state_ops.has_five_in_a_row(defender):
        return VCFResult(False, None, None, nodes=0, hit_cap=False)
    if state_ops.has_five_in_a_row(attacker):
        # Already won (shouldn't happen for a side *to move*, but be safe).
        return VCFResult(True, None, 0, nodes=0, hit_cap=False)

    counter = {"nodes": 0, "hit_cap": False}

    move, dist = _attack(attacker, defender, 1, max_depth, max_nodes, counter)
    if move is not None:
        return VCFResult(True, move, dist, nodes=counter["nodes"],
                         hit_cap=counter["hit_cap"])
    return VCFResult(False, None, None, nodes=counter["nodes"],
                     hit_cap=counter["hit_cap"])


def _attack(
    attacker: np.ndarray,
    defender: np.ndarray,
    depth: int,
    max_depth: int,
    max_nodes: int,
    counter: dict,
) -> tuple[int | None, int | None]:
    """OR-node: attacker tries four-making moves. Returns (winning_move, dist)
    where dist is the number of attacker moves remaining to mate along the line,
    or (None, None) if no forced win is provable within budget from here.

    `attacker`/`defender` are (9,9) bool planes (NOT flipped — attacker is
    always plane-0 semantics regardless of recursion depth, because we track
    the two planes explicitly).
    """
    counter["nodes"] += 1
    if counter["nodes"] >= max_nodes:
        counter["hit_cap"] = True
        return None, None
    if depth > max_depth:
        counter["hit_cap"] = True
        return None, None

    empty_plane = ~(attacker | defender)
    empty_idx = _empties_from_plane(empty_plane)
    if len(empty_idx) == 0:
        return None, None

    # 1. Immediate win: does the attacker have a move that makes five right now?
    #    (This covers the |completion|>=1 "straight win" without needing a block.)
    immediate = _five_completions(attacker, empty_plane)
    if immediate:
        return int(immediate[0]), 1

    # 2. Enumerate four-making moves among nearby empty cells. A move is a four
    #    iff after placing it the attacker gains >=1 five-completion. We order
    #    double-fours (>=2 completions) first: they are immediate wins (defender
    #    can only block one) provided the defender has no instant five.
    candidates = _candidate_cells_from_planes(attacker, defender, empty_idx)

    # Pre-compute the defender's current immediate-five squares once: if the
    # defender can make five on their reply we must ensure our four does not
    # simply hand them the move. (Their completion set only shrinks when we or
    # they occupy a cell, so the pre-move set is an upper bound we re-check.)
    four_moves: list[tuple[int, list[int]]] = []
    for m in candidates:
        attacker[m // BOARD_SIZE, m % BOARD_SIZE] = True
        comps = _five_completions(attacker, empty_plane & _without(m))
        attacker[m // BOARD_SIZE, m % BOARD_SIZE] = False
        if comps:
            four_moves.append((int(m), comps))

    if not four_moves:
        return None, None

    # Double-fours first (immediate win), then single fours.
    four_moves.sort(key=lambda mc: 0 if len(mc[1]) >= 2 else 1)

    for m, comps in four_moves:
        mr, mc = m // BOARD_SIZE, m % BOARD_SIZE
        attacker[mr, mc] = True
        try:
            new_empty = ~(attacker | defender)

            # Defender soundness check: can the defender make their OWN five now?
            # If so, this four does not force them to block — they win/escape.
            if _has_immediate_five(defender, new_empty):
                continue

            if len(comps) >= 2:
                # Double four: defender blocks at most one completion; the
                # attacker completes another next move. Defender has no instant
                # five (checked above), so this is a proven win in 2 attacker
                # moves (this move + the completion).
                return m, 2

            # Single four: the defender is FORCED to occupy the unique
            # completion square (else attacker makes five). Recurse with the
            # block placed for the defender.
            block = comps[0]
            br, bc = block // BOARD_SIZE, block % BOARD_SIZE
            defender[br, bc] = True
            try:
                sub_move, sub_dist = _attack(
                    attacker, defender, depth + 1, max_depth, max_nodes, counter
                )
            finally:
                defender[br, bc] = False
            if sub_move is not None:
                return m, (sub_dist + 1) if sub_dist is not None else None
            if counter["hit_cap"] and counter["nodes"] >= max_nodes:
                # Out of budget; stop exploring siblings to bound work.
                return None, None
        finally:
            attacker[mr, mc] = False

    return None, None


# ---- small plane helpers (kept tiny + local for hot-loop clarity) ----

def _empties_from_plane(empty_plane: np.ndarray) -> np.ndarray:
    return np.flatnonzero(empty_plane.reshape(-1))


def _without(m: int) -> np.ndarray:
    """A (9,9) bool plane that is True everywhere except cell `m`."""
    mask = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    mask[m // BOARD_SIZE, m % BOARD_SIZE] = False
    return mask


def _candidate_cells_from_planes(
    attacker: np.ndarray, defender: np.ndarray, empty_idx: np.ndarray
) -> np.ndarray:
    n = BOARD_SIZE
    occupied = attacker | defender
    if not occupied.any():
        return np.empty(0, dtype=np.int64)
    near = np.zeros((n, n), dtype=bool)
    occ_rows, occ_cols = np.nonzero(occupied)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            rr = occ_rows + dr
            cc = occ_cols + dc
            valid = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
            near[rr[valid], cc[valid]] = True
    near &= ~occupied
    near_flat = near.reshape(-1)
    return empty_idx[near_flat[empty_idx]]


def solve_vcf_from_planes(
    planes: np.ndarray,
    *,
    history_ply: int = state_ops.HISTORY_PLY,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> VCFResult:
    """Convenience: solve VCF from a network input-plane stack.

    The input-plane layout (see ``GameState.to_planes``) puts the side-to-move's
    current stones at plane 0 and the opponent's current stones at plane
    ``history_ply``. We reconstruct the ``(2, 9, 9)`` board from those two planes
    (mirrors ``self_play._gamestate_from_archive``) and solve.
    """
    planes = np.asarray(planes)
    attacker = planes[0].astype(bool)
    defender = planes[history_ply].astype(bool)
    board = np.stack([attacker, defender], axis=0)
    return solve_vcf(board, max_depth=max_depth, max_nodes=max_nodes)
