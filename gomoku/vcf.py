"""Exact VCF (Victory-by-Continuous-Fours) solver for 9x9 freestyle gomoku.

=============================================================================
OBSOLETE — REFERENCE / HISTORY ONLY. Superseded by the GPU megakernel.
=============================================================================
This pure-CPU recursive AND/OR solver (``solve_vcf`` / ``solve_vct`` and their
``*_from_planes`` wrappers) is SUPERSEDED for corpus VCT/VCF labeling and
shape-mining by the fully on-device GPU megakernel
``scripts/vct_metal/mega_vct_bb.py :: solve_vct_mega_bb`` (VCF twin:
``scripts/vct_metal/mega_vcf_bb.py :: solve_vcf_mega_bb``). See
``wiki/topics/gpu-vct-feasibility.md`` §8 and ``wiki/topics/vct-backward-mining.md``.

WHY IT IS RETIRED (not merely slower):
  * Soundness was proven EQUIVALENT, so the kernel no longer needs this oracle.
    Validation (wiki §8): the bitboard megakernel matched this solver with
    **0 false-positive / 0 false-negative** verdicts over **320 real VCT
    positions** (258 clean, non-cap agreements, 8 seeds), and its VCF twin over
    **360 real VCF positions** — never a phantom win in either direction. The
    only divergences are cap-boundary positions (excluded from the "clean"
    count), always the SAFE direction: the GPU's full expansion can PROVE a few
    roots this DFS abandons at its node cap, never the reverse, never a false
    positive. (One real over-generation bug in the kernel — far pre-existing
    fours spawning spurious threes — was caught and fixed against this solver
    during that bring-up; the equivalence above is the post-fix result.)
  * Performance black-hole: single-thread ``solve_vct`` is **~0.64 solves/s
    aggregate** and sharply BIMODAL — median ~30 ms (~33 solves/s typical) but
    ~14% of positions hit a **~90 s tail** at the production cap
    (``max_nodes=20000``); at the deeper depth-10 / ``max_nodes=100k`` mining
    config that tail stretches to **4–16+ min ("~20-min") "monster" boards**.
    The megakernel does ~850–1020 solves/s at B≈16k–32k (**~1600× aggregate**)
    precisely because its flat-in-batch wall hides that single-deep-board tail.
    This is the slow oracle Jason does NOT want to keep hand-syncing to an
    actively-evolving kernel.

RULES for this file going forward:
  * Kept ONLY for reference / history / readable-spec value — the megakernel and
    its ``scripts/vct_metal/*_ref.py`` validation scaffolding were derived from
    these exact helpers and soundness rules.
  * It will NOT be kept in sync as the megakernel evolves — expect it to DRIFT.
  * Do NOT use it to verify or cross-check the GPU solver. That equivalence is
    already established and archived (above); re-pinning the kernel to this
    frozen reference would only hold it back.

MIGRATION of the remaining callers (the eventual path):
  * Offline VCT mining / labeling — ``scripts/threat_shapes/mine_first_vct.py``,
    ``mine_vct_backward.py``, ``mine_vct_gpu.py``, ``mine_vct_gpu_flat.py``,
    ``mine_parallel.py``, ``build_corpus.py``, ``threats.py`` (and the v0
    ``scripts/gpu_vct_prototype.py``) — move to
    ``solve_vct_mega_bb(..., return_move=True)`` (its passive GPU root-move
    output, wiki §5). NB: that returns a *valid* (sound) first VCT move, not
    necessarily the *shortest* one ``solve_vct`` reports — a benign behavioral
    difference for move-extraction consumers.
  * Still-LIVE in-loop callers use the cheaper VCF path (``solve_vcf`` /
    ``has_four_threat``), NOT VCT: the self-play / MCTS / eval teachers
    (``gomoku/self_play.py``, ``gomoku/mcts.py``, ``gomoku/eval.py``,
    ``gomoku/white_defense.py``). They do single-position checks inside the
    training loop where a per-process batch GPU call does not yet fit — the last
    reason this module still imports, and the harder migration.
=============================================================================

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

import os
from dataclasses import dataclass

import numpy as np

from gomoku import state_ops


class CpuSolverRetired(RuntimeError):
    """Raised when a RETIRED CPU vcf solver entry point is reached at runtime.

    The CPU solver (``solve_vcf`` / ``solve_vct`` and their ``*_from_planes``
    wrappers) is retired as a *runtime* dependency. Set the env var
    ``GOMOKU_ALLOW_CPU_SOLVER=1`` to deliberately use it as a bootstrap/oracle
    (fixture regen / deep validation). See ``wiki/topics/mega-vct-solver.md``.
    """


_CPU_SOLVER_RETIRED_MSG = (
    "The CPU vcf solver is RETIRED as a runtime dependency (slow: ~0.65 ms/node, "
    "~90s tail on hard 15x15 boards). Use "
    "scripts.vct_metal.mega_vct_bb.solve_vct_mega_bb — the on-device GPU solver "
    "(~1600x, 0 FP/FN). The CPU solver is kept only as a bootstrap/oracle; to use "
    "it deliberately (fixture regen / deep validation) set env "
    "GOMOKU_ALLOW_CPU_SOLVER=1. See wiki/topics/mega-vct-solver.md."
)


def _check_cpu_solver_gate() -> None:
    """Throw unless the deliberate-use override is set.

    Bypassed iff ``GOMOKU_ALLOW_CPU_SOLVER == "1"``. Gates only the PUBLIC entry
    points; every internal helper stays fully intact (the kernel/ref scaffolding
    and the sanctioned fixture-gen / deep-validation paths import them directly).
    """
    if os.environ.get("GOMOKU_ALLOW_CPU_SOLVER") != "1":
        raise CpuSolverRetired(_CPU_SOLVER_RETIRED_MSG)

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


def has_four_threat(board: np.ndarray) -> bool:
    """Cheap, SOUND necessary-condition pre-scan for a VCF win by plane 0.

    A continuous-four win ALWAYS opens with a four: the attacker's very first
    move must either make five outright or create a five-completion (a four). If
    plane 0 (the attacker) has no move that yields any five-completion, no VCF
    win can exist, so a caller may safely skip the (expensive) full solve.

    This is a single, non-recursive pass over the near-stone candidate cells
    using the same collinear-completion prune the solver uses (one place at a
    cell, count completions through it, undo), so it is far cheaper than
    `solve_vcf` (which would enumerate the same fours AND recurse). It is the
    DANGER pre-scan for the defensive teacher: gate the swapped-plane solve on
    this so a QUIET position (the side-to-move's opponent has no four-making
    move at all) costs ZERO solver recursion.

    SOUNDNESS: returns False only when plane 0 has no four-making move. Since
    every VCF line opens with a four, a False here is a guaranteed
    `has_forced_win=False` from the full solver, so gating on it only ever skips
    provably-losing-free work — it never suppresses a win the solver would have
    proved, and (critically) never invents one. Never mutates ``board``.
    """
    board = np.ascontiguousarray(board, dtype=bool)
    attacker = board[0].copy()
    defender = board[1].copy()
    empty_plane = ~(attacker | defender)
    empty_idx = _empties_from_plane(empty_plane)
    if len(empty_idx) == 0:
        return False
    # Immediate five (attacker already has an open four) is a four threat.
    if _five_completions(attacker, empty_plane):
        return True
    occupied = attacker | defender
    candidates = _candidate_cells_from_planes(attacker, defender, empty_idx)
    for m in candidates:
        mr, mc = int(m) // BOARD_SIZE, int(m) % BOARD_SIZE
        attacker[mr, mc] = True
        occupied[mr, mc] = True
        comps = _completions_through(attacker, int(m), occupied)
        attacker[mr, mc] = False
        occupied[mr, mc] = False
        if comps:
            return True
    return False


def solve_vcf(
    board: np.ndarray,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> VCFResult:
    """Solve VCF for the side to move (plane 0 = attacker, plane 1 = defender).

    OBSOLETE (see module banner): superseded for batch labeling by the GPU
    megakernel ``scripts/vct_metal/mega_vcf_bb.py :: solve_vcf_mega_bb`` (0 FP/0 FN
    vs this over 360 real positions). Still imported by the in-loop VCF teachers
    (self_play / mcts / eval); NOT to be used as a GPU cross-check oracle.

    Returns a :class:`VCFResult`. ``has_forced_win`` is True only if an explicit
    continuous-four line to five was found. Never mutates ``board``.

    The position is assumed non-terminal (no side already has five). If the
    attacker already has an immediate five it is reported as a depth-1 win.
    """
    _check_cpu_solver_gate()
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


def _refutation_candidates(attacker: np.ndarray, defender: np.ndarray,
                           winning_move: int | None) -> list[int]:
    """The small, THREAT-RELEVANT set of defender moves that could break the
    attacker's forced VCF (issue #43, gen-cost bound).

    A defender (moving first, one tempo before the attacker) can only refute a
    continuous-four win by interfering with its OPENING four or by seizing the
    initiative with a counter-threat. So the only moves worth testing are:

      1. ``winning_move`` itself — deny the attacker its first forcing move.
      2. the completion square(s) of the four that ``winning_move`` makes — the
         literal block of that four (pre-occupying a completion makes the
         attacker's move no longer forcing).
      3. the defender's OWN four/five-making moves — a counter-four forces the
         attacker to respond, breaking the VCF line (and a five just wins).

    This replaces the old "all near-stone empties" candidate set, which on a dense
    mid-game 15x15 board is ~40 cells × a full re-solve each PER FIRE PER PLY — the
    measured throughput blowup (8 self-play games did not finish in 120 s). The
    targeted set is typically <10 cells. It can MISS a refutation that works only
    by disrupting a LATER move in the forcing line (rare); that is the safe
    direction — the teacher then simply does not fire (no false saving move, since
    every candidate is still re-solve-verified by the caller). Never mutates planes.
    """
    cand: set[int] = set()
    occupied = attacker | defender
    if winning_move is not None:
        wr, wc = winning_move // BOARD_SIZE, winning_move % BOARD_SIZE
        if not occupied[wr, wc]:
            cand.add(int(winning_move))
        # The four that winning_move makes: block its completion square(s).
        atk2 = attacker.copy()
        atk2[wr, wc] = True
        occ2 = atk2 | defender
        for comp in _completions_through(atk2, int(winning_move), occ2):
            cr, cc = comp // BOARD_SIZE, comp % BOARD_SIZE
            if not occupied[cr, cc]:
                cand.add(int(comp))
    # Defender counter-threats: any move giving the defender a four/five.
    empty_idx = _empties_from_plane(~occupied)
    for m in _candidate_cells_from_planes(attacker, defender, empty_idx):
        mr, mc = int(m) // BOARD_SIZE, int(m) % BOARD_SIZE
        defender[mr, mc] = True
        occupied[mr, mc] = True
        comps = _completions_through(defender, int(m), occupied)
        defender[mr, mc] = False
        occupied[mr, mc] = False
        if comps:
            cand.add(int(m))
    return sorted(cand)


def vcf_refutations(
    board: np.ndarray,
    *,
    winning_move: int | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_candidates: int | None = None,
) -> list[int]:
    """Defender refutation moves that BREAK a forced VCF win (issue #43).

    ``board`` is a ``(2, N, N)`` position where plane 0 (attacker) has a proven
    forced VCF win *as the side to move* but, in the real game, plane 1 (the
    DEFENDER) moves FIRST — one tempo before the attacker. Return the sorted flat
    indices of defender moves ``d`` such that, after the defender plays ``d``, the
    attacker (now to move) NO LONGER has a proven forced VCF win. An empty list
    means no refutation exists within budget (the position is genuinely lost).

    This is the "saving move" the defensive teacher stamps on the policy head: it
    is the generalizable lesson ("here is the move that refuses the forced four")
    rather than the value-only "you already lost" signal.

    ``winning_move`` is the attacker's first forcing move (from the detection
    :func:`solve_vcf`); pass it to avoid a redundant solve. When ``None`` it is
    recovered with one internal solve (and an empty result if there is no win).

    CANDIDATE SET: only the THREAT-RELEVANT moves (:func:`_refutation_candidates`)
    are tested — the opening four's block squares + the defender's own
    four/five-making moves — NOT every near-stone empty. This is the gen-cost
    bound (see that helper); it trades a little recall (may miss a refutation that
    only disrupts a later move in the line) for ~8x fewer re-solves on dense
    boards. The safe direction: a missed refutation just means the teacher does
    not fire.

    SOUNDNESS: a refutation is only reported when an explicit re-solve proves the
    attacker has *no* forced VCF after the defender's move AND that re-solve
    COMPLETED within budget (``hit_cap`` False). The cap guard is essential at the
    tight gen-time budgets self-play uses: a cap-hit ``has_forced_win=False`` is
    "unproven", NOT "proven safe", so trusting it as a refutation could stamp a
    move that does not actually escape — the unsound direction. Requiring a
    completed search means a reported move is a genuine escape from the detected
    VCF (no false saving moves) regardless of budget. ``max_candidates`` (None =
    all) further bounds the worst case. Never mutates ``board``.
    """
    board = np.ascontiguousarray(board, dtype=bool)
    attacker = board[0].copy()
    defender = board[1].copy()
    if winning_move is None:
        res0 = solve_vcf(board, max_depth=max_depth, max_nodes=max_nodes)
        if not res0.has_forced_win:
            return []
        winning_move = res0.winning_move
    candidates = _refutation_candidates(attacker, defender, winning_move)
    if max_candidates is not None and len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    saving: list[int] = []
    for d in candidates:
        dr, dc = int(d) // BOARD_SIZE, int(d) % BOARD_SIZE
        defender[dr, dc] = True
        try:
            test = np.stack([attacker, defender], axis=0)
            res = solve_vcf(test, max_depth=max_depth, max_nodes=max_nodes)
        finally:
            defender[dr, dc] = False
        # PROVEN escape only: no forced win AND the search completed (a cap-hit
        # "no win" is unproven and must NOT be trusted as a refutation).
        if not res.has_forced_win and not res.hit_cap:
            saving.append(int(d))
    return sorted(saving)


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
    _check_cpu_solver_gate()
    planes = np.asarray(planes)
    attacker = planes[0].astype(bool)
    defender = planes[history_ply].astype(bool)
    board = np.stack([attacker, defender], axis=0)
    return solve_vcf(board, max_depth=max_depth, max_nodes=max_nodes)


# ===========================================================================
# VCT (Victory by Continuous Threats): fours *and* forcing threes.
# ===========================================================================
#
# VCF only uses fours, which are absolutely forcing: a four leaves a unique
# completion the defender must occupy on the very next ply or lose. VCT widens
# the attacker's threat set to also include FORCING THREES — a move that makes a
# three which can be extended into an *open four* (two completion squares, hence
# unstoppable) on the attacker's next move. An open four is a one-ply win the
# defender cannot block, so a three that threatens an open four is itself a real
# threat the defender must answer.
#
# CORRECTNESS, the whole point: a false "forced win" poisons training, so the
# VCT solver is at least as conservative as VCF. The two new soundness rules on
# top of VCF's:
#
#   3. A four is forcing (defender pinned to one square); a THREE is NOT. After
#      an attacker three the defender may play *any* legal move — block either
#      end of the would-be open four, block the three itself, or ignore it and
#      start their own counterplay. So a three is only a proven win if the
#      attacker wins against EVERY defender reply (a real AND-node), not just
#      against the "obvious block".
#
#   4. Tempo guard: a three does not threaten the defender's *next* ply, so the
#      defender is free to seize the initiative with a four (or five) of their
#      own, which would force the ATTACKER to respond and break the VCT line. We
#      do NOT try to refute defender counter-fours; instead, conservatively, if
#      the defender has ANY four-making (or five-making) move available at the
#      three's AND-node we ABANDON that three (treat it as non-forcing). This can
#      only make us miss wins, never invent them — exactly the bias we want.
#
# Because the three branch genuinely branches on the defender side (rule 3), the
# VCT tree is far larger than VCF's; the depth/node caps below are
# correspondingly more important and default lower in depth.

# VCT caps. Depth counts attacker moves (each attacker move + its forced/AND
# defender reply is one ply of depth). The node cap is the global hard budget;
# hitting either cap returns "no forced win found" (has_forced_win=False) and
# never a false positive. The defaults are deliberately CONSERVATIVE: the
# continuous-threes tree fans out on the defender side and explodes on
# wide-open boards, so these caps are the hard safety valve that guarantees a
# single solve always terminates quickly even if a caller forgets to pass caps.
# Tune UP for deeper proofs on a known-tactical position; never rely on the
# defaults to prove a long line on an open board.
DEFAULT_VCT_MAX_DEPTH = 7
DEFAULT_VCT_MAX_NODES = 20_000

# Per-node branching cap on forcing threes (sound: only ever misses wins). Real
# continuous-threes mates come from the strongest few threes at each node; the
# weak tail just balloons the tree. Together with the depth/node caps this keeps
# even a pathological wide-open position's solve well under a second.
_MAX_THREE_BRANCH = 6


def _collinear_empties(m: int, empty_plane: np.ndarray) -> list[int]:
    """Empty cells within distance 4 of cell ``m`` along the 4 axes through it.

    An open four that uses the stone just placed at ``m`` must lie on one of the
    4 lines through ``m`` and within 4 cells of it (a five spans 5 cells, so any
    completing placement is within 4 of ``m``). Restricting follow-up candidates
    to this small set (<= 32 cells) instead of every near-stone empty is a sound
    constant-factor prune for the forcing-three test.
    """
    n = BOARD_SIZE
    mr, mc = m // n, m % n
    out: list[int] = []
    for dr, dc in _DIRS:
        for sign in (1, -1):
            for k in range(1, WIN_LEN):
                rr, cc = mr + sign * dr * k, mc + sign * dc * k
                if 0 <= rr < n and 0 <= cc < n:
                    if empty_plane[rr, cc]:
                        out.append(rr * n + cc)
                else:
                    break
    return out


def _completions_through(stones: np.ndarray, m: int,
                         occupied: np.ndarray) -> list[int]:
    """Five-completion cells that lie on a line through ``m`` (collinear prune).

    Counts, for the cell ``m`` (assumed already a ``stones`` stone), the empty
    cells along the 4 axes through ``m`` whose occupation would complete a
    >= WIN_LEN run. ``occupied`` is the (9,9) bool plane of all occupied cells
    (either side); an empty completion cell must be free of BOTH sides. This is a
    sound localized replacement for scanning every empty: a five that uses the
    stone at ``m`` must run through ``m``, so its completing square is collinear
    with ``m`` and within WIN_LEN-1 cells. Returns distinct completion cells.
    """
    n = BOARD_SIZE
    mr, mc = m // n, m % n
    comps: set[int] = set()
    for dr, dc in _DIRS:
        # Walk both ways from m; for each empty cell within reach along this axis,
        # test whether placing there makes a >= WIN_LEN run.
        for sign in (1, -1):
            for k in range(1, WIN_LEN):
                rr, cc = mr + sign * dr * k, mc + sign * dc * k
                if not (0 <= rr < n and 0 <= cc < n):
                    break
                if occupied[rr, cc]:
                    # Blocked along this ray; but a stone of our own continues the
                    # run, so only stop on an *opponent/empty-then-stone*? Simpler
                    # and sound: stop scanning this ray at the first occupied that
                    # is not our stone. We re-test via full run-length below.
                    if stones[rr, cc]:
                        continue
                    break
                # Empty candidate completion cell: test it.
                if _placement_makes_five(stones, rr, cc, dr, dc):
                    comps.add(rr * n + cc)
    return list(comps)


def _placement_makes_five(stones: np.ndarray, r: int, c: int,
                          dr: int, dc: int) -> bool:
    """True if placing a ``stones`` stone at (r,c) makes >= WIN_LEN along (dr,dc)."""
    n = BOARD_SIZE
    count = 1
    rr, cc = r + dr, c + dc
    while 0 <= rr < n and 0 <= cc < n and stones[rr, cc]:
        count += 1
        rr += dr
        cc += dc
    rr, cc = r - dr, c - dc
    while 0 <= rr < n and 0 <= cc < n and stones[rr, cc]:
        count += 1
        rr -= dr
        cc -= dc
    return count >= WIN_LEN


def _open_four_threats(attacker: np.ndarray, defender: np.ndarray,
                       empty_plane: np.ndarray,
                       candidates: list[int]) -> list[tuple[int, list[int]]]:
    """Open-four threats the attacker can make on its NEXT move.

    Returns ``(f, comps)`` for each empty cell ``f`` where placing an attacker
    stone yields an OPEN four: ``comps`` are the ``>= 2`` distinct five-completion
    squares of that four (so the defender could not block them all). These are
    exactly the moves a forcing three is *threatening* to make next. Completions
    are computed via the collinear prune (see :func:`_completions_through`).
    """
    occupied = attacker | defender
    out: list[tuple[int, list[int]]] = []
    for f in candidates:
        fr, fc = int(f) // BOARD_SIZE, int(f) % BOARD_SIZE
        attacker[fr, fc] = True
        occupied[fr, fc] = True
        comps = _completions_through(attacker, int(f), occupied)
        attacker[fr, fc] = False
        occupied[fr, fc] = False
        if len(comps) >= 2:
            out.append((int(f), comps))
    return out


def _defender_has_four_or_five(defender: np.ndarray, attacker: np.ndarray,
                               empty_plane: np.ndarray) -> bool:
    """Tempo guard (rule 4): True if the defender can make a four or five.

    A defender four would force the attacker to respond, breaking a VCT three
    line; a defender five is an outright loss for the attacker. Either means the
    attacker's three is not safely forcing, so we abandon it. We reuse the
    four-detection logic: a move is a four/five iff after it the defender has
    >= 1 five-completion.
    """
    # A pre-existing defender four (a completion square already open) is also a
    # tempo threat: the defender just completes it. Catch that first so the
    # per-move collinear check below need only find NEW (through-m) completions —
    # together they cannot UNDER-detect, which would be the dangerous direction
    # (it would make the tempo guard too lenient and could admit an unsound three).
    if _five_completions(defender, empty_plane):
        return True
    occupied = attacker | defender
    candidates = _candidate_cells_from_planes(attacker, defender, _empties_from_plane(empty_plane))
    for m in candidates:
        mr, mc = int(m) // BOARD_SIZE, int(m) % BOARD_SIZE
        defender[mr, mc] = True
        occupied[mr, mc] = True
        comps = _completions_through(defender, int(m), occupied)
        defender[mr, mc] = False
        occupied[mr, mc] = False
        if comps:
            return True
    return False


def solve_vct(
    board: np.ndarray,
    *,
    max_depth: int = DEFAULT_VCT_MAX_DEPTH,
    max_nodes: int = DEFAULT_VCT_MAX_NODES,
) -> VCFResult:
    """Solve VCT for the side to move (plane 0 = attacker, plane 1 = defender).

    OBSOLETE (see module banner): superseded by the GPU megakernel
    ``scripts/vct_metal/mega_vct_bb.py :: solve_vct_mega_bb`` (0 FP/0 FN vs this
    over 320 real positions; ~1600× throughput). Use ``return_move=True`` there
    for the winning-move extraction the mining scripts need. Reference/history
    only — will drift; do NOT cross-check the kernel against it.

    Returns the same :class:`VCFResult` shape as :func:`solve_vcf`.
    ``has_forced_win`` is True only when an explicit forcing line (of fours
    and/or forcing threes) to five was constructed; otherwise False. On cap
    exhaustion it returns ``has_forced_win=False`` with ``hit_cap=True`` — an
    "unproven", never a false positive. Never mutates ``board``.

    Every VCF win is a VCT win (the four-only subset), so VCT is a strict
    superset of VCF: it solves all VCF positions plus continuous-threes mates.
    """
    _check_cpu_solver_gate()
    board = np.ascontiguousarray(board, dtype=bool)
    attacker = board[0].copy()
    defender = board[1].copy()

    if state_ops.has_five_in_a_row(defender):
        return VCFResult(False, None, None, nodes=0, hit_cap=False)
    if state_ops.has_five_in_a_row(attacker):
        return VCFResult(True, None, 0, nodes=0, hit_cap=False)

    counter = {"nodes": 0, "hit_cap": False}
    move, dist = _vct_attack(attacker, defender, 1, max_depth, max_nodes, counter)
    if move is not None:
        return VCFResult(True, move, dist, nodes=counter["nodes"],
                         hit_cap=counter["hit_cap"])
    return VCFResult(False, None, None, nodes=counter["nodes"],
                     hit_cap=counter["hit_cap"])


def _vct_attack(
    attacker: np.ndarray,
    defender: np.ndarray,
    depth: int,
    max_depth: int,
    max_nodes: int,
    counter: dict,
) -> tuple[int | None, int | None]:
    """OR-node: attacker tries fours then forcing threes. Returns (move, dist)
    or (None, None). ``dist`` is the count of attacker moves remaining to mate
    along the found line.

    Move ordering (best-first): immediate five, double-fours, single fours,
    forcing threes. Fours keep VCF's forced-single-block recursion; threes open
    a genuine defender AND-node (rule 3) with the tempo guard (rule 4).
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

    # 1. Immediate five.
    immediate = _five_completions(attacker, empty_plane)
    if immediate:
        return int(immediate[0]), 1

    candidates = _candidate_cells_from_planes(attacker, defender, empty_idx)
    occupied = attacker | defender

    # 2. Enumerate four-making moves (VCF-style), using the collinear-completion
    #    prune: a four made by placing at m has its completion squares on a line
    #    through m, so we only scan those (sound + much cheaper than rescanning
    #    every empty at every node).
    four_moves: list[tuple[int, list[int]]] = []
    for m in candidates:
        mr, mc = int(m) // BOARD_SIZE, int(m) % BOARD_SIZE
        attacker[mr, mc] = True
        occupied[mr, mc] = True
        comps = _completions_through(attacker, int(m), occupied)
        attacker[mr, mc] = False
        occupied[mr, mc] = False
        if comps:
            four_moves.append((int(m), comps))

    four_moves.sort(key=lambda mc: 0 if len(mc[1]) >= 2 else 1)
    four_set = {m for m, _ in four_moves}

    for m, comps in four_moves:
        mr, mc = m // BOARD_SIZE, m % BOARD_SIZE
        attacker[mr, mc] = True
        try:
            new_empty = ~(attacker | defender)
            if _has_immediate_five(defender, new_empty):
                continue
            if len(comps) >= 2:
                return m, 2  # double four: unstoppable
            block = comps[0]
            br, bc = block // BOARD_SIZE, block % BOARD_SIZE
            defender[br, bc] = True
            try:
                sub_move, sub_dist = _vct_attack(
                    attacker, defender, depth + 1, max_depth, max_nodes, counter
                )
            finally:
                defender[br, bc] = False
            if sub_move is not None:
                return m, (sub_dist + 1) if sub_dist is not None else None
            if counter["hit_cap"] and counter["nodes"] >= max_nodes:
                return None, None
        finally:
            attacker[mr, mc] = False

    # 3. Forcing threes: a move that does NOT make a four but, after which, the
    #    attacker has a follow-up move creating an OPEN four (>=2 completions).
    #    These genuinely branch on the defender (AND-node), so they're explored
    #    after fours. We first collect each three with its open-four threat list,
    #    then try them BEST-FIRST: a three creating MORE distinct open-four
    #    threats (a fork) is both more likely to win and cheaper to refute, so it
    #    is searched before single-threat threes (which the defender usually
    #    parries and which lead to deep speculative lines).
    threes: list[tuple[int, list[tuple[int, list[int]]]]] = []
    for m in candidates:
        if m in four_set:
            continue  # already covered as a four above
        mr, mc = m // BOARD_SIZE, m % BOARD_SIZE
        attacker[mr, mc] = True
        new_empty = ~(attacker | defender)
        # Soundness: the three must not hand the defender an immediate five.
        if _has_immediate_five(defender, new_empty):
            attacker[mr, mc] = False
            continue
        # The open four the three threatens must lie on a line THROUGH the
        # just-placed stone m (m is the new attacking stone that creates the
        # three). So we only need to test follow-up squares collinear with m and
        # within reach (distance <= 4) — a big constant-factor prune over scanning
        # every near-stone empty, and it preserves soundness (a follow-up off all
        # lines through m cannot extend m's three into a four).
        of_cands = _collinear_empties(m, new_empty)
        threats = _open_four_threats(attacker, defender, new_empty, of_cands)
        attacker[mr, mc] = False
        if not threats:
            continue  # not a forcing three
        threes.append((int(m), threats))

    # Most-threats-first (forks before single threats), then keep only the top
    # MAX_THREE_BRANCH per node. Real continuous-threes mates run through the
    # best (most-threatening) threes; the long tail of weak single-threes only
    # spawns deep speculative lines that explode the tree. Capping the branching
    # bounds the search hard and is SOUND — it can only make us miss a win, never
    # invent one (the no-false-positive guarantee is preserved).
    threes.sort(key=lambda mt: -len(mt[1]))
    del threes[_MAX_THREE_BRANCH:]

    for m, threats in threes:
        mr, mc = m // BOARD_SIZE, m % BOARD_SIZE
        attacker[mr, mc] = True
        try:
            new_empty = ~(attacker | defender)
            # Tempo guard (rule 4) FIRST: if the defender can make a four/five of
            # their own they seize the initiative (their counter-four forces the
            # attacker to respond instead of completing the open four), so the
            # three is not forcing. This MUST precede the fast double-threat win
            # below — a defender counter-four refutes even a disjoint fork.
            if _defender_has_four_or_five(defender, attacker, new_empty):
                continue
            # Fast double-threat win: with no defender counter-four/five (guarded
            # above), if two of the threatened open fours share NO common
            # defeating cell, the defender cannot neutralize both with one move,
            # so the attacker makes an unstoppable open four next move regardless
            # of the block. Proven win in 2 attacker moves; no recursion needed.
            if _has_disjoint_threats(threats):
                return m, 2

            # AND-node: the attacker wins only if it beats every defender reply
            # that could possibly save them (the bounded threat-defeating set;
            # see _vct_defend for the soundness argument).
            sub_dist = _vct_defend(
                attacker, defender, threats, depth + 1, max_depth, max_nodes, counter
            )
            if sub_dist is not None:
                return m, sub_dist + 1
            if counter["hit_cap"] and counter["nodes"] >= max_nodes:
                return None, None
        finally:
            attacker[mr, mc] = False

    return None, None


def _has_disjoint_threats(threats: list[tuple[int, list[int]]]) -> bool:
    """True if two open-four threats share no single defeating cell.

    A defender move neutralizes an open-four threat ``(f, comps)`` only by
    occupying ``f`` or one of ``comps`` (its defeating set). If two threats have
    disjoint defeating sets, no single defender stone can stop both, so the
    attacker's open four is unstoppable — an immediate (2-attacker-move) win.
    """
    sets = [set([f, *comps]) for f, comps in threats]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i].isdisjoint(sets[j]):
                return True
    return False


def _vct_defend(
    attacker: np.ndarray,
    defender: np.ndarray,
    threats: list[tuple[int, list[int]]],
    depth: int,
    max_depth: int,
    max_nodes: int,
    counter: dict,
) -> int | None:
    """AND-node: a forcing three was just played against the defender (plane 1),
    threatening the open fours in ``threats`` (``(f, comps)`` pairs from
    :func:`_open_four_threats`). Return the worst-case attacker mate distance
    over the defender's *non-losing* replies if the attacker wins against all of
    them, else None.

    SOUNDNESS — why a bounded reply set is sufficient (no false positives):
    The attacker threatens to play some ``f`` next, making an open four with
    completion squares ``comps`` (|comps| >= 2). A defender move can only avoid
    losing to that specific open four if it occupies ``f`` itself or reduces the
    open four's completions below 2 by occupying one of its ``comps``. A defender
    move touching NONE of the threats' ``f``/``comps`` squares leaves at least one
    open four intact, so the attacker plays it and wins (an open four is a
    one-ply, unblockable win). Therefore every defender reply OUTSIDE the union
    of all threats' ``{f} ∪ comps`` squares is an immediate loss for the defender
    and need not be searched — and the only replies that could possibly save the
    defender ARE inside that union. We require the attacker to win against every
    candidate in the union; if so, the defender has no escape at all and the win
    is proven. (The parent's tempo guard already ruled out defender counter-fours,
    so a "pass-like" reply cannot seize the initiative.)

    Hitting a cap, or failing to refute any one reply, returns None.
    """
    counter["nodes"] += 1
    if counter["nodes"] >= max_nodes:
        counter["hit_cap"] = True
        return None
    if depth > max_depth:
        counter["hit_cap"] = True
        return None

    empty_plane = ~(attacker | defender)
    if not empty_plane.any():
        return None

    # Bounded, sound reply set: the union of every threat's open-four square and
    # its completion squares. Any cell outside this set leaves some open four
    # intact => the defender loses to it, so we need not search it.
    reply_set: set[int] = set()
    for f, comps in threats:
        reply_set.add(f)
        reply_set.update(comps)
    # Keep only still-empty cells (all should be, but be defensive).
    replies = [c for c in reply_set
               if not (attacker[c // BOARD_SIZE, c % BOARD_SIZE]
                       or defender[c // BOARD_SIZE, c % BOARD_SIZE])]
    if not replies:
        return None

    worst = 0
    for d in replies:
        dr, dc = d // BOARD_SIZE, d % BOARD_SIZE
        defender[dr, dc] = True
        try:
            # A defender block that itself completes the defender's five (e.g. an
            # overline through the blocked square) means the attacker has not won.
            if state_ops.has_five_in_a_row(defender):
                return None
            sub_move, sub_dist = _vct_attack(
                attacker, defender, depth + 1, max_depth, max_nodes, counter
            )
        finally:
            defender[dr, dc] = False
        if sub_move is None:
            # Attacker failed to win against this defender reply (or ran out of
            # budget): no proven forced win at this AND-node.
            return None
        if sub_dist is not None and sub_dist + 1 > worst:
            worst = sub_dist + 1
    return worst


def solve_vct_from_planes(
    planes: np.ndarray,
    *,
    history_ply: int = state_ops.HISTORY_PLY,
    max_depth: int = DEFAULT_VCT_MAX_DEPTH,
    max_nodes: int = DEFAULT_VCT_MAX_NODES,
) -> VCFResult:
    """Convenience: solve VCT from a network input-plane stack (mirrors
    :func:`solve_vcf_from_planes`)."""
    _check_cpu_solver_gate()
    planes = np.asarray(planes)
    attacker = planes[0].astype(bool)
    defender = planes[history_ply].astype(bool)
    board = np.stack([attacker, defender], axis=0)
    return solve_vct(board, max_depth=max_depth, max_nodes=max_nodes)
