"""Parallel self-play game generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import threading
import time
from typing import MutableMapping

import numpy as np

from gomoku.game import (
    BOARD_SIZE,
    GameState,
    HISTORY_PLY,
    N_ACTIONS,
    augment,
    augment_with_aux,
    augment_with_cell_targets,
)
from gomoku.mcts import (
    Evaluator,
    MCTSGame,
    policy_from_visits,
    run_batched_mcts,
    run_batched_mcts_waves,
)
from gomoku import native_mcts
from gomoku import vcf


ProfileStats = MutableMapping[str, float]


# Default value-target discount per attacker move-to-mate for the VCF teacher.
# dist == 1 (this move makes five) -> +1.0; deeper mates are discounted by
# VCF_VALUE_DISCOUNT**(dist-1), bottoming out at VCF_VALUE_FLOOR. A proven
# forced win is still a win, so the floor stays high.
VCF_VALUE_DISCOUNT = 0.98
VCF_VALUE_FLOOR = 0.90

# Per-process VCF solver budget for the teacher. Defaults to the vcf module's
# values (byte-identical to the pre-`vcf-deep` behavior); a worker can raise them
# once at startup via configure_vcf_teacher() so the solver proves LONGER forced
# wins (the Derby v5 'vcf-deep' lever) without threading the budget through every
# generator signature. Process-isolated: each selfplay_worker is one process.
_VCF_MAX_DEPTH = vcf.DEFAULT_MAX_DEPTH
_VCF_MAX_NODES = vcf.DEFAULT_MAX_NODES


def configure_vcf_teacher(max_depth: int | None = None,
                          max_nodes: int | None = None) -> None:
    """Set the process-wide VCF teacher solver budget (depth / node cap). None
    leaves a field at its current value. Call once before generation."""
    global _VCF_MAX_DEPTH, _VCF_MAX_NODES
    if max_depth is not None:
        _VCF_MAX_DEPTH = int(max_depth)
    if max_nodes is not None:
        _VCF_MAX_NODES = int(max_nodes)


# Per-process VCT solver budget for the TEACHER hot path (Derby 'x-vct' lever,
# beads derby-rxf / derby-b6r). VCT (Victory-by-Continuous-Threes) is a strict
# SUPERSET of VCF: it proves every VCF forced win plus wins that need forcing
# threes. Its tree fans out on the defender side, so an unbounded (or generously
# bounded) per-move solve is ruinous on the SELF-PLAY GENERATION hot path: bead
# derby-b6r raced derby-x-vct in Derby v8 with the library defaults (depth 7,
# nodes 20k) and got ZERO games / buf=0 in ~50s — the per-move solve never
# returned in time, fully starving generation (trainer spun at pl=nan).
#
# So the teacher defaults are DECOUPLED from the general-purpose solver library
# defaults (vcf.DEFAULT_VCT_MAX_* — depth 7 / nodes 20k, which stay as-is for
# direct solve_vct callers and tests) and set AGGRESSIVELY here: a wide-open
# position must bail to "no forced win / hit_cap" almost instantly so self-play
# never blocks. These find short tactical wins (open-four mate at dist 1, the
# double-three fork at dist 2) while bailing fast on explosive trees. A worker
# may override per-process via configure_vct_teacher() (the --vct-max-depth /
# --vct-max-nodes flags). On cap-hit the solver returns has_forced_win=False
# with hit_cap=True (never a false positive), so generation always proceeds.
_VCT_TEACHER_MAX_DEPTH = 4
_VCT_TEACHER_MAX_NODES = 800
_VCT_MAX_DEPTH = _VCT_TEACHER_MAX_DEPTH
_VCT_MAX_NODES = _VCT_TEACHER_MAX_NODES


def configure_vct_teacher(max_depth: int | None = None,
                          max_nodes: int | None = None) -> None:
    """Set the process-wide VCT teacher solver budget (depth / node cap). None
    leaves a field at its current value. Call once before generation. Mirrors
    :func:`configure_vcf_teacher`."""
    global _VCT_MAX_DEPTH, _VCT_MAX_NODES
    if max_depth is not None:
        _VCT_MAX_DEPTH = int(max_depth)
    if max_nodes is not None:
        _VCT_MAX_NODES = int(max_nodes)


# ---------------------------------------------------------------------------
# VCT-terminus self-play (issue #98): end a game at the FIRST cap50 VCT instead
# of playing out to five-in-a-row. A VCT is a forced win the GPU oracle
# (scripts.vct_metal.mega_vct_bb) both DETECTS and terminally VALUES (exact win +
# winning move), so the rollout tail past the VCT (empirically ~half the game;
# first VCT at median ply ~19) is pure label noise. Terminate there and take the
# oracle verdict: cheaper AND cleaner data, and the objective becomes "reach a
# VCT". Unlike the per-move --vct-teacher (which RELABELS a full playout via the
# retired CPU solver), this ENDS the game early using the batched GPU oracle.
#
# THE CALL-COST LAW (wiki gpu-vct-feasibility.md): the solver's wall is set by the
# single hardest board and is ~flat in batch size, so every caller MUST be
# bulk-synchronous — gather the WHOLE wave of live games into ONE solve per ply,
# never solve-in-a-loop. cap50 (max_nodes=50) is the near-complete first-VCT
# detector sweet spot (98.8% of VCTs, 96%+ of games; 40-850x cheaper than deep
# search). See wiki idea-pile.md #11 + vct-cascade-run-2026-06-30.md.
_VCT_TERMINUS_ENABLED = False
_VCT_TERMINUS_BUDGET = 50            # per-board node cap (cap50)
_vct_terminus_solver = None         # lazily-imported MLX solve_vct_mega_bb
# Moonshot VCT-defense labeler breadth cap. 0 (default) = enumerate ALL legal
# empty cells as blunder candidates per recorded position; K > 0 caps to the K
# empty cells nearest an existing stone (a per-ply gen-cost lever). Only read by
# _vct_defense_solve, which is only called when record_vct is on.
_VCT_DEFENSE_MAX_CANDS = 0


def configure_vct_terminus(enabled: bool | None = None,
                           budget: int | None = None,
                           defense_max_cands: int | None = None) -> None:
    """Enable / parameterize VCT-terminus self-play (process-wide). None leaves a
    field unchanged. Call once before generation, before any game loop. Gating is
    purely on this global, so the default-off path is byte-identical self-play and
    never imports MLX. `defense_max_cands` caps the VCT-defense labeler breadth
    (0 = all legal empty cells; K > 0 = K cells nearest existing stones)."""
    global _VCT_TERMINUS_ENABLED, _VCT_TERMINUS_BUDGET, _VCT_DEFENSE_MAX_CANDS
    if enabled is not None:
        _VCT_TERMINUS_ENABLED = bool(enabled)
    if budget is not None:
        _VCT_TERMINUS_BUDGET = int(budget)
    if defense_max_cands is not None:
        _VCT_DEFENSE_MAX_CANDS = int(defense_max_cands)


# Sound-world oracle veto (issue #107). When enabled, every self-play ply runs
# the bulk VCT-defense escape-solve (_vct_defense_solve, FULL breadth unless
# _VETO_MAX_CANDS staged escalation is configured) over the
# wave and the resulting per-cell blunder map is ACTED ON instead of merely
# recorded:
#   * moves proven to lose to a forced opponent VCT are masked OUT of the root
#     visit distribution — both the move actually played AND the recorded
#     policy target `pi` (on-policy by construction: the target stays the
#     net's own search, just constrained; no post-hoc relabeling);
#   * a position where EVERY legal move is a proven blunder is a defender
#     terminus: the game ends there, side-to-move loses (the exact mirror of
#     the attacker VCT-terminus), giving an opponent-independent value target.
# Composes with (but does not require) _VCT_TERMINUS_ENABLED and record_vct —
# the per-ply solve is shared by all three consumers. Native gen path only.
# Default OFF = byte-identical self-play.
_ORACLE_VETO_ENABLED = False

# Staged-escalation breadth cap for the VETO's escape-solve (big-board lever).
# 0 (default) = FULL breadth every ply (the original #107 semantics,
# byte-identical). K > 0 = stage 1 tests only the K empty cells nearest existing
# stones (the same _defense_candidate_cells rule as the labeler cap) and vetoes
# the proven blunders among THEM (conservative-safe: an untested cell is simply
# never vetoed); the one soundness-critical event — "ALL legal moves lose", the
# defender terminus — can only be declared at full breadth, so any position
# whose tested cells are ALL blunders is ESCALATED: its remaining untested legal
# cells are solved in a second bulk call before the partition runs. The
# partition condition np.all(vmap[legal] >= 0.5) is automatically sound under
# partial maps (untested legal cells read 0.0 -> the game survives).
_VETO_MAX_CANDS = 0

# Oracle/search overlap (perf, flag-gated): run the per-ply bulk mega-solve in a
# background thread WHILE the native MCTS wave searches on MPS, joining before
# the verdicts are consumed. MLX releases the GIL inside the Metal dispatch and
# the two libraries drive separate Metal queues, so the smaller of (solve,
# search) hides almost entirely (measured). NOT byte-identical to the serial
# order: games the oracle would have terminated pre-search are searched (and
# discarded) this ply, which changes evaluator batch shapes on firing plies —
# the same numeric class as a wave-size change; targets/records are unchanged
# in kind and the flag-on path is deterministic per seed. Default OFF.
_ORACLE_OVERLAP_ENABLED = False


def configure_oracle_veto(enabled: bool | None = None,
                          max_cands: int | None = None) -> None:
    """Enable / parameterize the gen-side oracle veto (process-wide). None
    leaves a field unchanged. Call once before generation. Gating is purely on
    these globals, so the default-off path is byte-identical self-play and never
    imports MLX. `max_cands` (0 = full breadth) is the staged-escalation stage-1
    breadth cap; see _VETO_MAX_CANDS."""
    global _ORACLE_VETO_ENABLED, _VETO_MAX_CANDS
    if enabled is not None:
        _ORACLE_VETO_ENABLED = bool(enabled)
    if max_cands is not None:
        _VETO_MAX_CANDS = int(max_cands)


def configure_oracle_overlap(enabled: bool | None = None) -> None:
    """Enable the oracle/search overlap (process-wide). None leaves it
    unchanged. Call once before generation. Native gen path only; see
    _ORACLE_OVERLAP_ENABLED for the semantics note. Default OFF."""
    global _ORACLE_OVERLAP_ENABLED
    if enabled is not None:
        _ORACLE_OVERLAP_ENABLED = bool(enabled)


def _load_mega_solver():
    """Lazy-import the MLX mega VCT solver (one-time Metal compile on first
    call; coexists with the PyTorch/MPS evaluator in-process, verified)."""
    global _vct_terminus_solver
    if _vct_terminus_solver is None:
        from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb
        _vct_terminus_solver = solve_vct_mega_bb
    return _vct_terminus_solver


_MEGA_WARMED = False


def _warm_mega_solver() -> None:
    """One tiny synchronous solve so the MLX import + Metal kernel compile
    happen on the MAIN thread before any background-thread solve (overlap
    mode). Idempotent, ~0.2s once per process."""
    global _MEGA_WARMED
    if not _MEGA_WARMED:
        solver = _load_mega_solver()
        solver(np.zeros((1, 2, BOARD_SIZE, BOARD_SIZE), dtype=bool),
               max_nodes=1, return_move=True)
        _MEGA_WARMED = True


def _terminus_boards(arrs):
    """(B, 2, N, N) bool solver batch (attacker = plane 0 = side to move,
    defender = plane HISTORY_PLY) from a list of to_planes() stacks."""
    attacker = np.stack([a[0] for a in arrs]).astype(bool)
    defender = np.stack([a[HISTORY_PLY] for a in arrs]).astype(bool)
    return np.stack([attacker, defender], axis=1)


def _vct_terminus_solve(planes_list):
    """Batched cap50 VCT test across the whole wave of live games. `planes_list`
    is one (N_INPUT_PLANES, N, N) input-plane stack per active game, in the SAME
    side-to-move-relative convention as :func:`vcf.solve_vct_from_planes`
    (attacker = plane 0, defender = plane HISTORY_PLY). Returns (win, move): win
    (B,) bool = the side to move has a forced VCT; move (B,) int = the oracle's
    winning first move (flat index, -1 where no win). ONE bulk-synchronous MLX
    solve — never per-game (the call-cost law)."""
    solver = _load_mega_solver()
    boards = _terminus_boards([np.asarray(p) for p in planes_list])
    win, _hit, move = solver(
        boards, max_nodes=_VCT_TERMINUS_BUDGET, return_move=True)
    return np.asarray(win).astype(bool), np.asarray(move).astype(np.int64)


def _defense_candidate_cells(empty_flat, occupied, max_cands):
    """Pick which empty cells to test as blunder candidates. max_cands <= 0 (or a
    board with fewer empties than the cap) => ALL legal empty cells; otherwise the
    `max_cands` empty cells with the smallest Chebyshev distance to any existing
    stone (breadth cap near the action). Returns a 1-D int array of flat indices."""
    empty_flat = np.asarray(empty_flat, dtype=np.int64)
    if max_cands is None or max_cands <= 0 or empty_flat.size <= max_cands:
        return empty_flat
    n = occupied.shape[0]
    stones = np.argwhere(occupied)                 # (S, 2) rows, cols
    if stones.size == 0:
        return empty_flat[:max_cands]
    rows = empty_flat // n
    cols = empty_flat % n
    rc = np.stack([rows, cols], axis=1)            # (E, 2)
    # Chebyshev distance from each empty cell to the NEAREST existing stone.
    dist = np.abs(rc[:, None, :] - stones[None, :, :]).max(axis=2).min(axis=1)
    order = np.argsort(dist, kind="stable")
    return empty_flat[order[:max_cands]]


def _vct_defense_solve(planes_list, max_cands=0):
    """Moonshot VCT-defense labeler. For each input position (side-to-move = me),
    produce a per-cell (N_ACTIONS,) 0/1 "blunder map": map[m] = 1.0 iff, after I
    play empty cell m, the OPPONENT has a forced VCT (i.e. m walks me into a lost
    position). `planes_list` is one (N_INPUT_PLANES, N, N) stack per position, in
    the same side-to-move-relative convention as :func:`_vct_terminus_solve`
    (attacker = plane 0 = my stones, defender = plane HISTORY_PLY = opp stones).

    For each candidate m the CHILD board is framed from the OPPONENT's
    perspective (they are now to move):
        attacker = opp stones (my defender-plane, HISTORY_PLY)
        defender = my stones (plane 0) with cell m set to 1
    ALL children of ALL positions are concatenated into ONE (Sum_L, 2, N, N) batch
    and solved with a SINGLE solve_vct_mega_bb call (the call-cost law) — never
    per-position. `max_cands` <= 0 tests every legal empty cell; K > 0 caps to the
    K empties nearest existing stones.

    Returns (maps, masks): maps is a list of (N_ACTIONS,) float32 arrays (1.0 on
    proven-blunder cells, 0.0 elsewhere), masks is a list of (N_ACTIONS,) bool
    arrays marking which cells were evaluated (candidate cells)."""
    solver = _load_mega_solver()
    arrs = [np.asarray(p) for p in planes_list]
    maps = [np.zeros(N_ACTIONS, dtype=np.float32) for _ in arrs]
    child_boards, owner_pos, owner_cell, masks = _defense_children(arrs, max_cands)
    if child_boards is None:
        return maps, masks
    win, _hit = solver(
        child_boards, max_nodes=_VCT_TERMINUS_BUDGET, return_move=False)
    win = np.asarray(win).astype(bool)
    for k in np.flatnonzero(win):
        maps[owner_pos[k]][owner_cell[k]] = 1.0
    return maps, masks


def _stack_children(me: np.ndarray, opp: np.ndarray, cands: np.ndarray):
    """Vectorized child-board build for ONE position: for each candidate cell m,
    the child is (attacker=opp, defender=me+m) — the OPPONENT-to-move frame the
    escape-solve tests. Returns (E, 2, N, N) bool in `cands` order."""
    E = int(cands.size)
    n = me.shape[0]
    child_def = np.repeat(me[None], E, axis=0)
    child_def.reshape(E, -1)[np.arange(E), cands] = True
    child_att = np.repeat(opp[None], E, axis=0)
    return np.stack([child_att, child_def], axis=1).reshape(E, 2, n, n)


def _defense_children(arrs, max_cands, build_for=None):
    """Build the escape-solve child batch for a list of to_planes() stacks.

    Returns (child_boards, owner_pos, owner_cell, masks): child_boards is the
    concatenated (Sum_L, 2, N, N) bool solver batch (None when no position has
    any candidate), owner_pos/owner_cell are (Sum_L,) int arrays mapping each
    child back to (position index, flat cell), and masks is the per-position
    (N_ACTIONS,) bool "which cells were evaluated" list. Same candidate rule and
    ordering as the historical per-cell loop, vectorized per position.

    `build_for` (optional (B,) bool) skips the child-board BUILD for positions
    marked False while still marking their candidate masks — the null-board
    precheck's contract: a position with a clean opp-no-VCT-on-pass has every
    child PROVEN no-win (freestyle monotonicity + solver 0-FP), so its cells
    count as evaluated (zero blunder map) without ever hitting the solver."""
    masks = [np.zeros(N_ACTIONS, dtype=bool) for _ in arrs]
    seg_boards: list[np.ndarray] = []
    seg_pos: list[np.ndarray] = []
    seg_cell: list[np.ndarray] = []
    for pos_idx, p in enumerate(arrs):
        me = p[0].astype(bool)               # my stones (plane 0)
        opp = p[HISTORY_PLY].astype(bool)    # opponent stones (plane HISTORY_PLY)
        occupied = me | opp
        empty_flat = np.flatnonzero(~occupied.reshape(-1))
        cands = np.asarray(
            _defense_candidate_cells(empty_flat, occupied, max_cands),
            dtype=np.int64)
        if cands.size == 0:
            continue
        masks[pos_idx][cands] = True
        if build_for is not None and not build_for[pos_idx]:
            continue
        seg_boards.append(_stack_children(me, opp, cands))
        seg_pos.append(np.full(cands.size, pos_idx, dtype=np.int64))
        seg_cell.append(cands)
    if not seg_boards:
        return None, None, None, masks
    return (np.concatenate(seg_boards, axis=0), np.concatenate(seg_pos),
            np.concatenate(seg_cell), masks)


# Null-board precheck (perf, byte-identical): before building the
# ~(legal cells) escape-solve children of a position, solve its NULL board
# (attacker = opponent, i.e. "the side to move passes"). A CLEAN no-win (win
# False AND hit_cap False) is an exhaustive proof that the opponent has no VCT
# even with a free tempo; every child adds one DEFENDER stone to that exact
# frame, and by freestyle monotonicity (a defender stone never makes an
# attacker VCT appear — documented mega-solver invariant, see return_w) no
# child can hold an attacker VCT either; the solver never returns win=True on
# a no-VCT board (0 FP), so every child would come back win=False. The whole
# children build+solve is therefore skippable with a provably identical zero
# blunder map. A capped null (hit_cap=True) is NOT skippable: a child's extra
# defender stone can prune the attacker's move list enough that the child
# search completes (and proves a win) within the same node budget. Measured on
# live sound-world gen positions: ~68% of plies are clean -> ~64% of children
# solver-work skipped (the veto-stretched endgame plies 40+ are ~all clean).
#
# DEFAULT OFF (2026-07-01 A/B verdict): at the 9x9 live config the per-call
# solver cost is ~CONSTANT (~44 ms at 48 or 151 boards/call — the call is
# call-count x tail-grind bound, width is FREE), so skipping 61% of the boards
# saved nothing while the phase-2 split added ~0.2 calls/ply => net SLOWER
# (oracle 3.55 -> 4.35 s/batch). Results are byte-identical either way; the
# flag stays as a BIG-BOARD experiment (N^2-children build cost and batch
# widths change the trade there — re-measure before enabling).
_ORACLE_PRECHECK_ENABLED = False


def configure_oracle_precheck(enabled: bool | None = None) -> None:
    """Enable/disable the null-board precheck (process-wide; A/B escape hatch —
    the results are byte-identical either way, only the cost changes)."""
    global _ORACLE_PRECHECK_ENABLED
    if enabled is not None:
        _ORACLE_PRECHECK_ENABLED = bool(enabled)


def _oracle_ply_solve(planes_list, *, want_terminus, want_defense,
                      defense_max_cands=0, profile=None):
    """Bulk mega-solve per ply for every oracle consumer (the call-cost law:
    gather everything, never solve-in-a-loop). Phase 1 concatenates the
    attacker-terminus boards (B) and — with the precheck on — the null boards
    (B) into ONE solve_vct_mega_bb dispatch; phase 2 solves the defense
    escape-children ONLY for positions whose null board came back win-or-cap
    (with the precheck off, the children ride along in phase 1 and there is no
    phase 2).

    Bit-identical to running _vct_terminus_solve + _vct_defense_solve
    separately: the megakernel runs one GPU thread per board with a PER-BOARD
    node budget (`int nodes=0` thread-local), so a board's verdict is
    independent of batch composition; `return_move` selects the SAME kernel
    (the move output is always computed, the flag only gates the Python
    return); and the precheck skip is a proof, not an approximation (see
    _ORACLE_PRECHECK_ENABLED). Verified by tests/test_oracle_merged_solve.py,
    a live-solver probe, and same-seed whole-gen hash receipts.

    Returns (win_t, move_t, vmaps, vmasks): the terminus pair is None unless
    `want_terminus`; the defense pair is None unless `want_defense`. Slot order
    follows `planes_list`."""
    solver = _load_mega_solver()
    B = len(planes_list)
    precheck = want_defense and _ORACLE_PRECHECK_ENABLED
    with _profile_timer(profile, "oracle_build_s"):
        arrs = [np.asarray(p) for p in planes_list]
        segs: list[np.ndarray] = []
        if want_terminus:
            segs.append(_terminus_boards(arrs))
        if precheck:
            null_att = np.stack([a[HISTORY_PLY] for a in arrs]).astype(bool)
            null_def = np.stack([a[0] for a in arrs]).astype(bool)
            segs.append(np.stack([null_att, null_def], axis=1))
        child_boards = owner_pos = owner_cell = masks = None
        if want_defense and not precheck:
            child_boards, owner_pos, owner_cell, masks = _defense_children(
                arrs, defense_max_cands)
            if child_boards is not None:
                segs.append(child_boards)
    win_t = move_t = None
    vmaps = vmasks = None
    if want_defense and not precheck:
        vmaps = [np.zeros(N_ACTIONS, dtype=np.float32) for _ in arrs]
        vmasks = masks
    if not segs:
        return win_t, move_t, vmaps, vmasks
    batch = segs[0] if len(segs) == 1 else np.concatenate(segs, axis=0)
    with _profile_timer(profile, "oracle_solve_s"):
        win, hit, move = solver(
            batch, max_nodes=_VCT_TERMINUS_BUDGET, return_move=True)
    _profile_add(profile, "oracle_solve_calls", 1.0)
    _profile_add(profile, "oracle_boards", float(batch.shape[0]))
    win = np.asarray(win).astype(bool)
    off = 0
    if want_terminus:
        win_t = win[:B]
        move_t = np.asarray(move)[:B].astype(np.int64)
        off = B
    if precheck:
        null_win = win[off:off + B]
        null_hit = np.asarray(hit).astype(bool)[off:off + B]
        need = null_win | null_hit          # only these can hold a blunder
        vmaps = [np.zeros(N_ACTIONS, dtype=np.float32) for _ in arrs]
        with _profile_timer(profile, "oracle_build_s"):
            child_boards, owner_pos, owner_cell, vmasks = _defense_children(
                arrs, defense_max_cands, build_for=need)
        _profile_add(profile, "oracle_precheck_skips", float(B - int(need.sum())))
        if child_boards is not None:
            with _profile_timer(profile, "oracle_solve_s"):
                child_win, _chit = solver(
                    child_boards, max_nodes=_VCT_TERMINUS_BUDGET,
                    return_move=False)
            _profile_add(profile, "oracle_solve_calls", 1.0)
            _profile_add(profile, "oracle_boards", float(child_boards.shape[0]))
            child_win = np.asarray(child_win).astype(bool)
            for k in np.flatnonzero(child_win):
                vmaps[owner_pos[k]][owner_cell[k]] = 1.0
    elif want_defense and child_boards is not None:
        child_win = win[off:]
        for k in np.flatnonzero(child_win):
            vmaps[owner_pos[k]][owner_cell[k]] = 1.0
    return win_t, move_t, vmaps, vmasks


def _escalate_all_blunder_positions(planes_list, surv_slots, vmaps, vmasks,
                                    profile=None):
    """Staged-escalation stage 2 (see _VETO_MAX_CANDS): among the surviving
    positions, find those whose TESTED cells are ALL proven blunders while
    untested legal cells remain — the only positions where the (soundness-
    critical, full-breadth-only) defender terminus could fire — and solve their
    remaining untested legal cells in ONE bulk call, updating vmaps/vmasks in
    place to full breadth. Positions with any tested-safe cell can never fire
    the terminus and keep their partial (conservative) map."""
    todo: list[tuple[int, np.ndarray]] = []
    for s in surv_slots:
        mask = vmasks[s]
        if not mask.any():
            continue
        if not np.all(vmaps[s][mask] >= 0.5):
            continue                       # a tested safe cell exists
        legal = _legal_mask_from_planes(planes_list[s])
        untested = legal & ~mask
        if not untested.any():
            continue                       # already full breadth
        todo.append((s, np.flatnonzero(untested).astype(np.int64)))
    if not todo:
        return
    solver = _load_mega_solver()
    with _profile_timer(profile, "oracle_escalation_s"):
        segs, seg_slot, seg_cell = [], [], []
        for s, cells in todo:
            p = np.asarray(planes_list[s])
            me = p[0].astype(bool)
            opp = p[HISTORY_PLY].astype(bool)
            segs.append(_stack_children(me, opp, cells))
            seg_slot.append(np.full(cells.size, s, dtype=np.int64))
            seg_cell.append(cells)
        batch = np.concatenate(segs, axis=0)
        win, _hit = solver(
            batch, max_nodes=_VCT_TERMINUS_BUDGET, return_move=False)
    win = np.asarray(win).astype(bool)
    slot_arr = np.concatenate(seg_slot)
    cell_arr = np.concatenate(seg_cell)
    for s, cells in todo:
        vmasks[s][cells] = True
    for k in np.flatnonzero(win):
        vmaps[slot_arr[k]][cell_arr[k]] = 1.0
    _profile_add(profile, "oracle_escalated_positions", float(len(todo)))
    _profile_add(profile, "oracle_escalation_boards", float(batch.shape[0]))


def _ply_for(ply, g_idx):
    """Per-game ply lookup. `ply` is either a global int (the lockstep paths,
    where every active game is at the same round) or a per-game indexable
    (the continuous-refill native path, where games start at different
    rounds). Values coincide in lockstep, so helpers using this are
    byte-identical on the legacy path."""
    return ply[g_idx] if hasattr(ply, "__getitem__") else ply


def _vct_terminus_partition(active, active_games, planes_list, ply, initial_plies,
                            trajectories, completed, final_state, record_ownership,
                            profile, win=None, move=None):
    """Run the batched VCT test over the wave and TERMINATE every game whose side
    to move has a forced VCT: record the decisive position (one-hot on the oracle
    winning move) as that game's final training example, credit that side the win,
    and drop it from the active set. Games with no VCT survive and continue to
    normal MCTS + play this ply. Returns the surviving (active, active_games).

    `win`/`move` may be precomputed by the merged per-ply oracle solve
    (:func:`_oracle_ply_solve`, slot-aligned with `active`); when None this runs
    its own bulk solve (the historical form; the Python gen path uses this).

    The terminal outcome flows through the SAME sign-flip + mate-distance discount
    as a real five-in-a-row terminal (:func:`_apply_teachers_to_trajectory`): the
    appended VCT position is the last trajectory element, so it gets plies-to-end
    = 0 (crisp +-1) and earlier positions get discounted back from it."""
    if win is None:
        with _profile_timer(profile, "vct_terminus_s"):
            win, move = _vct_terminus_solve(planes_list)
    if not win.any():
        return active, active_games
    keep = []
    for slot_idx, g_idx in enumerate(active):
        if not win[slot_idx]:
            keep.append(slot_idx)
            continue
        n_initial = initial_plies[g_idx]
        ply_g = _ply_for(ply, g_idx)
        side = (n_initial + ply_g) % 2        # the side to move HAS the forced win
        mv = int(move[slot_idx])
        pi_term = np.zeros(N_ACTIONS, dtype=np.float32)
        if 0 <= mv < N_ACTIONS:
            pi_term[mv] = 1.0                 # exact winning move = seek-VCT target
        else:                                 # win with no move: never expected
            pi_term[:] = 1.0 / N_ACTIONS
            _profile_add(profile, "vct_terminus_moveless", 1.0)
        trajectories[g_idx].append(
            (np.asarray(planes_list[slot_idx]).copy(), pi_term, side))
        outcome_for_black = 1.0 if side == 0 else -1.0
        if record_ownership:
            # A VCT terminus is NOT an actual five-in-a-row board, so there is no
            # honest final-ownership target — MASK it (like a max-plies draw).
            final_state[g_idx] = (None, 0)
        completed.append((g_idx, outcome_for_black, n_initial + ply_g + 1))
        _profile_add(profile, "vct_terminus_fired", 1.0)
    return [active[s] for s in keep], [active_games[s] for s in keep]


def _legal_mask_from_planes(planes) -> np.ndarray:
    """(N_ACTIONS,) bool legal (empty) mask from a to_planes() stack."""
    p = np.asarray(planes)
    occupied = p[0].astype(bool) | p[HISTORY_PLY].astype(bool)
    return ~occupied.reshape(-1)


def _veto_policy(pi: np.ndarray, vmap: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Mask proven-blunder cells out of a root visit distribution.

    `vmap` is the (N_ACTIONS,) 0/1 blunder map from _vct_defense_solve, `legal`
    the (N_ACTIONS,) bool legality mask. Returns a renormalized distribution
    with zero mass on proven blunders. If the search put ALL its mass on
    blunders (visited only losing moves), falls back to uniform over the legal
    non-blunder cells — the caller guarantees at least one exists (the all-lose
    case is partitioned out as a defender terminus before search)."""
    keep = vmap < 0.5
    masked = np.where(keep, pi, 0.0)
    s = masked.sum()
    if s > 0.0 and np.isfinite(s):
        return (masked / s).astype(np.float32)
    fallback = (legal & keep).astype(np.float32)
    n = fallback.sum()
    if n <= 0.0:                       # defensive: should be unreachable
        return (legal.astype(np.float32) / max(legal.sum(), 1)).astype(np.float32)
    return fallback / n


def _oracle_veto_partition(active, active_games, planes_list, pending_vct, ply,
                           initial_plies, trajectories, completed, final_state,
                           record_ownership, record_vct, vct_maps, profile):
    """Defender terminus: END every game whose side to move has NO non-losing
    move (every legal cell is a proven blunder on its map). The exact mirror of
    :func:`_vct_terminus_partition` — side-to-move LOSES and the game leaves
    the active set before any search is spent on it. Only positions with a FULL
    blunder map (veto forces max_cands=0) can terminate here; positions with no
    map (None) always survive. Returns the surviving (active, active_games).

    The doomed position itself records NO training example (2026-07-01 wound
    fix, #107): the original design appended it with a uniform-over-legal
    policy target ("no move is better; z does the teaching"), but at scale
    that IS the teaching — as black's attack sharpens, most games end here, so
    white's most common late-training examples became uniform noise over ~70
    cells and white's policy collapsed (e1982: 0W-20L-0D as white vs the old
    champion, vs all-draws at e1239; self-metrics saw nothing). Dropping the
    example loses nothing real: the trap-completing move (real MCTS pi, crisp
    discounted z) becomes the trajectory's final example and the loss still
    propagates through the mate-distance discount."""
    keep = []
    for slot_idx, g_idx in enumerate(active):
        vmap = pending_vct.get(g_idx)
        if vmap is None:
            keep.append(slot_idx)
            continue
        legal = _legal_mask_from_planes(planes_list[slot_idx])
        if not legal.any() or not np.all(vmap[legal] >= 0.5):
            keep.append(slot_idx)
            continue
        n_initial = initial_plies[g_idx]
        ply_g = _ply_for(ply, g_idx)
        side = (n_initial + ply_g) % 2     # the side to move is LOST
        outcome_for_black = -1.0 if side == 0 else 1.0
        if record_ownership:
            # Not a real five-in-a-row board -> ownership target MASKED.
            final_state[g_idx] = (None, 0)
        completed.append((g_idx, outcome_for_black, n_initial + ply_g))
        _profile_add(profile, "oracle_veto_all_lose", 1.0)
    return [active[s] for s in keep], [active_games[s] for s in keep]


def _apply_oracle_partitions(oracle_res, active, active_games, planes_list,
                             slot_of, ply, initial_plies, trajectories,
                             completed, final_state, record_ownership,
                             record_vct, vct_maps, profile):
    """Consume one ply's merged oracle verdicts (:func:`_oracle_ply_solve`):
    run the attacker-terminus partition, build `pending_vct` for the survivors,
    run staged escalation when the veto breadth is capped, then the
    defender-terminus/all-lose partition. `planes_list`/`slot_of` are aligned
    with the ACTIVE SET AT SOLVE TIME (pre-partition slots). Returns the
    surviving (active, active_games, pending_vct)."""
    win_t, move_t, vmaps, vmasks = oracle_res
    pending_vct: dict[int, np.ndarray | None] = {}
    if win_t is not None and active:
        active, active_games = _vct_terminus_partition(
            active, active_games, planes_list, ply, initial_plies,
            trajectories, completed, final_state, record_ownership, profile,
            win=win_t, move=move_t)
    if vmaps is not None and active:
        surv_slots = [slot_of[g_idx] for g_idx in active]
        if _ORACLE_VETO_ENABLED and _VETO_MAX_CANDS > 0:
            _escalate_all_blunder_positions(
                planes_list, surv_slots, vmaps, vmasks, profile)
        for g_idx, s in zip(active, surv_slots):
            pending_vct[g_idx] = vmaps[s] if vmasks[s].any() else None
        if _ORACLE_VETO_ENABLED:
            surv_planes = [planes_list[s] for s in surv_slots]
            active, active_games = _oracle_veto_partition(
                active, active_games, surv_planes, pending_vct, ply,
                initial_plies, trajectories, completed, final_state,
                record_ownership, record_vct, vct_maps, profile)
    return active, active_games, pending_vct


# Value-discount (Derby v6 'mate-discounted-value'): scale ordinary outcome value
# targets by gamma^(plies_to_end) so positions near a decisive end get crisp ±1 and
# far-from-end positions get hedged targets — generalizing the VCF mate-distance
# discount to ALL outcomes. Per-process (set once by the worker). 1.0 = OFF (flat
# ±1, byte-identical to pre-v6). Applied to the base z BEFORE the VCF teacher, so a
# proven forced win still overwrites with its own (steeper) mate-discounted value.
_VALUE_DISCOUNT = 1.0


# Draw-contempt (Derby 'x-draw-contempt', bead derby-9q4): a DECISIVENESS lever.
# When a game ends in a DRAW (raw outcome z=0), reshape the value TARGET to
# -DRAW_VALUE (mildly losing) instead of exactly 0, so the net learns that a draw
# is mildly worse than equal -> at MCTS/eval time the value head reports draws as
# slightly losing -> the search prefers non-drawing continuations. Per-process
# (set once by the worker / trainer). 0.0 = OFF (draws stay 0, byte-identical
# baseline). Composes with --value-discount the same way decisive outcomes do:
# the contempt magnitude DELTA is scaled by gamma^(plies_to_end), i.e. positions
# far from the (drawn) game end get a smaller contempt push, positions near the
# end get the full -DELTA. This preserves the existing math shape (z scaled by
# gamma^plies) and the sibling-of-mate-discount property.
_DRAW_VALUE = 0.0


# Search-contempt (Derby 'x-search-contempt', bead derby-qoq) — a SELF-PLAY
# POSITION-DISTRIBUTION lever, NOT a target reshape. With probability
# `_CONTEMPT_P` per move, REPLACE the temperature-sampled visit-policy move
# selection with a contempt-perturbed pick: weight legal-visited children by
# `softmax(-|child_Q| / max(tau, eps))` so moves that lead to the MOST CONTESTED
# child position (Q closest to 0) are preferred. The result is a self-play
# trajectory that oversamples hard-to-convert positions — exactly the regime
# where the v8 champion's lookahead4-as-black 100%-target gap clusters (draws,
# not losses). Source: Singh & Eindhoven 2025, arxiv 2504.07757 (Odds Chess).
#
# Crucial: this only changes the MOVE PLAYED. The training target (the visit-
# count `pi` appended to the trajectory BEFORE this call) is unchanged — only
# the position distribution that enters the buffer shifts. Per-process
# (set once by the worker). 0.0 = OFF (no roll, no W read, byte-identical
# baseline; the `_sample_action(pi, rng)` call site runs verbatim). Paper
# default is p=0.5; the cell defaults to 0.5. The Q used is child Q from the
# parent's (root side-to-move's) perspective, the standard MCTS backup
# convention (W[a] is incremented by +v on own side, -v after the negamax
# flip — so W[a]/N[a] is in [-1,+1] from the side-about-to-play's POV).
_CONTEMPT_P = 0.0


def configure_value_discount(gamma: float | None = None) -> None:
    """Set the process-wide value-target discount (gamma in (0,1]); 1.0 = flat
    outcomes (current behavior). Call once before generation."""
    global _VALUE_DISCOUNT
    if gamma is not None:
        _VALUE_DISCOUNT = float(gamma)


def configure_draw_value(delta: float | None = None) -> None:
    """Set the process-wide draw-contempt magnitude (DELTA >= 0); 0.0 = OFF (draws
    stay exactly 0, byte-identical baseline). Call once before generation."""
    global _DRAW_VALUE
    if delta is not None:
        _DRAW_VALUE = float(delta)


def configure_search_contempt(p: float | None = None) -> None:
    """Set the process-wide search-contempt per-move probability (p in [0,1]);
    0.0 = OFF (move selection runs the standard temperature-sampled visit-
    policy path, byte-identical baseline). Call once before generation."""
    global _CONTEMPT_P
    if p is not None:
        _CONTEMPT_P = float(p)


# Defense-teacher GENTLENESS knobs (GitHub issue #42, the #36 course-correction).
# The defense teacher (--defense-teacher) relabels the VALUE target of a position
# the opponent has a proven forced win against. Two per-process knobs soften it so
# it cannot saturate the value head ("white always loses") and corrupt the shared
# trunk (the #36 G15-defense crash: Δelo -458, vl 0.16->0.06, pl 1.25->3.4):
#
#   _DEFENSE_SOFT_VALUE   the value target stamped on a proven-lost position.
#                         -1.0 = hard "dead lost" (the original, byte-identical
#                         default). A softer value (e.g. -0.5) still teaches "you
#                         are losing" without collapsing the head to a delta at -1.
#   _DEFENSE_MAX_FRACTION the cap on the FRACTION of one game's recorded to-move
#                         positions the teacher may relabel. 1.0 = unbounded (the
#                         original, byte-identical default). A tighter cap (e.g.
#                         0.25) stops a wide-open losing game from stamping the
#                         soft loss on dozens of positions. When the cap binds, the
#                         budget is spent on the LATEST firing plies (closest to
#                         the mate) — the most informative "defend earlier" signal.
#
# Per-process (set once by the worker via configure_defense_teacher). The defaults
# (-1.0 / 1.0) reproduce the pre-#42 behavior bit-for-bit, so cells that do not
# pass the new flags are unaffected.
_DEFENSE_SOFT_VALUE = -1.0
_DEFENSE_MAX_FRACTION = 1.0

# Issue #43 (defense-teacher I2): when True, the defensive teacher stamps the
# SAVING (refutation) move(s) on the POLICY head and LEAVES the value target
# untouched, instead of the value-only -1.0 crush. The two modes share the same
# --defense-teacher gate + danger pre-scan; this flag only switches WHICH target
# is rewritten. Default False = the original value-only behavior (byte-identical).
_DEFENSE_POLICY_MODE = False

# Issue #43 audit (MINOR): hard cap on the number of refutation candidates the
# policy teacher re-solves per fire. The targeted candidate set is usually <10 but
# its size scales with the defender's four-making moves (K~60-75 on adversarial
# dense boards), and each candidate is a full (capped) solve_vcf. Bounding it keeps
# a pathological position from costing seconds; dropping a candidate only reduces
# recall (the safe direction — a missed refutation just means the teacher fires on
# fewer of that position's saving moves, never a false save).
_DEFENSE_REFUTE_MAX_CANDIDATES = 24

# White-defense "sparse bite" sampler (2026-06-19, the #43 follow-on). The EXACT
# VCF detection solve is the generation bottleneck (~180 ms/solve, ~21 four-threat
# plies/game on a 15x15 net ⇒ ~3.8 s/game; the whole teacher ~7 s/game ≈ 90% of
# gen wall, measured on g15-wdl@0). AlphaZero distills a defensive lesson over many
# epochs from a PRESENT signal — it does NOT need every forced loss stamped. So
# invoke the proper solver on only a FRACTION of danger plies: gen cost scales ~1:1
# with the fraction, the stamps stay EXACT (no new correctness surface), and even a
# 10% rate in a fresh 150k buffer is ~1000× denser than the #43 race that drowned
# in a 1.5M warm buffer. Default 1.0 = solve every danger ply (byte-identical to the
# pre-sampler behavior). Sampled with a dedicated process-local RNG so it never
# perturbs the gen/MCTS RNG stream. See wiki/topics/white-side-defense-plan.md.
_DEFENSE_DETECT_FRAC = 1.0
_DEFENSE_SAMPLE_RNG = np.random.default_rng()

# CONV block-teacher (2026-06-19, the white-defense "dense shallow" arm). The
# complement of the sparse-deep VCF policy teacher: instead of the EXACT (and
# expensive) solver firing on a sampled fraction of plies, a CHEAP vectorized
# board scan fires on EVERY ply and stamps the BLOCK to the opponent's IMMEDIATE
# threat onto the POLICY head, leaving the value target at the natural outcome.
# Cost is a couple of numpy line-scans per ply (~microseconds), NO tree search, so
# it never throttles generation the way the VCF teacher did (~7 s/game). Hypothesis:
# a present, dense "block the obvious threat" reflex is exactly the basic defensive
# skill the white net lacks — and the sparse-deep approach did NOT move it off the
# floor (a clean null), so this is the orthogonal bite. Two tiers:
#   Tier 1 (SOUND CORE): the opponent has exactly one immediate five-completion (a
#     bare four) -> that cell is the forced block -> one-hot policy stamp. Blocking
#     it or losing next move is unambiguous. A double-four (>=2 opp win cells) is
#     genuinely lost -> NO stamp (mirrors the exact teacher leaving true losses
#     untouched). The defender having its OWN five-completion -> NO stamp (convert,
#     don't defend; mirrors the StM-own-win guard).
#   Tier 2 (HEURISTIC BITE, _DEFENSE_CONV_TIER2): no opp four, but the opponent has
#     an OPEN THREE (a move that would make an unstoppable open four). Stamp the
#     defender move(s) that PREVENT every such open four (the intersection of the
#     threats' defeating cells); uniform if several. This is a strong heuristic, NOT
#     strictly forced — the opponent could have other replies — so it lives behind
#     its own sub-toggle, but is ON by default (it is where the real defensive bite
#     is; the net likely already blocks bare fours). When the intersection is empty
#     (no single move stops all open fours) we do NOT stamp (sound: never a false
#     save). Both globals default OFF so generation is byte-identical when the conv
#     lever is not set. See wiki/topics/white-side-defense-plan.md.
_DEFENSE_CONV_MODE = False
_DEFENSE_CONV_TIER2 = True


def configure_defense_teacher(soft_value: float | None = None,
                              max_fraction: float | None = None,
                              policy_mode: bool | None = None,
                              detect_frac: float | None = None,
                              sample_seed: int | None = None,
                              conv_mode: bool | None = None,
                              conv_tier2: bool | None = None) -> None:
    """Set the process-wide defense-teacher knobs (issues #42, #43). None leaves a
    field at its current value. Call once before generation.

    ``soft_value``  -> :data:`_DEFENSE_SOFT_VALUE` (default -1.0, the hard loss).
    ``max_fraction`` -> :data:`_DEFENSE_MAX_FRACTION` (default 1.0, unbounded).
    ``policy_mode``  -> :data:`_DEFENSE_POLICY_MODE` (default False, value-only).
    ``detect_frac`` -> :data:`_DEFENSE_DETECT_FRAC` (default 1.0, solve every
    danger ply). <1.0 invokes the EXACT solver on only that fraction of four-threat
    plies — the "sparse bite" gen-cost lever; cost scales ~1:1 with the fraction,
    stamps stay exact. ``sample_seed`` reseeds the sampler RNG (test determinism).
    ``conv_mode`` -> :data:`_DEFENSE_CONV_MODE` (default False). The cheap dense
    "block the opponent's immediate threat" POLICY teacher (no tree search). When
    on, it is a SIBLING of policy_mode in the per-game seam (the two are exclusive
    levers on the same gate); the value target is left at the natural outcome.
    ``conv_tier2`` -> :data:`_DEFENSE_CONV_TIER2` (default True): include the
    open-three -> open-four prevention heuristic tier (Tier 2). Set False to keep
    only the SOUND forced-four block (Tier 1).

    The defaults reproduce the original hard / unbounded value-only behavior
    byte-for-byte. ``policy_mode=True`` is the #43 saving-move-on-policy lever
    (``soft_value`` is then unused — the value target is left at the natural
    game outcome; ``max_fraction`` still bounds the per-game stamp count).
    """
    global _DEFENSE_SOFT_VALUE, _DEFENSE_MAX_FRACTION, _DEFENSE_POLICY_MODE
    global _DEFENSE_DETECT_FRAC, _DEFENSE_SAMPLE_RNG
    global _DEFENSE_CONV_MODE, _DEFENSE_CONV_TIER2
    if soft_value is not None:
        _DEFENSE_SOFT_VALUE = float(soft_value)
    if max_fraction is not None:
        _DEFENSE_MAX_FRACTION = float(max_fraction)
    if policy_mode is not None:
        _DEFENSE_POLICY_MODE = bool(policy_mode)
    if detect_frac is not None:
        _DEFENSE_DETECT_FRAC = float(detect_frac)
    if sample_seed is not None:
        _DEFENSE_SAMPLE_RNG = np.random.default_rng(sample_seed)
    if conv_mode is not None:
        _DEFENSE_CONV_MODE = bool(conv_mode)
    if conv_tier2 is not None:
        _DEFENSE_CONV_TIER2 = bool(conv_tier2)


def _defense_budget(n_positions: int) -> int:
    """Max number of positions the defense teacher may relabel in ONE game, given
    the game has ``n_positions`` recorded to-move positions. Derived from
    :data:`_DEFENSE_MAX_FRACTION`: ``floor(frac * n_positions)``.

    With the default frac == 1.0 this is ``n_positions`` (>= every possible firing
    count), so NOTHING is ever capped — byte-identical to the pre-#42 behavior.
    A tighter frac bounds the relabel count; the caller spends the budget on the
    LATEST firing plies (closest to the loss). Floored, so a frac that rounds below
    1 yields a 0 budget (the teacher is fully off for that game) — deterministic.
    """
    if _DEFENSE_MAX_FRACTION >= 1.0:
        return n_positions
    return int(math.floor(_DEFENSE_MAX_FRACTION * max(0, n_positions)))


def _relabel_defense_game(
    fired: list[tuple[int, float]],
    n_positions: int,
) -> dict[int, float]:
    """Apply the per-game defense-teacher FRACTION cap (#42).

    ``fired`` is ``[(ply_idx, new_z), ...]`` for every position in ONE game where
    the defense teacher proved a forced loss (already in ascending ply order, as
    produced by a forward pass). ``n_positions`` is the game's total recorded
    to-move position count (the fraction denominator).

    Returns ``{ply_idx: new_z}`` for the positions whose relabel is KEPT: at most
    :func:`_defense_budget` of them, chosen as the LATEST firing plies (closest to
    the loss — the most informative "defend earlier" signal). When the budget is
    >= the firing count (the default, frac == 1.0), every fire is kept, so the
    result is byte-identical to the pre-#42 unbounded behavior.
    """
    budget = _defense_budget(n_positions)
    if budget >= len(fired):
        return dict(fired)
    if budget <= 0:
        return {}
    # Keep the LATEST `budget` firing plies (fired is ascending by ply_idx).
    return dict(fired[len(fired) - budget:])


def _relabel_defense_policy_game(
    fired: list[tuple[int, np.ndarray]],
    n_positions: int,
) -> dict[int, np.ndarray]:
    """#43 mirror of :func:`_relabel_defense_game` for the POLICY-stamp teacher.

    ``fired`` is ``[(ply_idx, new_pi), ...]`` (ascending ply order) for every
    position where the policy teacher proved a refutable forced loss and built a
    saving-move policy target. Applies the same per-game FRACTION budget
    (:func:`_defense_budget`), keeping the LATEST firing plies (closest to the
    loss, where the defensive lesson is sharpest). With the default frac == 1.0
    every fire is kept.
    """
    budget = _defense_budget(n_positions)
    if budget >= len(fired):
        return dict(fired)
    if budget <= 0:
        return {}
    return dict(fired[len(fired) - budget:])


def _apply_teachers_to_trajectory(
    traj: list[tuple[np.ndarray, np.ndarray, int]],
    outcome_for_black: float,
    *,
    vcf_teacher: bool,
    vct_teacher: bool,
    defense_teacher: bool,
    profile: ProfileStats | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Run the offensive (VCF/VCT) + defensive teachers over ONE game's
    trajectory and return the finalized ``[(pi, z), ...]`` per ply.

    Centralizes the record-build teacher seam shared by every discounted
    generation path (native / native-Gumbel / Python-Gumbel / Python fallback) so
    the #42 per-game defense FRACTION cap is applied in exactly one place. The
    per-ply z base and the offensive-teacher behavior are byte-identical to the
    prior inline code; the only change is that the defensive relabel is now
    BUDGETED per game (latest firing plies first) rather than applied unbounded.

    With the default knobs (soft_value -1.0, max_fraction 1.0) the output is
    bit-for-bit identical to the prior inline loops.
    """
    n = len(traj)
    out: list[tuple[np.ndarray, float]] = []
    defense_fires: list[tuple[int, float]] = []
    defense_policy_candidates: list[tuple[int, tuple]] = []
    conv_fires: list[tuple[int, np.ndarray]] = []
    for ply_idx, (planes, pi, side) in enumerate(traj):
        z = outcome_for_black if side == 0 else -outcome_for_black
        z = _discount_z(z, n - 1 - ply_idx)
        vcf_fired = False
        if vct_teacher:
            # VCT is a strict superset of VCF; the cell uses it INSTEAD of
            # --vcf-teacher, so the deeper solver replaces the shallower.
            pi, z, vcf_fired = _apply_vct_teacher(planes, pi, z, side=int(side),
                                                  profile=profile)
        elif vcf_teacher:
            pi, z, vcf_fired = _apply_vcf_teacher(planes, pi, z, side=int(side),
                                                  profile=profile)
        if defense_teacher:
            if _DEFENSE_CONV_MODE:
                # Conv block-teacher: a CHEAP dense scan stamps the block to the
                # opponent's immediate threat on the POLICY head (value left at the
                # natural outcome). No tree search; fires on every qualifying ply.
                # Collected here and budget-capped after the loop (same per-game
                # FRACTION cap as the policy teacher, latest plies first) so
                # --defense-max-fraction still applies — though with the cheap scan
                # a dense (frac 1.0) stamp is the intended use.
                c_pi, _z, fired = _apply_defense_teacher_conv(
                    planes, pi, z, profile=profile)
                if fired:
                    conv_fires.append((ply_idx, c_pi))
            elif _DEFENSE_POLICY_MODE:
                # #43/#60: detect the refutation CANDIDATE cheaply now (prescan +
                # detection/own-win solves, NO refutation); the expensive saving-move
                # enumeration is deferred to the budgeted reverse pass below, so we
                # never refute plies the FRACTION budget would discard.
                cand = _defense_detect_candidate(
                    planes, vcf_already_fired=vcf_fired, profile=profile)
                if cand is not None:
                    defense_policy_candidates.append((ply_idx, cand))
            else:
                d_z, fired = _apply_defense_teacher(
                    planes, z, vcf_already_fired=vcf_fired, profile=profile)
                if fired:
                    defense_fires.append((ply_idx, d_z))
        out.append((pi, z))
    if defense_fires:
        kept = _relabel_defense_game(defense_fires, n)
        for ply_idx, d_z in kept.items():
            pi, _ = out[ply_idx]
            out[ply_idx] = (pi, d_z)
    if defense_policy_candidates:
        # #60: realize the per-game FRACTION budget (latest firing plies first) as a
        # LAZY reverse pass — refute candidates newest-ply-first and stamp until the
        # budget of FIRES is filled, so the expensive refutation runs ONLY on kept
        # plies (it is ~83% of the teacher's gen cost and the budget discards most
        # fires). The kept set is identical to the eager form — refute every
        # candidate, then keep the latest `budget` fires (see
        # _relabel_defense_policy_game) — and z is left as the natural outcome.
        budget = _defense_budget(n)
        kept = 0
        for ply_idx, (swapped_board, winning_move) in reversed(defense_policy_candidates):
            if kept >= budget:
                break
            new_pi = _defense_refute_stamp(
                swapped_board, winning_move, out[ply_idx][0], profile=profile)
            if new_pi is None:
                continue  # candidate had no saving move — genuinely lost, not a fire
            out[ply_idx] = (new_pi, out[ply_idx][1])
            kept += 1
    if conv_fires:
        # Same per-game FRACTION budget as the policy teacher: keep the LATEST
        # firing plies (closest to the loss / most informative). Default frac 1.0
        # keeps every conv stamp (the dense intent). Value left at the outcome.
        kept_conv = _relabel_defense_policy_game(conv_fires, n)
        for ply_idx, new_pi in kept_conv.items():
            out[ply_idx] = (new_pi, out[ply_idx][1])
    return out


def _discount_z(z: float, plies_to_end: int) -> float:
    """Scale an outcome value target by gamma^plies_to_end.

    Decisive outcomes (z != 0): scaled by gamma^plies_to_end (no-op when
    gamma>=1.0). Draws (z == 0): if draw-contempt is enabled (DRAW_VALUE > 0),
    the target becomes -DRAW_VALUE * gamma^plies_to_end (mildly losing, with the
    same gamma^plies shape as decisive outcomes — so contempt and mate-discount
    compose multiplicatively, exactly like z * gamma^plies for decisive games).
    When both DRAW_VALUE == 0 AND (gamma >= 1.0 or z == 0), the function is the
    pre-lever identity (byte-identical baseline)."""
    if z == 0.0:
        if _DRAW_VALUE <= 0.0:
            return z
        # Draw-contempt: -DELTA, then discounted by gamma^plies_to_end the same
        # way decisive outcomes are discounted (composes with --value-discount).
        if _VALUE_DISCOUNT >= 1.0:
            return -_DRAW_VALUE
        return -_DRAW_VALUE * (_VALUE_DISCOUNT ** max(0, plies_to_end))
    if _VALUE_DISCOUNT >= 1.0:
        return z
    return z * (_VALUE_DISCOUNT ** max(0, plies_to_end))


def _apply_vcf_teacher(
    planes: np.ndarray,
    pi: np.ndarray,
    z: float,
    *,
    side: int,
    profile: ProfileStats | None = None,
    max_depth: int | None = None,
    max_nodes: int | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Opt-in EXACT teacher: if the recorded position is a proven VCF forced
    win for the side to move, overwrite its policy/value targets with the exact
    solution.

    Returns ``(new_pi, new_z, fired)``. When no forced win is proved the inputs
    are returned unchanged and ``fired`` is False. This is only ever called when
    ``--vcf-teacher`` is set; the default-off path never enters here, keeping
    self-play byte-identical.

    Policy rewrite: a one-hot on the proven winning move (the exact best move —
    sharper and more correct than a 100-sim visit estimate).

    Value rewrite: the side to move at this position has a proven win, so its
    value target becomes a (mate-distance-discounted) +1.0. The stored ``z`` is
    from the recorded side's perspective and the solver runs on that same
    side-to-move position, so a positive proof maps directly to a positive ``z``.
    """
    if max_depth is None:
        max_depth = _VCF_MAX_DEPTH
    if max_nodes is None:
        max_nodes = _VCF_MAX_NODES
    with _profile_timer(profile, "vcf_solve_s"):
        res = vcf.solve_vcf_from_planes(
            planes, history_ply=HISTORY_PLY, max_depth=max_depth, max_nodes=max_nodes
        )
    _profile_add(profile, "vcf_calls", 1.0)
    if not res.has_forced_win or res.winning_move is None:
        return pi, z, False

    new_pi = np.zeros_like(pi)
    new_pi[res.winning_move] = 1.0

    dist = res.mate_distance if res.mate_distance is not None else 1
    value = VCF_VALUE_DISCOUNT ** max(0, dist - 1)
    if value < VCF_VALUE_FLOOR:
        value = VCF_VALUE_FLOOR
    _profile_add(profile, "vcf_fired", 1.0)
    return new_pi, float(value), True


def _apply_vct_teacher(
    planes: np.ndarray,
    pi: np.ndarray,
    z: float,
    *,
    side: int,
    profile: ProfileStats | None = None,
    max_depth: int | None = None,
    max_nodes: int | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Opt-in EXACT teacher: the VCT (Victory-by-Continuous-Threes) mirror of
    :func:`_apply_vcf_teacher`. VCT is a strict SUPERSET of VCF — it proves every
    forced win VCF proves, PLUS wins that need forcing threes — so it stamps more
    positions with exact mate labels. Everything else is identical to the VCF
    teacher: a one-hot policy on the proven winning move, and a mate-distance-
    discounted +1.0 value (floor ``VCF_VALUE_FLOOR``). The solver
    (:func:`vcf.solve_vct_from_planes`) returns the SAME ``VCFResult`` shape, so
    the result-handling is byte-for-byte the VCF logic.

    Returns ``(new_pi, new_z, fired)``; unchanged inputs with ``fired=False`` when
    no win is proved. Only ever called when ``--vct-teacher`` is set; the
    default-off path never enters here, keeping self-play byte-identical.

    Caps: the VCT search fans out on the defender side, so on the GENERATION hot
    path it respects an AGGRESSIVE per-process budget (``_VCT_TEACHER_MAX_DEPTH``
    / ``_VCT_TEACHER_MAX_NODES``, decoupled from and far tighter than the
    general-purpose ``vcf.DEFAULT_VCT_MAX_*``) so a wide-open position bails to
    ``hit_cap=True`` / no-forced-win almost instantly and never blocks self-play
    (bead derby-b6r). Override per-process via :func:`configure_vct_teacher`.
    """
    if max_depth is None:
        max_depth = _VCT_MAX_DEPTH
    if max_nodes is None:
        max_nodes = _VCT_MAX_NODES
    with _profile_timer(profile, "vct_solve_s"):
        res = vcf.solve_vct_from_planes(
            planes, history_ply=HISTORY_PLY, max_depth=max_depth, max_nodes=max_nodes
        )
    _profile_add(profile, "vct_calls", 1.0)
    if not res.has_forced_win or res.winning_move is None:
        return pi, z, False

    new_pi = np.zeros_like(pi)
    new_pi[res.winning_move] = 1.0

    dist = res.mate_distance if res.mate_distance is not None else 1
    value = VCF_VALUE_DISCOUNT ** max(0, dist - 1)
    if value < VCF_VALUE_FLOOR:
        value = VCF_VALUE_FLOOR
    _profile_add(profile, "vct_fired", 1.0)
    return new_pi, float(value), True


def _apply_defense_teacher(
    planes: np.ndarray,
    z: float,
    *,
    vcf_already_fired: bool,
    profile: ProfileStats | None = None,
    max_depth: int | None = None,
    max_nodes: int | None = None,
) -> tuple[float, bool]:
    """Opt-in EXACT *defensive* teacher (VALUE-ONLY): the mirror of
    :func:`_apply_vcf_teacher`. Where the offensive teacher proves a forced WIN
    for the side to move, this proves a forced win for the OPPONENT against the
    side to move — i.e. the recorded position is already lost — and relabels the
    value target to :data:`_DEFENSE_SOFT_VALUE` (default ``-1.0``; #42 allows a
    softer target) — teaching "you should have defended earlier".

    The per-game FRACTION cap (:data:`_DEFENSE_MAX_FRACTION`) is NOT enforced
    here (this is a single-position solve); the per-game caller selects which of
    the firing plies actually keep the relabel via :func:`_defense_budget`.

    Returns ``(new_z, fired)``. The POLICY target is never touched: defense is
    non-unique (many moves may equally delay the loss), so there is no single
    correct policy label — only the value is sound to stamp. (Policy/refutation
    mode is explicitly out of scope for this teacher.)

    Detection: solve the SAME position with the two planes SWAPPED so the
    OPPONENT becomes the attacker (plane 0) and the side-to-move becomes the
    defender (plane 1). A proven VCF win for that swapped attacker is a proven
    forced loss for the recorded side-to-move. We reuse :func:`vcf.solve_vcf`
    verbatim on the swapped board (mirroring the plane reconstruction in
    :func:`vcf.solve_vcf_from_planes`).

    GEN-COST GATE (two cheap guards before the expensive solve, so quiet
    positions cost ZERO solver recursion and we never double the gen cost):
      (a) If the offensive VCF teacher already fired here, the side to move has a
          PROVEN win, so it cannot also be in a proven loss — skip entirely.
      (b) Cheap danger pre-scan: only solve if the OPPONENT has at least one
          four-making move (:func:`vcf.has_four_threat` on the swapped board). A
          VCF line always opens with a four, so no four-threat => no opponent
          forced win => no solve needed. Sound (only ever skips work).

    Only ever called when ``--defense-teacher`` is set; the default-off path
    never enters here, so self-play stays byte-identical.
    """
    if vcf_already_fired:
        return z, False
    if max_depth is None:
        max_depth = _VCF_MAX_DEPTH
    if max_nodes is None:
        max_nodes = _VCF_MAX_NODES

    planes = np.asarray(planes)
    # Swap: opponent (plane HISTORY_PLY) becomes attacker, side-to-move (plane 0)
    # becomes defender. Mirrors vcf.solve_vcf_from_planes' reconstruction, swapped.
    opp = planes[HISTORY_PLY].astype(bool)
    me = planes[0].astype(bool)
    swapped_board = np.stack([opp, me], axis=0)

    _profile_add(profile, "defense_calls", 1.0)
    # (b) Cheap danger pre-scan: skip the solve unless the opponent has a four.
    with _profile_timer(profile, "defense_prescan_s"):
        danger = vcf.has_four_threat(swapped_board)
    if not danger:
        return z, False

    with _profile_timer(profile, "defense_solve_s"):
        res = vcf.solve_vcf(swapped_board, max_depth=max_depth, max_nodes=max_nodes)
    _profile_add(profile, "defense_solves", 1.0)
    if not res.has_forced_win:
        return z, False

    # The opponent has a proven forced win against the side to move: this
    # recorded position is lost. Value target = _DEFENSE_SOFT_VALUE (default -1.0,
    # the hard loss vertex; #42 allows a softer "you are losing" target). The
    # recorded z is from the recorded side's perspective, matching the side-to-move
    # of the un-swapped position, so a proven OPPONENT win maps directly to it.
    _profile_add(profile, "defense_fired", 1.0)
    return _DEFENSE_SOFT_VALUE, True


def _defense_detect_candidate(
    planes: np.ndarray,
    *,
    vcf_already_fired: bool,
    profile: ProfileStats | None = None,
    max_depth: int | None = None,
    max_nodes: int | None = None,
) -> tuple[np.ndarray, object] | None:
    """#43/#60 — the CHEAP first phase of the policy defense teacher: decide
    whether a ply is a refutation CANDIDATE WITHOUT paying for the (expensive)
    refutation enumeration.

    A ply is a candidate when (a) the offensive VCF teacher has not already fired,
    (b) the OPPONENT has a proven forced VCF win against the side to move (detection
    solve on the swapped board), and (c) the side to move does NOT itself have a
    forced win (the #43 self-contained "convert, don't defend" guard). Returns
    ``(swapped_board, winning_move)`` — the inputs the refutation needs — for a
    candidate, else ``None``. Whether a candidate actually FIRES (has a saving move)
    is known only after :func:`_defense_refute_stamp`; #60 defers that to the
    budgeted reverse pass so we never refute plies the budget would discard.

    The two cheap gen-cost gates (offensive-already-fired short-circuit, opponent
    four-threat pre-scan) are unchanged from the pre-#60 inline teacher. Increments
    ``defense_policy_candidates`` per candidate (the detection-level count, now
    decoupled from the budget-limited ``defense_policy_fired`` stamp count).
    """
    if vcf_already_fired:
        return None
    if max_depth is None:
        max_depth = _VCF_MAX_DEPTH
    if max_nodes is None:
        max_nodes = _VCF_MAX_NODES

    planes = np.asarray(planes)
    opp = planes[HISTORY_PLY].astype(bool)
    me = planes[0].astype(bool)
    swapped_board = np.stack([opp, me], axis=0)

    _profile_add(profile, "defense_calls", 1.0)
    with _profile_timer(profile, "defense_prescan_s"):
        danger = vcf.has_four_threat(swapped_board)
    if not danger:
        return None

    # Sparse-bite sampler (the #43 follow-on): invoke the EXACT solver on only a
    # fraction of danger plies. The cheap prescan above still runs on every ply;
    # this gate skips the expensive detection (and the guard + refute that follow)
    # for the un-sampled 90%. Default frac 1.0 keeps every solve (byte-identical).
    if _DEFENSE_DETECT_FRAC < 1.0 and _DEFENSE_SAMPLE_RNG.random() >= _DEFENSE_DETECT_FRAC:
        _profile_add(profile, "defense_detect_skipped", 1.0)
        return None

    with _profile_timer(profile, "defense_solve_s"):
        res = vcf.solve_vcf(swapped_board, max_depth=max_depth, max_nodes=max_nodes)
    _profile_add(profile, "defense_solves", 1.0)
    if not res.has_forced_win:
        return None

    # SELF-CONTAINED guard: if the side-to-move ITSELF has a proven forced win,
    # this is a position to CONVERT, not defend — do not overwrite its winning
    # policy. (Replaces the dead vcf_already_fired plumbing when --vcf-teacher is
    # off; also short-circuits before the expensive refutation on these positions.)
    own_board = np.stack([me, opp], axis=0)
    with _profile_timer(profile, "defense_solve_s"):
        own = vcf.solve_vcf(own_board, max_depth=max_depth, max_nodes=max_nodes)
    if own.has_forced_win:
        _profile_add(profile, "defense_own_win_skips", 1.0)
        return None

    _profile_add(profile, "defense_policy_candidates", 1.0)
    return swapped_board, res.winning_move


def _defense_refute_stamp(
    swapped_board: np.ndarray,
    winning_move: object,
    pi: np.ndarray,
    *,
    profile: ProfileStats | None = None,
    max_depth: int | None = None,
    max_nodes: int | None = None,
) -> np.ndarray | None:
    """#43/#60 — the EXPENSIVE second phase: enumerate the defender moves that
    BREAK the opponent's proven forced win and, if any exist, build the soft
    (uniform) saving-move policy target. Returns the new policy vector, or ``None``
    when the position is genuinely lost (no saving move) — the caller then leaves
    the record untouched (pure policy lever; value never crushed).

    Split out from :func:`_defense_detect_candidate` so the per-game caller can run
    it ONLY on the plies the FRACTION budget keeps (#60): the refutation is ~83% of
    the teacher's gen cost, and the budget discarded most fires, so refuting every
    candidate paid for work that was then thrown away. ``max_candidates`` bounds the
    per-fire re-solve count on pathologically dense boards.
    """
    if max_depth is None:
        max_depth = _VCF_MAX_DEPTH
    if max_nodes is None:
        max_nodes = _VCF_MAX_NODES
    with _profile_timer(profile, "defense_refute_s"):
        saving = vcf.vcf_refutations(
            swapped_board, winning_move=winning_move,
            max_depth=max_depth, max_nodes=max_nodes,
            max_candidates=_DEFENSE_REFUTE_MAX_CANDIDATES)
    if not saving:
        return None
    new_pi = np.zeros_like(pi)
    weight = 1.0 / len(saving)
    for mv in saving:
        new_pi[mv] = weight
    _profile_add(profile, "defense_fired", 1.0)
    _profile_add(profile, "defense_policy_fired", 1.0)
    return new_pi


def _apply_defense_teacher_policy(
    planes: np.ndarray,
    pi: np.ndarray,
    z: float,
    *,
    vcf_already_fired: bool,
    profile: ProfileStats | None = None,
    max_depth: int | None = None,
    max_nodes: int | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Opt-in EXACT *defensive* teacher (POLICY mode, issue #43): the I2 arm of
    #18's defense decomposition. Where :func:`_apply_defense_teacher` only crushes
    the VALUE to -1.0 ("you already lost" — a signal the net cannot act on, and
    which on a shared trunk contradicts the still-attacking policy, #41), this
    teaches the REFUTATION on the POLICY head and leaves the value at its natural
    discounted outcome.

    Detection is identical to the value teacher: swap planes so the OPPONENT is
    the attacker and confirm a proven forced VCF win against the side-to-move. The
    NEW step: because the side-to-move actually moves FIRST (one tempo before the
    attacker), enumerate the defender moves that BREAK that forced win
    (:func:`vcf.vcf_refutations`). If one or more SAVING moves exist, stamp a soft
    (uniform) policy target over them; the value is left untouched.

    Returns ``(new_pi, new_z, fired)``. ``fired`` is True only when (a) the
    opponent has a proven forced VCF win, (b) the side-to-move does NOT itself have
    a forced win (else it should CONVERT, not defend — see the self-contained guard
    below), AND (c) at least one refutation exists — a TRULY lost position (no
    saving move) returns ``(pi, z, False)`` and is left entirely untouched, keeping
    this a PURE policy lever (no value crushing). The gen-cost gates: skip when the
    offensive VCF teacher already fired, and a cheap opponent-four-threat pre-scan
    before the solve. Only ever called when ``--defense-teacher`` is set with policy
    mode on.

    SELF-CONTAINED "side-to-move already winning" guard (issue #43 audit fix): the
    ``vcf_already_fired`` parameter is the offensive teacher's signal that the
    side-to-move has a proven win — but that plumbing is DEAD whenever the cell runs
    ``--defense-teacher-policy`` WITHOUT ``--vcf-teacher`` (the actual G15-defense-i2
    configuration), so ``vcf_already_fired`` is always False there. Without an
    independent check the teacher would fire on positions where the side-to-move has
    its OWN forced win and OVERWRITE the correct sharp winning-move policy with a
    diluted defensive blend (policy says "defend" while value says "winning" — a
    contradictory target on a shared trunk). We therefore re-derive the condition
    HERE by solving the UN-swapped board (side-to-move as attacker); if the
    side-to-move has a forced win we bail and leave the record untouched, so the
    net keeps learning to convert. This also short-circuits before the expensive
    refutation enumeration on exactly those positions.
    """
    # (#60) Thin wrapper over the two-phase split — cheap detection then expensive
    # refutation. Single-position behavior and every profile counter are unchanged
    # (detection adds only the new additive `defense_policy_candidates` count); the
    # per-game caller in _apply_teachers_to_trajectory instead calls the two phases
    # directly so it refutes ONLY the budget-kept plies.
    cand = _defense_detect_candidate(
        planes, vcf_already_fired=vcf_already_fired, profile=profile,
        max_depth=max_depth, max_nodes=max_nodes)
    if cand is None:
        return pi, z, False
    swapped_board, winning_move = cand
    new_pi = _defense_refute_stamp(
        swapped_board, winning_move, pi, profile=profile,
        max_depth=max_depth, max_nodes=max_nodes)
    if new_pi is None:
        return pi, z, False
    return new_pi, z, True


def _conv_block_detect(
    planes: np.ndarray,
    *,
    tier2: bool = True,
) -> np.ndarray | None:
    """CHEAP, NO-tree-search defensive POLICY detector (the conv block-teacher).

    Reconstructs the board the same way the VCF defense teacher does — ``me =
    planes[0]`` is the DEFENDER (side to move), ``opp = planes[HISTORY_PLY]`` is
    the attacker — then, with a handful of numpy line scans (NO ``solve_vcf``, NO
    recursion), decides whether to stamp a defensive block and over which cells.

    Returns a SET of flat action indices (a 1-D int array) the defender should
    play, or ``None`` for "no stamp" (the caller leaves the record untouched). The
    caller turns a non-None result into a uniform policy target over those cells.

    Freestyle rules: >= 5 in a row wins (overlines included) — ``vcf._five_completions``
    already uses ``>=`` so a six-completion counts as a win cell.

    Detection rules (sound core first, then the heuristic bite):

    1. OWN-WIN guard. If the DEFENDER itself has any immediate five-completion,
       it should CONVERT, not defend -> return None (mirror of the policy teacher's
       StM-own-win guard / the offensive "side-to-move already winning" short-circuit).
    2. Opponent four. Let ``opp_win`` be the opponent's immediate five-completion
       cells (the cells where the opponent makes >= 5 next move).
         * exactly one  -> the unique FORCED BLOCK -> stamp that one cell. Blocking
           it or losing next move; unambiguously correct defense.
         * two or more  -> a double-four, genuinely lost -> return None (mirror of
           the exact teacher leaving truly-lost positions untouched).
    3. (Tier 2, ``tier2`` True) No opponent four, but the opponent has an OPEN
       THREE: a move ``f`` that would make an OPEN FOUR (>= 2 completion cells =
       unstoppable next move). The defender must occupy a cell that defeats EVERY
       such open-four threat. A defender move ``d`` defeats threat ``(f, comps)``
       iff ``d == f`` or ``d in comps`` (occupying the would-be open-four square or
       one of its completions). Stamp the INTERSECTION of all threats' defeating
       sets (uniform if several). If that intersection is empty (no single move
       stops all of them) return None — SOUND: never stamp a cell that does not
       actually prevent the open four (no false saves). This tier is a strong
       HEURISTIC, not strictly forced (the opponent has other replies), so it is
       behind a sub-toggle; on by default since the net likely already blocks bare
       fours and the open-three reflex is where the real bite is.

    Never mutates ``planes``. ``vcf`` helpers are reused only for the (cheap,
    non-recursive) collinear completion / open-four scans, never the solver.
    """
    planes = np.asarray(planes)
    me = planes[0].astype(bool)        # DEFENDER (side to move)
    opp = planes[HISTORY_PLY].astype(bool)  # attacker
    occupied = me | opp
    empty_plane = ~occupied

    # Rule 1: defender can win now -> convert, don't defend.
    if vcf._five_completions(me, empty_plane):
        return None

    # Rule 2: opponent's immediate five-completion cells (the opponent's "four").
    opp_win = vcf._five_completions(opp, empty_plane)
    if len(opp_win) == 1:
        return np.array([int(opp_win[0])], dtype=np.int64)  # forced block
    if len(opp_win) >= 2:
        return None  # double four -> genuinely lost, no stamp

    # Rule 3 (Tier 2): opponent open-three -> open-four prevention.
    if not tier2:
        return None
    empty_idx = vcf._empties_from_plane(empty_plane)
    if len(empty_idx) == 0:
        return None
    # Open-four-making moves for the opponent (>=2 completions), restricted to the
    # near-stone candidate cells (a four can only form next to existing stones).
    of_cands = vcf._candidate_cells_from_planes(opp, me, empty_idx)
    threats = vcf._open_four_threats(opp.copy(), me.copy(), empty_plane,
                                     [int(c) for c in of_cands])
    if not threats:
        return None
    # Intersection of each threat's defeating set {f} ∪ comps. A single defender
    # move in the intersection neutralizes every threatened open four at once; if
    # the intersection is empty, no one move suffices -> do not stamp (sound).
    defeating_sets = [set([f, *comps]) for f, comps in threats]
    common = set.intersection(*defeating_sets)
    # Only still-empty cells are legal stamps (all should be, but be defensive).
    common = {c for c in common
              if not occupied[c // vcf.BOARD_SIZE, c % vcf.BOARD_SIZE]}
    if not common:
        return None
    return np.array(sorted(common), dtype=np.int64)


def _apply_defense_teacher_conv(
    planes: np.ndarray,
    pi: np.ndarray,
    z: float,
    *,
    profile: ProfileStats | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Per-ply conv block-teacher: run :func:`_conv_block_detect` and, if it
    returns block cells, stamp a uniform policy target over them; the value ``z``
    is left untouched (pure policy lever, exactly like the #43 policy teacher).

    Returns ``(new_pi, z, fired)``; the inputs unchanged with ``fired=False`` when
    the detector declines (no threat / lost / own win). Profile counters:
    ``defense_conv_s`` (wall) and ``defense_conv_fired`` (stamp count).
    """
    with _profile_timer(profile, "defense_conv_s"):
        cells = _conv_block_detect(planes, tier2=_DEFENSE_CONV_TIER2)
    _profile_add(profile, "defense_conv_calls", 1.0)
    if cells is None or len(cells) == 0:
        return pi, z, False
    new_pi = np.zeros_like(pi)
    new_pi[cells] = 1.0 / float(len(cells))
    _profile_add(profile, "defense_conv_fired", 1.0)
    return new_pi, z, True


def _profile_add(profile: ProfileStats | None, key: str, value: float) -> None:
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + float(value)


class _profile_timer:
    def __init__(self, profile: ProfileStats | None, key: str):
        self.profile = profile
        self.key = key
        self.t0 = 0.0

    def __enter__(self):
        if self.profile is not None:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.profile is not None:
            _profile_add(self.profile, self.key, time.perf_counter() - self.t0)


@dataclass
class SelfPlayExample:
    """One training example: a canonical state, MCTS policy, and outcome.

    All three are from the SAME side-to-move perspective (canonical).
    """

    planes: np.ndarray   # (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE) float32
    pi: np.ndarray       # (N_ACTIONS,) float32, sums to 1 over legal actions
    z: float             # value in [-1, 1]
    side: int = 0        # 0=black (first mover), 1=white — for per-color diagnostics
    ply: int = 0         # ply count at which this position was captured
    # V3 auxiliary opponent-reply target: the MCTS policy `pi` recorded at the
    # NEXT ply in the same game (the opponent is to move there). None for the
    # LAST position of a game (no next ply) and for every position when the aux
    # lever is off — masked out of the aux loss in either case. Under D4
    # augmentation this vector is transformed by the SAME symmetry as `pi`.
    aux_pi: np.ndarray | None = None  # (N_ACTIONS,) float32 or None
    # V4 auxiliary ownership target: per-cell (81,) final control of the
    # played-out game, in the SAME absolute board frame as this position's
    # planes. +1 where the eventual WINNER's stone sits at the final board, -1
    # for the LOSER's stone, 0 for empty (all-zeros on a draw). Constant for
    # every position of a game (the game has one final board); transformed by
    # the SAME D4 symmetry as planes/pi/aux_pi under augmentation. None when the
    # ownership lever is off — masked out of the ownership loss.
    ownership: np.ndarray | None = None  # (N_ACTIONS,) float32 or None
    # Moonshot VCT-defense target: per-cell (81,) 0/1 "blunder map" for THIS
    # position (side-to-move = me). vct[m] = 1.0 iff playing at empty cell m walks
    # the side to move into a forced VCT for the opponent, 0.0 otherwise. Unlike
    # `ownership` (one per-GAME final board), this is a PER-PLY target computed by
    # the escape-search labeler at the position that is being recorded. Under D4
    # augmentation it is permuted by the SAME symmetry as planes/pi/ownership (it
    # is a per-cell board map). None when the vct lever is off or the position was
    # not labeled — masked out of the vct loss.
    vct: np.ndarray | None = None  # (N_ACTIONS,) float32 or None


@dataclass
class ChoiceExample:
    """One swap2 NEGOTIATION-choice training example for the choice head.

    Distinct from :class:`SelfPlayExample`: it lives at a swap2 CHOICE node (the
    responder's {STAY, SWAP, PLACE2} or the opener's pick-color), carries an
    N_CHOICES-wide legal mask, and its target is OUTCOME-driven (v2a): the slot
    actually played, re-signed to this chooser's frame by ``chooser_z`` (the
    game's outcome in [-1, 1] from the chooser's perspective). NOT D4-augmented
    (choice slots are not spatial), and NEVER stored in the main ReplayBuffer
    (which is shaped for the policy width) — it rides a separate ChoiceBuffer.
    """

    planes: np.ndarray       # (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE) float32 — the choice node
    legal_mask: np.ndarray   # (N_CHOICES,) bool — legal choice slots
    chosen: int              # the slot actually applied during negotiation
    chooser_z: float         # game outcome in [-1, 1] from THIS chooser's perspective


@dataclass
class GameRecord:
    """A completed game's training examples plus some metadata."""

    examples: list[SelfPlayExample]
    plies: int
    outcome: float  # +1 if first-mover (black) won, -1 if second-mover (white) won, 0 draw
    archive_start: bool = False  # WL5: True if game was seeded from validation archive
    # v2a swap2 choice-head examples. One per negotiation CHOICE node of the
    # game's opening (0 when the game did not use a swap2 opening). Default is an
    # empty list so every non-swap2 path stays byte-identical.
    choice_examples: list[ChoiceExample] = field(default_factory=list)


def _choice_examples_for_game(res, outcome_for_black: float) -> list[ChoiceExample]:
    """Build the v2a choice-head examples for one game from its swap2 result.

    `res` is the game's :class:`~gomoku.swap2_search.Swap2Result` (None when the
    game did not use a swap2 opening → no choice examples). `outcome_for_black`
    is +1 / -1 / 0 from the HAND-OFF MOVER's perspective: because `to_normal()`
    frames board[0] as the mover at hand-off, `outcome_for_black` is really the
    outcome for `res.mover_actor`. For each choice record we re-sign that to the
    chooser's frame: the actor-form of `backup_sign` is `+1 iff the chooser at
    that node IS the hand-off mover`, so
        chooser_z = (+1 if cr.to_act == res.mover_actor else -1) * outcome_for_black.
    Choice examples are NOT D4-augmented (the slots are not spatial).
    """
    if res is None or not res.choice_records:
        return []
    out: list[ChoiceExample] = []
    for cr in res.choice_records:
        sign = 1.0 if cr.to_act == res.mover_actor else -1.0
        out.append(
            ChoiceExample(
                planes=cr.planes,
                legal_mask=cr.legal_mask,
                chosen=int(cr.chosen),
                chooser_z=float(sign * outcome_for_black),
            )
        )
    return out


def _aux_target_for(
    traj: list[tuple[np.ndarray, np.ndarray, int]], ply_idx: int, side: int
) -> np.ndarray | None:
    """Opponent-reply target for the position at `ply_idx` in `traj`.

    The target is the MCTS policy `pi` recorded at the NEXT trajectory entry —
    the opponent is to move there. Returns None (→ masked out of the aux loss)
    when:
      * this is the LAST recorded ply (no next entry), or
      * the next recorded entry is the SAME side to move (can happen under
        Playout-Cap Randomization, where an unrecorded fast move sits between
        two recorded plies — then the "next recorded" position is NOT the
        immediate opponent reply, so we decline to use it as a label rather
        than feed a misaligned target).

    The returned vector is in the same board coordinate frame as the position's
    own `pi`; D4 alignment is handled downstream by augment_with_aux.
    """
    nxt = ply_idx + 1
    if nxt >= len(traj):
        return None
    next_pi, next_side = traj[nxt][1], traj[nxt][2]
    if int(next_side) == int(side):
        return None
    return next_pi


def _ownership_target(
    final_planes: np.ndarray, term_side_abs: int, outcome_for_black: float
) -> np.ndarray | None:
    """Per-cell (81,) ownership target from a game's FINAL board.

    `final_planes` is the canonical plane stack of the TERMINAL root (plane 0 =
    stones of the side to move at terminal, plane HISTORY_PLY = the other side).
    `term_side_abs` is the ABSOLUTE side to move at terminal (0 = black/first-
    mover, 1 = white). `outcome_for_black` is +1 if black won, -1 if white won,
    0 on a draw.

    Returns an (81,) float vector in ABSOLUTE board coordinates: +1 at the
    eventual WINNER's stones, -1 at the LOSER's stones, 0 at empty cells. On a
    draw (outcome 0) returns all zeros (no winner/loser to credit). The same
    vector applies to every position of the game; D4 alignment to each
    position is handled downstream by augment_with_cell_targets.

    Returns None only if the final planes are unusable (defensive).
    """
    if final_planes is None:
        return None
    fp = np.asarray(final_planes)
    # Map the terminal canonical planes to absolute black/white stone masks.
    cur = fp[0].reshape(-1).astype(np.float32)            # side-to-move @ terminal
    opp = fp[HISTORY_PLY].reshape(-1).astype(np.float32)  # the other side
    if int(term_side_abs) == 0:
        black, white = cur, opp
    else:
        black, white = opp, cur
    if outcome_for_black > 0:        # black won
        winner, loser = black, white
    elif outcome_for_black < 0:      # white won
        winner, loser = white, black
    else:                            # draw — nobody credited
        return np.zeros(N_ACTIONS, dtype=np.float32)
    return (winner - loser).astype(np.float32)


def _build_examples(
    planes: np.ndarray,
    pi: np.ndarray,
    z: float,
    side: int,
    ply_at_capture: int,
    aux_pi: np.ndarray | None,
    augment_symmetries: bool,
    profile: ProfileStats | None = None,
    ownership: np.ndarray | None = None,
    vct: np.ndarray | None = None,
) -> list[SelfPlayExample]:
    """Expand one recorded position into training example(s).

    With D4 augmentation, emits 8 symmetry variants; the per-cell aux targets
    (opponent-reply `aux_pi`, `ownership`, and/or the vct-defense blunder map,
    when present) are transformed by the SAME symmetry as planes+pi via
    augment_with_cell_targets, so each label stays board-aligned with its
    position. Without augmentation, emits the single position. A None target
    propagates to the example unchanged → masked out of that target's loss.

    Byte-identical guarantee: when aux_pi, ownership, AND vct are all None, this
    uses the plain `augment` (the pre-aux path), so the off-case output is
    unchanged.
    """
    out: list[SelfPlayExample] = []
    if augment_symmetries:
        if aux_pi is None and ownership is None and vct is None:
            with _profile_timer(profile, "d4_augment_s"):
                augmented = list(augment(planes, pi))
            for aug_planes, aug_pi in augmented:
                with _profile_timer(profile, "example_create_s"):
                    out.append(SelfPlayExample(
                        aug_planes, aug_pi.astype(np.float32), z,
                        side=int(side), ply=int(ply_at_capture),
                        aux_pi=None, ownership=None, vct=None,
                    ))
        else:
            # One or more per-cell targets present: carry them through the
            # identical D4 symmetry as planes+pi. Slot order is fixed
            # [aux_pi, ownership, vct]; a missing target keeps its slot as None so
            # the example field stays None (masked out of that loss).
            cell_targets = []
            aux_slot = ownership_slot = vct_slot = None
            if aux_pi is not None:
                aux_slot = len(cell_targets)
                cell_targets.append(aux_pi)
            if ownership is not None:
                ownership_slot = len(cell_targets)
                cell_targets.append(ownership)
            if vct is not None:
                vct_slot = len(cell_targets)
                cell_targets.append(vct)
            with _profile_timer(profile, "d4_augment_s"):
                augmented = list(augment_with_cell_targets(planes, pi, cell_targets))
            for aug_planes, aug_pi, aug_targets in augmented:
                with _profile_timer(profile, "example_create_s"):
                    out.append(SelfPlayExample(
                        aug_planes, aug_pi.astype(np.float32), z,
                        side=int(side), ply=int(ply_at_capture),
                        aux_pi=(aug_targets[aux_slot].astype(np.float32)
                                if aux_slot is not None else None),
                        ownership=(aug_targets[ownership_slot].astype(np.float32)
                                   if ownership_slot is not None else None),
                        vct=(aug_targets[vct_slot].astype(np.float32)
                             if vct_slot is not None else None),
                    ))
    else:
        with _profile_timer(profile, "example_create_s"):
            out.append(SelfPlayExample(
                planes, pi.astype(np.float32), z,
                side=int(side), ply=int(ply_at_capture),
                aux_pi=(aux_pi.astype(np.float32) if aux_pi is not None else None),
                ownership=(ownership.astype(np.float32) if ownership is not None else None),
                vct=(vct.astype(np.float32) if vct is not None else None),
            ))
    return out


def _sample_action(pi: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an action from a probability distribution. Falls back to argmax
    if the distribution is degenerate (zero/NaN sum) or contains NaN entries.

    NaN guard is load-bearing: WL3 (wandb 0o75gws5) crashed all 8 workers
    at e825 from `ValueError: Probabilities contain NaN` because native
    MCTS occasionally emits NaN visit-policies. Without the guard, `s <= 0`
    returned False on NaN (NaN comparisons are False), the divide
    propagated NaN, and rng.choice rejected. The underlying MCTS NaN
    source is the real fix — this is the safety net.
    """
    s = pi.sum()
    if not np.isfinite(s) or s <= 0:
        return int(np.argmax(np.nan_to_num(pi, nan=-np.inf)))
    pi = pi / s
    if not np.all(np.isfinite(pi)):
        return int(np.argmax(np.nan_to_num(pi, nan=-np.inf)))
    return int(rng.choice(len(pi), p=pi))


def _contempt_distribution(
    visits: np.ndarray,
    child_w: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Build the contempt-perturbed action distribution from child visit counts
    and cumulative win values (W[a] from the C MCTS backup, in the parent
    side-to-move's perspective).

    For each child with N[a] > 0, the contempt SCORE is `-|child_Q[a]|` where
    `child_Q[a] = W[a] / N[a]` ∈ [-1, +1]. Children with the most CONTESTED
    Q (closest to 0) score highest, so the resulting softmax peaks on moves
    that lead to harder-to-convert positions (the paper's mechanism). The
    softmax uses `max(tau, 1e-2)` as a temperature floor so the play
    distribution doesn't collapse to one-hot near the late-game tau=0.1
    regime (the recorded `pi` already collapses; we want some exploration in
    the contempt-perturbed play distribution to keep the position-flow
    diverse). Children with N[a] == 0 (never visited by MCTS) score `-inf`
    (no Q estimate, never picked by contempt). Illegal/unvisited actions get
    zero probability mass.

    Returns a length-`N_ACTIONS` float64 array summing to 1.0 over the visited
    legal children, or all-zeros if no children have visits (degenerate).
    """
    n = np.asarray(visits, dtype=np.int64)
    w = np.asarray(child_w, dtype=np.float64)
    out = np.zeros_like(w, dtype=np.float64)
    visited = n > 0
    if not visited.any():
        return out
    q = np.zeros_like(w, dtype=np.float64)
    q[visited] = w[visited] / n[visited]
    score = np.full_like(q, -np.inf, dtype=np.float64)
    score[visited] = -np.abs(q[visited])
    tau_eff = max(float(tau), 1e-2)
    # numerically stable softmax over the visited subset
    finite_scores = score[visited]
    m = float(finite_scores.max())
    shifted = (finite_scores - m) / tau_eff
    exps = np.exp(shifted)
    Z = float(exps.sum())
    if Z <= 0 or not np.isfinite(Z):
        # All-equal-or-degenerate fallback: uniform over visited children.
        idx = np.flatnonzero(visited)
        out[idx] = 1.0 / float(len(idx))
        return out
    probs = exps / Z
    idx = np.flatnonzero(visited)
    out[idx] = probs
    return out


def _contempt_sample_action(
    visits: np.ndarray,
    child_w: np.ndarray,
    pi: np.ndarray,
    tau: float,
    rng: np.random.Generator,
) -> int:
    """Sample a move from the CONTEMPT-perturbed action distribution.

    See `_contempt_distribution` for the math. Falls back to the visit-policy
    `pi` (via `_sample_action`) if the contempt distribution is degenerate
    (no visited children — happens at the very first call before any sims
    have backed up). Guarantees a finite, legal action choice (the caller's
    `_sample_action` NaN safety net is the final fallback).
    """
    contempt_pi = _contempt_distribution(visits, child_w, tau)
    s = float(contempt_pi.sum())
    if not np.isfinite(s) or s <= 0:
        return _sample_action(pi, rng)
    return int(rng.choice(len(contempt_pi), p=contempt_pi / s))


def _random_opening_state(rng: np.random.Generator, n_moves: int) -> tuple[GameState, int]:
    """Play `n_moves` uniform-random legal moves from an empty board, then return
    the resulting state and the ply count (= n_moves, unless the random play hit
    terminal first — in which case we restart from empty).

    Used by generate_games / generate_games_vs_baseline to inject opening diversity.
    """
    if n_moves <= 0:
        return GameState.initial(), 0
    while True:
        state = GameState.initial()
        plies = 0
        for _ in range(n_moves):
            legal = state.legal_actions()
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))
            state = state.apply(action)
            plies += 1
            done, _ = state.is_terminal()
            if done:
                # Unlucky — random play accidentally ended the game. Try again.
                break
        else:
            return state, plies
        # restart from scratch


# Rapfi's 9 hand-curated BALANCED swap2 openings, defined for a 15x15 board.
# VERIFIED faithful to Rapfi source @6e0a1329 (Rapfi/search/opening.cpp, the static
# SWAP2 list for board.size()==15): these exact 9 triples, stored as bare Pos(x,y)
# with NO explicit colors -- color is implied purely by placement ORDER from an
# empty BLACK-first board, so stone[0]=BLACK, stone[1]=WHITE, stone[2]=BLACK
# (2 black + 1 white) and after 3 plies currentSide=~BLACK => WHITE to move. That
# is exactly the board Rapfi EMITS to the swap2 responder ("Rapfi says go"): a
# pre-decision 3-stone, white-to-move position; neither color is illegal (free-
# style). Our construction below reproduces it byte-for-byte (cross-checked against
# swap2.OpeningState.to_normal() in tests/test_fixed_openings.py). So we capture
# Rapfi's MOVE, not just its shape -- white-to-move is the protocol-correct seat,
# not an assumption. (The broad swap2 protocol CAN also settle to a 5-stone,
# black-to-move board via the responder's place-2 branch, but Rapfi's hardcoded
# canon -- and thus ours -- is strictly the 3-stone white-to-move board.)
# Used by the fixed-opening training mode (#73): we hand the net these fair
# positions DIRECTLY and skip the (unfair) negotiation entirely.
_RAPFI_BALANCED_OPENINGS_15: tuple[tuple[tuple[int, int], ...], ...] = (
    ((6, 7), (6, 4), (4, 2)),
    ((3, 3), (5, 5), (6, 6)),
    ((3, 2), (5, 4), (4, 5)),
    ((5, 2), (1, 5), (1, 6)),
    ((8, 5), (5, 8), (6, 7)),
    ((5, 5), (8, 8), (7, 7)),
    ((13, 12), (13, 9), (10, 12)),
    ((11, 7), (10, 6), (13, 5)),
    ((3, 7), (1, 8), (0, 4)),
)

# Fair-opening LADDER (#73/#74): the SAME Rapfi shapes RE-CENTERED onto the smaller
# boards (translate each 3-stone cluster so its bounding box sits at board center;
# the footprints are <=3x6 so all 9 fit on 9/11/13 -- none dropped). The relative
# arrangement (and thus the rough balance) is preserved; centering keeps them off
# the edges. 15 keeps Rapfi's native placements. So a fresh net trains the same
# canned fair openers at every rung, climbing 9->11->13->15.
_FAIR_OPENINGS: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    9: (
        ((5, 7), (5, 4), (3, 2)), ((3, 3), (5, 5), (6, 6)), ((3, 2), (5, 4), (4, 5)),
        ((6, 2), (2, 5), (2, 6)), ((6, 3), (3, 6), (4, 5)), ((3, 3), (6, 6), (5, 5)),
        ((5, 6), (5, 3), (2, 6)), ((3, 5), (2, 4), (5, 3)), ((5, 5), (3, 6), (2, 2)),
    ),
    11: (
        ((6, 7), (6, 4), (4, 2)), ((3, 3), (5, 5), (6, 6)), ((4, 4), (6, 6), (5, 7)),
        ((7, 3), (3, 6), (3, 7)), ((6, 3), (3, 6), (4, 5)), ((3, 3), (6, 6), (5, 5)),
        ((7, 6), (7, 3), (4, 6)), ((5, 6), (4, 5), (7, 4)), ((7, 6), (5, 7), (4, 3)),
    ),
    13: (
        ((7, 9), (7, 6), (5, 4)), ((5, 5), (7, 7), (8, 8)), ((5, 4), (7, 6), (6, 7)),
        ((8, 4), (4, 7), (4, 8)), ((8, 5), (5, 8), (6, 7)), ((5, 5), (8, 8), (7, 7)),
        ((7, 8), (7, 5), (4, 8)), ((5, 7), (4, 6), (7, 5)), ((7, 7), (5, 8), (4, 4)),
    ),
    15: _RAPFI_BALANCED_OPENINGS_15,
}


def _active_fixed_openings(
    board_size: int | None = None,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """The fixed opening book for ``board_size`` (default: active), minus any
    indices listed in ``$GOMOKU_DROP_OPENERS`` (comma-separated indices into the
    ORIGINAL book).

    The black-advantage prune gate (#73, 9x9): if a rung shows a persistent black
    edge, the orchestrator identifies the most black-favoring opener (offline
    balance probe) and re-runs with that index dropped via this env var -- no
    source edit, reversible, indices stay stable across drops. Raises if the drop
    list would empty the book.
    """
    n = BOARD_SIZE if board_size is None else board_size
    openings = _FAIR_OPENINGS.get(n)
    if openings is None:
        raise ValueError(
            f"_active_fixed_openings: no fixed opening book for board size "
            f"{n}; have {sorted(_FAIR_OPENINGS)} (set GOMOKU_BOARD_SIZE)"
        )
    raw = os.environ.get("GOMOKU_DROP_OPENERS", "").strip()
    if not raw:
        return openings
    drop = {int(t) for t in raw.replace(",", " ").split() if t.strip() != ""}
    kept = tuple(o for i, o in enumerate(openings) if i not in drop)
    if not kept:
        raise ValueError(
            f"GOMOKU_DROP_OPENERS={raw!r} would drop every opener for "
            f"board {n} (book has {len(openings)})"
        )
    return kept


def _fixed_opening_state(
    rng: np.random.Generator,
    openings: tuple[tuple[tuple[int, int], ...], ...] | None = None,
) -> tuple[GameState, int]:
    """Pick one fixed BALANCED opening uniformly and place its 3 stones directly.

    Stones are colored BLACK, WHITE, BLACK in list order (the swap2 opener order),
    yielding 2 black + 1 white -> white to move. Returns ``(GameState, plies=3)``,
    mirroring ``_random_opening_state``'s contract so the gen loops drop it into the
    same opening branch. NO swap2 negotiation, NO net call, NO choice records -- the
    construction is byte-identical to ``swap2.OpeningState.to_normal()`` for the
    2-black-1-white SWAP outcome (plane 0 = mover/white stones, plane 1 = black).

    The opening book is selected by the active board size (9/11/13/15); raises on
    any other size.
    """
    if openings is None:
        openings = _active_fixed_openings(BOARD_SIZE)
    opening = openings[int(rng.integers(0, len(openings)))]
    black = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    white = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    for i, (x, y) in enumerate(opening):  # x=col, y=row -> board[y, x]
        (black if i % 2 == 0 else white)[y, x] = True  # i=0,2 black; i=1 white
    # 2 black + 1 white -> nb == nw + 1 -> WHITE to move; plane 0 = mover (white).
    board = np.stack([white, black]).astype(bool)
    state = GameState(board=board, move_count=3, history=())
    assert not state.is_terminal()[0]  # 3 scattered stones can never be 5-in-a-row
    return state, 3


def _make_swap2_oracle(evaluator: Evaluator):
    """Wrap a batch `evaluator` as a single-state swap2 oracle.

    The negotiator wants `Callable[[GameState], (policy_probs, value)]` where
    `policy_probs` is a true probability distribution over `N_ACTIONS` (sums to
    1) and `value` is a float in [-1, 1] from the state's side-to-move. The
    evaluator returns RAW LOGITS for a batch, so we softmax the single row. The
    opening is ~30 forwards/game (negligible), so a batch-of-1 call is fine.
    """

    def oracle(gs: GameState) -> tuple[np.ndarray, float]:
        priors, values = evaluator([gs])
        logits = np.asarray(priors[0], dtype=np.float64)
        logits = logits - logits.max()  # numerically stable softmax
        e = np.exp(logits)
        probs = e / e.sum()
        return probs, float(values[0])

    return oracle


def _swap2_opening_state(
    evaluator: Evaluator, rng: np.random.Generator
) -> tuple[GameState, int, "Swap2Result"]:
    """Negotiate a swap2 opening and return `(normal_state, opening_plies, res)`.

    Mirrors `_random_opening_state`'s `(state, plies)` contract so the generation
    loops can swap it in at the same opening seam: the returned state is the
    canonical, legal, non-terminal position normal play begins from, and
    `opening_plies` is its move count (the stones placed during negotiation,
    which — like the random opening prefix — are NOT recorded as policy/value
    training examples). v2a ALSO threads out the full `Swap2Result` so the caller
    can fold the negotiation `choice_records` into choice-head examples (the
    `res.choice_records` / `res.mover_actor` were previously discarded).
    """
    from gomoku.swap2_search import negotiate

    oracle = _make_swap2_oracle(evaluator)
    res = negotiate(oracle, rng)
    start_state = res.normal_state
    return start_state, start_state.move_count, res


def _gamestate_from_archive(archive: dict, idx: int) -> GameState:
    """WL5 archive-start: build a GameState from one archived position.

    Reads plane 0 (current-side stones) and plane HISTORY_PLY (opponent
    stones) at archive index `idx` and constructs a fresh `GameState`
    with `move_count` taken from `archive["ply"][idx]`. History tuple is
    left empty — the archive only persists planes, not move order, so
    planes 1..H-1 / H+1..2H-1 will read as zeros from the new state. The
    archived current+opponent positions and side-to-move are preserved;
    only the per-ply history snapshots are lost. Acceptable per the WL5
    design doc (history is "hard without move order").
    """
    planes = archive["planes"][idx].cpu().numpy()
    current = planes[0].astype(bool)
    opponent = planes[HISTORY_PLY].astype(bool)
    board = np.stack([current, opponent], axis=0)
    move_count = int(archive["ply"][idx].item())
    return GameState(board=board, move_count=move_count, history=())


def _can_use_native_mcts(evaluator: Evaluator) -> bool:
    return bool(
        native_mcts.USING_NATIVE_MCTS
        and native_mcts.NativeMCTSGame is not None
        and hasattr(evaluator, "evaluate_planes")
    )


def _can_use_native_gumbel(evaluator: Evaluator) -> bool:
    """True iff the native engine is available AND it implements the Gumbel
    batch path (the built .so exposes gumbel_search_batch). On a source-only or
    pre-port .so this is False and the pure-Python fallback is used.
    """
    return bool(_can_use_native_mcts(evaluator) and native_mcts.has_native_gumbel())


def _generate_games_native_gumbel(
    n_games: int,
    evaluator: Evaluator,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.25,
    c_puct_base: float = 19652.0,
    max_plies: int | None = None,
    rng: np.random.Generator | None = None,
    augment_symmetries: bool = True,
    wave_size: int = 16,
    random_opening_moves: int = 0,
    swap2: bool = False,
    fixed_openings: bool = False,
    archive: dict | None = None,
    archive_start_frac: float = 0.0,
    profile: ProfileStats | None = None,
    gumbel_m: int = 16,
    gumbel_c_visit: float = 50.0,
    gumbel_c_scale: float = 1.0,
    vcf_teacher: bool = False,
    vct_teacher: bool = False,
    defense_teacher: bool = False,
    record_aux: bool = False,
    record_ownership: bool = False,
) -> list[GameRecord]:
    """Self-play using the NATIVE Gumbel root + Sequential Halving path.

    Mirrors `_generate_games_native` (wave-batched native C engine, subtree
    reuse via advance_root) but:
      - constructs games with `gumbel_root=1` so the native engine forces the
        root edge to the Sequential-Halving-chosen candidate;
      - drives search with `native_mcts.gumbel_search_batch` (one batched leaf
        eval per slot, shared across games — the wall-fairness win);
      - the move played is `g.gumbel_selected_action()` (the SH argmax), NOT a
        temperature-sampled visit-count draw;
      - the training target is `g.gumbel_policy()` (the COMPLETED-POLICY), NOT
        visit counts.

    Identical training-signal math to `_generate_games_gumbel` (the Python
    fallback), just running in the fast native engine.
    """
    rng = rng or np.random.default_rng()
    if max_plies is None:
        max_plies = N_ACTIONS

    planes_evaluator = evaluator.evaluate_planes  # type: ignore[attr-defined]

    def _timed_planes_evaluator(planes_batch):
        t0 = time.perf_counter()
        try:
            return planes_evaluator(planes_batch)
        finally:
            dt = time.perf_counter() - t0
            _profile_add(profile, "evaluator_s", dt)
            try:
                batch_n = int(getattr(planes_batch, "shape", [len(planes_batch)])[0])
            except Exception:
                batch_n = 0
            _profile_add(profile, "evaluator_calls", 1.0)
            _profile_add(profile, "evaluator_positions", float(batch_n))

    search_evaluator = _timed_planes_evaluator if profile is not None else planes_evaluator

    games = []
    initial_plies: list[int] = []
    archive_start_flags: list[bool] = []
    swap2_results: list = []  # per-game Swap2Result (None unless a swap2 opening)
    n_archive = 0 if archive is None else int(archive["planes"].shape[0])
    with _profile_timer(profile, "game_setup_s"):
        for _ in range(n_games):
            from_archive = (
                archive is not None
                and n_archive > 0
                and float(rng.random()) < archive_start_frac
            )
            swap2_res = None
            if from_archive:
                idx = int(rng.integers(0, n_archive))
                start_state = _gamestate_from_archive(archive, idx)
                opening_plies = start_state.move_count
            elif swap2:
                start_state, opening_plies, swap2_res = _swap2_opening_state(evaluator, rng)
            elif fixed_openings:
                start_state, opening_plies = _fixed_opening_state(rng)
            elif random_opening_moves > 0:
                start_state, opening_plies = _random_opening_state(rng, random_opening_moves)
            else:
                start_state, opening_plies = GameState.initial(), 0
            games.append(
                native_mcts.NativeMCTSGame(
                    start_state,
                    c_puct=c_puct,
                    c_puct_base=c_puct_base,
                    seed=int(rng.integers(1, 2**63 - 1)),
                    gumbel_root=1,
                    gumbel_m=gumbel_m,
                    gumbel_c_visit=gumbel_c_visit,
                    gumbel_c_scale=gumbel_c_scale,
                )
            )
            initial_plies.append(opening_plies)
            archive_start_flags.append(from_archive)
            swap2_results.append(swap2_res)

    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    active: list[int] = list(range(n_games))
    completed: list[tuple[int, float, int]] = []
    # Ownership: final-board planes + terminal absolute side per game, captured
    # at the moment of termination. Only populated when record_ownership is on.
    final_state: dict[int, tuple[np.ndarray, int]] = {}

    ply = 0
    while active and ply < max_plies:
        with _profile_timer(profile, "active_list_s"):
            active_games = [games[i] for i in active]
        with _profile_timer(profile, "native_search_batch_s"):
            native_mcts.gumbel_search_batch(
                active_games,
                search_evaluator,
                n_simulations=n_simulations,
                wave_size=wave_size,
            )
        _profile_add(profile, "search_calls", 1.0)

        next_active: list[int] = []
        with _profile_timer(profile, "post_search_loop_s"):
            for slot_idx, g_idx in enumerate(active):
                g = active_games[slot_idx]
                with _profile_timer(profile, "policy_export_s"):
                    pi = g.gumbel_policy()
                n_initial = initial_plies[g_idx]
                side = (n_initial + ply) % 2
                with _profile_timer(profile, "policy_sanitize_s"):
                    if not np.all(np.isfinite(pi)):
                        pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
                        s = pi.sum()
                        pi = pi / s if s > 0 else np.full_like(pi, 1.0 / len(pi))
                with _profile_timer(profile, "root_planes_s"):
                    planes = g.root_planes()
                with _profile_timer(profile, "trajectory_append_s"):
                    trajectories[g_idx].append((planes, pi.copy(), side))

                # The move is the Sequential-Halving argmax (NO temperature
                # sampling — the Gumbel noise IS the exploration).
                action = int(g.gumbel_selected_action())
                if action < 0:
                    # Degenerate (terminal/no legal); fall back to argmax of pi.
                    action = int(np.argmax(pi))
                # Search-contempt (Derby 'x-search-contempt', bead derby-qoq):
                # with probability _CONTEMPT_P, REPLACE the SH-chosen move with
                # a contempt-perturbed pick that favors children with Q closest
                # to 0 (most contested). Uses tau=1.0 for the softmax (Gumbel
                # has no temperature schedule of its own; the recorded training
                # target `pi` is the COMPLETED-POLICY and is unchanged — only
                # the position distribution shifts). OFF (_CONTEMPT_P <= 0):
                # no roll, no W read, byte-identical baseline.
                if _CONTEMPT_P > 0.0 and float(rng.random()) < _CONTEMPT_P:
                    ds = g.gumbel_debug_state()
                    action = _contempt_sample_action(
                        g.visit_counts(), ds["W"], pi, 1.0, rng,
                    )
                with _profile_timer(profile, "advance_root_s"):
                    g.advance_root(action)
                with _profile_timer(profile, "terminal_check_s"):
                    done, term_val = g.is_terminal()
                if done:
                    if term_val == -1.0:
                        winner_side = side
                        outcome_for_black = 1.0 if winner_side == 0 else -1.0
                    else:
                        outcome_for_black = 0.0
                    if record_ownership:
                        # After advance_root the root is the TERMINAL board; its
                        # side-to-move (plane 0) is the player who did NOT just
                        # move, parity (n_initial+ply+1)%2 in absolute terms.
                        final_state[g_idx] = (
                            np.asarray(g.root_planes()).copy(),
                            (n_initial + ply + 1) % 2,
                        )
                    completed.append((g_idx, outcome_for_black, n_initial + ply + 1))
                else:
                    next_active.append(g_idx)

        active = next_active
        ply += 1

    for g_idx in active:
        if record_ownership:
            # Game hit max_plies without terminating: scored a draw. final_planes
            # is None => _ownership_target returns None => this game is MASKED OUT
            # of the ownership loss (not scored). A genuine board-full draw, by
            # contrast, carries real final_planes and IS scored as all-zeros.
            final_state[g_idx] = (None, 0)
        completed.append((g_idx, 0.0, initial_plies[g_idx] + ply))

    records: list[GameRecord] = []
    with _profile_timer(profile, "record_build_s"):
        for g_idx, outcome_for_black, plies in sorted(completed):
            examples: list[SelfPlayExample] = []
            n_initial = initial_plies[g_idx]
            traj = trajectories[g_idx]
            ownership = None
            if record_ownership:
                fp, term_side = final_state.get(g_idx, (None, 0))
                ownership = _ownership_target(fp, term_side, outcome_for_black)
            finalized = _apply_teachers_to_trajectory(
                traj, outcome_for_black, vcf_teacher=vcf_teacher,
                vct_teacher=vct_teacher, defense_teacher=defense_teacher,
                profile=profile)
            for ply_idx, (planes, _pi_orig, side) in enumerate(traj):
                pi, z = finalized[ply_idx]
                ply_at_capture = n_initial + ply_idx
                aux_pi = _aux_target_for(traj, ply_idx, side) if record_aux else None
                examples.extend(_build_examples(
                    planes, pi, z, side, ply_at_capture, aux_pi,
                    augment_symmetries, profile, ownership=ownership,
                ))
            records.append(GameRecord(
                examples=examples,
                plies=plies,
                outcome=outcome_for_black,
                archive_start=archive_start_flags[g_idx],
                choice_examples=_choice_examples_for_game(
                    swap2_results[g_idx], outcome_for_black),
            ))
    return records


def _generate_games_gumbel(
    n_games: int,
    evaluator: Evaluator,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.25,
    c_puct_base: float = 19652.0,
    temperature_moves: int = 8,  # unused: Gumbel selects greedily by its own score
    temperature_final: float = 0.1,  # unused
    dirichlet_alpha: float = 0.3,  # unused: Gumbel uses Gumbel noise, not Dirichlet
    dirichlet_eps: float = 0.25,  # unused
    max_plies: int | None = None,
    rng: np.random.Generator | None = None,
    augment_symmetries: bool = True,
    random_opening_moves: int = 0,
    swap2: bool = False,
    fixed_openings: bool = False,
    archive: dict | None = None,
    archive_start_frac: float = 0.0,
    profile: ProfileStats | None = None,
    gumbel_m: int = 16,
    gumbel_c_visit: float = 50.0,
    gumbel_c_scale: float = 1.0,
    vcf_teacher: bool = False,
    vct_teacher: bool = False,
    defense_teacher: bool = False,
    record_aux: bool = False,
    record_ownership: bool = False,
) -> list[GameRecord]:
    """Self-play generation using Gumbel AlphaZero root selection + Sequential
    Halving (the pure-Python `gomoku.mcts.Node` tree path).

    Differences from the standard path:
      - Root action is chosen by `gumbel_search_root` (Gumbel-top-k candidate
        sampling, Sequential Halving over the sim budget, argmax of the SH score).
        There is NO temperature sampling and NO Dirichlet noise — the Gumbel
        noise IS the exploration, and the SH argmax IS the move.
      - The training policy target is the COMPLETED-POLICY (softmax of
        logits + sigma(q_hat) with v-mix completion for unvisited actions),
        NOT the visit-count distribution.

    Each game keeps its own `MCTSGame` so subtrees are reused across plies.
    """
    from gomoku.mcts import MCTSGame
    from gomoku.gumbel import gumbel_search_root

    rng = rng or np.random.default_rng()
    if max_plies is None:
        max_plies = N_ACTIONS
    _ = (temperature_moves, temperature_final, dirichlet_alpha, dirichlet_eps)

    games: list[MCTSGame] = []
    initial_plies: list[int] = []
    archive_start_flags: list[bool] = []
    swap2_results: list = []  # per-game Swap2Result (None unless a swap2 opening)
    n_archive = 0 if archive is None else int(archive["planes"].shape[0])
    for _ in range(n_games):
        from_archive = (
            archive is not None and n_archive > 0
            and float(rng.random()) < archive_start_frac
        )
        swap2_res = None
        if from_archive:
            idx = int(rng.integers(0, n_archive))
            start_state = _gamestate_from_archive(archive, idx)
            opening_plies = start_state.move_count
        elif swap2:
            start_state, opening_plies, swap2_res = _swap2_opening_state(evaluator, rng)
        elif fixed_openings:
            start_state, opening_plies = _fixed_opening_state(rng)
        elif random_opening_moves > 0:
            start_state, opening_plies = _random_opening_state(rng, random_opening_moves)
        else:
            start_state, opening_plies = GameState.initial(), 0
        games.append(
            MCTSGame(
                start_state,
                c_puct=c_puct,
                c_puct_base=c_puct_base,
                rng=np.random.default_rng(rng.integers(0, 2**31)),
            )
        )
        initial_plies.append(opening_plies)
        archive_start_flags.append(from_archive)
        swap2_results.append(swap2_res)

    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    active: list[int] = list(range(n_games))
    completed: list[tuple[int, float, int]] = []
    final_state: dict[int, tuple[np.ndarray, int]] = {}

    ply = 0
    while active and ply < max_plies:
        with _profile_timer(profile, "gumbel_search_s"):
            for g_idx in active:
                g = games[g_idx]
                # Per-move Gumbel noise: draw from the game's own rng.
                result = gumbel_search_root(
                    g.root,
                    evaluator,
                    n_simulations=n_simulations,
                    m=gumbel_m,
                    c_visit=gumbel_c_visit,
                    c_scale=gumbel_c_scale,
                    c_puct_init=c_puct,
                    c_puct_base=c_puct_base,
                    rng=g.rng,
                )
                n_initial = initial_plies[g_idx]
                side = (n_initial + ply) % 2
                pi = result.pi
                # Sanitize the target (mirrors the native path's NaN guard).
                if not np.all(np.isfinite(pi)):
                    pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
                    s = pi.sum()
                    pi = pi / s if s > 0 else np.full_like(pi, 1.0 / len(pi))
                trajectories[g_idx].append((g.root.state.to_planes(), pi.copy(), side))
                g.advance_root(result.action)

        next_active: list[int] = []
        for g_idx in active:
            g = games[g_idx]
            n_initial = initial_plies[g_idx]
            done, term_val = g.root.state.is_terminal()
            if done:
                if term_val == -1.0:
                    winner_side = (n_initial + ply) % 2
                    outcome_for_black = 1.0 if winner_side == 0 else -1.0
                else:
                    outcome_for_black = 0.0
                if record_ownership:
                    # g.root.state is the TERMINAL state; plane 0 = side-to-move
                    # there (the non-mover), abs parity (n_initial+ply+1)%2.
                    final_state[g_idx] = (
                        g.root.state.to_planes(), (n_initial + ply + 1) % 2,
                    )
                completed.append((g_idx, outcome_for_black, n_initial + ply + 1))
            else:
                next_active.append(g_idx)
        active = next_active
        ply += 1

    for g_idx in active:
        if record_ownership:
            final_state[g_idx] = (None, 0)  # max-plies draw -> ownership MASKED OUT (None), not scored
        completed.append((g_idx, 0.0, initial_plies[g_idx] + ply))

    records: list[GameRecord] = []
    for g_idx, outcome_for_black, plies in sorted(completed):
        examples: list[SelfPlayExample] = []
        n_initial = initial_plies[g_idx]
        traj = trajectories[g_idx]
        ownership = None
        if record_ownership:
            fp, term_side = final_state.get(g_idx, (None, 0))
            ownership = _ownership_target(fp, term_side, outcome_for_black)
        finalized = _apply_teachers_to_trajectory(
            traj, outcome_for_black, vcf_teacher=vcf_teacher,
            vct_teacher=vct_teacher, defense_teacher=defense_teacher,
            profile=profile)
        for ply_idx, (planes, _pi_orig, side) in enumerate(traj):
            pi, z = finalized[ply_idx]
            ply_at_capture = n_initial + ply_idx
            aux_pi = _aux_target_for(traj, ply_idx, side) if record_aux else None
            examples.extend(_build_examples(
                planes, pi, z, side, ply_at_capture, aux_pi,
                augment_symmetries, None, ownership=ownership,
            ))
        records.append(GameRecord(
            examples=examples,
            plies=plies,
            outcome=outcome_for_black,
            archive_start=archive_start_flags[g_idx],
            choice_examples=_choice_examples_for_game(
                swap2_results[g_idx], outcome_for_black),
        ))
    return records


def _generate_games_native(
    n_games: int,
    evaluator: Evaluator,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.25,
    c_puct_base: float = 19652.0,
    temperature_moves: int = 8,
    temperature_final: float = 0.1,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    max_plies: int | None = None,
    rng: np.random.Generator | None = None,
    augment_symmetries: bool = True,
    wave_size: int = 1,
    random_opening_moves: int = 0,
    swap2: bool = False,
    fixed_openings: bool = False,
    archive: dict | None = None,
    archive_start_frac: float = 0.0,
    playout_cap_frac: float = 1.0,
    playout_cap_fast_sims: int = 0,
    forced_playout_k: float = 0.0,
    profile: ProfileStats | None = None,
    vcf_teacher: bool = False,
    vct_teacher: bool = False,
    defense_teacher: bool = False,
    record_aux: bool = False,
    record_ownership: bool = False,
    record_vct: bool = False,
    concurrent_games: int = 0,
    flush_records=None,
    flush_games: int = 32,
    refresh_evaluator=None,
) -> list[GameRecord]:
    rng = rng or np.random.default_rng()
    if max_plies is None:
        max_plies = N_ACTIONS

    # Playout-Cap Randomization (KataGo, Wu 2019). When `playout_cap_frac < 1.0`,
    # each move is, with probability `playout_cap_frac`, a FULL-search move (run
    # at `n_simulations` AND recorded as a training target); otherwise it is a
    # FAST move (run at `fast_sims` and NOT recorded — used only to advance the
    # game). Concentrates expensive search on the moves that actually train the
    # net. `playout_cap_frac >= 1.0` (default) => every move is full + recorded =
    # byte-identical to the pre-PCR production path. `fast_sims == 0` (default)
    # means "same as n_simulations", so the lever is inert unless BOTH the frac
    # is reduced and a smaller fast budget is set.
    pcr_active = playout_cap_frac < 1.0
    fast_sims = playout_cap_fast_sims if playout_cap_fast_sims > 0 else n_simulations

    # Evaluator indirection: streaming mode (refresh_evaluator) hot-swaps the
    # net between rounds — games in flight keep their trees, subsequent leaf
    # evals use the new weights (the standard continuous-self-play shape).
    eval_box = {"ev": evaluator}

    def planes_evaluator(planes_batch):
        return eval_box["ev"].evaluate_planes(planes_batch)  # type: ignore[attr-defined]

    def _timed_planes_evaluator(planes_batch):
        t0 = time.perf_counter()
        try:
            return planes_evaluator(planes_batch)
        finally:
            dt = time.perf_counter() - t0
            _profile_add(profile, "evaluator_s", dt)
            try:
                batch_n = int(getattr(planes_batch, "shape", [len(planes_batch)])[0])
            except Exception:
                batch_n = 0
            _profile_add(profile, "evaluator_calls", 1.0)
            _profile_add(profile, "evaluator_positions", float(batch_n))

    search_evaluator = _timed_planes_evaluator if profile is not None else planes_evaluator

    games = []
    initial_plies: list[int] = []
    archive_start_flags: list[bool] = []
    swap2_results: list = []  # per-game Swap2Result (None unless a swap2 opening)
    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = []
    # Moonshot VCT-defense per-ply targets, kept in LOCKSTEP with `trajectories`:
    # vct_maps[g_idx][k] is the (N_ACTIONS,) blunder map (or None) for the k-th
    # RECORDED position of game g_idx. Terminal VCT-terminus one-hot positions are
    # appended to `trajectories` by the partition (which knows nothing of vct) and
    # get NO vct_maps entry — they sit at the end, so the finalize loop reads a
    # None (masked out) for any ply_idx past the vct_maps length. Off (record_vct
    # False) => never populated, byte-identical to the pre-vct path.
    vct_maps: list[list] = []
    # Game-local ply counts (moves made since gen start, per game). On the
    # legacy lockstep path every active game shares the same value; continuous
    # refill (below) is what desynchronizes them.
    ply_of: list[int] = []
    n_archive = 0 if archive is None else int(archive["planes"].shape[0])

    def _seed_game() -> int:
        """Create one new game (same opening logic + RNG draw order as the
        historical setup loop) and append it to every per-game structure.
        Returns the new game index."""
        from_archive = (
            archive is not None
            and n_archive > 0
            and float(rng.random()) < archive_start_frac
        )
        swap2_res = None
        if from_archive:
            idx = int(rng.integers(0, n_archive))
            start_state = _gamestate_from_archive(archive, idx)
            opening_plies = start_state.move_count
        elif swap2:
            start_state, opening_plies, swap2_res = _swap2_opening_state(
                eval_box["ev"], rng)
        elif fixed_openings:
            start_state, opening_plies = _fixed_opening_state(rng)
        elif random_opening_moves > 0:
            start_state, opening_plies = _random_opening_state(rng, random_opening_moves)
        else:
            start_state, opening_plies = GameState.initial(), 0
        games.append(
            native_mcts.NativeMCTSGame(
                start_state,
                c_puct=c_puct,
                c_puct_base=c_puct_base,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                seed=int(rng.integers(1, 2**63 - 1)),
                forced_playout_k=forced_playout_k,
            )
        )
        initial_plies.append(opening_plies)
        archive_start_flags.append(from_archive)
        swap2_results.append(swap2_res)
        trajectories.append([])
        vct_maps.append([])
        ply_of.append(0)
        return len(games) - 1

    # Continuous refill (issue #112): cap the ACTIVE set at `concurrent_games`
    # and seed a replacement the moment a game completes, so every merged
    # oracle solve (one TAIL ~= 44 ms regardless of width) and every MPS
    # search wave serves a full-width batch instead of the thinning tail of a
    # lockstep batch. 0 (default) = legacy lockstep, byte-identical.
    refill = 0 < concurrent_games < n_games
    width = concurrent_games if refill else n_games
    with _profile_timer(profile, "game_setup_s"):
        for _ in range(width):
            _seed_game()

    active: list[int] = list(range(width))
    completed: list[tuple[int, float, int]] = []
    final_state: dict[int, tuple[np.ndarray, int]] = {}
    flush_queue: list[tuple[int, float, int]] = []

    def _build_record(g_idx: int, outcome_for_black: float, plies: int) -> GameRecord:
        examples: list[SelfPlayExample] = []
        n_initial = initial_plies[g_idx]
        traj = trajectories[g_idx]
        ownership = None
        if record_ownership:
            fp, term_side = final_state.get(g_idx, (None, 0))
            ownership = _ownership_target(fp, term_side, outcome_for_black)
        finalized = _apply_teachers_to_trajectory(
            traj, outcome_for_black, vcf_teacher=vcf_teacher,
            vct_teacher=vct_teacher, defense_teacher=defense_teacher,
            profile=profile)
        vmaps = vct_maps[g_idx] if record_vct else None
        for ply_idx, (planes, _pi_orig, side) in enumerate(traj):
            pi, z = finalized[ply_idx]
            ply_at_capture = n_initial + ply_idx
            aux_pi = _aux_target_for(traj, ply_idx, side) if record_aux else None
            # Per-ply vct map (None for terminal VCT-terminus positions that
            # sit past the recorded-position count → masked out of the loss).
            vct = (vmaps[ply_idx]
                   if vmaps is not None and ply_idx < len(vmaps) else None)
            examples.extend(_build_examples(
                planes, pi, z, side, ply_at_capture, aux_pi,
                augment_symmetries, profile, ownership=ownership, vct=vct,
            ))
        return GameRecord(
            examples=examples,
            plies=plies,
            outcome=outcome_for_black,
            archive_start=archive_start_flags[g_idx],
            choice_examples=_choice_examples_for_game(
                swap2_results[g_idx], outcome_for_black),
        )

    while active:
        # Streaming (issue #112): hot-swap the evaluator between rounds (games
        # in flight keep their trees; later leaf evals use the new weights —
        # the standard continuous-self-play shape).
        if refresh_evaluator is not None:
            new_ev = refresh_evaluator()
            if new_ev is not None:
                eval_box["ev"] = new_ev
        n_done_before = len(completed)
        with _profile_timer(profile, "active_list_s"):
            active_games = [games[i] for i in active]

        # Oracle phase (issues #98/#107): the attacker VCT-terminus test, the
        # VCT-defense blunder maps (recorded aux target and/or root-policy
        # veto), and the defender terminus all consume ONE merged bulk
        # mega-solve per ply (_oracle_ply_solve — one solver call costs one
        # TAIL; the historical two-call form cost two; per-board verdicts are
        # independent of batch composition, so the merge is bit-identical).
        # The veto's escape-solve runs FULL breadth by default — an untested
        # cell can't be vetoed, and "all moves lose" is only sound when every
        # legal cell was tested; --oracle-veto-max-cands K > 0 switches to the
        # staged-escalation form (K nearest cells, full breadth only for
        # positions whose tested cells ALL lose). With _ORACLE_OVERLAP_ENABLED
        # the solve runs in a background thread WHILE the MPS search below
        # runs, and the partitions apply post-search (flag-gated; see the
        # semantics note at the flag). Default features OFF = byte-identical.
        pending_vct: dict[int, np.ndarray | None] = {}
        oracle_wanted = _VCT_TERMINUS_ENABLED or record_vct or _ORACLE_VETO_ENABLED
        oracle_box: dict = {}
        oracle_thread = None
        planes_list: list | None = None
        slot_of: dict[int, int] = {}
        if oracle_wanted:
            with _profile_timer(profile, "oracle_planes_s"):
                planes_list = [g.root_planes() for g in active_games]
            slot_of = {g_idx: s for s, g_idx in enumerate(active)}
            defense_cands = (
                _VETO_MAX_CANDS if _ORACLE_VETO_ENABLED else _VCT_DEFENSE_MAX_CANDS)
            want_defense = record_vct or _ORACLE_VETO_ENABLED
            if _ORACLE_OVERLAP_ENABLED:
                _warm_mega_solver()   # compile on the main thread, once

                def _oracle_bg(pl=planes_list, dc=defense_cands,
                               wt=_VCT_TERMINUS_ENABLED, wd=want_defense):
                    try:
                        oracle_box["res"] = _oracle_ply_solve(
                            pl, want_terminus=wt, want_defense=wd,
                            defense_max_cands=dc, profile=profile)
                    except BaseException as exc:  # pragma: no cover - re-raised
                        oracle_box["err"] = exc

                oracle_thread = threading.Thread(target=_oracle_bg, daemon=True)
                oracle_thread.start()
            else:
                oracle_box["res"] = _oracle_ply_solve(
                    planes_list, want_terminus=_VCT_TERMINUS_ENABLED,
                    want_defense=want_defense, defense_max_cands=defense_cands,
                    profile=profile)
                active, active_games, pending_vct = _apply_oracle_partitions(
                    oracle_box["res"], active, active_games, planes_list,
                    slot_of, ply_of, initial_plies, trajectories, completed,
                    final_state, record_ownership, record_vct, vct_maps,
                    profile)
                if not active and not refill:
                    break

        # Playout-Cap Randomization: decide per-game-per-ply whether this is a
        # full-search (recorded) move or a fast (non-recorded) move. The native
        # search_batch requires a single sim count per call, so when PCR is
        # active we split the active wave into two homogeneous sub-batches and
        # call search_batch once per sub-batch with its own sim count. The
        # `is_full` flag (indexed by slot) gates the training-target append
        # below; game advancement happens identically for both kinds of move.
        if pcr_active:
            is_full = rng.random(len(active_games)) < playout_cap_frac
            # Recorded-move gating is keyed by g_idx (not slot) so it stays
            # correct when the overlap path partitions `active` post-search;
            # in the default serial path `active` is unchanged between here
            # and the post-search loop, so this is byte-identical to the old
            # slot indexing.
            full_set = {active[s] for s in range(len(active_games)) if is_full[s]}
            full_slots = [s for s in range(len(active_games)) if is_full[s]]
            fast_slots = [s for s in range(len(active_games)) if not is_full[s]]
            with _profile_timer(profile, "native_search_batch_s"):
                if full_slots:
                    native_mcts.search_batch(
                        [active_games[s] for s in full_slots],
                        search_evaluator,
                        n_simulations=n_simulations,
                        wave_size=wave_size,
                        add_root_noise=True,
                    )
                    _profile_add(profile, "search_calls", 1.0)
                if fast_slots:
                    # Fast moves are not training targets, so they carry no
                    # exploration obligation: skip Dirichlet root noise to keep
                    # the cheap search greedier (KataGo intent — exploration
                    # noise belongs on the recorded/full moves). Temperature
                    # (the play-sampling knob) still applies to both kinds.
                    native_mcts.search_batch(
                        [active_games[s] for s in fast_slots],
                        search_evaluator,
                        n_simulations=fast_sims,
                        wave_size=wave_size,
                        add_root_noise=False,
                    )
                    _profile_add(profile, "search_calls", 1.0)
        else:
            is_full = None  # every move is full + recorded (production path)
            full_set = None
            if active_games:  # refill: a round can be emptied pre-search
                with _profile_timer(profile, "native_search_batch_s"):
                    native_mcts.search_batch(
                        active_games,
                        search_evaluator,
                        n_simulations=n_simulations,
                        wave_size=wave_size,
                        add_root_noise=True,
                    )
                _profile_add(profile, "search_calls", 1.0)

        # Oracle/search overlap: join the background solve and apply the
        # partitions AFTER the search (games the oracle terminates this ply
        # were searched anyway and are simply dropped — their extra search is
        # the price of hiding the solve under the MPS wave).
        if oracle_thread is not None:
            with _profile_timer(profile, "oracle_join_stall_s"):
                oracle_thread.join()
            if "err" in oracle_box:
                raise oracle_box["err"]
            active, active_games, pending_vct = _apply_oracle_partitions(
                oracle_box["res"], active, active_games, planes_list, slot_of,
                ply_of, initial_plies, trajectories, completed, final_state,
                record_ownership, record_vct, vct_maps, profile)
            if not active and not refill:
                break

        next_active: list[int] = []
        with _profile_timer(profile, "post_search_loop_s"):
            for slot_idx, g_idx in enumerate(active):
                g = active_games[slot_idx]
                ply_g = ply_of[g_idx]
                record_target = is_full is None or g_idx in full_set
                tau = 1.0 if ply_g < temperature_moves else temperature_final
                with _profile_timer(profile, "policy_export_s"):
                    pi = g.policy(temperature=tau)
                # Sound-world oracle veto (issue #107): zero out proven-blunder
                # cells and renormalize BEFORE pi is either recorded as the
                # training target or sampled for play — one masking point, both
                # consumers. The target stays the net's own visit distribution
                # (on-policy), just constrained to non-losing moves. NOTE: not
                # composed with search-contempt (the contempt pick reads raw
                # visits and could select a vetoed move; don't run both).
                if _ORACLE_VETO_ENABLED:
                    vmap = pending_vct.get(g_idx)
                    if vmap is not None and vmap.any():
                        with _profile_timer(profile, "oracle_veto_mask_s"):
                            # The root hasn't advanced since the oracle solve,
                            # so this ply's planes are reusable (same values as
                            # a fresh g.root_planes() export).
                            legal = _legal_mask_from_planes(
                                planes_list[slot_of[g_idx]])
                            pi = _veto_policy(pi, vmap, legal)
                            _profile_add(profile, "oracle_veto_masked", 1.0)
                n_initial = initial_plies[g_idx]
                side = (n_initial + ply_g) % 2
                # Sanitize pi before recording the training example: NaN entries
                # from the native MCTS policy export must not enter the buffer.
                # _sample_action handles NaN for the *play* path, but trajectories
                # are stored independently and feed the trainer's cross-entropy
                # target. A NaN target poisons the loss for the entire minibatch.
                # Replace NaN with 0 and re-normalize; if everything is NaN, fall
                # back to a uniform distribution (lowest-information target —
                # better than corrupting the buffer).
                if record_target:
                    with _profile_timer(profile, "policy_sanitize_s"):
                        if not np.all(np.isfinite(pi)):
                            pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
                            s = pi.sum()
                            if s <= 0:
                                pi = np.full_like(pi, 1.0 / len(pi))
                            else:
                                pi = pi / s
                    with _profile_timer(profile, "root_planes_s"):
                        # Reuse this ply's oracle plane export when available
                        # (the root hasn't advanced; same values, one fewer
                        # native materialization). Nothing mutates these
                        # arrays, so storing the shared reference is safe.
                        if planes_list is not None:
                            planes = planes_list[slot_of[g_idx]]
                        else:
                            planes = g.root_planes()
                    with _profile_timer(profile, "trajectory_append_s"):
                        trajectories[g_idx].append((planes, pi.copy(), side))
                        if record_vct:
                            # Keep vct_maps in lockstep with the recorded traj.
                            vct_maps[g_idx].append(pending_vct.get(g_idx))

                with _profile_timer(profile, "sample_action_s"):
                    # Search-contempt (Derby 'x-search-contempt', bead derby-qoq):
                    # with probability _CONTEMPT_P, REPLACE the standard temperature-
                    # sampled visit-policy pick with a contempt-perturbed pick that
                    # favors children with Q closest to 0 (most contested). The
                    # recorded training target `pi` above is UNCHANGED — only the
                    # MOVE PLAYED (and thus the position distribution) shifts.
                    # OFF (_CONTEMPT_P <= 0): no roll, no W read, byte-identical
                    # baseline.
                    if _CONTEMPT_P > 0.0 and float(rng.random()) < _CONTEMPT_P:
                        ds = g.gumbel_debug_state()
                        action = _contempt_sample_action(
                            g.visit_counts(), ds["W"], pi, tau, rng,
                        )
                    else:
                        action = _sample_action(pi, rng)
                with _profile_timer(profile, "advance_root_s"):
                    g.advance_root(action)
                with _profile_timer(profile, "terminal_check_s"):
                    done, term_val = g.is_terminal()
                if done:
                    if term_val == -1.0:
                        winner_side = side
                        outcome_for_black = 1.0 if winner_side == 0 else -1.0
                    else:
                        outcome_for_black = 0.0
                    if record_ownership:
                        # Post-advance root is the TERMINAL board; side-to-move
                        # (plane 0) is the non-mover, abs parity (n_initial+ply_g+1)%2.
                        final_state[g_idx] = (
                            np.asarray(g.root_planes()).copy(),
                            (n_initial + ply_g + 1) % 2,
                        )
                    completed.append((g_idx, outcome_for_black, n_initial + ply_g + 1))
                elif ply_g + 1 >= max_plies:
                    # Per-game max-plies draw (the legacy loop's post-while
                    # retirement, moved inline for the per-game ply world).
                    if record_ownership:
                        final_state[g_idx] = (None, 0)  # ownership MASKED OUT (None), not scored
                    completed.append((g_idx, 0.0, n_initial + max_plies))
                else:
                    next_active.append(g_idx)
                ply_of[g_idx] = ply_g + 1

        # Continuous refill: seed replacements the moment games finish, so the
        # next round's merged solve + search wave stay at full width.
        if refill and len(games) < n_games:
            with _profile_timer(profile, "game_setup_s"):
                while len(games) < n_games and len(next_active) < width:
                    next_active.append(_seed_game())

        # Free finished games' native trees now (never touched again); on the
        # streaming path also build + hand off chunks and drop their per-game
        # state so memory stays flat over an unbounded run.
        if completed and len(completed) > n_done_before:
            for g_idx, _o, _p in completed[n_done_before:]:
                games[g_idx] = None
            if flush_records is not None:
                flush_queue.extend(completed[n_done_before:])
                del completed[n_done_before:]
                if len(flush_queue) >= flush_games:
                    with _profile_timer(profile, "record_build_s"):
                        chunk = [_build_record(g, o, p) for g, o, p in flush_queue]
                    for g_idx, _o, _p in flush_queue:
                        trajectories[g_idx] = []
                        vct_maps[g_idx] = []
                        final_state.pop(g_idx, None)
                    flush_queue.clear()
                    flush_records(chunk)

        active = next_active

    if flush_records is not None:
        # Streaming: everything goes through the callback; flush the remainder.
        if flush_queue:
            with _profile_timer(profile, "record_build_s"):
                chunk = [_build_record(g, o, p) for g, o, p in flush_queue]
            flush_queue.clear()
            flush_records(chunk)
        return []

    records: list[GameRecord] = []
    with _profile_timer(profile, "record_build_s"):
        for g_idx, outcome_for_black, plies in sorted(completed):
            records.append(_build_record(g_idx, outcome_for_black, plies))

    return records


def generate_games(
    n_games: int,
    evaluator: Evaluator,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.25,
    c_puct_base: float = 19652.0,
    temperature_moves: int = 8,
    temperature_final: float = 0.1,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    max_plies: int | None = None,
    rng: np.random.Generator | None = None,
    augment_symmetries: bool = True,
    wave_size: int = 1,
    random_opening_moves: int = 0,
    swap2: bool = False,
    fixed_openings: bool = False,
    archive: dict | None = None,
    archive_start_frac: float = 0.0,
    playout_cap_frac: float = 1.0,
    playout_cap_fast_sims: int = 0,
    forced_playout_k: float = 0.0,
    profile: ProfileStats | None = None,
    gumbel_root: bool = False,
    gumbel_m: int = 16,
    gumbel_c_visit: float = 50.0,
    gumbel_c_scale: float = 1.0,
    vcf_teacher: bool = False,
    vct_teacher: bool = False,
    defense_teacher: bool = False,
    record_aux: bool = False,
    record_ownership: bool = False,
    record_vct: bool = False,
    concurrent_games: int = 0,
    flush_records=None,
    flush_games: int = 32,
    refresh_evaluator=None,
) -> list[GameRecord]:
    """Generate `n_games` self-play games in parallel.

    All games advance in lockstep: at each ply we batch-MCTS across active games,
    sample an action per game, apply it, and remove any games that ended.

    `concurrent_games` > 0 (native path only) caps the active set at that width
    and seeds a replacement game the moment one completes (continuous refill,
    issue #112): every merged oracle solve costs one ~44 ms tail regardless of
    width, so keeping the batch full amortizes it over `concurrent_games` games
    instead of a lockstep batch's thinning tail. 0 (default) = legacy lockstep,
    byte-identical. Ignored (with the lockstep behavior) on the pure-Python and
    Gumbel fallback paths.

    `flush_records` (native path only) switches to streaming delivery: completed
    games are built into GameRecords and handed to the callback in chunks of
    `flush_games` (their per-game state is freed, so memory stays flat over an
    unbounded run) and the function returns []. `refresh_evaluator` (native path
    only) is called once per round; returning a new evaluator hot-swaps the net
    between rounds — in-flight games keep their trees and finish under the new
    weights (the standard continuous-self-play shape).

    `gumbel_root=True` switches the ROOT to Gumbel AlphaZero selection + Sequential
    Halving (Danihelka et al. 2022): the root samples `gumbel_m` candidate actions
    via the Gumbel-top-k trick, allocates the sim budget with Sequential Halving,
    and the training policy target is the COMPLETED-POLICY (not visit counts).
    Internal nodes keep PUCT. Default `gumbel_root=False` is byte-identical to the
    standard path. The Gumbel path runs on the pure-Python `gomoku.mcts` tree
    (the native C engine does not implement Gumbel — see the C-port spec in
    `gomoku/gumbel.py`).

    `wave_size` > 1 enables zeb-style wave-batched MCTS with virtual loss:
    each round collects `wave_size` leaves per game in one batched evaluator
    call. wave_size=1 reduces to the original per-sim batching.

    `random_opening_moves` > 0 starts each game with that many uniform-random
    legal moves played (alternating sides); MCTS only takes over after that.
    No training examples are recorded for the random opening — only for moves
    chosen by MCTS. Breaks the "always-same-opening" collapse mode by forcing
    the model to learn from a diverse set of starting positions.

    `playout_cap_frac` < 1.0 enables Playout-Cap Randomization (KataGo, Wu 2019):
    each move is, with probability `playout_cap_frac`, a full-search move
    (`n_simulations` sims, recorded as a training target); otherwise a fast move
    (`playout_cap_fast_sims` sims, NOT recorded — used only to advance the
    game). Defaults (frac=1.0, fast_sims=0) preserve the current behavior
    exactly: every move is full-search and recorded. NOTE: PCR is only honored on
    the native MCTS path; the pure-Python fallback below ignores it.

    `forced_playout_k` > 0 enables KataGo forced playouts + policy-target
    pruning (Wu 2019). Root-only forcing during search, forced visits removed
    from the training target. 0.0 (default) is OFF == byte-identical legacy.
    Composes with PCR (forced-k shapes search + prunes the recorded target).

    `vcf_teacher` (Derby v4 'Tactically-exact' lever) enables the EXACT VCF
    teacher. Default OFF == byte-identical: the solver is never called and no
    target is touched. When ON, every RECORDED training position is run through
    the VCF solver (gomoku/vcf.py); if the side to move has a proven forced win
    by continuous fours, that position's policy target is overwritten with a
    one-hot on the exact winning move and its value target with a
    mate-distance-discounted +1.0. Applied uniformly across all paths (native,
    native-Gumbel, Python-Gumbel, Python fallback) at the record-build seam, so
    the control flag composes with gumbel_root / PCR / forced-playouts.

    `defense_teacher` (Derby 'x-defense' lever) enables the EXACT *defensive*
    teacher — the value-only mirror of `vcf_teacher`. Default OFF == byte-
    identical (solver never called). When ON, for each recorded position where
    the OPPONENT has a proven forced VCF win against the side to move, the value
    target is relabeled to -1.0 (the position is lost; the policy target is left
    untouched because defense is non-unique). Gen-cost-gated: it is skipped when
    the offensive teacher already fired here, and runs a cheap opponent-four-
    threat pre-scan before the (expensive) swapped-plane solve, so quiet
    positions cost zero solver calls. Applied at the same record-build seam.
    """
    rng = rng or np.random.default_rng()
    if swap2 and random_opening_moves > 0:
        # Swap2 owns the opening; the two openings are mutually exclusive.
        raise ValueError(
            "swap2 and random_opening_moves are mutually exclusive "
            "(swap2 negotiates the opening; set random_opening_moves=0)"
        )
    if fixed_openings and (swap2 or random_opening_moves > 0):
        # The fixed balanced opening book places the opening directly; it is
        # mutually exclusive with swap2 negotiation and random openings.
        raise ValueError(
            "fixed_openings is mutually exclusive with swap2 and "
            "random_opening_moves (the opening book owns the opening)"
        )
    if gumbel_root:
        if _VCT_TERMINUS_ENABLED:
            raise NotImplementedError(
                "--vct-terminus is not wired into the Gumbel generation paths; "
                "run the native / Python (non-Gumbel) path (drop gumbel_root).")
        # Gumbel root selection + Sequential Halving. When the native C engine
        # is available AND implements the Gumbel batch path, use it — it
        # inherits the wave-batching that keeps Gumbel wall-comparable to native
        # PUCT (the whole point of the C port). Otherwise fall back to the pure-
        # Python tree path (identical target math, ~5x slower). See the C-port
        # spec at the bottom of gomoku/gumbel.py.
        if _can_use_native_gumbel(evaluator):
            return _generate_games_native_gumbel(
                n_games,
                evaluator,
                n_simulations=n_simulations,
                c_puct=c_puct,
                c_puct_base=c_puct_base,
                max_plies=max_plies,
                rng=rng,
                augment_symmetries=augment_symmetries,
                wave_size=wave_size,
                random_opening_moves=random_opening_moves,
                swap2=swap2,
                fixed_openings=fixed_openings,
                archive=archive,
                archive_start_frac=archive_start_frac,
                profile=profile,
                gumbel_m=gumbel_m,
                gumbel_c_visit=gumbel_c_visit,
                gumbel_c_scale=gumbel_c_scale,
                vcf_teacher=vcf_teacher,
                vct_teacher=vct_teacher,
                defense_teacher=defense_teacher,
                record_aux=record_aux,
                record_ownership=record_ownership,
            )
        return _generate_games_gumbel(
            n_games,
            evaluator,
            n_simulations=n_simulations,
            c_puct=c_puct,
            c_puct_base=c_puct_base,
            temperature_moves=temperature_moves,
            temperature_final=temperature_final,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            max_plies=max_plies,
            rng=rng,
            augment_symmetries=augment_symmetries,
            random_opening_moves=random_opening_moves,
            swap2=swap2,
            fixed_openings=fixed_openings,
            archive=archive,
            archive_start_frac=archive_start_frac,
            profile=profile,
            gumbel_m=gumbel_m,
            gumbel_c_visit=gumbel_c_visit,
            gumbel_c_scale=gumbel_c_scale,
            vcf_teacher=vcf_teacher,
            vct_teacher=vct_teacher,
            defense_teacher=defense_teacher,
            record_aux=record_aux,
            record_ownership=record_ownership,
        )
    if _can_use_native_mcts(evaluator):
        return _generate_games_native(
            n_games,
            evaluator,
            n_simulations=n_simulations,
            c_puct=c_puct,
            c_puct_base=c_puct_base,
            temperature_moves=temperature_moves,
            temperature_final=temperature_final,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            max_plies=max_plies,
            rng=rng,
            augment_symmetries=augment_symmetries,
            wave_size=wave_size,
            random_opening_moves=random_opening_moves,
            swap2=swap2,
            fixed_openings=fixed_openings,
            archive=archive,
            archive_start_frac=archive_start_frac,
            playout_cap_frac=playout_cap_frac,
            playout_cap_fast_sims=playout_cap_fast_sims,
            forced_playout_k=forced_playout_k,
            profile=profile,
            vcf_teacher=vcf_teacher,
            vct_teacher=vct_teacher,
            defense_teacher=defense_teacher,
            record_aux=record_aux,
            record_ownership=record_ownership,
            record_vct=record_vct,
            concurrent_games=concurrent_games,
            flush_records=flush_records,
            flush_games=flush_games,
            refresh_evaluator=refresh_evaluator,
        )
    if max_plies is None:
        max_plies = N_ACTIONS  # full-board fallback (game can't have more than this)

    games: list[MCTSGame] = []
    initial_plies: list[int] = []
    swap2_results: list = []  # per-game Swap2Result (None unless a swap2 opening)
    for _ in range(n_games):
        swap2_res = None
        if swap2:
            start_state, opening_plies, swap2_res = _swap2_opening_state(evaluator, rng)
        elif fixed_openings:
            start_state, opening_plies = _fixed_opening_state(rng)
        elif random_opening_moves > 0:
            start_state, opening_plies = _random_opening_state(rng, random_opening_moves)
        else:
            start_state, opening_plies = GameState.initial(), 0
        games.append(MCTSGame(start_state, c_puct=c_puct, c_puct_base=c_puct_base,
                              dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
                              forced_playout_k=forced_playout_k,
                              rng=np.random.default_rng(rng.integers(0, 2**31))))
        initial_plies.append(opening_plies)
        swap2_results.append(swap2_res)

    # Per-game trajectory of (planes, pi, side_to_move_at_that_ply)
    # side_to_move is encoded as 0 for the player who moved first ("black"), 1 for the other.
    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    # Moonshot VCT-defense per-ply targets, in LOCKSTEP with `trajectories` (see
    # the native path for the alignment contract). Off => never populated.
    vct_maps: list[list] = [[] for _ in range(n_games)]
    active: list[int] = list(range(n_games))
    completed: list[tuple[int, float, int]] = []  # (game_idx, outcome_for_black, plies)
    final_state: dict[int, tuple[np.ndarray, int]] = {}

    ply = 0
    while active and ply < max_plies:
        active_games = [games[i] for i in active]
        # VCT-terminus (issue #98): same batched pre-search test as the native
        # path; the Python game exposes planes via g.root.state.to_planes().
        pending_vct: dict[int, np.ndarray | None] = {}
        if _VCT_TERMINUS_ENABLED:
            planes_list = [g.root.state.to_planes() for g in active_games]
            active, active_games = _vct_terminus_partition(
                active, active_games, planes_list, ply, initial_plies,
                trajectories, completed, final_state, record_ownership, profile)
            if not active:
                break
            # Moonshot VCT-defense labeler over the surviving positions (one bulk
            # escape-solve per ply); attached to this ply's recorded position.
            if record_vct:
                surv_planes = [g.root.state.to_planes() for g in active_games]
                vmaps_ply, vmasks_ply = _vct_defense_solve(
                    surv_planes, max_cands=_VCT_DEFENSE_MAX_CANDS)
                for k, g_idx in enumerate(active):
                    pending_vct[g_idx] = (
                        vmaps_ply[k] if vmasks_ply[k].any() else None)
        if wave_size > 1:
            run_batched_mcts_waves(
                active_games,
                evaluator,
                n_simulations=n_simulations,
                wave_size=wave_size,
                add_root_noise=True,
            )
        else:
            run_batched_mcts(
                active_games,
                evaluator,
                n_simulations=n_simulations,
                add_root_noise=True,
            )

        next_active: list[int] = []
        for slot_idx, g_idx in enumerate(active):
            g = active_games[slot_idx]
            tau = 1.0 if ply < temperature_moves else temperature_final
            pi = policy_from_visits(
                g.root, tau,
                forced_playout_k=forced_playout_k,
                c_puct_init=c_puct, c_puct_base=c_puct_base,
            )
            # Total moves played so far in THIS game = initial_plies[g_idx] (random
            # opening) + ply (MCTS moves applied so far). The side ABOUT to move
            # at this point has parity = total_moves % 2.
            n_initial = initial_plies[g_idx]
            side = (n_initial + ply) % 2
            trajectories[g_idx].append((g.root.state.to_planes(), pi.copy(), side))
            if record_vct:
                vct_maps[g_idx].append(pending_vct.get(g_idx))

            action = _sample_action(pi, rng)
            g.advance_root(action)
            done, term_val = g.root.state.is_terminal()
            if done:
                # term_val is from the NEW root's side-to-move perspective.
                # The new side-to-move is the player who DIDN'T just move.
                # If term_val == -1, the player who just moved (side) won.
                # If term_val == 0, draw.
                if term_val == -1.0:
                    winner_side = side
                    outcome_for_black = 1.0 if winner_side == 0 else -1.0
                else:
                    outcome_for_black = 0.0
                if record_ownership:
                    final_state[g_idx] = (
                        g.root.state.to_planes(), (n_initial + ply + 1) % 2,
                    )
                completed.append((g_idx, outcome_for_black, n_initial + ply + 1))
            else:
                next_active.append(g_idx)

        active = next_active
        ply += 1

    # Any games still active at max_plies are scored as draws. Total plies is
    # the MCTS-loop ply plus the per-game random opening prefix.
    for g_idx in active:
        if record_ownership:
            final_state[g_idx] = (None, 0)  # max-plies draw -> ownership MASKED OUT (None), not scored
        completed.append((g_idx, 0.0, initial_plies[g_idx] + ply))

    # Build records, applying symmetry augmentation.
    records: list[GameRecord] = []
    for g_idx, outcome_for_black, plies in sorted(completed):
        examples: list[SelfPlayExample] = []
        n_initial = initial_plies[g_idx]
        traj = trajectories[g_idx]
        ownership = None
        if record_ownership:
            fp, term_side = final_state.get(g_idx, (None, 0))
            ownership = _ownership_target(fp, term_side, outcome_for_black)
        finalized = _apply_teachers_to_trajectory(
            traj, outcome_for_black, vcf_teacher=vcf_teacher,
            vct_teacher=vct_teacher, defense_teacher=defense_teacher,
            profile=profile)
        vmaps = vct_maps[g_idx] if record_vct else None
        for ply_idx, (planes, _pi_orig, side) in enumerate(traj):
            pi, z = finalized[ply_idx]
            ply_at_capture = n_initial + ply_idx
            aux_pi = _aux_target_for(traj, ply_idx, side) if record_aux else None
            # Per-ply vct map (None for terminal VCT-terminus positions past the
            # recorded-position count → masked out of the loss).
            vct = (vmaps[ply_idx]
                   if vmaps is not None and ply_idx < len(vmaps) else None)
            examples.extend(_build_examples(
                planes, pi, z, side, ply_at_capture, aux_pi,
                augment_symmetries, None, ownership=ownership, vct=vct,
            ))
        records.append(GameRecord(
            examples=examples, plies=plies, outcome=outcome_for_black,
            choice_examples=_choice_examples_for_game(
                swap2_results[g_idx], outcome_for_black),
        ))

    return records


def generate_games_vs_baseline(
    n_games: int,
    evaluator: Evaluator,
    opponent_picker,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.25,
    c_puct_base: float = 19652.0,
    temperature_moves: int = 8,
    temperature_final: float = 0.1,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    max_plies: int | None = None,
    rng: np.random.Generator | None = None,
    augment_symmetries: bool = True,
    wave_size: int = 1,
    model_first_frac: float = 0.5,
    random_opening_moves: int = 0,
    forced_playout_k: float = 0.0,
    vcf_teacher: bool = False,
    vct_teacher: bool = False,
    defense_teacher: bool = False,
) -> list[GameRecord]:
    """Generate games where the model plays a fixed opponent picker.

    The model uses MCTS (with the same `wave_size` / `dirichlet` / `c_puct` knobs
    as self-play). The opponent uses `opponent_picker(state, rng)` directly —
    no MCTS, no examples recorded for opponent moves. Training examples come
    only from the model's plies, with `z` set to the game's outcome from the
    model's perspective.

    `model_first_frac` is the fraction of games where the model plays the first
    move (side 0); the rest the model plays second. Default 0.5 so the model
    sees both sides equally.

    `GameRecord.outcome` is from the MODEL's perspective (+1 win, -1 loss, 0 draw)
    rather than first-mover's.
    """
    rng = rng or np.random.default_rng()
    if max_plies is None:
        max_plies = N_ACTIONS

    games: list[MCTSGame] = []
    initial_plies: list[int] = []
    for _ in range(n_games):
        if random_opening_moves > 0:
            start_state, opening_plies = _random_opening_state(rng, random_opening_moves)
        else:
            start_state, opening_plies = GameState.initial(), 0
        games.append(MCTSGame(start_state, c_puct=c_puct, c_puct_base=c_puct_base,
                              dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
                              forced_playout_k=forced_playout_k,
                              rng=np.random.default_rng(rng.integers(0, 2**31))))
        initial_plies.append(opening_plies)
    # side the model plays in each game: 0 = first mover, 1 = second
    model_side = np.where(rng.random(n_games) < model_first_frac, 0, 1).astype(np.int8)

    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    active: list[int] = list(range(n_games))
    completed: list[tuple[int, float, int]] = []  # (game_idx, outcome_for_model, plies)

    # initial_plies is constant across games (always == random_opening_moves), so
    # all games share the same `side_to_move` at any loop ply.
    n_initial = random_opening_moves
    ply = 0
    while active and ply < max_plies:
        side_to_move = (n_initial + ply) % 2
        model_turn = [i for i in active if model_side[i] == side_to_move]
        opp_turn = [i for i in active if model_side[i] != side_to_move]

        # Model: batched MCTS on its subset of games.
        if model_turn:
            mcts_games = [games[i] for i in model_turn]
            if wave_size > 1:
                run_batched_mcts_waves(
                    mcts_games, evaluator,
                    n_simulations=n_simulations, wave_size=wave_size,
                    add_root_noise=True,
                )
            else:
                run_batched_mcts(
                    mcts_games, evaluator,
                    n_simulations=n_simulations, add_root_noise=True,
                )
            for slot_idx, g_idx in enumerate(model_turn):
                g = mcts_games[slot_idx]
                tau = 1.0 if ply < temperature_moves else temperature_final
                pi = policy_from_visits(
                    g.root, tau,
                    forced_playout_k=forced_playout_k,
                    c_puct_init=c_puct, c_puct_base=c_puct_base,
                )
                trajectories[g_idx].append((g.root.state.to_planes(), pi.copy(), n_initial + ply))
                action = _sample_action(pi, rng)
                g.advance_root(action)

        # Opponent: just call picker per game.
        for g_idx in opp_turn:
            g = games[g_idx]
            action = int(opponent_picker(g.root.state, rng))
            g.advance_root(action)

        # Terminal check (same for both subsets).
        next_active: list[int] = []
        for g_idx in active:
            g = games[g_idx]
            done, term_val = g.root.state.is_terminal()
            if done:
                # The player who just moved is `side_to_move`. If term_val == -1
                # at the new root (whose side is the player who DIDN'T move),
                # the mover just won.
                if term_val == -1.0:
                    winner_side = side_to_move
                    outcome_for_model = 1.0 if winner_side == int(model_side[g_idx]) else -1.0
                else:
                    outcome_for_model = 0.0
                completed.append((g_idx, outcome_for_model, n_initial + ply + 1))
            else:
                next_active.append(g_idx)
        active = next_active
        ply += 1

    # Max-plies fallthrough = draw.
    for g_idx in active:
        completed.append((g_idx, 0.0, n_initial + ply))

    records: list[GameRecord] = []
    for g_idx, outcome_for_model, plies in sorted(completed):
        examples: list[SelfPlayExample] = []
        side = int(model_side[g_idx])
        traj = trajectories[g_idx]
        # First pass: offensive teacher + collect defense fires (so the #42
        # per-game fraction cap can be applied before any example is emitted).
        finalized: list[tuple[np.ndarray, np.ndarray, float, int]] = []
        defense_fires: list[tuple[int, float]] = []
        for ply_idx, (planes, pi, ply_at_capture) in enumerate(traj):
            # planes are canonical (plane 0 = side-to-move = model at the moment
            # the example was recorded), so z is directly outcome_for_model.
            z = outcome_for_model
            vcf_fired = False
            if vct_teacher:
                # VCT is a strict superset of VCF; the cell uses it INSTEAD of
                # --vcf-teacher, so the deeper solver replaces the shallower.
                pi, z, vcf_fired = _apply_vct_teacher(planes, pi, z, side=side)
            elif vcf_teacher:
                pi, z, vcf_fired = _apply_vcf_teacher(planes, pi, z, side=side)
            if defense_teacher:
                d_z, fired = _apply_defense_teacher(
                    planes, z, vcf_already_fired=vcf_fired)
                if fired:
                    defense_fires.append((ply_idx, d_z))
            finalized.append((planes, pi, z, ply_at_capture))
        kept = _relabel_defense_game(defense_fires, len(traj)) if defense_fires else {}
        for ply_idx, (planes, pi, z, ply_at_capture) in enumerate(finalized):
            if ply_idx in kept:
                z = kept[ply_idx]
            if augment_symmetries:
                for aug_planes, aug_pi in augment(planes, pi):
                    examples.append(SelfPlayExample(
                        aug_planes, aug_pi.astype(np.float32), z,
                        side=side, ply=int(ply_at_capture),
                    ))
            else:
                examples.append(SelfPlayExample(
                    planes, pi, z,
                    side=side, ply=int(ply_at_capture),
                ))
        records.append(GameRecord(examples=examples, plies=plies, outcome=outcome_for_model))

    return records
