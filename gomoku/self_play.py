"""Parallel self-play game generation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import MutableMapping

import numpy as np

from gomoku.game import (
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


# Per-process VCT solver budget for the teacher (Derby 'x-vct' lever, bead
# derby-rxf). VCT (Victory-by-Continuous-Threes) is a strict SUPERSET of VCF:
# it proves every VCF forced win plus wins that need forcing threes. Its tree
# fans out on the defender side, so its caps default LOWER than VCF's (see
# vcf.DEFAULT_VCT_MAX_* — depth 7, nodes 20k) and are the hard safety valve
# that keeps a single gen-hot-path solve cheap. We deliberately do NOT uncap.
_VCT_MAX_DEPTH = vcf.DEFAULT_VCT_MAX_DEPTH
_VCT_MAX_NODES = vcf.DEFAULT_VCT_MAX_NODES


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


# Value-discount (Derby v6 'mate-discounted-value'): scale ordinary outcome value
# targets by gamma^(plies_to_end) so positions near a decisive end get crisp ±1 and
# far-from-end positions get hedged targets — generalizing the VCF mate-distance
# discount to ALL outcomes. Per-process (set once by the worker). 1.0 = OFF (flat
# ±1, byte-identical to pre-v6). Applied to the base z BEFORE the VCF teacher, so a
# proven forced win still overwrites with its own (steeper) mate-discounted value.
_VALUE_DISCOUNT = 1.0


def configure_value_discount(gamma: float | None = None) -> None:
    """Set the process-wide value-target discount (gamma in (0,1]); 1.0 = flat
    outcomes (current behavior). Call once before generation."""
    global _VALUE_DISCOUNT
    if gamma is not None:
        _VALUE_DISCOUNT = float(gamma)


def _discount_z(z: float, plies_to_end: int) -> float:
    """Scale an outcome value target by gamma^plies_to_end. No-op when gamma>=1.0 or
    z==0 (draws stay 0). plies_to_end = recorded-trajectory steps to game end."""
    if _VALUE_DISCOUNT >= 1.0 or z == 0.0:
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

    Caps: the VCT search fans out on the defender side, so it respects the
    (conservative) per-process VCT budget (``vcf.DEFAULT_VCT_MAX_DEPTH`` / nodes);
    we do NOT uncap it on the gen hot path.
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
    value target ``z = -1.0`` (teaches "you should have defended earlier").

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
    # recorded position is lost. Value target = -1.0 (loss vertex). The recorded
    # z is from the recorded side's perspective, matching the side-to-move of the
    # un-swapped position, so a proven OPPONENT win maps directly to z = -1.0.
    _profile_add(profile, "defense_fired", 1.0)
    return -1.0, True


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


@dataclass
class GameRecord:
    """A completed game's training examples plus some metadata."""

    examples: list[SelfPlayExample]
    plies: int
    outcome: float  # +1 if first-mover (black) won, -1 if second-mover (white) won, 0 draw
    archive_start: bool = False  # WL5: True if game was seeded from validation archive


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
) -> list[SelfPlayExample]:
    """Expand one recorded position into training example(s).

    With D4 augmentation, emits 8 symmetry variants; the per-cell aux targets
    (opponent-reply `aux_pi` and/or `ownership`, when present) are transformed
    by the SAME symmetry as planes+pi via augment_with_cell_targets, so each
    label stays board-aligned with its position. Without augmentation, emits the
    single position. A None target propagates to the example unchanged → masked
    out of that target's loss.

    Byte-identical guarantee: when BOTH aux_pi and ownership are None, this uses
    the plain `augment` (the pre-aux path), so the off-case output is unchanged.
    """
    out: list[SelfPlayExample] = []
    if augment_symmetries:
        if aux_pi is None and ownership is None:
            with _profile_timer(profile, "d4_augment_s"):
                augmented = list(augment(planes, pi))
            for aug_planes, aug_pi in augmented:
                with _profile_timer(profile, "example_create_s"):
                    out.append(SelfPlayExample(
                        aug_planes, aug_pi.astype(np.float32), z,
                        side=int(side), ply=int(ply_at_capture),
                        aux_pi=None, ownership=None,
                    ))
        else:
            # One or both per-cell targets present: carry them through the
            # identical D4 symmetry as planes+pi. Slot order is fixed
            # [aux_pi, ownership]; a missing target keeps its slot as None so
            # the example field stays None (masked out of that loss).
            cell_targets = []
            aux_slot = ownership_slot = None
            if aux_pi is not None:
                aux_slot = len(cell_targets)
                cell_targets.append(aux_pi)
            if ownership is not None:
                ownership_slot = len(cell_targets)
                cell_targets.append(ownership)
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
                    ))
    else:
        with _profile_timer(profile, "example_create_s"):
            out.append(SelfPlayExample(
                planes, pi.astype(np.float32), z,
                side=int(side), ply=int(ply_at_capture),
                aux_pi=(aux_pi.astype(np.float32) if aux_pi is not None else None),
                ownership=(ownership.astype(np.float32) if ownership is not None else None),
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
    n_archive = 0 if archive is None else int(archive["planes"].shape[0])
    with _profile_timer(profile, "game_setup_s"):
        for _ in range(n_games):
            from_archive = (
                archive is not None
                and n_archive > 0
                and float(rng.random()) < archive_start_frac
            )
            if from_archive:
                idx = int(rng.integers(0, n_archive))
                start_state = _gamestate_from_archive(archive, idx)
                opening_plies = start_state.move_count
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
            for ply_idx, (planes, pi, side) in enumerate(traj):
                z = outcome_for_black if side == 0 else -outcome_for_black
                z = _discount_z(z, len(traj) - 1 - ply_idx)
                ply_at_capture = n_initial + ply_idx
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
                    z, _ = _apply_defense_teacher(
                        planes, z, vcf_already_fired=vcf_fired, profile=profile)
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
    n_archive = 0 if archive is None else int(archive["planes"].shape[0])
    for _ in range(n_games):
        from_archive = (
            archive is not None and n_archive > 0
            and float(rng.random()) < archive_start_frac
        )
        if from_archive:
            idx = int(rng.integers(0, n_archive))
            start_state = _gamestate_from_archive(archive, idx)
            opening_plies = start_state.move_count
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
        for ply_idx, (planes, pi, side) in enumerate(traj):
            z = outcome_for_black if side == 0 else -outcome_for_black
            z = _discount_z(z, len(traj) - 1 - ply_idx)
            ply_at_capture = n_initial + ply_idx
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
                z, _ = _apply_defense_teacher(
                    planes, z, vcf_already_fired=vcf_fired, profile=profile)
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
    n_archive = 0 if archive is None else int(archive["planes"].shape[0])
    with _profile_timer(profile, "game_setup_s"):
        for _ in range(n_games):
            from_archive = (
                archive is not None
                and n_archive > 0
                and float(rng.random()) < archive_start_frac
            )
            if from_archive:
                idx = int(rng.integers(0, n_archive))
                start_state = _gamestate_from_archive(archive, idx)
                opening_plies = start_state.move_count
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

    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    active: list[int] = list(range(n_games))
    completed: list[tuple[int, float, int]] = []
    final_state: dict[int, tuple[np.ndarray, int]] = {}

    ply = 0
    while active and ply < max_plies:
        with _profile_timer(profile, "active_list_s"):
            active_games = [games[i] for i in active]

        # Playout-Cap Randomization: decide per-game-per-ply whether this is a
        # full-search (recorded) move or a fast (non-recorded) move. The native
        # search_batch requires a single sim count per call, so when PCR is
        # active we split the active wave into two homogeneous sub-batches and
        # call search_batch once per sub-batch with its own sim count. The
        # `is_full` flag (indexed by slot) gates the training-target append
        # below; game advancement happens identically for both kinds of move.
        if pcr_active:
            is_full = rng.random(len(active_games)) < playout_cap_frac
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
            with _profile_timer(profile, "native_search_batch_s"):
                native_mcts.search_batch(
                    active_games,
                    search_evaluator,
                    n_simulations=n_simulations,
                    wave_size=wave_size,
                    add_root_noise=True,
                )
            _profile_add(profile, "search_calls", 1.0)

        next_active: list[int] = []
        with _profile_timer(profile, "post_search_loop_s"):
            for slot_idx, g_idx in enumerate(active):
                g = active_games[slot_idx]
                record_target = is_full is None or bool(is_full[slot_idx])
                tau = 1.0 if ply < temperature_moves else temperature_final
                with _profile_timer(profile, "policy_export_s"):
                    pi = g.policy(temperature=tau)
                n_initial = initial_plies[g_idx]
                side = (n_initial + ply) % 2
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
                        planes = g.root_planes()
                    with _profile_timer(profile, "trajectory_append_s"):
                        trajectories[g_idx].append((planes, pi.copy(), side))

                with _profile_timer(profile, "sample_action_s"):
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
                        # (plane 0) is the non-mover, abs parity (n_initial+ply+1)%2.
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
            final_state[g_idx] = (None, 0)  # max-plies draw -> ownership MASKED OUT (None), not scored
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
            for ply_idx, (planes, pi, side) in enumerate(traj):
                z = outcome_for_black if side == 0 else -outcome_for_black
                z = _discount_z(z, len(traj) - 1 - ply_idx)
                ply_at_capture = n_initial + ply_idx
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
                    z, _ = _apply_defense_teacher(
                        planes, z, vcf_already_fired=vcf_fired, profile=profile)
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
            ))

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
) -> list[GameRecord]:
    """Generate `n_games` self-play games in parallel.

    All games advance in lockstep: at each ply we batch-MCTS across active games,
    sample an action per game, apply it, and remove any games that ended.

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
    if gumbel_root:
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
        )
    if max_plies is None:
        max_plies = N_ACTIONS  # full-board fallback (game can't have more than this)

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

    # Per-game trajectory of (planes, pi, side_to_move_at_that_ply)
    # side_to_move is encoded as 0 for the player who moved first ("black"), 1 for the other.
    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    active: list[int] = list(range(n_games))
    completed: list[tuple[int, float, int]] = []  # (game_idx, outcome_for_black, plies)
    final_state: dict[int, tuple[np.ndarray, int]] = {}

    ply = 0
    while active and ply < max_plies:
        active_games = [games[i] for i in active]
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
        for ply_idx, (planes, pi, side) in enumerate(traj):
            z = outcome_for_black if side == 0 else -outcome_for_black
            z = _discount_z(z, len(traj) - 1 - ply_idx)
            ply_at_capture = n_initial + ply_idx
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
                z, _ = _apply_defense_teacher(
                    planes, z, vcf_already_fired=vcf_fired, profile=profile)
            aux_pi = _aux_target_for(traj, ply_idx, side) if record_aux else None
            examples.extend(_build_examples(
                planes, pi, z, side, ply_at_capture, aux_pi,
                augment_symmetries, None, ownership=ownership,
            ))
        records.append(GameRecord(examples=examples, plies=plies, outcome=outcome_for_black))

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
        for planes, pi, ply_at_capture in trajectories[g_idx]:
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
                z, _ = _apply_defense_teacher(
                    planes, z, vcf_already_fired=vcf_fired)
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
