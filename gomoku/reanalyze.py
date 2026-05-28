"""Reanalyze engine — re-MCTS old buffer positions with the CURRENT net.

Subtask 1/3 of the reanalyze epic (bead derby-fm9; parent derby-3vs).
This module is the ENGINE only — it does NOT decide cadence or register a
derby cell (those are subtasks 2/3 and 3/3). It exposes a single function
`reanalyze_cycle()` that the trainer calls with the live `ReplayBuffer`,
the current model, and a small bounded sample fraction; the function
samples a subset of stored positions, re-runs MCTS with the current net,
and **overwrites the stored policy targets in place**.

DESIGN INVARIANTS:

* **Default OFF byte-identical.** The flag-gated import sites in `train.py`
  never call this module unless `--reanalyze` is set, and this module
  never mutates `buffer` rows it did not sample. With the flag off, the
  buffer's `(planes, pi, z, weight_version, side, ply, …)` columns are
  bitwise unchanged.

* **Compose, don't replace.** The engine produces an MCTS-visit-derived
  policy `pi'` for each sampled position and writes it into `buffer.pi`.
  It does **not** rewrite the target-build math (value-discount, vcf
  teacher, draw-value reshape) — those live in `self_play.py` and apply
  to fresh self-play examples on their way INTO the buffer. The point of
  reanalyze is that the stored target row is what the SGD path samples,
  so overwriting `buffer.pi[i]` (and optionally `buffer.z[i]`) is
  sufficient. (Value relabel is gated separately — see `relabel_value`
  in `reanalyze_cycle`.)

* **Bound the cost.** A single cycle re-runs MCTS on at most
  `min(int(fraction * buffer.size), max_positions)` rows, in fixed-size
  micro-batches (`mcts_batch`). Per-call sim count is configurable
  (default deliberately conservative).

* **Trainer-owned single-writer.** Workers keep dropping game files
  unchanged; the trainer's reanalyze pass is the single mutator of buffer
  target columns. This module assumes the caller holds the buffer.

* **Reconstruct-from-planes (no history fidelity).** The buffer stores
  the per-position INPUT planes (current side + opponent + history); we
  reconstruct a `GameState` from the current-frame planes only (history
  is dropped). The MCTS still explores forward from the same root board;
  the net's evaluation of the root and of explored leaves will use
  HISTORY_PLY=0 planes — a documented approximation. This keeps the
  engine pure-Python and avoids requiring extra columns in the ring
  buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from gomoku.game import BOARD_SIZE, HISTORY_PLY, N_ACTIONS, GameState
from gomoku.mcts import (
    MCTSGame,
    make_torch_evaluator,
    policy_from_visits,
    run_batched_mcts,
)


# ---------- defaults (bounded-cost; conservative) -------------------------

# Per-cycle fraction of the buffer to re-MCTS. 5–10% is the spec range.
DEFAULT_FRACTION: float = 0.05

# Hard cap on positions per cycle, regardless of fraction. Stops the engine
# itself running away if the buffer grows or the fraction is mis-set.
DEFAULT_MAX_POSITIONS: int = 1024

# MCTS simulation budget per position. Conservative — the engine's value is
# *deeper* search than fresh self-play (which uses ~100 sims), but we keep
# the engine's own cost capped. The runner can tune this for the cell.
DEFAULT_SIMS: int = 200

# Positions per MCTS batch call. Independent of fraction/max — controls the
# evaluator's per-call batch size (cross-game leaf batching is in mcts.py).
DEFAULT_MCTS_BATCH: int = 32

# Optional value relabel: if True, also overwrite buffer.z with the MCTS
# root value. Default OFF — relabeling z is more aggressive and the spec's
# core lever is the policy target. The scheduler subtask (2/3) decides
# when/whether to enable it.
DEFAULT_RELABEL_VALUE: bool = False


# -------------------------------------------------------------------------


@dataclass
class ReanalyzeMetrics:
    """Per-cycle bookkeeping returned to the trainer for logging."""

    sampled_n: int          # rows actually re-MCTS'd this cycle
    skipped_terminal: int   # rows whose reconstructed root was terminal (no MCTS)
    mcts_batches: int       # MCTS batch calls executed
    sims_per_pos: int       # the sim cap used
    fraction: float         # the effective fraction applied
    relabel_value: bool     # whether z was overwritten
    # Row indices the engine ACTUALLY wrote to this cycle (post-terminal-skip).
    # The scheduler reads this to feed `record_cycle()` (the cooldown's
    # post-cycle update). None when nothing was written. Not logged to wandb
    # — log keys above are the scalar surface; this is the structural hand-off.
    written_rows: Optional[np.ndarray] = None


# -------------------------------------------------------------------------


def _state_from_planes(planes: np.ndarray) -> GameState:
    """Reconstruct a minimal `GameState` from the buffer's stored input planes.

    Buffer plane layout (gomoku/game.py): plane[0]=side-to-move's stones,
    plane[HISTORY_PLY]=opponent's stones, plus history planes we deliberately
    drop. The reconstructed state's `move_count` is recovered from the stone
    count (canonical: move_count = total stones on board), and the perspective
    is implicit in `board[0]` already being side-to-move's stones.
    """
    if planes.ndim != 3 or planes.shape[1] != BOARD_SIZE or planes.shape[2] != BOARD_SIZE:
        raise ValueError(f"reanalyze: planes shape {planes.shape} not (C,N,N)")
    me = planes[0].astype(bool)
    opp = planes[HISTORY_PLY].astype(bool)
    board = np.stack([me, opp], axis=0)
    move_count = int(me.sum() + opp.sum())
    return GameState(board=board, move_count=move_count, history=())


def _select_indices(
    buffer_size: int,
    fraction: float,
    max_positions: int,
    rng: np.random.Generator,
    eligibility_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Pick which buffer rows to re-MCTS this cycle.

    Bounded by min(fraction * buffer_size, max_positions). Returns a
    np.int64 array of row indices (unique, no replacement).

    If `eligibility_filter` is given, it is called once with the array of
    ALL candidate row indices (``arange(buffer_size)``) and must return the
    subset of indices that are currently eligible (the scheduler's
    feedback-loop guard lives there). Sampling then draws from that subset.
    With no filter, every row is eligible (the original behavior).

    Filter contract: pure function, returns a subset of the input by value
    (np.int64); no in-place mutation of the input. The engine never mutates
    the scheduler's state — `record_cycle()` is the only mutator, called
    after the engine returns.
    """
    if buffer_size <= 0:
        return np.empty((0,), dtype=np.int64)
    fraction = max(0.0, min(1.0, float(fraction)))
    target = int(round(fraction * buffer_size))
    target = min(target, int(max_positions), int(buffer_size))
    if target <= 0:
        return np.empty((0,), dtype=np.int64)
    candidates = np.arange(buffer_size, dtype=np.int64)
    if eligibility_filter is not None:
        candidates = np.asarray(eligibility_filter(candidates), dtype=np.int64)
        if candidates.size == 0:
            return np.empty((0,), dtype=np.int64)
    # `target` is the engine's cost cap; the filter may have shrunk the pool
    # below it, in which case we draw the whole eligible pool (target capped
    # by available unique-without-replacement count).
    target = min(target, int(candidates.size))
    # Sample WITHOUT replacement so we don't re-MCTS the same row twice
    # within a single cycle (wasted cost; same row, same net = same answer).
    chosen = rng.choice(candidates, size=target, replace=False)
    return np.asarray(chosen, dtype=np.int64)


def reanalyze_cycle(
    buffer,
    model,
    device,
    *,
    fraction: float = DEFAULT_FRACTION,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    sims: int = DEFAULT_SIMS,
    mcts_batch: int = DEFAULT_MCTS_BATCH,
    relabel_value: bool = DEFAULT_RELABEL_VALUE,
    c_puct: float = 1.25,
    c_puct_base: float = 19652.0,
    rng: np.random.Generator | None = None,
    eligibility_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> ReanalyzeMetrics:
    """Sample buffer positions, re-MCTS with `model`, overwrite their targets.

    Single-writer mutator of `buffer.pi` (and optionally `buffer.z`) on the
    sampled rows. All other buffer columns (`weight_version`, `side`, `ply`,
    aux, ownership, pos_key) are LEFT UNCHANGED — the engine refreshes the
    policy/value teacher signal, not the position identity.

    Returns ReanalyzeMetrics for logging; the caller decides whether/how to
    surface them to wandb.
    """
    if rng is None:
        rng = np.random.default_rng()

    idx = _select_indices(
        int(buffer.size), fraction, max_positions, rng,
        eligibility_filter=eligibility_filter,
    )
    if idx.size == 0:
        return ReanalyzeMetrics(
            sampled_n=0, skipped_terminal=0, mcts_batches=0,
            sims_per_pos=int(sims), fraction=float(fraction),
            relabel_value=bool(relabel_value),
        )

    # Pull the sampled planes off the buffer ONCE (one copy to CPU/numpy).
    # The buffer may live on MPS; the GameState reconstruction is pure-Python
    # numpy. The MCTS evaluator (`make_torch_evaluator`) re-derives planes
    # from the GameState and ships them back to `device` for the model.
    sampled_planes = buffer.planes[idx].detach().cpu().numpy()

    # Reconstruct minimal GameStates per row. Terminal positions are skipped
    # (cannot meaningfully re-MCTS; their target is undefined under the
    # search-from-this-state convention).
    games: list[MCTSGame] = []
    keep: list[int] = []   # indices into `idx` that we'll write back to
    skipped_terminal = 0
    for j, planes_row in enumerate(sampled_planes):
        state = _state_from_planes(planes_row)
        done, _ = state.is_terminal()
        if done:
            skipped_terminal += 1
            continue
        # Per-position RNG seed so the engine's exploration is reproducible
        # across runs but distinct across positions. Dirichlet noise is OFF
        # in our root expansion below — reanalyze wants the *sharper* search,
        # not exploration noise. (run_batched_mcts adds root noise only if
        # add_root_noise=True; we pass False.)
        per_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        g = MCTSGame(state, c_puct=c_puct, c_puct_base=c_puct_base, rng=per_rng)
        games.append(g)
        keep.append(j)

    if not games:
        return ReanalyzeMetrics(
            sampled_n=0, skipped_terminal=skipped_terminal, mcts_batches=0,
            sims_per_pos=int(sims), fraction=float(fraction),
            relabel_value=bool(relabel_value),
        )

    # Build one evaluator backed by the current model. `make_torch_evaluator`
    # handles eval-mode / no-grad / device transfer — the trainer's model
    # may be in train() mode when we get here; the evaluator saves+restores it.
    evaluator = make_torch_evaluator(model, device)

    mcts_batches = 0
    batch = max(1, int(mcts_batch))
    # Re-MCTS in fixed-size chunks. Each chunk shares one evaluator pool
    # via run_batched_mcts (leaf-batched). add_root_noise=False so reanalyze
    # produces a sharper target than fresh self-play would.
    for start in range(0, len(games), batch):
        chunk = games[start:start + batch]
        run_batched_mcts(
            chunk, evaluator, n_simulations=int(sims), add_root_noise=False,
        )
        mcts_batches += 1

    # Materialize new (pi, z) targets and write them back to the buffer's
    # sampled rows in place. We use temperature=1 (proportional-to-visits)
    # which matches the AlphaZero pi target for non-greedy positions; the
    # spec is "overwrite policy target" and the buffer's stored target is
    # already a normalized distribution (not a greedy one-hot).
    new_pi = np.zeros((len(games), N_ACTIONS), dtype=np.float32)
    new_z = np.zeros((len(games),), dtype=np.float32)
    for k, g in enumerate(games):
        new_pi[k] = policy_from_visits(g.root, temperature=1.0)
        if relabel_value:
            n = float(g.root.N.sum())
            new_z[k] = float(g.root.W.sum() / n) if n > 0 else 0.0

    write_rows = idx[np.array(keep, dtype=np.int64)]
    write_idx_t = torch.as_tensor(write_rows, dtype=torch.long, device=buffer.pi.device)
    new_pi_t = torch.from_numpy(new_pi).to(buffer.pi.device, dtype=buffer.pi.dtype)
    buffer.pi.index_copy_(0, write_idx_t, new_pi_t)
    if relabel_value:
        new_z_t = torch.from_numpy(new_z).to(buffer.z.device, dtype=buffer.z.dtype)
        buffer.z.index_copy_(0, write_idx_t, new_z_t)

    return ReanalyzeMetrics(
        sampled_n=int(write_rows.size),
        skipped_terminal=skipped_terminal,
        mcts_batches=mcts_batches,
        sims_per_pos=int(sims),
        fraction=float(fraction),
        relabel_value=bool(relabel_value),
        written_rows=np.asarray(write_rows, dtype=np.int64),
    )


# =========================================================================
# Scheduler — subtask 2/3 (bead derby-1nt, parent derby-3vs)
# =========================================================================
#
# The ENGINE above decides *what happens* in a reanalyze cycle (sample,
# re-MCTS, write). The SCHEDULER below decides:
#
#   1. *When* a cycle fires (cadence).
#   2. *Which* rows are eligible this cycle (the feedback-loop guard).
#
# Per-cycle fraction stays the engine's `--reanalyze-fraction` knob — this
# class only decides cadence + cooldown. Design constraints (spec):
#
#   * Default OFF byte-identical — the master `--reanalyze` flag in train.py
#     remains the kill-switch; the scheduler is INSTANTIATED only inside
#     the gated branch. Aux-head discipline: when off, this class isn't
#     constructed and `record_cycle`/`should_fire` are never called.
#
#   * Compose with the engine — the scheduler doesn't re-implement
#     sampling; it passes its eligibility filter into `reanalyze_cycle`
#     via the `eligibility_filter` parameter wired above.
#
#   * Feedback-loop guard — over-reanalyzing recently-visited rows lets
#     the net certify its own current biases (the spec's flagged risk).
#     Guard mechanism: for each buffer row we remember the cycle index
#     and the buffer's `weight_version[row]` AT the moment we wrote it.
#     A row is INELIGIBLE this cycle iff:
#       (a) we reanalyzed it before, AND
#       (b) fewer than `cooldown_cycles` cycles have elapsed since, AND
#       (c) the buffer hasn't overwritten that row with new self-play data
#           (i.e. `buffer.weight_version[row] == stored_weight_version`).
#     Condition (c) catches the ring-buffer overwrite case: if a new
#     self-play position has been written into the slot, the cooldown is
#     stale (it's a NEW position) and the row is eligible again — without
#     this, the guard would underflow the eligible pool as the buffer
#     cycles. Optional secondary cooldown:
#     `cooldown_positions > 0` requires
#     `cumulative_new_positions - stored_cumulative >= cooldown_positions`.
#
# Cadence — two independent triggers (OR-combined; either fires the cycle):
#
#   * `every_epochs`: fire every N epochs (default 1 = every epoch, the
#     1/3 engine's pre-scheduler behavior — backward-compat default).
#   * `every_positions`: fire after K new positions have been ingested
#     since the last fire (0 = disabled; default 0).
#
# Note: with both at their defaults (every_epochs=1, every_positions=0),
# `should_fire` returns True every epoch — preserving the engine 1/3
# behavior. A user explicitly sets `every_epochs>1` to space cycles out,
# or `every_positions>0` for an ingest-driven cadence.

# Conservative defaults: fire every epoch (preserve 1/3 behavior), with
# a 3-cycle cooldown that lets rows breathe before another sharpening pass.
DEFAULT_EVERY_EPOCHS: int = 1
DEFAULT_EVERY_POSITIONS: int = 0   # OFF; opt-in for ingest-driven cadence
DEFAULT_COOLDOWN_CYCLES: int = 3
DEFAULT_COOLDOWN_POSITIONS: int = 0   # OFF; opt-in for ingest-driven cooldown


class ReanalyzeScheduler:
    """Cadence + per-row cooldown for the reanalyze engine.

    Construct ONCE per training run, AFTER the buffer is allocated (so the
    per-row cooldown tables can size to `buffer.capacity`). Then for each
    epoch:

        if scheduler.should_fire(epoch, cumulative_new_positions):
            metrics = reanalyze_cycle(
                ..., eligibility_filter=scheduler.eligibility_filter(buffer),
            )
            scheduler.record_cycle(metrics.written_rows, buffer,
                                   cumulative_new_positions)

    The scheduler is OFF-path silent: if the trainer never builds an
    instance (because `--reanalyze` is off), the class is never imported
    and the buffer/state-dict are bitwise unchanged.
    """

    def __init__(
        self,
        capacity: int,
        *,
        every_epochs: int = DEFAULT_EVERY_EPOCHS,
        every_positions: int = DEFAULT_EVERY_POSITIONS,
        cooldown_cycles: int = DEFAULT_COOLDOWN_CYCLES,
        cooldown_positions: int = DEFAULT_COOLDOWN_POSITIONS,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"ReanalyzeScheduler: capacity must be >0 (got {capacity})")
        if every_epochs < 1:
            raise ValueError(f"every_epochs must be >=1 (got {every_epochs})")
        if every_positions < 0:
            raise ValueError(f"every_positions must be >=0 (got {every_positions})")
        if cooldown_cycles < 0:
            raise ValueError(f"cooldown_cycles must be >=0 (got {cooldown_cycles})")
        if cooldown_positions < 0:
            raise ValueError(f"cooldown_positions must be >=0 (got {cooldown_positions})")
        self.capacity = int(capacity)
        self.every_epochs = int(every_epochs)
        self.every_positions = int(every_positions)
        self.cooldown_cycles = int(cooldown_cycles)
        self.cooldown_positions = int(cooldown_positions)
        # Per-row cooldown state. -1 sentinel = never reanalyzed.
        # int32 is plenty for cycle counts (a long run < 2**31 epochs).
        self._row_last_cycle = np.full((self.capacity,), -1, dtype=np.int32)
        # The buffer's weight_version when we last reanalyzed this row.
        # int64 to match buffer.weight_version dtype.
        self._row_last_weight_version = np.zeros((self.capacity,), dtype=np.int64)
        # Cumulative_new_positions at the moment we last reanalyzed this row.
        self._row_last_cum_positions = np.zeros((self.capacity,), dtype=np.int64)
        # Scheduler-internal cycle counter (incremented by record_cycle).
        self._cycle = 0
        # Last firing's bookkeeping (for the cadence check).
        self._last_fire_epoch: int | None = None
        self._last_fire_cum_positions: int | None = None

    # ---------- cadence ----------

    def should_fire(self, epoch: int, cumulative_new_positions: int) -> bool:
        """Return True iff a reanalyze cycle should run at this epoch.

        Triggers (OR-combined):
          * every_epochs trigger: epoch >= last_fire_epoch + every_epochs
            (or never fired yet AND epoch is a multiple of every_epochs).
          * every_positions trigger (if every_positions > 0):
            cumulative_new_positions >= last_fire_cum_positions + every_positions
            (or never fired yet AND cumulative_new_positions >= every_positions).

        With every_epochs=1 + every_positions=0 (defaults), this fires every
        epoch — preserving the 1/3 engine's behavior unchanged.
        """
        epoch = int(epoch)
        cum = int(cumulative_new_positions)
        # Epoch trigger
        if self._last_fire_epoch is None:
            epoch_fires = (epoch % self.every_epochs == 0)
        else:
            epoch_fires = (epoch - self._last_fire_epoch >= self.every_epochs)
        # Positions trigger (opt-in)
        positions_fires = False
        if self.every_positions > 0:
            if self._last_fire_cum_positions is None:
                positions_fires = (cum >= self.every_positions)
            else:
                positions_fires = (cum - self._last_fire_cum_positions >= self.every_positions)
        return epoch_fires or positions_fires

    # ---------- feedback-loop guard ----------

    def eligibility_filter(self, buffer) -> Callable[[np.ndarray], np.ndarray]:
        """Return the function the engine should call to filter candidates.

        Captures the buffer's CURRENT weight_version column (so the engine
        can compare per-row against the version stored at last reanalysis).
        The returned closure is pure (no scheduler-state mutation).

        Eligibility rule per row (see class docstring): row is eligible iff
          * we never reanalyzed it before, OR
          * cooldown_cycles have elapsed since we did, OR
          * the buffer's row has been overwritten with new self-play data
            since we reanalyzed it (weight_version mismatch), AND
          * (if cooldown_positions > 0) enough new positions have been
            ingested since.
        """
        # Snapshot buffer.weight_version once; the cycle is single-threaded.
        # buffer.weight_version is sized to buffer.capacity; we only consult
        # rows in [0, buffer.size) (matches the candidate set the engine
        # passes in).
        wver_now = buffer.weight_version.detach().cpu().numpy().astype(np.int64)
        last_cycle = self._row_last_cycle
        last_wver = self._row_last_weight_version
        last_cum = self._row_last_cum_positions
        cooldown_cycles = self.cooldown_cycles
        cooldown_positions = self.cooldown_positions
        # The cycle index the NEXT fire will record (so on the FIRST fire,
        # comparing to last_cycle=-1 yields elapsed = current_cycle+1 >=
        # cooldown_cycles for any cooldown >= 0). _cycle is incremented by
        # record_cycle AFTER the cycle.
        current_cycle = self._cycle
        last_fire_cum_positions = self._last_fire_cum_positions

        def _filter(candidates: np.ndarray) -> np.ndarray:
            cand = np.asarray(candidates, dtype=np.int64)
            if cand.size == 0:
                return cand
            never = (last_cycle[cand] < 0)
            # Cooldown elapsed in cycles.
            cycle_elapsed = (current_cycle - last_cycle[cand]) >= cooldown_cycles
            # Buffer overwrote the row with new self-play data since we
            # reanalyzed (weight_version mismatch). Cheap O(N) compare.
            row_replaced = (wver_now[cand] != last_wver[cand])
            eligible = never | cycle_elapsed | row_replaced
            # Optional ingest-cooldown (only constrains when >0).
            if cooldown_positions > 0 and last_fire_cum_positions is not None:
                pos_elapsed = (
                    (last_fire_cum_positions - last_cum[cand]) >= cooldown_positions
                )
                # A row passes only if BOTH guards admit it (intersection).
                # But `never` should still admit unconditionally (no prior
                # ingest delta to compute).
                eligible = never | (eligible & pos_elapsed)
            return cand[eligible]

        return _filter

    # ---------- post-cycle update ----------

    def record_cycle(
        self,
        written_rows: Optional[np.ndarray],
        buffer,
        cumulative_new_positions: int,
        *,
        epoch: Optional[int] = None,
    ) -> None:
        """Update cooldown bookkeeping after an engine cycle returned.

        `written_rows` is `ReanalyzeMetrics.written_rows` — the rows the
        engine actually wrote (post-terminal-skip). Pass the engine's
        return value through. None or empty => only the cadence trackers
        update; the per-row tables stay put.
        """
        cum = int(cumulative_new_positions)
        # Advance the scheduler's own counter regardless of whether rows
        # were written (cadence trackers reflect that we FIRED, not just
        # that we successfully wrote).
        self._last_fire_cum_positions = cum
        if epoch is not None:
            self._last_fire_epoch = int(epoch)
        # Per-row tables
        if written_rows is not None:
            rows = np.asarray(written_rows, dtype=np.int64)
            if rows.size > 0:
                # Defensive clamp — rows must index into [0, capacity).
                if rows.min() < 0 or rows.max() >= self.capacity:
                    raise ValueError(
                        f"record_cycle: written_rows out of range "
                        f"[0,{self.capacity}); got [{rows.min()},{rows.max()}]"
                    )
                wver_now = buffer.weight_version.detach().cpu().numpy().astype(np.int64)
                self._row_last_cycle[rows] = np.int32(self._cycle)
                self._row_last_weight_version[rows] = wver_now[rows]
                self._row_last_cum_positions[rows] = cum
        # Bump the cycle index for the NEXT fire's cooldown math.
        self._cycle += 1

    # ---------- introspection (handy for logs / tests) ----------

    @property
    def cycle(self) -> int:
        """Cycles that have been recorded (incremented after each
        `record_cycle`)."""
        return int(self._cycle)

    def eligible_count(self, buffer) -> int:
        """How many of buffer.size rows are eligible RIGHT NOW.

        Useful for logging — if this is shrinking toward zero, the
        cooldown is too tight (or the buffer isn't cycling fast enough).
        """
        cand = np.arange(int(buffer.size), dtype=np.int64)
        return int(self.eligibility_filter(buffer)(cand).size)
