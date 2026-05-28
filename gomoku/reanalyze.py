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
) -> np.ndarray:
    """Pick which buffer rows to re-MCTS this cycle.

    Bounded by min(fraction * buffer_size, max_positions). Returns a
    np.int64 array of row indices (unique, no replacement).
    """
    if buffer_size <= 0:
        return np.empty((0,), dtype=np.int64)
    fraction = max(0.0, min(1.0, float(fraction)))
    target = int(round(fraction * buffer_size))
    target = min(target, int(max_positions), int(buffer_size))
    if target <= 0:
        return np.empty((0,), dtype=np.int64)
    # Sample WITHOUT replacement so we don't re-MCTS the same row twice
    # within a single cycle (wasted cost; same row, same net = same answer).
    return rng.choice(buffer_size, size=target, replace=False).astype(np.int64)


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

    idx = _select_indices(int(buffer.size), fraction, max_positions, rng)
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
    )
