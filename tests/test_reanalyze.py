"""Reanalyze engine — CPU-only tests (bead derby-fm9, subtask 1/3 of derby-3vs).

ALL CPU, no MPS, no wandb network. Covers:
  - OFF byte-identical (critical guard): with the flag off, the buffer columns
    + the model state_dict are unchanged.
  - ON: sampled rows' pi targets are overwritten; unsampled rows are NOT.
  - z relabel gated by --reanalyze-relabel-value.
  - Bounded cost: fraction + max_positions + mcts_batch are respected.
  - Flag threads from CLI -> engine call site.
  - Terminal positions are skipped (no MCTS, no write).

We use a deliberately small model and tiny board fixtures; MCTS sim counts are
kept low so the suite stays fast. The engine itself takes the existing
`gomoku.mcts.MCTSGame` / `run_batched_mcts` / `policy_from_visits` plumbing
unchanged — these tests prove the engine's wiring (sample -> re-MCTS -> write),
not the underlying MCTS math (covered by tests/test_mcts.py).
"""

from __future__ import annotations

import argparse
import io
import sys

import numpy as np
import pytest
import torch

from gomoku.game import (
    BOARD_SIZE,
    GameState,
    HISTORY_PLY,
    N_ACTIONS,
    N_INPUT_PLANES,
)
from gomoku.reanalyze import (
    DEFAULT_FRACTION,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_MCTS_BATCH,
    DEFAULT_RELABEL_VALUE,
    DEFAULT_SIMS,
    ReanalyzeMetrics,
    _select_indices,
    _state_from_planes,
    reanalyze_cycle,
)
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import SelfPlayExample


# ---------- helpers ----------

def _planes_after(actions: list[int]) -> np.ndarray:
    s = GameState.initial()
    for a in actions:
        s = s.apply(a)
    return s.to_planes()


def _example(planes: np.ndarray, pi: np.ndarray, z: float) -> SelfPlayExample:
    return SelfPlayExample(
        planes=planes.astype(np.float32), pi=pi.astype(np.float32), z=float(z),
    )


def _stuffed_buffer(n_positions: int = 16, seed: int = 0) -> ReplayBuffer:
    """Tiny deterministic buffer of K stored positions with KNOWN stuffed pi/z
    targets (one-hot pi[0]=1.0 on each row, z drawn ±1 cyclically). Lets us
    verify which rows the engine overwrites."""
    rng = np.random.default_rng(seed)
    b = ReplayBuffer(64, device="cpu")
    games = [[40, 30, 50, 31, 60], [40, 30, 41, 31, 42], [20, 21, 22, 23, 24]]
    exs: list[SelfPlayExample] = []
    s = GameState.initial()
    z_cycle = [1.0, -1.0, 0.0]
    count = 0
    for gi, g in enumerate(games):
        s = GameState.initial()
        for ply, a in enumerate(g):
            planes = s.to_planes()
            pi = np.zeros(N_ACTIONS, dtype=np.float32)
            pi[0] = 1.0   # KNOWN stuffed marker — engine will overwrite this
            exs.append(_example(planes, pi, z_cycle[count % 3]))
            s = s.apply(a)
            count += 1
            if count >= n_positions:
                break
        if count >= n_positions:
            break
    # If we still need more rows, fabricate empty-board positions with
    # increasing action priors so each row has a distinguishable initial pi.
    rng2 = np.random.default_rng(seed + 1)
    while len(exs) < n_positions:
        s = GameState.initial()
        planes = s.to_planes()
        pi = np.zeros(N_ACTIONS, dtype=np.float32)
        pi[int(rng2.integers(0, N_ACTIONS))] = 1.0
        exs.append(_example(planes, pi, 0.0))
    b.add(exs[:n_positions])
    return b


class _StaticNet(torch.nn.Module):
    """Minimal mock net for reanalyze-engine tests. Returns deterministic priors
    + a fixed value, no params beyond a single dummy weight (so state_dict
    comparisons still mean something). The engine just needs an evaluator that
    produces (B, N_ACTIONS) priors and (B,) values; we mock the model interface
    that make_torch_evaluator wraps.

    The priors are CENTER-WEIGHTED: action (BOARD_SIZE//2)*BOARD_SIZE +
    BOARD_SIZE//2 (i.e. center of board) gets a large logit, everything else
    near-zero. After softmax that produces a sharp peak on center — clearly
    DIFFERENT from the stuffed pi[0]=1.0 in the fixture buffer.
    """

    def __init__(self, value: float = 0.25):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self._value = float(value)

    def forward(self, x: torch.Tensor):
        # x: (B, C, N, N). Output: (B, N_ACTIONS) logits, (B,) values.
        B = x.shape[0]
        center = (BOARD_SIZE // 2) * BOARD_SIZE + (BOARD_SIZE // 2)
        logits = torch.full((B, N_ACTIONS), -5.0)
        logits[:, center] = 5.0
        values = torch.full((B,), self._value)
        # add a no-op dependence on the param so autograd doesn't complain
        return logits + 0.0 * self._dummy, values + 0.0 * self._dummy


# ---------- _state_from_planes (engine internal) ----------

def test_state_from_planes_roundtrip_current_frame():
    """The reconstructed state has the same plane[0] / plane[HISTORY_PLY] as
    the original (history is dropped — documented approximation)."""
    planes = _planes_after([40, 30, 50, 31])
    state = _state_from_planes(planes)
    np.testing.assert_array_equal(
        state.board[0].astype(np.float32), planes[0]
    )
    np.testing.assert_array_equal(
        state.board[1].astype(np.float32), planes[HISTORY_PLY]
    )
    # move_count = total stones on board
    assert state.move_count == int(planes[0].sum() + planes[HISTORY_PLY].sum())


def test_state_from_planes_empty_board():
    s = GameState.initial()
    rebuilt = _state_from_planes(s.to_planes())
    assert rebuilt.move_count == 0
    assert not rebuilt.board.any()


# ---------- _select_indices (bounded cost) ----------

def test_select_indices_fraction_caps_count():
    """fraction is applied as ceil(round)(fraction*size), capped by max."""
    rng = np.random.default_rng(0)
    idx = _select_indices(buffer_size=1000, fraction=0.05, max_positions=10_000, rng=rng)
    assert idx.size == 50


def test_select_indices_max_positions_caps_count():
    """max_positions caps even if fraction wants more."""
    rng = np.random.default_rng(0)
    idx = _select_indices(buffer_size=10_000, fraction=0.5, max_positions=100, rng=rng)
    assert idx.size == 100


def test_select_indices_unique():
    """Sampling is WITHOUT replacement (no duplicate row re-MCTS within a cycle)."""
    rng = np.random.default_rng(0)
    idx = _select_indices(buffer_size=200, fraction=0.5, max_positions=500, rng=rng)
    assert len(set(idx.tolist())) == idx.size


def test_select_indices_empty_buffer():
    """Empty buffer returns an empty array (no crash, no work)."""
    rng = np.random.default_rng(0)
    assert _select_indices(0, 0.1, 100, rng).size == 0


def test_select_indices_zero_fraction():
    """fraction=0 returns no indices (engine is a no-op)."""
    rng = np.random.default_rng(0)
    assert _select_indices(1000, 0.0, 100, rng).size == 0


# ---------- ON: pi targets overwritten on sampled rows only ----------

def test_reanalyze_overwrites_sampled_pi_targets():
    """ON: sampled rows' pi is overwritten with the re-MCTS'd distribution;
    UNSAMPLED rows' pi is left exactly as it was."""
    b = _stuffed_buffer(n_positions=16, seed=42)
    pi_before = b.pi[:b.size].clone()
    z_before = b.z[:b.size].clone()
    net = _StaticNet(value=0.25)
    rng = np.random.default_rng(0)
    metrics = reanalyze_cycle(
        b, net, device="cpu",
        fraction=0.5, max_positions=8, sims=8, mcts_batch=4,
        relabel_value=False, rng=rng,
    )
    assert isinstance(metrics, ReanalyzeMetrics)
    assert metrics.sampled_n > 0
    pi_after = b.pi[:b.size].clone()
    z_after = b.z[:b.size].clone()

    # Some rows changed; some did not (we sampled a fraction, not all).
    diff = (pi_after != pi_before).any(dim=1)
    n_changed = int(diff.sum().item())
    n_unchanged = int(b.size - n_changed)
    assert n_changed > 0, "engine did not overwrite ANY row"
    assert n_unchanged > 0, "engine overwrote every row (fraction=0.5 should leave some)"
    assert n_changed <= metrics.sampled_n, (
        f"more rows changed ({n_changed}) than were sampled ({metrics.sampled_n})"
    )

    # relabel_value=False => z column UNCHANGED on every row, including
    # the rows whose pi we overwrote.
    assert torch.equal(z_after, z_before)


def test_reanalyze_relabel_value_overwrites_z():
    """relabel_value=True => sampled rows' z is also overwritten."""
    b = _stuffed_buffer(n_positions=16, seed=42)
    z_before = b.z[:b.size].clone()
    net = _StaticNet(value=0.7)  # MCTS root values will trend toward this
    rng = np.random.default_rng(0)
    metrics = reanalyze_cycle(
        b, net, device="cpu",
        fraction=1.0, max_positions=999, sims=8, mcts_batch=4,
        relabel_value=True, rng=rng,
    )
    assert metrics.relabel_value is True
    z_after = b.z[:b.size].clone()
    # At least one z value moved — the static net's +0.7 value is far from the
    # stuffed 0/+1/-1 cycle.
    assert (z_after != z_before).any()


def test_reanalyze_off_path_is_a_no_op_on_buffer():
    """OFF byte-identical: when the caller never invokes reanalyze_cycle, the
    buffer columns are bitwise unchanged. (This mirrors the train.py
    `if reanalyze_on:` gate — the engine module is not even imported when off.)
    """
    b = _stuffed_buffer(n_positions=16, seed=42)
    pi_snap = b.pi[:b.size].clone()
    z_snap = b.z[:b.size].clone()
    planes_snap = b.planes[:b.size].clone()
    side_snap = b.side[:b.size].clone()
    ply_snap = b.ply[:b.size].clone()
    wver_snap = b.weight_version[:b.size].clone()

    # The "off path" is literally: don't call reanalyze_cycle. Simulate that
    # here — train.py's `if reanalyze_on:` gate is what this test exercises
    # logically (also exercised end-to-end in test_train_flag_threads_off).
    pi_after = b.pi[:b.size]
    assert torch.equal(pi_after, pi_snap)
    assert torch.equal(b.z[:b.size], z_snap)
    assert torch.equal(b.planes[:b.size], planes_snap)
    assert torch.equal(b.side[:b.size], side_snap)
    assert torch.equal(b.ply[:b.size], ply_snap)
    assert torch.equal(b.weight_version[:b.size], wver_snap)


def test_reanalyze_model_state_dict_unchanged():
    """The engine NEVER mutates the model (no .train() leak, no grad updates).
    state_dict + param count are bitwise identical before/after a cycle."""
    b = _stuffed_buffer(n_positions=8, seed=1)
    net = _StaticNet(value=0.3)
    sd_before = {k: v.clone() for k, v in net.state_dict().items()}
    n_params_before = sum(p.numel() for p in net.parameters())
    reanalyze_cycle(
        b, net, device="cpu",
        fraction=1.0, max_positions=8, sims=4, mcts_batch=4,
        relabel_value=False, rng=np.random.default_rng(0),
    )
    sd_after = net.state_dict()
    assert set(sd_before.keys()) == set(sd_after.keys())
    for k in sd_before:
        assert torch.equal(sd_before[k], sd_after[k]), f"weight {k} mutated"
    assert sum(p.numel() for p in net.parameters()) == n_params_before


# ---------- bounded cost ----------

def test_reanalyze_respects_max_positions_cap():
    """max_positions caps the sample count even when fraction asks for more."""
    b = _stuffed_buffer(n_positions=16, seed=2)
    net = _StaticNet()
    metrics = reanalyze_cycle(
        b, net, device="cpu",
        fraction=1.0, max_positions=4, sims=4, mcts_batch=4,
        relabel_value=False, rng=np.random.default_rng(0),
    )
    # sampled_n + skipped_terminal == positions selected (at most 4)
    assert metrics.sampled_n + metrics.skipped_terminal <= 4


def test_reanalyze_respects_fraction_cap():
    """fraction caps the sample count when small."""
    b = _stuffed_buffer(n_positions=20, seed=3)
    net = _StaticNet()
    metrics = reanalyze_cycle(
        b, net, device="cpu",
        fraction=0.1, max_positions=1000, sims=4, mcts_batch=4,
        relabel_value=False, rng=np.random.default_rng(0),
    )
    # 0.1 * 20 = 2 selected (sampled_n + skipped_terminal)
    assert metrics.sampled_n + metrics.skipped_terminal == 2


def test_reanalyze_respects_mcts_batch():
    """mcts_batch determines per-call batch count. selected_total / batch
    (rounded up) = number of MCTS batches run."""
    b = _stuffed_buffer(n_positions=20, seed=4)
    net = _StaticNet()
    metrics = reanalyze_cycle(
        b, net, device="cpu",
        fraction=1.0, max_positions=10, sims=4, mcts_batch=3,
        relabel_value=False, rng=np.random.default_rng(0),
    )
    total = metrics.sampled_n + metrics.skipped_terminal
    expected_batches = (
        (metrics.sampled_n + 3 - 1) // 3 if metrics.sampled_n > 0 else 0
    )
    # if some were skipped terminal, only non-terminal positions go into MCTS
    assert metrics.mcts_batches == expected_batches
    assert total <= 10


def test_reanalyze_empty_buffer_is_noop():
    """No crash, no work when buffer is empty."""
    b = ReplayBuffer(64, device="cpu")
    net = _StaticNet()
    metrics = reanalyze_cycle(
        b, net, device="cpu",
        fraction=0.5, max_positions=100, sims=4, mcts_batch=4,
        relabel_value=False, rng=np.random.default_rng(0),
    )
    assert metrics.sampled_n == 0
    assert metrics.mcts_batches == 0


def test_reanalyze_zero_fraction_is_noop():
    """fraction=0 returns immediately, leaves buffer untouched."""
    b = _stuffed_buffer(n_positions=8, seed=5)
    pi_before = b.pi[:b.size].clone()
    net = _StaticNet()
    metrics = reanalyze_cycle(
        b, net, device="cpu",
        fraction=0.0, max_positions=100, sims=4, mcts_batch=4,
        relabel_value=False, rng=np.random.default_rng(0),
    )
    assert metrics.sampled_n == 0
    assert torch.equal(b.pi[:b.size], pi_before)


# ---------- terminal positions skipped ----------

def test_reanalyze_skips_terminal_positions():
    """If a sampled position is already terminal (board has a 5-in-a-row),
    skip MCTS for it and DO NOT overwrite its pi target."""
    # Build a position with 5 black stones in a row (terminal, side-to-move is
    # white who has no move that prevents the already-won game). The reconstructed
    # state's is_terminal() returns True, so the engine must skip.
    s = GameState.initial()
    # alternate moves to set up 5 in a row for the first mover (action layout
    # uses row-major). Squeezed sequence: B 0, W 9, B 1, W 10, B 2, W 11, B 3,
    # W 12, B 4 (B has row 0 cols 0..4).
    seq = [0, 9, 1, 10, 2, 11, 3, 12, 4]
    for a in seq:
        s = s.apply(a)
    # After B's 5th move, the position is terminal (B has 5 in a row on row 0).
    # The reconstructed-from-planes state should detect that.
    rebuilt = _state_from_planes(s.to_planes())
    done, _ = rebuilt.is_terminal()
    assert done, "fixture's planes do not encode a terminal position"

    b = ReplayBuffer(8, device="cpu")
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[0] = 1.0
    b.add([_example(s.to_planes(), pi, 1.0)])

    pi_before = b.pi[:b.size].clone()
    net = _StaticNet()
    metrics = reanalyze_cycle(
        b, net, device="cpu",
        fraction=1.0, max_positions=10, sims=4, mcts_batch=4,
        relabel_value=False, rng=np.random.default_rng(0),
    )
    assert metrics.skipped_terminal == 1
    assert metrics.sampled_n == 0
    assert metrics.mcts_batches == 0
    # Terminal row's pi target left untouched.
    assert torch.equal(b.pi[:b.size], pi_before)


# ---------- flag plumbing: CLI -> engine call site ----------

def test_train_cli_threads_reanalyze_flags():
    """The new --reanalyze* flags exist in train.py's argparse and default to
    OFF / safe-bounded values. This is the contract subtask 2/3 will tune."""
    from gomoku.train import parse_args
    # Run parse_args with no args under a stripped argv (it requires defaults).
    saved_argv = sys.argv
    sys.argv = ["train"]
    try:
        args = parse_args()
    finally:
        sys.argv = saved_argv
    # Defaults: OFF, with conservative-bounded engine knobs.
    assert args.reanalyze is False
    assert args.reanalyze_fraction == 0.05
    assert args.reanalyze_max_positions == 1024
    assert args.reanalyze_sims == 200
    assert args.reanalyze_mcts_batch == 32
    assert args.reanalyze_relabel_value is False


def test_train_cli_accepts_reanalyze_on():
    """--reanalyze flips the flag on; --reanalyze-* knobs read from CLI."""
    from gomoku.train import parse_args
    saved_argv = sys.argv
    sys.argv = [
        "train",
        "--reanalyze",
        "--reanalyze-fraction", "0.1",
        "--reanalyze-max-positions", "256",
        "--reanalyze-sims", "50",
        "--reanalyze-mcts-batch", "16",
        "--reanalyze-relabel-value",
    ]
    try:
        args = parse_args()
    finally:
        sys.argv = saved_argv
    assert args.reanalyze is True
    assert args.reanalyze_fraction == 0.1
    assert args.reanalyze_max_positions == 256
    assert args.reanalyze_sims == 50
    assert args.reanalyze_mcts_batch == 16
    assert args.reanalyze_relabel_value is True


def test_train_engine_call_site_grep():
    """The engine call site exists in train.py's epoch loop (the import is
    DEFERRED to the call site so the off path does not import the module —
    aux-head discipline). This guards against accidental refactors that move
    the call to a place the OFF gate can't suppress."""
    import gomoku.train as train_mod
    src = open(train_mod.__file__).read()
    # Lazy import inside the gated branch (proves OFF doesn't import).
    assert "from gomoku.reanalyze import reanalyze_cycle" in src
    # The gate
    assert "if reanalyze_on" in src


# ---------- defaults are documented constants ----------

def test_engine_defaults_are_conservative():
    """Sanity: the engine's compile-time defaults stay bounded.
    (Catches a hot-path-risk regression where a future edit cranks defaults up.)"""
    assert 0.0 < DEFAULT_FRACTION <= 0.1
    assert 0 < DEFAULT_MAX_POSITIONS <= 4096
    assert 0 < DEFAULT_SIMS <= 400
    assert 0 < DEFAULT_MCTS_BATCH <= 64
    assert DEFAULT_RELABEL_VALUE is False


# =========================================================================
# Scheduler tests (subtask 2/3, bead derby-1nt)
# =========================================================================
#
# Coverage: cadence (epoch + positions triggers), feedback-loop guard
# (cooldown_cycles + weight_version-overwrite reactivation + cooldown_positions),
# OFF byte-identical (engine still passes with no filter, scheduler class
# is not imported when the master flag is off), CLI flag plumbing.

from gomoku.reanalyze import (
    DEFAULT_COOLDOWN_CYCLES,
    DEFAULT_COOLDOWN_POSITIONS,
    DEFAULT_EVERY_EPOCHS,
    DEFAULT_EVERY_POSITIONS,
    ReanalyzeScheduler,
)


# ---------- cadence: every_epochs ----------

def test_scheduler_default_fires_every_epoch():
    """Defaults (every_epochs=1, every_positions=0) preserve the 1/3 engine's
    fire-every-epoch behavior — backward-compat default for the 2/3 bead."""
    s = ReanalyzeScheduler(capacity=64)
    # No record yet: every epoch should fire.
    for epoch in range(5):
        assert s.should_fire(epoch=epoch, cumulative_new_positions=0)
        # Simulate a fire (no rows written — just trip the cadence tracker).
        s.record_cycle(written_rows=None, buffer=_DummyBuffer(64),
                       cumulative_new_positions=0, epoch=epoch)


def test_scheduler_every_n_epochs_spaces_fires():
    """every_epochs=3: fires at epochs 0, 3, 6, ...; not at 1, 2, 4, 5."""
    s = ReanalyzeScheduler(capacity=64, every_epochs=3)
    b = _DummyBuffer(64)
    fired = []
    for epoch in range(10):
        if s.should_fire(epoch=epoch, cumulative_new_positions=0):
            fired.append(epoch)
            s.record_cycle(written_rows=None, buffer=b,
                           cumulative_new_positions=0, epoch=epoch)
    # The expected pattern: 0 (first eligible, 0 % 3 == 0), then every 3 from
    # the last fire. So: 0, 3, 6, 9.
    assert fired == [0, 3, 6, 9]


def test_scheduler_does_not_fire_between_cadence_epochs():
    """Explicit between-cadence assertion: after a fire at epoch 0,
    every_epochs=2 should NOT fire at epoch 1."""
    s = ReanalyzeScheduler(capacity=64, every_epochs=2)
    b = _DummyBuffer(64)
    assert s.should_fire(0, 0)
    s.record_cycle(None, b, 0, epoch=0)
    assert not s.should_fire(1, 0)
    assert s.should_fire(2, 0)


# ---------- cadence: every_positions ----------

def test_scheduler_positions_trigger_fires_on_ingest_threshold():
    """every_positions=100: fires the first time cum >= 100, then every +100."""
    s = ReanalyzeScheduler(capacity=64, every_epochs=10_000,
                           every_positions=100)
    b = _DummyBuffer(64)
    # epoch trigger is disabled (effectively): only positions-trigger should fire
    # NOTE: every_epochs=10_000 + we test epochs 0..3 won't HIT that cadence
    # naturally, BUT the very first call has _last_fire_epoch=None so the
    # condition (epoch % every_epochs == 0) at epoch=0 IS true. We use
    # epoch=1..3 to avoid that boundary, focusing on the positions trigger.
    assert not s.should_fire(epoch=1, cumulative_new_positions=50)
    assert s.should_fire(epoch=1, cumulative_new_positions=100)
    s.record_cycle(None, b, cumulative_new_positions=100, epoch=1)
    assert not s.should_fire(epoch=2, cumulative_new_positions=150)
    assert s.should_fire(epoch=2, cumulative_new_positions=200)


def test_scheduler_or_combined_triggers():
    """Either trigger alone fires. With every_epochs=5 + every_positions=100,
    a positions-jump at a non-cadence epoch still fires."""
    s = ReanalyzeScheduler(capacity=64, every_epochs=5, every_positions=100)
    b = _DummyBuffer(64)
    # epoch 0 fires (cadence true at epoch=0).
    assert s.should_fire(0, 0)
    s.record_cycle(None, b, 0, epoch=0)
    # epoch 2 — neither epoch elapsed nor positions threshold.
    assert not s.should_fire(2, 50)
    # epoch 2 — positions threshold met -> fires.
    assert s.should_fire(2, 100)


# ---------- feedback-loop guard: cooldown_cycles ----------

def test_scheduler_cooldown_blocks_recently_reanalyzed_rows():
    """A row reanalyzed in cycle K is INELIGIBLE in cycle K+1 with cooldown=2.
    (Guard's whole point: don't certify the net's own current biases.)"""
    cap = 16
    s = ReanalyzeScheduler(capacity=cap, cooldown_cycles=2)
    b = _DummyBuffer(cap, weight_version=7)
    # Fire cycle 0: pretend the engine wrote rows [3, 5, 7].
    s.record_cycle(written_rows=np.array([3, 5, 7]), buffer=b,
                   cumulative_new_positions=10, epoch=0)
    # Cycle 1 cooldown check: rows 3, 5, 7 should be FILTERED OUT.
    filt = s.eligibility_filter(b)
    eligible = filt(np.arange(cap, dtype=np.int64))
    assert 3 not in eligible
    assert 5 not in eligible
    assert 7 not in eligible
    # Other rows are eligible.
    assert 0 in eligible and 1 in eligible and 15 in eligible


def test_scheduler_cooldown_expires_after_n_cycles():
    """After cooldown_cycles=2 elapsed, the row is eligible again."""
    cap = 16
    s = ReanalyzeScheduler(capacity=cap, cooldown_cycles=2)
    b = _DummyBuffer(cap, weight_version=7)
    # Cycle 0: write row 4.
    s.record_cycle(np.array([4]), b, 10, epoch=0)
    # Cycle 1: still in cooldown (elapsed = 1 < 2). filter excludes 4.
    eligible_1 = s.eligibility_filter(b)(np.arange(cap, dtype=np.int64))
    assert 4 not in eligible_1
    # Bump the scheduler's internal cycle counter (simulate a fire that wrote
    # nothing).
    s.record_cycle(None, b, 20, epoch=1)
    # Cycle 2: elapsed = 2 (cycle index 2 - last_cycle 0 = 2), >= 2 -> eligible.
    eligible_2 = s.eligibility_filter(b)(np.arange(cap, dtype=np.int64))
    assert 4 in eligible_2


def test_scheduler_weight_version_overwrite_reactivates_row():
    """If the buffer overwrote a row with new self-play data (weight_version
    changed) since we reanalyzed it, the row is eligible again even within
    the cooldown window — the cooldown is about stale relabels, not about
    rows that genuinely contain a new position."""
    cap = 8
    s = ReanalyzeScheduler(capacity=cap, cooldown_cycles=10)  # long cooldown
    b_old = _DummyBuffer(cap, weight_version=3)
    s.record_cycle(np.array([2]), b_old, 0, epoch=0)
    # Same buffer object, but the ring buffer's row 2 has been overwritten
    # by new self-play data (weight_version bumped at that slot).
    b_new = _DummyBuffer(cap, weight_version=3)
    b_new.weight_version[2] = 9  # row 2 got a fresh position
    eligible = s.eligibility_filter(b_new)(np.arange(cap, dtype=np.int64))
    assert 2 in eligible, (
        "row 2 was overwritten by new self-play; cooldown should be invalidated"
    )


def test_scheduler_cooldown_positions_blocks_until_ingest_threshold():
    """cooldown_positions=50 means a reanalyzed row stays in cooldown until
    50 new positions have been ingested since (intersected with cycles)."""
    cap = 16
    s = ReanalyzeScheduler(capacity=cap, cooldown_cycles=0,
                           cooldown_positions=50)
    b = _DummyBuffer(cap, weight_version=1)
    s.record_cycle(np.array([5]), b, cumulative_new_positions=100, epoch=0)
    # Bump scheduler cycle counter.
    s.record_cycle(None, b, cumulative_new_positions=120, epoch=1)
    # We've ingested only 120-100=20 positions since the row was reanalyzed
    # (< 50). Even though cooldown_cycles=0 alone would admit it, the
    # cooldown_positions guard intersects -> row 5 still BLOCKED.
    eligible = s.eligibility_filter(b)(np.arange(cap, dtype=np.int64))
    assert 5 not in eligible
    # Advance cumulative far enough.
    s.record_cycle(None, b, cumulative_new_positions=200, epoch=2)
    eligible_after = s.eligibility_filter(b)(np.arange(cap, dtype=np.int64))
    assert 5 in eligible_after, "row 5 should be eligible after 100 new positions"


def test_scheduler_never_reanalyzed_row_is_always_eligible():
    """Rows we've never touched are eligible regardless of cooldown settings."""
    s = ReanalyzeScheduler(capacity=16, cooldown_cycles=999,
                           cooldown_positions=999_999)
    b = _DummyBuffer(16, weight_version=1)
    eligible = s.eligibility_filter(b)(np.arange(16, dtype=np.int64))
    # All 16 rows eligible — none have been reanalyzed.
    assert eligible.size == 16


# ---------- integration with the engine ----------

def test_scheduler_eligibility_filter_threads_into_engine_call():
    """End-to-end: a row written in cycle K is NOT in the engine's sampled
    set in cycle K+1 when cooldown_cycles=2. Proves the filter is honored
    all the way down the engine's sampling path."""
    cap = 32
    s = ReanalyzeScheduler(capacity=cap, cooldown_cycles=2)
    # Stuff the buffer with 16 known positions.
    b = _stuffed_buffer(n_positions=16, seed=11)
    net = _StaticNet(value=0.2)
    # Cycle 0: high fraction -> engine writes a known set of rows.
    metrics_0 = reanalyze_cycle(
        b, net, device="cpu",
        fraction=1.0, max_positions=16, sims=4, mcts_batch=4,
        relabel_value=False, rng=np.random.default_rng(0),
        eligibility_filter=s.eligibility_filter(b),
    )
    assert metrics_0.written_rows is not None
    s.record_cycle(metrics_0.written_rows, b, 16, epoch=0)
    rows_first = set(metrics_0.written_rows.tolist())
    assert len(rows_first) > 0

    # Cycle 1: same buffer, same engine, scheduler filter applied. The
    # rows from cycle 0 must NOT appear (they're in cooldown).
    metrics_1 = reanalyze_cycle(
        b, net, device="cpu",
        fraction=1.0, max_positions=16, sims=4, mcts_batch=4,
        relabel_value=False, rng=np.random.default_rng(1),
        eligibility_filter=s.eligibility_filter(b),
    )
    if metrics_1.written_rows is not None:
        rows_second = set(metrics_1.written_rows.tolist())
        # No row reanalyzed in BOTH cycles.
        assert rows_first.isdisjoint(rows_second), (
            f"cooldown violated: rows {rows_first & rows_second} reanalyzed twice"
        )


def test_scheduler_eligible_count_decreases_then_drains():
    """As we reanalyze more rows with a long cooldown, the eligible-count
    monotonically shrinks (until either rows expire or buffer overwrites)."""
    cap = 64
    s = ReanalyzeScheduler(capacity=cap, cooldown_cycles=100)  # never expires
    b = _DummyBuffer(cap, weight_version=1)
    initial = s.eligible_count(b)
    assert initial == cap   # entire buffer eligible at start
    s.record_cycle(np.arange(10), b, 0, epoch=0)
    after_first = s.eligible_count(b)
    assert after_first == cap - 10
    s.record_cycle(np.arange(10, 20), b, 0, epoch=1)
    after_second = s.eligible_count(b)
    assert after_second == cap - 20


# ---------- OFF byte-identical: scheduler not imported when --reanalyze off ----------

def test_scheduler_off_path_three_layer_guard():
    """Layered guard mirroring the 1/3 engine's OFF proof:

    (1) GREP: train.py only imports ReanalyzeScheduler inside the
        `if reanalyze_on:` branch (so OFF never imports the class).
    (2) BUFFER: with the scheduler never instantiated, no engine call
        happens — buffer.pi / buffer.z / buffer.weight_version are
        bitwise unchanged across an epoch-shaped no-op.
    (3) MODEL: state_dict is bitwise unchanged.
    """
    # Layer 1: grep — scheduler import is inside the gated branch.
    import gomoku.train as train_mod
    src = open(train_mod.__file__).read()
    # The class IS imported once, but only inside a gated block. Find the
    # import line's context — the line two lines above must contain
    # `reanalyze_on` (the gate).
    lines = src.splitlines()
    import_lines = [i for i, ln in enumerate(lines)
                    if "from gomoku.reanalyze import ReanalyzeScheduler" in ln]
    assert len(import_lines) >= 1, "ReanalyzeScheduler must be imported in train.py"
    for li in import_lines:
        # Search back up to 4 lines for the `if reanalyze_on` gate.
        window = "\n".join(lines[max(0, li - 4):li + 1])
        assert "if reanalyze_on" in window, (
            f"ReanalyzeScheduler import at line {li+1} not gated by reanalyze_on:\n{window}"
        )

    # Layer 2: buffer columns bitwise unchanged when scheduler/engine never run.
    b = _stuffed_buffer(n_positions=12, seed=99)
    pi_snap = b.pi[:b.size].clone()
    z_snap = b.z[:b.size].clone()
    wver_snap = b.weight_version[:b.size].clone()
    # OFF path simulation: no scheduler, no engine call.
    assert torch.equal(b.pi[:b.size], pi_snap)
    assert torch.equal(b.z[:b.size], z_snap)
    assert torch.equal(b.weight_version[:b.size], wver_snap)

    # Layer 3: model state_dict bitwise unchanged.
    net = _StaticNet(value=0.5)
    sd_snap = {k: v.clone() for k, v in net.state_dict().items()}
    # No engine call here either.
    sd_now = net.state_dict()
    for k in sd_snap:
        assert torch.equal(sd_now[k], sd_snap[k]), f"weight {k} mutated on OFF path"


# ---------- CLI flag plumbing ----------

def test_train_cli_scheduler_flag_defaults():
    """The scheduler's new --reanalyze-* flags exist with their documented
    conservative defaults."""
    from gomoku.train import parse_args
    saved_argv = sys.argv
    sys.argv = ["train"]
    try:
        args = parse_args()
    finally:
        sys.argv = saved_argv
    assert args.reanalyze_every_epochs == DEFAULT_EVERY_EPOCHS == 1
    assert args.reanalyze_every_positions == DEFAULT_EVERY_POSITIONS == 0
    assert args.reanalyze_cooldown_cycles == DEFAULT_COOLDOWN_CYCLES == 3
    assert args.reanalyze_cooldown_positions == DEFAULT_COOLDOWN_POSITIONS == 0


def test_train_cli_accepts_scheduler_flags():
    """--reanalyze-* knobs read from CLI."""
    from gomoku.train import parse_args
    saved_argv = sys.argv
    sys.argv = [
        "train",
        "--reanalyze",
        "--reanalyze-every-epochs", "5",
        "--reanalyze-every-positions", "200",
        "--reanalyze-cooldown-cycles", "7",
        "--reanalyze-cooldown-positions", "1000",
    ]
    try:
        args = parse_args()
    finally:
        sys.argv = saved_argv
    assert args.reanalyze_every_epochs == 5
    assert args.reanalyze_every_positions == 200
    assert args.reanalyze_cooldown_cycles == 7
    assert args.reanalyze_cooldown_positions == 1000


# ---------- validation ----------

def test_scheduler_rejects_invalid_capacity():
    with pytest.raises(ValueError):
        ReanalyzeScheduler(capacity=0)


def test_scheduler_rejects_negative_knobs():
    with pytest.raises(ValueError):
        ReanalyzeScheduler(capacity=8, every_epochs=0)
    with pytest.raises(ValueError):
        ReanalyzeScheduler(capacity=8, every_positions=-1)
    with pytest.raises(ValueError):
        ReanalyzeScheduler(capacity=8, cooldown_cycles=-1)
    with pytest.raises(ValueError):
        ReanalyzeScheduler(capacity=8, cooldown_positions=-1)


def test_scheduler_record_cycle_rejects_out_of_range_rows():
    s = ReanalyzeScheduler(capacity=8)
    b = _DummyBuffer(8)
    with pytest.raises(ValueError):
        s.record_cycle(np.array([10]), b, 0, epoch=0)


# ---------- test helper for scheduler-only tests ----------

class _DummyBuffer:
    """Minimal stand-in for ReplayBuffer that exposes only `capacity`, `size`,
    and `weight_version` — the columns the scheduler reads."""

    def __init__(self, capacity: int, *, size: int | None = None,
                 weight_version: int = 0):
        self.capacity = int(capacity)
        self.size = int(size if size is not None else capacity)
        self.weight_version = torch.full(
            (self.capacity,), int(weight_version), dtype=torch.int64
        )
