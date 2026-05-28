"""Tests for the draw-contempt lever (bead derby-9q4, cell derby-x-draw-contempt).

`--draw-value DELTA` is a training-side value-TARGET reshape: when DELTA > 0 and
a game ends in a DRAW (raw outcome z=0), the value target is set to -DELTA
(mildly losing) instead of exactly 0. Sibling of `--value-discount`; composes
with it via the same gamma^plies shape so the contempt magnitude decays with
distance-to-end exactly like the mate-discount does for decisive outcomes.

The load-bearing tests:
  * OFF byte-identical: --draw-value 0.0 (default) leaves _discount_z exactly
    as it was pre-lever for ALL outcomes (decisive AND draws), including
    composition with --value-discount.
  * ON for draws (scalar): --draw-value 0.05 -> _discount_z(0, plies) = -0.05.
  * Composition with --value-discount: --draw-value 0.05 + gamma=0.98 -> a
    drawn position N plies from the end has z = -0.05 * 0.98^N (same shape as
    decisive outcomes' z * gamma^plies).
  * ON does NOT affect decisive outcomes (the lever is draw-only, by design).
  * WDL re-expression: the existing wdl_target_from_z maps z=-DELTA to
    (0, 1-DELTA, DELTA), shifting DELTA probability mass from draw->loss vertex
    (per the bead's acceptance criteria).
  * Flag threading: --draw-value parses on both the trainer and the worker
    argparser; the cell `derby-x-draw-contempt` is registered with the
    expected verbatim-clone + ONLY-draw-value-0.05 delta.

CPU-only, no GPU, no wandb network.
"""

from __future__ import annotations

import torch

from gomoku.self_play import (
    _DRAW_VALUE,
    _VALUE_DISCOUNT,
    _discount_z,
    configure_draw_value,
    configure_value_discount,
)


def _reset_globals() -> None:
    """Reset the process-wide self_play globals to their byte-identical defaults
    so a previous test's `configure_*` call doesn't leak."""
    configure_value_discount(1.0)
    configure_draw_value(0.0)


# --- OFF byte-identical (the critical guard) ----------------------------------


def test_off_byte_identical_decisive_no_discount():
    """Default config (no value-discount, no draw-contempt): _discount_z is the
    identity on every input — exactly the pre-lever behavior."""
    _reset_globals()
    for z in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for plies in (0, 1, 5, 20):
            assert _discount_z(z, plies) == z


def test_off_byte_identical_draws_with_value_discount():
    """value-discount ON, draw-contempt OFF: draws (z=0) MUST stay exactly 0
    (this is the pre-9q4 invariant — _discount_z was a no-op on z==0). Decisive
    outcomes scale by gamma^plies. Asserts byte-exact equality on a fixed
    'batch' of (z, plies) inputs."""
    _reset_globals()
    configure_value_discount(0.98)
    try:
        # Decisive outcomes: z * gamma^plies (the pre-lever discount path).
        gamma = 0.98
        for z_in, plies in [(+1.0, 0), (+1.0, 5), (-1.0, 10), (+0.5, 3)]:
            expected = z_in * (gamma ** plies)
            got = _discount_z(z_in, plies)
            # Use floating-point exact equality — _discount_z is the SAME
            # arithmetic as the pre-lever path on the decisive branch.
            assert got == expected, (z_in, plies, got, expected)
        # Draws stay exactly 0 (byte-identical to pre-lever).
        for plies in (0, 1, 5, 20):
            assert _discount_z(0.0, plies) == 0.0
    finally:
        _reset_globals()


def test_off_byte_identical_full_batch_torch_equal():
    """Stronger guard: build a fixed-seed batch of (z, plies) pairs and assert
    the lever-OFF output is `torch.equal` to a tensor computed by the pre-lever
    arithmetic (z * gamma^plies if gamma<1 and z!=0, else z)."""
    _reset_globals()
    configure_value_discount(0.98)
    try:
        torch.manual_seed(0)
        # 32 examples: half decisive (z in {-1,+1}), half draws (z=0).
        zs = torch.cat([
            torch.tensor([1.0, -1.0, 1.0, -1.0] * 4),
            torch.zeros(16),
        ])
        # Random ply counts in [0, 30].
        plies = torch.randint(0, 31, (32,))
        # Pre-lever expected: z * gamma^plies for nonzero z; 0 for z==0.
        gamma = 0.98
        expected = torch.tensor([
            (zs[i].item() * (gamma ** int(plies[i].item())))
            if zs[i].item() != 0.0
            else 0.0
            for i in range(32)
        ])
        # Lever-OFF output via _discount_z (the actual code path).
        got = torch.tensor([
            _discount_z(zs[i].item(), int(plies[i].item()))
            for i in range(32)
        ])
        # Same arithmetic, same operand order -> floats are bit-identical.
        assert torch.equal(got, expected)
    finally:
        _reset_globals()


# --- ON for draws --------------------------------------------------------------


def test_draw_contempt_scalar_no_discount():
    """With --draw-value 0.05 and no value-discount, a drawn game's target is
    exactly -0.05 at every ply (the contempt isn't decayed when gamma>=1.0)."""
    _reset_globals()
    configure_draw_value(0.05)
    try:
        for plies in (0, 1, 5, 20):
            assert _discount_z(0.0, plies) == -0.05
    finally:
        _reset_globals()


def test_draw_contempt_composes_with_value_discount():
    """Composition with --value-discount: a drawn position N plies from the end
    gets z = -DELTA * gamma^N, the SAME gamma^plies shape that decisive
    outcomes use (the bead's 'preserves the existing math shape' requirement)."""
    _reset_globals()
    configure_value_discount(0.98)
    configure_draw_value(0.05)
    try:
        gamma = 0.98
        delta = 0.05
        for plies in (0, 1, 5, 20):
            expected = -delta * (gamma ** plies)
            assert _discount_z(0.0, plies) == expected
    finally:
        _reset_globals()


def test_draw_contempt_does_not_affect_decisive_outcomes():
    """The lever is DRAW-only by design. A decisive z (!= 0) is unchanged by
    --draw-value; only the gamma^plies discount (if any) applies."""
    _reset_globals()
    configure_value_discount(0.98)
    configure_draw_value(0.05)
    try:
        gamma = 0.98
        for z_in in (+1.0, -1.0, +0.5, -0.5):
            for plies in (0, 1, 5, 20):
                expected = z_in * (gamma ** plies)
                assert _discount_z(z_in, plies) == expected
    finally:
        _reset_globals()


# --- WDL re-expression (bead acceptance criterion) -----------------------------


def test_wdl_target_from_z_with_draw_contempt():
    """Per the bead: with DELTA=0.05, z_draw = -0.05 maps via wdl_target_from_z
    to (0, 1-DELTA, DELTA) = (0, 0.95, 0.05) — DELTA probability mass shifts
    from the draw vertex to the LOSS vertex. (DELTA=0 keeps the (0,1,0) draw
    one-hot, the byte-identical baseline.)"""
    from gomoku.train import wdl_target_from_z

    # DELTA=0 (OFF): z=0 -> (0, 1, 0) exact one-hot draw vertex.
    t_off = wdl_target_from_z(torch.tensor([0.0]))[0]
    assert torch.equal(t_off, torch.tensor([0.0, 1.0, 0.0]))
    # DELTA=0.05 (ON): z=-0.05 -> (0, 0.95, 0.05).
    delta = 0.05
    t_on = wdl_target_from_z(torch.tensor([-delta]))[0]
    assert torch.allclose(t_on, torch.tensor([0.0, 1.0 - delta, delta]), atol=1e-7)
    # Distribution stays normalized.
    assert torch.allclose(t_on.sum(), torch.tensor(1.0), atol=1e-7)


# --- Flag threading ------------------------------------------------------------


def test_draw_value_threads_through_trainer_argparser(monkeypatch):
    """--draw-value parses on the trainer CLI with default 0.0 (byte-identical
    OFF) and accepts a positive float when supplied."""
    from gomoku.train import parse_args

    # Default OFF.
    monkeypatch.setattr("sys.argv", ["gomoku-train"])
    args = parse_args()
    assert hasattr(args, "draw_value")
    assert args.draw_value == 0.0
    # Explicit ON.
    monkeypatch.setattr("sys.argv", ["gomoku-train", "--draw-value", "0.05"])
    args = parse_args()
    assert args.draw_value == 0.05


def test_draw_value_threads_through_worker_argparser(monkeypatch):
    """--draw-value parses on the selfplay_worker CLI with default 0.0."""
    from gomoku.selfplay_worker import parse_args

    base = [
        "gomoku.selfplay_worker",
        "--worker-id", "w0",
        "--weights-path", "/tmp/x.pt",
        "--output-dir", "/tmp/out",
    ]
    monkeypatch.setattr("sys.argv", base)
    args = parse_args()
    assert hasattr(args, "draw_value")
    assert args.draw_value == 0.0
    monkeypatch.setattr("sys.argv", base + ["--draw-value", "0.05"])
    args = parse_args()
    assert args.draw_value == 0.05


def test_cell_derby_x_draw_contempt_registered():
    """The cell `derby-x-draw-contempt` is registered in run_sweep.CELLS as a
    verbatim clone of `derby-v7-mate-discount` + ONLY the --draw-value 0.05
    extra-worker-arg (the bead's acceptance criterion)."""
    from scripts.run_sweep import CELLS

    assert "derby-x-draw-contempt" in CELLS
    assert "derby-v7-mate-discount" in CELLS
    base = CELLS["derby-v7-mate-discount"]
    cell = CELLS["derby-x-draw-contempt"]

    # Identical to the champion on every Cell field EXCEPT name + extra_worker_args.
    # (We compare the dataclass fields explicitly rather than asdict so the
    # diff is visible if a future Cell field is added.)
    for field in (
        "sgd_per_game", "buffer_size", "games_per_epoch", "size", "stem_padding",
        "n_simulations", "n_workers", "games_per_batch", "wave_mode",
        "c_puct", "c_puct_base", "dirichlet_alpha", "dirichlet_eps",
        "temperature_moves", "temperature_final", "sgd_per_position",
        "save_buffer_every", "ema_tau", "grad_accum_steps",
        "opponent_mix_recent", "opponent_mix_history",
        "opponent_mix_recent_window", "weights_poll_min_sec",
        "weights_poll_max_sec", "epochs", "random_opening_moves",
        "global_pool", "extra_train_args",
    ):
        assert getattr(cell, field) == getattr(base, field), field

    # extra_worker_args = base + ONLY '--draw-value 0.05'.
    expected_worker = list(base.extra_worker_args) + ["--draw-value", "0.05"]
    assert list(cell.extra_worker_args) == expected_worker
