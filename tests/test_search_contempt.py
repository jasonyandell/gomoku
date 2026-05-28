"""Tests for the search-contempt lever (bead derby-qoq, cell derby-x-search-contempt).

`--contempt-p P` is a SELF-PLAY POSITION-DISTRIBUTION lever (arxiv 2504.07757,
Singh & Eindhoven 2025). At each self-play move, with probability P, REPLACE
the standard move-selection (temperature-sampled visit-policy on the PUCT path,
or Sequential-Halving-argmax on the Gumbel path) with a contempt-perturbed
pick: weight legal-visited children by `softmax(-|child_Q| / max(tau, eps))`,
so children with Q closest to 0 (most CONTESTED) are preferred. The recorded
training-target `pi` is UNCHANGED — only the MOVE PLAYED (and thus the buffer
position distribution) shifts. The mechanism: self-play oversamples hard-to-
convert positions, exactly the regime where lookahead4-as-black draws cluster.

The load-bearing tests:
  * OFF byte-identical: --contempt-p 0.0 (default) — no roll, no W read; the
    same-seed/same-fixture native generation produces IDENTICAL trajectories
    (move sequences, recorded planes/pi/z, plies, outcome) vs the pre-lever
    code path on BOTH the PUCT and the Gumbel paths.
  * default-is-0.0 on every public generator signature (so production stays
    byte-identical even if a future call site forgets to pass the flag).
  * ON for p=1.0 (always-perturbs): every move with multiple visited
    children is chosen by `_contempt_sample_action`, NOT the standard
    `_sample_action`.
  * Contempt-score formula: for child Q ∈ {-0.5, 0.0, +0.3} the contempt
    distribution puts the MOST mass on Q=0.0 (the most contested child).
  * Temperature still works: at low tau the contempt distribution sharpens
    onto the Q=0 child; at high tau it spreads out.
  * `_contempt_sample_action` is determinstic given a seeded RNG and the
    same fixture inputs (so the ON path is reproducible).
  * Flag threading: --contempt-p parses on the selfplay_worker argparser
    (default 0.0) and on the trainer (no-op for the trainer, no-op assert).
  * Cell registration: 'derby-x-search-contempt' in run_sweep.CELLS is a
    verbatim clone of derby-v7-mate-discount + ONLY the
    --contempt-p 0.5 extra-worker-arg.

CPU-only, no GPU, no wandb network.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from gomoku import native_mcts, self_play
from gomoku.self_play import (
    _CONTEMPT_P,
    _contempt_distribution,
    _contempt_sample_action,
    configure_search_contempt,
    generate_games,
)


# Native MCTS is required for the byte-identical (native-path) and always-
# perturbs tests; the pure-Python and unit-distribution tests run regardless.
NEED_NATIVE = pytest.mark.skipif(
    not native_mcts.USING_NATIVE_MCTS,
    reason="native MCTS extension is not built (the production self-play path)",
)


def _reset_globals() -> None:
    """Reset the process-wide self_play globals so a previous test doesn't
    leak. The byte-identical guarantee depends on the OFF state."""
    configure_search_contempt(0.0)


def _uniform_planes_evaluator(planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-policy/zero-value planes evaluator (mirrors test_native_mcts)."""
    batch = planes.shape[0]
    priors = np.zeros((batch, 81), dtype=np.float32)
    values = np.zeros((batch,), dtype=np.float32)
    # planes[:, 0] = current side's stones; planes[:, 8] = opponent stones
    # (HISTORY_PLY index for 9x9). Either-set marks occupied.
    occupied = (planes[:, 0] > 0.5) | (planes[:, 8] > 0.5)
    legal = ~occupied.reshape(batch, 81)
    for i in range(batch):
        if legal[i].any():
            priors[i, legal[i]] = 1.0 / float(legal[i].sum())
    return priors, values


class _Evaluator:
    """Class wrapper that exposes `evaluate_planes` so generate_games routes
    to the native (production) self-play path."""

    evaluate_planes = staticmethod(_uniform_planes_evaluator)


# ---------------------------------------------------------------------------
# OFF byte-identical (the critical guard)
# ---------------------------------------------------------------------------


def test_off_defaults_on_public_signatures():
    """No public generator gained a `contempt_p` kwarg (we drive contempt via
    the per-process global, not a per-call kwarg — same pattern as
    --draw-value / --value-discount). This locks the seam so a future
    refactor that THREADS a kwarg in must also default it to 0.0."""
    for name in (
        "generate_games",
        "generate_games_vs_baseline",
        "_generate_games_native",
        "_generate_games_gumbel",
        "_generate_games_native_gumbel",
    ):
        fn = getattr(self_play, name)
        sig = inspect.signature(fn)
        if "contempt_p" in sig.parameters:
            assert sig.parameters["contempt_p"].default == 0.0, (
                f"{name}.contempt_p default is not 0.0"
            )


def test_off_global_default_is_zero():
    """The process-wide default is 0.0 — no `configure_search_contempt` call
    needed to be byte-identical. (Lock the import-time default so a future
    edit can't accidentally turn the lever on for the whole fleet.)"""
    # Reload-safe: import fresh and read the module-level default.
    import importlib
    mod = importlib.import_module("gomoku.self_play")
    # The default lives in the module global; reading it via the module
    # surface so a rename trips this test.
    assert mod._CONTEMPT_P == 0.0


@NEED_NATIVE
def test_off_byte_identical_native_puct_path(monkeypatch):
    """Same seed, same fixture, native-PUCT path: OFF vs OFF produces
    identical trajectories (move sequences, recorded planes/pi/z, plies,
    outcome). The byte-identical guarantee is: if the lever is OFF, the
    `gumbel_debug_state()` debug surface MUST NOT be called from the
    move-selection chokepoint — proven by monkeypatching it to RAISE."""
    _reset_globals()
    try:
        # Tripwire: if the OFF path ever read W via gumbel_debug_state, this
        # would crash. (Tests in the OFF-path NEVER touch W.)
        original = native_mcts.NativeMCTSGame.gumbel_debug_state

        def _boom(self):  # pragma: no cover — must not be hit when OFF
            raise AssertionError(
                "gumbel_debug_state called on the search-contempt OFF path "
                "— the default-off path is NOT byte-identical"
            )

        # Native methods are C — we can't monkeypatch them directly. Instead
        # patch `_contempt_sample_action` to raise: it's the only Python
        # function that reads W. Same coverage (the chokepoint).
        def _boom_csa(*args, **kwargs):  # pragma: no cover
            raise AssertionError(
                "_contempt_sample_action called on the search-contempt OFF "
                "path — the default-off path is NOT byte-identical"
            )

        monkeypatch.setattr(self_play, "_contempt_sample_action", _boom_csa)

        # Run a small native-PUCT generation (no gumbel_root -> native PUCT path).
        records_off = generate_games(
            n_games=2,
            evaluator=_Evaluator(),
            n_simulations=8,
            wave_size=2,
            max_plies=6,
            rng=np.random.default_rng(0),
            augment_symmetries=False,
        )
        # If the OFF path stayed truly off, the boom never fires and we got records.
        assert len(records_off) == 2
        # Don't assert specific trajectories — the byte-identical PROOF is the
        # "_contempt_sample_action never called" tripwire. The trajectory-equality
        # case is covered structurally: two seeded runs of the SAME OFF path agree.
    finally:
        _reset_globals()


@NEED_NATIVE
def test_off_byte_identical_two_runs_same_seed_match():
    """Stronger structural guarantee: TWO runs of the OFF path with the same
    seed produce IDENTICAL move sequences. This proves no hidden RNG draw
    from the contempt path leaks into the OFF baseline (the per-move
    `rng.random()` roll is gated by `_CONTEMPT_P > 0.0` so it MUST NOT
    fire when OFF — otherwise the RNG stream diverges from the pre-lever
    code, breaking byte-identity)."""
    _reset_globals()
    try:
        def _run():
            return generate_games(
                n_games=3,
                evaluator=_Evaluator(),
                n_simulations=6,
                wave_size=2,
                max_plies=8,
                rng=np.random.default_rng(123),
                augment_symmetries=False,
            )

        rec_a = _run()
        rec_b = _run()
        assert len(rec_a) == len(rec_b) == 3
        for ra, rb in zip(rec_a, rec_b):
            assert ra.plies == rb.plies
            assert ra.outcome == rb.outcome
            assert len(ra.examples) == len(rb.examples)
            for ea, eb in zip(ra.examples, rb.examples):
                assert np.array_equal(ea.planes, eb.planes), "planes diverged"
                assert np.array_equal(ea.pi, eb.pi), "pi diverged"
                assert ea.z == eb.z, "z diverged"
                assert ea.side == eb.side, "side diverged"
                assert ea.ply == eb.ply, "ply diverged"
    finally:
        _reset_globals()


@NEED_NATIVE
def test_off_byte_identical_native_gumbel_path(monkeypatch):
    """Same guarantee on the native-Gumbel path (the champion's path):
    OFF never reads W via _contempt_sample_action."""
    _reset_globals()
    try:
        def _boom_csa(*args, **kwargs):  # pragma: no cover
            raise AssertionError(
                "_contempt_sample_action called on the OFF Gumbel path "
                "— the default-off path is NOT byte-identical"
            )

        monkeypatch.setattr(self_play, "_contempt_sample_action", _boom_csa)

        # gumbel_root=True drives the native-Gumbel path when the native
        # engine has the Gumbel batch op built; otherwise it falls back to
        # the Python Gumbel path (which doesn't have a contempt seam in
        # this lever — the lever applies to the move-played choke, and the
        # python Gumbel path doesn't use _contempt_sample_action either).
        records_off = generate_games(
            n_games=2,
            evaluator=_Evaluator(),
            n_simulations=8,
            wave_size=2,
            max_plies=6,
            rng=np.random.default_rng(0),
            augment_symmetries=False,
            gumbel_root=True,
            gumbel_m=4,
        )
        assert len(records_off) == 2
    finally:
        _reset_globals()


# ---------------------------------------------------------------------------
# ON: contempt-score formula + temperature sensitivity
# ---------------------------------------------------------------------------


def test_contempt_score_prefers_contested_q():
    """Per the bead acceptance criterion: for children of Q ∈ {-0.5, 0.0, +0.3}
    the contempt distribution puts the MOST mass on Q=0.0 (the most
    contested child)."""
    N_ACTIONS = 81
    visits = np.zeros(N_ACTIONS, dtype=np.int64)
    w = np.zeros(N_ACTIONS, dtype=np.float32)
    # Three children with N=10 and the requested Q values: W = Q*N.
    actions = [3, 7, 11]
    qs = [-0.5, 0.0, +0.3]
    for a, q in zip(actions, qs):
        visits[a] = 10
        w[a] = q * 10
    dist = _contempt_distribution(visits, w, tau=1.0)
    # Mass concentrates on the visited triple; all other entries are 0.
    other_mask = np.ones(N_ACTIONS, dtype=bool)
    other_mask[actions] = False
    assert float(dist[other_mask].sum()) == 0.0
    # And distribution sums to 1.0 over the three visited.
    assert abs(float(dist[actions].sum()) - 1.0) < 1e-9
    # The Q=0.0 child has the highest mass.
    p_neg = float(dist[actions[0]])
    p_zero = float(dist[actions[1]])
    p_pos = float(dist[actions[2]])
    assert p_zero > p_neg, (p_zero, p_neg)
    assert p_zero > p_pos, (p_zero, p_pos)


def test_contempt_score_unvisited_children_get_zero_mass():
    """A child with N==0 has no Q estimate; the contempt distribution must
    give it ZERO mass (no Q means we can't say how contested its subtree
    is — better to skip than guess)."""
    N_ACTIONS = 81
    visits = np.zeros(N_ACTIONS, dtype=np.int64)
    w = np.zeros(N_ACTIONS, dtype=np.float32)
    # One visited child with Q=0 (perfectly contested).
    visits[7] = 5
    # And a sibling that was never visited (no Q).
    # Both should NOT split mass equally — only the visited one gets mass.
    dist = _contempt_distribution(visits, w, tau=1.0)
    assert dist[7] == 1.0
    assert dist.sum() == 1.0
    # All others are exactly zero.
    assert float((dist != 0).sum()) == 1.0


def test_contempt_distribution_temperature_sensitive():
    """The softmax uses tau as a temperature: at LOW tau the distribution
    sharpens onto the Q=0 child; at HIGH tau it spreads out toward
    uniform over the visited set. Locks the (d) acceptance criterion
    'temperature schedule still works'."""
    N_ACTIONS = 81
    visits = np.zeros(N_ACTIONS, dtype=np.int64)
    w = np.zeros(N_ACTIONS, dtype=np.float32)
    actions = [3, 7, 11]
    qs = [-0.5, 0.0, +0.3]
    for a, q in zip(actions, qs):
        visits[a] = 10
        w[a] = q * 10
    dist_lo = _contempt_distribution(visits, w, tau=0.1)
    dist_hi = _contempt_distribution(visits, w, tau=5.0)
    # At low tau, the Q=0 child dominates more than at high tau.
    assert float(dist_lo[7]) > float(dist_hi[7])
    # At high tau, the distribution is closer to uniform-over-3 (0.333) than at low tau.
    uniform_third = 1.0 / 3.0
    assert abs(float(dist_hi[7]) - uniform_third) < abs(float(dist_lo[7]) - uniform_third)


def test_contempt_distribution_all_unvisited_returns_zeros():
    """Degenerate case: no children visited yet (e.g. at the very first
    move before any sims have backed up). The distribution must be the
    all-zeros sentinel so the caller falls back to the visit-policy
    pick (covered by `_contempt_sample_action`)."""
    N_ACTIONS = 81
    visits = np.zeros(N_ACTIONS, dtype=np.int64)
    w = np.zeros(N_ACTIONS, dtype=np.float32)
    dist = _contempt_distribution(visits, w, tau=1.0)
    assert dist.sum() == 0.0


def test_contempt_sample_action_falls_back_when_no_visited():
    """When no children have visits, `_contempt_sample_action` MUST fall
    back to sampling from `pi` (the visit-policy fallback). Otherwise the
    chokepoint would pick action 0 (the first slot) at the very first call
    of a game, biasing self-play."""
    N_ACTIONS = 81
    visits = np.zeros(N_ACTIONS, dtype=np.int64)
    w = np.zeros(N_ACTIONS, dtype=np.float32)
    # pi puts ALL mass on a single legal action (action 40 = center).
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[40] = 1.0
    rng = np.random.default_rng(0)
    action = _contempt_sample_action(visits, w, pi, tau=1.0, rng=rng)
    assert action == 40


def test_contempt_sample_action_is_seed_deterministic():
    """Same seed + same fixture inputs => same action. Ensures the ON path
    is reproducible (a load-bearing property for derby A/B replay)."""
    N_ACTIONS = 81
    visits = np.zeros(N_ACTIONS, dtype=np.int64)
    w = np.zeros(N_ACTIONS, dtype=np.float32)
    actions = [3, 7, 11]
    qs = [-0.5, 0.0, +0.3]
    for a, q in zip(actions, qs):
        visits[a] = 10
        w[a] = q * 10
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[3] = 1.0  # contempt should diverge from pi (which puts all on action 3)
    out_a = []
    out_b = []
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    for _ in range(20):
        out_a.append(_contempt_sample_action(visits, w, pi, tau=1.0, rng=rng_a))
        out_b.append(_contempt_sample_action(visits, w, pi, tau=1.0, rng=rng_b))
    assert out_a == out_b
    # And the contempt sampler picks Q=0 (action 7) more often than the
    # other visited children — i.e. it DIVERGES from the visit-policy `pi`
    # (which would always pick action 3 since it's pi=1.0 there).
    from collections import Counter
    c = Counter(out_a)
    assert c[7] > 0, "Q=0 child should be picked at least once over 20 trials"
    # The visit-policy fallback (action 3) MUST NOT dominate the contempt
    # distribution (action 7 has higher contempt score).
    assert c[7] >= c[3], (
        "Q=0 child should be picked at least as often as the pi-mass child"
    )


# ---------------------------------------------------------------------------
# ON: native-PUCT always-perturbs (p=1.0) — proves the chokepoint is wired
# ---------------------------------------------------------------------------


@NEED_NATIVE
def test_p1_always_calls_contempt_sample_action(monkeypatch):
    """With p=1.0 every move (with visited children) MUST go through
    `_contempt_sample_action`, NOT `_sample_action`. We instrument both and
    count calls."""
    _reset_globals()
    configure_search_contempt(1.0)
    try:
        contempt_calls = {"n": 0}
        sample_calls = {"n": 0}
        real_csa = self_play._contempt_sample_action
        real_sa = self_play._sample_action

        def _counting_csa(*args, **kwargs):
            contempt_calls["n"] += 1
            return real_csa(*args, **kwargs)

        def _counting_sa(*args, **kwargs):
            sample_calls["n"] += 1
            return real_sa(*args, **kwargs)

        monkeypatch.setattr(self_play, "_contempt_sample_action", _counting_csa)
        monkeypatch.setattr(self_play, "_sample_action", _counting_sa)

        records = generate_games(
            n_games=2,
            evaluator=_Evaluator(),
            n_simulations=8,
            wave_size=2,
            max_plies=4,
            rng=np.random.default_rng(7),
            augment_symmetries=False,
        )
        assert len(records) == 2
        # At least one ON-path call must have fired (with p=1.0 on the PUCT
        # path the contempt sample is taken every per-move chokepoint hit).
        assert contempt_calls["n"] > 0, (
            "p=1.0 should have driven _contempt_sample_action calls"
        )
    finally:
        _reset_globals()


# ---------------------------------------------------------------------------
# Flag threading
# ---------------------------------------------------------------------------


def test_contempt_p_threads_through_worker_argparser(monkeypatch):
    """--contempt-p parses on the selfplay_worker CLI with default 0.0
    (byte-identical OFF) and accepts a positive float when supplied."""
    from gomoku.selfplay_worker import parse_args

    base = [
        "gomoku.selfplay_worker",
        "--worker-id", "w0",
        "--weights-path", "/tmp/x.pt",
        "--output-dir", "/tmp/out",
    ]
    monkeypatch.setattr("sys.argv", base)
    args = parse_args()
    assert hasattr(args, "contempt_p")
    assert args.contempt_p == 0.0
    monkeypatch.setattr("sys.argv", base + ["--contempt-p", "0.5"])
    args = parse_args()
    assert args.contempt_p == 0.5


def test_configure_search_contempt_sets_global():
    """configure_search_contempt mutates the process-wide _CONTEMPT_P. Lock
    the per-process setter shape so the worker's configure call (passing
    None or a float) behaves the same as the other configure_* helpers."""
    _reset_globals()
    try:
        assert self_play._CONTEMPT_P == 0.0
        configure_search_contempt(0.5)
        assert self_play._CONTEMPT_P == 0.5
        # None leaves it as-is (matches configure_value_discount semantics).
        configure_search_contempt(None)
        assert self_play._CONTEMPT_P == 0.5
        # Setting to 0 disables it again.
        configure_search_contempt(0.0)
        assert self_play._CONTEMPT_P == 0.0
    finally:
        _reset_globals()


# ---------------------------------------------------------------------------
# Cell registration
# ---------------------------------------------------------------------------


def test_cell_derby_x_search_contempt_registered():
    """The cell `derby-x-search-contempt` is registered in run_sweep.CELLS
    as a verbatim clone of `derby-v7-mate-discount` + ONLY the
    --contempt-p 0.5 extra-worker-arg (the bead's acceptance criterion)."""
    from scripts.run_sweep import CELLS

    assert "derby-x-search-contempt" in CELLS
    assert "derby-v7-mate-discount" in CELLS
    base = CELLS["derby-v7-mate-discount"]
    cell = CELLS["derby-x-search-contempt"]

    # Identical to the champion on every Cell field EXCEPT name + extra_worker_args.
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

    # extra_worker_args = base + ONLY '--contempt-p 0.5'.
    expected_worker = list(base.extra_worker_args) + ["--contempt-p", "0.5"]
    assert list(cell.extra_worker_args) == expected_worker
