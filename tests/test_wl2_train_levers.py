"""WL2 trainer levers: EMA self-play weights + gradient accumulation.

Design ref: wiki/topics/wl2-scale-emulation-design.md
"""

import copy

import torch
import torch.nn as nn

from gomoku.model import build_model
from gomoku.train import average_state_dicts, ema_l2_distance, ema_update, train_step


# --------------------------- EMA update math --------------------------------


def _flat_params(m: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in m.parameters()])


# --------------------- V3 SWA tail-averaging math ---------------------------


def _models_with_distinct_bn_counts(nfwds):
    """Build tiny models with distinct weights AND distinct BN batch-counters
    (num_batches_tracked) so 'latest-wins' on int buffers is observable."""
    models = []
    for seed, nfwd in nfwds:
        torch.manual_seed(seed)
        m = build_model("tiny")
        m.train()
        for _ in range(nfwd):
            m(torch.randn(2, m.cfg.n_input_planes, 9, 9))
        models.append(m)
    return models


def test_swa_average_equals_elementwise_float_mean():
    """Every FLOAT tensor in the SWA output is the elementwise mean of inputs."""
    models = _models_with_distinct_bn_counts([(0, 1), (1, 3), (2, 7)])
    sds = [m.state_dict() for m in models]
    avg = average_state_dicts(sds)
    n_float = 0
    for k, ref in sds[-1].items():
        if torch.is_tensor(ref) and ref.is_floating_point():
            expected = sum(sd[k].double() for sd in sds) / len(sds)
            assert torch.allclose(avg[k].double(), expected, atol=1e-6), k
            n_float += 1
    assert n_float > 0


def test_swa_int_buffers_take_latest_not_average():
    """num_batches_tracked (int64) takes the LATEST value, never an average."""
    models = _models_with_distinct_bn_counts([(0, 1), (1, 3), (2, 7)])
    sds = [m.state_dict() for m in models]
    avg = average_state_dicts(sds)
    n_int = 0
    for k, ref in sds[-1].items():
        if torch.is_tensor(ref) and not ref.is_floating_point():
            assert torch.equal(avg[k], sds[-1][k]), k
            n_int += 1
        if "num_batches_tracked" in k:
            assert int(avg[k]) == 7  # latest model had 7 forward passes
    assert n_int > 0


def test_swa_single_window_is_identity():
    """A window of one checkpoint publishes that checkpoint unchanged."""
    m = build_model("tiny")
    sd = m.state_dict()
    avg = average_state_dicts([sd])
    for k, v in sd.items():
        if torch.is_tensor(v):
            assert torch.equal(avg[k], v), k


def test_swa_does_not_mutate_inputs():
    """Averaging must not alias/mutate the source state_dicts (ring buffer safety)."""
    models = _models_with_distinct_bn_counts([(0, 1), (1, 5)])
    sds = [m.state_dict() for m in models]
    before = {k: v.clone() for k, v in sds[0].items() if torch.is_tensor(v)}
    _ = average_state_dicts(sds)
    for k, v in before.items():
        assert torch.equal(sds[0][k], v), k


def test_swa_output_loads_into_fresh_model():
    """The averaged state_dict has full key/shape parity (loadable for publish)."""
    models = _models_with_distinct_bn_counts([(0, 2), (1, 4), (2, 6)])
    avg = average_state_dicts([m.state_dict() for m in models])
    fresh = build_model("tiny")
    fresh.load_state_dict(avg)  # raises on any missing/extra/shape mismatch


def test_ema_tau_zero_copies_model_into_ema():
    """tau=0 means ema is replaced by model on every step."""
    torch.manual_seed(0)
    model = build_model("tiny")
    ema = build_model("tiny")
    # Start with deliberately different params so the update is observable.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    assert ema_l2_distance(ema, model) > 0.0
    ema_update(ema, model, tau=0.0)
    assert torch.allclose(_flat_params(ema), _flat_params(model))


def test_ema_tau_one_freezes_ema():
    """tau=1 means ema never updates — stays at its initial value."""
    torch.manual_seed(0)
    model = build_model("tiny")
    ema = build_model("tiny")
    ema_before = _flat_params(ema).clone()
    # Mutate the model so the EMA would move if tau<1.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    for _ in range(10):
        ema_update(ema, model, tau=1.0)
    assert torch.allclose(_flat_params(ema), ema_before)


def test_ema_converges_to_model_under_repeated_updates():
    """With tau<1 and a fixed target, ema_l2_distance decays toward 0."""
    torch.manual_seed(0)
    model = build_model("tiny")
    ema = build_model("tiny")
    # Push the model away from ema.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
    d0 = ema_l2_distance(ema, model)
    assert d0 > 0.0
    tau = 0.9
    distances = [d0]
    for _ in range(50):
        ema_update(ema, model, tau=tau)
        distances.append(ema_l2_distance(ema, model))
    # Strict monotone decrease toward 0 (fixed target, geometric decay).
    for a, b in zip(distances, distances[1:]):
        assert b < a, "ema_l2_distance should strictly decrease toward a fixed target"
    # After 50 steps at tau=0.9, distance multiplier is 0.9**50 ≈ 5e-3.
    assert distances[-1] < 0.05 * d0


def test_ema_geometric_decay_rate_matches_tau():
    """Analytic check: after k steps with tau and fixed target, distance is tau**k * d0."""
    torch.manual_seed(0)
    model = build_model("tiny")
    ema = build_model("tiny")
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3)
    d0 = ema_l2_distance(ema, model)
    tau = 0.7
    k = 20
    for _ in range(k):
        ema_update(ema, model, tau=tau)
    expected = (tau ** k) * d0
    actual = ema_l2_distance(ema, model)
    assert abs(actual - expected) < 1e-5, f"expected {expected}, got {actual}"


# --------------------- Gradient accumulation correctness --------------------


class TinyHead(nn.Module):
    """BN-free stand-in for GomokuNet. Grad-accumulation equivalence to a
    single big batch is broken by BatchNorm's per-batch normalization (it
    sees different statistics across the two schedules even in eval mode if
    we call model.train()). Use a plain linear stack for the accumulation
    arithmetic test."""

    def __init__(self, in_planes: int = 3):
        super().__init__()
        self.fc_shared = nn.Linear(in_planes * 9 * 9, 32)
        self.fc_policy = nn.Linear(32, 81)
        self.fc_value = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc_shared(x.flatten(1)))
        logits = self.fc_policy(h)
        v = torch.tanh(self.fc_value(h)).squeeze(-1)
        return logits, v


def _make_fake_batch(in_planes: int, batch_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    planes = torch.randn(batch_size, in_planes, 9, 9, generator=g)
    logits_for_target = torch.randn(batch_size, 81, generator=g)
    pi = torch.softmax(logits_for_target, dim=-1)
    z = torch.empty(batch_size).uniform_(-1, 1, generator=g)
    return planes, pi, z


def test_grad_accumulation_matches_single_big_step():
    """Single optimizer.step over a B-sized batch == N accumulated steps over B/N microbatches.

    Uses a BN-free model so the only thing being tested is the accumulation
    arithmetic in train_step. (The real model has BatchNorm; with BN the two
    schedules will not match bit-for-bit because BN normalizes per-batch — an
    intrinsic property of grad accumulation, not a bug in our implementation.)
    """
    torch.manual_seed(1234)
    base = TinyHead(in_planes=3)
    big_batch = 8
    accum_n = 4
    micro_batch = big_batch // accum_n  # 2

    planes, pi, z = _make_fake_batch(in_planes=3, batch_size=big_batch, seed=42)

    # --- Path A: single optimizer step on the full batch ---
    model_a = copy.deepcopy(base)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01, momentum=0.0)
    train_step(
        model_a, opt_a, planes, pi, z,
        value_weight=1.0, l2_weight=0.0,  # disable L2 to keep math equivalent
        accum_scale=1.0, do_optimizer_step=True, zero_grad=True,
    )

    # --- Path B: N microbatches, accumulated, single optimizer step at end ---
    model_b = copy.deepcopy(base)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01, momentum=0.0)
    for i in range(accum_n):
        s = i * micro_batch
        e = s + micro_batch
        is_first = (i == 0)
        is_last = (i == accum_n - 1)
        train_step(
            model_b, opt_b, planes[s:e], pi[s:e], z[s:e],
            value_weight=1.0, l2_weight=0.0,
            accum_scale=1.0 / accum_n,
            do_optimizer_step=is_last,
            zero_grad=is_first,
        )

    # --- Compare final parameters ---
    max_abs_diff = 0.0
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        max_abs_diff = max(max_abs_diff, float((pa - pb).detach().abs().max()))
    # fp32 tolerance: tiny float-ordering discrepancy at sub-1e-6.
    assert max_abs_diff < 1e-5, f"params diverged: max |Δ| = {max_abs_diff}"


def test_grad_accumulation_n1_equals_legacy_train_step():
    """grad_accum_steps=1 is a no-op — defaults preserve legacy single-step behavior."""
    torch.manual_seed(7)
    base = TinyHead(in_planes=3)
    planes, pi, z = _make_fake_batch(in_planes=3, batch_size=4, seed=99)

    # --- Path A: legacy defaults (accum_scale=1, step + zero_grad both on) ---
    model_a = copy.deepcopy(base)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.05)
    train_step(model_a, opt_a, planes, pi, z, l2_weight=0.0)

    # --- Path B: explicit accum_n=1 (also a single step) ---
    model_b = copy.deepcopy(base)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.05)
    train_step(
        model_b, opt_b, planes, pi, z, l2_weight=0.0,
        accum_scale=1.0, do_optimizer_step=True, zero_grad=True,
    )

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb, atol=0.0), "n=1 path must be bit-identical to legacy"
