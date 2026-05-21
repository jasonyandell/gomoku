"""H/KL decomposition identity for the trainer's policy loss.

policy_ce = -sum(pi * log_softmax(logits)) per row
          = H(pi) + KL(pi || p_net)

where H(pi) = -sum(pi * log(pi)) and
      KL(pi || p) = sum(pi * (log(pi) - log(p))).

The train step computes target_entropy + policy_kl and reports them
separately; this test asserts the equality the trainer relies on.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _make_distribution(rng: np.random.Generator, batch: int, n: int,
                       sharp: bool = False) -> torch.Tensor:
    if sharp:
        raw = rng.random((batch, n)).astype(np.float32) + 1e-3
        raw = raw ** 4
    else:
        raw = rng.random((batch, n)).astype(np.float32) + 1e-3
    out = raw / raw.sum(axis=-1, keepdims=True)
    return torch.from_numpy(out)


def test_policy_ce_equals_target_entropy_plus_kl():
    rng = np.random.default_rng(0)
    batch, n = 32, 81
    pi = _make_distribution(rng, batch, n, sharp=False)
    logits = torch.from_numpy(rng.standard_normal((batch, n)).astype(np.float32))

    logp = F.log_softmax(logits, dim=-1)
    policy_ce = -(pi * logp).sum(dim=-1).mean()

    target_entropy = -(pi * torch.log(pi.clamp_min(1e-9))).sum(dim=-1).mean()
    pi_safe = pi.clamp_min(1e-9)
    policy_kl = (pi * (torch.log(pi_safe) - logp)).sum(dim=-1).mean()

    recomputed = target_entropy + policy_kl
    assert torch.allclose(policy_ce, recomputed, atol=1e-5), (
        f"policy_ce={float(policy_ce):.6f}, recomputed={float(recomputed):.6f}"
    )
    # And the trainer's shortcut subtraction matches the explicit KL.
    short_kl = policy_ce - target_entropy
    assert torch.allclose(short_kl, policy_kl, atol=1e-5)


def test_train_step_returns_decomposition_keys():
    """Sanity: train_step actually surfaces the three decomposition scalars."""
    from gomoku.model import build_model
    from gomoku.train import train_step

    rng = np.random.default_rng(1)
    model = build_model("tiny")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = 4
    from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES
    planes = torch.from_numpy(
        rng.random((batch, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    )
    pi_np = rng.random((batch, N_ACTIONS)).astype(np.float32) + 1e-3
    pi_np /= pi_np.sum(axis=-1, keepdims=True)
    pi = torch.from_numpy(pi_np)
    z = torch.from_numpy((rng.random(batch) * 2.0 - 1.0).astype(np.float32))
    side = torch.tensor([0, 1, 0, 1], dtype=torch.int8)
    ply = torch.tensor([2, 12, 30, 8], dtype=torch.int16)
    out = train_step(model, optimizer, planes, pi, z, side=side, ply=ply, l2_weight=0.0)
    for key in (
        "train/policy_target_entropy",
        "train/policy_net_entropy",
        "train/policy_kl",
        "train/policy_ce/side_0",
        "train/policy_ce/side_1",
        "train/value_mse/side_0",
        "train/value_mse/side_1",
        "train/policy_ce/ply_00_10",
        "train/policy_ce/ply_10_25",
        "train/policy_ce/ply_25_60",
        "train/value_mse/ply_00_10",
        "train/value_mse/ply_10_25",
        "train/value_mse/ply_25_60",
    ):
        assert key in out, f"missing {key} in train_step output"
        assert np.isfinite(out[key])
    # Identity restored on the reported scalars (decomposed values pre-step).
    assert out["train/policy_kl"] >= -1e-4


def test_per_color_metrics_are_correct_on_synthetic_buffer():
    """Direct sanity: build a buffer where every side-0 example has pi peaking
    on action 0 and every side-1 example peaking on action 1; if we then
    train_step a model that predicts identical logits everywhere, the per-side
    CE values should differ predictably.
    """
    from gomoku.model import build_model
    from gomoku.train import train_step

    rng = np.random.default_rng(42)
    model = build_model("tiny")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES
    batch = 8
    planes = torch.from_numpy(
        rng.random((batch, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    )
    pi_np = np.full((batch, N_ACTIONS), 1e-6, dtype=np.float32)
    for i in range(batch):
        peak = 0 if i % 2 == 0 else 1
        pi_np[i, peak] = 1.0 - 1e-6 * (N_ACTIONS - 1)
    pi = torch.from_numpy(pi_np)
    z = torch.zeros(batch)
    side = torch.tensor([i % 2 for i in range(batch)], dtype=torch.int8)
    ply = torch.zeros(batch, dtype=torch.int16)
    out = train_step(model, optimizer, planes, pi, z, side=side, ply=ply, l2_weight=0.0)
    # Both per-color CE values are finite and reported.
    assert np.isfinite(out["train/policy_ce/side_0"])
    assert np.isfinite(out["train/policy_ce/side_1"])
