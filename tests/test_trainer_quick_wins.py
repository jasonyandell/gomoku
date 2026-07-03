"""Equivalence tests for the #115 trainer quick-wins.

(a) `_l2_sum_of_squares` (fused torch._foreach_pow) must reproduce the old
    per-parameter Python loop: its VALUE to <=1e-6 relative, and — the part that
    actually drives training — the GRADIENT it feeds backward() BITWISE.
(b) train_step's batched per-bucket CE/MSE diagnostics (one packed host transfer)
    must produce the exact same keys and values as the old per-bucket
    `float(ce[mask].mean())` loop on a fixed batch, including omitting empty
    buckets.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F


def _old_l2_loop(model):
    return sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)


def test_l2_value_matches_old_loop():
    from gomoku.model import build_model
    from gomoku.train import _l2_sum_of_squares

    torch.manual_seed(0)
    model = build_model("small")
    ref = float(_old_l2_loop(model).detach())
    new = float(_l2_sum_of_squares(model).detach())
    rel = abs(new - ref) / max(abs(ref), 1e-12)
    assert rel <= 1e-6, f"L2 value rel_err {rel:.2e} exceeds 1e-6 (old={ref}, new={new})"


def test_l2_gradient_is_bitwise_identical():
    """The optimizer consumes the GRADIENT, not the loss value. The fused form
    must produce the identical 2*p per-parameter gradient as the old loop, or the
    (deliberate, on-top-of-weight_decay) L2 regularization semantics would drift.
    """
    from gomoku.model import build_model
    from gomoku.train import _l2_sum_of_squares

    torch.manual_seed(1)
    model = build_model("small")
    l2_weight = 1e-4

    def grads(fn):
        model.zero_grad(set_to_none=True)
        (l2_weight * fn(model)).backward()
        return [p.grad.detach().clone() for p in model.parameters() if p.requires_grad]

    ref = grads(_old_l2_loop)
    new = grads(_l2_sum_of_squares)
    for a, b in zip(ref, new):
        assert torch.equal(a, b), (
            f"L2 gradient differs by {float((a - b).abs().max()):.3e} (must be bitwise)"
        )


def _old_bucket_diagnostics(ce, ve, side, ply):
    """Reference: the pre-#115 per-bucket host-sync loop, verbatim."""
    ref: dict[str, float] = {}
    side_long = side.long()
    for s in (0, 1):
        mask = side_long == s
        if bool(mask.any()):
            ref[f"train/policy_ce/side_{s}"] = float(ce[mask].mean())
            ref[f"train/value_mse/side_{s}"] = float(ve[mask].mean())
    ply_long = ply.long()
    for lo, hi, label in ((0, 10, "ply_00_10"), (10, 25, "ply_10_25"), (25, 60, "ply_25_60")):
        mask = (ply_long >= lo) & (ply_long < hi)
        if bool(mask.any()):
            ref[f"train/policy_ce/{label}"] = float(ce[mask].mean())
            ref[f"train/value_mse/{label}"] = float(ve[mask].mean())
    return ref


def test_batched_diagnostics_match_old_loop_and_omit_empty_buckets():
    from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES
    from gomoku.model import build_model
    from gomoku.train import train_step

    rng = np.random.default_rng(7)
    model = build_model("tiny")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = 6
    planes = torch.from_numpy(
        rng.random((batch, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    )
    pi_np = rng.random((batch, N_ACTIONS)).astype(np.float32) + 1e-3
    pi_np /= pi_np.sum(axis=-1, keepdims=True)
    pi = torch.from_numpy(pi_np)
    z = torch.from_numpy((rng.random(batch) * 2.0 - 1.0).astype(np.float32))
    # side all 0 -> side_1 bucket empty; all ply < 25 -> ply_25_60 bucket empty.
    side = torch.zeros(batch, dtype=torch.int8)
    ply = torch.tensor([2, 12, 5, 8, 12, 2], dtype=torch.int16)

    # Snapshot pre-step weights so we can recompute the exact logits train_step
    # saw (its optimizer.step() mutates `model` in place afterward).
    model_copy = copy.deepcopy(model)
    out = train_step(model, optimizer, planes, pi, z, side=side, ply=ply, l2_weight=0.0)

    # Reference: same forward on pre-step weights, then the old bucket loop.
    model_copy.train()
    with torch.no_grad():
        logits, v = model_copy(planes)
        logp = F.log_softmax(logits, dim=-1)
        per_policy_ce = -(pi * logp).sum(dim=-1)
        per_value_se = (v - z) ** 2
    ref = _old_bucket_diagnostics(per_policy_ce, per_value_se, side, ply)

    bucket_keys = {
        k for k in list(out) if k.startswith(("train/policy_ce/", "train/value_mse/"))
    }
    # Same keys emitted (empty buckets omitted in both).
    assert bucket_keys == set(ref), f"key mismatch: got {bucket_keys}, ref {set(ref)}"
    for k in ("train/policy_ce/side_1", "train/value_mse/side_1",
              "train/policy_ce/ply_25_60", "train/value_mse/ply_25_60"):
        assert k not in out, f"empty bucket {k} should be omitted"
    # Same values — the batched path uses the identical masked_select().mean().
    for k, want in ref.items():
        assert out[k] == want, f"{k}: batched={out[k]!r} old={want!r}"
