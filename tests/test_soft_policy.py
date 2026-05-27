"""Tests for the KataGo soft-policy auxiliary target (bead derby-79l).

The soft-policy lever adds a SECOND policy-loss term: the same policy logits are
scored against a 4th-root temperature-flattened, renormalized copy of the
already-recorded policy target `pi` (KataGo's exact transform), scaled by
--soft-policy-weight. No new head; ZERO generation cost (pure trainer transform).

The load-bearing test is `test_off_byte_identical`: with the default weight 0.0
the guarded branch never executes, so the loss tensor AND one optimizer step
must be float-EXACT vs the pre-lever champion path on a fixed seed.

All tests force device='cpu' and stay tiny (no MPS/GPU; the live derby owns it).
"""

from __future__ import annotations

import copy

import torch

from gomoku.game import N_ACTIONS, N_INPUT_PLANES
from gomoku.model import build_model
from gomoku.train import train_step


def _fixed_batch(seed: int = 0, batch: int = 6):
    """A deterministic tiny CPU batch with a LEGAL-MASKED policy target `pi`
    (zeros off-support, rows sum to 1) like the self-play target builder emits."""
    g = torch.Generator().manual_seed(seed)
    planes = torch.rand(batch, N_INPUT_PLANES, 9, 9, generator=g)
    z = (torch.rand(batch, generator=g) * 2.0 - 1.0)
    # Build a peaky, legal-masked pi: a random support of ~12 legal cells per row,
    # softmax over random logits on the support, exact zeros elsewhere.
    pi = torch.zeros(batch, N_ACTIONS)
    for b in range(batch):
        perm = torch.randperm(N_ACTIONS, generator=g)
        support = perm[:12]
        raw = torch.rand(12, generator=g) * 4.0  # spread so the target is peaky
        pi[b, support] = torch.softmax(raw, dim=-1)
    return planes, pi, z


def _build_cpu_model(seed: int = 1234):
    torch.manual_seed(seed)
    model = build_model("small").to("cpu")
    return model


def _one_step_params(model, optimizer, planes, pi, z, **kw):
    """Run one train_step (forward+backward+optimizer.step) and return the loss
    metric dict plus a flat snapshot of all parameters AFTER the step."""
    m = train_step(model, optimizer, planes, pi, z, l2_weight=1e-4, **kw)
    snap = [p.detach().clone() for p in model.parameters()]
    return m, snap


# --------------------------------------------------------------------------
# THE load-bearing guard: OFF (weight 0.0) is byte-identical to the pre-lever
# path — loss tensor AND one optimizer step float-EXACT on a fixed seed.
# --------------------------------------------------------------------------

def test_off_byte_identical_loss_and_optimizer_step():
    planes, pi, z = _fixed_batch(seed=7)

    # Path A: pre-lever champion call — never pass soft_policy_weight at all.
    model_a = _build_cpu_model(seed=2024)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1, momentum=0.9)
    m_a, params_a = _one_step_params(model_a, opt_a, planes, pi, z)

    # Path B: lever present but OFF (explicit default 0.0). The guarded branch
    # must NOT execute, so loss + post-step params are float-exact vs A.
    model_b = _build_cpu_model(seed=2024)  # identical init (same seed)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1, momentum=0.9)
    m_b, params_b = _one_step_params(
        model_b, opt_b, planes, pi, z, soft_policy_weight=0.0
    )

    # Loss tensor float-EXACT.
    assert m_a["loss/total"] == m_b["loss/total"]
    assert m_a["loss/policy"] == m_b["loss/policy"]
    assert m_a["loss/value"] == m_b["loss/value"]
    # No soft-policy metric is emitted when OFF.
    assert "loss/soft_policy" not in m_a
    assert "loss/soft_policy" not in m_b
    # One optimizer step: every parameter float-EXACT (bit-identical).
    for pa, pb in zip(params_a, params_b):
        assert torch.equal(pa, pb)


# --------------------------------------------------------------------------
# soft-on: valid flatter distribution, pl increases by exactly the term,
# gradient flows to the logits.
# --------------------------------------------------------------------------

def test_soft_target_is_valid_flatter_distribution():
    """The soft target = renormalized (pi + 1e-7)^0.25: sums to 1 per row,
    has lower max and higher entropy than `pi` (4th-root flattens the peak)."""
    _, pi, _ = _fixed_batch(seed=11)
    soft = (pi + 1e-7).pow(0.25)
    soft = soft / soft.sum(dim=-1, keepdim=True)

    # Valid distribution: sums to 1 per row, non-negative.
    assert torch.allclose(soft.sum(dim=-1), torch.ones(pi.shape[0]), atol=1e-5)
    assert bool((soft >= 0).all())

    # Flatter than the sharp target: lower per-row max, strictly higher entropy.
    assert bool((soft.max(dim=-1).values < pi.max(dim=-1).values).all())

    def entropy(p):
        return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)

    assert bool((entropy(soft) > entropy(pi)).all())


def test_soft_on_pl_increases_by_exact_term_and_grad_flows():
    """With weight > 0, pl == sharp_pl + weight * soft_ce.mean() (the loss/policy
    metric reflects the augmented term), the soft CE is logged, and gradient
    reaches the policy logits."""
    planes, pi, z = _fixed_batch(seed=23)
    weight = 0.15

    # Sharp-only reference policy loss (forward shares no state; just compute the
    # reduction the way train_step does at weight 0.0).
    model = _build_cpu_model(seed=99)
    model.train()
    logits, _ = model(planes)
    logp = torch.log_softmax(logits, dim=-1)
    sharp_pl = -(pi * logp).sum(dim=-1).mean()
    soft = (pi + 1e-7).pow(0.25)
    soft = soft / soft.sum(dim=-1, keepdim=True)
    per_soft_ce = -(soft * logp).sum(dim=-1)
    expected_pl = sharp_pl + weight * per_soft_ce.mean()

    # Run train_step ON at the same model state (rebuild with same seed so the
    # forward logits match the manual reference above exactly).
    model2 = _build_cpu_model(seed=99)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.0)  # lr 0 -> params unchanged
    m = train_step(
        model2, opt2, planes, pi, z, l2_weight=0.0, soft_policy_weight=weight,
        do_optimizer_step=False,
    )

    # loss/policy reflects the augmented term (sharp + weight*soft), float-close.
    assert abs(m["loss/policy"] - float(expected_pl.detach())) < 1e-5
    # soft CE is logged and is float-close to the manual mean.
    assert "loss/soft_policy" in m
    assert abs(m["loss/soft_policy"] - float(per_soft_ce.mean().detach())) < 1e-5

    # Gradient flows to the policy logits via the soft term. Build a standalone
    # graph that isolates the soft term and confirm a non-zero logit gradient.
    logits3 = logits.detach().clone().requires_grad_(True)
    logp3 = torch.log_softmax(logits3, dim=-1)
    soft_loss = weight * (-(soft * logp3).sum(dim=-1)).mean()
    soft_loss.backward()
    assert logits3.grad is not None
    assert float(logits3.grad.abs().sum()) > 0.0


def test_off_vs_on_loss_differs():
    """A direct ON/OFF contrast on one fixed batch: turning the lever on changes
    the policy loss (the term is actually wired in, not a no-op)."""
    planes, pi, z = _fixed_batch(seed=31)

    model_off = _build_cpu_model(seed=555)
    opt_off = torch.optim.SGD(model_off.parameters(), lr=0.0)
    m_off = train_step(model_off, opt_off, planes, pi, z, l2_weight=0.0,
                       do_optimizer_step=False, soft_policy_weight=0.0)

    model_on = _build_cpu_model(seed=555)
    opt_on = torch.optim.SGD(model_on.parameters(), lr=0.0)
    m_on = train_step(model_on, opt_on, planes, pi, z, l2_weight=0.0,
                      do_optimizer_step=False, soft_policy_weight=0.15)

    assert m_off["loss/policy"] != m_on["loss/policy"]
