"""Teacher distillation mix-in in train_step: OFF byte-identical, ON learns."""

from __future__ import annotations

import torch

from gomoku.game import N_ACTIONS, N_INPUT_PLANES
from gomoku.model import build_model
from gomoku.train import train_step


def _fixed_batch(seed: int = 0, batch: int = 6):
    g = torch.Generator().manual_seed(seed)
    planes = torch.rand(batch, N_INPUT_PLANES, 9, 9, generator=g)
    z = torch.rand(batch, generator=g) * 2.0 - 1.0
    pi = torch.zeros(batch, N_ACTIONS)
    for b in range(batch):
        perm = torch.randperm(N_ACTIONS, generator=g)
        support = perm[:12]
        pi[b, support] = torch.softmax(torch.rand(12, generator=g) * 4.0, dim=-1)
    return planes, pi, z


def _teacher_batch(seed: int = 99, batch: int = 6):
    g = torch.Generator().manual_seed(seed)
    planes = torch.rand(batch, N_INPUT_PLANES, 9, 9, generator=g)
    moves = torch.randint(0, N_ACTIONS, (batch,), generator=g)
    tpi = torch.zeros(batch, N_ACTIONS)
    tpi[torch.arange(batch), moves] = 1.0
    return planes, tpi


# --- OFF is byte-identical -------------------------------------------------
def test_teacher_off_byte_identical():
    planes, pi, z = _fixed_batch(7)

    torch.manual_seed(2024)
    model_a = build_model("small").to("cpu")
    opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1, momentum=0.9)
    m_a = train_step(model_a, opt_a, planes, pi, z, l2_weight=1e-4)

    torch.manual_seed(2024)
    model_b = build_model("small").to("cpu")
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1, momentum=0.9)
    m_b = train_step(
        model_b, opt_b, planes, pi, z, l2_weight=1e-4,
        teacher_weight=0.0, teacher_planes=None, teacher_pi=None,
    )

    assert m_a["loss/total"] == m_b["loss/total"]
    assert m_a["loss/policy"] == m_b["loss/policy"]
    assert "loss/teacher" not in m_a
    assert "loss/teacher" not in m_b
    # Post-step params bit-identical.
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)


def test_teacher_weight_set_but_no_data_is_off():
    """teacher_weight > 0 but teacher_planes None -> guarded off (no extra forward)."""
    planes, pi, z = _fixed_batch(3)
    torch.manual_seed(1)
    model = build_model("small").to("cpu")
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    m = train_step(model, opt, planes, pi, z, teacher_weight=0.5, teacher_planes=None, teacher_pi=None)
    assert "loss/teacher" not in m


# --- ON emits the metric and actually learns the teacher move --------------
def test_teacher_on_emits_metric():
    planes, pi, z = _fixed_batch(5)
    tplanes, tpi = _teacher_batch()
    torch.manual_seed(1)
    model = build_model("small").to("cpu")
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    m = train_step(
        model, opt, planes, pi, z,
        teacher_planes=tplanes, teacher_pi=tpi, teacher_weight=0.5,
    )
    assert "loss/teacher" in m
    assert m["loss/teacher"] == m["loss/teacher"]  # not NaN


def test_teacher_distillation_reduces_teacher_ce():
    planes, pi, z = _fixed_batch(5)
    tplanes, tpi = _teacher_batch(seed=42, batch=4)
    torch.manual_seed(1)
    model = build_model("small").to("cpu")
    opt = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    first = train_step(
        model, opt, planes, pi, z,
        teacher_planes=tplanes, teacher_pi=tpi, teacher_weight=1.0,
    )["loss/teacher"]
    for _ in range(40):
        last = train_step(
            model, opt, planes, pi, z,
            teacher_planes=tplanes, teacher_pi=tpi, teacher_weight=1.0,
        )["loss/teacher"]
    # The model should fit the (fixed) teacher targets: CE drops substantially.
    assert last < first * 0.7
