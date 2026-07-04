"""Tests for the swap2 v2a choice-head TRAINING loss (`gomoku.train.choice_step`).

v2a TRAINS the choice head into the loss with an OUTCOME-driven soft target (the
slot actually played, re-signed by the game outcome from the chooser's frame); it
does NOT change how the negotiator SELECTS (still the one-ply heuristic). These
tests pin: (i) one choice_step writes a real gradient into choice_fc; (ii) a plain
(non-swap2) train_step never touches the choice path (byte-identical-when-off);
(iii) the head overfits a fixed won-vs-lost batch — loss strictly decreases and the
argmax converges to the won slot; (iv) an always-illegal slot gets exactly zero
gradient (the legal mask is honored).

Tiny CPU model throughout (mirrors tests/test_model_choice_head.py).
"""

from __future__ import annotations

import numpy as np
import torch

from gomoku.game import BOARD_SIZE, N_INPUT_PLANES
from gomoku.model import build_model
from gomoku.swap2 import N_CHOICES
from gomoku.train import choice_step, train_step


def _planes(batch: int, *, seed: int = 0) -> torch.Tensor:
    """A batch of distinct random binary-ish planes (shape the net expects)."""
    rng = np.random.default_rng(seed)
    arr = rng.random((batch, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    return torch.from_numpy((arr > 0.7).astype(np.float32))


def _fresh_model_opt(lr: float = 0.05):
    torch.manual_seed(0)
    m = build_model("tiny")
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    return m, opt


def test_choice_step_writes_grad_into_choice_fc():
    m, opt = _fresh_model_opt()
    b = 8
    planes = _planes(b, seed=1)
    legal = torch.ones((b, N_CHOICES), dtype=torch.bool)
    chosen = torch.zeros((b,), dtype=torch.int64)
    chooser_z = torch.ones((b,), dtype=torch.float32)  # all won

    # Grad is populated by the backward INSIDE choice_step; capture it before the
    # next zero_grad would clear it by reading right after the call (choice_step
    # does opt.step() but leaves .grad populated).
    out = choice_step(m, opt, planes, legal, chosen, chooser_z, choice_weight=1.0)
    assert "loss/choice" in out and np.isfinite(out["loss/choice"])
    g = m.choice_fc.weight.grad
    assert g is not None, "choice_fc.weight.grad should be populated by choice_step"
    assert torch.any(g != 0), "choice_fc.weight.grad should be nonzero"


def test_non_swap2_train_step_leaves_choice_path_untouched():
    # A normal policy/value train_step must NEVER run the choice head — its grad
    # stays None (byte-identical-when-off guard).
    m, opt = _fresh_model_opt()
    b = 8
    planes = _planes(b, seed=2)
    pi = torch.full((b, BOARD_SIZE * BOARD_SIZE), 1.0 / (BOARD_SIZE * BOARD_SIZE))
    z = torch.zeros((b,), dtype=torch.float32)

    assert m.choice_fc.weight.grad is None
    train_step(m, opt, planes, pi, z, l2_weight=0.0)
    assert m.choice_fc.weight.grad is None, (
        "a non-swap2 train_step must not touch the choice head"
    )


def test_choice_head_overfits_won_vs_lost_batch():
    # A fixed 4-row batch: 2 WON rows (chooser_z=+1) prefer their chosen slot, 2
    # LOST rows (chooser_z=-1) push AWAY from theirs. The chosen slots are picked
    # so the supervised pressure has a consistent argmax winner per row. We assert
    # the loss is finite and strictly decreasing, and the head's argmax converges
    # to the WON rows' chosen slot.
    m, opt = _fresh_model_opt(lr=0.05)
    b = 4
    # Same planes per ROLE so the head can actually separate them: rows 0,1 share
    # one position (won, slot 0), rows 2,3 share another (lost, slot 1).
    base = _planes(2, seed=3)
    planes = torch.stack([base[0], base[0], base[1], base[1]])
    legal = torch.ones((b, N_CHOICES), dtype=torch.bool)
    chosen = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    chooser_z = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float32)

    losses = []
    for _ in range(60):
        out = choice_step(m, opt, planes, legal, chosen, chooser_z, choice_weight=1.0)
        losses.append(out["loss/choice"])

    assert all(np.isfinite(losses)), "choice loss went non-finite"
    # Strictly decreasing across the run (compare smoothed endpoints to avoid
    # Adam micro-noise tripping a step-by-step strict check).
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"
    first5 = float(np.mean(losses[:5]))
    last5 = float(np.mean(losses[-5:]))
    assert last5 < first5, f"loss not decreasing (first5={first5}, last5={last5})"

    # On the WON position (row 0), the head's argmax should land on slot 0.
    m.eval()
    with torch.no_grad():
        _, _, logits = m.forward_with_choice(planes)
    won_pred = int(logits[0].argmax())
    assert won_pred == 0, f"won-row argmax converged to slot {won_pred}, expected 0"


def test_illegal_slot_gets_zero_gradient():
    # Slot 2 is illegal for EVERY row, so its masked logit is a constant (min) and
    # its choice_fc row/bias must receive exactly zero gradient.
    m, opt = _fresh_model_opt()
    b = 6
    illegal = 2
    planes = _planes(b, seed=4)
    legal = torch.ones((b, N_CHOICES), dtype=torch.bool)
    legal[:, illegal] = False
    chosen = torch.zeros((b,), dtype=torch.int64)  # always a legal slot
    chooser_z = torch.ones((b,), dtype=torch.float32)

    choice_step(m, opt, planes, legal, chosen, chooser_z, choice_weight=1.0)
    wgrad = m.choice_fc.weight.grad
    bgrad = m.choice_fc.bias.grad
    assert wgrad is not None and bgrad is not None
    assert torch.all(wgrad[illegal] == 0.0), "illegal slot's weight grad must be zero"
    assert bgrad[illegal] == 0.0, "illegal slot's bias grad must be zero"
    # And a LEGAL slot must still receive nonzero gradient (sanity: loss is live).
    assert torch.any(wgrad[0] != 0.0), "legal slot should receive gradient"
