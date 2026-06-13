"""Tests for the WDL (win/draw/loss) value-representation lever (bead derby-cgf).

The lever replaces the scalar tanh value head with a categorical {win,draw,loss}
head: the net emits 3 logits; everywhere the codebase needs a scalar value it
derives v = P(win) - P(loss). The value-discount and VCF-stamp targets are
re-expressed NATIVELY in WDL inside the trainer (a deterministic function of the
existing scalar z), so this stays ONE lever — the value REPRESENTATION.

The load-bearing tests:
  * test_scalar_off_byte_identical_* — the default (--value-head scalar) is
    byte-identical to a model predating the value_head field (same state_dict
    keys, param count, AND forward output). The critical guard.
  * test_wdl_forward_3_logits_and_derived_scalar — WDL forward returns the
    derived scalar v=P(win)-P(loss); return_value_logits exposes the 3 logits.
  * test_wdl_target_from_z_* — the WDL target math (one-hot, discount-toward-
    draw-vertex, VCF floor-0.90 stamp) on worked examples.
  * test_wdl_value_loss_is_cross_entropy — the WDL value loss is CE, not MSE.
  * test_wdl_checkpoint_roundtrip_derives_scalar — a saved WDL checkpoint loads
    and produces a scalar usable by the anchor-ladder eval.

All CPU-only and tiny; the native MCTS ext is not required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES, GameState
from gomoku.mcts import (
    MCTSGame,
    make_torch_evaluator,
    policy_from_visits,
    run_batched_mcts,
)
from gomoku.model import (
    GomokuNet,
    ModelConfig,
    build_model,
    fuse_model_for_inference,
    load_checkpoint,
    n_params,
    save_checkpoint,
)
from gomoku.train import train_step, wdl_target_from_z

DEV = torch.device("cpu")
GAMMA = 0.98


def _x(b: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(b, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE, generator=g)


# --------------------------------------------------------------------------
# Scalar OFF == byte-identical to the pre-WDL model. The critical guard.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("global_pool", [None, True, 2])
def test_scalar_off_byte_identical_state_dict_and_params(global_pool):
    """Default value_head='scalar' (and the build_model default) has the SAME
    state_dict keys + param count as a model whose ModelConfig has no WDL FC.
    The WDL FC must not exist; the scalar value_fc2 must exist."""
    base = build_model("small", global_pool=global_pool)
    explicit = build_model("small", global_pool=global_pool, value_head="scalar")
    assert base.cfg.value_head == "scalar"
    base_keys = set(base.state_dict().keys())
    explicit_keys = set(explicit.state_dict().keys())
    assert base_keys == explicit_keys
    assert n_params(base) == n_params(explicit)
    # The scalar head's final FC is present; the WDL FC is absent.
    assert any(k.startswith("value_fc2") for k in base_keys)
    assert not any(k.startswith("value_wdl_fc") for k in base_keys)


def test_scalar_off_byte_identical_forward_output():
    """A scalar model loaded from a pre-WDL state_dict produces the IDENTICAL
    forward output as the model it was saved from — proving the value_head field
    does not perturb the scalar inference graph."""
    torch.manual_seed(1)
    m = build_model("small")  # default scalar
    m.eval()
    x = _x()
    with torch.no_grad():
        p0, v0 = m(x)
    # Round-trip the weights into a freshly-built scalar model.
    m2 = build_model("small", value_head="scalar")
    m2.load_state_dict(m.state_dict())
    m2.eval()
    with torch.no_grad():
        p1, v1 = m2(x)
    assert torch.equal(p0, p1)
    assert torch.equal(v0, v1)
    assert v0.shape == (x.shape[0],)  # scalar value, one per sample


def test_scalar_off_default_two_tuple_and_no_wdl_logits():
    m = build_model("small")
    x = _x()
    out = m(x)
    assert len(out) == 2
    with pytest.raises(RuntimeError):
        m(x, return_value_logits=True)  # WDL FC not constructed


# --------------------------------------------------------------------------
# WDL forward: 3 logits + derived scalar v = P(win) - P(loss).
# --------------------------------------------------------------------------

def test_wdl_forward_3_logits_and_derived_scalar():
    m = build_model("small", value_head="wdl")
    assert m.cfg.value_head == "wdl"
    m.eval()
    x = _x()
    with torch.no_grad():
        p, v = m(x)
        p2, v2, logits = m(x, return_value_logits=True)
    assert v.shape == (x.shape[0],)
    assert logits.shape == (x.shape[0], 3)
    # The default-call scalar must equal P(win)-P(loss) from the 3 logits.
    wdl = torch.softmax(logits, dim=-1)
    derived = wdl[:, 0] - wdl[:, 2]
    assert torch.allclose(v, derived, atol=1e-6)
    assert torch.equal(v, v2)
    # Derived scalar is a valid value in [-1, 1].
    assert torch.all(v <= 1.0) and torch.all(v >= -1.0)


def test_wdl_has_wdl_fc_not_scalar_fc():
    m = build_model("small", value_head="wdl")
    keys = set(m.state_dict().keys())
    assert any(k.startswith("value_wdl_fc") for k in keys)
    assert not any(k.startswith("value_fc2") for k in keys)


def test_wdl_fuse_keeps_derived_scalar():
    """Conv/BN fusion (inference) is unchanged for WDL — forward still returns
    the derived scalar; the value_conv/value_bn fusion is shared with scalar."""
    m = build_model("small", value_head="wdl")
    m.eval()
    x = _x()
    with torch.no_grad():
        _, v_pre = m(x)
    fuse_model_for_inference(m)
    with torch.no_grad():
        out = m(x)
    assert len(out) == 2
    # Fusion is numerically equivalent for an eval model.
    assert torch.allclose(out[1], v_pre, atol=1e-5)


# --------------------------------------------------------------------------
# WDL target math: the EXACT generalization of the scalar targets.
# --------------------------------------------------------------------------

def test_wdl_target_one_hot_undiscounted_outcomes():
    """z in {+1, 0, -1} -> one-hot {(1,0,0),(0,1,0),(0,0,1)}."""
    z = torch.tensor([1.0, 0.0, -1.0])
    t = wdl_target_from_z(z)
    assert torch.allclose(t[0], torch.tensor([1.0, 0.0, 0.0]))  # win
    assert torch.allclose(t[1], torch.tensor([0.0, 1.0, 0.0]))  # draw
    assert torch.allclose(t[2], torch.tensor([0.0, 0.0, 1.0]))  # loss
    # Rows are proper distributions.
    assert torch.allclose(t.sum(dim=-1), torch.ones(3))


def test_wdl_target_value_discount_interpolates_toward_draw():
    """value-discount z = gamma^plies * sign maps to
    gamma^plies * onehot(sign) + (1-gamma^plies) * (0,1,0)."""
    plies = 5
    w = GAMMA ** plies  # 0.98^5 ~ 0.9039
    # A WIN this far from the end has scalar z = +w.
    z_win = torch.tensor([w])
    t_win = wdl_target_from_z(z_win)[0]
    assert torch.allclose(t_win, torch.tensor([w, 1.0 - w, 0.0]), atol=1e-6)
    # A LOSS: scalar z = -w -> w on the loss vertex, (1-w) on draw.
    z_loss = torch.tensor([-w])
    t_loss = wdl_target_from_z(z_loss)[0]
    assert torch.allclose(t_loss, torch.tensor([0.0, 1.0 - w, w]), atol=1e-6)


def test_wdl_target_vcf_stamp_floor_090():
    """A proven VCF win produces scalar z = f with f = max(0.90, gamma^(d-1)).
    The WDL target is f*(1,0,0) + (1-f)*(0,1,0) — win + residual draw mass."""
    # mate distance d=1 -> f = 1.0 (the immediate mate), then deeper, floored 0.90.
    for d in (1, 2, 6, 20):
        f = max(0.90, GAMMA ** max(0, d - 1))
        t = wdl_target_from_z(torch.tensor([f]))[0]
        assert torch.allclose(t, torch.tensor([f, 1.0 - f, 0.0]), atol=1e-6)
        # The floor: no VCF stamp ever puts <0.90 on the win vertex (float32 tol).
        assert t[0].item() >= 0.90 - 1e-5
        # No mass on the loss vertex for a proven win.
        assert t[2].item() == 0.0


# --------------------------------------------------------------------------
# WDL value loss is cross-entropy (not MSE) on the WDL path.
# --------------------------------------------------------------------------

def _batch(b: int = 4):
    torch.manual_seed(2)
    planes = _x(b)
    pi = torch.softmax(torch.randn(b, N_ACTIONS), dim=-1)
    z = torch.tensor([1.0, -1.0, 0.0, 0.5])[:b]
    return planes, pi, z


def test_wdl_value_loss_is_cross_entropy():
    """On the WDL path, loss/value equals the WDL cross-entropy, NOT the scalar
    MSE the same z would give under a scalar head."""
    m = build_model("small", value_head="wdl")
    opt = torch.optim.SGD(m.parameters(), lr=0.0)  # lr=0: weights frozen, metric only
    planes, pi, z = _batch()
    metrics = train_step(m, opt, planes, pi, z, l2_weight=0.0)
    # Recompute the expected CE from the model's own 3 logits. lr=0 left the
    # weights unchanged; recompute in the SAME train() mode train_step used so
    # the BatchNorm batch-stats path matches exactly (not eval running stats).
    m.train()
    with torch.no_grad():
        _, _, logits = m(planes, return_value_logits=True)
    logp = torch.log_softmax(logits, dim=-1)
    target = wdl_target_from_z(z)
    expected_ce = float(-(target * logp).sum(dim=-1).mean())
    assert metrics["loss/value"] == pytest.approx(expected_ce, abs=1e-5)
    # And it is NOT the scalar MSE the same z would have produced.
    with torch.no_grad():
        _, v = m(planes)
    mse = float(((v - z) ** 2).mean())
    assert not math.isclose(metrics["loss/value"], mse, abs_tol=1e-4)


def test_scalar_value_loss_still_mse():
    """The scalar path's loss/value is still the MSE — byte-identical loss."""
    m = build_model("small")  # scalar
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    planes, pi, z = _batch()
    metrics = train_step(m, opt, planes, pi, z, l2_weight=0.0)
    m.train()  # match train_step's BN mode (lr=0 left weights unchanged)
    with torch.no_grad():
        _, v = m(planes)
    expected_mse = float(((v - z) ** 2).mean())
    assert metrics["loss/value"] == pytest.approx(expected_mse, abs=1e-5)


def test_wdl_train_step_runs_and_lowers_value_loss():
    """A few real SGD steps on a fixed batch drive the WDL value loss down —
    proves the 3-logit head + CE target actually train end to end (CPU)."""
    m = build_model("small", value_head="wdl")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    planes, pi, z = _batch()
    first = train_step(m, opt, planes, pi, z, l2_weight=0.0)["loss/value"]
    last = first
    for _ in range(20):
        last = train_step(m, opt, planes, pi, z, l2_weight=0.0)["loss/value"]
    assert last < first


# --------------------------------------------------------------------------
# Checkpoint save/load of a WDL model derives a scalar for the anchor ladder.
# --------------------------------------------------------------------------

def test_wdl_checkpoint_roundtrip_derives_scalar(tmp_path):
    """Save a WDL model, reload via load_checkpoint (the path eval/the worker
    use), and confirm the reloaded model emits the SAME derived scalar value —
    so the anchor-ladder eval (which calls model(x) -> scalar) works on WDL."""
    torch.manual_seed(3)
    m = build_model("small", value_head="wdl")
    m.eval()
    x = _x()
    with torch.no_grad():
        _, v_orig = m(x)
    ckpt = tmp_path / "wdl.pt"
    save_checkpoint(str(ckpt), m, epoch=7, total_games=99)
    loaded, payload = load_checkpoint(str(ckpt), device="cpu")
    assert loaded.cfg.value_head == "wdl"
    assert payload["model_config"]["value_head"] == "wdl"
    loaded.eval()
    with torch.no_grad():
        out = loaded(x)
        assert len(out) == 2  # scalar-value contract for eval consumers
        _, v_loaded = out
    assert torch.allclose(v_orig, v_loaded, atol=1e-6)
    assert v_loaded.shape == (x.shape[0],)


def test_old_checkpoint_without_value_head_loads_as_scalar(tmp_path):
    """A pre-WDL checkpoint config (no 'value_head' key) loads as scalar — the
    dataclass default — so existing scalar checkpoints are unaffected."""
    torch.manual_seed(4)
    m = build_model("small")  # scalar
    ckpt = tmp_path / "old.pt"
    save_checkpoint(str(ckpt), m, epoch=1)
    # Simulate a pre-WDL payload: drop the value_head key from the saved config.
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    payload["model_config"].pop("value_head", None)
    torch.save(payload, str(ckpt))
    loaded, _ = load_checkpoint(str(ckpt), device="cpu")
    assert loaded.cfg.value_head == "scalar"
    x = _x()
    loaded.eval()
    with torch.no_grad():
        assert len(loaded(x)) == 2


# --------------------------------------------------------------------------
# MCTS consumes a WDL net's DERIVED scalar value transparently (python path).
# This is the load-bearing end-to-end gate: a WDL evaluator must plug into the
# same MCTS backup as a scalar evaluator, because make_torch_evaluator only ever
# sees model(x) -> (priors, scalar_value) — the 3 logits are scalarized to
# P(win)-P(loss) INSIDE forward(), upstream of the evaluator boundary. The
# native MCTS engine consumes that SAME scalar via evaluate_planes(), so it needs
# no WDL awareness (see the G15-wdl cell comment in scripts/run_sweep.py).
# --------------------------------------------------------------------------

def test_mcts_runs_with_wdl_evaluator_python_path():
    """A WDL net wrapped by make_torch_evaluator drives run_batched_mcts to a
    valid visit policy — proving the derived scalar feeds MCTS backup exactly
    like a scalar net. The evaluator returns (priors, values) where values are
    the WDL-derived P(win)-P(loss); MCTS never knows it is a WDL net."""
    torch.manual_seed(5)
    m = build_model("small", value_head="wdl")
    m.eval()
    ev = make_torch_evaluator(m, DEV)

    # The evaluator's value output is the derived scalar in [-1, 1], one per state.
    states = [GameState.initial(), GameState.initial().apply(BOARD_SIZE * BOARD_SIZE // 2)]
    priors, values = ev(states)
    assert priors.shape == (2, N_ACTIONS)
    assert values.shape == (2,)
    assert np.all(values <= 1.0 + 1e-5) and np.all(values >= -1.0 - 1e-5)
    # It must match the model's own derived scalar (the forward() P(W)-P(L)).
    import numpy as _np  # local alias to keep the assert self-contained
    x = _np.stack([s.to_planes() for s in states]).astype(_np.float32)
    with torch.no_grad():
        _, v_model = m(torch.from_numpy(x))
    assert _np.allclose(values, v_model.numpy(), atol=1e-5)

    # End to end: MCTS runs and produces a normalized policy over legal actions.
    games = [MCTSGame(GameState.initial()) for _ in range(3)]
    run_batched_mcts(games, ev, n_simulations=24, add_root_noise=False)
    for g in games:
        pi = policy_from_visits(g.root, temperature=1.0)
        assert pi.shape == (N_ACTIONS,)
        assert abs(float(pi.sum()) - 1.0) < 1e-4
        # All visit mass sits on legal actions only.
        assert float(pi[~g.root.legal_mask].sum()) == 0.0
        assert g.root.total_visits() >= 24
