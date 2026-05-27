"""Tests for the Mish activation-function lever (bead derby-sib, cell derby-x-mish).

The lever swaps the residual-tower nonlinearity ReLU -> Mish via a new flag
--activation {relu,mish} (default relu = today, byte-identical). Mish is
nn.Mish(x) = x*tanh(softplus(x)). It is an ORTHOGONAL architecture axis: ZERO
added params, IDENTICAL state_dict keys (Mish/ReLU are parameter-free), so a
checkpoint of matching shape loads either way and only the activation math
changes. It lives ENTIRELY in model.py — the native-C MCTS engine does tree ops
and calls back into the PyTorch evaluator for the forward, so no C kernel changes.

The load-bearing tests:
  * test_relu_default_byte_identical_* — the default (--activation relu, and the
    build_model default) is byte-identical to today's exact tower graph: same
    state_dict keys, param count, AND a forward output equal to an independent
    functional-ReLU recomputation of the network — across global_pool in
    {None, True, 2}. The CRITICAL guard.
  * test_mish_forward_differs_and_modules_are_mish — --activation mish constructs
    nn.Mish in EVERY tower nonlinearity, the forward differs from relu, and the
    param count + state_dict KEYS are unchanged (Mish has no params).
  * test_mish_checkpoint_records_activation + test_*_consistency — the checkpoint
    config records `activation`; an old checkpoint without it loads as relu.

All CPU-only and tiny; the native MCTS ext is NOT required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from gomoku.game import BOARD_SIZE, N_INPUT_PLANES
from gomoku.model import (
    GlobalPoolResBlock,
    GomokuNet,
    ResBlock,
    build_model,
    load_checkpoint,
    make_activation,
    n_params,
    save_checkpoint,
)

DEV = torch.device("cpu")


def _x(b: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(b, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE, generator=g)


def _functional_relu_forward(m: GomokuNet, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent recomputation of the network forward using F.relu at every
    tower nonlinearity — i.e. EXACTLY the pre-lever functional graph. The model
    must be in eval() so BatchNorm uses running stats. We read the model's own
    fused/unfused submodules so this is a faithful re-implementation of the
    today's-graph forward, not the model's (module-activation) code path."""
    cfg = m.cfg
    # Stem: conv -> bn -> relu.
    h = m.stem[0](x)
    h = m.stem[1](h)
    h = F.relu(h)
    # Residual tower.
    for block in m.tower:
        if isinstance(block, GlobalPoolResBlock):
            hb = F.relu(block.bn1(block.conv1(h)))
            mean = hb.mean(dim=(2, 3))
            mx = hb.amax(dim=(2, 3))
            bias = block.pool_fc(torch.cat([mean, mx], dim=1))
            hb = hb + bias[:, :, None, None]
            hb = block.bn2(block.conv2(hb))
            h = F.relu(h + hb)
        else:
            hb = F.relu(block.bn1(block.conv1(h)))
            hb = block.bn2(block.conv2(hb))
            h = F.relu(h + hb)
    # Policy head (unchanged — F.relu by design).
    p = F.relu(m.policy_bn(m.policy_conv(h)))
    p = m.policy_fc(p.flatten(1))
    # Value head (scalar, unchanged).
    v = F.relu(m.value_bn(m.value_conv(h)))
    v = F.relu(m.value_fc1(v.flatten(1)))
    v = torch.tanh(m.value_fc2(v)).squeeze(-1)
    return p, v


# --------------------------------------------------------------------------
# relu default == byte-identical to today's exact tower graph. CRITICAL guard.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("global_pool", [None, True, 2])
def test_relu_default_byte_identical_state_dict_and_params(global_pool):
    """Default activation='relu' (and the build_model default) has the SAME
    state_dict keys + param count as an explicit relu build, across every
    global_pool setting. Activation modules add no parameters and no keys."""
    base = build_model("small", global_pool=global_pool)
    explicit = build_model("small", global_pool=global_pool, activation="relu")
    assert base.cfg.activation == "relu"
    assert set(base.state_dict().keys()) == set(explicit.state_dict().keys())
    assert n_params(base) == n_params(explicit)


@pytest.mark.parametrize("global_pool", [None, True, 2])
def test_relu_default_byte_identical_forward_output(global_pool):
    """The relu-default forward output is BITWISE-identical to an independent
    functional-ReLU recomputation of the network (the pre-lever graph) — proving
    the activation-module refactor does not perturb the today's-graph math.
    Checked across global_pool in {None, True, 2}."""
    torch.manual_seed(1)
    m = build_model("small", global_pool=global_pool)  # default relu
    m.eval()
    x = _x()
    with torch.no_grad():
        p0, v0 = m(x)
        p_ref, v_ref = _functional_relu_forward(m, x)
    assert torch.equal(p0, p_ref)
    assert torch.equal(v0, v_ref)
    assert v0.shape == (x.shape[0],)


def test_relu_default_state_dict_roundtrip_forward_identical():
    """Round-trip relu weights into a freshly-built relu model -> identical
    forward (the value_head test's analogue; weights load between builds)."""
    torch.manual_seed(2)
    m = build_model("small")
    m.eval()
    x = _x()
    with torch.no_grad():
        p0, v0 = m(x)
    m2 = build_model("small", activation="relu")
    m2.load_state_dict(m.state_dict())
    m2.eval()
    with torch.no_grad():
        p1, v1 = m2(x)
    assert torch.equal(p0, p1)
    assert torch.equal(v0, v1)


def test_relu_factory_is_inplace_relu():
    """make_activation('relu') is exactly nn.ReLU(inplace=True) — the module the
    stem's nn.ReLU(inplace=True) and the ResBlock F.relu calls are replaced by."""
    a = make_activation("relu")
    assert isinstance(a, nn.ReLU)
    assert a.inplace is True


# --------------------------------------------------------------------------
# Mish ON: nn.Mish in every tower nonlinearity, forward differs, no new params.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("global_pool", [None, True, 2])
def test_mish_same_params_and_keys_as_relu(global_pool):
    """activation='mish' keeps the SAME param count + state_dict KEYS as relu —
    Mish is parameter-free, so a checkpoint of matching shape loads either way."""
    relu_m = build_model("small", global_pool=global_pool)  # default relu
    mish_m = build_model("small", global_pool=global_pool, activation="mish")
    assert mish_m.cfg.activation == "mish"
    assert set(relu_m.state_dict().keys()) == set(mish_m.state_dict().keys())
    assert n_params(relu_m) == n_params(mish_m)


def test_mish_tower_modules_are_nn_mish():
    """EVERY residual-tower nonlinearity (stem + each ResBlock/GlobalPoolResBlock)
    is nn.Mish when activation='mish'; none is when relu. The policy/value HEAD
    nonlinearities are deliberately untouched (F.relu) — this is a tower lever."""
    m = build_model("small", global_pool=2, activation="mish")
    # Stem activation (index 2 of the Sequential: conv, bn, act).
    assert isinstance(m.stem[2], nn.Mish)
    # Both per-block activations on every tower block.
    n_block_acts = 0
    for block in m.tower:
        assert isinstance(block, (ResBlock, GlobalPoolResBlock))
        assert isinstance(block.act1, nn.Mish)
        assert isinstance(block.act2, nn.Mish)
        n_block_acts += 2
    assert n_block_acts == 2 * m.cfg.n_blocks
    # The relu build has ReLU modules in exactly the same spots (no Mish).
    r = build_model("small", global_pool=2)
    assert isinstance(r.stem[2], nn.ReLU)
    for block in r.tower:
        assert isinstance(block.act1, nn.ReLU)
        assert isinstance(block.act2, nn.ReLU)


@pytest.mark.parametrize("global_pool", [None, True])
def test_mish_forward_differs_from_relu_and_grad_flows(global_pool):
    """Mish forward output DIFFERS from relu (same weights, same input) and a
    backward pass produces finite grads through the Mish tower."""
    torch.manual_seed(3)
    relu_m = build_model("small", global_pool=global_pool)
    mish_m = build_model("small", global_pool=global_pool, activation="mish")
    # Same weights in both towers so the only difference is the activation.
    mish_m.load_state_dict(relu_m.state_dict())
    relu_m.eval()
    mish_m.eval()
    x = _x()
    with torch.no_grad():
        pr, vr = relu_m(x)
        pm, vm = mish_m(x)
    # Identical params + input, different nonlinearity -> different output.
    assert not torch.equal(pr, pm)
    assert not torch.equal(vr, vm)
    # Grad flows through the Mish tower.
    mish_m.train()
    p, v = mish_m(x)
    loss = p.float().pow(2).mean() + v.float().pow(2).mean()
    loss.backward()
    grads = [g.grad for g in mish_m.parameters() if g.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_mish_matches_reference_formula():
    """make_activation('mish') matches x*tanh(softplus(x)) on a sample tensor."""
    a = make_activation("mish")
    x = torch.linspace(-4.0, 4.0, 17)
    ref = x * torch.tanh(F.softplus(x))
    assert torch.allclose(a(x), ref, atol=1e-6)


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        make_activation("swish")
    with pytest.raises(ValueError):
        build_model("small", activation="gelu")


# --------------------------------------------------------------------------
# Checkpoint config records `activation`; consistency on load.
# --------------------------------------------------------------------------

def test_mish_checkpoint_records_activation_and_roundtrips(tmp_path):
    """A saved Mish model records activation='mish' in its config and reloads to
    an identical-forward Mish model (the path the worker/eval use)."""
    torch.manual_seed(5)
    m = build_model("small", activation="mish")
    m.eval()
    x = _x()
    with torch.no_grad():
        p0, v0 = m(x)
    ckpt = tmp_path / "mish.pt"
    save_checkpoint(str(ckpt), m, epoch=3, total_games=42)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert payload["model_config"]["activation"] == "mish"
    loaded, _ = load_checkpoint(str(ckpt), device="cpu")
    assert loaded.cfg.activation == "mish"
    loaded.eval()
    with torch.no_grad():
        p1, v1 = loaded(x)
    assert torch.equal(p0, p1)
    assert torch.equal(v0, v1)


def test_old_checkpoint_without_activation_loads_as_relu(tmp_path):
    """A pre-lever checkpoint config (no 'activation' key) loads as relu — the
    dataclass default — so existing relu checkpoints are unaffected."""
    torch.manual_seed(6)
    m = build_model("small")  # relu
    ckpt = tmp_path / "old.pt"
    save_checkpoint(str(ckpt), m, epoch=1)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    payload["model_config"].pop("activation", None)
    torch.save(payload, str(ckpt))
    loaded, _ = load_checkpoint(str(ckpt), device="cpu")
    assert loaded.cfg.activation == "relu"
