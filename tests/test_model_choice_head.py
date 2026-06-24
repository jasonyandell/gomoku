"""Tests for the swap2 choice head on GomokuNet.

The choice head is a small width-N_CHOICES (=3) head off the value head's
penultimate hidden activation, reached ONLY via forward_with_choice(). The
existing forward(x) -> (policy, value) hot path and every `policy, value =
model(x)` caller must stay byte-identical, and load_checkpoint must warm-start an
OLD (pre-swap2) checkpoint that lacks the choice-head weights.
"""

import torch

from gomoku.game import GameState
from gomoku.model import build_model, load_checkpoint, save_checkpoint
from gomoku.swap2 import N_CHOICES


def _input(batch: int = 4) -> torch.Tensor:
    s = GameState.initial()
    x = torch.from_numpy(s.to_planes()).unsqueeze(0)
    return x.repeat(batch, 1, 1, 1)


def test_forward_with_choice_shapes():
    m = build_model("tiny")
    m.eval()
    x = _input(4)
    with torch.no_grad():
        p, v, c = m.forward_with_choice(x)
    assert p.shape == (4, 81)
    assert v.shape == (4,)
    assert c.shape == (4, N_CHOICES)
    assert c.shape[1] == 3
    assert torch.all(v >= -1) and torch.all(v <= 1)


def test_forward_still_two_tuple_unchanged_shapes():
    m = build_model("tiny")
    m.eval()
    x = _input(4)
    with torch.no_grad():
        out = m(x)
    # forward(x) must return exactly (policy, value) — a 2-tuple, no choice head.
    assert isinstance(out, tuple) and len(out) == 2
    p, v = out
    assert p.shape == (4, 81)
    assert v.shape == (4,)


def test_forward_and_forward_with_choice_agree_on_policy_value():
    # forward_with_choice must reproduce the same (policy, value) as forward.
    m = build_model("tiny")
    m.eval()
    x = _input(3)
    with torch.no_grad():
        p0, v0 = m(x)
        p1, v1, _ = m.forward_with_choice(x)
    assert torch.allclose(p0, p1, atol=1e-6)
    assert torch.allclose(v0, v1, atol=1e-6)


def test_warm_start_from_checkpoint_without_choice_head(tmp_path):
    # Simulate the champion: a checkpoint whose state_dict has NO choice-head
    # weights. load_checkpoint must succeed, the model is usable, and the choice
    # head is fresh-initialized (forward_with_choice works).
    m = build_model("tiny")
    sd = m.state_dict()
    choice_keys = [k for k in sd if k.startswith("choice_")]
    assert choice_keys, "expected choice-head keys to exist in the new model"
    for k in choice_keys:
        del sd[k]
    payload = {
        "model_state_dict": sd,
        "model_config": {**m.cfg.__dict__},
        "epoch": 0,
        "total_games": 0,
    }
    # Also strip choice_head from the config to fully simulate a pre-swap2 ckpt.
    payload["model_config"].pop("choice_head", None)
    p = tmp_path / "champion.pt"
    torch.save(payload, str(p))

    m2, _ = load_checkpoint(str(p))
    assert m2.cfg.choice_head is True
    m2.eval()
    x = _input(2)
    with torch.no_grad():
        _, _, c = m2.forward_with_choice(x)
    assert c.shape == (2, N_CHOICES)


def test_warm_start_does_not_mask_missing_core_weights(tmp_path):
    # Tolerance is choice-head-ONLY: a genuinely-missing CORE weight must still
    # raise, not be silently filled from init.
    m = build_model("tiny")
    sd = m.state_dict()
    del sd["policy_fc.weight"]  # a core weight
    payload = {
        "model_state_dict": sd,
        "model_config": {**m.cfg.__dict__},
    }
    p = tmp_path / "broken.pt"
    torch.save(payload, str(p))
    try:
        load_checkpoint(str(p))
    except (RuntimeError, KeyError):
        pass
    else:
        raise AssertionError("expected a missing core weight to raise on load")


def test_new_checkpoint_roundtrips_choice_head_bit_for_bit(tmp_path):
    # A NEW checkpoint that carries the choice head must round-trip the choice
    # logits exactly (the fallback must NOT trigger when the weights are present).
    torch.manual_seed(0)
    m = build_model("tiny")
    m.eval()
    x = _input(3)
    with torch.no_grad():
        _, _, c0 = m.forward_with_choice(x)

    p = tmp_path / "new.pt"
    save_checkpoint(str(p), m)
    m2, _ = load_checkpoint(str(p))
    m2.eval()
    with torch.no_grad():
        _, _, c1 = m2.forward_with_choice(x)
    assert torch.equal(c0, c1)
    # And the choice-head params are equal tensor-for-tensor.
    sd1, sd2 = m.state_dict(), m2.state_dict()
    for k in sd1:
        if k.startswith("choice_"):
            assert torch.equal(sd1[k], sd2[k])


def test_choice_logits_deterministic_in_eval(tmp_path):
    m = build_model("tiny")
    m.eval()
    x = _input(2)
    with torch.no_grad():
        _, _, c_a = m.forward_with_choice(x)
        _, _, c_b = m.forward_with_choice(x)
    assert torch.equal(c_a, c_b)
