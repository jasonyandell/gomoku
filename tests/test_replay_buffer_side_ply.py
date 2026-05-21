"""Side+ply tagging on the replay buffer: add/sample roundtrip + backward-compat."""

from __future__ import annotations

import numpy as np
import torch

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import SelfPlayExample


def _fake_example(rng: np.random.Generator, side: int, ply: int) -> SelfPlayExample:
    planes = rng.random((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    pi = rng.random(N_ACTIONS).astype(np.float32)
    pi /= pi.sum()
    z = float(rng.random() * 2.0 - 1.0)
    return SelfPlayExample(planes, pi, z, side=side, ply=ply)


def test_add_sample_returns_side_and_ply():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=32, device="cpu")
    examples = [_fake_example(rng, side=i % 2, ply=i * 3) for i in range(16)]
    buf.add(examples)
    planes, pi, z, side, ply = buf.sample(8)
    assert planes.shape == (8, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert pi.shape == (8, N_ACTIONS)
    assert z.shape == (8,)
    assert side.shape == (8,)
    assert ply.shape == (8,)
    assert side.dtype == torch.int8
    assert ply.dtype == torch.int16


def test_side_ply_persist_through_state_dict_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    buf = ReplayBuffer(capacity=64, device="cpu")
    examples = [_fake_example(rng, side=i % 2, ply=i + 1) for i in range(40)]
    buf.add(examples)
    sd = buf.state_dict()
    path = tmp_path / "buf.pt"
    torch.save(sd, path)

    sd_loaded = torch.load(path, map_location="cpu", weights_only=False)
    buf2 = ReplayBuffer(capacity=64, device="cpu")
    buf2.load_state_dict(sd_loaded)
    assert buf2.size == 40
    assert torch.equal(buf2.side[:40], buf.side[:40])
    assert torch.equal(buf2.ply[:40], buf.ply[:40])


def test_backward_compat_missing_side_ply_zero_fills(capsys):
    """A 'fake old' state_dict (no side/ply keys) loads cleanly with zeros."""
    rng = np.random.default_rng(2)
    n = 20
    planes = torch.from_numpy(
        rng.random((n, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    )
    pi = torch.from_numpy(rng.random((n, N_ACTIONS)).astype(np.float32))
    pi = pi / pi.sum(dim=-1, keepdim=True)
    z = torch.from_numpy((rng.random(n) * 2.0 - 1.0).astype(np.float32))
    weight_version = torch.zeros((n,), dtype=torch.int64)
    sd_old = {
        "capacity": 64,
        "head": n,
        "size": n,
        "planes": planes,
        "pi": pi,
        "z": z,
        "weight_version": weight_version,
        "current_weight_version": 0,
    }
    buf = ReplayBuffer(capacity=64, device="cpu")
    buf.load_state_dict(sd_old)
    assert buf.size == n
    assert torch.all(buf.side[:n] == 0)
    assert torch.all(buf.ply[:n] == 0)
    captured = capsys.readouterr()
    assert "side tag missing" in captured.out
    assert "ply tag missing" in captured.out


def test_examples_without_side_ply_default_to_zero():
    """SelfPlayExample default values keep callers that don't set side/ply working."""
    rng = np.random.default_rng(3)
    planes = rng.random((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    pi = rng.random(N_ACTIONS).astype(np.float32)
    pi /= pi.sum()
    ex = SelfPlayExample(planes, pi, 0.5)
    assert ex.side == 0
    assert ex.ply == 0
    buf = ReplayBuffer(capacity=8, device="cpu")
    buf.add([ex])
    _, _, _, side, ply = buf.sample(1)
    assert int(side[0]) == 0
    assert int(ply[0]) == 0
