"""Cheap-test gate for the bit-packed replay buffer (issue #25).

Per wiki/topics/buffer-bit-packing.md, before/with the refactor we assert:
  1. pack/unpack round-trip fidelity at the active board size (run at 9 AND 15),
  2. add/sample correctness under packing == the unpacked path (the net cannot
     tell — identical planes/pi/z/side/ply),
  3. memory reduction (packed bytes/pos << float32 bytes/pos),
  4. OFF is byte-identical (default buffer schema/values unchanged, and a packed
     checkpoint cross-loads into a float32 buffer reproducing the exact planes).

Run both sizes:
    PYTHONPATH=$PWD pytest tests/test_bitpack_buffer.py -q
    GOMOKU_BOARD_SIZE=15 PYTHONPATH=$PWD pytest tests/test_bitpack_buffer.py -q
"""

from __future__ import annotations

import numpy as np
import torch

from gomoku.game import BOARD_SIZE, GameState, N_ACTIONS, N_INPUT_PLANES
from gomoku.plane_packing import packed_bytes_per_position, pack_planes, unpack_planes
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import SelfPlayExample

PLANE_SHAPE = (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
PLANE_ELEMS = N_INPUT_PLANES * BOARD_SIZE * BOARD_SIZE


def _random_binary_planes(rng: np.random.Generator, n: int) -> np.ndarray:
    """A batch of binary occupancy-style planes (0/1), like production data."""
    return (rng.random((n, *PLANE_SHAPE)) < 0.3).astype(np.float32)


def _real_game_planes(seed: int, n_moves: int) -> np.ndarray:
    """Planes from an actual played-out game — genuinely binary + constant plane,
    exactly the production encoding (State.to_planes)."""
    rng = np.random.default_rng(seed)
    st = GameState.initial()
    out = [st.to_planes()]
    for _ in range(n_moves):
        legal = st.legal_actions()
        if legal.size == 0:
            break
        st = st.apply(int(rng.choice(legal)))
        out.append(st.to_planes())
        done, _ = st.is_terminal()
        if done:
            break
    return np.stack(out)


def _example(planes: np.ndarray, rng: np.random.Generator, side: int, ply: int) -> SelfPlayExample:
    pi = rng.random(N_ACTIONS).astype(np.float32)
    pi /= pi.sum()
    z = float(rng.random() * 2.0 - 1.0)
    return SelfPlayExample(planes.astype(np.float32), pi, z, side=side, ply=ply)


# ---------------------------------------------------------------------------
# 1. pack/unpack round-trip fidelity (exact) — at the active board size.
# ---------------------------------------------------------------------------

def test_roundtrip_exact_random_binary():
    rng = np.random.default_rng(0)
    planes = _random_binary_planes(rng, 37)
    packed = pack_planes(planes, PLANE_ELEMS)
    restored = unpack_planes(packed, PLANE_ELEMS, PLANE_SHAPE)
    assert restored.shape == planes.shape
    assert np.array_equal(restored, planes)  # bit-exact, not approximate


def test_roundtrip_exact_real_game():
    planes = _real_game_planes(seed=7, n_moves=40)
    packed = pack_planes(planes, PLANE_ELEMS)
    restored = unpack_planes(packed, PLANE_ELEMS, PLANE_SHAPE)
    assert np.array_equal(restored, planes)
    # The production constant plane (all ones) must survive intact.
    assert np.all(restored[:, N_INPUT_PLANES - 1] == 1.0)


def test_roundtrip_edges_all_zero_all_one():
    z = np.zeros((3, *PLANE_SHAPE), dtype=np.float32)
    o = np.ones((3, *PLANE_SHAPE), dtype=np.float32)
    assert np.array_equal(unpack_planes(pack_planes(z, PLANE_ELEMS), PLANE_ELEMS, PLANE_SHAPE), z)
    assert np.array_equal(unpack_planes(pack_planes(o, PLANE_ELEMS), PLANE_ELEMS, PLANE_SHAPE), o)


def test_pack_rejects_nonbinary():
    bad = np.full((1, *PLANE_SHAPE), 0.5, dtype=np.float32)
    try:
        pack_planes(bad, PLANE_ELEMS)
    except ValueError:
        return
    raise AssertionError("pack_planes must reject non-binary planes")


# ---------------------------------------------------------------------------
# 2. add/sample correctness under packing == the unpacked path.
# ---------------------------------------------------------------------------

def test_packed_sample_identical_to_float32():
    """Same examples into a packed buffer and a float32 buffer; sampling the SAME
    indices must yield byte-identical planes (and identical pi/z/side/ply). The
    net must not be able to tell which storage mode produced the batch."""
    rng = np.random.default_rng(11)
    cap = 64
    n = 50
    planes_b = _random_binary_planes(rng, n)
    examples = [_example(planes_b[i], rng, side=i % 2, ply=i) for i in range(n)]

    buf_f32 = ReplayBuffer(capacity=cap, device="cpu", pack_planes=False)
    buf_pak = ReplayBuffer(capacity=cap, device="cpu", pack_planes=True)
    buf_f32.add(examples)
    buf_pak.add(examples)

    assert buf_pak.planes is None and buf_pak.packed_planes is not None
    assert buf_f32.packed_planes is None and buf_f32.planes is not None

    # Compare the full live region directly (deterministic, index-aligned).
    idx = torch.arange(n)
    p_f32 = buf_f32._planes_at(idx)
    p_pak = buf_pak._planes_at(idx)
    assert torch.equal(p_f32, p_pak)
    assert torch.equal(buf_f32.pi[:n], buf_pak.pi[:n])
    assert torch.equal(buf_f32.z[:n], buf_pak.z[:n])
    assert torch.equal(buf_f32.side[:n], buf_pak.side[:n])
    assert torch.equal(buf_f32.ply[:n], buf_pak.ply[:n])

    # sample() returns the right shapes/dtypes under packing.
    s_planes, s_pi, s_z, s_side, s_ply = buf_pak.sample(8)
    assert s_planes.shape == (8, *PLANE_SHAPE)
    assert s_planes.dtype == torch.float32
    assert s_pi.shape == (8, N_ACTIONS)


def test_packed_ring_wraparound_matches_float32():
    """Overfill past capacity: the ring's surviving rows must match between modes."""
    rng = np.random.default_rng(13)
    cap = 16
    n = 40  # forces ~2.5 wraps
    planes_b = _random_binary_planes(rng, n)
    examples = [_example(planes_b[i], rng, side=i % 2, ply=i) for i in range(n)]
    buf_f32 = ReplayBuffer(capacity=cap, device="cpu", pack_planes=False)
    buf_pak = ReplayBuffer(capacity=cap, device="cpu", pack_planes=True)
    buf_f32.add(examples)
    buf_pak.add(examples)
    assert buf_f32.size == cap and buf_pak.size == cap
    assert buf_f32.head == buf_pak.head
    idx = torch.arange(cap)
    assert torch.equal(buf_f32._planes_at(idx), buf_pak._planes_at(idx))
    assert torch.equal(buf_f32.pi[:cap], buf_pak.pi[:cap])


# ---------------------------------------------------------------------------
# 3. memory reduction.
# ---------------------------------------------------------------------------

def test_packed_bytes_much_smaller_than_f32():
    packed = packed_bytes_per_position(PLANE_ELEMS)
    f32 = PLANE_ELEMS * 4
    assert packed == (PLANE_ELEMS + 7) // 8
    # 8 cells/byte vs 4 bytes/cell => ~32x. Assert a strong margin (>=16x) so the
    # gate would catch a regression to int8/bool storage.
    assert f32 >= 16 * packed, f"packed {packed}B vs f32 {f32}B is only {f32/packed:.1f}x"


def test_buffer_packed_store_smaller_than_f32_store():
    cap = 1000
    buf_f32 = ReplayBuffer(capacity=cap, device="cpu", pack_planes=False)
    buf_pak = ReplayBuffer(capacity=cap, device="cpu", pack_planes=True)
    f32_bytes = buf_f32.planes.numel() * buf_f32.planes.element_size()
    pak_bytes = buf_pak.packed_planes.numel() * buf_pak.packed_planes.element_size()
    assert pak_bytes * 16 <= f32_bytes


# ---------------------------------------------------------------------------
# 4. OFF is byte-identical + cross-mode checkpoint compatibility.
# ---------------------------------------------------------------------------

def test_off_state_dict_schema_is_byte_identical():
    """An OFF (default) buffer's state_dict must carry the pre-#25 'planes' key,
    a float32 tensor, and NO packing marker / packed_planes key."""
    rng = np.random.default_rng(21)
    planes_b = _random_binary_planes(rng, 12)
    examples = [_example(planes_b[i], rng, side=0, ply=i) for i in range(12)]
    buf = ReplayBuffer(capacity=32, device="cpu")  # default => pack_planes False
    assert buf.pack_planes is False
    buf.add(examples)
    sd = buf.state_dict()
    assert "planes" in sd and sd["planes"].dtype == torch.float32
    assert "packed_planes" not in sd
    assert "pack_planes" not in sd


def test_off_buffer_planes_bit_identical_to_examples():
    """OFF buffer stores planes verbatim (float32, no transform)."""
    rng = np.random.default_rng(22)
    planes_b = _random_binary_planes(rng, 10)
    examples = [_example(planes_b[i], rng, side=0, ply=i) for i in range(10)]
    buf = ReplayBuffer(capacity=16, device="cpu")
    buf.add(examples)
    assert torch.equal(buf.planes[:10], torch.from_numpy(planes_b))


def test_packed_checkpoint_loads_into_f32_buffer_exact(tmp_path):
    """A packed buffer's checkpoint must restore EXACTLY into a default (OFF)
    float32 buffer — the cross-mode unpack path, bit-identical planes."""
    rng = np.random.default_rng(23)
    planes_b = _random_binary_planes(rng, 30)
    examples = [_example(planes_b[i], rng, side=i % 2, ply=i) for i in range(30)]
    src = ReplayBuffer(capacity=64, device="cpu", pack_planes=True)
    src.add(examples)
    sd = src.state_dict()
    path = tmp_path / "packed.pt"
    torch.save(sd, path)
    sd2 = torch.load(path, map_location="cpu", weights_only=False)

    dst_f32 = ReplayBuffer(capacity=64, device="cpu", pack_planes=False)
    dst_f32.load_state_dict(sd2)
    assert dst_f32.size == 30
    assert torch.equal(dst_f32.planes[:30], torch.from_numpy(planes_b))
    assert torch.equal(dst_f32.pi[:30], src.pi[:30])
    assert torch.equal(dst_f32.z[:30], src.z[:30])


def test_f32_checkpoint_loads_into_packed_buffer_exact(tmp_path):
    """An old-style float32 checkpoint must restore EXACTLY into a packed buffer
    (pack-on-load); planes must round-trip bit-for-bit."""
    rng = np.random.default_rng(24)
    planes_b = _random_binary_planes(rng, 30)
    examples = [_example(planes_b[i], rng, side=i % 2, ply=i) for i in range(30)]
    src = ReplayBuffer(capacity=64, device="cpu", pack_planes=False)
    src.add(examples)
    sd = src.state_dict()  # carries "planes" float32
    path = tmp_path / "f32.pt"
    torch.save(sd, path)
    sd2 = torch.load(path, map_location="cpu", weights_only=False)

    dst_pak = ReplayBuffer(capacity=64, device="cpu", pack_planes=True)
    dst_pak.load_state_dict(sd2)
    assert dst_pak.size == 30
    assert torch.equal(dst_pak._planes_at(torch.arange(30)), torch.from_numpy(planes_b))
    assert torch.equal(dst_pak.pi[:30], src.pi[:30])


def test_packed_roundtrip_state_dict_into_packed(tmp_path):
    """Packed save -> packed load keeps the compact form and exact planes."""
    rng = np.random.default_rng(25)
    planes_b = _random_binary_planes(rng, 20)
    examples = [_example(planes_b[i], rng, side=0, ply=i) for i in range(20)]
    src = ReplayBuffer(capacity=32, device="cpu", pack_planes=True)
    src.add(examples)
    sd = src.state_dict()
    assert "packed_planes" in sd and sd["packed_planes"].dtype == torch.uint8
    path = tmp_path / "pp.pt"
    torch.save(sd, path)
    sd2 = torch.load(path, map_location="cpu", weights_only=False)
    dst = ReplayBuffer(capacity=32, device="cpu", pack_planes=True)
    dst.load_state_dict(sd2)
    assert torch.equal(dst.packed_planes[:20], src.packed_planes[:20])
    assert torch.equal(dst._planes_at(torch.arange(20)), torch.from_numpy(planes_b))
