"""Teacher dataset: action permutations, D4 augmentation alignment, npz io."""

from __future__ import annotations

import numpy as np
import torch

from gomoku.board_config import BOARD_SIZE, N_ACTIONS
from gomoku.game import GameState, N_INPUT_PLANES
from gomoku.teacher import (
    TeacherDataset,
    TeacherExample,
    _action_perms,
    bfs_mine,
    save_teacher_npz,
)


def _onehot_planes(r: int, c: int) -> np.ndarray:
    """Planes with a single distinctive stone in plane 0 at (r, c)."""
    p = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    p[0, r, c] = 1.0
    return p


# --- the core invariant: plane-transform and move-permutation agree --------
def test_action_perm_matches_plane_transform():
    perm = _action_perms(BOARD_SIZE, N_ACTIONS)
    r, c = 2, 3
    a0 = r * BOARD_SIZE + c
    for s in range(8):
        rot, flip = s % 4, s // 4
        plane = torch.zeros(1, 1, BOARD_SIZE, BOARD_SIZE)
        plane[0, 0, r, c] = 1.0
        p2 = plane
        if rot:
            p2 = torch.rot90(p2, rot, dims=(-2, -1))
        if flip:
            p2 = torch.flip(p2, dims=(-1,))
        rr, cc = (p2[0, 0] == 1).nonzero()[0].tolist()
        # The stone's new cell must equal where the action permutation sends a0.
        assert int(perm[s, a0]) == rr * BOARD_SIZE + cc


def test_perm_is_a_permutation():
    perm = _action_perms(BOARD_SIZE, N_ACTIONS)
    for s in range(8):
        assert sorted(perm[s].tolist()) == list(range(N_ACTIONS))


# --- TeacherDataset.sample -------------------------------------------------
def test_sample_shapes_and_onehot_no_augment():
    r, c = 4, 1
    a = r * BOARD_SIZE + c
    ds = TeacherDataset(
        np.stack([_onehot_planes(r, c)]),
        np.array([a], dtype=np.int64),
        device="cpu",
        augment=False,
    )
    planes, pi = ds.sample(5)
    assert planes.shape == (5, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert pi.shape == (5, N_ACTIONS)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(5))
    # Without augmentation the target is exactly the stored move.
    assert torch.all(pi.argmax(dim=-1) == a)


def test_sample_augment_keeps_planes_and_move_aligned():
    r, c = 2, 5
    a = r * BOARD_SIZE + c
    ds = TeacherDataset(
        np.stack([_onehot_planes(r, c)]),
        np.array([a], dtype=np.int64),
        device="cpu",
        augment=True,
    )
    # Over many samples, whatever symmetry is drawn, the plane-0 stone cell must
    # equal the one-hot target cell (they transform together).
    for _ in range(40):
        planes, pi = ds.sample(1)
        stone = (planes[0, 0] == 1).nonzero()
        assert stone.shape[0] == 1
        rr, cc = stone[0].tolist()
        assert int(pi.argmax(dim=-1).item()) == rr * BOARD_SIZE + cc


# --- npz round-trip --------------------------------------------------------
def test_npz_roundtrip(tmp_path):
    exs = [
        TeacherExample(planes=_onehot_planes(1, 1), move=1 * BOARD_SIZE + 1, side=0, ply=3),
        TeacherExample(planes=_onehot_planes(2, 2), move=2 * BOARD_SIZE + 2, side=1, ply=4),
    ]
    path = str(tmp_path / "t.npz")
    save_teacher_npz(path, exs)
    ds = TeacherDataset.load(path, device="cpu", augment=False)
    assert ds.n == 2
    assert ds.soft_policy is None  # v1 (hard) npz has no dense soft target
    planes, pi = ds.sample(8)
    # Every sampled target is one of the two stored moves.
    targets = set(pi.argmax(dim=-1).tolist())
    assert targets <= {1 * BOARD_SIZE + 1, 2 * BOARD_SIZE + 2}


# --- SOFT teacher (v2): dense winrate target ------------------------------
def _soft_example(r: int, c: int, wr: dict[int, float]) -> TeacherExample:
    best = max(wr.items(), key=lambda kv: kv[1])[0]
    return TeacherExample(
        planes=_onehot_planes(r, c), move=best, side=0, ply=4, winrates=dict(wr)
    )


def test_soft_npz_roundtrip_carries_soft_policy(tmp_path):
    a0 = 3 * BOARD_SIZE + 3
    a1 = 4 * BOARD_SIZE + 4
    a2 = 5 * BOARD_SIZE + 5
    ex = _soft_example(3, 3, {a0: 0.8, a1: 0.4, a2: 0.1})
    path = str(tmp_path / "soft.npz")
    save_teacher_npz(path, [ex])
    # version bumped to 2 and a dense soft_policy is present.
    d = np.load(path)
    assert int(d["version"]) == 2
    assert "soft_policy" in d.files
    assert d["soft_policy"].shape == (1, N_ACTIONS)
    # only the three scored cells carry mass; everything else is exactly 0.
    sp = d["soft_policy"][0]
    assert (sp > 0).sum() == 3
    assert np.isclose(sp[a0], 0.8, atol=1e-3)
    # `moves` kept (=argmax) for back-compat.
    assert int(d["moves"][0]) == a0

    ds = TeacherDataset.load(path, device="cpu", augment=False, teacher_temp=0.10)
    assert ds.soft_policy is not None
    planes, pi = ds.sample(4)
    assert pi.shape == (4, N_ACTIONS)
    # the target is a proper distribution living only on the scored support.
    assert torch.allclose(pi.sum(dim=-1), torch.ones(4), atol=1e-4)
    support = {a0, a1, a2}
    nz = set((pi[0] > 0).nonzero().flatten().tolist())
    assert nz <= support
    # higher winrate -> higher probability.
    assert pi[0, a0] > pi[0, a1] > pi[0, a2]


def test_soft_target_collapses_to_one_hot_ce_as_temp_to_zero():
    """As teacher-temp -> 0, the masked softmax target sharpens to a one-hot at
    the max-winrate move, so its cross-entropy vs the v1 one-hot CE -> 0."""
    a_best = 6 * BOARD_SIZE + 6
    a_mid = 2 * BOARD_SIZE + 2
    a_lo = 1 * BOARD_SIZE + 8
    wr = {a_best: 0.9, a_mid: 0.5, a_lo: 0.2}
    soft = np.zeros((1, N_ACTIONS), dtype=np.float32)
    for a, w in wr.items():
        soft[0, a] = w
    moves = np.array([a_best], dtype=np.int64)
    planes = np.stack([_onehot_planes(6, 6)])

    onehot = torch.zeros(N_ACTIONS)
    onehot[a_best] = 1.0

    def ce_to_onehot(temp: float) -> float:
        ds = TeacherDataset(
            planes, moves, device="cpu", augment=False,
            soft_policy=soft, teacher_temp=temp,
        )
        _, pi = ds.sample(1)
        # CE of the one-hot label under the soft target's distribution: -log p(best).
        return float(-torch.log(pi[0, a_best].clamp_min(1e-12)))

    ce_warm = ce_to_onehot(1.0)
    ce_cool = ce_to_onehot(0.10)
    ce_cold = ce_to_onehot(0.001)
    # Cooling monotonically concentrates mass on the best move -> CE shrinks to ~0.
    assert ce_warm > ce_cool > ce_cold
    assert ce_cold < 1e-2


def test_soft_augment_keeps_dense_target_aligned():
    """Under any D4 symmetry the dense winrate row permutes WITH the planes: the
    plane-0 stone cell stays the argmax cell of the soft target."""
    r, c = 2, 5
    a = r * BOARD_SIZE + c
    soft = np.zeros((1, N_ACTIONS), dtype=np.float32)
    soft[0, a] = 0.9
    soft[0, 0] = 0.3  # a second scored cell so it's a real distribution
    ds = TeacherDataset(
        np.stack([_onehot_planes(r, c)]),
        np.array([a], dtype=np.int64),
        device="cpu", augment=True, soft_policy=soft, teacher_temp=0.05,
    )
    for _ in range(40):
        planes, pi = ds.sample(1)
        stone = (planes[0, 0] == 1).nonzero()
        assert stone.shape[0] == 1
        rr, cc = stone[0].tolist()
        # the dominant soft-target cell moves exactly where the stone moves.
        assert int(pi.argmax(dim=-1).item()) == rr * BOARD_SIZE + cc


# --- model-free BFS mining (bfs_mine over a stub pool, no engine) -----------
class _StubPool:
    """Offline stand-in for RapfiPool: scores each board's first few legal
    actions with descending winrates. Deterministic, no engine, no network —
    enough to exercise bfs_mine's expansion / dedup / limit logic."""

    size = 4

    def __init__(self, support: int = 6) -> None:
        self.support = support
        self.analyzed = 0

    def analyze_states(self, states, *, max_node=20000, max_pv=None):
        out = []
        for s in states:
            legal = sorted(int(a) for a in s.legal_actions())
            cap = self.support if max_pv is None else min(self.support, max_pv)
            legal = legal[:cap]
            out.append({a: 1.0 - 0.1 * i for i, a in enumerate(legal)})
            self.analyzed += 1
        return out


def test_bfs_mine_respects_limit_dedups_and_is_soft():
    pool = _StubPool()
    start = GameState.initial()
    ex = bfs_mine(pool, start, expand_k=3, limit=20, max_pv=4)

    # the BFS tree is far larger than 20 → exactly the limit, no over-mining
    assert len(ex) == 20
    assert pool.analyzed == 20  # one analyze per produced example, dedup'd

    # model-free mining is ALWAYS soft: every example carries a winrate map,
    # and max_pv caps the scored support
    assert all(e.winrates for e in ex)
    assert all(len(e.winrates) <= 4 for e in ex)

    # BFS order: the root is first, plies are non-decreasing down the levels
    assert ex[0].ply == start.move_count
    assert all(ex[i].ply <= ex[i + 1].ply for i in range(len(ex) - 1))

    # move is the argmax of the soft target (the highest-winrate scored action)
    for e in ex:
        assert e.move == max(e.winrates.items(), key=lambda kv: kv[1])[0]


def test_bfs_mine_expand_k_one_is_a_single_chain():
    pool = _StubPool()
    start = GameState.initial()
    ex = bfs_mine(pool, start, expand_k=1, limit=5)
    # k=1 → one child per level → a strict chain, ply increments by one each step
    assert [e.ply for e in ex] == [start.move_count + i for i in range(5)]
