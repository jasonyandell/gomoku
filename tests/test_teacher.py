"""Teacher dataset: action permutations, D4 augmentation alignment, npz io."""

from __future__ import annotations

import numpy as np
import torch

from gomoku.board_config import BOARD_SIZE, N_ACTIONS
from gomoku.game import N_INPUT_PLANES
from gomoku.teacher import (
    TeacherDataset,
    TeacherExample,
    _action_perms,
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
    planes, pi = ds.sample(8)
    # Every sampled target is one of the two stored moves.
    targets = set(pi.argmax(dim=-1).tolist())
    assert targets <= {1 * BOARD_SIZE + 1, 2 * BOARD_SIZE + 2}
