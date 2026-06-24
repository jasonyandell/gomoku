"""Tests for the V3 auxiliary opponent-reply policy head.

The load-bearing test is `test_d4_alignment_*`: the aux target (the opponent's
next-ply MCTS policy, an 81-vector over board cells) MUST be transformed by the
SAME D4 symmetry as its position's planes+pi, or the label is rotated/reflected
relative to the board it describes. This file proves that explicitly, plus the
byte-identical-OFF guarantees and the masking of last-ply positions.
"""

from __future__ import annotations

import numpy as np
import torch

from gomoku.game import (
    BOARD_SIZE,
    N_ACTIONS,
    N_INPUT_PLANES,
    _sym_board,
    _sym_policy,
    augment,
    augment_with_aux,
)
from gomoku.model import build_model, fuse_model_for_inference, n_params
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import SelfPlayExample, _aux_target_for, _build_examples


# --------------------------------------------------------------------------
# D4 alignment — the highest-risk correctness item.
# --------------------------------------------------------------------------

def _onehot(action: int) -> np.ndarray:
    v = np.zeros(N_ACTIONS, dtype=np.float32)
    v[action] = 1.0
    return v


def test_d4_alignment_aux_tracks_board_transform():
    """augment_with_aux must permute the aux target by the identical symmetry
    as planes+pi. We use one-hot policies at known cells so the transform is
    unambiguous: after symmetry s, the hot cell of the aux target must land at
    exactly the cell the same symmetry sends the original aux cell to."""
    planes = np.random.default_rng(0).random(
        (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    ).astype(np.float32)
    # current pi peaks at cell (1,2); aux (opponent reply) peaks at cell (7,5).
    pi_action = 1 * BOARD_SIZE + 2
    aux_action = 7 * BOARD_SIZE + 5
    pi = _onehot(pi_action)
    aux = _onehot(aux_action)

    out = augment_with_aux(planes, pi, aux)
    ref_planes = augment(planes, pi)  # the canonical 2-tuple augment
    assert len(out) == 8

    for s in range(8):
        aug_planes, aug_pi, aug_aux = out[s]
        # planes + pi must be byte-identical to the canonical augment(planes,pi)
        ref_p, ref_pi = ref_planes[s]
        assert np.array_equal(aug_planes, ref_p), f"planes mismatch at s={s}"
        assert np.array_equal(aug_pi, ref_pi), f"pi mismatch at s={s}"
        # aux must be exactly _sym_policy(aux, s) — same symmetry as the position
        expected_aux = _sym_policy(aux, s)
        assert np.array_equal(aug_aux, expected_aux), f"aux mismatch at s={s}"
        # And the hot cell must land where the symmetry sends the original cell.
        expected_cell = int(_sym_board(_onehot(aux_action).reshape(BOARD_SIZE, BOARD_SIZE), s).argmax())
        assert int(aug_aux.argmax()) == expected_cell, (
            f"aux hot cell misaligned at s={s}: {int(aug_aux.argmax())} != {expected_cell}"
        )


def test_d4_alignment_identity_and_rot90():
    """Spot-check two concrete symmetries by hand to catch a wrong-axis bug."""
    aux_action = 0 * BOARD_SIZE + 1  # row 0, col 1
    aux = _onehot(aux_action)
    # s=0 is identity
    assert int(_sym_policy(aux, 0).argmax()) == aux_action
    # s=1 is rot90 in axes (-2,-1): np.rot90 maps (r,c) -> (BOARD_SIZE-1-c, r)
    rotated = _sym_policy(aux, 1).reshape(BOARD_SIZE, BOARD_SIZE)
    rr, cc = np.unravel_index(int(rotated.argmax()), (BOARD_SIZE, BOARD_SIZE))
    assert (rr, cc) == (BOARD_SIZE - 1 - 1, 0), f"rot90 sent (0,1) to {(rr, cc)}"


# --------------------------------------------------------------------------
# Aux target sourcing + masking.
# --------------------------------------------------------------------------

def test_aux_target_is_next_ply_pi():
    """The aux target for ply i is the recorded pi at ply i+1 (opponent moves
    there). Built a fake trajectory of alternating sides."""
    pis = [_onehot(k) for k in range(5)]
    traj = [(np.zeros(1), pis[i], i % 2) for i in range(5)]
    # ply 0 (side 0): aux target is ply 1's pi (side 1)
    t0 = _aux_target_for(traj, 0, side=0)
    assert t0 is not None and int(t0.argmax()) == 1
    # ply 3 (side 1): aux target is ply 4's pi (side 0)
    t3 = _aux_target_for(traj, 3, side=1)
    assert t3 is not None and int(t3.argmax()) == 4


def test_aux_target_last_ply_masked():
    """The LAST recorded ply has no next entry -> aux target is None (masked)."""
    pis = [_onehot(k) for k in range(3)]
    traj = [(np.zeros(1), pis[i], i % 2) for i in range(3)]
    assert _aux_target_for(traj, 2, side=0) is None  # last ply


def test_aux_target_same_side_next_masked():
    """If the next recorded ply is the SAME side (e.g. an unrecorded PCR fast
    move sat between them), decline the target rather than misalign."""
    # ply 0 side 0, ply 1 side 0 (same) -> mask
    traj = [(np.zeros(1), _onehot(1), 0), (np.zeros(1), _onehot(2), 0)]
    assert _aux_target_for(traj, 0, side=0) is None


def test_build_examples_aux_propagates_and_masks():
    """_build_examples with aug: 8 examples, each carrying a D4-transformed aux
    target; with aux_pi=None, 8 examples with aux_pi None (masked downstream)."""
    planes = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), np.float32)
    pi = _onehot(10)
    aux = _onehot(20)
    with_aux = _build_examples(planes, pi, 1.0, 0, 5, aux, augment_symmetries=True)
    assert len(with_aux) == 8
    assert all(e.aux_pi is not None for e in with_aux)
    # each aux must equal the matching symmetry of the original
    for s, e in enumerate(with_aux):
        assert np.array_equal(e.aux_pi, _sym_policy(aux, s))
    none_aux = _build_examples(planes, pi, 1.0, 0, 5, None, augment_symmetries=True)
    assert len(none_aux) == 8
    assert all(e.aux_pi is None for e in none_aux)


# --------------------------------------------------------------------------
# Byte-identical OFF guarantees.
# --------------------------------------------------------------------------

def test_model_off_byte_identical():
    """small model with aux OFF == small without the aux field: same keys+params.

    This isolates the AUX head's byte-identical-OFF guarantee, so it builds the
    baseline with choice_head=False — the swap2 choice head is an UNRELATED
    optional head (default-on; +choice_fc.{weight,bias}, +value_hidden*N_CHOICES+
    N_CHOICES params) and must not be conflated with the aux-OFF baseline. With it
    off, the baseline is the pure pre-aux-field small model (344458 params, 72
    keys), and turning aux ON must add EXACTLY the 8 aux keys and nothing else.
    """
    import dataclasses

    from gomoku.model import SIZE_PRESETS, GomokuNet

    base_cfg = dataclasses.replace(SIZE_PRESETS["small"], choice_head=False)
    off = GomokuNet(base_cfg)
    on = GomokuNet(dataclasses.replace(base_cfg, aux_opponent_reply=True))
    off_keys = set(off.state_dict().keys())
    on_keys = set(on.state_dict().keys())
    assert off.cfg.aux_opponent_reply is False
    assert n_params(off) == 344458
    assert len(off_keys) == 72
    # ON strictly adds the 8 aux keys, nothing else changes.
    assert on_keys - off_keys == {
        "aux_policy_conv.weight",
        "aux_policy_bn.weight",
        "aux_policy_bn.bias",
        "aux_policy_bn.running_mean",
        "aux_policy_bn.running_var",
        "aux_policy_bn.num_batches_tracked",
        "aux_policy_fc.weight",
        "aux_policy_fc.bias",
    }
    assert off_keys - on_keys == set()


def test_forward_off_two_tuple_on_three_tuple():
    x = torch.randn(2, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    off = build_model("small")
    assert len(off(x)) == 2
    import pytest

    with pytest.raises(RuntimeError):
        off(x, return_aux=True)
    on = build_model("small", aux_opponent_reply=True)
    assert len(on(x)) == 2  # default path never runs aux head
    p, v, a = on(x, return_aux=True)
    assert a.shape == (2, N_ACTIONS)


def test_fuse_drops_aux():
    """fuse_model_for_inference must not run or require the aux head; the
    default forward stays a 2-tuple after fusing."""
    x = torch.randn(2, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    on = build_model("small", aux_opponent_reply=True)
    on.eval()
    fuse_model_for_inference(on)
    assert len(on(x)) == 2  # aux head untouched, zero aux FLOPs


def test_buffer_off_no_aux_tensors():
    b = ReplayBuffer(50, aux_opponent_reply=False)
    assert b.aux_pi is None and b.aux_mask is None
    ex = [
        SelfPlayExample(
            np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), np.float32),
            np.ones(N_ACTIONS, np.float32) / N_ACTIONS,
            1.0,
        )
        for _ in range(4)
    ]
    b.add(ex)
    sd = b.state_dict()
    assert not any("aux" in k for k in sd)
    assert len(b.sample(4)) == 5


def test_buffer_on_aux_roundtrip_and_tolerate_missing():
    aux = _onehot(7)
    b = ReplayBuffer(50, aux_opponent_reply=True)
    exa = [
        SelfPlayExample(
            np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), np.float32),
            np.ones(N_ACTIONS, np.float32) / N_ACTIONS,
            1.0,
            aux_pi=aux,
        ),
        SelfPlayExample(
            np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), np.float32),
            np.ones(N_ACTIONS, np.float32) / N_ACTIONS,
            1.0,
            aux_pi=None,
        ),
    ]
    b.add(exa)
    assert b.aux_mask[0].item() is True or bool(b.aux_mask[0])
    assert not bool(b.aux_mask[1])
    assert int(b.aux_pi[0].argmax()) == 7
    sd = b.state_dict()
    b2 = ReplayBuffer(50, aux_opponent_reply=True)
    b2.load_state_dict(sd)
    assert bool(b2.aux_mask[0]) and not bool(b2.aux_mask[1])
    # ON buffer loading an OFF (aux-less) checkpoint: tolerate-missing, mask off
    off = ReplayBuffer(50, aux_opponent_reply=False)
    off.add(exa)
    b3 = ReplayBuffer(50, aux_opponent_reply=True)
    b3.load_state_dict(off.state_dict())
    assert not bool(b3.aux_mask[: b3.size].any())
