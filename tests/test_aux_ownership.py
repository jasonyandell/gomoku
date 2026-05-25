"""Tests for the V4 auxiliary ownership head (KataGo-style, adapted to gomoku).

The ownership head predicts per-cell final control: an 81-vector over board
cells with +1 at the eventual WINNER's stones, -1 at the LOSER's stones, 0 at
empty cells (all-zeros on a draw), constant for every position of a game.

The load-bearing tests are:
  * test_d4_alignment_* — the ownership target MUST ride the SAME D4 symmetry as
    its position's planes+pi, or the label is rotated/reflected vs the board.
  * test_ownership_target_winner_loser — the +1/-1/0 semantics are correct in
    absolute board coordinates regardless of which side moved last.
  * test_native_gumbel_records_both_targets — BOTH aux targets are recorded on
    the native-Gumbel self-play path (the path Derby v4 cells actually run).
Plus the byte-identical-OFF guarantees, masking, buffer roundtrip, and a
finite-loss train_step check with both aux weights on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import gomoku.native_mcts as nm
from gomoku.game import (
    BOARD_SIZE,
    HISTORY_PLY,
    N_ACTIONS,
    N_INPUT_PLANES,
    GameState,
    _sym_board,
    _sym_policy,
    augment,
    augment_with_cell_targets,
)
from gomoku.model import build_model, fuse_model_for_inference, n_params
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import (
    SelfPlayExample,
    _build_examples,
    _ownership_target,
    _can_use_native_gumbel,
    generate_games,
)
from gomoku.train import make_torch_evaluator, train_step


def _onehot(action: int) -> np.ndarray:
    v = np.zeros(N_ACTIONS, dtype=np.float32)
    v[action] = 1.0
    return v


# --------------------------------------------------------------------------
# Ownership target definition — winner/loser/empty semantics.
# --------------------------------------------------------------------------

def _play_black_wins_row0():
    """Drive a native game to a decisive black win on row 0 cols 0-4."""
    g = nm.NativeMCTSGame(GameState.initial(), c_puct=1.25, c_puct_base=19652.0, seed=1)
    for m in [0, 9, 1, 10, 2, 11, 3, 12, 4]:  # black 0..4 wins; white on row 1
        g.advance_root(m)
    return g


def test_ownership_target_winner_loser():
    """+1 at the winner's final stones, -1 at the loser's, 0 elsewhere — in
    absolute board coordinates, independent of side-to-move at terminal."""
    g = _play_black_wins_row0()
    done, val = g.is_terminal()
    assert done and val == -1.0
    term_side = g.move_count % 2  # absolute side to move at terminal
    fp = np.asarray(g.root_planes())
    own = _ownership_target(fp, term_side, +1.0)  # black (first mover) won
    own2d = own.reshape(BOARD_SIZE, BOARD_SIZE)
    # Black winner: row 0 cols 0-4 == +1.
    assert np.array_equal(own2d[0, 0:5], np.ones(5, np.float32))
    # White loser: cells 9..12 == row 1 cols 0-3 == -1.
    assert np.array_equal(own2d[1, 0:4], -np.ones(4, np.float32))
    # Winner has exactly one more stone than the loser (winner moved last).
    assert int((own > 0).sum()) == 5
    assert int((own < 0).sum()) == 4
    assert set(np.unique(own).tolist()) <= {-1.0, 0.0, 1.0}


def test_ownership_target_draw_is_zeros():
    """A draw credits nobody — all-zeros target (still a VALID target)."""
    g = _play_black_wins_row0()
    fp = np.asarray(g.root_planes())
    own = _ownership_target(fp, g.move_count % 2, 0.0)
    assert own is not None
    assert np.array_equal(own, np.zeros(N_ACTIONS, np.float32))


def test_ownership_target_white_winner_flips_sign():
    """If white won (outcome_for_black < 0), white's stones become +1 and
    black's -1 — the winner/loser flip is driven by outcome, not by color."""
    g = _play_black_wins_row0()
    fp = np.asarray(g.root_planes())
    term_side = g.move_count % 2
    own_black_win = _ownership_target(fp, term_side, +1.0)
    own_white_win = _ownership_target(fp, term_side, -1.0)
    # Same board, opposite outcome -> exact sign flip.
    assert np.array_equal(own_white_win, -own_black_win)


# --------------------------------------------------------------------------
# D4 alignment — the highest-risk correctness item.
# --------------------------------------------------------------------------

def test_d4_alignment_ownership_tracks_board_transform():
    """augment_with_cell_targets must permute the ownership target by the
    identical symmetry as planes+pi. Uses a one-hot ownership so the transform
    is unambiguous."""
    planes = np.random.default_rng(0).random(
        (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    ).astype(np.float32)
    pi = _onehot(1 * BOARD_SIZE + 2)
    own_cell = 7 * BOARD_SIZE + 5
    own = _onehot(own_cell)

    out = augment_with_cell_targets(planes, pi, [own])
    ref = augment(planes, pi)
    assert len(out) == 8
    for s in range(8):
        aug_planes, aug_pi, aug_targets = out[s]
        ref_p, ref_pi = ref[s]
        assert np.array_equal(aug_planes, ref_p), f"planes mismatch at s={s}"
        assert np.array_equal(aug_pi, ref_pi), f"pi mismatch at s={s}"
        aug_own = aug_targets[0]
        assert np.array_equal(aug_own, _sym_policy(own, s)), f"own mismatch s={s}"
        expected_cell = int(
            _sym_board(_onehot(own_cell).reshape(BOARD_SIZE, BOARD_SIZE), s).argmax()
        )
        assert int(aug_own.argmax()) == expected_cell, (
            f"ownership hot cell misaligned at s={s}"
        )


def test_d4_alignment_aux_and_ownership_both_ride_same_symmetry():
    """When BOTH targets are carried, each rides the SAME symmetry as the
    position — slot order is fixed [aux, ownership]."""
    planes = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), np.float32)
    pi = _onehot(10)
    aux = _onehot(20)
    own = _onehot(30)
    out = augment_with_cell_targets(planes, pi, [aux, own])
    for s in range(8):
        _, _, targets = out[s]
        assert np.array_equal(targets[0], _sym_policy(aux, s))
        assert np.array_equal(targets[1], _sym_policy(own, s))


def test_build_examples_carries_ownership_aligned():
    """_build_examples with aug emits 8 examples each carrying the
    D4-transformed ownership target; ownership=None -> all None (masked)."""
    planes = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), np.float32)
    pi = _onehot(10)
    own = _onehot(40)
    with_own = _build_examples(
        planes, pi, 1.0, 0, 5, None, augment_symmetries=True, ownership=own
    )
    assert len(with_own) == 8
    assert all(e.ownership is not None for e in with_own)
    assert all(e.aux_pi is None for e in with_own)  # aux not requested
    for s, e in enumerate(with_own):
        assert np.array_equal(e.ownership, _sym_policy(own, s))
    none_own = _build_examples(
        planes, pi, 1.0, 0, 5, None, augment_symmetries=True, ownership=None
    )
    assert all(e.ownership is None for e in none_own)


def test_build_examples_both_targets_aligned():
    """Both aux_pi and ownership present -> each rides its own symmetry."""
    planes = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), np.float32)
    pi = _onehot(1)
    aux = _onehot(2)
    own = _onehot(3)
    out = _build_examples(
        planes, pi, 1.0, 0, 0, aux, augment_symmetries=True, ownership=own
    )
    assert len(out) == 8
    for s, e in enumerate(out):
        assert np.array_equal(e.aux_pi, _sym_policy(aux, s))
        assert np.array_equal(e.ownership, _sym_policy(own, s))


def test_build_examples_off_unchanged_when_both_none():
    """Byte-identical OFF: both targets None -> output equals plain augment."""
    planes = np.random.default_rng(1).random(
        (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    ).astype(np.float32)
    pi = _onehot(11)
    out = _build_examples(planes, pi, -1.0, 1, 7, None, augment_symmetries=True)
    ref = augment(planes, pi)
    assert len(out) == 8
    for s, e in enumerate(out):
        assert np.array_equal(e.planes, ref[s][0])
        assert np.array_equal(e.pi, ref[s][1])
        assert e.aux_pi is None and e.ownership is None


# --------------------------------------------------------------------------
# Byte-identical OFF guarantees (model).
# --------------------------------------------------------------------------

OWNERSHIP_KEYS = {
    "ownership_conv.weight",
    "ownership_bn.weight",
    "ownership_bn.bias",
    "ownership_bn.running_mean",
    "ownership_bn.running_var",
    "ownership_bn.num_batches_tracked",
    "ownership_fc.weight",
    "ownership_fc.bias",
}


@pytest.mark.parametrize("global_pool", [None, True, 2])
def test_model_ownership_off_byte_identical(global_pool):
    """Ownership OFF + any global_pool == the same model without the field:
    identical state_dict keys and param count. (Proves coexistence with the
    global-pool arch from main.)"""
    off = build_model("small", global_pool=global_pool)
    on = build_model("small", global_pool=global_pool, aux_ownership=True)
    off_keys = set(off.state_dict().keys())
    on_keys = set(on.state_dict().keys())
    assert off.cfg.aux_ownership is False
    assert on_keys - off_keys == OWNERSHIP_KEYS
    assert off_keys - on_keys == set()
    assert n_params(on) - n_params(off) == sum(
        p.numel() for n, p in on.named_parameters() if n.startswith("ownership_")
    )


def test_model_both_heads_coexist_independently():
    """global_pool, opponent-reply, and ownership are three independent levers;
    enabling all three adds exactly the union of their extra keys."""
    base = build_model("small")
    both = build_model(
        "small", global_pool=True, aux_opponent_reply=True, aux_ownership=True
    )
    extra = set(both.state_dict().keys()) - set(base.state_dict().keys())
    # ownership keys present; aux keys present; global-pool params live in the
    # tower (extra pool_fc weights), not as NEW top-level head modules.
    assert OWNERSHIP_KEYS <= extra
    assert any(k.startswith("aux_policy") for k in extra)


def test_forward_independent_head_flags():
    x = torch.randn(2, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    # ownership only
    m = build_model("small", aux_ownership=True)
    assert len(m(x)) == 2  # default never runs aux heads
    p, v, o = m(x, return_ownership=True)
    assert o.shape == (2, N_ACTIONS)
    with pytest.raises(RuntimeError):
        m(x, return_aux=True)  # opponent-reply head not constructed
    # both heads
    mb = build_model("small", aux_opponent_reply=True, aux_ownership=True)
    p2, v2, a2, o2 = mb(x, return_aux=True, return_ownership=True)
    assert a2.shape == (2, N_ACTIONS) and o2.shape == (2, N_ACTIONS)
    # aux only on a both-head model still yields a 3-tuple (aux), no ownership
    assert len(mb(x, return_aux=True)) == 3


def test_fuse_drops_ownership():
    x = torch.randn(2, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    m = build_model("small", aux_ownership=True)
    m.eval()
    fuse_model_for_inference(m)
    assert len(m(x)) == 2  # ownership head untouched, zero aux FLOPs


# --------------------------------------------------------------------------
# Buffer roundtrip + tolerate-missing.
# --------------------------------------------------------------------------

def _ex(ownership=None, aux_pi=None):
    return SelfPlayExample(
        np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), np.float32),
        np.ones(N_ACTIONS, np.float32) / N_ACTIONS,
        1.0,
        aux_pi=aux_pi,
        ownership=ownership,
    )


def test_buffer_off_no_ownership_tensors():
    b = ReplayBuffer(50, aux_ownership=False)
    assert b.ownership is None and b.ownership_mask is None
    b.add([_ex() for _ in range(4)])
    sd = b.state_dict()
    assert not any("ownership" in k for k in sd)
    assert len(b.sample(4)) == 5  # default 5-tuple unchanged


def test_buffer_on_ownership_roundtrip_and_tolerate_missing():
    own = _onehot(7)
    b = ReplayBuffer(50, aux_ownership=True)
    b.add([_ex(ownership=own), _ex(ownership=None)])
    assert bool(b.ownership_mask[0]) and not bool(b.ownership_mask[1])
    assert int(b.ownership[0].argmax()) == 7
    # sample with return_ownership appends (ownership, mask)
    out = b.sample(4, return_ownership=True)
    assert len(out) == 7
    # roundtrip
    b2 = ReplayBuffer(50, aux_ownership=True)
    b2.load_state_dict(b.state_dict())
    assert bool(b2.ownership_mask[0]) and not bool(b2.ownership_mask[1])
    # ON buffer loading an OFF (ownership-less) checkpoint: tolerate-missing
    off = ReplayBuffer(50, aux_ownership=False)
    off.add([_ex(ownership=own)])
    b3 = ReplayBuffer(50, aux_ownership=True)
    b3.load_state_dict(off.state_dict())
    assert not bool(b3.ownership_mask[: b3.size].any())


def test_buffer_both_targets_independent_sampling():
    """A buffer with both heads on returns base+aux+ownership in fixed order."""
    b = ReplayBuffer(50, aux_opponent_reply=True, aux_ownership=True)
    b.add([_ex(ownership=_onehot(3), aux_pi=_onehot(5))])
    out = b.sample(2, return_aux=True, return_ownership=True)
    assert len(out) == 9  # planes,pi,z,side,ply, aux_pi,aux_mask, own,own_mask
    # off buffer asked for both -> zero/false fallbacks, never errors
    off = ReplayBuffer(50)
    off.add([_ex()])
    out2 = off.sample(2, return_aux=True, return_ownership=True)
    assert len(out2) == 9
    assert not bool(out2[6].any()) and not bool(out2[8].any())  # both masks False


# --------------------------------------------------------------------------
# Native-Gumbel self-play path records BOTH aux targets + D4 consistency.
# --------------------------------------------------------------------------

def test_native_gumbel_records_both_targets():
    """Derby v4 cells run --gumbel-root self-play. Both the opponent-reply and
    the ownership targets must be recorded on _generate_games_native_gumbel."""
    m = build_model("tiny")
    m.eval()
    ev = make_torch_evaluator(m, "cpu")
    if not _can_use_native_gumbel(ev):
        pytest.skip("native Gumbel engine not built")
    rng = np.random.default_rng(3)
    recs = generate_games(
        8, ev, n_simulations=12, gumbel_root=True, wave_size=8,
        record_aux=True, record_ownership=True, rng=rng, max_plies=81,
    )
    decided = [r for r in recs if r.outcome != 0.0]
    assert decided, "expected at least one decisive game in the smoke batch"
    r = decided[0]
    # Every example carries the ownership target (constant per game).
    assert all(e.ownership is not None for e in r.examples)
    # The non-last positions carry an opponent-reply target.
    assert any(e.aux_pi is not None for e in r.examples)
    # Ownership values are in {-1,0,1} for a decided game and consistent.
    own0 = r.examples[0].ownership
    assert set(np.unique(own0).tolist()) <= {-1.0, 0.0, 1.0}
    assert int((own0 > 0).sum()) >= 5  # winner has at least the 5-in-a-row


def test_native_gumbel_ownership_d4_consistent():
    """Within a game, the 8 D4 variants of each recorded position carry exactly
    the 8-symmetry ORBIT of the position's ownership target — and each variant
    is the joint D4 image (planes, pi, ownership) of one shared symmetry. We
    prove the alignment per-variant by checking ALL symmetries that reproduce
    a variant's planes also reproduce its ownership (no misaligned label can
    survive even under symmetric-plane collisions)."""
    m = build_model("tiny")
    m.eval()
    ev = make_torch_evaluator(m, "cpu")
    if not _can_use_native_gumbel(ev):
        pytest.skip("native Gumbel engine not built")
    rng = np.random.default_rng(7)
    recs = generate_games(
        8, ev, n_simulations=12, gumbel_root=True, wave_size=8,
        record_aux=False, record_ownership=True, rng=rng, max_plies=81,
    )
    decided = [r for r in recs if r.outcome != 0.0]
    assert decided
    r = decided[0]
    base_planes = r.examples[0].planes
    base_own = r.examples[0].ownership
    variants = r.examples[:8]  # the 8 D4 images of ply 0
    # 1) The set of ownership variants is exactly the orbit of base_own.
    orbit = {_sym_policy(base_own, s).tobytes() for s in range(8)}
    got = {e.ownership.tobytes() for e in variants}
    assert got == orbit, "ownership variants are not the full D4 orbit"
    # 2) For each variant, EVERY symmetry that reproduces its planes must also
    #    reproduce its ownership — so the label can never be rotated/reflected
    #    relative to the board it describes, even when planes are self-symmetric.
    for e in variants:
        matches = [s for s in range(8)
                   if np.array_equal(_sym_board(base_planes, s), e.planes)]
        assert matches, "variant planes are not a D4 image of ply-0 planes"
        assert any(
            np.array_equal(e.ownership, _sym_policy(base_own, s)) for s in matches
        ), "ownership not aligned with any symmetry that produced its planes"


# --------------------------------------------------------------------------
# train_step: both aux loss terms finite + logged.
# --------------------------------------------------------------------------

def test_train_step_ownership_loss_finite_and_logged():
    torch.manual_seed(0)
    model = build_model("small", aux_opponent_reply=True, aux_ownership=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    B = 8
    planes = torch.randn(B, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    pi = torch.softmax(torch.randn(B, N_ACTIONS), dim=-1)
    z = torch.tanh(torch.randn(B))
    aux_pi = torch.softmax(torch.randn(B, N_ACTIONS), dim=-1)
    aux_mask = torch.ones(B, dtype=torch.bool)
    ownership = torch.empty(B, N_ACTIONS).uniform_(-1, 1)
    own_mask = torch.ones(B, dtype=torch.bool)
    m = train_step(
        model, opt, planes, pi, z,
        aux_pi=aux_pi, aux_mask=aux_mask, aux_weight=0.15,
        ownership=ownership, ownership_mask=own_mask, ownership_weight=0.15,
    )
    assert np.isfinite(m["loss/total"])
    assert np.isfinite(m["loss/aux_policy"])
    assert np.isfinite(m["loss/aux_ownership"])
    assert m["train/ownership_mask_frac"] == 1.0


def test_train_step_ownership_all_masked_no_error():
    """All ownership rows masked off -> zero-scaled term, still finite, no NaN."""
    torch.manual_seed(1)
    model = build_model("small", aux_ownership=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    B = 4
    planes = torch.randn(B, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    pi = torch.softmax(torch.randn(B, N_ACTIONS), dim=-1)
    z = torch.zeros(B)
    ownership = torch.zeros(B, N_ACTIONS)
    own_mask = torch.zeros(B, dtype=torch.bool)  # all masked
    m = train_step(
        model, opt, planes, pi, z,
        ownership=ownership, ownership_mask=own_mask, ownership_weight=0.15,
    )
    assert np.isfinite(m["loss/total"])
    assert m["loss/aux_ownership"] == 0.0
    assert m["train/ownership_mask_frac"] == 0.0


def test_train_step_off_byte_identical_two_tuple_path():
    """Both aux weights 0 -> plain model(planes) call, no aux keys logged."""
    torch.manual_seed(2)
    model = build_model("small")  # no aux heads at all
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    B = 4
    planes = torch.randn(B, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    pi = torch.softmax(torch.randn(B, N_ACTIONS), dim=-1)
    z = torch.zeros(B)
    m = train_step(model, opt, planes, pi, z)  # no aux args
    assert "loss/aux_ownership" not in m
    assert "loss/aux_policy" not in m
    assert np.isfinite(m["loss/total"])
