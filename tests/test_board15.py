"""15x15 board-size tests (Phase 2 of the 15x15 plan).

Run with the board size set BEFORE pytest imports the package:

    GOMOKU_BOARD_SIZE=15 pytest tests/test_board15.py

In a default (9x9) run of the full suite every test here is skipped — board
size is a process-level constant, so the 15x15 checks need their own process.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gomoku import game, native_mcts, state_ops
from gomoku.game import GameState, augment
from gomoku.model import (
    ModelConfig,
    GomokuNet,
    build_model,
    load_checkpoint,
    n_params,
    save_checkpoint,
)
from gomoku.vcf import solve_vcf

pytestmark = pytest.mark.skipif(
    game.BOARD_SIZE != 15,
    reason="15x15 tests need GOMOKU_BOARD_SIZE=15 (board size is fixed per process)",
)

N = game.BOARD_SIZE


def _a(r: int, c: int) -> int:
    return r * N + c


def _play_line(attacker_cells, filler_start=200):
    """Alternate attacker moves through `attacker_cells` with harmless
    defender filler moves; return the state after the LAST attacker move."""
    state = GameState.initial()
    filler = filler_start
    for i, (r, c) in enumerate(attacker_cells):
        state = state.apply(_a(r, c))
        if i < len(attacker_cells) - 1:
            state = state.apply(filler)
            filler -= 1
    return state


# ---------------------------------------------------------------- game logic


def test_board_constants():
    assert game.BOARD_SIZE == 15
    assert game.N_ACTIONS == 225
    assert GameState.initial().board.shape == (2, 15, 15)
    assert GameState.initial().legal_mask().shape == (225,)


@pytest.mark.parametrize(
    "cells",
    [
        # row win hugging the right edge (row 0, cols 10..14)
        [(0, 10), (0, 11), (0, 12), (0, 13), (0, 14)],
        # column win hugging the bottom edge (col 14, rows 10..14)
        [(10, 14), (11, 14), (12, 14), (13, 14), (14, 14)],
        # main diagonal ending in the bottom-right corner
        [(10, 10), (11, 11), (12, 12), (13, 13), (14, 14)],
        # anti-diagonal ending in the bottom-left corner
        [(10, 4), (11, 3), (12, 2), (13, 1), (14, 0)],
        # a win through the center for good measure
        [(7, 5), (7, 6), (7, 7), (7, 8), (7, 9)],
    ],
)
def test_win_detection_at_edges(cells):
    state = _play_line(cells)
    done, value = state.is_terminal()
    assert done and value == -1.0  # the mover just won; perspective flipped


def test_no_phantom_win_across_row_wrap():
    # Flat indices (0,12)..(0,14),(1,0),(1,1) are contiguous in a flattened
    # 225-vector. A naive bit-shift five-in-a-row would wrap across the row
    # boundary; the real rules must NOT call this a win. Guards the multi-word
    # native bitboard as well as the numpy path.
    cells = [(0, 12), (0, 13), (0, 14), (1, 0), (1, 1)]
    state = _play_line(cells)
    done, _ = state.is_terminal()
    assert not done


def test_four_is_not_a_win():
    state = _play_line([(5, 5), (5, 6), (5, 7), (5, 8)])
    done, _ = state.is_terminal()
    assert not done


def test_d4_augmentation_alignment_and_identity():
    rng = np.random.default_rng(0)
    planes = rng.random((game.N_INPUT_PLANES, N, N)).astype(np.float32)
    # One-hot policy on an asymmetric cell so every symmetry moves it.
    policy = np.zeros(game.N_ACTIONS, dtype=np.float32)
    policy[_a(2, 11)] = 1.0
    marker = np.zeros((game.N_INPUT_PLANES, N, N), dtype=np.float32)
    marker[0, 2, 11] = 1.0

    pairs = augment(planes, policy)
    marker_pairs = augment(marker, policy)
    assert len(pairs) == 8

    # Identity element reproduces the input exactly.
    assert np.array_equal(pairs[0][0], planes)
    assert np.array_equal(pairs[0][1], policy)

    # Alignment: under every symmetry the policy's argmax lands exactly where
    # the marked board cell went.
    seen = set()
    for aug_marker, aug_policy in marker_pairs:
        m_idx = int(np.argmax(aug_marker[0].reshape(-1)))
        p_idx = int(np.argmax(aug_policy))
        assert m_idx == p_idx
        seen.add(p_idx)
    assert len(seen) == 8  # all 8 images distinct for an asymmetric cell


def test_to_planes_shape_and_history():
    state = GameState.initial().apply(_a(7, 7)).apply(_a(0, 0)).apply(_a(7, 8))
    planes = state.to_planes()
    assert planes.shape == (game.N_INPUT_PLANES, N, N)
    # Current frame: side to move owns (0,0); opponent owns (7,7),(7,8).
    assert planes[0, 0, 0] == 1.0
    assert planes[game.HISTORY_PLY, 7, 7] == 1.0
    assert planes[game.HISTORY_PLY, 7, 8] == 1.0
    assert planes[2 * game.HISTORY_PLY].min() == 1.0  # const plane


# --------------------------------------------------------------------- model


@pytest.mark.parametrize("size", ["tiny", "96x8"])
def test_model_forward_shapes(size):
    model = build_model(size)
    assert model.cfg.board_size == 15
    x = torch.zeros(3, game.N_INPUT_PLANES, N, N)
    p, v = model(x)
    assert p.shape == (3, 225)
    assert v.shape == (3,)
    assert torch.all(v.abs() <= 1.0)


def test_96x8_preset_is_the_planned_net():
    model = build_model("96x8")
    assert model.cfg.n_filters == 96
    assert model.cfg.n_blocks == 8
    # ~1.45M params per the feasibility bench (sanity band, not exact).
    assert 1_000_000 < n_params(model) < 2_500_000


def test_checkpoint_roundtrip_embeds_board_size(tmp_path):
    model = build_model("tiny")
    path = str(tmp_path / "ck15.pt")
    save_checkpoint(path, model, epoch=3, total_games=7)

    loaded, payload = load_checkpoint(path)
    assert payload["model_config"]["board_size"] == 15
    assert loaded.cfg.board_size == 15
    x = torch.full((2, game.N_INPUT_PLANES, N, N), 0.25)
    model.eval(), loaded.eval()
    with torch.no_grad():
        p0, v0 = model(x)
        p1, v1 = loaded(x)
    assert torch.allclose(p0, p1)
    assert torch.allclose(v0, v1)


def test_checkpoint_board_size_mismatch_rejected(tmp_path):
    # A 9x9 net is still constructible in a 15x15 process (cfg drives shapes);
    # loading its checkpoint into this process must fail loudly.
    cfg9 = ModelConfig(board_size=9, n_filters=32, n_blocks=2, value_hidden=32)
    model9 = GomokuNet(cfg9)
    path = str(tmp_path / "ck9.pt")
    save_checkpoint(path, model9)
    with pytest.raises(ValueError, match="board size"):
        load_checkpoint(path)
    # Explicit opt-out for offline inspection still works.
    loaded, payload = load_checkpoint(path, expect_board_size=None)
    assert payload["model_config"]["board_size"] == 9


def test_legacy_config_without_board_size_loads_as_9(tmp_path):
    cfg9 = ModelConfig(board_size=9, n_filters=32, n_blocks=2, value_hidden=32)
    model9 = GomokuNet(cfg9)
    path = str(tmp_path / "legacy.pt")
    save_checkpoint(path, model9)
    payload = torch.load(path, weights_only=False)
    del payload["model_config"]["board_size"]  # simulate a pre-15x15 checkpoint
    torch.save(payload, path)
    with pytest.raises(ValueError, match="board size"):
        load_checkpoint(path)  # defaults to 9, mismatches the active 15


# ----------------------------------------------------------------- self-play


def _uniform_planes_evaluator(batch_in) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-prior evaluator for BOTH evaluator calling conventions: the
    native path passes a (B, planes, N, N) array, the python MCTS path passes
    a list of GameState."""
    if isinstance(batch_in, np.ndarray):
        planes = batch_in
    else:
        planes = np.stack([s.to_planes() for s in batch_in])
    batch = planes.shape[0]
    priors = np.zeros((batch, game.N_ACTIONS), dtype=np.float32)
    values = np.zeros((batch,), dtype=np.float32)
    occupied = (planes[:, 0] > 0.5) | (planes[:, game.HISTORY_PLY] > 0.5)
    legal = ~occupied.reshape(batch, game.N_ACTIONS)
    for i in range(batch):
        if legal[i].any():
            priors[i, legal[i]] = 1.0 / legal[i].sum()
    return priors, values


def test_short_selfplay_game_completes():
    from gomoku.self_play import generate_games

    records = generate_games(
        2,
        _uniform_planes_evaluator,
        n_simulations=8,
        wave_size=4,
        rng=np.random.default_rng(123),
        augment_symmetries=False,
    )
    assert len(records) == 2
    for rec in records:
        assert rec.plies == len(rec.examples)
        assert 5 <= rec.plies <= 225
        assert rec.outcome in (-1.0, 0.0, 1.0)
        ex = rec.examples[0]
        assert ex.planes.shape == (game.N_INPUT_PLANES, N, N)
        assert ex.pi.shape == (225,)
        assert np.isclose(ex.pi.sum(), 1.0, atol=1e-5)


# ----------------------------------------------------------------------- VCF


def test_vcf_open_four_is_mate_in_one():
    b = np.zeros((2, N, N), dtype=bool)
    for c in (9, 10, 11, 12):  # open four near the right edge: _XXXX_
        b[0, 7, c] = True
    b[1, 0, 0] = b[1, 0, 1] = b[1, 1, 0] = True
    res = solve_vcf(b)
    assert res.has_forced_win
    assert res.mate_distance == 1
    assert res.winning_move in (_a(7, 8), _a(7, 13))


def test_vcf_no_false_positive_on_quiet_position():
    b = np.zeros((2, N, N), dtype=bool)
    b[0, 7, 7] = b[0, 8, 8] = True
    b[1, 6, 6] = b[1, 9, 9] = True
    res = solve_vcf(b)
    assert not res.has_forced_win


# ----------------------------------------------------- native vs python A/B


def _random_position(rng: np.random.Generator, n_stones: int = 40) -> GameState:
    state = GameState.initial()
    for _ in range(n_stones):
        done, _ = state.is_terminal()
        if done:
            break
        legal = state.legal_actions()
        state = state.apply(int(rng.choice(legal)))
    return state


def test_native_state_ops_agree_with_numpy():
    assert state_ops.USING_NATIVE, "15x15 native state-ops build missing"
    rng = np.random.default_rng(7)
    for _ in range(10):
        state = _random_position(rng)
        b = state.board
        # legal mask
        np.testing.assert_array_equal(
            state_ops.legal_mask(b), ~(b[0] | b[1]).reshape(-1)
        )
        # five-in-a-row, native vs numpy fallback
        for plane in (b[0], b[1]):
            assert state_ops.has_five_in_a_row(plane) == (
                state_ops._has_five_in_a_row_numpy(plane)
            )
        # terminal value matches the numpy definition
        done, value = state_ops.terminal_value(b, state.move_count)
        py_done = state_ops._has_five_in_a_row_numpy(b[1])
        if py_done:
            assert (done, value) == (True, -1.0)
        elif state.move_count >= 225:
            assert (done, value) == (True, 0.0)
        else:
            assert not done
        # apply parity on a random legal move
        legal = np.flatnonzero(state_ops.legal_mask(b))
        if legal.size == 0 or done:
            continue
        action = int(rng.choice(legal))
        nb, nh = state_ops.apply_move_arrays(b, state.history, action)
        snapshot = b.copy()
        ref = b.copy()
        ref[0, action // N, action % N] = True
        ref = ref[::-1]
        np.testing.assert_array_equal(nb, ref)
        np.testing.assert_array_equal(nh[0], snapshot)


def test_native_and_python_mcts_agree_on_forced_win():
    assert native_mcts.USING_NATIVE_MCTS, "15x15 native MCTS build missing"
    from gomoku.mcts import MCTSGame, run_batched_mcts

    # Side to move has four in a row; a9 (8,0)..(8,3); winning square (8,4).
    state = GameState.initial()
    for mine, theirs in [((8, 0), (0, 0)), ((8, 1), (0, 1)), ((8, 2), (0, 2))]:
        state = state.apply(_a(*mine)).apply(_a(*theirs))
    state = state.apply(_a(8, 3)).apply(_a(0, 3))
    win = _a(8, 4)

    ngame = native_mcts.NativeMCTSGame(state)
    native_mcts.search_batch(
        [ngame],
        _uniform_planes_evaluator,
        n_simulations=600,
        wave_size=16,
        add_root_noise=False,
    )
    pgame = MCTSGame(state)
    run_batched_mcts([pgame], _uniform_planes_evaluator, n_simulations=600,
                     add_root_noise=False)

    n_visits = ngame.visit_counts()
    p_visits = pgame.root.N
    assert n_visits.shape == (225,)
    assert p_visits.shape == (225,)
    assert int(np.argmax(n_visits)) == win
    assert int(np.argmax(p_visits)) == win


def test_native_root_planes_match_game_state_at_15():
    state = GameState.initial()
    rng = np.random.default_rng(11)
    for _ in range(12):
        state = state.apply(int(rng.choice(state.legal_actions())))
    ngame = native_mcts.NativeMCTSGame(state)
    np.testing.assert_array_equal(ngame.root_planes(), state.to_planes())
    assert ngame.move_count == state.move_count
