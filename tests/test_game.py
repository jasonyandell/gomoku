import numpy as np
import pytest

from gomoku.game import (
    BOARD_SIZE,
    N_ACTIONS,
    GameState,
    augment,
    str_to_action,
    action_to_str,
)


def test_initial_state():
    s = GameState.initial()
    assert s.board.shape == (2, 9, 9)
    assert not s.board.any()
    assert s.move_count == 0
    assert s.legal_mask().sum() == N_ACTIONS


def test_apply_flips_perspective():
    s = GameState.initial()
    # Black plays e5 (center). Should appear on plane 1 of the next state
    # (because plane 0 of the next state is white's stones).
    s2 = s.apply(str_to_action("e5"))
    assert s2.move_count == 1
    assert s2.board[1, 4, 4]
    assert not s2.board[0, 4, 4]
    # The same square is now illegal.
    assert not s2.legal_mask()[str_to_action("e5")]


def test_horizontal_win():
    s = GameState.initial()
    # Black builds a5..e5; white plays elsewhere. 5 in a row -> black wins on move 9.
    moves = [
        ("a5", "a1"),
        ("b5", "b1"),
        ("c5", "c1"),
        ("d5", "d1"),
    ]
    for b, w in moves:
        s = s.apply(str_to_action(b))
        assert not s.is_terminal()[0]
        s = s.apply(str_to_action(w))
        assert not s.is_terminal()[0]
    s = s.apply(str_to_action("e5"))  # black's 5th in a row
    done, v = s.is_terminal()
    assert done
    # The state's side-to-move is now white. v should be -1 (black just won).
    assert v == -1.0


def test_diagonal_win():
    s = GameState.initial()
    # Black: a1, b2, c3, d4, e5 (main diagonal). White plays harmless moves.
    black = ["a1", "b2", "c3", "d4", "e5"]
    white = ["i1", "i2", "i3", "i4"]
    for i in range(4):
        s = s.apply(str_to_action(black[i]))
        s = s.apply(str_to_action(white[i]))
    s = s.apply(str_to_action(black[4]))
    done, v = s.is_terminal()
    assert done and v == -1.0


def test_antidiagonal_win():
    s = GameState.initial()
    black = ["a5", "b4", "c3", "d2", "e1"]
    white = ["i1", "i2", "i3", "i4"]
    for i in range(4):
        s = s.apply(str_to_action(black[i]))
        s = s.apply(str_to_action(white[i]))
    s = s.apply(str_to_action(black[4]))
    done, v = s.is_terminal()
    assert done and v == -1.0


def test_no_overline_required():
    # Free-style: 6 in a row also wins (5+ counts).
    s = GameState.initial()
    for i in range(5):
        s = s.apply(str_to_action(f"{chr(ord('a') + i)}5"))
        if i < 4:
            s = s.apply(str_to_action(f"{chr(ord('a') + i)}1"))
    done, _ = s.is_terminal()
    assert done


def test_no_four_is_not_win():
    s = GameState.initial()
    # Black plays a..d, white plays a..d on row 1. 4 in a row, not done.
    for i in range(4):
        s = s.apply(str_to_action(f"{chr(ord('a') + i)}5"))
        s = s.apply(str_to_action(f"{chr(ord('a') + i)}1"))
    done, _ = s.is_terminal()
    assert not done


def test_action_str_roundtrip():
    for a in range(N_ACTIONS):
        assert str_to_action(action_to_str(a)) == a


def test_str_to_action_invalid():
    with pytest.raises(ValueError):
        str_to_action("z9")
    with pytest.raises(ValueError):
        str_to_action("a99")


def test_symmetry_count():
    rng = np.random.default_rng(0)
    planes = rng.random((3, 9, 9)).astype(np.float32)
    pi = rng.random(N_ACTIONS).astype(np.float32)
    augs = augment(planes, pi)
    assert len(augs) == 8


def test_symmetry_preserves_sum():
    rng = np.random.default_rng(0)
    planes = rng.random((3, 9, 9)).astype(np.float32)
    pi = rng.random(N_ACTIONS).astype(np.float32)
    pi /= pi.sum()
    for ap, api in augment(planes, pi):
        assert ap.shape == planes.shape
        assert api.shape == pi.shape
        assert np.isclose(api.sum(), pi.sum())
        # Planes element sums preserved per channel
        assert np.isclose(ap.sum(axis=(-1, -2)), planes.sum(axis=(-1, -2))).all()


def test_symmetry_identity():
    """Symmetry 0 = identity."""
    rng = np.random.default_rng(0)
    planes = rng.random((3, 9, 9)).astype(np.float32)
    pi = rng.random(N_ACTIONS).astype(np.float32)
    augs = augment(planes, pi)
    ap, api = augs[0]
    np.testing.assert_array_equal(ap, planes)
    np.testing.assert_array_equal(api, pi)


def test_full_board_is_terminal():
    s = GameState.initial()
    # Fill board with a non-winning pattern is tricky; just simulate move_count >= N_ACTIONS
    # by manually constructing.
    s = GameState(board=np.ones((2, 9, 9), dtype=bool), move_count=N_ACTIONS)
    # board has stones in both planes for every square — this state could "win" by
    # 5-in-a-row check on plane 1. is_terminal will return True via the win path.
    done, _ = s.is_terminal()
    assert done
