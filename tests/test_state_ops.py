import numpy as np
import pytest

from gomoku.game import BOARD_SIZE, HISTORY_PLY, N_ACTIONS, GameState, str_to_action
from gomoku import state_ops


DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def reference_legal_mask(board):
    return ~((board[0] | board[1]).reshape(-1))


def reference_has_five(plane):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if not plane[row, col]:
                continue
            for dr, dc in DIRS:
                end_row = row + dr * 4
                end_col = col + dc * 4
                if not (0 <= end_row < BOARD_SIZE and 0 <= end_col < BOARD_SIZE):
                    continue
                if all(plane[row + dr * k, col + dc * k] for k in range(5)):
                    return True
    return False


def reference_terminal_value(board, move_count):
    if reference_has_five(board[1]):
        return True, -1.0
    if move_count >= N_ACTIONS:
        return True, 0.0
    return False, 0.0


def reference_apply_move_arrays(board, history, action):
    row, col = divmod(action, BOARD_SIZE)
    if board[0, row, col] or board[1, row, col]:
        raise ValueError(f"illegal move {action} on occupied square")
    snapshot = board.copy()
    new_board = board.copy()
    new_board[0, row, col] = True
    new_board = new_board[::-1].copy()
    return new_board, (snapshot,) + history[: HISTORY_PLY - 1]


def test_state_ops_match_reference_over_random_legal_games():
    rng = np.random.default_rng(20260520)
    for _ in range(20):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        history = ()
        move_count = 0
        for _ in range(50):
            np.testing.assert_array_equal(
                state_ops.legal_mask(board),
                reference_legal_mask(board),
            )
            assert state_ops.terminal_value(board, move_count) == reference_terminal_value(
                board, move_count
            )

            done, _, mask = state_ops.init_node_status(board, move_count)
            ref_done, _ = reference_terminal_value(board, move_count)
            assert done == ref_done
            if ref_done:
                assert mask is None
                break
            assert mask is not None
            np.testing.assert_array_equal(mask, reference_legal_mask(board))

            legal = np.flatnonzero(reference_legal_mask(board))
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))

            expected_board, expected_history = reference_apply_move_arrays(
                board, history, action
            )
            board, history = state_ops.apply_move_arrays(board, history, action)
            move_count += 1
            np.testing.assert_array_equal(board, expected_board)
            assert len(history) == len(expected_history)
            for actual, expected in zip(history, expected_history):
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "moves",
    [
        ["a5", "a1", "b5", "b1", "c5", "c1", "d5", "d1", "e5"],
        ["a1", "i1", "b2", "i2", "c3", "i3", "d4", "i4", "e5"],
        ["a5", "i1", "b4", "i2", "c3", "i3", "d2", "i4", "e1"],
    ],
)
def test_game_state_uses_state_ops_boundary_for_wins(moves):
    state = GameState.initial()
    for move in moves:
        state = state.apply(str_to_action(move))
    assert state.is_terminal() == (True, -1.0)


def test_apply_move_arrays_rejects_occupied_square():
    state = GameState.initial().apply(str_to_action("e5"))
    with pytest.raises(ValueError, match="illegal move"):
        state_ops.apply_move_arrays(state.board, state.history, str_to_action("e5"))
