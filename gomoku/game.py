"""9x9 free-style gomoku: first to 5-in-a-row wins, no opening restrictions.

State is stored as two boolean planes (current-player stones, opponent stones).
After every move we flip perspective so the side-to-move is always plane 0.
This canonical form is what gets fed to the network and the MCTS tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOARD_SIZE = 9
N_ACTIONS = BOARD_SIZE * BOARD_SIZE
WIN_LEN = 5

# 4 line directions (the other 4 are their negatives, so 4 covers both ways)
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass
class GameState:
    """Canonical: plane 0 = side-to-move's stones, plane 1 = opponent's stones."""

    board: np.ndarray  # (2, 9, 9) bool
    move_count: int

    @classmethod
    def initial(cls) -> "GameState":
        return cls(board=np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=bool), move_count=0)

    def legal_mask(self) -> np.ndarray:
        """Return (81,) bool — True where empty (legal to place)."""
        occupied = self.board[0] | self.board[1]
        return ~occupied.reshape(-1)

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.legal_mask())

    def apply(self, action: int) -> "GameState":
        """Place a stone for the side-to-move at action, then flip perspective.

        The returned state has perspective = the next player to move.
        """
        r, c = divmod(action, BOARD_SIZE)
        if self.board[0, r, c] or self.board[1, r, c]:
            raise ValueError(f"illegal move {action} on occupied square")
        new_board = self.board.copy()
        new_board[0, r, c] = True
        # Flip planes so plane 0 is now the next side-to-move.
        new_board = new_board[::-1].copy()
        return GameState(board=new_board, move_count=self.move_count + 1)

    def is_terminal(self) -> tuple[bool, float]:
        """Check for terminal state.

        Returns (done, value_from_side_to_move_perspective).
        value is +1 if side-to-move just won (impossible — see below),
                -1 if opponent just won, 0 for draw, undefined otherwise.

        Convention: this is called AFTER apply(), so plane 1 holds the player who
        just moved. We check plane 1 for a 5-in-a-row.
        """
        if _has_five_in_a_row(self.board[1]):
            # The player who just moved (now on plane 1) won.
            # From the current side-to-move's perspective this is a loss.
            return True, -1.0
        if self.move_count >= N_ACTIONS:
            return True, 0.0
        return False, 0.0

    def to_planes(self) -> np.ndarray:
        """Return (3, 9, 9) float32 input for the network.

        Plane 0: side-to-move's stones.
        Plane 1: opponent's stones.
        Plane 2: constant 1.0 (acts as a bias / side-to-move indicator slot —
                 kept here so the model has somewhere to learn "always full" features).
        """
        out = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        out[0] = self.board[0]
        out[1] = self.board[1]
        out[2] = 1.0
        return out


def _has_five_in_a_row(plane: np.ndarray) -> bool:
    for dr, dc in _DIRS:
        if _has_five_in_dir(plane, dr, dc):
            return True
    return False


def _has_five_in_dir(plane: np.ndarray, dr: int, dc: int) -> bool:
    # Slide a length-5 window along (dr, dc) and AND across the 5 shifts.
    n = BOARD_SIZE
    for r0 in range(n):
        for c0 in range(n):
            if not plane[r0, c0]:
                continue
            r_end = r0 + dr * (WIN_LEN - 1)
            c_end = c0 + dc * (WIN_LEN - 1)
            if not (0 <= r_end < n and 0 <= c_end < n):
                continue
            ok = True
            for k in range(1, WIN_LEN):
                if not plane[r0 + dr * k, c0 + dc * k]:
                    ok = False
                    break
            if ok:
                return True
    return False


# ----- 8-fold symmetry for data augmentation -----
# A square board has 8 symmetries (D4): 4 rotations × 2 reflections.
# For each, the action index (r, c) maps to (r', c') under the same transform.

def _sym_board(board: np.ndarray, sym: int) -> np.ndarray:
    """Apply one of 8 D4 symmetries to a (..., 9, 9) array."""
    rot = sym % 4
    flip = sym // 4
    out = np.rot90(board, rot, axes=(-2, -1))
    if flip:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def _sym_policy(policy: np.ndarray, sym: int) -> np.ndarray:
    """Apply same symmetry to a (81,) policy vector."""
    p = policy.reshape(BOARD_SIZE, BOARD_SIZE)
    return _sym_board(p, sym).reshape(-1)


def augment(planes: np.ndarray, policy: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return all 8 (planes, policy) pairs under D4 symmetry."""
    return [(_sym_board(planes, s), _sym_policy(policy, s)) for s in range(8)]


def action_to_str(action: int) -> str:
    r, c = divmod(action, BOARD_SIZE)
    return f"{chr(ord('a') + c)}{r + 1}"


def str_to_action(s: str) -> int:
    s = s.strip().lower()
    if len(s) < 2:
        raise ValueError(f"bad move: {s!r}")
    col = ord(s[0]) - ord('a')
    row = int(s[1:]) - 1
    if not (0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE):
        raise ValueError(f"out of range: {s!r}")
    return row * BOARD_SIZE + col


def render(state: GameState, *, last_action: int | None = None) -> str:
    """Render board to a string. Shows X for side-to-move's stones, O for opponent."""
    # We render from "the player about to move" perspective:
    # 'X' = side to move's stones, 'O' = opponent's.
    lines = []
    header = "   " + " ".join(chr(ord('a') + c) for c in range(BOARD_SIZE))
    lines.append(header)
    for r in range(BOARD_SIZE):
        row = [f"{r + 1:2d} "]
        for c in range(BOARD_SIZE):
            if state.board[0, r, c]:
                ch = 'X'
            elif state.board[1, r, c]:
                ch = 'O'
            else:
                ch = '.'
            if last_action is not None and last_action == r * BOARD_SIZE + c:
                ch = f"[{ch}]"
                row.append(ch)
            else:
                row.append(f" {ch}")
        lines.append("".join(row))
    return "\n".join(lines)
