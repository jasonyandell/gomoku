"""Free-style gomoku: first to 5-in-a-row wins, no opening restrictions.

State is stored as two boolean planes (current-player stones, opponent stones).
After every move we flip perspective so the side-to-move is always plane 0.
This canonical form is what gets fed to the network and the MCTS tree.

AlphaZero-style history: the `to_planes()` output includes `HISTORY_PLY` past
canonical board snapshots per side, so the network can see threats forming
across multiple plies — defense is much easier to learn when "opponent just
placed a 3rd stone in a row" is visible directly rather than inferred from a
static snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BOARD_SIZE = 9
N_ACTIONS = BOARD_SIZE * BOARD_SIZE
WIN_LEN = 5
HISTORY_PLY = 8                                  # past plies of each side in the input
N_INPUT_PLANES = 2 * HISTORY_PLY + 1             # 8 me-hist + 8 opp-hist + 1 const = 17

# 4 line directions (the other 4 are their negatives, so 4 covers both ways)
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass
class GameState:
    """Canonical: plane 0 = side-to-move's stones, plane 1 = opponent's stones.

    `history` holds up to `HISTORY_PLY - 1` past canonical boards, most-recent
    first. Each entry is a (2, BOARD_SIZE, BOARD_SIZE) bool array that was the
    canonical `board` immediately before that move's apply() flipped perspective.
    """

    board: np.ndarray              # (2, N, N) bool
    move_count: int
    history: tuple = field(default_factory=tuple)  # tuple of (2, N, N) bool arrays

    @classmethod
    def initial(cls) -> "GameState":
        return cls(
            board=np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=bool),
            move_count=0,
            history=(),
        )

    def legal_mask(self) -> np.ndarray:
        """Return (N_ACTIONS,) bool — True where empty (legal to place)."""
        occupied = self.board[0] | self.board[1]
        return ~occupied.reshape(-1)

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.legal_mask())

    def apply(self, action: int) -> "GameState":
        """Place a stone for the side-to-move at action, then flip perspective.

        The returned state has perspective = the next player to move. The pre-move
        canonical board (`board` BEFORE the stone was placed and BEFORE the flip —
        i.e. what the mover *observed* when deciding) is pushed onto the head of
        `history`. That way each history slot encodes a real past observation
        distinct from the current frame.
        """
        r, c = divmod(action, BOARD_SIZE)
        if self.board[0, r, c] or self.board[1, r, c]:
            raise ValueError(f"illegal move {action} on occupied square")
        # Snapshot the state-as-the-mover-observed-it: BEFORE we place their stone
        # and BEFORE we flip. Each ply produces a distinct snapshot of "what did
        # the player see when they made this decision."
        snapshot = self.board.copy()
        new_board = self.board.copy()
        new_board[0, r, c] = True
        # Flip planes so plane 0 is now the next side-to-move.
        new_board = new_board[::-1].copy()
        # Push snapshot; keep at most HISTORY_PLY - 1 (current frame is `board`).
        new_history = (snapshot,) + self.history[: HISTORY_PLY - 1]
        return GameState(
            board=new_board, move_count=self.move_count + 1, history=new_history
        )

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
        """Return (N_INPUT_PLANES, N, N) float32 input for the network.

        Layout (mirrors AlphaZero Go, scaled down to HISTORY_PLY plies):
          0           : my stones at t   (= side-to-move's stones, current frame)
          1..H-1      : my stones at t-1, t-2, ..., t-(H-1)
          H           : opp stones at t
          H+1..2H-1   : opp stones at t-1, ..., t-(H-1)
          2H          : constant 1.0

        For history step k (1..H-1) the *meaning* of "my" rotates because the
        perspective flipped each ply: at step k the canonical mover was the same
        side as now iff k is even, opposite iff k is odd. So we read the matching
        plane out of `history[k-1]`.
        """
        H = HISTORY_PLY
        out = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        out[0] = self.board[0]
        out[H] = self.board[1]
        for k, past_board in enumerate(self.history, start=1):
            if k >= H:
                break
            if k % 2 == 0:
                # Same parity → past mover same side as now → past_board[0] == my color
                out[k] = past_board[0]
                out[H + k] = past_board[1]
            else:
                # Opposite parity → swap
                out[k] = past_board[1]
                out[H + k] = past_board[0]
        out[2 * H] = 1.0
        return out


def _has_five_in_a_row(plane: np.ndarray) -> bool:
    # Vectorized: AND together 5 shifted views along each direction. ~2× faster
    # than the Python-loop variant on a 9×9 board.
    p = plane
    h = p[:, :-4] & p[:, 1:-3] & p[:, 2:-2] & p[:, 3:-1] & p[:, 4:]
    if h.any():
        return True
    v = p[:-4, :] & p[1:-3, :] & p[2:-2, :] & p[3:-1, :] & p[4:, :]
    if v.any():
        return True
    d = p[:-4, :-4] & p[1:-3, 1:-3] & p[2:-2, 2:-2] & p[3:-1, 3:-1] & p[4:, 4:]
    if d.any():
        return True
    a = p[:-4, 4:] & p[1:-3, 3:-1] & p[2:-2, 2:-2] & p[3:-1, 1:-3] & p[4:, :-4]
    return bool(a.any())


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
