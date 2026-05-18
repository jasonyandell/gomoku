// TypeScript port of gomoku/game.py.
// 9x9 free-style gomoku: first to 5-in-a-row wins. State is canonicalized so
// plane 0 is always the side-to-move's stones; plane 1 is opponent stones.
// `apply()` flips perspective. `to_planes()` adds a third constant-1 plane
// so the network sees (3, 9, 9).

export const BOARD_SIZE = 9;
export const N_ACTIONS = BOARD_SIZE * BOARD_SIZE; // 81
export const WIN_LEN = 5;

const PLANE = BOARD_SIZE * BOARD_SIZE; // 81

const DIRS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

/** Canonical GameState. board is length 162 = 2 planes * 81. */
export class GameState {
  // Plane 0 (offsets 0..80)   = side-to-move's stones.
  // Plane 1 (offsets 81..161) = opponent's stones.
  readonly board: Uint8Array;
  readonly moveCount: number;

  constructor(board: Uint8Array, moveCount: number) {
    this.board = board;
    this.moveCount = moveCount;
  }

  static initial(): GameState {
    return new GameState(new Uint8Array(2 * PLANE), 0);
  }

  /** Length-81 mask of legal (empty) actions. */
  legalMask(): Uint8Array {
    const out = new Uint8Array(N_ACTIONS);
    const b = this.board;
    for (let i = 0; i < PLANE; i++) {
      out[i] = b[i] === 0 && b[i + PLANE] === 0 ? 1 : 0;
    }
    return out;
  }

  legalActions(): number[] {
    const mask = this.legalMask();
    const out: number[] = [];
    for (let i = 0; i < N_ACTIONS; i++) if (mask[i]) out.push(i);
    return out;
  }

  /** Return a NEW state with `action` played by the side-to-move; perspective flipped. */
  apply(action: number): GameState {
    if (action < 0 || action >= N_ACTIONS) {
      throw new Error(`out of range: ${action}`);
    }
    if (this.board[action] || this.board[action + PLANE]) {
      throw new Error(`illegal move ${action} on occupied square`);
    }
    const next = new Uint8Array(2 * PLANE);
    // Swap planes during copy: my (plane 0) becomes opponent (plane 1) for next state.
    // First: place my new stone in current plane 0, then copy planes swapped.
    // We'll just copy directly: out[0..81] = old[81..162] (opponent's old stones)
    //                          out[81..162] = old[0..81] + this new stone
    for (let i = 0; i < PLANE; i++) {
      next[i] = this.board[i + PLANE];
      next[i + PLANE] = this.board[i];
    }
    next[action + PLANE] = 1;
    return new GameState(next, this.moveCount + 1);
  }

  /** Returns [done, value_from_side_to_move_perspective].
   *
   * Called AFTER apply(), so plane 1 holds the player who just moved.
   * If they have 5-in-a-row, the current side-to-move just lost (value=-1).
   */
  isTerminal(): { done: boolean; value: number } {
    if (hasFiveInARow(this.board, PLANE)) {
      return { done: true, value: -1.0 };
    }
    if (this.moveCount >= N_ACTIONS) return { done: true, value: 0.0 };
    return { done: false, value: 0.0 };
  }

  /** Return Float32Array of shape [3, 9, 9] flattened: 243 elements. */
  toPlanes(): Float32Array {
    const out = new Float32Array(3 * PLANE);
    for (let i = 0; i < PLANE; i++) {
      out[i] = this.board[i];
      out[i + PLANE] = this.board[i + PLANE];
      out[i + 2 * PLANE] = 1.0;
    }
    return out;
  }
}

/** Check if `plane` (length-81 slice starting at `offset` in `board`) has a 5-in-a-row. */
function hasFiveInARow(board: Uint8Array, offset: number): boolean {
  for (const [dr, dc] of DIRS) {
    for (let r0 = 0; r0 < BOARD_SIZE; r0++) {
      for (let c0 = 0; c0 < BOARD_SIZE; c0++) {
        if (!board[offset + r0 * BOARD_SIZE + c0]) continue;
        const rEnd = r0 + dr * (WIN_LEN - 1);
        const cEnd = c0 + dc * (WIN_LEN - 1);
        if (rEnd < 0 || rEnd >= BOARD_SIZE || cEnd < 0 || cEnd >= BOARD_SIZE) continue;
        let ok = true;
        for (let k = 1; k < WIN_LEN; k++) {
          if (!board[offset + (r0 + dr * k) * BOARD_SIZE + (c0 + dc * k)]) {
            ok = false;
            break;
          }
        }
        if (ok) return true;
      }
    }
  }
  return false;
}

export function actionToStr(action: number): string {
  const r = Math.floor(action / BOARD_SIZE);
  const c = action % BOARD_SIZE;
  return `${String.fromCharCode("a".charCodeAt(0) + c)}${r + 1}`;
}

export function strToAction(s: string): number {
  s = s.trim().toLowerCase();
  if (s.length < 2) throw new Error(`bad move: ${s}`);
  const col = s.charCodeAt(0) - "a".charCodeAt(0);
  const row = parseInt(s.slice(1), 10) - 1;
  if (!Number.isInteger(row) || col < 0 || col >= BOARD_SIZE || row < 0 || row >= BOARD_SIZE) {
    throw new Error(`out of range: ${s}`);
  }
  return row * BOARD_SIZE + col;
}
