// TypeScript port of gomoku/game.py.
// 9x9 free-style gomoku: first to 5-in-a-row wins. State is canonicalized so
// plane 0 is always the side-to-move's stones; plane 1 is opponent stones.
// `apply()` flips perspective and snapshots the pre-move canonical board onto
// the head of `history` (most-recent first; up to HISTORY_PLY-1 entries) —
// mirrors gomoku/state_ops.py:apply_move_arrays.
// `toPlanes(n)` supports two layouts:
//   n=3:  [my, opp, const-1]               — legacy single-frame (epoch-100 model)
//   n=17: [my×H, opp×H, const-1]  H=8     — AlphaZero-style history (WL series)

export const BOARD_SIZE = 9;
export const N_ACTIONS = BOARD_SIZE * BOARD_SIZE; // 81
export const WIN_LEN = 5;
export const HISTORY_PLY = 8; // matches gomoku/game.py HISTORY_PLY

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
  // Up to HISTORY_PLY-1 (=7) past canonical boards, most-recent first.
  // Each entry is the pre-flip canonical board (length 2*PLANE) as observed
  // by the mover at that ply.
  readonly history: ReadonlyArray<Uint8Array>;

  constructor(
    board: Uint8Array,
    moveCount: number,
    history: ReadonlyArray<Uint8Array> = [],
  ) {
    this.board = board;
    this.moveCount = moveCount;
    this.history = history;
  }

  static initial(): GameState {
    return new GameState(new Uint8Array(2 * PLANE), 0, []);
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
    // Snapshot the pre-move canonical board (matches Python's state_ops.apply_move_arrays).
    const snapshot = new Uint8Array(this.board);
    const next = new Uint8Array(2 * PLANE);
    // Swap planes during copy: my (plane 0) becomes opponent (plane 1) for next state.
    for (let i = 0; i < PLANE; i++) {
      next[i] = this.board[i + PLANE];
      next[i + PLANE] = this.board[i];
    }
    next[action + PLANE] = 1;
    const nextHistory = [snapshot, ...this.history.slice(0, HISTORY_PLY - 1)];
    return new GameState(next, this.moveCount + 1, nextHistory);
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

  /** Return Float32Array of shape [nPlanes, 9, 9] flattened.
   *
   * nPlanes=3:  [my, opp, const-1]                — legacy
   * nPlanes=17: AlphaZero-style 2*HISTORY_PLY history planes + const-1, matching
   *             gomoku/game.py:to_planes layout. Slots beyond available history
   *             stay zero.
   */
  toPlanes(nPlanes: number = 3): Float32Array {
    if (nPlanes === 3) {
      const out = new Float32Array(3 * PLANE);
      for (let i = 0; i < PLANE; i++) {
        out[i] = this.board[i];
        out[i + PLANE] = this.board[i + PLANE];
        out[i + 2 * PLANE] = 1.0;
      }
      return out;
    }
    if (nPlanes === 2 * HISTORY_PLY + 1) {
      const H = HISTORY_PLY;
      const out = new Float32Array(nPlanes * PLANE);
      // Plane 0 (my, t) and plane H (opp, t)
      for (let i = 0; i < PLANE; i++) {
        out[i] = this.board[i];
        out[i + H * PLANE] = this.board[i + PLANE];
      }
      // Planes 1..H-1 / H+1..2H-1: past frames, perspective-corrected.
      // history[k-1] = canonical board observed by mover at t-k. If k is odd
      // the mover was the opposite side, so swap planes when reading.
      for (let k = 1; k < H; k++) {
        const past = this.history[k - 1];
        if (!past) break;
        const myOffset = k % 2 === 0 ? 0 : PLANE;
        const oppOffset = k % 2 === 0 ? PLANE : 0;
        for (let i = 0; i < PLANE; i++) {
          out[k * PLANE + i] = past[myOffset + i];
          out[(H + k) * PLANE + i] = past[oppOffset + i];
        }
      }
      // Last plane (2H): constant 1.0
      for (let i = 0; i < PLANE; i++) out[2 * H * PLANE + i] = 1.0;
      return out;
    }
    throw new Error(`unsupported nPlanes=${nPlanes} (expected 3 or 17)`);
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
