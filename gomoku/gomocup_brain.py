"""Brain-side Gomocup/Piskvork-protocol engine — the MIRROR of `external_engine.py`.

`external_engine.py` is the *consumer* half: it SPAWNS a Gomocup engine subprocess
and DRIVES it (our net plays against Rapfi/Embryo/…). This module is the *brain*
half: it IS that subprocess. It loads one of our trained checkpoints and answers
the protocol on stdin/stdout, so our live net becomes a first-class,
path-registerable Gomocup engine — drivable by GomocupJudge, the real piskvork
manager, OR our own panel-derby harness (`gomoku.match`'s `external:cmd=` spec).

The dev-time unlock (issue #31, parent #30): every net in `sweep_runs/` becomes a
registrable engine by `--checkpoint PATH` — no zip packaging, no rebuilds. The
closed-loop proof is `external_engine.py` (client) driving this module (brain):
champion-vs-champion must be ~50% and play legal games, validating BOTH protocol
halves at once.

Protocol surface (text over stdin/stdout, 0-based `X,Y` with X=column, Y=row):
- `START <size>`   -> `OK` (or `ERROR ...` if size != this net's board size)
- `INFO <k> <v>`   -> stored, no reply (timeout_turn/time_left are soft hints;
                      `--sims` is the real strength knob)
- `BEGIN`          -> bare `X,Y` opening move on an empty board
- `TURN <X>,<Y>`   -> apply opponent move, reply our bare `X,Y` (incremental)
- `BOARD` … `DONE` -> rebuild the position from `X,Y,field` lines (field 1 = own/
                      side-to-move, 2 = opponent, 3 = renju marker → ignored),
                      reply our bare `X,Y`. Re-entrant: this is the authoritative
                      resync (process reuse / harness re-dumps every move).
- `ABOUT`          -> identity line
- `RESTART`        -> `OK`, board cleared (size preserved)
- `END` / EOF      -> exit cleanly

Coordinate mapping (single source of truth, mirrors `external_engine.py`):
    flat_action = Y * BOARD_SIZE + X        # == row * BOARD_SIZE + col
    X = col = action %  BOARD_SIZE
    Y = row = action // BOARD_SIZE
Board arrays index `board[plane, row, col] == board[plane, Y, X]`.

Two load-bearing invariants:
- **Flush every reply.** The manager reads via a blocking reader thread; an
  unflushed move line reads as a timeout (TLE) and forfeits the game.
- **Board size before import.** `gomoku.board_config` locks `BOARD_SIZE` at import,
  so `--board-size`/`GOMOKU_BOARD_SIZE` MUST be set before any `import gomoku.*`.
  We pre-parse `--board-size` from `sys.argv` at module top, before importing
  `gomoku.game` (the shell wrapper also exports `GOMOKU_BOARD_SIZE=15`).

History note: under the harness, `external_engine.py` sends a full `BOARD` dump
every move (board-scan order, NOT play order), so true move-order history is
unrecoverable — we rebuild with empty history. `to_planes()` reads the CURRENT
board straight from `board[0]`/`board[1]` (only the *older* recency planes come
from history), so the net still sees a fully-correct position; only the recency
cues are zeroed, and symmetrically across both engines. Under pure `TURN`
incremental play (real piskvork) history accumulates naturally via `apply()`.
"""

from __future__ import annotations

import argparse
import os
import sys


def _preparse_board_size(argv: list[str]) -> None:
    """Set ``GOMOKU_BOARD_SIZE`` from ``--board-size`` BEFORE importing gomoku.

    ``gomoku.board_config`` resolves ``BOARD_SIZE`` once at import; the env var
    must be set first. The shell wrapper also exports it, so this is a belt-and-
    suspenders path for a bare ``python -m gomoku.gomocup_brain --board-size N``.
    """
    for i, a in enumerate(argv):
        if a == "--board-size" and i + 1 < len(argv):
            os.environ["GOMOKU_BOARD_SIZE"] = argv[i + 1]
            return
        if a.startswith("--board-size="):
            os.environ["GOMOKU_BOARD_SIZE"] = a.split("=", 1)[1]
            return


_preparse_board_size(sys.argv)

import numpy as np  # noqa: E402  (cheap; after the board-size pre-parse)

# Safe: gomoku.game is import-light (no torch). BOARD_SIZE is locked here, so the
# env must already be correct (wrapper export or the pre-parse above).
from gomoku.game import BOARD_SIZE, GameState  # noqa: E402


def _stdout_emit(line: str) -> None:
    """Write one reply line and FLUSH (an unflushed reply = a forfeit)."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _parse_xy(token: str, board_size: int) -> tuple[int, int] | None:
    """Parse a bare ``X,Y`` token into in-range ``(x, y)``, else ``None``."""
    parts = token.split(",")
    if len(parts) < 2:
        return None
    try:
        x = int(parts[0])
        y = int(parts[1])
    except ValueError:
        return None
    if 0 <= x < board_size and 0 <= y < board_size:
        return x, y
    return None


class GomocupBrain:
    """Pure protocol line-handler — testable without torch or a subprocess.

    ``pick`` maps a ``GameState`` (whose side-to-move == us) to a flat action int.
    ``emit`` writes one reply line (defaults to stdout+flush). Tests inject a stub
    ``pick`` and a list-appending ``emit``; ``main()`` wires the real MCTS picker
    and the flushing stdout emitter.
    """

    def __init__(
        self,
        pick,
        *,
        board_size: int = BOARD_SIZE,
        name: str = "gomoku-az",
        version: str = "1.0",
        author: str = "jasonyandell",
        emit=None,
    ):
        self.pick = pick
        self.board_size = board_size
        self.name = name
        self.version = version
        self.author = author
        self._emit = emit or _stdout_emit
        self.info: dict[str, str] = {}
        self.state = GameState.initial()
        self._in_board = False
        self._board_lines: list[tuple[int, int, int]] = []

    # -- helpers ---------------------------------------------------------

    def _coord(self, action: int) -> str:
        col = action % self.board_size
        row = action // self.board_size
        return f"{col},{row}"  # X=col, Y=row

    def _move_and_emit(self) -> None:
        """Pick a move for the side-to-move, apply it, emit exactly one ``X,Y``.

        Guards against an out-of-range / occupied pick: we NEVER emit an illegal
        coordinate (it would be scored an illegal forfeit). The MCTS picker only
        returns legal moves, but the fallback keeps a stub/garbage path safe.
        """
        legal = self.state.legal_actions()
        if len(legal) == 0:
            # Board full / terminal — nothing legal to play. Stay silent rather
            # than emit a bad coord; the manager adjudicates the finished board.
            return
        action = int(self.pick(self.state))
        if action not in set(int(a) for a in legal):
            action = int(legal[0])
        self.state = self.state.apply(action)
        self._emit(self._coord(action))

    # -- BOARD … DONE collection ----------------------------------------

    def _collect_board_line(self, line: str) -> None:
        parts = line.split(",")
        if len(parts) < 3:
            return  # malformed — ignore (illegal-input safety)
        try:
            x = int(parts[0])
            y = int(parts[1])
            field = int(parts[2])
        except ValueError:
            return
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            self._board_lines.append((x, y, field))

    def _finish_board(self) -> None:
        """Rebuild the GameState from the collected field lines (re-entrant).

        field 1 = own (side-to-move = us) -> plane 0; field 2 = opponent -> plane
        1; field 3 (renju winning/forbidden marker) is not a stone -> ignored.
        History is left empty (see module docstring). After rebuild ``board[0]``
        is us, so ``pick`` chooses our move directly.
        """
        self._in_board = False
        board = np.zeros((2, self.board_size, self.board_size), dtype=bool)
        for x, y, field in self._board_lines:
            if field == 1:
                board[0, y, x] = True
            elif field == 2:
                board[1, y, x] = True
            # field == 3 (renju marker) or anything else: not a stone, skip.
        self._board_lines = []
        self.state = GameState(board=board, move_count=int(board.sum()), history=())
        self._move_and_emit()

    # -- command dispatch ------------------------------------------------

    def handle(self, raw: str) -> bool:
        """Process one input line. Return False iff the engine should exit."""
        line = raw.rstrip("\r\n").strip()

        # Inside a BOARD block: collect field lines until DONE.
        if self._in_board:
            if line.upper() == "DONE":
                self._finish_board()
            else:
                self._collect_board_line(line)
            return True

        if not line:
            return True
        token = line.split()[0].upper()

        if token == "START":
            return self._cmd_start(line)
        if token == "INFO":
            self._cmd_info(line)
            return True
        if token == "BEGIN":
            # Empty board, we move first. (START/RESTART left an empty state.)
            self._move_and_emit()
            return True
        if token == "TURN":
            self._cmd_turn(line)
            return True
        if token == "BOARD":
            self._in_board = True
            self._board_lines = []
            return True
        if token == "ABOUT":
            self._emit(
                f'name="{self.name}", version="{self.version}", '
                f'author="{self.author}", country="USA", '
                f'www="https://github.com/jasonyandell/gomoku"'
            )
            return True
        if token == "RESTART":
            self.state = GameState.initial()
            self._emit("OK")
            return True
        if token == "END":
            return False
        # Unrecognized (incl. RECTSTART / TAKEBACK / SWAP2BOARD …): reply UNKNOWN
        # per protocol and — critically — NEVER a bare X,Y (which would be eaten
        # as our move). The manager ignores UNKNOWN/MESSAGE chatter.
        self._emit(f"UNKNOWN unsupported command: {line}")
        return True

    def _cmd_start(self, line: str) -> bool:
        parts = line.split()
        try:
            size = int(parts[1])
        except (IndexError, ValueError):
            self._emit("ERROR bad START (expected: START <size>)")
            return True
        if size != self.board_size:
            self._emit(
                f"ERROR unsupported board size {size} "
                f"(this engine is fixed at {self.board_size}x{self.board_size})"
            )
            return True
        self.state = GameState.initial()
        self.info = {}
        self._emit("OK")
        return True

    def _cmd_info(self, line: str) -> None:
        parts = line.split(None, 2)
        if len(parts) >= 3:
            self.info[parts[1].lower()] = parts[2]
        elif len(parts) == 2:
            self.info[parts[1].lower()] = ""
        # No reply (per protocol).

    def _cmd_turn(self, line: str) -> None:
        parts = line.split()
        token = parts[1] if len(parts) > 1 else ""
        xy = _parse_xy(token, self.board_size)
        if xy is None:
            self._emit("MESSAGE ignoring malformed TURN")
            return
        x, y = xy
        opp_action = y * self.board_size + x
        if opp_action in set(int(a) for a in self.state.legal_actions()):
            self.state = self.state.apply(opp_action)
        # else: opponent move illegal/occupied — ignore the apply but still reply
        # legally rather than crash or desync.
        self._move_and_emit()


# -- pickers -------------------------------------------------------------


def _first_legal_pick(state: "GameState") -> int:
    """Torch-free stub picker (lowest-index legal cell) for protocol tests."""
    return int(state.legal_actions()[0])


def _build_net_picker(args):
    """Build the real MCTS picker. Torch imports stay here (lazy) so the protocol
    layer and ``--stub-picker`` mode stay torch-free."""
    from gomoku.eval import mcts_picker
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint
    from gomoku.util import pick_device

    device = pick_device(args.device or os.environ.get("GOMOKU_DEVICE"))
    model, payload = load_checkpoint(args.checkpoint, device=device)
    if args.ema and isinstance(payload, dict) and payload.get("ema_model_state_dict"):
        try:
            model.load_state_dict(payload["ema_model_state_dict"])
            print("MESSAGE loaded EMA weights", file=sys.stderr, flush=True)
        except Exception as e:  # pragma: no cover - best-effort
            print(f"MESSAGE EMA load failed ({e}); using live weights",
                  file=sys.stderr, flush=True)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, device)
    player = mcts_picker(evaluator, n_simulations=args.sims, c_puct=args.c_puct)
    rng = np.random.default_rng(args.seed)

    def pick(state):
        return int(player(state, rng))

    return pick


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gomoku-brain",
        description="Brain-side Gomocup/Piskvork engine wrapping a trained net.",
    )
    p.add_argument("--checkpoint", help="Path to a .pt checkpoint (embeds arch).")
    p.add_argument("--sims", type=int, default=200, help="MCTS simulations per move.")
    p.add_argument("--c-puct", type=float, default=1.5, dest="c_puct")
    p.add_argument("--board-size", type=int, default=None, dest="board_size",
                   help="Must match the checkpoint; sets GOMOKU_BOARD_SIZE pre-import.")
    p.add_argument("--device", default=None, help="torch device (default: env/mps).")
    p.add_argument("--seed", type=int, default=0, help="Tie-break rng seed.")
    p.add_argument("--name", default="gomoku-az", help="Engine name for ABOUT.")
    p.add_argument("--ema", action="store_true",
                   help="Load ema_model_state_dict if present.")
    p.add_argument("--stub-picker", action="store_true", dest="stub_picker",
                   help="Lowest-legal-cell picker; no checkpoint/torch (protocol smoke).")
    return p


def _run_loop(brain: GomocupBrain, stream) -> None:
    for raw in stream:
        if not brain.handle(raw):
            break  # END


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.board_size is not None and args.board_size != BOARD_SIZE:
        # Pre-parse should have set the env; if not, fail loudly rather than
        # silently serve the wrong size.
        print(
            f"ERROR --board-size {args.board_size} but engine locked at "
            f"{BOARD_SIZE} (set GOMOKU_BOARD_SIZE before launch)",
            file=sys.stderr, flush=True,
        )
        return 2
    if args.stub_picker:
        pick = _first_legal_pick
    else:
        if not args.checkpoint:
            _build_arg_parser().error("--checkpoint is required (or use --stub-picker)")
        pick = _build_net_picker(args)
    brain = GomocupBrain(pick, board_size=BOARD_SIZE, name=args.name)
    _run_loop(brain, sys.stdin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
