"""Piskvork/Gomocup-protocol subprocess player (an external-engine YARDSTICK).

This wraps a Gomocup-protocol engine (e.g. Rapfi) as a `gomoku.eval.Picker`,
so a checkpoint can be scored against a rated external engine instead of only
against the in-repo anchor ladder (random/heuristic/lookahead). The anchor
ladder saturates around ~1700 Elo; a rated engine like Rapfi (Gomocup freestyle
Elo 2625) is the only honest yardstick past that ceiling.

EVAL-ONLY. Do not mix an external engine into self-play training.

Protocol surface used (text over stdin/stdout, zero-based X,Y coords):
- `START <size>`        -> engine replies `OK` (or `ERROR ...` if size unsupported)
- `INFO rule <id>`      -> 0 = freestyle (no reply)
- `INFO timeout_turn <ms>` -> per-move wall budget (no reply)
- per move: `BOARD` + lines `X,Y,field` (1=own/side-to-move, 2=opponent) + `DONE`
            -> engine replies a move `X,Y` (after zero or more `MESSAGE ...`
               / `DEBUG ...` chatter lines, which we skip).

Board <-> protocol mapping. `GameState` is canonical: `state.board[0]` is the
side-to-move's stones and `state.board[1]` is the opponent's. The picker is
called when it is *this* player's turn, so the external engine *is* the
side-to-move. Therefore `board[0]` stones are encoded as field `1` (own) and
`board[1]` stones as field `2` (opponent).

Coordinate mapping. An action index is `row * BOARD_SIZE + col`
(`row = action // BOARD_SIZE`, `col = action % BOARD_SIZE`). The protocol's
`X,Y` is zero-based with X = column and Y = row, so `X = col`, `Y = row`.
"""

from __future__ import annotations

import shlex
import subprocess
import threading
from dataclasses import dataclass

import numpy as np

from gomoku.game import BOARD_SIZE, GameState

WRAPPER_VERSION = "1"

# Lines the engine may emit before its move; not a coordinate reply.
_CHATTER_PREFIXES = ("MESSAGE", "DEBUG", "INFO", "ERROR", "UNKNOWN", "SUGGEST")


class ExternalEngineError(RuntimeError):
    """Raised when the external engine misbehaves (bad size, illegal move, crash)."""


def _coord_to_xy(action: int) -> tuple[int, int]:
    row, col = divmod(action, BOARD_SIZE)
    return col, row  # X=col, Y=row


def _xy_to_action(x: int, y: int) -> int:
    return y * BOARD_SIZE + x  # row=y, col=x


@dataclass
class ExternalEngineConfig:
    cmd: str
    timeout_ms: int = 1000
    label: str = "external"
    rule: int = 0  # 0 = freestyle
    board_size: int = BOARD_SIZE
    # Hard ceiling on a single move reply, independent of the engine's own
    # timeout_turn budget. Generous slack over timeout_ms so a busy engine
    # still gets to answer.
    read_timeout_s: float = 30.0


class ExternalEnginePlayer:
    """A stateful Gomocup-protocol subprocess player usable as a `Picker`.

    Spawns the engine once on construction, sends `START <size>` / `INFO rule`
    / `INFO timeout_turn`, and replays the full board to the engine on every
    move (stateless from the engine's point of view — robust to color
    alternation across games). Call `close()` when done.
    """

    def __init__(self, config: ExternalEngineConfig):
        self.config = config
        argv = shlex.split(config.cmd)
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError as e:
            raise ExternalEngineError(f"cannot launch engine: {config.cmd!r}: {e}") from e

        self._handshake()

    # -- protocol I/O ----------------------------------------------------

    def _send(self, line: str) -> None:
        if self._proc.poll() is not None:
            raise ExternalEngineError("engine process has exited")
        assert self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _read_line(self, timeout_s: float) -> str:
        """Read one stdout line, with a hard wall-clock timeout."""
        assert self._proc.stdout is not None
        result: list[str | None] = [None]

        def _do_read() -> None:
            result[0] = self._proc.stdout.readline()  # type: ignore[union-attr]

        t = threading.Thread(target=_do_read, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            raise ExternalEngineError(
                f"engine timed out after {timeout_s:.1f}s waiting for a reply"
            )
        line = result[0]
        if line == "" or line is None:
            raise ExternalEngineError("engine closed stdout (EOF) unexpectedly")
        return line.strip()

    def _read_until_coord_or_ok(self, *, expect_ok: bool) -> str:
        """Read lines, skipping chatter, until a coordinate or OK line.

        If `expect_ok`, an `OK` reply is the success token; an `ERROR ...`
        line is a hard failure. Otherwise we want a coordinate reply (and an
        `ERROR ...` is also a hard failure).
        """
        deadline = self.config.read_timeout_s
        while True:
            line = self._read_line(deadline)
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("ERROR"):
                raise ExternalEngineError(f"engine returned: {line}")
            if expect_ok:
                if upper == "OK":
                    return line
                # Some chatter before OK is tolerated.
                continue
            # Want a coordinate; skip known chatter prefixes.
            if any(upper.startswith(p) for p in _CHATTER_PREFIXES):
                continue
            return line

    def _handshake(self) -> None:
        cfg = self.config
        self._send(f"START {cfg.board_size}")
        try:
            self._read_until_coord_or_ok(expect_ok=True)
        except ExternalEngineError as e:
            raise ExternalEngineError(
                f"engine rejected START {cfg.board_size} "
                f"(not a {cfg.board_size}x{cfg.board_size} fit): {e}"
            ) from e
        self._send(f"INFO rule {cfg.rule}")
        self._send(f"INFO timeout_turn {cfg.timeout_ms}")
        # timeout_match 0 => no whole-game cap; rely on per-turn budget.
        self._send("INFO timeout_match 0")

    # -- Picker interface ------------------------------------------------

    def __call__(self, state: GameState, rng: np.random.Generator) -> int:
        """Return the engine's chosen flat action index for `state`.

        `rng` is unused (the engine is deterministic-ish given timeout); it is
        in the signature to satisfy the `Picker` protocol.
        """
        del rng
        own = state.board[0]  # side-to-move == this engine -> field 1
        opp = state.board[1]  # opponent -> field 2
        legal = set(int(a) for a in state.legal_actions())

        self._send("BOARD")
        for y in range(self.config.board_size):
            for x in range(self.config.board_size):
                if own[y, x]:
                    self._send(f"{x},{y},1")
                elif opp[y, x]:
                    self._send(f"{x},{y},2")
        self._send("DONE")

        reply = self._read_until_coord_or_ok(expect_ok=False)
        action = self._parse_move(reply, legal)
        return action

    def _parse_move(self, reply: str, legal: set[int]) -> int:
        # Reply may be "X,Y" or (for swap variants) more tokens; take the
        # first coordinate token only.
        tok = reply.split()[0]
        if "," not in tok:
            raise ExternalEngineError(f"unparseable move reply: {reply!r}")
        parts = tok.split(",")
        try:
            x = int(parts[0])
            y = int(parts[1])
        except (ValueError, IndexError) as e:
            raise ExternalEngineError(f"bad coordinate in reply {reply!r}: {e}") from e
        n = self.config.board_size
        if not (0 <= x < n and 0 <= y < n):
            raise ExternalEngineError(f"move {x},{y} out of range for {n}x{n}")
        action = _xy_to_action(x, y)
        if action not in legal:
            raise ExternalEngineError(
                f"engine returned illegal/occupied move {x},{y} (action {action})"
            )
        return action

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._send("END")
            except ExternalEngineError:
                pass
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- provenance ------------------------------------------------------

    def provenance(self) -> dict:
        """Explicit fields for eval JSONL records."""
        return {
            "engine": self.config.label,
            "cmd": self.config.cmd,
            "timeout_ms": self.config.timeout_ms,
            "board_size": self.config.board_size,
            "rule": self.config.rule,
            "wrapper_version": WRAPPER_VERSION,
        }


def build_external_player(kwargs: dict[str, str]) -> ExternalEnginePlayer:
    """Construct an ExternalEnginePlayer from a parsed player-spec kwargs dict.

    Recognised keys: cmd (required), timeout_ms (default 1000), label,
    rule (default 0 = freestyle), size (default BOARD_SIZE).
    """
    cmd = kwargs.get("cmd")
    if not cmd:
        raise SystemExit("external spec needs cmd=PATH")
    timeout_ms = int(kwargs.get("timeout_ms", "1000"))
    label = kwargs.get("label", "external")
    rule = int(kwargs.get("rule", "0"))
    size = int(kwargs.get("size", str(BOARD_SIZE)))
    cfg = ExternalEngineConfig(
        cmd=cmd, timeout_ms=timeout_ms, label=label, rule=rule, board_size=size
    )
    return ExternalEnginePlayer(cfg)
