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
- per move: `RESTART` (reply `OK`) + `BOARD` + lines `X,Y,field`
            (1=own/side-to-move, 2=opponent) + `DONE`
            -> engine replies a move `X,Y` (after zero or more chatter lines,
               which we skip — see "Reading the move" below).
            On a *truly empty* board (engine to make the opening move) we send
            `BEGIN` instead of an empty `BOARD`/`DONE` — some engines (Zetor)
            resign rather than answer an empty `BOARD`.

Why `RESTART` before every `BOARD`. We replay the *full* board on every move
(stateless from the engine's point of view, robust to color alternation across
games). But several Gomocup engines treat `BOARD` as a one-shot full-board
*initialisation* that may only be used once per game — a second `BOARD` on a
non-empty internal board desyncs them. Observed real failures (15x15, via wine):

- Zetor2017: a second `BOARD` emits ``ERROR Board isn't initialized. Use:
  'START size'`` and then reads the following ``X,Y,field`` stone lines as
  top-level commands (``UNKNOWN command '7,7,2'``) — the desync where board
  lines get sent as commands.
- Yixin2018 / Pela2023 / Eulring2016: re-feeding a board whose history they
  already tracked makes them emit benign-but-confusing diagnostics like
  ``ERROR my move [7,7]`` / ``ERROR opponents's move [7,7]`` *before* the real
  coordinate reply.

``RESTART`` (canonical Piskvork "reset board to empty, keep size + settings")
fixes both at the root: it clears the engine's internal board/history so the
``BOARD`` block is always a fresh initialisation, and the engines stop emitting
the spurious ``ERROR my move`` chatter. It replies ``OK`` (which we consume) and
is accepted by every engine tested, including Rapfi (the previously-working
yardstick).

Reading the move. The move reply is the FIRST line that parses as a bare
``X,Y`` coordinate. Everything else is skipped as chatter — not just the known
``MESSAGE`` / ``DEBUG`` / ``INFO`` prefixes but ANY non-coordinate line,
including engine-name banners, ``DATABASE`` notices, lone ``?``, ``SUGGEST``,
and (when waiting for a move, not the handshake) stray ``ERROR ...`` diagnostics
that some engines emit non-fatally. A bounded skip count guards against an
engine that never answers.

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

WRAPPER_VERSION = "2"

# Safety cap on how many non-coordinate chatter lines we skip while waiting for
# a single move/OK reply before giving up (guards against an engine that streams
# diagnostics forever and never answers).
_MAX_CHATTER_LINES = 2000


class ExternalEngineError(RuntimeError):
    """Raised when the external engine misbehaves (bad size, illegal move, crash)."""


def _parse_coord(line: str) -> tuple[int, int] | None:
    """Return ``(x, y)`` if ``line`` IS a bare Gomocup coordinate reply, else None.

    A move reply is exactly an ``X,Y`` pair (optionally with trailing swap-variant
    tokens separated by whitespace, e.g. ``"7,7 ..."``). Anything else — banners,
    ``MESSAGE``/``DEBUG``/``INFO``/``ERROR``/``UNKNOWN``/``SUGGEST``/``DATABASE``
    chatter, a lone ``?``, ``my move [7,7]`` style diagnostics — returns None and
    is skipped. This positive-match gate is what makes the reader robust: we take
    a line as the move ONLY when it parses as a coordinate, never by prefix
    exclusion.
    """
    tok = line.split()[0] if line.split() else ""
    if "," not in tok:
        return None
    parts = tok.split(",")
    if len(parts) < 2:
        return None
    try:
        x = int(parts[0])
        y = int(parts[1])
    except ValueError:
        return None
    return x, y


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
    # Hard floor on the per-reply wall-clock ceiling, independent of the engine's
    # own timeout_turn budget. The effective ceiling (see `_read_deadline_s`) is
    # max(read_timeout_s, timeout_ms/1000 * read_timeout_slack) so a slow engine
    # under a long timeout_turn still gets to answer, and a GPU-contended engine
    # (e.g. Vulkan-backed Embryo) gets generous slack over its move budget rather
    # than tripping the old hard 30s wall.
    read_timeout_s: float = 30.0
    # Multiplier applied to the per-move budget (timeout_ms) to derive the read
    # deadline. 3x leaves room for wine startup jitter + GPU contention.
    read_timeout_slack: float = 3.0


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

    def _read_deadline_s(self) -> float:
        """Effective per-reply wall-clock ceiling.

        ``max(read_timeout_s, timeout_ms/1000 * read_timeout_slack)`` — a hard
        floor for fast engines, but scales up with the engine's own move budget
        so a slow / GPU-contended engine under a long ``timeout_turn`` still gets
        to answer instead of tripping a fixed wall.
        """
        cfg = self.config
        budget_s = (cfg.timeout_ms / 1000.0) * cfg.read_timeout_slack
        return max(cfg.read_timeout_s, budget_s)

    def _expect_ok(self) -> None:
        """Read until an ``OK`` line. Here ``ERROR ...`` IS a hard failure.

        Used for the ``START`` / ``RESTART`` confirmations, where ``ERROR``
        genuinely signals rejection (e.g. an unsupported board size). Chatter
        (banners, ``MESSAGE`` lines) before ``OK`` is tolerated.
        """
        deadline = self._read_deadline_s()
        for _ in range(_MAX_CHATTER_LINES):
            line = self._read_line(deadline)
            if not line:
                continue
            upper = line.upper()
            if upper == "OK":
                return
            if upper.startswith("ERROR"):
                raise ExternalEngineError(f"engine returned: {line}")
            # Tolerate banners / MESSAGE / DEBUG chatter before OK.
        raise ExternalEngineError(
            f"engine emitted {_MAX_CHATTER_LINES}+ lines without an OK reply"
        )

    def _read_move(self) -> tuple[int, int]:
        """Read until a bare ``X,Y`` coordinate, skipping ALL other lines.

        The move reply is identified POSITIVELY (it must parse as a coordinate),
        so any chatter is skipped regardless of prefix — including stray
        non-fatal ``ERROR ...`` diagnostics (``ERROR my move [7,7]``) that some
        engines emit before their real move. A bare ``X,Y`` is the only thing
        taken as the answer. If the engine closes stdout (EOF) we surface that
        as a hard error; a chatter flood past ``_MAX_CHATTER_LINES`` also fails.
        """
        deadline = self._read_deadline_s()
        for _ in range(_MAX_CHATTER_LINES):
            line = self._read_line(deadline)
            if not line:
                continue
            coord = _parse_coord(line)
            if coord is not None:
                return coord
            # else: chatter (banner / MESSAGE / DEBUG / INFO / ERROR diagnostic /
            # DATABASE / '?' / 'my move [..]' / SUGGEST ...) — skip and keep
            # reading for the actual coordinate.
        raise ExternalEngineError(
            f"engine emitted {_MAX_CHATTER_LINES}+ lines without a coordinate move"
        )

    def _handshake(self) -> None:
        cfg = self.config
        self._send(f"START {cfg.board_size}")
        try:
            self._expect_ok()
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

        # Reset the engine's internal board+history before replaying the full
        # position. RESTART makes our "replay full board every move" strategy
        # re-entrant (engines whose BOARD is one-shot otherwise desync) and stops
        # them emitting stale "ERROR my move [..]" diagnostics. See module docstring.
        self._send("RESTART")
        self._expect_ok()

        # Collect occupied cells once so we can branch on the empty-board case.
        own_cells = [
            (x, y)
            for y in range(self.config.board_size)
            for x in range(self.config.board_size)
            if own[y, x]
        ]
        opp_cells = [
            (x, y)
            for y in range(self.config.board_size)
            for x in range(self.config.board_size)
            if opp[y, x]
        ]

        if not own_cells and not opp_cells:
            # Truly empty board: the engine makes the opening move. Some engines
            # (Zetor) resign on an empty BOARD/DONE; BEGIN is the correct command
            # to ask for the first move. (With --random-opening-moves > 0 this
            # branch never triggers, but the picker must still be correct for the
            # cold-start path.)
            self._send("BEGIN")
        else:
            self._send("BOARD")
            for x, y in own_cells:
                self._send(f"{x},{y},1")
            for x, y in opp_cells:
                self._send(f"{x},{y},2")
            self._send("DONE")

        x, y = self._read_move()
        return self._to_action(x, y, legal)

    def _to_action(self, x: int, y: int, legal: set[int]) -> int:
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
