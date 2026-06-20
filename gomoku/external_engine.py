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
from dataclasses import dataclass, field
from enum import Enum

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


# ---------------------------------------------------------------------------
# Swap2 opening negotiation (the SWAP2BOARD protocol path).
#
# The manager drives a swap2 opening by sending the engine a `SWAP2BOARD`
# block — the literal line `SWAP2BOARD`, then 0 / 3 / 5 stone lines in board
# order, then `DONE` — and reading ONE reply line. The arity of the reply is
# fixed by how many stones were on the board (see `_read_swap2_reply`):
#
#   0 stones (engine OPENS)       -> three coords "x,y x,y x,y" (2B+1W)
#   3 stones (engine RESPONDS)    -> `SWAP` | one coord | two coords
#   5 stones (engine PICKS color) -> `SWAP` | one coord
#
# `SWAP` and `DONE` are exact-uppercase literals; `SWAP` is NOT a coordinate
# (it has no comma) so the positive-match `_parse_coord` gate skips it as
# chatter — we test for the `SWAP` token explicitly before coordinate parsing.
# Coords are zero-based X=col,Y=row, mapped to actions via `_xy_to_action`,
# identical to the move path.
# ---------------------------------------------------------------------------

_SWAP2_TOKEN = "SWAP"

# Allowed reply arities (number of coords) keyed by stones already on the board.
# `None` is the SWAP literal, which is admissible alongside the coord arities.
_SWAP2_ARITY: dict[int, tuple[int, ...]] = {
    0: (3,),          # opener: must place exactly 3 opening stones
    3: (1, 2),        # responder: keep-color (1) or place-two (2); or SWAP
    5: (1,),          # opener picks: play the move (1); or SWAP
}
# Stone counts whose reply may legally be the bare `SWAP` literal.
_SWAP2_SWAP_OK: frozenset[int] = frozenset({3, 5})


class Swap2Option(Enum):
    """Which swap2 reply the engine gave, disambiguated from the stone count."""

    # Engine OPENED: it returned the 2B+1W opening placement (3 coords).
    OPEN_THREE = "open_three"
    # Engine took the opposite color (the literal `SWAP`); no new stone.
    SWAP = "swap"
    # Engine kept its color / played the next single move (1 coord). As a
    # responder this is the "keep color, play the 4th move" reply; as a picker
    # it is the chosen continuation move.
    ONE_COORD = "one_coord"
    # Engine (as responder) placed two more stones (4th + 5th); the OPPONENT
    # then picks a color (our PLACE2-equivalent reply from the engine's side).
    TWO_COORDS = "two_coords"


@dataclass
class Swap2Reply:
    """A parsed, validated swap2 negotiation reply.

    `coords` are the engine's returned `(x, y)` pairs in board/reply order
    (empty for `SWAP`). `actions` maps them to flat action indices with the
    same `action = row*BOARD_SIZE + col` convention as the move path. The eval
    harness branches on `option` and consumes `coords` / `actions`.
    """

    option: Swap2Option
    coords: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    @property
    def is_swap(self) -> bool:
        return self.option is Swap2Option.SWAP

    @property
    def actions(self) -> tuple[int, ...]:
        return tuple(_xy_to_action(x, y) for x, y in self.coords)


def _parse_coord_list(line: str) -> list[tuple[int, int]] | None:
    """Parse a whitespace-separated list of bare `x,y` coords, or None.

    Every whitespace token must itself be a coordinate; one non-coord token
    (a banner word, `SWAP`, ...) disqualifies the whole line, so chatter is
    never half-accepted. Returns the list (length >= 1) on success.
    """
    toks = line.split()
    if not toks:
        return None
    coords: list[tuple[int, int]] = []
    for tok in toks:
        c = _parse_coord(tok)
        if c is None:
            return None
        coords.append(c)
    return coords


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
    # Incremental TURN-mode driving (default off). When True, after an initial
    # BOARD sync we feed the opponent's single new move as `TURN x,y` rather than
    # re-dumping the whole board every move. This lets a STATEFUL brain engine
    # (our own `gomocup_brain`) accumulate real move history via `apply()`, which
    # a history-conditioned net's recency input planes need. A full BOARD re-dump
    # every move gives the brain EMPTY history — a self-contradictory, OOD input
    # (full board, zero recency) that craters strength: measured 100% -> 25% vs
    # the heuristic on the same checkpoint. Classical external engines don't use
    # history planes, so they stay on the robust re-entrant BOARD path
    # (incremental=False); only re-set this for our own brain wrapper.
    incremental: bool = False


class ExternalEnginePlayer:
    """A stateful Gomocup-protocol subprocess player usable as a `Picker`.

    Spawns the engine once on construction, sends `START <size>` / `INFO rule`
    / `INFO timeout_turn`, and replays the full board to the engine on every
    move (stateless from the engine's point of view — robust to color
    alternation across games). Call `close()` when done.
    """

    def __init__(self, config: ExternalEngineConfig):
        self.config = config
        # Incremental-mode tracking: our believed board (sets of (x,y) cells) as
        # of our last reply, so the next call can diff out the opponent's single
        # new move and send it as `TURN x,y`. None until the first synced move.
        self._incremental = config.incremental
        self._prev_own: set[tuple[int, int]] | None = None
        self._prev_opp: set[tuple[int, int]] | None = None
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

    def _read_swap2_reply(self, n_stones_on_board: int) -> Swap2Reply:
        """Read one swap2 negotiation reply, classified + arity-checked.

        Skips leading chatter exactly like `_read_move`: a line is taken as the
        reply only when it is positively the `SWAP` literal OR a whitespace-list
        of bare `x,y` coords; everything else (banners, MESSAGE/DEBUG/ERROR
        diagnostics, lone `?`) is skipped. The first qualifying line is the
        reply.

        `n_stones_on_board` is how many stones we sent in the SWAP2BOARD block
        (0 / 3 / 5); it fixes the legal arity. A `SWAP` reply is only legal at
        3 or 5 stones; a coord count outside `_SWAP2_ARITY[n]` raises
        `ExternalEngineError` (arity mismatch). An unknown stone count raises.
        """
        if n_stones_on_board not in _SWAP2_ARITY:
            raise ExternalEngineError(
                f"swap2 reply requested for unsupported stone count "
                f"{n_stones_on_board} (expected one of {sorted(_SWAP2_ARITY)})"
            )
        allowed = _SWAP2_ARITY[n_stones_on_board]
        deadline = self._read_deadline_s()
        for _ in range(_MAX_CHATTER_LINES):
            line = self._read_line(deadline)
            if not line:
                continue
            if line.strip().upper() == _SWAP2_TOKEN:
                if n_stones_on_board not in _SWAP2_SWAP_OK:
                    raise ExternalEngineError(
                        f"swap2: engine replied SWAP at {n_stones_on_board} "
                        f"stones, where SWAP is not a legal reply"
                    )
                return Swap2Reply(option=Swap2Option.SWAP)
            coords = _parse_coord_list(line)
            if coords is None:
                # Chatter (banner / MESSAGE / DEBUG / ERROR diagnostic / '?') —
                # skip and keep reading for the real reply, just like _read_move.
                continue
            self._validate_swap2_coords(coords, n_stones_on_board)
            n = len(coords)
            if n not in allowed:
                raise ExternalEngineError(
                    f"swap2: engine returned {n} coord(s) at "
                    f"{n_stones_on_board} stones; expected {allowed} (or SWAP)"
                )
            option = {
                3: Swap2Option.OPEN_THREE,
                1: Swap2Option.ONE_COORD,
                2: Swap2Option.TWO_COORDS,
            }[n]
            return Swap2Reply(option=option, coords=tuple(coords))
        raise ExternalEngineError(
            f"engine emitted {_MAX_CHATTER_LINES}+ lines without a swap2 reply"
        )

    def _validate_swap2_coords(
        self, coords: list[tuple[int, int]], n_stones_on_board: int
    ) -> None:
        """Range-check returned coords (no occupancy check: we don't track the
        engine-side board here, only that every coord is on the board)."""
        n = self.config.board_size
        for x, y in coords:
            if not (0 <= x < n and 0 <= y < n):
                raise ExternalEngineError(
                    f"swap2: returned coord {x},{y} out of range for {n}x{n}"
                )

    def _send_swap2board(self, stones: list[tuple[int, int]]) -> None:
        """Send a SWAP2BOARD block: the literal, the stones in order, then DONE.

        `stones` are `(x, y)` pairs in the board order the spec expects (0 for
        the engine-opens probe, 3 for a responder query, 5 for a color pick).
        Each is sent as a bare `x,y` line (the spec form Rapfi drives on).
        RESTART first so the block is a fresh, re-entrant initialisation — same
        rationale as the move path (see module docstring).
        """
        self._send("RESTART")
        self._expect_ok()
        self._send("SWAP2BOARD")
        for x, y in stones:
            self._send(f"{x},{y}")
        self._send("DONE")

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

    # -- swap2 opening negotiation --------------------------------------

    def swap2_open(self) -> Swap2Reply:
        """Ask the engine to OPEN: place the 2B+1W opening (0 stones in).

        Sends `SWAP2BOARD` + no stones + `DONE`; returns a `Swap2Reply` with
        `option=OPEN_THREE` and the three `(x, y)` coords (and `.actions`).
        """
        self._send_swap2board([])
        return self._read_swap2_reply(0)

    def swap2_respond(self, stones: list[tuple[int, int]]) -> Swap2Reply:
        """Ask the engine to RESPOND to our 3-stone opening.

        `stones` are our 3 opening stones as `(x, y)` pairs in board order
        (2 black + 1 white by placement order, per the spec). Returns a
        `Swap2Reply`: `SWAP` (take the other color), `ONE_COORD` (keep color /
        play the 4th move), or `TWO_COORDS` (place 4th+5th, we then pick a
        color). Raises if we did not pass exactly 3 stones.
        """
        if len(stones) != 3:
            raise ExternalEngineError(
                f"swap2_respond expects exactly 3 stones, got {len(stones)}"
            )
        self._send_swap2board(list(stones))
        return self._read_swap2_reply(3)

    def swap2_pick(self, stones: list[tuple[int, int]]) -> Swap2Reply:
        """Ask the engine to PICK a color after we did PLACE2 (5 stones in).

        `stones` are the 5 stones on the board (3 black + 2 white by placement
        order). Returns a `Swap2Reply`: `SWAP` (take the other color) or
        `ONE_COORD` (keep color and play the next move). Raises if we did not
        pass exactly 5 stones.
        """
        if len(stones) != 5:
            raise ExternalEngineError(
                f"swap2_pick expects exactly 5 stones, got {len(stones)}"
            )
        self._send_swap2board(list(stones))
        return self._read_swap2_reply(5)

    # -- Picker interface ------------------------------------------------

    def __call__(self, state: GameState, rng: np.random.Generator) -> int:
        """Return the engine's chosen flat action index for `state`.

        `rng` is unused (the engine is deterministic-ish given timeout); it is
        in the signature to satisfy the `Picker` protocol.
        """
        del rng
        n = self.config.board_size
        own = state.board[0]  # side-to-move == this engine -> field 1
        opp = state.board[1]  # opponent -> field 2
        legal = set(int(a) for a in state.legal_actions())
        own_cells = {(x, y) for y in range(n) for x in range(n) if own[y, x]}
        opp_cells = {(x, y) for y in range(n) for x in range(n) if opp[y, x]}

        if self._incremental and self._can_turn(own_cells, opp_cells):
            # Clean incremental continuation: the opponent added exactly one stone
            # since our last reply and our stones are unchanged. Feed just that
            # move as `TURN x,y` — NO RESTART (which would wipe the engine's
            # history). A stateful brain applies it via apply() and keeps real
            # move history (the whole point of incremental mode).
            ox, oy = next(iter(opp_cells - self._prev_opp))
            self._send(f"TURN {ox},{oy}")
            x, y = self._read_move()
        else:
            # (Re)sync via a full board replay. In BOARD mode this is every move;
            # in incremental mode only the first move of a game or a detected
            # desync (a fresh/smaller board). RESTART makes the replay re-entrant
            # (engines whose BOARD is one-shot otherwise desync) and stops stale
            # "ERROR my move [..]" diagnostics. See module docstring.
            self._send("RESTART")
            self._expect_ok()
            if not own_cells and not opp_cells:
                # Truly empty board: engine makes the opening move. Some engines
                # (Zetor) resign on an empty BOARD/DONE; BEGIN is correct here.
                self._send("BEGIN")
            else:
                self._send("BOARD")
                for x, y in sorted(own_cells):
                    self._send(f"{x},{y},1")
                for x, y in sorted(opp_cells):
                    self._send(f"{x},{y},2")
                self._send("DONE")
            x, y = self._read_move()

        action = self._to_action(x, y, legal)
        if self._incremental:
            # Remember our believed board AFTER this reply: our stones now include
            # the move we just played; the opponent's are unchanged until they
            # reply. Next call diffs against this to recover their single move.
            self._prev_own = set(own_cells) | {(x, y)}
            self._prev_opp = set(opp_cells)
        return action

    def _can_turn(
        self, own_cells: set[tuple[int, int]], opp_cells: set[tuple[int, int]]
    ) -> bool:
        """True iff this is a clean incremental continuation of the prior move.

        Requires that since our last reply (a) our own stones are unchanged and
        (b) the opponent added EXACTLY ONE stone (a superset by one). Anything
        else — first move of a game, a new game (smaller board), or any
        mismatch — returns False so `__call__` falls back to a full BOARD resync.
        """
        if self._prev_own is None or self._prev_opp is None:
            return False
        if own_cells != self._prev_own:
            return False
        if not self._prev_opp <= opp_cells:
            return False
        return len(opp_cells - self._prev_opp) == 1

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
            "incremental": self.config.incremental,
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
    # incremental=1 drives a stateful brain via TURN (keeps move history); only
    # set it for our own gomocup_brain wrapper, never for classical engines.
    incremental = kwargs.get("incremental", "0").lower() not in ("0", "false", "no", "")
    cfg = ExternalEngineConfig(
        cmd=cmd, timeout_ms=timeout_ms, label=label, rule=rule, board_size=size,
        incremental=incremental,
    )
    return ExternalEnginePlayer(cfg)
