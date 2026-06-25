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

import queue
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
# diagnostics forever and never answers). Sized for the QUIET single-reply
# protocols (_expect_ok / _read_move / _read_swap2_reply): a few banner/MESSAGE
# lines, never thousands.
_MAX_CHATTER_LINES = 2000

# A multiPV analysis stream is a DIFFERENT shape: iterative deepening RE-EMITS
# all `pv_count` PV blocks every depth round, so its legitimate size scales with
# pv_count × (number of emitting depth rounds). A forced-mate position deepens
# FAR (measured: an 8-stone tactical board with EVAL -M4 ran to DEPTH 26 →
# 2302 lines, terminator and all — past the 2000 quiet-belt, hence the crash).
# So `_read_analysis` gets its OWN pv-scaled cap, not the quiet belt. Budget is
# generous lines-per-PV (≈ deepening rounds × ~7 lines/block, ×safety): the real
# terminator (a bare-coord bestmove) reliably arrives once max_node bounds the
# search, so this is only a runaway backstop. Line-count (not wall-clock) keeps
# it contention-immune — a saturated 60-engine pool won't false-trip a counter.
_ANALYSIS_LINES_PER_PV = 600


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


def _eval_to_winrate(token: str) -> float | None:
    """Map an ``INFO EVAL`` token to a side-to-move winrate fallback.

    Used ONLY when ``INFO WINRATE`` is absent (mate / infinite scores, where
    Rapfi may emit a symbolic EVAL): ``+M<k>`` / ``VAL_INF`` with a ``+`` sign
    -> 1.0 (we win), ``-M<k>`` / mated / ``VAL_INF`` with a ``-`` sign -> 0.0.
    A plain integer EVAL returns None (the real WINRATE line is preferred and
    always present alongside it); an unparseable token returns None.
    """
    if not token:
        return None
    t = token.strip().upper()
    if t.startswith("+M") or t == "VAL_INF" or t.startswith("+VAL_INF"):
        return 1.0
    if t.startswith("-M") or t.startswith("-VAL_INF") or t == "MATED":
        return 0.0
    return None


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
# Stone lines carry COLOR. Each opening stone is sent as `x,y,color` (the same
# `x,y,field` grammar as the move path's BOARD block), with color in {1, 2}:
# **1 = black, 2 = white**. Rapfi REJECTS bare `x,y` stone lines with
# ``ERROR Color is not a valid value, must be one of [1, 2, 3]`` and then opens
# on an empty board (an arity mismatch downstream). With `x,y,color` Rapfi loads
# its NNUE and negotiates correctly (confirmed by driving the real binary:
# OPEN -> 3 coords, RESPOND -> SWAP|1 coord, PICK -> SWAP|1 coord, no errors).
# The opening is 2 black + 1 white (then +1 or +2 stones); colors are read off
# the absolute (black, white) planes, never assumed from placement parity.
#
# `SWAP` and `DONE` are exact-uppercase literals; `SWAP` is NOT a coordinate
# (it has no comma) so the positive-match `_parse_coord` gate skips it as
# chatter — we test for the `SWAP` token explicitly before coordinate parsing.
# Reply coords carry NO color (they are coords/SWAP only), so the reply parser
# is unchanged. Coords are zero-based X=col,Y=row, mapped to actions via
# `_xy_to_action`, identical to the move path.
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

        # ONE persistent reader thread per engine drains stdout into a queue, so
        # _read_line is a cheap queue.get instead of spawning a fresh thread PER
        # LINE. A multiPV analysis emits ~2000 lines; the old thread-per-line cost
        # (~0.024ms × 2000 ≈ 50ms) dominated each analyze and starved the engine.
        # The reader exits on EOF (puts a None sentinel). daemon=True so it never
        # blocks interpreter exit.
        self._rq: "queue.Queue[str | None]" = queue.Queue()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        self._handshake()

    def _reader_loop(self) -> None:
        out = self._proc.stdout
        assert out is not None
        try:
            for line in out:            # blocks in C; yields complete lines
                self._rq.put(line)
        except (ValueError, OSError):
            pass                        # stdout closed under us
        finally:
            self._rq.put(None)          # EOF sentinel

    # -- protocol I/O ----------------------------------------------------

    def _send(self, line: str) -> None:
        if self._proc.poll() is not None:
            raise ExternalEngineError("engine process has exited")
        assert self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def _read_line(self, timeout_s: float) -> str:
        """Read one stdout line, with a hard wall-clock timeout.

        Pulls from the persistent reader thread's queue (one thread per engine,
        started in __init__) — NOT a fresh thread per line. Same semantics:
        timeout -> 'timed out'; EOF sentinel -> 'closed stdout'.
        """
        try:
            line = self._rq.get(timeout=timeout_s)
        except queue.Empty:
            raise ExternalEngineError(
                f"engine timed out after {timeout_s:.1f}s waiting for a reply"
            )
        if line is None or line == "":
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

    def _send_swap2board(self, stones: list[tuple[int, int, int]]) -> None:
        """Send a SWAP2BOARD block: the literal, the stones in order, then DONE.

        `stones` are `(x, y, color)` triples in the board order the spec expects
        (0 for the engine-opens probe, 3 for a responder query, 5 for a color
        pick). COLOR is in {1, 2} (1 = black, 2 = white). Each is sent as an
        `x,y,color` line — the same `x,y,field` grammar as the move-path BOARD
        block; Rapfi rejects a bare `x,y` stone line outright (see module
        docstring). RESTART first so the block is a fresh, re-entrant
        initialisation — same rationale as the move path.
        """
        self._send("RESTART")
        self._expect_ok()
        self._send("SWAP2BOARD")
        for x, y, color in stones:
            if color not in (1, 2):
                raise ExternalEngineError(
                    f"swap2 stone color must be 1 (black) or 2 (white), got {color}"
                )
            self._send(f"{x},{y},{color}")
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

    def swap2_respond(self, stones: list[tuple[int, int, int]]) -> Swap2Reply:
        """Ask the engine to RESPOND to our 3-stone opening.

        `stones` are our 3 opening stones as `(x, y, color)` triples in board
        order (2 black + 1 white, with color in {1, 2}). Returns a `Swap2Reply`:
        `SWAP` (take the other color), `ONE_COORD` (keep color / play the 4th
        move), or `TWO_COORDS` (place 4th+5th, we then pick a color). Raises if
        we did not pass exactly 3 stones.
        """
        if len(stones) != 3:
            raise ExternalEngineError(
                f"swap2_respond expects exactly 3 stones, got {len(stones)}"
            )
        self._send_swap2board(list(stones))
        return self._read_swap2_reply(3)

    def swap2_pick(self, stones: list[tuple[int, int, int]]) -> Swap2Reply:
        """Ask the engine to PICK a color after we did PLACE2 (5 stones in).

        `stones` are the 5 stones on the board as `(x, y, color)` triples (3
        black + 2 white, with color in {1, 2}; each stone's true color is read
        off the OpeningState planes, not assumed from placement order). Returns
        a `Swap2Reply`: `SWAP` (take the other color) or `ONE_COORD` (keep color
        and play the next move). Raises if we did not pass exactly 5 stones.
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

    # -- whole-board analysis (multiPV soft-target distillation) ---------

    def analyze(
        self, state: GameState, *, max_node: int = 20000, max_pv: int | None = None
    ) -> dict[int, float]:
        """Score EVERY candidate root move for ``state`` -> ``{action: winrate}``.

        Rapfi's gomocup protocol can emit a per-move EVAL+WINRATE for the whole
        board (a multiPV search) — the signal the SOFT teacher distils. We DRIVE
        the existing binary; nothing in the engine is rebuilt (GPL untouched).
        Returns winrates in [0, 1] from the SIDE-TO-MOVE's perspective, keyed by
        the flat action index ``y*BOARD_SIZE + x``. Pruned-away (far-from-action)
        cells are simply absent — their mass is correctly ~0.

        Drive sequence (a one-shot stateless BOARD dump; assumes RESTART-fresh):
        set a small node budget + SHOW_DETAIL/CAUTION so the search both emits the
        per-PV blocks and widens its candidate range, dump the position, then
        ``YXNBEST 999`` to run multiPV = movesLeft(). ``DONE`` triggers a stray
        default multiPV=1 think whose bestmove reply we DRAIN and discard. Each
        ``INFO PV <idx>`` block carries an ``INFO EVAL`` / ``INFO WINRATE`` /
        ``INFO BESTLINE <root> ...``; iterative deepening RE-EMITS every PV per
        depth, so we keep the LAST block seen for each ``PV <idx>``. The whole
        analysis is terminated by a bare ``x,y`` bestmove line.

        Only meaningful for a stateless (``incremental=False``) engine: we always
        RESTART + full-BOARD dump, carrying NO state between calls (the same
        re-entrant contract the pool relies on for safe reuse).
        """
        n = self.config.board_size
        own = state.board[0]  # side-to-move == this engine -> field 1
        opp = state.board[1]  # opponent -> field 2
        own_cells = {(x, y) for y in range(n) for x in range(n) if own[y, x]}
        opp_cells = {(x, y) for y in range(n) for x in range(n) if opp[y, x]}
        moves_left = n * n - len(own_cells) - len(opp_cells)
        if moves_left <= 0:
            return {}

        # Configure multiPV-friendly analysis. max_node bounds the search so a
        # whole-board scan stays fast; SHOW_DETAIL 2 emits the INFO PV/EVAL/WINRATE
        # blocks; CAUTION_FACTOR widens the candidate set toward the whole board.
        self._send(f"INFO max_node {int(max_node)}")
        self._send("INFO SHOW_DETAIL 2")
        self._send("INFO CAUTION_FACTOR 5")
        # Re-entrant full-board dump (same contract as the move path's BOARD mode).
        self._send("RESTART")
        self._expect_ok()
        self._send("BOARD")
        for x, y in sorted(own_cells):
            self._send(f"{x},{y},1")
        for x, y in sorted(opp_cells):
            self._send(f"{x},{y},2")
        self._send("DONE")
        # DONE triggers a default multiPV=1 think — read & discard its bestmove.
        self._read_move()
        # Now run the multiPV scan and parse it. ``max_pv`` caps the number of PV
        # lines (the soft target's scored support): the whole-board scan is
        # ``YXNBEST moves_left``, but on a near-empty board that is 200+ lines and
        # the per-node cost balloons. Capping to the top ``max_pv`` moves keeps the
        # support information-equivalent for a temperature-softmax target (far cells
        # carry ~0 mass) while bounding cost so a warm pool stays saturated.
        pv_count = moves_left if max_pv is None else max(1, min(int(max_pv), moves_left))
        self._send(f"YXNBEST {pv_count}")
        return self._read_analysis(pv_count=pv_count)

    def _read_analysis(self, *, pv_count: int) -> dict[int, float]:
        """Parse the multiPV stream until the terminating bare ``x,y`` bestmove.

        Keeps, per ``PV <idx>``, the winrate from the LAST (deepest) block — the
        root move is the FIRST token of that block's ``INFO BESTLINE``. EVAL is
        used only to recover a winrate when WINRATE is absent (mate/inf scores):
        ``+M<k>``/``VAL_INF+`` -> 1.0, ``-M<k>``/mated/``VAL_INF-`` -> 0.0.
        Returns ``{action: winrate}`` with winrate clamped to [0, 1].

        The read is bounded by a pv-scaled line cap (``_ANALYSIS_LINES_PER_PV``),
        NOT the quiet ``_MAX_CHATTER_LINES`` belt: a mate-driven deepening stream
        legitimately runs into the thousands (see the constant's note).
        """
        deadline = self._read_deadline_s()
        line_cap = max(_MAX_CHATTER_LINES, int(pv_count) * _ANALYSIS_LINES_PER_PV)
        # Per PV index, the latest (eval_winrate, root_action) seen.
        by_pv: dict[int, tuple[float | None, int | None]] = {}
        cur_pv: int | None = None
        cur_winrate: float | None = None
        cur_eval_wr: float | None = None
        cur_root: int | None = None

        def _commit() -> None:
            if cur_pv is None or cur_root is None:
                return
            wr = cur_winrate if cur_winrate is not None else cur_eval_wr
            if wr is None:
                return
            by_pv[cur_pv] = (wr, cur_root)

        for _ in range(line_cap):
            line = self._read_line(deadline)
            if not line:
                continue
            # The whole analysis ends with a bare coordinate bestmove line.
            if not line.startswith("INFO") and _parse_coord(line) is not None:
                _commit()
                break
            if not line.startswith("INFO"):
                continue  # MESSAGE / DEBUG / banner chatter
            body = line[len("INFO"):].strip()
            if body.startswith("PV "):
                # Start of a new PV block: commit the previous block first.
                _commit()
                try:
                    cur_pv = int(body[3:].strip())
                except ValueError:
                    cur_pv = None
                cur_winrate = cur_eval_wr = cur_root = None
            elif body == "PV DONE":
                _commit()
                cur_pv = None
                cur_winrate = cur_eval_wr = cur_root = None
            elif body.startswith("WINRATE"):
                try:
                    cur_winrate = float(body.split()[1])
                except (IndexError, ValueError):
                    cur_winrate = None
            elif body.startswith("EVAL"):
                cur_eval_wr = _eval_to_winrate(body.split()[1] if len(body.split()) > 1 else "")
            elif body.startswith("BESTLINE"):
                toks = body.split()
                if len(toks) >= 2:
                    c = _parse_coord(toks[1])
                    cur_root = _xy_to_action(*c) if c is not None else None
        else:
            raise ExternalEngineError(
                f"engine emitted {line_cap}+ lines (pv_count={pv_count}) without "
                f"ending the multiPV analysis (no bestmove terminator)"
            )

        out: dict[int, float] = {}
        for wr, root in by_pv.values():
            if root is None or wr is None:
                continue
            out[int(root)] = float(min(1.0, max(0.0, wr)))
        return out

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
