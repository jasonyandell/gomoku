"""Swap2 opening protocol — the pure game-logic state machine.

Why this exists
===============
15x15 freestyle gomoku is a proven *first-player win*: from an empty or random
opening, white (the second player) is a (near-)lost role, so AlphaZero self-play
collapses — every game is a black win, the value head only ever sees "white =
lost", and the policy never gets gradient on *winnable* white positions because
its own data contains none. No defense teacher can fix a distribution that holds
no balanced games. (Full evidence: ``wiki/topics/white-side-defense-plan.md``.)

Swap2 is Gomocup's balancing protocol. It removes the doomed *forced* role: a
player is never compelled onto the lost side, so self-play generates ~50/50 data
and white positions finally become winnable in the training set. This module is
the load-bearing first piece — the negotiation as provably-correct game logic,
with no network and no search. MCTS / self-play / eval integration build on top.

The negotiation (cross-confirmed: Wikipedia "Gomoku", RenjuNet rule 11, the
Gomocup ``SWAP2BOARD`` protocol)
==============================================================================
1. The OPENER places 3 stones = **2 black + 1 white**, anywhere.
2. The RESPONDER chooses one of:
   - **STAY**  : take white and *place a 2nd white stone* (this placement IS
                 white's move) -> board 2B+2W, the OPENER moves next as black.
   - **SWAP**  : take black (no new stone) -> board 2B+1W, the OPENER moves next
                 as white.
   - **PLACE2**: add 1 black + 1 white -> board 3B+2W, then the OPENER picks a
                 color; white moves next either way (the pick only decides *which
                 player* is white — the board is identical).

Representation choices
======================
- **Absolute colors.** Unlike normal :class:`gomoku.game.GameState` (which is
  perspective-canonical — plane 0 is always side-to-move), the opening is *not*
  alternating-color play: the opener places three stones of mixed color in a row.
  So the opening is tracked in absolute (black, white) planes, and is converted
  to a canonical ``GameState`` only at the hand-off to normal play
  (:meth:`OpeningState.to_normal`).

- **Color fixed by placement order — zero expressiveness lost.** Each opening
  stone's *color* is determined by its step index (black, white, black, ...),
  and only its *square* is a decision. Because the squares are chosen freely,
  every 2B+1W configuration is still reachable (just place the white stone
  whenever you want it). This keeps every placement a plain spatial move over
  the board policy head — no action-space growth for placements.

- **Only the two negotiation moments are abstract.** The RESPONDER's
  {STAY, SWAP, PLACE2} and the OPENER's {pick-black, pick-white} are the only
  non-spatial decisions. They are exposed as a small choice space (width
  :data:`N_CHOICES`) for a dedicated choice head; everything else is a square.

- **Value attribution via explicit actor tags.** The opening breaks the
  "flip perspective every ply" invariant (the opener acts three times in a row),
  so value sign during search is driven by each node's :attr:`OpeningState.to_act`
  (OPENER vs RESPONDER), not by ply parity. See :func:`backup_sign`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from gomoku.board_config import BOARD_SIZE
from gomoku.game import GameState

# ---------------------------------------------------------------------------
# Choice space (the only non-spatial actions in the whole protocol).
# A choice NODE is always one of two kinds; a width-3 head serves both because
# the position (3 stones vs 5 stones) disambiguates the slot meaning:
#   responder node (3 stones): slot 0=STAY, 1=SWAP, 2=PLACE2
#   opener pick node (5 stones): slot 0=PICK_BLACK, 1=PICK_WHITE, (2 illegal)
# ---------------------------------------------------------------------------
N_CHOICES = 3

# Responder-node slot meanings.
RESP_STAY = 0
RESP_SWAP = 1
RESP_PLACE2 = 2

# Opener pick-node slot meanings.
PICK_BLACK = 0
PICK_WHITE = 1


class Color(IntEnum):
    BLACK = 0
    WHITE = 1


class Actor(IntEnum):
    """Which player acts at a node — for value attribution through the opening."""

    OPENER = 0
    RESPONDER = 1


class Phase(IntEnum):
    PLACE = 0   # a stone placement; legal actions are empty board squares
    CHOICE = 1  # an abstract negotiation choice; legal actions are choice slots
    DONE = 2    # opening complete; hand to normal play via to_normal()


class _Node(IntEnum):
    """The position in the fixed negotiation sequence."""

    P1_S1 = 0       # opener places stone 1 (black)
    P1_S2 = 1       # opener places stone 2 (white)
    P1_S3 = 2       # opener places stone 3 (black)
    P2_CHOICE = 3   # responder chooses STAY / SWAP / PLACE2
    P2_STAY_W = 4   # responder (chose STAY) places its white stone
    P2_P2_B = 5     # responder (chose PLACE2) places a black stone
    P2_P2_W = 6     # responder (chose PLACE2) places a white stone
    P1_PICK = 7     # opener picks black or white
    DONE = 8


# Per-node static descriptors: (phase, place_color_or_None, to_act).
_NODE_INFO: dict[_Node, tuple[Phase, Color | None, Actor]] = {
    _Node.P1_S1: (Phase.PLACE, Color.BLACK, Actor.OPENER),
    _Node.P1_S2: (Phase.PLACE, Color.WHITE, Actor.OPENER),
    _Node.P1_S3: (Phase.PLACE, Color.BLACK, Actor.OPENER),
    _Node.P2_CHOICE: (Phase.CHOICE, None, Actor.RESPONDER),
    _Node.P2_STAY_W: (Phase.PLACE, Color.WHITE, Actor.RESPONDER),
    _Node.P2_P2_B: (Phase.PLACE, Color.BLACK, Actor.RESPONDER),
    _Node.P2_P2_W: (Phase.PLACE, Color.WHITE, Actor.RESPONDER),
    _Node.P1_PICK: (Phase.CHOICE, None, Actor.OPENER),
    _Node.DONE: (Phase.DONE, None, Actor.OPENER),  # to_act unused once DONE
}


class Swap2Error(ValueError):
    """Raised on an illegal opening action (bad square, illegal choice slot)."""


@dataclass
class OpeningState:
    """One node of the swap2 negotiation, in absolute (black, white) colors.

    Immutable by convention: :meth:`apply` always returns a fresh state with new
    arrays; the planes are never mutated in place. ``mover_color`` /
    ``opener_color`` are populated only once the node is :data:`Phase.DONE`.
    """

    black: np.ndarray            # (N, N) bool — black stones, absolute
    white: np.ndarray            # (N, N) bool — white stones, absolute
    node: _Node
    # Populated only at DONE:
    mover_color: Color | None = None   # color to move when normal play begins
    opener_color: Color | None = None  # color the OPENER plays the whole game

    # -- descriptors ----------------------------------------------------

    @property
    def phase(self) -> Phase:
        return _NODE_INFO[self.node][0]

    @property
    def place_color(self) -> Color | None:
        return _NODE_INFO[self.node][1]

    @property
    def to_act(self) -> Actor:
        return _NODE_INFO[self.node][2]

    def is_done(self) -> bool:
        return self.node is _Node.DONE

    @property
    def n_black(self) -> int:
        return int(self.black.sum())

    @property
    def n_white(self) -> int:
        return int(self.white.sum())

    # -- legality -------------------------------------------------------

    def legal_choice_mask(self) -> np.ndarray:
        """(N_CHOICES,) bool over choice slots; all-False unless a CHOICE node."""
        mask = np.zeros(N_CHOICES, dtype=bool)
        if self.node is _Node.P2_CHOICE:
            mask[RESP_STAY] = mask[RESP_SWAP] = mask[RESP_PLACE2] = True
        elif self.node is _Node.P1_PICK:
            mask[PICK_BLACK] = mask[PICK_WHITE] = True
        return mask

    def legal_square_mask(self) -> np.ndarray:
        """(N*N,) bool over board squares; all-False unless a PLACE node."""
        if self.phase is not Phase.PLACE:
            return np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=bool)
        empty = ~(self.black | self.white)
        return empty.reshape(-1)

    def legal_actions(self) -> np.ndarray:
        """Legal action indices: board squares at a PLACE node, choice slots at
        a CHOICE node, empty when DONE. Interpret against :attr:`phase`."""
        if self.phase is Phase.PLACE:
            return np.flatnonzero(self.legal_square_mask())
        if self.phase is Phase.CHOICE:
            return np.flatnonzero(self.legal_choice_mask())
        return np.empty(0, dtype=np.int64)

    # -- transition -----------------------------------------------------

    def apply(self, action: int) -> "OpeningState":
        """Advance the negotiation by one action (a square or a choice slot)."""
        if self.phase is Phase.PLACE:
            return self._apply_place(int(action))
        if self.phase is Phase.CHOICE:
            return self._apply_choice(int(action))
        raise Swap2Error("cannot apply an action to a DONE opening")

    def _apply_place(self, square: int) -> "OpeningState":
        if not (0 <= square < BOARD_SIZE * BOARD_SIZE):
            raise Swap2Error(f"square {square} out of range")
        r, c = divmod(square, BOARD_SIZE)
        if self.black[r, c] or self.white[r, c]:
            raise Swap2Error(f"square {square} is occupied")
        black = self.black.copy()
        white = self.white.copy()
        if self.place_color is Color.BLACK:
            black[r, c] = True
        else:
            white[r, c] = True
        nxt = {
            _Node.P1_S1: _Node.P1_S2,
            _Node.P1_S2: _Node.P1_S3,
            _Node.P1_S3: _Node.P2_CHOICE,
            _Node.P2_STAY_W: _Node.DONE,   # responder took white -> opener (black) moves
            _Node.P2_P2_B: _Node.P2_P2_W,
            _Node.P2_P2_W: _Node.P1_PICK,
        }[self.node]
        if nxt is _Node.DONE:
            return _finish(black, white, opener_color=Color.BLACK)
        return OpeningState(black=black, white=white, node=nxt)

    def _apply_choice(self, slot: int) -> "OpeningState":
        if not (0 <= slot < N_CHOICES) or not self.legal_choice_mask()[slot]:
            raise Swap2Error(f"illegal choice slot {slot} at {self.node.name}")
        if self.node is _Node.P2_CHOICE:
            if slot == RESP_STAY:
                return OpeningState(
                    black=self.black.copy(), white=self.white.copy(),
                    node=_Node.P2_STAY_W,
                )
            if slot == RESP_SWAP:
                # No new stone; responder takes black, opener plays white.
                return _finish(self.black.copy(), self.white.copy(),
                               opener_color=Color.WHITE)
            # PLACE2
            return OpeningState(
                black=self.black.copy(), white=self.white.copy(),
                node=_Node.P2_P2_B,
            )
        # P1_PICK
        opener_color = Color.BLACK if slot == PICK_BLACK else Color.WHITE
        return _finish(self.black.copy(), self.white.copy(), opener_color=opener_color)

    # -- hand-off to normal play ---------------------------------------

    def to_normal(self) -> GameState:
        """Convert a DONE opening to a canonical, perspective-correct GameState.

        ``board[0]`` is the side-to-move's (``mover_color``) stones; history is
        empty (the opening carries no per-ply recency). The opener<->color mapping
        for scoring lives in :attr:`opener_color` / :meth:`mover_actor`, not in the
        color-blind GameState."""
        if not self.is_done():
            raise Swap2Error("opening is not complete")
        assert self.mover_color is not None
        mover = self.black if self.mover_color is Color.BLACK else self.white
        other = self.white if self.mover_color is Color.BLACK else self.black
        board = np.stack([mover, other]).astype(bool)
        return GameState(board=board, move_count=self.n_black + self.n_white, history=())

    def mover_actor(self) -> Actor:
        """Which player is to move when normal play begins."""
        if not self.is_done():
            raise Swap2Error("opening is not complete")
        return Actor.OPENER if self.mover_color == self.opener_color else Actor.RESPONDER

    # -- network framing (for priors only; value comes from backup) ----

    def net_state(self) -> GameState:
        """A GameState whose planes frame this node for the network's prior.

        PLACE node: side-to-move == the color being placed ("I add a [color] stone
        here"). CHOICE node: framed white-to-move (white is the next mover in every
        continuation of both choice nodes), so the choice head reads a consistent
        position. The *value* of a choice/place node is never read off this frame —
        it is backed up from children with :func:`backup_sign`."""
        if self.phase is Phase.PLACE:
            mine = self.black if self.place_color is Color.BLACK else self.white
            opp = self.white if self.place_color is Color.BLACK else self.black
        elif self.phase is Phase.CHOICE:
            mine, opp = self.white, self.black  # white-to-move framing
        else:
            return self.to_normal()
        board = np.stack([mine, opp]).astype(bool)
        return GameState(board=board, move_count=self.n_black + self.n_white, history=())


def _finish(black: np.ndarray, white: np.ndarray, *, opener_color: Color) -> OpeningState:
    """Build a DONE state, deriving who-moves-next from the stone counts.

    Invariant: black has the same number of stones as white, or exactly one more.
    Equal counts -> black to move; black-ahead-by-one -> white to move."""
    nb, nw = int(black.sum()), int(white.sum())
    if nb == nw:
        mover_color = Color.BLACK
    elif nb == nw + 1:
        mover_color = Color.WHITE
    else:
        raise Swap2Error(f"illegal opening stone counts: {nb} black, {nw} white")
    return OpeningState(
        black=black, white=white, node=_Node.DONE,
        mover_color=mover_color, opener_color=opener_color,
    )


def initial_opening() -> OpeningState:
    """The root of the negotiation: an empty board, opener to place stone 1."""
    z = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    return OpeningState(black=z.copy(), white=z.copy(), node=_Node.P1_S1)


def backup_sign(parent: OpeningState, child_to_act: Actor) -> int:
    """Sign to apply to a child's value when backing it up to ``parent``.

    Value at every node is in its :attr:`~OpeningState.to_act` player's
    perspective. Crossing an edge negates iff the acting player changes — this
    is the explicit replacement for "flip perspective every ply", which does not
    hold across the opener's three-in-a-row placements. ``child_to_act`` is the
    child node's acting player (for a normal-play child, the
    :meth:`OpeningState.mover_actor` of the hand-off)."""
    return 1 if parent.to_act == child_to_act else -1
