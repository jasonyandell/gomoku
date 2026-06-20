"""Tests for the swap2 opening state machine (gomoku/swap2.py).

These pin the negotiation *rules* as executable game logic — stone counts,
colors, who-moves-next, and the value-attribution sign — independent of any
network or search. Board size is whatever the process default is (9); the
negotiation logic is board-size agnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

from gomoku.board_config import BOARD_SIZE
from gomoku.swap2 import (
    Actor,
    Color,
    OpeningState,
    Phase,
    Swap2Error,
    PICK_BLACK,
    PICK_WHITE,
    RESP_PLACE2,
    RESP_STAY,
    RESP_SWAP,
    backup_sign,
    initial_opening,
)

N = BOARD_SIZE


def sq(r: int, c: int) -> int:
    return r * N + c


def _cells(plane: np.ndarray) -> set[int]:
    return {int(i) for i in np.flatnonzero(plane.reshape(-1))}


def _place3(op: OpeningState, a: int, b: int, c: int) -> OpeningState:
    """Walk the opener's three placements (colors black, white, black)."""
    return op.apply(a).apply(b).apply(c)


def opener_done(a: int, b: int, c: int) -> OpeningState:
    return _place3(initial_opening(), a, b, c)


# -- the opener's three stones -------------------------------------------------

def test_initial_is_opener_placing_black():
    op = initial_opening()
    assert op.phase is Phase.PLACE
    assert op.place_color is Color.BLACK
    assert op.to_act is Actor.OPENER
    assert not op.is_done()
    assert op.n_black == 0 and op.n_white == 0
    # All squares legal on an empty board.
    assert len(op.legal_actions()) == N * N


def test_opener_places_two_black_one_white():
    op = opener_done(sq(3, 3), sq(3, 4), sq(4, 4))
    assert op.n_black == 2 and op.n_white == 1
    # The white stone is the *second* placement (color-by-order).
    assert _cells(op.white) == {sq(3, 4)}
    assert _cells(op.black) == {sq(3, 3), sq(4, 4)}
    # Now it is the responder's choice.
    assert op.phase is Phase.CHOICE
    assert op.to_act is Actor.RESPONDER
    assert list(op.legal_actions()) == [RESP_STAY, RESP_SWAP, RESP_PLACE2]


def test_color_by_order_reaches_any_white_square():
    # Any 2B+1W config is reachable: place the desired white square 2nd.
    for white_sq in (sq(0, 0), sq(7, 7), sq(2, 6)):
        op = opener_done(sq(1, 1), white_sq, sq(5, 5))
        assert _cells(op.white) == {white_sq}


# -- responder: SWAP -----------------------------------------------------------

def test_swap_branch():
    op = opener_done(sq(3, 3), sq(3, 4), sq(4, 4)).apply(RESP_SWAP)
    assert op.is_done()
    assert op.n_black == 2 and op.n_white == 1   # no new stone
    assert op.mover_color is Color.WHITE          # black-ahead-by-one -> white moves
    assert op.opener_color is Color.WHITE         # opener took white on a swap
    assert op.mover_actor() is Actor.OPENER       # opener (white) moves next
    gs = op.to_normal()
    assert gs.move_count == 3
    assert _cells(gs.board[0]) == {sq(3, 4)}       # side-to-move == white
    assert _cells(gs.board[1]) == {sq(3, 3), sq(4, 4)}
    assert gs.is_terminal()[0] is False


# -- responder: STAY (take white, place a white stone) -------------------------

def test_stay_branch():
    mid = opener_done(sq(3, 3), sq(3, 4), sq(4, 4)).apply(RESP_STAY)
    # STAY is an abstract choice followed by a white placement.
    assert not mid.is_done()
    assert mid.phase is Phase.PLACE
    assert mid.place_color is Color.WHITE
    assert mid.to_act is Actor.RESPONDER
    op = mid.apply(sq(5, 5))
    assert op.is_done()
    assert op.n_black == 2 and op.n_white == 2
    assert op.mover_color is Color.BLACK          # equal counts -> black moves
    assert op.opener_color is Color.BLACK          # responder took white
    assert op.mover_actor() is Actor.OPENER        # opener (black) moves next
    gs = op.to_normal()
    assert gs.move_count == 4
    assert _cells(gs.board[0]) == {sq(3, 3), sq(4, 4)}   # side-to-move == black
    assert _cells(gs.board[1]) == {sq(3, 4), sq(5, 5)}


# -- responder: PLACE2 then opener picks ---------------------------------------

def _place2_to_pick(op: OpeningState) -> OpeningState:
    """Drive PLACE2: responder places a black then a white, reaching the pick."""
    s = op.apply(RESP_PLACE2)
    assert s.place_color is Color.BLACK and s.to_act is Actor.RESPONDER
    s = s.apply(sq(6, 6))                  # responder's black
    assert s.place_color is Color.WHITE
    s = s.apply(sq(6, 7))                  # responder's white
    assert s.phase is Phase.CHOICE and s.to_act is Actor.OPENER
    assert list(s.legal_actions()) == [PICK_BLACK, PICK_WHITE]
    return s


def test_place2_pick_black():
    pick = _place2_to_pick(opener_done(sq(3, 3), sq(3, 4), sq(4, 4)))
    op = pick.apply(PICK_BLACK)
    assert op.is_done()
    assert op.n_black == 3 and op.n_white == 2
    assert op.mover_color is Color.WHITE
    assert op.opener_color is Color.BLACK
    assert op.mover_actor() is Actor.RESPONDER     # responder is white, moves next
    gs = op.to_normal()
    assert gs.move_count == 5
    assert _cells(gs.board[0]) == {sq(3, 4), sq(6, 7)}    # white to move


def test_place2_pick_white():
    pick = _place2_to_pick(opener_done(sq(3, 3), sq(3, 4), sq(4, 4)))
    op = pick.apply(PICK_WHITE)
    assert op.is_done()
    assert op.mover_color is Color.WHITE
    assert op.opener_color is Color.WHITE
    assert op.mover_actor() is Actor.OPENER


def test_pick_changes_attribution_not_the_board():
    """The PLACE2 color-pick only decides which *player* is white; the dealt
    position (3B+2W, white to move) is byte-identical either way."""
    pick = _place2_to_pick(opener_done(sq(3, 3), sq(3, 4), sq(4, 4)))
    black_gs = pick.apply(PICK_BLACK).to_normal()
    white_gs = pick.apply(PICK_WHITE).to_normal()
    assert np.array_equal(black_gs.board, white_gs.board)
    assert black_gs.move_count == white_gs.move_count


# -- legality / errors ---------------------------------------------------------

def test_place_excludes_occupied_squares():
    op = opener_done(sq(3, 3), sq(3, 4), sq(4, 4)).apply(RESP_STAY)
    legal = set(int(a) for a in op.legal_actions())
    assert sq(3, 3) not in legal and sq(3, 4) not in legal and sq(4, 4) not in legal
    assert len(legal) == N * N - 3


def test_apply_occupied_raises():
    op = initial_opening().apply(sq(3, 3))
    with pytest.raises(Swap2Error):
        op.apply(sq(3, 3))


def test_apply_out_of_range_raises():
    with pytest.raises(Swap2Error):
        initial_opening().apply(N * N)


def test_illegal_choice_slot_raises():
    op = opener_done(sq(3, 3), sq(3, 4), sq(4, 4))
    with pytest.raises(Swap2Error):
        op.apply(99)                       # not a valid choice slot
    pick = _place2_to_pick(op)
    with pytest.raises(Swap2Error):
        pick.apply(RESP_PLACE2)            # slot 2 illegal at a 2-way pick node


def test_apply_to_done_raises():
    op = opener_done(sq(3, 3), sq(3, 4), sq(4, 4)).apply(RESP_SWAP)
    with pytest.raises(Swap2Error):
        op.apply(sq(0, 0))


def test_to_normal_requires_done():
    with pytest.raises(Swap2Error):
        initial_opening().to_normal()


# -- invariants ----------------------------------------------------------------

def test_apply_does_not_mutate_parent():
    op = initial_opening()
    before_black = op.black.copy()
    _ = op.apply(sq(3, 3))
    assert np.array_equal(op.black, before_black)
    assert op.n_black == 0


@pytest.mark.parametrize("branch", ["swap", "stay", "place2_black", "place2_white"])
def test_black_ahead_by_at_most_one_through_every_node(branch):
    states = [initial_opening()]
    op = opener_done(sq(3, 3), sq(3, 4), sq(4, 4))
    # record the three opener placements too
    s = initial_opening()
    seq = [s]
    for a in (sq(3, 3), sq(3, 4), sq(4, 4)):
        s = s.apply(a)
        seq.append(s)
    if branch == "swap":
        seq.append(op.apply(RESP_SWAP))
    elif branch == "stay":
        mid = op.apply(RESP_STAY)
        seq.extend([mid, mid.apply(sq(5, 5))])
    else:
        pick = _place2_to_pick(op)
        seq.append(pick)
        seq.append(pick.apply(PICK_BLACK if branch == "place2_black" else PICK_WHITE))
    for st in seq:
        diff = st.n_black - st.n_white
        assert diff in (0, 1), f"{st.node.name}: {st.n_black}B {st.n_white}W"


# -- network framing -----------------------------------------------------------

def test_net_state_place_frames_placing_color_as_side_to_move():
    # Opener placing white (stone 2): board[0] should be the white stones so far.
    op = initial_opening().apply(sq(3, 3))   # placed black; now placing white
    assert op.place_color is Color.WHITE
    ns = op.net_state()
    # so-far white is empty, black has one stone -> board[1] holds the black stone
    assert _cells(ns.board[0]) == set()
    assert _cells(ns.board[1]) == {sq(3, 3)}


def test_net_state_choice_frames_white_to_move():
    op = opener_done(sq(3, 3), sq(3, 4), sq(4, 4))   # responder choice node
    ns = op.net_state()
    assert _cells(ns.board[0]) == {sq(3, 4)}          # white framed as side-to-move
    assert _cells(ns.board[1]) == {sq(3, 3), sq(4, 4)}


# -- value-attribution sign ----------------------------------------------------

def test_backup_sign_no_flip_within_opener_placements():
    op = initial_opening()                    # OPENER
    child = op.apply(sq(3, 3))                # OPENER (stone 2)
    assert backup_sign(op, child.to_act) == 1


def test_backup_sign_flips_into_responder_choice():
    op = opener_done(sq(3, 3), sq(3, 4), sq(4, 4))   # to_act RESPONDER
    parent = initial_opening().apply(sq(3, 3)).apply(sq(3, 4))  # P1_S3, OPENER
    assert parent.to_act is Actor.OPENER
    assert backup_sign(parent, op.to_act) == -1


def test_backup_sign_uses_mover_actor_at_handoff():
    # Responder chooses SWAP -> opener moves next; value flips to the responder.
    choice = opener_done(sq(3, 3), sq(3, 4), sq(4, 4))   # RESPONDER
    done = choice.apply(RESP_SWAP)
    assert backup_sign(choice, done.mover_actor()) == -1   # RESPONDER vs OPENER
