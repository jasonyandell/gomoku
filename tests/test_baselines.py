import numpy as np
import pytest

from gomoku.baselines import (
    heuristic_player,
    lookahead_player,
    random_player,
)
from gomoku.eval import MatchResult, play_match_pickers
from gomoku.game import GameState, str_to_action


def _apply_moves(state: GameState, moves: list[str]) -> GameState:
    for m in moves:
        state = state.apply(str_to_action(m))
    return state


def _setup_black_to_win_at_e5() -> GameState:
    """Black has a5..d5, white has 4 harmless moves. Black to move, e5 wins."""
    state = GameState.initial()
    pairs = [("a5", "a1"), ("b5", "b1"), ("c5", "c1"), ("d5", "d1")]
    for b, w in pairs:
        state = state.apply(str_to_action(b))
        state = state.apply(str_to_action(w))
    # Black to move; placing at e5 yields a5..e5 = 5-in-a-row.
    return state


def test_heuristic_takes_immediate_win():
    state = _setup_black_to_win_at_e5()
    rng = np.random.default_rng(0)
    action = heuristic_player(state, rng)
    assert action == str_to_action("e5")


def test_heuristic_blocks_immediate_threat():
    """Opp has 4 in a row a5..d5, no win for us — must block at e5."""
    # We need: state with side-to-move = white, white has no immediate win,
    # black has a5..d5 threatening e5. Engineer it by playing as black: a5,b5,c5,d5
    # interleaved with white moves that DON'T create a 4-in-a-row.
    state = GameState.initial()
    # Black: a5, b5, c5, d5. White: a8, b8, c8, d8 (4 spread out, but they're 4-in-a-row too!)
    # Use non-aligned white moves instead.
    moves = [
        ("a5", "i1"),
        ("b5", "i9"),
        ("c5", "a9"),
        ("d5", "i5"),
    ]
    for b, w in moves:
        state = state.apply(str_to_action(b))
        state = state.apply(str_to_action(w))
    # Now black just played d5; white to move. White has no 4-in-a-row.
    # Black threatens to play e5 for the win. White must block.
    rng = np.random.default_rng(0)
    action = heuristic_player(state, rng)
    assert action == str_to_action("e5"), \
        f"expected block at e5, got {action}"


def test_lookahead_finds_mate_in_one():
    """Depth-2 lookahead must find the immediate winning move (mate in 1)."""
    state = _setup_black_to_win_at_e5()
    player = lookahead_player(depth=2)
    rng = np.random.default_rng(0)
    action = player(state, rng)
    assert action == str_to_action("e5")


def test_lookahead_finds_forced_mate_in_two():
    """Set up a position where depth-4 lookahead must see a 2-move-ahead forced win.

    Construction: side to move has TWO immediate threats (a double-four). The
    opponent can only block one; we then complete the other for the win on our
    next move. This is a forced mate-in-3-plies (we play, they block, we win)
    so depth=4 finds it cleanly.
    """
    # Black builds two simultaneous 4-in-a-row threats:
    #   Horizontal a3..d3 — threatens e3 to make 5.
    #   Vertical   e6..e9 — threatens e5 to make 5.
    # (We pick squares that don't overlap and white plays harmlessly.)
    state = GameState.initial()
    # Setup: alternate moves so black ends up with both threats and it's black's turn.
    # Black needs 8 stones, white needs 8 stones. After 16 moves black plays the
    # 9th stone — that's the move that creates the double-four.
    # Easier: hand-craft via apply.
    black_pre = ["a3", "b3", "c3", "e6", "e7", "e8"]
    # We want black to make TWO 4-stone lines (horizontal a3..d3 and vertical e6..e9)
    # by playing one final move. So we set up 3 black stones in each line, then
    # the move "d3" creates the horizontal 4, and we need a separate 4th-line setup.
    # Simpler: skip "split" approach and use a position where black has a3,b3,c3
    # and e7,e8,e9. Black plays d3 -> horizontal 4 threatening e3. But that's
    # only one threat. We need two threats from one move. The classic "double four"
    # requires a stone that completes two 4-in-a-rows simultaneously.
    #
    # Even simpler test: position where black has a3..d3 (4 in a row, e3 wins)
    # AND e7..e9 + e5 (3 in a row at e5,e7,e8,e9 — wait those aren't contiguous).
    #
    # Use the simplest forced-mate-in-3: side to move has TWO existing
    # 4-in-a-row threats (open fours OR two fours requiring different blocks).
    # If both fours are already on the board, side-to-move wins immediately
    # (caught by mate-in-1). So we need a SINGLE move that creates two fours.
    #
    # Classic shape: "double 3" doesn't work (opp can block both ends with
    # single move blocks each, but we have 2 free turns). Use a sword:
    # black has stones at b3, c3, d3 (3 in a row, gap at a3 and e3) AND
    # d3, d4, d5 (3 in a row vertically, gap at d2 and d6). Black plays d3...
    # wait, d3 is already a stone.
    #
    # OK, let's just do the most direct: black has b3, c3, d3 horizontally and
    # d4, d5, d6 vertically — both 3-in-a-rows. Black to move plays d3? d3 is
    # already played. Black plays at a position that creates two fours: e.g.
    # add stone at e3 (forms b3..e3, 4-in-a-row threatening a3 or f3) and... no,
    # one stone can only be in lines through it.
    #
    # The simplest construction: side-to-move has 4 in a row open on both
    # ends, e.g. b3..e3 with both a3 and f3 empty. That's an "open four" —
    # opp blocks one end, we play the other for the win. Mate in 3.
    state = GameState.initial()
    # Black: b3, c3, d3, e3 (4 in a row, both ends a3 and f3 empty).
    # We need black to move AFTER this is set up. Play black moves on b3..e3
    # interleaved with non-threatening white moves, then black makes one more
    # move so it's white's turn... no, we need it to be BLACK's turn with the
    # open four already on the board so black plays one more stone to make 5.
    # But "open four with both ends open" IS already mate-in-1 (just play a3
    # or f3 to make 5)! So it's mate in 1 not mate in 3.
    #
    # For mate-in-3 we want: black plays a move that CREATES the open four;
    # white blocks one end; black plays the other end for 5.
    # Setup: black has b3, c3, d3 (3-in-row), e3 empty, a3 empty, f3 empty.
    # Black to move plays e3 -> b3..e3 (4-in-row, open on a3 and f3).
    # That's mate in 3. Let's build it.
    state = GameState.initial()
    # Place: black b3,c3,d3; white: 3 harmless stones; it's black to move.
    state = state.apply(str_to_action("b3"))  # black
    state = state.apply(str_to_action("i1"))  # white
    state = state.apply(str_to_action("c3"))  # black
    state = state.apply(str_to_action("i9"))  # white
    state = state.apply(str_to_action("d3"))  # black
    state = state.apply(str_to_action("a9"))  # white
    # Now black to move with b3..d3 in place; squares a3, e3, f3 all empty.
    # Black should play e3 (or symmetrically c3, but c3 occupied). Best move
    # creates open four — depth 4 must find it.
    player = lookahead_player(depth=4)
    rng = np.random.default_rng(0)
    action = player(state, rng)
    # e3 creates b3..e3 open four (a3 and f3 both empty). Also a3 creates
    # a3..d3 with e3 empty but the other end of the 4 — only the f-side end
    # is open (a3 itself is now the leftmost stone, so left end is closed by
    # the board edge — wait, a3 is the edge; so a3..d3 is a "half-open four"
    # threatening only e3, which white blocks). So e3 is the unique winning
    # move (the only one creating an OPEN four).
    assert action == str_to_action("e3"), \
        f"expected e3 (creates open four for mate-in-3), got {action}"


def test_lookahead_only_plays_legal():
    """Run lookahead on a handful of random positions and check legality."""
    player = lookahead_player(depth=2)
    rng = np.random.default_rng(42)
    for _ in range(5):
        state = GameState.initial()
        # Random opening of 6 plies.
        for _ in range(6):
            legal = state.legal_actions()
            a = int(rng.choice(legal))
            state = state.apply(a)
            done, _ = state.is_terminal()
            if done:
                break
        done, _ = state.is_terminal()
        if done:
            continue
        action = player(state, rng)
        assert state.legal_mask()[action], \
            f"lookahead played illegal action {action} on a position with " \
            f"{state.legal_mask().sum()} legal moves"


def test_play_match_pickers_counts():
    """play_match_pickers returns a MatchResult with consistent counts."""
    res = play_match_pickers(random_player, random_player, n_games=6, seed=0)
    assert isinstance(res, MatchResult)
    assert res.n_games == 6
    assert res.wins + res.losses + res.draws == 6


def test_heuristic_beats_random():
    """Sanity: the heuristic should crush random over a small sample."""
    res = play_match_pickers(heuristic_player, random_player, n_games=10, seed=0)
    assert res.wins + 0.5 * res.draws >= 8, \
        f"heuristic should dominate random, got {res}"
