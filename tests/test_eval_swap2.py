"""Tests for the swap2 eval-vs-engine harness (``gomoku.eval_swap2``).

The protocol <-> OpeningState mapping is the risk, so every engine branch is
guarded by driving a full opening against the shipped fake Piskvork engine
(``tests/fake_piskvork_engine.py``, swap2_* modes) and asserting the resulting
``OpeningState.to_normal()`` board equals an INDEPENDENTLY hand-constructed
expected board — correct squares AND colors. The color check is what catches the
move-parity bug in the engine's PLACE2 (TWO_COORDS) reply.

Net side: a tiny deterministic stub oracle (no real net, no GPU) that spikes a
chosen square and returns a constant value, so our agent's placements/choices are
fully determined and the assertions are exact.

No real engine binary is required: the fake engine is launched as a subprocess
exactly like the external-engine swap2 tests.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

from gomoku.board_config import BOARD_SIZE, N_ACTIONS
from gomoku.external_engine import ExternalEngineConfig, ExternalEnginePlayer
from gomoku.game import GameState
from gomoku.swap2 import (
    Actor,
    Color,
    OpeningState,
    PICK_BLACK,
    PICK_WHITE,
    Phase,
    RESP_PLACE2,
    RESP_STAY,
    RESP_SWAP,
    initial_opening,
)
from gomoku.eval_swap2 import (
    GameOutcome,
    Swap2EvalResult,
    _engine_open,
    _engine_pick,
    _engine_respond,
    _game_rng,
    _game_role,
    _split_game_indices,
    _sq,
    aggregate_outcomes,
    eval_swap2_h2h,
    eval_swap2_vs_engine,
    play_swap2_game,
    play_swap2_game_h2h,
)

_STUB = os.path.join(os.path.dirname(__file__), "fake_piskvork_engine.py")


def _engine(mode: str) -> ExternalEnginePlayer:
    cmd = f"{sys.executable} {_STUB} {mode}"
    return ExternalEnginePlayer(
        ExternalEngineConfig(cmd=cmd, timeout_ms=500, label=f"stub:{mode}")
    )


# ---------------------------------------------------------------------------
# Stub oracle. Spikes a target square (passed in via a mutable list so the test
# can steer each placement deterministically), constant value otherwise.
# ---------------------------------------------------------------------------


def _spike_oracle(square_plan: list[int], value: float = 0.0):
    """Oracle that places stones on the squares in ``square_plan`` in order.

    Each PLACE-node query pops the next planned square off ``square_plan`` and
    spikes the policy there (mass 1.0). Choice-node queries use the constant
    ``value`` so choice selection is well-defined. This makes the WHOLE opening
    deterministic so an expected board can be hand-constructed.
    """
    plan = list(square_plan)
    cursor = {"i": 0}

    def oracle(gs: GameState) -> tuple[np.ndarray, float]:
        p = np.zeros(N_ACTIONS, dtype=np.float64)
        # Choose the next planned square if it is still empty; else fall back to
        # the lowest empty square (keeps the policy legal for choice rollouts).
        empty = ~(gs.board[0] | gs.board[1])
        flat = empty.reshape(-1)
        target = None
        if cursor["i"] < len(plan) and flat[plan[cursor["i"]]]:
            target = plan[cursor["i"]]
        if target is None:
            target = int(np.flatnonzero(flat)[0])
        p[target] = 1.0
        return p, float(value)

    def consume() -> None:
        cursor["i"] += 1

    oracle.consume = consume  # type: ignore[attr-defined]
    return oracle


def _board_from_xy(black_xy: list[tuple[int, int]], white_xy: list[tuple[int, int]]):
    """Hand-build (black, white) bool planes from (x=col, y=row) coordinate lists."""
    black = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    white = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    for x, y in black_xy:
        black[y, x] = True
    for x, y in white_xy:
        white[y, x] = True
    return black, white


def _normal_from_state(state: OpeningState):
    """``to_normal()`` board re-expanded into absolute (black, white) planes.

    ``to_normal()`` is perspective-canonical (board[0] = mover's stones); we
    undo that here so the assertion compares ABSOLUTE colors against the hand-
    built board, which is what catches a color bug.
    """
    gs = state.to_normal()
    assert state.mover_color is not None
    if state.mover_color == Color.BLACK:
        black, white = gs.board[0], gs.board[1]
    else:
        white, black = gs.board[0], gs.board[1]
    return black, white


# ---------------------------------------------------------------------------
# Branch: engine is OPENER (swap2_open -> 3 coords, colors b,w,b).
# ---------------------------------------------------------------------------


def test_engine_opener_open_board():
    """Engine opens (lowest-empty fake): 3 coords (0,0),(1,0),(2,0) in move order
    -> colors black, white, black. We (responder) then act; with a SWAP choice
    being unavailable here we just check the post-open OpeningState board."""
    eng = _engine("swap2_open")
    try:
        state = initial_opening()
        # Engine is OPENER at the start.
        assert state.node_is_opener_start()
        state = _engine_open(state, eng)
    finally:
        eng.close()

    # Fake fills lowest empties: (0,0)=move1 BLACK, (1,0)=move2 WHITE, (2,0)=move3 BLACK.
    exp_black, exp_white = _board_from_xy([(0, 0), (2, 0)], [(1, 0)])
    assert np.array_equal(state.black, exp_black), "opener black squares wrong"
    assert np.array_equal(state.white, exp_white), "opener white squares wrong"
    assert state.node.name == "P2_CHOICE"
    assert state.to_act == Actor.RESPONDER


# ---------------------------------------------------------------------------
# Branch: engine is RESPONDER, replies SWAP.
# ---------------------------------------------------------------------------


def test_engine_responder_swap_board():
    """Our opener places (0,0)B,(1,0)W,(2,0)B; engine responds SWAP -> takes
    black, opener (us) is white. Board stays 2B+1W, white(opener)-to-move."""
    # Our placements: black (0,0), white (1,0), black (2,0).
    oracle = _spike_oracle([_sq(0, 0), _sq(1, 0), _sq(2, 0)])
    eng = _engine("swap2_swap")
    try:
        state = initial_opening()
        # Three OUR placements (OPENER role).
        for _ in range(3):
            from gomoku.swap2_search import agent_act

            action, _rec = agent_act(oracle, state, np.random.default_rng(0), greedy=True)
            oracle.consume()
            state = state.apply(action)
        assert state.node.name == "P2_CHOICE"
        state = _engine_respond(state, eng)
    finally:
        eng.close()

    assert state.is_done()
    exp_black, exp_white = _board_from_xy([(0, 0), (2, 0)], [(1, 0)])
    black, white = _normal_from_state(state)
    assert np.array_equal(black, exp_black), "SWAP black squares wrong"
    assert np.array_equal(white, exp_white), "SWAP white squares wrong"
    assert state.opener_color == Color.WHITE  # responder took black -> opener white
    assert state.mover_color == Color.WHITE    # 2B+1W -> white to move
    assert state.mover_actor() == Actor.OPENER  # opener IS white here


# ---------------------------------------------------------------------------
# Branch: engine is RESPONDER, replies ONE_COORD (STAY white, plays move 4).
# ---------------------------------------------------------------------------


def test_engine_responder_one_coord_board():
    """Our opener: (0,0)B,(1,0)W,(2,0)B. Engine STAYS white and plays move 4
    (white) at the lowest empty (3,0). Board = 2B + 2W, black(opener)-to-move."""
    oracle = _spike_oracle([_sq(0, 0), _sq(1, 0), _sq(2, 0)])
    eng = _engine("swap2_one")
    try:
        state = initial_opening()
        for _ in range(3):
            from gomoku.swap2_search import agent_act

            action, _rec = agent_act(oracle, state, np.random.default_rng(0), greedy=True)
            oracle.consume()
            state = state.apply(action)
        state = _engine_respond(state, eng)
    finally:
        eng.close()

    assert state.is_done()
    # White move 4 at lowest empty: stones present are (0,0),(1,0),(2,0) -> lowest
    # empty is (3,0).
    exp_black, exp_white = _board_from_xy([(0, 0), (2, 0)], [(1, 0), (3, 0)])
    black, white = _normal_from_state(state)
    assert np.array_equal(black, exp_black), "ONE_COORD black squares wrong"
    assert np.array_equal(white, exp_white), "ONE_COORD white squares wrong (move-4 color)"
    assert state.opener_color == Color.BLACK   # responder kept white -> opener black
    assert state.mover_color == Color.BLACK     # 2B+2W -> black to move
    assert state.mover_actor() == Actor.OPENER  # opener IS black here


# ---------------------------------------------------------------------------
# Branch: engine is RESPONDER, replies TWO_COORDS (PLACE2). THE color-parity test.
# ---------------------------------------------------------------------------


def test_engine_responder_two_coords_board_move_parity():
    """Our opener: (0,0)B,(1,0)W,(2,0)B. Engine PLACE2 returns two coords in reply
    order ``coords[0]`` then ``coords[1]``; the fake returns the two lowest empties
    (3,0) then (4,0). By GLOBAL ply parity move 4 = WHITE = coords[0] = (3,0) and
    move 5 = BLACK = coords[1] = (4,0). The driver must color (4,0) black and
    (3,0) white — NOT in reply order."""
    oracle = _spike_oracle([_sq(0, 0), _sq(1, 0), _sq(2, 0)])
    eng = _engine("swap2_two")
    try:
        state = initial_opening()
        for _ in range(3):
            from gomoku.swap2_search import agent_act

            action, _rec = agent_act(oracle, state, np.random.default_rng(0), greedy=True)
            oracle.consume()
            state = state.apply(action)
        state = _engine_respond(state, eng)
    finally:
        eng.close()

    # After PLACE2 we are at the opener PICK node (OUR node), NOT done.
    assert not state.is_done()
    assert state.node.name == "P1_PICK"
    # Hand-built expected ABSOLUTE board after the 5 placed stones:
    #   black: (0,0),(2,0)  [opener moves 1,3]  +  (4,0)  [move 5, coords[1]]
    #   white: (1,0)        [opener move 2]      +  (3,0)  [move 4, coords[0]]
    exp_black, exp_white = _board_from_xy([(0, 0), (2, 0), (4, 0)], [(1, 0), (3, 0)])
    assert np.array_equal(state.black, exp_black), "PLACE2 black squares wrong (move-5 parity)"
    assert np.array_equal(state.white, exp_white), "PLACE2 white squares wrong (move-4 parity)"
    assert state.n_black == 3 and state.n_white == 2


# ---------------------------------------------------------------------------
# Branch: engine is OPENER picking color (5 stones), replies SWAP.
# ---------------------------------------------------------------------------


def _drive_to_pick_with_our_place2(oracle, eng_first_three_xy):
    """Helper: build a 5-stone board where WE (responder) did PLACE2, reaching the
    engine's PICK node. The engine OPENED the 3 stones; we then PLACE2 two stones;
    the engine is OPENER so it PICKS. Returns the OpeningState at the engine PICK
    node and the engine handle (caller closes)."""
    # Not used directly — the SWAP/ONE pick branches drive from our own PLACE2 so
    # the 5-stone board is fully deterministic.
    raise NotImplementedError


def test_engine_opener_pick_swap_board():
    """We (responder) PLACE2 to 5 stones, engine (opener) PICKs SWAP -> takes
    BLACK; we are white and move next. The 5-stone board is unchanged by SWAP."""
    # WE are the RESPONDER here, so WE open? No: roles — to reach an engine PICK,
    # the OPENER must be the engine. But PLACE2 is the RESPONDER's action, and the
    # PICK is the OPENER's. So: engine=OPENER opens 3, we=RESPONDER PLACE2 to 5,
    # engine=OPENER PICKs. Drive that explicitly.
    eng = _engine("swap2_swap")  # engine replies SWAP at the pick (5 stones)
    # We need the engine to OPEN first, then us PLACE2, then engine PICK. But the
    # swap2_swap fake replies SWAP at ALL stone counts (3 and 5), and OPEN (0) is
    # an arity error. So drive the open with a separate opener and the pick with
    # swap2_swap is impossible in one process. Instead construct the 5-stone board
    # directly: engine opener (swap2_open mode) opens 3, we PLACE2, then a SECOND
    # engine handle in swap2_swap mode answers the pick.
    eng.close()

    eng_open = _engine("swap2_open")
    eng_pick = _engine("swap2_swap")
    oracle = _spike_oracle([])  # our PLACE2 placements use lowest-empty fallback
    try:
        state = initial_opening()
        # Engine OPENS (b,w,b at (0,0),(1,0),(2,0)).
        state = _engine_open(state, eng_open)
        assert state.node.name == "P2_CHOICE" and state.to_act == Actor.RESPONDER
        # WE (responder) PLACE2: apply RESP_PLACE2 then two placements via agent.
        from gomoku.swap2_search import agent_act

        slot, _rec = agent_act(oracle, state, np.random.default_rng(0), greedy=True)
        # Force the PLACE2 branch (value is constant so any legal slot is argmax;
        # apply PLACE2 explicitly to land on the 5-stone pick node).
        state = state.apply(RESP_PLACE2)
        for _ in range(2):
            action, _rec = agent_act(oracle, state, np.random.default_rng(0), greedy=True)
            state = state.apply(action)
        assert state.node.name == "P1_PICK"
        # Now the ENGINE (opener) PICKs SWAP.
        state, engine_first = _engine_pick(state, eng_pick)
    finally:
        eng_open.close()
        eng_pick.close()

    assert engine_first is None  # SWAP reveals no engine move
    assert state.is_done()
    # 5-stone board: engine opener b,w,b at (0,0),(1,0),(2,0); our PLACE2 fills
    # lowest empties (3,0) black-step then (4,0) white-step (P2_P2_B then P2_P2_W):
    #   black: (0,0),(2,0),(3,0)   white: (1,0),(4,0)
    exp_black, exp_white = _board_from_xy([(0, 0), (2, 0), (3, 0)], [(1, 0), (4, 0)])
    black, white = _normal_from_state(state)
    assert np.array_equal(black, exp_black), "pick-SWAP black squares wrong"
    assert np.array_equal(white, exp_white), "pick-SWAP white squares wrong"
    assert state.opener_color == Color.BLACK   # engine(opener) took black
    assert state.mover_color == Color.WHITE     # 3B+2W -> white to move
    assert state.mover_actor() == Actor.RESPONDER  # responder (us) is white


# ---------------------------------------------------------------------------
# Branch: engine is OPENER picking color (5 stones), replies ONE_COORD.
# ---------------------------------------------------------------------------


def test_engine_opener_pick_one_coord_board_and_reveal():
    """Engine (opener) opens 3, we PLACE2 to 5, engine PICKs ONE_COORD -> takes
    WHITE and plays move 6 at the lowest empty (5,0). The 5-stone opening board is
    unchanged by the pick; the engine's FIRST normal move is revealed (5,0)."""
    eng_open = _engine("swap2_open")
    eng_pick = _engine("swap2_one")
    oracle = _spike_oracle([])
    try:
        state = initial_opening()
        state = _engine_open(state, eng_open)
        state = state.apply(RESP_PLACE2)
        from gomoku.swap2_search import agent_act

        for _ in range(2):
            action, _rec = agent_act(oracle, state, np.random.default_rng(0), greedy=True)
            state = state.apply(action)
        assert state.node.name == "P1_PICK"
        state, engine_first = _engine_pick(state, eng_pick)
    finally:
        eng_open.close()
        eng_pick.close()

    assert state.is_done()
    # 5-stone opening board (unchanged by the WHITE pick):
    #   black: (0,0),(2,0),(3,0)   white: (1,0),(4,0)
    exp_black, exp_white = _board_from_xy([(0, 0), (2, 0), (3, 0)], [(1, 0), (4, 0)])
    black, white = _normal_from_state(state)
    assert np.array_equal(black, exp_black), "pick-ONE_COORD black squares wrong"
    assert np.array_equal(white, exp_white), "pick-ONE_COORD white squares wrong"
    assert state.opener_color == Color.WHITE       # engine kept white
    assert state.mover_color == Color.WHITE         # 3B+2W -> white to move
    assert state.mover_actor() == Actor.OPENER      # opener (engine) is white
    # The engine's first normal move is revealed (lowest empty after 5 stones).
    assert engine_first == _sq(5, 0)


# ---------------------------------------------------------------------------
# End-to-end: a full game completes and returns a well-formed outcome.
# ---------------------------------------------------------------------------


def _stub_picker(square_plan: list[int] | None = None):
    """A deterministic normal-play picker: lowest empty legal square (or follow a
    plan). No MCTS, no net — just exercises the driver's normal-play loop."""

    def pick(gs: GameState, rng: np.random.Generator) -> int:
        legal = set(int(a) for a in gs.legal_actions())
        return min(legal)

    return pick


def test_full_game_completes_and_is_well_formed():
    oracle = _spike_oracle([])
    eng = _engine("swap2_open")
    try:
        out = play_swap2_game(
            oracle, eng, Actor.RESPONDER, np.random.default_rng(0),
            model_picker=_stub_picker(),
        )
    finally:
        eng.close()
    assert isinstance(out, GameOutcome)
    assert out.result in ("win", "loss", "draw")
    assert out.our_role == Actor.RESPONDER
    assert out.opener_color in (Color.BLACK, Color.WHITE)
    assert out.our_color in (Color.BLACK, Color.WHITE)
    assert out.normal_plies >= 0
    assert out.opening_stones in (3, 4, 5)


def test_roles_alternate_and_both_colors_reachable():
    """Across games, roles alternate (opener/responder) and both color outcomes
    for our net are reachable. We drive the per-game loop directly with the fake
    engine + stub picker (no checkpoint load)."""
    outcomes: list[GameOutcome] = []
    for g in range(8):
        role = Actor.OPENER if g % 2 == 0 else Actor.RESPONDER
        oracle = _spike_oracle([])
        # The engine mode must match the engine's ROLE, which is the OPPOSITE of
        # ours: when WE respond the engine OPENS (must reply 3 coords -> the
        # arity of swap2_open at 0 stones); when WE open the engine RESPONDS
        # (must reply SWAP / one / two coords). Within the opener-for-us games we
        # alternate the responder reply between SWAP and ONE_COORD so BOTH of our
        # color outcomes arise deterministically: SWAP -> engine takes black ->
        # we (opener) play WHITE; ONE_COORD (engine STAYs white) -> we keep BLACK.
        if role == Actor.RESPONDER:
            mode = "swap2_open"  # engine is the OPENER here
        else:
            # engine is the RESPONDER here; pick a respond-style reply
            mode = "swap2_swap" if (g // 2) % 2 == 0 else "swap2_one"
        eng = _engine(mode)
        try:
            out = play_swap2_game(
                oracle, eng, role, np.random.default_rng(g),
                model_picker=_stub_picker(),
            )
        finally:
            eng.close()
        outcomes.append(out)

    roles = {o.our_role for o in outcomes}
    assert roles == {Actor.OPENER, Actor.RESPONDER}, "roles did not alternate"
    colors = {o.our_color for o in outcomes}
    assert Color.BLACK in colors and Color.WHITE in colors, "a color outcome unreachable"


# ---------------------------------------------------------------------------
# Aggregation: the pure result builder and its invariants.
# ---------------------------------------------------------------------------


def test_aggregate_outcomes_splits_consistent():
    outs = [
        GameOutcome(Actor.OPENER, Color.BLACK, Color.BLACK, "win", 10, 3),
        GameOutcome(Actor.RESPONDER, Color.BLACK, Color.WHITE, "loss", 12, 3),
        GameOutcome(Actor.OPENER, Color.WHITE, Color.WHITE, "draw", 80, 5),
        GameOutcome(Actor.RESPONDER, Color.WHITE, Color.BLACK, "win", 9, 4),
    ]
    res = aggregate_outcomes(outs)
    assert isinstance(res, Swap2EvalResult)
    assert res.n_games == 4
    assert (res.wins, res.losses, res.draws) == (2, 1, 1)
    # black[i] + white[i] == opener[i] + responder[i] == aggregate count, each i.
    for i, agg in enumerate((res.wins, res.losses, res.draws)):
        assert res.black[i] + res.white[i] == agg
        assert res.opener[i] + res.responder[i] == agg
    # opener_color distribution sums to n.
    assert res.opener_color_black + res.opener_color_white == res.n_games
    d = res.to_dict()
    assert d["win_rate"] == pytest.approx((2 + 0.5 * 1) / 4)
    assert set(d.keys()) >= {
        "n_games", "wins", "losses", "draws", "win_rate",
        "by_color", "by_role", "opener_color_dist",
    }


# ---------------------------------------------------------------------------
# Parallel (--jobs) determinism: the partition + per-game (role, rng) helpers,
# then an end-to-end jobs=1 vs jobs=3 EXACT-match against the fake engine.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_games,jobs,expected",
    [
        (6, 1, [[0, 1, 2, 3, 4, 5]]),
        (6, 3, [[0, 1], [2, 3], [4, 5]]),
        (6, 4, [[0, 1], [2, 3], [4], [5]]),   # remainder spread over leading chunks
        (5, 2, [[0, 1, 2], [3, 4]]),
        (3, 8, [[0], [1], [2]]),              # jobs capped at n_games (no empty chunks)
        (0, 4, []),
        (1, 4, [[0]]),
    ],
)
def test_split_game_indices_covers_range(n_games, jobs, expected):
    chunks = _split_game_indices(n_games, jobs)
    assert chunks == expected
    # Chunks concatenate back to range(n_games) exactly -> same multiset of games.
    flat = [i for c in chunks for i in c]
    assert flat == list(range(n_games))
    assert all(c for c in chunks), "no empty chunks"


def test_game_role_alternates_from_opener():
    assert _game_role(0) == Actor.OPENER
    assert _game_role(1) == Actor.RESPONDER
    assert _game_role(2) == Actor.OPENER
    assert _game_role(3) == Actor.RESPONDER


def test_game_rng_is_index_and_seed_deterministic():
    """game i's RNG depends only on (seed, i): same (seed, i) -> same stream;
    different i (or seed) -> different stream. This is what makes the union of
    games identical across any --jobs partition."""
    a = _game_rng(7, 6, 2).random(8)
    b = _game_rng(7, 6, 2).random(8)
    assert np.array_equal(a, b), "same (seed, i) must reproduce the stream"
    c = _game_rng(7, 6, 3).random(8)
    assert not np.array_equal(a, c), "different game index must differ"
    d = _game_rng(99, 6, 2).random(8)
    assert not np.array_equal(a, d), "different seed must differ"


@pytest.fixture(scope="module")
def tiny_checkpoint_path():
    """A fresh tiny CPU model saved once for the module (mirrors the eval_vs_rapfi
    parallel test). Lets us drive the real ``eval_swap2_vs_engine`` end-to-end."""
    from gomoku.model import build_model, save_checkpoint

    model = build_model("tiny")
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(path, model, optimizer=None, epoch=0, total_games=0)
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _fake_engine_cfg(mode: str = "swap2_open"):
    """An ExternalEngineConfig that launches the shipped fake Piskvork engine — no
    real Rapfi, no MPS. The fake plays the swap2 negotiation AND normal play."""
    return ExternalEngineConfig(
        cmd=f"{sys.executable} {_STUB} {mode}",
        timeout_ms=500,
        label=f"stub:{mode}",
    )


def test_eval_swap2_jobs1_well_formed(tiny_checkpoint_path):
    """jobs=1 (serial) returns the full result-dict shape with consistent splits."""
    out = eval_swap2_vs_engine(
        tiny_checkpoint_path, _fake_engine_cfg(), n_games=4,
        sims=4, seed=0, device="cpu", jobs=1,
    )
    assert out["n_games"] == 4
    assert out["wins"] + out["losses"] + out["draws"] == 4
    assert out["jobs"] == 1
    assert set(out.keys()) >= {
        "n_games", "wins", "losses", "draws", "win_rate",
        "by_color", "by_role", "opener_color_dist", "protocol", "wall_secs",
    }
    # by_color and by_role splits each sum to n.
    bc, br = out["by_color"], out["by_role"]
    for i in range(3):
        assert bc["black"][i] + bc["white"][i] == [out["wins"], out["losses"], out["draws"]][i]
        assert br["opener"][i] + br["responder"][i] == [out["wins"], out["losses"], out["draws"]][i]


def test_eval_swap2_jobs3_equals_jobs1_exact(tiny_checkpoint_path):
    """The correctness guarantee: jobs=3 plays the SAME multiset of games as jobs=1
    for the same seed/n, so every aggregate + per-color/role/opener-color split is
    IDENTICAL. Determinism holds because each game's (role, rng) is derived only
    from (seed, game_index), the tiny net is deterministic on CPU, and the fake
    engine is deterministic — so the process partition cannot change any outcome."""
    serial = eval_swap2_vs_engine(
        tiny_checkpoint_path, _fake_engine_cfg(), n_games=6,
        sims=4, seed=123, device="cpu", jobs=1,
    )
    parallel = eval_swap2_vs_engine(
        tiny_checkpoint_path, _fake_engine_cfg(), n_games=6,
        sims=4, seed=123, device="cpu", jobs=3,
    )

    assert serial["n_games"] == parallel["n_games"] == 6
    assert serial["wins"] + serial["losses"] + serial["draws"] == 6
    for key in ("wins", "losses", "draws", "win_rate", "by_color", "by_role",
                "opener_color_dist"):
        assert serial[key] == parallel[key], f"{key} differs between jobs=1 and jobs=3"


# ---------------------------------------------------------------------------
# NET-vs-NET head-to-head (the #72 progress gate).
#
# Two tiny CPU nets negotiate swap2 against each other; the result is attributed
# to NET A. No external engine, so the whole thing is pure-torch/CPU and the
# jobs=1 == jobs=2 aggregate is EXACT (the correctness guarantee).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_tiny_checkpoints():
    """Two distinct tiny CPU nets, saved once for the module. Different build
    seeds give genuinely different policies/values so A-vs-B is a real contest
    (and not a degenerate A-vs-A self-play)."""
    import torch

    from gomoku.model import build_model, save_checkpoint

    paths = []
    for s in (11, 22):
        torch.manual_seed(s)
        model = build_model("tiny")
        model.eval()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        save_checkpoint(path, model, optimizer=None, epoch=0, total_games=0)
        paths.append(path)
    try:
        yield paths[0], paths[1]
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def test_play_swap2_game_h2h_well_formed():
    """One H2H game between two stub oracles + stub pickers completes and is
    attributed to NET A (our_role == role_a)."""
    oracle_a = _spike_oracle([])
    oracle_b = _spike_oracle([])
    out = play_swap2_game_h2h(
        oracle_a, _stub_picker(), oracle_b, _stub_picker(),
        Actor.OPENER, np.random.default_rng(0),
    )
    assert isinstance(out, GameOutcome)
    assert out.result in ("win", "loss", "draw")
    assert out.our_role == Actor.OPENER          # role_a is A's role
    assert out.our_color in (Color.BLACK, Color.WHITE)
    assert out.opener_color in (Color.BLACK, Color.WHITE)
    # A opened -> A's color is opener_color.
    assert out.our_color == out.opener_color
    assert out.normal_plies >= 0
    assert out.opening_stones in (3, 4, 5)


def test_play_swap2_game_h2h_responder_role_attribution():
    """When role_a is RESPONDER, NET A's color is the NON-opener color."""
    oracle_a = _spike_oracle([])
    oracle_b = _spike_oracle([])
    out = play_swap2_game_h2h(
        oracle_a, _stub_picker(), oracle_b, _stub_picker(),
        Actor.RESPONDER, np.random.default_rng(1),
    )
    assert out.our_role == Actor.RESPONDER
    # A responded -> A's color is the OTHER color from opener_color.
    assert out.our_color != out.opener_color


def test_eval_swap2_h2h_well_formed(two_tiny_checkpoints):
    """eval_swap2_h2h returns a well-formed result dict; W+L+D == n; the
    by_color / by_role splits each sum to n; both checkpoint paths are recorded."""
    ckpt_a, ckpt_b = two_tiny_checkpoints
    out = eval_swap2_h2h(
        ckpt_a, ckpt_b, n_games=6, sims=4, seed=0, jobs=1, device="cpu",
    )
    assert out["n_games"] == 6
    assert out["wins"] + out["losses"] + out["draws"] == 6
    assert out["mode"] == "h2h"
    assert out["jobs"] == 1
    assert out["checkpoint_a"] == os.path.abspath(ckpt_a)
    assert out["checkpoint_b"] == os.path.abspath(ckpt_b)
    assert 0.0 <= out["win_rate"] <= 1.0
    assert set(out.keys()) >= {
        "n_games", "wins", "losses", "draws", "win_rate",
        "by_color", "by_role", "opener_color_dist", "protocol", "mode",
        "checkpoint_a", "checkpoint_b", "wall_secs",
    }
    wld = [out["wins"], out["losses"], out["draws"]]
    bc, br = out["by_color"], out["by_role"]
    for i in range(3):
        assert bc["black"][i] + bc["white"][i] == wld[i], "by_color split must sum to n"
        assert br["opener"][i] + br["responder"][i] == wld[i], "by_role split must sum to n"
    # opener_color distribution sums to n.
    ocd = out["opener_color_dist"]
    assert ocd["black"] + ocd["white"] == 6


def test_eval_swap2_h2h_self_play_completes(two_tiny_checkpoints):
    """Self-play sanity: NET A vs the SAME net (A vs A) over many games completes
    and sums correctly. Nets are random-init so we don't over-constrain the score
    (just that it's a valid aggregate over all n games)."""
    ckpt_a, _ = two_tiny_checkpoints
    out = eval_swap2_h2h(
        ckpt_a, ckpt_a, n_games=8, sims=4, seed=5, jobs=1, device="cpu",
    )
    assert out["n_games"] == 8
    assert out["wins"] + out["losses"] + out["draws"] == 8
    assert 0.0 <= out["win_rate"] <= 1.0


def test_eval_swap2_h2h_jobs1_equals_jobs2_exact(two_tiny_checkpoints):
    """THE H2H correctness guarantee: with NO engine, two pure-torch CPU nets are
    fully deterministic and each game's (role, rng) is derived only from
    (seed, game_index), so jobs=2 plays the EXACT SAME games as jobs=1 — every
    aggregate + per-color/role/opener-color split is BYTE-identical. (Stronger
    than the vs-engine 'equivalent': there is no engine nondeterminism here.)"""
    ckpt_a, ckpt_b = two_tiny_checkpoints
    serial = eval_swap2_h2h(
        ckpt_a, ckpt_b, n_games=6, sims=4, seed=321, jobs=1, device="cpu",
    )
    parallel = eval_swap2_h2h(
        ckpt_a, ckpt_b, n_games=6, sims=4, seed=321, jobs=2, device="cpu",
    )
    assert serial["n_games"] == parallel["n_games"] == 6
    for key in ("wins", "losses", "draws", "win_rate", "by_color", "by_role",
                "opener_color_dist"):
        assert serial[key] == parallel[key], f"{key} differs between jobs=1 and jobs=2"
