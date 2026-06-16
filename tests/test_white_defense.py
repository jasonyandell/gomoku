"""White-defense EVAL SUITE tests (GH #45) — 15x15.

Board size is a process-level constant, so these run only under
``GOMOKU_BOARD_SIZE=15``:

    GOMOKU_BOARD_SIZE=15 pytest tests/test_white_defense.py

Covers the four verification rungs from the issue:
  1. planes <-> GameState round-trip is lossless (the correctness keystone).
  2. defend_one_position STARTS from the injected position (not initial()).
  3. binomial_ci is correct on a known proportion (CI-math sanity).
  4. a mined fixture loads and is non-empty + white-to-move + threatened.
"""

from __future__ import annotations

import numpy as np
import pytest

from gomoku import game
from gomoku.game import GameState

pytestmark = pytest.mark.skipif(
    game.BOARD_SIZE != 15,
    reason="white-defense suite tests need GOMOKU_BOARD_SIZE=15 (size fixed per process)",
)

N = game.BOARD_SIZE


def _a(r: int, c: int) -> int:
    return r * N + c


def _states_equivalent(s1: GameState, s2: GameState) -> bool:
    """Equivalent for everything the H2H loop reads: same board, same
    side-to-move, same legal moves, same network input planes."""
    if not np.array_equal(s1.board, s2.board):
        return False
    if (s1.move_count % 2) != (s2.move_count % 2):
        return False
    if not np.array_equal(s1.legal_mask(), s2.legal_mask()):
        return False
    if not np.array_equal(s1.to_planes(), s2.to_planes()):
        return False
    return True


def _play_some(moves: list[int]) -> GameState:
    state = GameState.initial()
    for m in moves:
        state = state.apply(m)
    return state


# --------------------------------------------------------------------------- #
# 1. Round-trip: GameState -> planes -> GameState yields an equivalent state.   #
# --------------------------------------------------------------------------- #
def test_planes_state_roundtrip_lossless():
    from gomoku.white_defense import planes_to_state

    # A mid-game state with real history (>= HISTORY_PLY plies), white to move.
    moves = [
        _a(7, 7), _a(7, 8), _a(8, 7), _a(6, 8), _a(9, 7),
        _a(5, 8), _a(10, 7), _a(4, 8), _a(11, 7),  # 9 plies -> white to move
    ]
    state = _play_some(moves)
    assert state.move_count % 2 == 1, "expected white to move (odd ply)"

    planes = state.to_planes()
    rt = planes_to_state(planes, ply=state.move_count)

    assert _states_equivalent(state, rt), "round-trip changed the state"
    # The reconstructed planes must be byte-identical (the net sees the same input).
    assert np.array_equal(state.to_planes(), rt.to_planes())
    # Side-to-move preserved.
    assert (state.move_count % 2) == (rt.move_count % 2)
    # Legal actions preserved exactly.
    assert np.array_equal(state.legal_actions(), rt.legal_actions())


def test_roundtrip_empty_and_early_positions():
    from gomoku.white_defense import planes_to_state

    # Empty board (black to move): no history, round-trips trivially.
    s0 = GameState.initial()
    rt0 = planes_to_state(s0.to_planes(), ply=0)
    assert _states_equivalent(s0, rt0)

    # One stone in (white to move, 1 ply of history).
    s1 = GameState.initial().apply(_a(7, 7))
    rt1 = planes_to_state(s1.to_planes(), ply=1)
    assert _states_equivalent(s1, rt1)


# --------------------------------------------------------------------------- #
# 2. Inject-start-state: the game STARTS from the injected position.           #
# --------------------------------------------------------------------------- #
def test_defend_starts_from_injected_position():
    from gomoku.white_defense import defend_one_position

    # White to move with some stones already on the board.
    moves = [_a(7, 7), _a(0, 0), _a(7, 8), _a(0, 1), _a(7, 9)]  # 5 plies, white to move
    start = _play_some(moves)
    assert start.move_count % 2 == 1
    n_stones_start = int(start.board[0].sum() + start.board[1].sum())
    assert n_stones_start == 5

    seen_first_state = {}

    def recording_white(state: GameState, rng: np.random.Generator) -> int:
        # On the FIRST white move, capture the board it was handed.
        seen_first_state.setdefault("board", state.board.copy())
        seen_first_state.setdefault("stones", int(state.board[0].sum() + state.board[1].sum()))
        return int(state.legal_actions()[0])

    def passive_black(state: GameState, rng: np.random.Generator) -> int:
        return int(state.legal_actions()[-1])

    rng = np.random.default_rng(0)
    defend_one_position(recording_white, passive_black, start, rng=rng)

    # The defender's first decision must have been from the injected board
    # (5 stones), NOT from GameState.initial() (0 stones).
    assert seen_first_state["stones"] == n_stones_start
    assert np.array_equal(seen_first_state["board"], start.board)


def test_defend_already_won_position_is_draw_guard():
    # An already-terminal injected position returns a draw (defensive guard),
    # never crashes the loop.
    from gomoku.white_defense import defend_one_position

    # Build a black 5-in-a-row, then a state where it's "white to move" but black
    # already has five (terminal). We fabricate by applying through a win.
    state = GameState.initial()
    # black plays a row of 5; white plays harmless filler far away.
    blacks = [_a(0, c) for c in range(5)]
    fillers = [_a(14, c) for c in range(4)]
    for i, b in enumerate(blacks):
        state = state.apply(b)
        if i < len(fillers):
            state = state.apply(fillers[i])
    done, _ = state.is_terminal()
    assert done  # black has five
    out = defend_one_position(lambda s, r: int(s.legal_actions()[0]),
                              lambda s, r: int(s.legal_actions()[0]),
                              state, rng=np.random.default_rng(0))
    assert out == "draw"


# --------------------------------------------------------------------------- #
# 3. CI-math sanity: binomial_ci on a known proportion.                        #
# --------------------------------------------------------------------------- #
def test_binomial_ci_known_proportion():
    from scripts.delta_e_harness import binomial_ci

    # 50/100 -> Wilson 95% CI is ~[0.404, 0.596], symmetric-ish around 0.5.
    lo, hi = binomial_ci(50.0, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert abs((lo + hi) / 2.0 - 0.5) < 0.01  # Wilson center ~ p for p=0.5
    assert 0.39 < lo < 0.41
    assert 0.59 < hi < 0.61

    # Tighter at larger n: half-width shrinks ~1/sqrt(n).
    lo2, hi2 = binomial_ci(500.0, 1000)
    assert (hi2 - lo2) < (hi - lo)

    # Degenerate / edge cases are clamped to [0, 1].
    lo3, hi3 = binomial_ci(0.0, 0)
    assert (lo3, hi3) == (0.0, 1.0)
    lo4, hi4 = binomial_ci(0.0, 50)
    assert lo4 == pytest.approx(0.0, abs=1e-9) and 0.0 < hi4 < 0.2
    lo5, hi5 = binomial_ci(50.0, 50)
    assert hi5 == pytest.approx(1.0, abs=1e-9) and 0.8 < lo5 < 1.0


# --------------------------------------------------------------------------- #
# 4. Fixture loads, is non-empty, and is genuinely white-defense.              #
# --------------------------------------------------------------------------- #
def test_mine_save_load_fixture_roundtrip(tmp_path):
    from gomoku.white_defense import (
        black_threat_present,
        load_fixture,
    )
    from scripts.white_defense_suite import cmd_mine, parse_args

    out = tmp_path / "wd_fixture.json"
    args = parse_args(["mine", "--out", str(out), "--target", "12", "--games", "30", "--seed", "0"])
    rc = cmd_mine(args)
    assert rc == 0
    assert out.exists()

    fx = load_fixture(out)
    assert len(fx) > 0, "fixture must be non-empty"
    # Every fixture position is white-to-move with a live black threat.
    for st in fx.states():
        assert st.move_count % 2 == 1, "fixture position must be white-to-move"
        assert black_threat_present(st), "fixture position must face a live black threat"
    # Provenance + meta carry the regen command (the documented regenerator).
    assert "regen_command" in fx.meta
    # Stored as move sequences: each position replays to a ply == len(moves).
    assert all("moves" in p and "ply" in p for p in fx.positions)
    assert all(p["ply"] == len(p["moves"]) for p in fx.positions)


def test_gate_seam_signature_matches_candidate_white_loss_tally(tmp_path):
    """white_defense_tally (fixture-bound via partial) drops into run_gate's
    white_loss_fn seam: it accepts the SAME positional+kw call the seam makes and
    returns the SAME (rate, defense_score, n) tuple shape as
    candidate_white_loss_tally — with NO change to decide_verdict."""
    import functools
    import inspect

    from gomoku.model import build_model, save_checkpoint
    from gomoku.white_defense import white_defense_tally
    from scripts.sliding_gate import candidate_white_loss_tally
    from scripts.white_defense_suite import cmd_mine, parse_args

    # A tiny fixture + a tiny random-init net (CPU) keep this fast.
    fixture = tmp_path / "seam_fixture.json"
    cmd_mine(parse_args(["mine", "--out", str(fixture), "--target", "6",
                         "--games", "20", "--seed", "0"]))
    ckpt = str(tmp_path / "tiny.pt")
    save_checkpoint(ckpt, build_model("small"))

    # Bind the fixture-specific args; the result is a drop-in white_loss_fn.
    fn = functools.partial(
        white_defense_tally, fixture_path=str(fixture),
        attacker="lookahead:depth=1", n_seeds=1,
    )
    # The EXACT kwargs run_gate passes to white_loss_fn (sliding_gate.py:522).
    seam_kwargs = dict(n_games=4, sims=8, c_puct=1.5, seed=0,
                       device="cpu", opening_plies=0)
    rate, defense_score, n = fn(ckpt, ckpt, **seam_kwargs)

    assert isinstance(n, int) and n > 0
    assert isinstance(defense_score, float)
    assert rate is None or isinstance(rate, float)
    # defense_score = successful defenses = n - losses.
    assert defense_score == float(n - round(rate * n))
    # Same declared return shape as the function it replaces in the seam.
    assert (
        inspect.signature(candidate_white_loss_tally).return_annotation
        == "tuple[Optional[float], float, int]"
    )


def test_checked_in_fixture_present_and_valid():
    """The committed fixture must exist, load, and be white-defense-valid.

    Skips (rather than fails) if the committed file is absent — keeps the test
    portable to a fresh checkout before the fixture is generated, while still
    guarding the artifact when it IS present."""
    from pathlib import Path

    from gomoku.white_defense import black_threat_present, load_fixture

    repo = Path(__file__).resolve().parent.parent
    path = repo / "fixtures" / f"white_defense_{N}x{N}_v1.json"
    if not path.exists():
        pytest.skip(f"committed fixture not present at {path}")
    fx = load_fixture(path)
    assert len(fx) > 0
    assert fx.board_size == N
    for st in fx.states():
        assert st.move_count % 2 == 1
        assert black_threat_present(st)
