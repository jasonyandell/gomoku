"""Eval core: fixed-opening start state, the net picker, rulers, panel rows."""

from __future__ import annotations

import os

import numpy as np
import pytest

from gomoku.board_config import BOARD_SIZE
from gomoku.eval import MatchResult, play_match_pickers
from gomoku.eval_panel import (
    EvaluatorCache,
    Ruler,
    eval_vs_ruler,
    fixed_opening_state,
    make_net_picker,
    result_to_row,
)
from gomoku.game import GameState
from gomoku.model import build_model, save_checkpoint


# --- fixed_opening_state ---------------------------------------------------
def test_fixed_opening_state_places_three_stones():
    s = fixed_opening_state(((3, 2), (5, 4), (4, 5)))
    assert s.move_count == 3
    # 3 stones placed -> 2 for the first mover, 1 for the second; white to move.
    total = int(s.board[0].sum() + s.board[1].sum())
    assert total == 3


def test_fixed_opening_state_rejects_collision():
    with pytest.raises(ValueError):
        fixed_opening_state(((4, 4), (4, 4)))


# --- play_match_pickers start_state (the additive extension) ---------------
def _capture_picker(seen):
    def pick(state: GameState, rng) -> int:
        seen.append(state.move_count)
        return int(state.legal_actions()[0])

    return pick


def test_start_state_every_game_begins_at_opening():
    start = fixed_opening_state(((3, 2), (5, 4), (4, 5)))
    seen: list[int] = []
    res = play_match_pickers(
        _capture_picker(seen), _capture_picker(seen),
        n_games=4, seed=0, start_state=start,
    )
    assert res.n_games == 4
    # No game ever starts before the opening's 3 plies.
    assert min(seen) >= 3
    assert 0 not in seen


def test_default_no_start_state_begins_at_move_zero():
    seen: list[int] = []
    play_match_pickers(
        _capture_picker(seen), _capture_picker(seen), n_games=2, seed=0
    )
    assert 0 in seen  # games start from the empty board, byte-identical to legacy


# --- net picker ------------------------------------------------------------
def test_make_net_picker_returns_legal_argmax():
    model = build_model("small")
    from gomoku.mcts import make_torch_evaluator

    ev = make_torch_evaluator(model, "cpu")
    pick = make_net_picker(ev, sims=8, temp_until_ply=0)
    s = GameState.initial()
    a = pick(s, np.random.default_rng(0))
    assert a in set(int(x) for x in s.legal_actions())


def test_net_picker_temperature_gives_variety():
    model = build_model("small")
    from gomoku.mcts import make_torch_evaluator

    ev = make_torch_evaluator(model, "cpu")
    pick = make_net_picker(ev, sims=16, temp_until_ply=99, temperature=1.5)
    s = GameState.initial()
    moves = {pick(s, np.random.default_rng(k)) for k in range(12)}
    # Temperature sampling on the empty board should produce >1 distinct move.
    assert len(moves) > 1


# --- result_to_row: white reported separately (issue #34) ------------------
def test_result_to_row_splits_white_and_black():
    res = MatchResult(
        n_games=4, wins=2, losses=2, draws=0,
        black_w=1, black_l=1, black_d=0,
        white_w=1, white_l=1, white_d=0,
    )
    row = result_to_row("champ", res)
    assert row["ruler"] == "champ"
    assert row["white_games"] == 2 and row["black_games"] == 2
    assert row["white_loss_rate"] == 0.5
    assert row["white_score"] == 0.5
    # white must be its own field, never folded into the aggregate
    assert "white_score" in row and "black_score" in row


# --- eval_vs_ruler end-to-end vs a baseline (no Rapfi needed) --------------
def test_eval_vs_baseline_ruler(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    save_checkpoint(ckpt, build_model("small"), epoch=7)
    cache = EvaluatorCache(device="cpu")
    ruler = Ruler(label="look1", opponent="lookahead:depth=1", n_games=2, sims=6, temp_until_ply=99)
    res = eval_vs_ruler(ckpt, ruler, cache=cache, seed=0)
    assert res.n_games == 2
    assert res.wins + res.losses + res.draws == 2


def test_evaluator_cache_reloads_on_mtime_change(tmp_path):
    ckpt = str(tmp_path / "c.pt")
    save_checkpoint(ckpt, build_model("small"), epoch=11)
    cache = EvaluatorCache(device="cpu")
    assert cache.epoch(ckpt) == 11
    # Rewrite with a new epoch and bump mtime -> cache must reload.
    save_checkpoint(ckpt, build_model("small"), epoch=22)
    os.utime(ckpt, (os.path.getmtime(ckpt) + 5, os.path.getmtime(ckpt) + 5))
    assert cache.epoch(ckpt) == 22
