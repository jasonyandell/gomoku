"""WL5 archive-start mechanism tests.

Covers:
- `_gamestate_from_archive` produces a `GameState` whose current/opponent
  planes match the archived planes (history slots are not preserved by
  design; only plane 0 and plane HISTORY_PLY are asserted equal).
- `_generate_games_native` with `archive_start_frac=1.0` starts every
  game from an archived position rather than the empty board.
- The cell wiring in `scripts/run_sweep.py` dispatches the new flags
  for trainer + worker commands.

Design ref: wiki/topics/wl5-diagnostics-archive-start-design.md
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gomoku import native_mcts
from gomoku.game import (
    BOARD_SIZE,
    GameState,
    HISTORY_PLY,
    N_ACTIONS,
    N_INPUT_PLANES,
    str_to_action,
)
from gomoku.self_play import _gamestate_from_archive, _generate_games_native
from scripts.run_sweep import CELLS, trainer_cmd, worker_cmd, cell_dirs


native_only = pytest.mark.skipif(
    not native_mcts.USING_NATIVE_MCTS,
    reason="native MCTS extension is not built",
)


def _build_synthetic_archive(n: int = 5) -> dict:
    """Return a tiny torch-load-shaped archive whose positions come from
    short, hand-played sequences. Index 0 is the empty board; later
    indices add more stones."""
    states: list[GameState] = []
    s = GameState.initial()
    states.append(s)
    move_seqs = [
        ["e5"],
        ["e5", "d5"],
        ["e5", "d5", "e6"],
        ["e5", "d5", "e6", "d6"],
    ]
    for seq in move_seqs[: max(0, n - 1)]:
        s = GameState.initial()
        for m in seq:
            s = s.apply(str_to_action(m))
        states.append(s)

    n = len(states)
    planes = torch.zeros((n, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    pi_mcts = torch.zeros((n, N_ACTIONS), dtype=torch.float32)
    pi_mcts[:, 0] = 1.0  # arbitrary; not used by archive-start path
    z = torch.zeros((n,), dtype=torch.float32)
    side = torch.zeros((n,), dtype=torch.int64)
    ply = torch.zeros((n,), dtype=torch.int64)
    for i, st in enumerate(states):
        planes[i] = torch.from_numpy(st.to_planes())
        side[i] = st.move_count % 2
        ply[i] = st.move_count
    return {
        "planes": planes,
        "pi_mcts": pi_mcts,
        "z": z,
        "provenance": ["synthetic"] * n,
        "side": side,
        "ply": ply,
    }


def test_gamestate_from_archive_roundtrips_current_and_opponent_planes():
    archive = _build_synthetic_archive(5)
    for i in range(len(archive["provenance"])):
        s = _gamestate_from_archive(archive, i)
        orig = archive["planes"][i].numpy()
        new = s.to_planes()
        # Current-side stones (plane 0) and opponent stones (plane H) must
        # match exactly — that's what the archive carries through.
        assert np.array_equal(new[0], orig[0]), f"plane 0 mismatch at idx {i}"
        assert np.array_equal(new[HISTORY_PLY], orig[HISTORY_PLY]), (
            f"plane {HISTORY_PLY} mismatch at idx {i}"
        )
        # move_count + parity preserved.
        assert s.move_count == int(archive["ply"][i].item())
        assert (s.move_count % 2) == int(archive["side"][i].item())


@native_only
def test_generate_games_native_uses_archive_when_frac_is_one():
    """With frac=1.0 every game must start from a non-empty archived
    position. We use indices 1..n (skipping the empty board) so the
    starting board has stones."""
    archive = _build_synthetic_archive(5)
    # Drop the empty-board index so every archive draw lands on a
    # non-trivial position — makes the assertion unambiguous.
    keep = slice(1, None)
    archive = {
        "planes": archive["planes"][keep],
        "pi_mcts": archive["pi_mcts"][keep],
        "z": archive["z"][keep],
        "provenance": archive["provenance"][1:],
        "side": archive["side"][keep],
        "ply": archive["ply"][keep],
    }

    class Eval:
        @staticmethod
        def evaluate_planes(planes_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            batch = planes_arr.shape[0]
            priors = np.zeros((batch, N_ACTIONS), dtype=np.float32)
            values = np.zeros((batch,), dtype=np.float32)
            occupied = (planes_arr[:, 0] > 0.5) | (planes_arr[:, HISTORY_PLY] > 0.5)
            legal = ~occupied.reshape(batch, N_ACTIONS)
            for i in range(batch):
                if legal[i].any():
                    priors[i, legal[i]] = 1.0 / legal[i].sum()
            return priors, values

    n_games = 6
    records = _generate_games_native(
        n_games,
        Eval(),
        n_simulations=2,
        wave_size=2,
        max_plies=1,
        rng=np.random.default_rng(0),
        augment_symmetries=False,
        archive=archive,
        archive_start_frac=1.0,
    )
    assert len(records) == n_games
    assert all(r.archive_start for r in records), "every game should be archive-seeded"
    # plies count includes the archived prefix; every starting position has
    # at least one stone, so plies > 1 (1 for the archived prefix at minimum,
    # plus the 1 MCTS move played before max_plies hits).
    for r in records:
        assert r.plies >= 1


@native_only
def test_generate_games_native_skips_archive_when_frac_is_zero():
    archive = _build_synthetic_archive(5)

    class Eval:
        @staticmethod
        def evaluate_planes(planes_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            batch = planes_arr.shape[0]
            priors = np.full((batch, N_ACTIONS), 1.0 / N_ACTIONS, dtype=np.float32)
            values = np.zeros((batch,), dtype=np.float32)
            return priors, values

    records = _generate_games_native(
        4,
        Eval(),
        n_simulations=2,
        wave_size=2,
        max_plies=1,
        rng=np.random.default_rng(0),
        augment_symmetries=False,
        archive=archive,
        archive_start_frac=0.0,
    )
    assert all(not r.archive_start for r in records)


# --------------------------- cell wiring ------------------------------------


def test_wl5_cell_exists_and_carries_archive_fields():
    cell = CELLS["WL5"]
    assert cell.validation_archive_path == "archives/wl5_validation_v1.pt"
    assert cell.archive_start_path == "archives/wl5_validation_v1.pt"
    assert cell.archive_start_frac == 0.15
    # WL4 levers preserved.
    assert cell.ema_tau == 0.99
    assert cell.opponent_mix_recent == 0.4
    assert cell.opponent_mix_history == 0.1
    assert cell.weights_poll_min_sec == 2.0
    assert cell.weights_poll_max_sec == 8.0
    assert cell.grad_accum_steps == 4
    assert cell.random_opening_moves == 0


def test_wl5_trainer_cmd_includes_validation_archive_flag():
    cell = CELLS["WL5"]
    dirs = cell_dirs(cell)
    cmd = trainer_cmd(cell, dirs)
    assert "--validation-archive-path" in cmd
    assert cmd[cmd.index("--validation-archive-path") + 1] == cell.validation_archive_path


def test_wl5_worker_cmd_includes_archive_start_flags():
    cell = CELLS["WL5"]
    dirs = cell_dirs(cell)
    cmd = worker_cmd(cell, dirs, worker_id="w0", seed=0)
    assert "--archive-start-path" in cmd
    assert cmd[cmd.index("--archive-start-path") + 1] == cell.archive_start_path
    assert "--archive-start-frac" in cmd
    assert cmd[cmd.index("--archive-start-frac") + 1] == str(cell.archive_start_frac)


def test_wl4_worker_cmd_omits_archive_flags():
    """Sanity: cells without archive_start_path don't get the flag."""
    cell = CELLS["WL4"]
    dirs = cell_dirs(cell)
    cmd = worker_cmd(cell, dirs, worker_id="w0", seed=0)
    assert "--archive-start-path" not in cmd
    assert "--archive-start-frac" not in cmd
