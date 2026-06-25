"""Unit tests for the DAgger aggregate loader (no engine / no GPU needed).

The full gather→label→train→gate loop needs Rapfi + MPS and is exercised by a
live smoke round; here we cover the new pure-data logic: ``_load_dirs`` unions
the dagger store with a capped random sample of the mine (the D ∪ Dᵢ retention
mix), reading the same teacher-v2 shard schema the mine writes.
"""
from __future__ import annotations

import numpy as np

from gomoku.board_config import BOARD_SIZE, N_ACTIONS
from gomoku.game import GameState
from gomoku.rapfimine.canonical import canonical_key
from gomoku.rapfimine.dagger import _load_dirs
from gomoku.rapfimine.store import ShardWriter


def _state(moves) -> GameState:
    s = GameState.initial()
    for a in moves:
        s = s.apply(int(a))
    return s


def _write_store(path, *, worker_id, n, shard_size):
    w = ShardWriter(path, worker_id=worker_id, shard_size=shard_size)
    for i in range(n):
        s = _state([i, BOARD_SIZE + i, 2 * BOARD_SIZE + (i % 4)])
        la = [int(a) for a in s.legal_actions()[:3]]
        w.add(planes=np.asarray(s.to_planes(), dtype=np.float16),
              winrates={la[0]: 0.7, la[1]: 0.2, la[2]: 0.1},
              key=canonical_key(s), side=int(s.move_count % 2), ply=s.move_count)
    w.close()


def test_load_dirs_dagger_only(tmp_path):
    dag = str(tmp_path / "dag")
    _write_store(dag, worker_id=0, n=6, shard_size=4)  # 2 shards
    planes, soft = _load_dirs(dag, None, 0, seed=0, on_log=lambda *_: None)
    assert planes.shape[0] == 6
    assert soft.shape == (6, N_ACTIONS)
    assert planes.dtype == np.float16
    # soft rows carry the scored support (> 0 at the labelled actions).
    assert (soft > 0).any()


def test_load_dirs_unions_mine_sample(tmp_path):
    dag = str(tmp_path / "dag")
    mine = str(tmp_path / "mine")
    _write_store(dag, worker_id=0, n=5, shard_size=10)          # 1 dagger shard, 5 rows
    for wid in range(4):                                         # 4 mine shards, 3 rows each
        _write_store(mine, worker_id=wid, n=3, shard_size=10)
    # Sample 2 of the 4 mine shards → 5 dagger + 2*3 mine = 11 rows.
    planes, _ = _load_dirs(dag, mine, 2, seed=0, on_log=lambda *_: None)
    assert planes.shape[0] == 5 + 2 * 3
    # Sampling is bounded by available shards: asking for more than exist uses all.
    planes_all, _ = _load_dirs(dag, mine, 99, seed=0, on_log=lambda *_: None)
    assert planes_all.shape[0] == 5 + 4 * 3
    # 0 mine shards → dagger only.
    planes_dag, _ = _load_dirs(dag, mine, 0, seed=0, on_log=lambda *_: None)
    assert planes_dag.shape[0] == 5
