"""Unit tests for the idx-2 Rapfi mine harness (no engine needed).

Covers the reusable, correctness-critical pieces: D4-canonical hashing
(symmetry-invariant, history-consistent) and the append-only sharded store
(atomic write, resume rebuilds the dedup-set). The coordinator/worker live loop
is exercised by the real mine, not here (needs Rapfi).
"""
from __future__ import annotations

import numpy as np

from gomoku.board_config import BOARD_SIZE, N_ACTIONS
from gomoku.game import GameState
from gomoku.rapfimine.canonical import (
    canonical_key, canonical_state, transform_state,
)
from gomoku.rapfimine.store import (
    ShardWriter, count_examples, load_seen_keys,
)


def _some_state(moves) -> GameState:
    s = GameState.initial()
    for a in moves:
        s = s.apply(int(a))
    return s


def test_canonical_key_invariant_under_all_8_symmetries():
    # A mildly asymmetric position so the 8 D4 images are genuinely distinct
    # boards that must still share one canonical key.
    s = _some_state([0, 1, BOARD_SIZE + 2, 2 * BOARD_SIZE + 5])
    keys = {canonical_key(transform_state(s, sym)) for sym in range(8)}
    assert len(keys) == 1


def test_canonical_distinct_positions_distinct_keys():
    a = _some_state([BOARD_SIZE * 7 + 7, BOARD_SIZE * 7 + 8])
    b = _some_state([BOARD_SIZE * 7 + 7, BOARD_SIZE * 8 + 8])
    assert canonical_key(a) != canonical_key(b)


def test_canonical_state_idempotent_and_frame_consistent():
    s = _some_state([0, 1, BOARD_SIZE + 2, 2 * BOARD_SIZE + 5])
    c1, _ = canonical_state(s)
    c2, _ = canonical_state(c1)
    assert c1.board.tobytes() == c2.board.tobytes()
    # History carried through the same transform → planes shape intact.
    assert c1.to_planes().shape == (s.to_planes().shape)
    # Same number of stones (transform is a permutation, not a mutation).
    assert int(c1.board.sum()) == int(s.board.sum())


def test_shard_writer_roundtrip_and_resume(tmp_path):
    out = str(tmp_path / "mine")
    w = ShardWriter(out, worker_id=0, shard_size=3)
    keys = []
    for i in range(5):
        s = _some_state([i, BOARD_SIZE + i])
        k = canonical_key(s)
        keys.append(k)
        w.add(planes=np.asarray(s.to_planes(), dtype=np.float16),
              winrates={int(s.legal_actions()[0]): 0.7,
                        int(s.legal_actions()[1]): 0.3},
              key=k, side=0, ply=2)
    w.close()
    # 5 examples → one full shard (3) + one partial (2) flushed on close.
    assert count_examples(out) == 5
    seen = load_seen_keys(out)
    assert seen == set(keys)


def test_shard_payload_matches_teacher_v2_format(tmp_path):
    out = str(tmp_path / "mine")
    w = ShardWriter(out, worker_id=1, shard_size=10)
    s = _some_state([BOARD_SIZE * 7 + 7])
    a0, a1 = int(s.legal_actions()[0]), int(s.legal_actions()[1])
    w.add(planes=np.asarray(s.to_planes(), dtype=np.float16),
          winrates={a0: 0.9, a1: 0.1}, key=canonical_key(s), side=1, ply=1)
    w.close()
    import glob
    (path,) = glob.glob(str(tmp_path / "mine" / "shard_w1_*.npz"))
    with np.load(path) as z:
        assert z["planes"].dtype == np.float16
        assert z["soft_policy"].shape == (1, N_ACTIONS)
        assert int(z["teacher_version"]) == 2
        assert int(z["board_size"]) == BOARD_SIZE
        # argmax move is the higher-winrate action; soft_policy carries both.
        assert int(z["moves"][0]) == a0
        assert z["soft_policy"][0, a0] > z["soft_policy"][0, a1] > 0
