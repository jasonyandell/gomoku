"""Regression guard for the vectorized lookahead-eval helpers (lane LA1).

`_find_immediate_wins`, `_candidate_moves`, and `_score_cells` were vectorized
for ~6.3x faster alpha-beta lookahead (an Elo anchor used by eval_worker). The
speedup was proven byte-identical to the old loop logic at the time, but that
proof was an ephemeral harness. This test pins the optimized helpers against
INDEPENDENT brute-force references so a future edit can't silently shift the
anchor's move selection.
"""
from __future__ import annotations

import numpy as np

from gomoku import baselines as B
from gomoku.game import BOARD_SIZE, GameState

N = BOARD_SIZE


def _gen_positions(n: int, seed: int = 0) -> list[GameState]:
    """Deterministic spread of heuristic-vs-heuristic midgame positions."""
    rng = np.random.default_rng(seed)
    out: list[GameState] = []
    snap = {3, 5, 7, 9, 11, 14, 17, 20, 24}
    while len(out) < n:
        st = GameState.initial()
        ply = 0
        want = set(snap)
        while want and ply <= max(snap):
            done, _ = st.is_terminal()
            if done:
                break
            if ply in want:
                out.append(st)
                want.discard(ply)
            st = st.apply(int(B.heuristic_player(st, rng)))
            ply += 1
    return out[:n]


def _brute_has_five(mine_flat: np.ndarray) -> bool:
    p = mine_flat.reshape(N, N)
    for r in range(N):
        for c in range(N):
            if not p[r, c]:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if all(
                    0 <= r + dr * k < N and 0 <= c + dc * k < N and p[r + dr * k, c + dc * k]
                    for k in range(5)
                ):
                    return True
    return False


def _brute_wins(mine_flat: np.ndarray, legal: np.ndarray) -> np.ndarray:
    wins = []
    for a in sorted(int(x) for x in legal):
        m2 = mine_flat.copy()
        m2[a] = True
        if _brute_has_five(m2):
            wins.append(a)
    return np.asarray(wins, dtype=np.int32)


def _brute_candidates(state: GameState) -> np.ndarray:
    occ = (state.board[0] | state.board[1]).reshape(-1)
    occ_idx = np.flatnonzero(occ)
    if occ_idx.size == 0:
        return np.asarray([(N // 2) * N + (N // 2)], dtype=np.int32)
    near = np.zeros(N * N, dtype=bool)
    for i in occ_idx:
        r0, c0 = divmod(int(i), N)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < N and 0 <= cc < N:
                    near[rr * N + cc] = True
    cand = near & ~occ
    return np.flatnonzero(cand).astype(np.int32)


def test_find_immediate_wins_matches_bruteforce():
    for st in _gen_positions(40, seed=1):
        mine, opp = B._flat_planes(st)
        legal = st.legal_actions()
        for m in (mine, opp):
            got = B._find_immediate_wins(m, opp, legal)
            want = _brute_wins(m, legal)
            assert np.array_equal(np.sort(got), want), (got, want)


def test_candidate_moves_matches_bruteforce():
    for st in _gen_positions(40, seed=2):
        got = B._candidate_moves(st)
        want = _brute_candidates(st)
        assert np.array_equal(got, want), (got, want)


def test_score_cells_matches_score_all_moves():
    for st in _gen_positions(40, seed=3):
        mine, opp = B._flat_planes(st)
        cells = B._candidate_moves(st)
        got = B._score_cells(mine, opp, cells)
        want = B._score_all_moves(mine, opp)[cells]
        assert np.allclose(got, want), (got, want)


def test_find_immediate_wins_empty_legal():
    mine = np.zeros(N * N, dtype=bool)
    opp = np.zeros(N * N, dtype=bool)
    assert B._find_immediate_wins(mine, opp, np.asarray([], dtype=np.int32)).size == 0
