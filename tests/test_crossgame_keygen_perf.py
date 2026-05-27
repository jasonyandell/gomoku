"""Cross-game keygen perf fix (bead derby-eda): ply-gate + vectorized keygen.

ALL CPU, bounded — never touches MPS / the live derby. Covers:
  - PROPERTY TEST: the vectorized ``canonical_key_from_board`` is BYTE-IDENTICAL
    to the scalar reference ``canonical_key_from_board_scalar`` over thousands of
    random boards AND all 8 D4 symmetries of random boards (a wrong key silently
    corrupts value targets, so this is the load-bearing correctness guarantee).
  - REGRESSION: per-position ingest cost stays ~flat as INFLOW grows 10x (the
    keygen no longer dominates / scales with flooding); the keygen is NOT called
    on ply>=max_ply positions (it is paid ONLY for positions that get stored).

The CPU benchmark here is necessary-but-NOT-sufficient: it cannot reproduce the
live MPS flooding regime. The derby runner's full-load (epoch 50+) re-race is the
real gate; this just pins correctness + the scaling shape.
"""

from __future__ import annotations

import time

import numpy as np

from gomoku.game import BOARD_SIZE, N_INPUT_PLANES
from gomoku.position_stats import (
    PositionStats,
    canonical_key_from_board,
    canonical_key_from_board_scalar,
    canonical_key_from_planes,
)

_N_CELLS = BOARD_SIZE * BOARD_SIZE
_OPP_PLANE = N_INPUT_PLANES // 2


# ---------- helpers ----------

def _random_board(rng: np.random.Generator) -> np.ndarray:
    """A random legal-ish (2, N, N) bool occupancy: disjoint mine/opp stones,
    a random number of stones (0..81), alternating ownership."""
    n = int(rng.integers(0, _N_CELLS + 1))
    cells = rng.choice(_N_CELLS, size=n, replace=False)
    mine = np.zeros(_N_CELLS, dtype=bool)
    opp = np.zeros(_N_CELLS, dtype=bool)
    for i, c in enumerate(cells):
        if i % 2 == 0:
            mine[c] = True
        else:
            opp[c] = True
    return np.stack([mine.reshape(BOARD_SIZE, BOARD_SIZE),
                     opp.reshape(BOARD_SIZE, BOARD_SIZE)])


def _sym_board2(board2: np.ndarray, s: int) -> np.ndarray:
    """The same D4 transform the keygen uses (rot then optional flip)."""
    rot = s % 4
    flip = s // 4
    out = np.rot90(board2, rot, axes=(-2, -1))
    if flip:
        out = np.flip(out, axis=-1)
    return out.copy()


def _planes_from_board2(board2: np.ndarray) -> np.ndarray:
    planes = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    planes[0] = board2[0].astype(np.float32)
    planes[_OPP_PLANE] = board2[1].astype(np.float32)
    return planes


# ---------- PROPERTY TEST: vectorized == scalar reference, byte-for-byte ----------

def test_vectorized_key_equals_scalar_reference_random_boards():
    """Over thousands of random boards, the vectorized canonical key is
    BYTE-IDENTICAL (tuple-equal) to the scalar reference. A single mismatch
    silently corrupts value targets, so this must be exact."""
    rng = np.random.default_rng(12345)
    for _ in range(4000):
        b = _random_board(rng)
        assert canonical_key_from_board(b) == canonical_key_from_board_scalar(b)


def test_vectorized_key_equals_scalar_reference_all_symmetries():
    """For random boards, EVERY one of the 8 D4 orientations must produce a key
    that (a) equals the scalar reference for that orientation, and (b) equals the
    base orientation's key (D4-invariance holds in BOTH implementations)."""
    rng = np.random.default_rng(999)
    for _ in range(1500):
        b = _random_board(rng)
        base_vec = canonical_key_from_board(b)
        base_ref = canonical_key_from_board_scalar(b)
        assert base_vec == base_ref
        for s in range(8):
            t = _sym_board2(b, s)
            assert canonical_key_from_board(t) == base_vec, f"vec broke sym {s}"
            assert canonical_key_from_board_scalar(t) == base_ref, f"ref broke sym {s}"


def test_vectorized_key_from_planes_matches_board_path():
    """The planes entry-point composes the same canonical key as the board path
    (and equals the scalar reference)."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        b = _random_board(rng)
        planes = _planes_from_board2(b)
        assert canonical_key_from_planes(planes) == canonical_key_from_board_scalar(b)


def test_empty_board_key_is_zero_triple():
    empty = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=bool)
    assert canonical_key_from_board(empty) == (0, 0, 0)
    assert canonical_key_from_board_scalar(empty) == (0, 0, 0)


# ---------- REGRESSION: ply-gate the keygen + flood-scaling stays flat ----------

class _Ex:
    __slots__ = ("planes", "z", "ply")

    def __init__(self, planes, z, ply):
        self.planes = planes
        self.z = z
        self.ply = ply


def _flood(rng, n, max_ply_seen=80):
    exs = []
    for _ in range(n):
        ply = int(rng.integers(0, max_ply_seen))
        cells = rng.choice(_N_CELLS, size=min(ply, _N_CELLS), replace=False)
        planes = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for i, c in enumerate(cells):
            r, col = divmod(int(c), BOARD_SIZE)
            planes[0 if i % 2 == 0 else _OPP_PLANE, r, col] = 1.0
        exs.append(_Ex(planes, float(rng.uniform(-1, 1)), ply))
    return exs


def _ingest_ply_gated(ps: PositionStats, exs, counter=None):
    """The train.py post-fix ingest path: ply-gate FIRST, keygen only for
    positions that will be stored. ``counter`` (optional list[int]) records the
    number of keygen calls actually made."""
    ps.decay()
    for e in exs:
        ply = e.ply
        if not ps._ply_in_cap(ply):
            continue
        if counter is not None:
            counter[0] += 1
        ps.add(canonical_key_from_planes(e.planes), e.z, ply=ply)


def test_keygen_not_called_on_capped_positions():
    """The keygen is paid ONLY for ply<max_ply positions. With a max_ply=10 cap
    and a flood that is mostly mid/late game, keygen calls == the opening count,
    far below the total inflow."""
    rng = np.random.default_rng(3)
    n = 4000
    exs = _flood(rng, n)
    n_opening = sum(1 for e in exs if e.ply < 10)
    assert 0 < n_opening < n  # sanity: the flood really is mixed-ply
    counter = [0]
    _ingest_ply_gated(PositionStats(max_ply=10), exs, counter=counter)
    assert counter[0] == n_opening, (
        f"keygen called {counter[0]}x, expected {n_opening} (opening-only)")


def test_ingest_cost_stays_flat_as_inflow_grows():
    """FLOOD-SCALING regression (bead derby-eda): 10x the inflow must NOT ~10x
    the per-position wall. The old keygen-everything path made the per-epoch wall
    grow with the store/inflow; the ply-gate + vectorized keygen makes per-epoch
    ingest ~linear in inflow with a small, ~flat per-position constant.

    We assert the per-position cost at 10x inflow is no worse than ~2.5x the 1x
    per-position cost (generous slack for CPU noise; the runaway was ~10x+)."""
    base_n = 3000

    def per_pos_us(mult):
        exs = _flood(np.random.default_rng(mult), base_n * mult)
        # warm once (import/JIT-ish numpy paths), then time
        _ingest_ply_gated(PositionStats(max_ply=10), exs)
        best = min(
            _timed(lambda: _ingest_ply_gated(PositionStats(max_ply=10), exs))
            for _ in range(3)
        )
        return 1e6 * best / (base_n * mult)

    p1 = per_pos_us(1)
    p10 = per_pos_us(10)
    assert p10 <= 2.5 * p1, (
        f"per-pos cost did not stay flat: 1x={p1:.2f}us 10x={p10:.2f}us "
        f"(ratio {p10/p1:.2f})")


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
