"""STEP-1 profile: where does per-epoch cross-game ingest time go under FLOODING?

Simulates a high-inflow ingest cycle: tens of thousands of FRESH positions per
epoch with a realistic mixed-ply distribution (NOT a tiny 60-opening toy), then
profiles keygen vs add vs decay vs the full train.py ingest path.

CPU-only, bounded. Run:
    uv run python scripts/bench_crossgame_ingest.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np

from gomoku.game import BOARD_SIZE, GameState, N_INPUT_PLANES
from gomoku.position_stats import (
    PositionStats,
    canonical_key_from_board_scalar,
    canonical_key_from_planes,
)


def canonical_key_from_planes_scalar(planes):
    """Pre-fix keygen: the SCALAR reference path (what shipped before derby-eda
    vectorization), so the benchmark can attribute the speedup honestly."""
    board2 = np.stack([planes[0] > 0.5, planes[_OPP_PLANE] > 0.5])
    return canonical_key_from_board_scalar(board2)

_N_CELLS = BOARD_SIZE * BOARD_SIZE
_OPP_PLANE = N_INPUT_PLANES // 2


def random_planes(rng: np.random.Generator, n_stones: int) -> np.ndarray:
    """A plausible side-relative input stack with `n_stones` placed (alternating).
    Only plane 0 (mine) and plane _OPP_PLANE (opp) carry occupancy — that's all
    the keygen reads. History planes left zero (keygen ignores them)."""
    planes = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    cells = rng.choice(_N_CELLS, size=n_stones, replace=False)
    # alternate ownership: even index -> mine, odd -> opp
    for i, c in enumerate(cells):
        r, col = divmod(int(c), BOARD_SIZE)
        if i % 2 == 0:
            planes[0, r, col] = 1.0
        else:
            planes[_OPP_PLANE, r, col] = 1.0
    return planes


class _Example:
    __slots__ = ("planes", "z", "ply")

    def __init__(self, planes, z, ply):
        self.planes = planes
        self.z = z
        self.ply = ply


def build_flood(rng, n_positions, max_ply_seen=80):
    """A flood cycle: n_positions fresh examples with a realistic mixed-ply
    distribution. Self-play games are ~20-50 plies; most positions are mid/late
    game (ply >= max_ply gate of 10), only a small opening fraction is < 10."""
    examples = []
    for _ in range(n_positions):
        # mixed plies: skew toward mid/late game like real self-play
        ply = int(rng.integers(0, max_ply_seen))
        n_stones = min(ply, _N_CELLS)
        planes = random_planes(rng, max(n_stones, 0))
        z = float(rng.uniform(-1.0, 1.0))
        examples.append(_Example(planes, z, ply))
    return examples


def ingest_old(ps: PositionStats, examples):
    """PRE-FIX train.py: keygen for EVERY example, ply gate inside add()."""
    ps.decay()
    for e in examples:
        ps.add(canonical_key_from_planes(e.planes), e.z, ply=getattr(e, "ply", 0))


def ingest_new(ps: PositionStats, examples):
    """POST-FIX train.py: ply-gate FIRST (cheap int check), keygen only for
    positions that will be stored. Keygen itself is now vectorized."""
    ps.decay()
    for e in examples:
        ply = getattr(e, "ply", 0)
        if not ps._ply_in_cap(ply):
            continue
        ps.add(canonical_key_from_planes(e.planes), e.z, ply=ply)


# alias kept for the cProfile section below
ingest_current = ingest_old


def main():
    rng = np.random.default_rng(0)
    N = 20000  # tens of thousands of fresh positions per epoch (flood)
    print(f"Building flood of {N} mixed-ply positions (max_ply gate=10)...")
    examples = build_flood(rng, N)
    n_opening = sum(1 for e in examples if e.ply < 10)
    print(f"  opening (ply<10): {n_opening}  ({100*n_opening/N:.1f}%)   "
          f"mid/late (ply>=10): {N - n_opening}")

    # --- coarse timing: SCALAR keygen (pre-fix) so keygen dominance is clear ---
    t0 = time.perf_counter()
    keys = [canonical_key_from_planes_scalar(e.planes) for e in examples]
    t_keygen_scalar = time.perf_counter() - t0

    # vectorized keygen (post-fix) over the SAME positions
    t0 = time.perf_counter()
    keys_v = [canonical_key_from_planes(e.planes) for e in examples]
    t_keygen_vec = time.perf_counter() - t0
    assert keys == keys_v, "vectorized keygen disagrees with scalar reference!"

    ps2 = PositionStats(max_ply=10)
    t0 = time.perf_counter()
    ps2.decay()
    for k, e in zip(keys, examples):
        ps2.add(k, e.z, ply=e.ply)
    t_add = time.perf_counter() - t0

    ps3 = PositionStats(max_ply=10)
    t0 = time.perf_counter()
    ps3.decay()
    t_decay = time.perf_counter() - t0

    print("\n=== COARSE TIMING (one flood epoch) ===")
    print(f"  scalar keygen (all {N}):  {t_keygen_scalar*1000:8.1f} ms   "
          f"({1e6*t_keygen_scalar/N:6.1f} us/pos)")
    print(f"  vector keygen (all {N}):  {t_keygen_vec*1000:8.1f} ms   "
          f"({1e6*t_keygen_vec/N:6.1f} us/pos)")
    print(f"  add (after keys ready):   {t_add*1000:8.1f} ms")
    print(f"  decay (O(1)):             {t_decay*1e6:8.1f} us")
    print(f"  -> scalar keygen share of (keygen+add+decay): "
          f"{100*t_keygen_scalar/(t_keygen_scalar+t_add+t_decay):5.1f}%")

    # --- BEFORE/AFTER the FULL fix at this inflow ---
    # Build a TRUE pre-fix ingest (scalar keygen, no ply-gate before keygen).
    def ingest_prefix(ps, exs):
        ps.decay()
        for e in exs:
            ps.add(canonical_key_from_planes_scalar(e.planes), e.z,
                   ply=getattr(e, "ply", 0))

    t0 = time.perf_counter()
    ingest_prefix(PositionStats(max_ply=10), examples)
    t_pre = time.perf_counter() - t0
    t0 = time.perf_counter()
    ingest_new(PositionStats(max_ply=10), examples)
    t_new = time.perf_counter() - t0
    print(f"\n=== BEFORE / AFTER (one flood epoch, {N} positions) ===")
    print(f"  PRE-FIX  (scalar keygen, no ply-gate):  {t_pre*1000:8.1f} ms")
    print(f"  POST-FIX (ply-gate + vectorized):       {t_new*1000:8.1f} ms")
    print(f"  total speedup:                          {t_pre/t_new:6.1f}x")

    # --- FLOOD-SCALING: does cost stay ~flat as INFLOW grows 10x? ---
    print("\n=== FLOOD SCALING (post-fix ingest; cost vs inflow) ===")
    base = None
    for mult in (1, 2, 5, 10):
        exs = build_flood(np.random.default_rng(mult), N * mult)
        t0 = time.perf_counter()
        ingest_new(PositionStats(max_ply=10), exs)
        t = time.perf_counter() - t0
        per = 1e6 * t / (N * mult)
        if base is None:
            base = per
        print(f"  inflow {N*mult:>7} pos:  {t*1000:8.1f} ms   "
              f"({per:6.2f} us/pos, {per/base:4.2f}x per-pos vs 1x)")

    # --- cProfile of the PRE-FIX path (proves keygen dominated) ---
    print("\n=== cProfile (PRE-FIX ingest path: scalar keygen, no ply-gate) ===")
    pr = cProfile.Profile()
    pr.enable()
    ingest_prefix(PositionStats(max_ply=10), examples)
    pr.disable()
    s = io.StringIO()
    ps_stats = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps_stats.print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    main()
