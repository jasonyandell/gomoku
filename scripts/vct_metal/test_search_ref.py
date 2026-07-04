"""Validate solve_vct_ref verdict against gomoku.vcf.solve_vct on CLEAN cases.

    GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_metal.test_search_ref

Clean = neither solver hit its cap. There the AND/OR proof value is
order-independent, so verdicts must agree. Cap-boundary disagreements are
expected (and tallied), never counted as failures -- matching the v0 spike.
"""
from __future__ import annotations

import sys
import time
import numpy as np

from gomoku import vcf
from scripts.vct_metal import search_ref as S
from scripts.vct_metal.test_detect_ref import gen_clustered

MAX_DEPTH = 6
MAX_NODES = 1500


def run(B: int = 60, seed: int = 0):
    own, opp = gen_clustered(B, seed, lo=8, hi=28)
    clean_agree = clean_disagree = cap = 0
    wins = 0
    fails = []
    t = time.time()
    for b in range(B):
        board = np.stack([own[b], opp[b]])
        mw, mh = S.solve_vct_ref(board, max_depth=MAX_DEPTH, max_nodes=MAX_NODES)
        print(".", end="", flush=True)
        v = vcf.solve_vct(board, max_depth=MAX_DEPTH, max_nodes=MAX_NODES)
        if mw == v.has_forced_win:
            if not (mh or v.hit_cap):
                clean_agree += 1
        else:
            if mh or v.hit_cap:
                cap += 1
            else:
                clean_disagree += 1
                fails.append((b, mw, v.has_forced_win))
        wins += int(v.has_forced_win)
    dt = time.time() - t

    print(f"N={S.N} boards={B} seed={seed} caps(d={MAX_DEPTH},n={MAX_NODES}) "
          f"{dt:.1f}s ({dt/B*1000:.0f} ms/board)")
    print(f"  vcf wins={wins}  clean_agree={clean_agree}  "
          f"cap_boundary={cap}  clean_DISAGREE={clean_disagree}")
    if fails:
        print(f"  REAL FAILURES (clean disagreement), first 8: {fails[:8]}")
    else:
        print("  PASS  no clean disagreement with gomoku.vcf.solve_vct")
    return fails


def test_search_ref_matches_vcf():
    assert run(B=80, seed=1) == []


if __name__ == "__main__":
    bad = []
    for s in (0, 1, 2):
        bad += run(B=60, seed=s)
        print()
    raise SystemExit(1 if bad else 0)
