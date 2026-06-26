"""Validate the bitboard VCT megakernel (mega_vct_bb) on real Rapfi positions.

    GOMOKU_BOARD_SIZE=15 timeout 115 uv run python -m scripts.vct_metal.test_mega_vct_bb

Cross-checks solve_vct_mega_bb against the cell-scan mega_vct.solve_vct_mega (itself
validated vs gomoku.vcf.solve_vct) on clean (non-cap) verdicts. mega_vct is slow, so
this uses a small batch; a background run validated mega_vct_bb directly vs
gomoku.vcf.solve_vct over real positions (see TRAINING_WIKI 2026-06-26).
"""
from __future__ import annotations

import numpy as np

from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb
from scripts.vct_metal.mega_vct import solve_vct_mega
from scripts.vct_metal.positions import load_position_stack


def run(B: int = 16, seed: int = 0, max_nodes: int = 600):
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=40)
    wb, hb = solve_vct_mega_bb(st, max_nodes=max_nodes)
    wo, ho = solve_vct_mega(st, max_nodes=max_nodes)
    clean = ~(hb | ho)
    disagree = np.where(clean & (wb != wo))[0]
    print(f"seed={seed} B={B} clean={int(clean.sum())}/{B} "
          f"bb_wins={int(wb.sum())} old_wins={int(wo.sum())} "
          f"bb_cap={int(hb.sum())} old_cap={int(ho.sum())} disagree={list(disagree)}")
    return list(disagree)


def test_mega_vct_bb_matches_mega_vct():
    bad = run(B=16, seed=0)
    assert not bad


if __name__ == "__main__":
    import sys
    bad = 0
    for s in (0, 1):
        bad += len(run(B=16, seed=s))
    print("PASS" if not bad else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)
