"""Validate the bitboard VCF megakernel (mega_vcf_bb) on real Rapfi positions.

    GOMOKU_BOARD_SIZE=15 timeout 115 uv run python -m scripts.vct_metal.test_mega_vcf_bb

Cross-checks solve_vcf_mega_bb against the already-oracle-validated cell-scan
mega_vcf.solve_vcf_mega on clean (non-cap) cases — a fast proxy for the gomoku.vcf
oracle (mega_vcf is validated 0 FP/FN vs vcf). A separate background run validated
mega_vcf_bb directly vs gomoku.vcf.solve_vcf: 0 FP / 0 FN over 360 real positions.
"""
from __future__ import annotations

import numpy as np

from scripts.vct_metal.mega_vcf_bb import solve_vcf_mega_bb
from scripts.vct_metal.mega_vcf import solve_vcf_mega
from scripts.vct_metal.positions import load_position_stack


def run(B: int = 60, seed: int = 0, max_nodes: int = 20000):
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=60)
    wb, hb = solve_vcf_mega_bb(st, max_nodes=max_nodes)
    wo, ho = solve_vcf_mega(st, max_nodes=max_nodes)
    clean = ~(hb | ho)
    disagree = np.where(clean & (wb != wo))[0]
    print(f"seed={seed} B={B} clean={int(clean.sum())}/{B} "
          f"bb_wins={int(wb.sum())} old_wins={int(wo.sum())} disagree={list(disagree)}")
    return list(disagree)


def test_mega_vcf_bb_matches_mega_vcf():
    bad = run(B=60, seed=1)
    assert not bad


if __name__ == "__main__":
    import sys
    bad = 0
    for s in (0, 1, 2):
        bad += len(run(B=80, seed=s))
    print("PASS" if not bad else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)
