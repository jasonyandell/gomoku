"""Validate the bitboard VCT megakernel (mega_vct_bb) on real Rapfi positions.

    GOMOKU_BOARD_SIZE=15 timeout 115 uv run python -m scripts.vct_metal.test_mega_vct_bb

Cross-checks solve_vct_mega_bb against the cell-scan mega_vct.solve_vct_mega (itself
validated vs gomoku.vcf.solve_vct) on clean (non-cap) verdicts. mega_vct is slow, so
this uses a small batch; a background run validated mega_vct_bb directly vs
gomoku.vcf.solve_vct over real positions (see TRAINING_WIKI 2026-06-26).
"""
from __future__ import annotations

import numpy as np

from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb, cells_from_words, N
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


def run_support_complete(B: int = 32, seed: int = 0, max_nodes: int = 500):
    """Invariants for the return_support / complete outputs (fast, no per-move
    gold — that is covered by tmp/gold_complete.py / TRAINING_WIKI 2026-06-27):

      * return_support leaves (win, hit, move) byte-identical to default
      * complete `win` == default `win` on boards neither solve capped
      * the default move (on a clean win) is a member of complete's winmask
      * support cells are EMPTY at root, contain the move on a win, empty on loss
    """
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=40)
    w0, h0, m0 = solve_vct_mega_bb(st, max_nodes=max_nodes, return_move=True)
    ws, hs, ms, supp = solve_vct_mega_bb(
        st, max_nodes=max_nodes, return_move=True, return_support=True)
    wc, hc, mc, winmask = solve_vct_mega_bb(
        st, max_nodes=max_nodes, return_move=True, complete=True)

    assert np.array_equal(w0, ws) and np.array_equal(h0, hs) \
        and np.array_equal(m0, ms), "return_support changed (win, hit, move)"
    clean = ~h0 & ~hc
    assert np.array_equal(w0[clean], wc[clean]), "complete win != default win"

    occ = (st[:, 0] | st[:, 1]).reshape(B, -1)
    for b in range(B):
        cells = cells_from_words(supp[b])
        assert all(not occ[b, c] for c in cells), f"support hit occupied cell @b{b}"
        if w0[b] and not h0[b]:
            assert cells and int(m0[b]) in cells, f"move not in support @b{b}"
            if not hc[b]:
                assert int(m0[b]) in set(cells_from_words(winmask[b])), \
                    f"default move not in winmask @b{b}"
        elif not w0[b] and not h0[b]:
            assert not cells, f"non-win has non-empty support @b{b}"
    return B


def test_support_and_complete_invariants():
    run_support_complete(B=32, seed=0)


if __name__ == "__main__":
    import sys
    bad = 0
    for s in (0, 1):
        bad += len(run(B=16, seed=s))
    run_support_complete(B=32, seed=0)
    run_support_complete(B=32, seed=1)
    print("support/complete invariants PASS")
    print("PASS" if not bad else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)
