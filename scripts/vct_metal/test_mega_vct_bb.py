"""FAST-tier validation for the bitboard VCT megakernel (mega_vct_bb).

    GOMOKU_BOARD_SIZE=15 uv run pytest scripts/vct_metal/test_mega_vct_bb.py

This tier touches NEITHER the retired CPU solver NOR the slow cell-scan
``mega_vct``. It diffs ``solve_vct_mega_bb`` against a small COMMITTED golden
fixture (``fixtures/vct_golden.npz``, regenerated on-demand by
``regen_vct_fixture.py`` — the only place the CPU oracle is invoked), plus a
self-oracle structural-invariants test on the support/complete outputs.

Tiering: the live-vcf verdict check, the winmask soundness+completeness GOLD
check, and the larger-n sweep all live in the DEEP tier
``scripts/vct_metal/validate_deep.py`` (run on-demand, not in the gate). See
wiki/topics/mega-vct-solver.md § CPU solver retired.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.vct_metal.mega_vct_bb import cells_from_words, solve_vct_mega_bb, N
from scripts.vct_metal.positions import load_position_stack

FIXTURE = Path(__file__).parent / "fixtures" / "vct_golden.npz"
# Tight budget: a non-capped result at this budget is definitive and MUST match
# the high-budget golden truth (non-capped verdicts are budget-independent).
FAST_MAX_NODES = 500


@pytest.fixture(scope="module")
def golden():
    if not FIXTURE.exists():
        pytest.skip(f"missing {FIXTURE}; regenerate via regen_vct_fixture.py")
    data = np.load(FIXTURE)
    if int(data["board_size"]) != N:
        pytest.skip(
            f"fixture board_size={int(data['board_size'])} != active N={N} "
            "(run with GOMOKU_BOARD_SIZE=15)"
        )
    return data


def test_verdict_matches_golden(golden):
    """GPU verdict at the tight budget matches the CPU-oracle golden truth on
    every board the GPU does not cap (hit_cap -> skip)."""
    boards = golden["boards"]
    truth = golden["win"].astype(bool)
    wg, hg = solve_vct_mega_bb(boards, max_nodes=FAST_MAX_NODES)
    clean = ~hg
    assert clean.any(), "every fixture board capped at the fast budget — fixture too hard"
    fp = np.where(wg & ~truth & clean)[0]
    fn = np.where(~wg & truth & clean)[0]
    assert fp.size == 0, f"false positives vs golden on boards {list(fp)}"
    assert fn.size == 0, f"false negatives vs golden on boards {list(fn)}"


def test_winmask_matches_golden(golden):
    """The kernel's `complete` winmask at the tight budget matches the stored
    high-budget winmask on every board the GPU does not cap."""
    boards = golden["boards"]
    gold_wm = golden["winmask"].astype(np.uint64)
    wc, hc, winmask = solve_vct_mega_bb(boards, max_nodes=FAST_MAX_NODES, complete=True)
    clean = ~hc
    bad = []
    for b in np.where(clean)[0]:
        if set(cells_from_words(winmask[b])) != set(cells_from_words(gold_wm[b])):
            bad.append(int(b))
    assert not bad, f"winmask diverged from golden on boards {bad}"


def run_support_complete(B: int = 32, seed: int = 0, max_nodes: int = 500):
    """Self-oracle structural invariants for the return_support / complete outputs
    (fast; no per-move gold — that is the DEEP tier validate_deep.py):

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
    if not FIXTURE.exists():
        print(f"FAIL: missing {FIXTURE} (run regen_vct_fixture.py)")
        sys.exit(1)
    data = np.load(FIXTURE)
    test_verdict_matches_golden(data)
    test_winmask_matches_golden(data)
    run_support_complete(B=32, seed=0)
    run_support_complete(B=32, seed=1)
    print("FAST tier PASS")
