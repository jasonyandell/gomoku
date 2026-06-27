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


# --------------------------------------------------------------------------- #
# return_carriers — the load-bearing OWN stones (the `B` channel) complementing
# support's required-openings (`./p`). Issue #88.
# --------------------------------------------------------------------------- #
def _board_from(own_cells, opp_cells=()):
    b = np.zeros((2, N, N), bool)
    for c in own_cells:
        b[0, c // N, c % N] = True
    for c in opp_cells:
        b[1, c // N, c % N] = True
    return b


def test_carriers_golden_shapes():
    """On immediate-five boards, carriers == EXACTLY the stones forming the five
    and support == the required-opening cells — the literal `.BBBB.` ask of #88
    (we wanted the four B's; support returns the `.`'s; carriers returns the B's)."""
    open_four = [7 * N + 4 + i for i in range(4)]          # .BBBB. (ends = openings)
    gapped = [5 * N + 2, 5 * N + 3, 5 * N + 5, 5 * N + 6]  # BB.BB  (gap  = opening)
    batch = np.stack([_board_from(open_four), _board_from(gapped)])
    win, hit, move, supp, carr = solve_vct_mega_bb(
        batch, max_nodes=500, return_move=True, return_support=True, return_carriers=True)
    for i, oc in enumerate([open_four, gapped]):
        assert win[i] and not hit[i], f"golden board {i} not a clean win"
        s = set(cells_from_words(supp[i]))
        cr = set(cells_from_words(carr[i]))
        assert cr == set(oc), f"carriers {sorted(cr)} != stones {sorted(oc)} @board{i}"
        assert s, f"empty support @board{i}"
        assert not (s & set(oc)), f"support hit a stone @board{i}"
        assert not (s & cr), f"support/carriers overlap @board{i}"


def run_carriers_invariants(B: int = 48, seed: int = 0, max_nodes: int = 500):
    """Structural invariants for the return_carriers output:

      * return_carriers leaves (win, hit, move, support) byte-identical
      * carriers ⊆ occupied OWN stones at root (the `B` channel, not openings)
      * carriers ∩ support == ∅ (stones vs played-empties are disjoint)
      * a non-win (clean) has empty carriers
    """
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=40)
    ws, hs, ms, supp = solve_vct_mega_bb(
        st, max_nodes=max_nodes, return_move=True, return_support=True)
    wk, hk, mk, sk, carr = solve_vct_mega_bb(
        st, max_nodes=max_nodes, return_move=True, return_support=True, return_carriers=True)

    assert np.array_equal(ws, wk) and np.array_equal(hs, hk) \
        and np.array_equal(ms, mk) and np.array_equal(supp, sk), \
        "return_carriers changed (win, hit, move, support)"

    own = st[:, 0].reshape(B, -1)
    for b in range(B):
        cr = cells_from_words(carr[b])
        s = set(cells_from_words(sk[b]))
        assert all(own[b, c] for c in cr), f"carrier not an own stone @b{b}"
        assert not (set(cr) & s), f"carrier overlaps support @b{b}"
        if not (wk[b] and not hk[b]):
            assert not cr, f"clean non-win has carriers @b{b}"
    return B


def test_carriers_invariants():
    run_carriers_invariants(B=48, seed=0)


# --------------------------------------------------------------------------- #
# return_w — the OPP MIRROR of carriers: the (over-inclusive) load-bearing
# DEFENDER stones (the `W` channel).  w = opp ∩ ⋃_support COLLIN.  Issue #90.
# --------------------------------------------------------------------------- #
def test_w_golden_shapes():
    """`.BBBB.` open four with defender stones at known distances from a support
    (open-end) cell: `w` returns EXACTLY the defenders collinear within 4 of a
    support cell — the same COLLIN-within-4 domain as `carriers`, but on OPP.  The
    attacker keeps its immediate five (the defenders sit off both open ends), so
    win / support / carriers are unchanged by adding them."""
    own = [7 * N + 4 + i for i in range(4)]      # .BBBB. at row 7, cols 4..7
    # support is the two open ends, col 3 (=7N+3) and col 8 (=7N+8). Defenders:
    near = [7 * N + 9, 7 * N + 12]               # col 9 (dist 1), col 12 (dist 4) -> in w
    far = [7 * N + 13, 0]                         # col 13 (dist 5), corner         -> not in w
    batch = np.stack([_board_from(own, near + far)])
    win, hit, move, supp, carr, wch = solve_vct_mega_bb(
        batch, max_nodes=500, return_move=True, return_support=True,
        return_carriers=True, return_w=True)
    assert win[0] and not hit[0], "golden W board not a clean win"
    s = set(cells_from_words(supp[0]))
    cr = set(cells_from_words(carr[0]))
    wc = set(cells_from_words(wch[0]))
    assert cr == set(own), f"carriers {sorted(cr)} != stones {sorted(own)}"
    assert wc == set(near), f"w {sorted(wc)} != expected near defenders {sorted(near)}"
    assert not (wc & set(far)), "w included a defender beyond collinear-within-4"
    assert not (wc & s), "w overlaps support (must be disjoint: opp vs empty)"
    assert not (wc & cr), "w overlaps carriers (must be disjoint: opp vs own)"


def run_w_invariants(B: int = 48, seed: int = 0, max_nodes: int = 500):
    """Structural invariants for the return_w output:

      * return_w leaves (win, hit, move, support, carriers) byte-identical
      * w ⊆ occupied OPP stones at root (the `W` channel, the defender mirror)
      * a clean non-win has empty w
    """
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=40)
    wc, hc, mc, sc, crc = solve_vct_mega_bb(
        st, max_nodes=max_nodes, return_move=True, return_support=True,
        return_carriers=True)
    ww, hw, mw, sw, crw, wch = solve_vct_mega_bb(
        st, max_nodes=max_nodes, return_move=True, return_support=True,
        return_carriers=True, return_w=True)

    assert np.array_equal(wc, ww) and np.array_equal(hc, hw) \
        and np.array_equal(mc, mw) and np.array_equal(sc, sw) \
        and np.array_equal(crc, crw), \
        "return_w changed (win, hit, move, support, carriers)"

    opp = st[:, 1].reshape(B, -1)
    for b in range(B):
        wcells = cells_from_words(wch[b])
        assert all(opp[b, c] for c in wcells), f"w cell not an opp stone @b{b}"
        if not (ww[b] and not hw[b]):
            assert not wcells, f"clean non-win has w @b{b}"
    return B


def test_w_invariants():
    run_w_invariants(B=48, seed=0)


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
    test_carriers_golden_shapes()
    run_carriers_invariants(B=48, seed=0)
    run_carriers_invariants(B=48, seed=1)
    test_w_golden_shapes()
    run_w_invariants(B=48, seed=0)
    run_w_invariants(B=48, seed=1)
    print("FAST tier PASS")
