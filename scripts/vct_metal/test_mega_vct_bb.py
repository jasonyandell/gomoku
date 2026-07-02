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

from scripts.vct_metal.mega_vct_bb import (
    MAXD, cells_from_words, solve_md_min, solve_vct_mega_bb, solve_vct_streaming,
    N, _build_src, _src)
from scripts.vct_metal.positions import load_position_stack

FIXTURE = Path(__file__).parent / "fixtures" / "vct_golden.npz"
MD_FIXTURE = Path(__file__).parent / "fixtures" / "vct_md_golden.npz"
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


# --------------------------------------------------------------------------- #
# depth_cap / max_depth — the per-board mate-distance variant (issue #91). md_min
# is GPU-self-validated (order-independent depth-capped binary search), NO CPU
# oracle.  See wiki/topics/mega-vct-solver.md `max_depth` / invariant #9.
# --------------------------------------------------------------------------- #
def _inject_depth_cap(src):
    """The two strings _build_src injects for depth_cap — mirrored here so the test
    fails loudly if the kernel injection drifts (a silent-wrong-answer guard)."""
    src = src.replace(
        "    int maxnodes=max_nodes[0]; int nodes=0; bool hitcap=false;",
        "    int maxnodes=max_nodes[0]; int nodes=0; bool hitcap=false;"
        " int maxd=max_depth[gid];", 1)
    src = src.replace(
        "        if (typ[sp]==0){ // ---- OR node (attacker) ----",
        "        if (sp >= maxd){ ret=0; entering=false; continue; }\n"
        "        if (typ[sp]==0){ // ---- OR node (attacker) ----", 1)
    return src


def test_depth_cap_byte_identical_and_composes():
    """The depth_cap flag is purely additive: for every (support,complete,carriers,w)
    the depth_cap=False source equals the default-arg source, and depth_cap=True adds
    EXACTLY the two injected strings — so no existing compiled variant changes
    (invariant #9)."""
    combos = [(s, c, cr, w) for s in (0, 1) for c in (0, 1)
              for cr in (0, 1) for w in (0, 1)]
    for s, c, cr, w in combos:
        base = _build_src(bool(s), bool(c), bool(cr), bool(w))
        assert _build_src(bool(s), bool(c), bool(cr), bool(w), depth_cap=False) == base, \
            f"depth_cap=False changed source @ {(s, c, cr, w)}"
        assert _build_src(bool(s), bool(c), bool(cr), bool(w), depth_cap=True) \
            == _inject_depth_cap(base), f"depth_cap injection drifted @ {(s, c, cr, w)}"
    assert _build_src(False, False, False, False, True) == _inject_depth_cap(_src())


def test_depth_cap_ceiling_equals_default(B: int = 48, seed: int = 0):
    """max_depth=MAXD-1 reproduces the default (win,hit,move): at sp=MAXD-1 the
    structural cap fires first, so the depth cut is provably dead code there."""
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=40)
    w0, h0, m0 = solve_vct_mega_bb(st, max_nodes=500, return_move=True)
    wc, hc, mc = solve_vct_mega_bb(st, max_nodes=500, return_move=True, max_depth=MAXD - 1)
    assert np.array_equal(w0, wc) and np.array_equal(h0, hc) and np.array_equal(m0, mc), \
        "max_depth=MAXD-1 != default (win,hit,move)"


def test_depth_monotonic(B: int = 48, seed: int = 0, dmax: int = 12):
    """win(d) is monotone non-decreasing in d on boards clean at both d and d+1
    (the single most important md-correctness gate — a True->False flip is a kernel
    bug, e.g. hitcap leaking into the depth cut)."""
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=40)
    prev = None
    for d in range(1, dmax + 1):
        wd, hd = solve_vct_mega_bb(st, max_nodes=500, max_depth=d)
        if prev is not None:
            wp, hp = prev
            clean = (~hp) & (~hd)
            assert not (wp & ~wd & clean).any(), f"win({d-1})&~win({d}) on a clean board"
        prev = (wd, hd)


def test_md_min_bracket(B: int = 48, seed: int = 0):
    """solve_md_min defines the mate distance: win@md is a clean win, win@(md-1) is a
    clean no-win, md>=1 iff the board is a (clean) default win."""
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=40)
    w0, h0 = solve_vct_mega_bb(st, max_nodes=500)
    md, capped = solve_md_min(st, max_nodes=500, hi=MAXD - 2)
    alive = md >= 1
    # md>=1 <=> default clean win, on boards neither solve withheld
    ok = (~capped) & (~h0)
    assert np.array_equal(alive[ok], w0[ok]), "md>=1 != default win on clean boards"
    # bracket via per-board probes
    wa, ha = solve_vct_mega_bb(st, max_nodes=500, max_depth=np.where(alive, md, 1))
    assert all(wa[b] and not ha[b] for b in np.where(alive)[0]), "win@md not clean"
    a2 = alive & (md >= 2)
    wl, hl = solve_vct_mega_bb(st, max_nodes=500, max_depth=np.where(a2, md - 1, 1))
    assert all((not wl[b]) and (not hl[b]) for b in np.where(a2)[0]), \
        "win@(md-1) not a clean no-win"


@pytest.fixture(scope="module")
def md_golden():
    if not MD_FIXTURE.exists():
        pytest.skip(f"missing {MD_FIXTURE}; regenerate via regen_vct_md_fixture.py")
    data = np.load(MD_FIXTURE)
    if int(data["board_size"]) != N:
        pytest.skip(f"md fixture board_size={int(data['board_size'])} != N={N}")
    return data


def test_md_matches_golden(md_golden):
    """Tight-budget md_min matches the high-budget golden md on every board the
    tight search does not cap (a non-capped md is budget-independent)."""
    boards = md_golden["boards"]
    gold_md = md_golden["md"].astype(np.int32)
    md, capped = solve_md_min(boards, max_nodes=500, hi=MAXD - 2)
    ok = (~capped) & (gold_md >= 1)            # compare on clean wins with known md
    bad = [(int(b), int(md[b]), int(gold_md[b])) for b in np.where(ok)[0] if md[b] != gold_md[b]]
    assert not bad, f"md != golden on {bad[:8]}"


# --------------------------------------------------------------------------- #
# work_steal (Option A, issue #93) — the persistent work-stealing dispatch must
# leave the (win, hit, move) verdict BYTE-IDENTICAL to the base kernel (same
# per-board search, same max_nodes); only the dispatch shape differs. And
# solve_vct_streaming (Option B) must agree with the base verdict at its deepest
# budget on every mutually-clean board.
# --------------------------------------------------------------------------- #
def test_work_steal_source_untouched_default():
    """The default (non-work_steal) source is unchanged, and work_steal injects
    exactly the cursor-pull loop + atomic output stores."""
    assert _build_src(False, False) == _src(), "default source drifted"
    ws = _build_src(False, False, work_steal=True)
    assert "atomic_fetch_add_explicit(&cursor[0]" in ws
    assert "if (gid >= (uint)nboards[0]) break;" in ws
    assert "atomic_store_explicit(&win[gid]" in ws
    assert "thread_position_in_grid.x" not in ws, "work_steal must not keep the thread-id gid"


def test_work_steal_rejects_optional_outputs():
    st = load_position_stack(4, seed=0, min_ply=6, max_ply=40)
    for kw in (dict(return_support=True), dict(complete=True),
               dict(return_carriers=True), dict(return_w=True), dict(max_depth=8)):
        with pytest.raises(ValueError):
            solve_vct_mega_bb(st, max_nodes=200, work_steal=True, **kw)


def run_work_steal_identical(B=1200, seed=0, budget=500, resident=1024):
    """work_steal verdict is byte-identical to the base kernel."""
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=40)
    wb, hb, mb = solve_vct_mega_bb(st, max_nodes=budget, return_move=True)
    ww, hw, mw = solve_vct_mega_bb(st, max_nodes=budget, return_move=True,
                                   work_steal=True, resident=resident)
    assert np.array_equal(wb, ww), "work_steal win != base"
    assert np.array_equal(hb, hw), "work_steal hit != base"
    assert np.array_equal(mb, mw), "work_steal move != base"
    return B


def test_work_steal_byte_identical():
    # resident < B (streaming), ~B, and >> B; plus B below a threadgroup.
    for resident in (256, 1024, 8192):
        run_work_steal_identical(B=1200, seed=0, budget=500, resident=resident)
    for B in (1, 31, 33):
        run_work_steal_identical(B=B, seed=7, budget=400, resident=1024)


def test_streaming_consistent_with_base():
    """solve_vct_streaming latches clean verdicts across deepening budgets; on any
    board both it and a single deepest-budget base call leave clean, they agree —
    and streaming resolves at least as many boards. Checked for BOTH the default
    base-kernel deepening (#94) and the work_steal=True opt-in — they must give
    identical verdicts (the substrate is a perf choice, not a correctness one)."""
    budgets = (250, 1000, 4000)
    st = load_position_stack(1500, seed=3, min_ply=6, max_ply=40)
    wb, hb, mb = solve_vct_mega_bb(st, max_nodes=budgets[-1], return_move=True)
    for ws in (False, True):
        sw, sh, sm = solve_vct_streaming(st, budgets=budgets, work_steal=ws,
                                         return_move=True)
        both_clean = ~sh & ~hb
        assert np.array_equal(sw[both_clean], wb[both_clean]), f"streaming win != base (ws={ws})"
        assert np.array_equal(sm[both_clean], mb[both_clean]), f"streaming move != base (ws={ws})"
        assert int((~sh).sum()) >= int((~hb).sum()), f"streaming resolved fewer (ws={ws})"


def test_lanes_source_untouched_default():
    """lanes=1 (the default) source is unchanged; lanes>1 injects exactly the
    lane-partitioned scans + cluster reductions (issue #114)."""
    assert _build_src(False, False) == _src(), "default source drifted"
    ls = _build_src(False, False, lanes=8)
    assert "uint gid = _tid / 8u;" in ls
    assert "simd_shuffle_xor" in ls
    assert "if ((ii++ % 8u) != _lane) continue;" in ls
    assert "if (_lane == 0u){" in ls


def test_lanes_rejects_optional_outputs():
    st = load_position_stack(4, seed=0, min_ply=6, max_ply=40)
    for kw in (dict(return_support=True), dict(complete=True),
               dict(return_carriers=True), dict(return_w=True),
               dict(max_depth=8), dict(work_steal=True)):
        with pytest.raises(ValueError):
            solve_vct_mega_bb(st, max_nodes=200, lanes=8, **kw)
    with pytest.raises(ValueError):
        solve_vct_mega_bb(st, max_nodes=200, lanes=3)


def run_lanes_identical(B=1200, seed=0, budget=500):
    """lanes=K verdict is bit-identical to the base kernel for every K: the
    lane-partitioned scans commit to min(winning cell) == the sequential scan's
    lowbit4 order, and the node count / hitcap latch are untouched."""
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=40)
    wb, hb, mb = solve_vct_mega_bb(st, max_nodes=budget, return_move=True)
    for K in (2, 4, 8, 16, 32):
        w, h, m = solve_vct_mega_bb(st, max_nodes=budget, return_move=True,
                                    lanes=K)
        assert np.array_equal(w, wb), f"lanes={K} win != base"
        assert np.array_equal(h, hb), f"lanes={K} hit != base"
        assert np.array_equal(m, mb), f"lanes={K} move != base"


def test_lanes_byte_identical():
    run_lanes_identical(B=1200, seed=0, budget=500)
    run_lanes_identical(B=33, seed=1, budget=50)    # sub-threadgroup B
    run_lanes_identical(B=1, seed=2, budget=4000)   # single board


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
    test_depth_cap_byte_identical_and_composes()
    test_depth_cap_ceiling_equals_default()
    test_depth_monotonic()
    test_md_min_bracket()
    test_work_steal_source_untouched_default()
    test_work_steal_rejects_optional_outputs()
    test_work_steal_byte_identical()
    test_streaming_consistent_with_base()
    test_lanes_source_untouched_default()
    run_lanes_identical(B=1200, seed=0, budget=500)
    run_lanes_identical(B=33, seed=1, budget=50)
    run_lanes_identical(B=1, seed=2, budget=4000)
    if MD_FIXTURE.exists():
        test_md_matches_golden(np.load(MD_FIXTURE))
    print("FAST tier PASS")
