"""Falsification harness for the stencil CERTIFICATE property (#88 measured, #89 hardened).

Claim under test (the "do we have a proof?" question): a VCT stencil that wins **in
isolation** — its carrier stones alone on an empty board, support cells empty — is a
forced win on ANY board where it fits and the defender has no *immediate counter-tempo*,
by the SAME forcing line. This is the soundness of Allis's dependency-based / threat-space
search (1994) made operational on our GPU-mined stencils; here we try to BREAK it.

#88 measured the property with a CONSERVATIVE filter ("defender has no VCT of its own").
#89 sharpens it to the EXACT boundary the megakernel itself uses — `def_tempo(opp,empty)`
= `completion_mask(opp)` (defender can make a five) ∪ `gen_forcing(opp)` (defender can make
a four) — and pushes on three fronts:

  A. ``defender_has_four_or_five(boards)`` — a cheap per-board probe of the EXACT
     immediate-tempo condition (the kernel's `def_tempo`, lifted out as one MLX kernel),
     cross-checked against ``bb.probe`` (the completion_mask the kernel uses for fives).
  B. transfer test re-run under the EXACT immediate-tempo filter (tempo-safe = no
     defender four/five AT START) instead of "no defender VCT". More permissive ⇒ a
     stronger test (it admits boards where the defender even has a *slower* VCT).
  C. ADVERSARIAL mid-sequence test: a forced defender BLOCK can itself create a four —
     counter-tempo born mid-sequence, invisible to a start-of-board filter. We place
     defender stones collinear to the shape's support cells (the forced-block cells) to
     manufacture it, keep boards tempo-safe AT START, and check for refutations. If any
     refute, we CHARACTERIZE the exact condition (the sharpened theorem).
  D. NEAR-EDGE translations: stamp self-contained shapes flush against each board edge /
     corner (breathing room cut by the boundary, not a stone) and confirm "fits on-board
     (all support+carrier cells in-bounds)" is sufficient — or find an edge failure.
  E. scale: multiple seeds and a larger pool.

Pipeline (all bulk-synchronous per the call-cost law; batches auto-split at <=16384,
which is verdict-identical since per-board verdicts are independent):
  1. mine clean attacker VCTs from a pool of real positions (support + carriers).
  2. self-containment: does each win reproduce from carriers ALONE on an empty board?
  3. transfer / adversarial / edge perturbations + the EXACT immediate-tempo filter.

Run (from repo root):
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python \
      scripts/threat_shapes/certificate_falsification.py --pool 4096 --seeds 0,1
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python \
      scripts/threat_shapes/certificate_falsification.py --self-test   # probe unit-check
"""

from __future__ import annotations

import argparse

import numpy as np
import mlx.core as mx

from scripts.vct_metal import bb, mega_vcf_bb
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb, cells_from_words, N

# Call-cost-law cap. Per-board verdicts are independent (one GPU thread per board,
# thread-local node budget), so splitting a >16384 batch into <=16384 chunks is
# verdict-identical to one call — it only honors the <=16384 batch constraint.
MAX_BATCH = 16384
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


# --------------------------------------------------------------------------- #
# A. EXACT immediate-tempo probe — the megakernel's def_tempo(opp, empty), per
#    board. completion_mask(opp) [defender can complete a five] OR gen_forcing(opp)
#    [defender can make a four]. These are *exactly* the ingredients the VCT kernel
#    uses to reject a non-forcing attacker three; here they are lifted out as a
#    cheap standalone probe. Additive: solve_vct_mega_bb is untouched.
# --------------------------------------------------------------------------- #
_TEMPO_SRC = """
    uint gid = thread_position_in_grid.x;
    uint base = gid * 4u;
    thread ulong own[4]; thread ulong opp[4]; thread ulong empty[4];
    for (uint w=0; w<4u; w++){ own[w]=own_in[base+w]; opp[w]=opp_in[base+w]; }
    for (uint w=0; w<4u; w++){ empty[w] = (~(own[w]|opp[w])); }
    empty[3] &= TOPMASK;
    // EXACT def_tempo(opp, empty): defender five-completion OR defender four-move.
    ulong c[4]; completion_mask(opp, empty, c);
    ulong f[4]; gen_forcing(opp, empty, f);
    tempo_out[gid] = (any4(c) || any4(f)) ? 1 : 0;
"""

_TEMPO_KERNEL = mx.fast.metal_kernel(
    name="def_tempo_probe",
    input_names=["own_in", "opp_in"],
    output_names=["tempo_out"],
    source=_TEMPO_SRC,
    header=mega_vcf_bb._HEADER,  # has completion_mask + gen_forcing + any4 + TOPMASK
)


def _tempo_once(boards: np.ndarray) -> np.ndarray:
    own, opp = bb.pack_words(boards)
    B = boards.shape[0]
    o = mx.array(own.reshape(-1))
    p = mx.array(opp.reshape(-1))
    (t,) = _TEMPO_KERNEL(
        inputs=[o, p], grid=(B, 1, 1), threadgroup=(min(B, 64), 1, 1),
        output_shapes=[(B,)], output_dtypes=[mx.uint8])
    mx.eval(t)
    return np.array(t).astype(bool)


def defender_has_four_or_five(boards: np.ndarray) -> np.ndarray:
    """(B,2,N,N) bool -> (B,) bool: the DEFENDER (board[1]) has an immediate
    four-move OR five-completion — the megakernel's EXACT ``def_tempo(opp, empty)``.
    This is the precise start-of-board immediate-tempo condition (#89). Cheap: one
    set-algebra probe, no AND/OR search. Auto-chunked at MAX_BATCH."""
    B = boards.shape[0]
    if B <= MAX_BATCH:
        return _tempo_once(boards)
    return np.concatenate([_tempo_once(boards[i:i + MAX_BATCH])
                           for i in range(0, B, MAX_BATCH)])


def _defender_has_five(boards: np.ndarray) -> np.ndarray:
    """(B,) bool: defender ALREADY has five-in-a-row (terminal / illegal as a test
    board). Reuses bb.probe's has_five on the flipped board."""
    five, _ = bb.probe(boards[:, [1, 0]].copy())
    return five


# --------------------------------------------------------------------------- #
# Bulk-synchronous solve, auto-chunked at MAX_BATCH (verdict-identical to one call).
# --------------------------------------------------------------------------- #
def _solve_chunked(boards: np.ndarray, *, max_nodes: int, **kw):
    B = boards.shape[0]
    if B <= MAX_BATCH:
        return solve_vct_mega_bb(boards, max_nodes=max_nodes, **kw)
    parts = [solve_vct_mega_bb(boards[i:i + MAX_BATCH], max_nodes=max_nodes, **kw)
             for i in range(0, B, MAX_BATCH)]
    n = len(parts[0])
    return tuple(np.concatenate([p[j] for p in parts], axis=0) for j in range(n))


def _iso_board(carrier_cells) -> np.ndarray:
    """A board with only the carrier stones as own (attacker to move), else empty."""
    bd = np.zeros((2, N, N), bool)
    for c in carrier_cells:
        bd[0, c // N, c % N] = True
    return bd


def _line_cells(s: int, step, k_from: int, k_to: int):
    """Cells s + step*k for k in [k_from, k_to] (step = (dr, dc)); None if any cell
    falls off-board (no column wrap — the steps are genuine board lines)."""
    sr, sc = divmod(s, N)
    dr, dc = step
    out = []
    for k in range(k_from, k_to + 1):
        r, c = sr + dr * k, sc + dc * k
        if not (0 <= r < N and 0 <= c < N):
            return None
        out.append(r * N + c)
    return out


# --------------------------------------------------------------------------- #
# Shared mining: clean attacker VCTs -> (carriers, support); keep self-contained.
# --------------------------------------------------------------------------- #
def mine_self_contained(pool_boards: np.ndarray, *, max_nodes: int):
    win, hit, supp, carr = _solve_chunked(
        pool_boards, max_nodes=max_nodes,
        return_support=True, return_carriers=True)
    wins = np.flatnonzero(win & ~hit)
    if len(wins) == 0:
        return wins, wins, supp, carr
    iso = np.stack([_iso_board(cells_from_words(carr[b])) for b in wins])
    wi, hi = _solve_chunked(iso, max_nodes=max_nodes)
    sc = wins[wi & ~hi]
    return wins, sc, supp, carr


# --------------------------------------------------------------------------- #
# B. Transfer test under the EXACT immediate-tempo filter (+ the #88 no-VCT filter
#    for comparison, and the immediate-tempo control).
# --------------------------------------------------------------------------- #
def run_transfer(sc, supp, carr, *, rng, max_nodes, per_shape, n_opp, cap):
    sc = sc[:cap]
    pert, owner = [], []
    for b in sc:
        forbidden = set(cells_from_words(supp[b])) | set(cells_from_words(carr[b]))
        base = _iso_board(cells_from_words(carr[b]))
        empties = [c for c in range(N * N) if c not in forbidden]
        for _ in range(per_shape):
            bd = base.copy()
            for c in rng.choice(empties, size=min(n_opp, len(empties)), replace=False):
                bd[1, c // N, c % N] = True
            pert.append(bd)
            owner.append(int(b))
    if not pert:
        print("  transfer: no self-contained shapes to perturb")
        return {"refuted_tempo": 0, "tested_tempo": 0}
    pert = np.stack(pert)
    owner = np.array(owner)

    tempo = defender_has_four_or_five(pert)                       # EXACT start tempo
    wf, hf = _solve_chunked(pert[:, [1, 0]].copy(), max_nodes=max_nodes)  # defender VCT
    wa, ha = _solve_chunked(pert, max_nodes=max_nodes)            # attacker VCT
    clean = ~ha & ~_defender_has_five(pert)                       # not capped, not terminal

    # cross-check: any defender five-completion (bb.probe flipped) must trip def_tempo.
    _, dcomp = bb.probe(pert[:, [1, 0]].copy())
    five_comp = np.array([len(cells_from_words(dcomp[i])) > 0 for i in range(len(pert))])
    assert np.all(tempo[five_comp]), "def_tempo probe missed a defender five-completion"

    safe_t = ~tempo & clean                  # EXACT immediate-tempo filter (#89)
    safe_v = ~wf & ~hf & clean               # #88 conservative "no defender VCT"
    ctrl_t = tempo & clean                   # immediate tempo present (control)
    slow_v = ~tempo & wf & clean             # tempo-safe BUT defender has a (slower) VCT

    def rate(mask):
        n = int(mask.sum())
        return int((wa & mask).sum()), int((~wa & mask).sum()), n

    pres_t, ref_t, n_t = rate(safe_t)
    pres_v, ref_v, n_v = rate(safe_v)
    pres_c, ref_c, n_c = rate(ctrl_t)
    pres_s, ref_s, n_s = rate(slow_v)

    print(f"  transfer: {len(pert)} boards ({len(sc)} shapes x {per_shape}, "
          f"{n_opp} random opp stones each)")
    print(f"    [#89 EXACT immediate-tempo filter] tempo-safe & clean: {n_t}")
    print(f"        attacker STILL WINS : {pres_t}")
    print(f"        attacker REFUTED    : {ref_t}   <-- counterexamples")
    print(f"    [#88 no-defender-VCT filter]        no-VCT-safe & clean: {n_v}  "
          f"(wins {pres_v}, refuted {ref_v})")
    print(f"    [sharpening] tempo-safe BUT defender HAS a slower VCT: {n_s}  "
          f"(wins {pres_s}, refuted {ref_s})  <-- attacker-moves-first dominance")
    print(f"    [control] defender HAS immediate tempo: {n_c}  "
          f"(wins {pres_c}, refuted {ref_c})  <-- tempo is the breaker")

    # THEOREM restated: EVERY refutation has immediate defender tempo <=> NO
    # tempo-safe board refutes. And the #88 "no-VCT" filter is NOT sound — it
    # admits lone-four-move boards (immediate tempo, but not a full VCT) that
    # refute; the #89 exact-tempo filter excludes exactly those.
    all_ref = ~wa & clean
    safe_ref = int((all_ref & ~tempo).sum())
    v88_unsound = int((~wa & safe_v & tempo).sum())
    print(f"    [THEOREM] {int(all_ref.sum())} refutations among clean boards; "
          f"immediate-tempo {int((all_ref & tempo).sum())}; tempo-SAFE refutations {safe_ref}")
    print(f"    [#88 unsound] of its {ref_v} refutations, {v88_unsound} have immediate "
          f"tempo (#89 correctly excludes these)")

    if ref_t:
        _dump_examples("transfer", np.flatnonzero((~wa) & safe_t)[:3],
                       pert, owner, supp, carr)
    return {"refuted_tempo": ref_t, "tested_tempo": n_t}


# --------------------------------------------------------------------------- #
# C. Adversarial mid-sequence test: manufacture a defender four via a forced block.
# --------------------------------------------------------------------------- #
def run_adversarial(sc, supp, carr, *, rng, max_nodes, cap, combo_per_shape=4, combo_k=3):
    sc = sc[:cap]
    cand, owner, ptype, anchor = [], [], [], []
    for b in sc:
        carr_cells = cells_from_words(carr[b])
        supp_cells = cells_from_words(supp[b])
        forbidden = set(carr_cells) | set(supp_cells)
        base = _iso_board(carr_cells)
        for s in supp_cells:                              # each support cell = a forced-block candidate
            for dr, dc in DIRS:
                for sign in (1, -1):
                    step = (dr * sign, dc * sign)
                    # 2-stone: D at s+1,s+2 -> after a forced block at s the defender
                    # has 3-in-line = a four-MOVE (def_tempo). Only 2 opp stones, so
                    # the board is start-tempo-safe BY CONSTRUCTION.
                    c2 = _line_cells(s, step, 1, 2)
                    if c2 and all(c not in forbidden for c in c2):
                        bd = base.copy()
                        for c in c2:
                            bd[1, c // N, c % N] = True
                        cand.append(bd); owner.append(int(b))
                        ptype.append("2stone"); anchor.append(s)
                    # 3-stone: D at s+1,s+2,s+3 -> a four-MOVE ALREADY AT START
                    # (control: the start-tempo filter MUST drop these).
                    c3 = _line_cells(s, step, 1, 3)
                    if c3 and all(c not in forbidden for c in c3):
                        bd = base.copy()
                        for c in c3:
                            bd[1, c // N, c % N] = True
                        cand.append(bd); owner.append(int(b))
                        ptype.append("3stone"); anchor.append(s)
        # combo: 2-stone setups at a random subset of support cells at once (chase a
        # rarer two-block real-four). The filter drops any that trip start-tempo.
        if len(supp_cells) >= 2:
            subsets = [rng.choice(supp_cells, size=min(combo_k, len(supp_cells)),
                                  replace=False) for _ in range(combo_per_shape)]
            subsets.append(np.array(supp_cells))   # "loaded": prime EVERY support cell
            for chosen in subsets:
                bd = base.copy(); used = set(); placed = False
                for s in chosen:
                    dr, dc = DIRS[int(rng.integers(4))]
                    sign = 1 if rng.integers(2) else -1
                    c2 = _line_cells(int(s), (dr * sign, dc * sign), 1, 2)
                    if c2 and all(c not in forbidden and c not in used for c in c2):
                        for c in c2:
                            bd[1, c // N, c % N] = True
                            used.add(c)
                        placed = True
                if placed:
                    cand.append(bd); owner.append(int(b))
                    ptype.append("combo"); anchor.append(-1)
    if not cand:
        print("  adversarial: no candidates generated")
        return {"refuted_tempo": 0, "tested_tempo": 0}
    cand = np.stack(cand)
    owner = np.array(owner); ptype = np.array(ptype); anchor = np.array(anchor)

    tempo = defender_has_four_or_five(cand)
    wa, ha = _solve_chunked(cand, max_nodes=max_nodes)
    clean = ~ha & ~_defender_has_five(cand)
    safe = ~tempo & clean
    refuted = (~wa) & safe

    # PRIMED diagnostic (proves the attack is real, not vacuous): for each 2stone
    # board, fill the anchor support cell s with a defender stone (= the defender's
    # forced block) and re-probe. "primed" = that block manufactures a defender
    # four/five = exactly the mid-sequence counter-tempo we are hunting. If the
    # defender is ever forced through s, tempo is born — yet the attacker still wins.
    is2 = ptype == "2stone"
    blocked = cand.copy()
    for i in np.flatnonzero(is2):
        s = int(anchor[i]); blocked[i, 1, s // N, s % N] = True
    primed_tempo = defender_has_four_or_five(blocked)
    primed2 = int((is2 & ~tempo & primed_tempo).sum())   # start-safe 2stone that blocking primes
    base2 = int((is2 & ~tempo).sum())
    print(f"  adversarial: {len(cand)} candidate boards "
          f"(defender stones placed collinear to support / forced-block cells)")
    print(f"    PRIMED check: {primed2}/{base2} start-tempo-safe 2-stone boards become "
          f"tempo-UNSAFE if the defender is forced to block the anchor cell "
          f"(=mechanism is live)")
    for pt in ("2stone", "3stone", "combo"):
        m = ptype == pt
        n = int(m.sum())
        if not n:
            continue
        n_safe = int((m & ~tempo).sum())
        n_caught = int((m & tempo).sum())
        n_tested = int((m & safe).sum())
        n_ref = int((m & refuted).sum())
        print(f"    {pt:8s}: {n:6d}  start-tempo-safe {n_safe:6d} / caught-by-filter "
              f"{n_caught:6d} | tested {n_tested:6d}  refuted {n_ref}")
    n_ref_all = int(refuted.sum())
    all_ref = ~wa & clean
    print(f"    TOTAL start-tempo-safe & clean: {int(safe.sum())}   "
          f"attacker REFUTED: {n_ref_all}   <-- mid-sequence counterexamples")
    print(f"    [THEOREM] {int(all_ref.sum())} refutations among clean boards; "
          f"immediate-tempo {int((all_ref & tempo).sum())}; tempo-SAFE refutations "
          f"{int((all_ref & ~tempo).sum())}")

    if n_ref_all:
        _dump_adversarial(np.flatnonzero(refuted)[:5], cand, owner, anchor, supp, carr)
    return {"refuted_tempo": n_ref_all, "tested_tempo": int(safe.sum())}


# --------------------------------------------------------------------------- #
# D. Near-edge translations: does "fits on-board" suffice?
# --------------------------------------------------------------------------- #
def run_edge(sc, supp, carr, *, max_nodes, cap):
    sc = sc[:cap]
    boards, owner, meta = [], [], []
    n_dont_fit = 0
    for b in sc:
        carr_cells = cells_from_words(carr[b])
        foot = carr_cells + cells_from_words(supp[b])          # support+carrier footprint
        rows = [c // N for c in foot]; cols = [c % N for c in foot]
        rmin, rmax, cmin, cmax = min(rows), max(rows), min(cols), max(cols)
        dr_opt = {"top": -rmin, "bot": (N - 1) - rmax}
        dc_opt = {"left": -cmin, "right": (N - 1) - cmax}
        trans = {}
        for rk, drv in dr_opt.items():
            trans[rk] = (drv, 0)
        for ck, dcv in dc_opt.items():
            trans[ck] = (0, dcv)
        for rk, drv in dr_opt.items():
            for ck, dcv in dc_opt.items():
                trans[f"{rk}-{ck}"] = (drv, dcv)
        seen = set()
        for name, (dr, dc) in trans.items():
            if (dr, dc) in seen:
                continue
            seen.add((dr, dc))
            shifted = [((c // N) + dr, (c % N) + dc) for c in foot]
            if not all(0 <= r < N and 0 <= cc < N for r, cc in shifted):
                n_dont_fit += 1
                continue                                       # doesn't fit -> not claimed
            bd = np.zeros((2, N, N), bool)
            for c in carr_cells:
                bd[0, (c // N) + dr, (c % N) + dc] = True
            boards.append(bd); owner.append(int(b)); meta.append((name, dr, dc))
    if not boards:
        print("  edge: no fitting translations")
        return {"fail": 0, "tested": 0}
    boards = np.stack(boards)
    wa, ha = _solve_chunked(boards, max_nodes=max_nodes)
    clean = ~ha
    fits = len(boards)
    won = int((wa & clean).sum())
    lost = int((~wa & clean).sum())
    capd = int(ha.sum())
    print(f"  edge: {fits} fitting flush translations ({n_dont_fit} would-be off-board, "
          f"not claimed)")
    print(f"        attacker WINS {won} / lost {lost} / capped {capd}   "
          f"<-- fit-but-LOSE = edge failures")
    if lost:
        for idx in np.flatnonzero(~wa & clean)[:5]:
            b = owner[idx]; name, dr, dc = meta[idx]
            print(f"    EDGE FAILURE: shape {b} translated {name} by ({dr},{dc}) loses")
            print("      carriers(orig):", cells_from_words(carr[b]))
            print("      support(orig) :", cells_from_words(supp[b]))
    return {"fail": lost, "tested": won + lost}


# --------------------------------------------------------------------------- #
# Example dumps + characterization.
# --------------------------------------------------------------------------- #
def _dump_examples(tag, idxs, boards, owner, supp, carr):
    for idx in idxs:
        b = owner[idx]
        opp = [c for c in range(N * N) if boards[idx, 1, c // N, c % N]]
        print(f"\n  --- {tag} counterexample (shape from board {b}) ---")
        print("    carriers:", cells_from_words(carr[b]))
        print("    support :", cells_from_words(supp[b]))
        print("    added opp:", opp)


def _dump_adversarial(idxs, cand, owner, anchor, supp, carr):
    """For each adversarial refutation, confirm the mechanism: the start board is
    tempo-safe, but FILLING the anchor support cell s (the forced block) with a
    defender stone makes the defender immediately have a four/five = mid-sequence
    counter-tempo."""
    for idx in idxs:
        b = owner[idx]; s = int(anchor[idx])
        opp = [c for c in range(N * N) if cand[idx, 1, c // N, c % N]]
        print(f"\n  --- ADVERSARIAL counterexample (shape from board {b}) ---")
        print("    carriers :", cells_from_words(carr[b]))
        print("    support  :", cells_from_words(supp[b]))
        print("    added opp:", opp, " anchor support cell s =", s)
        start_safe = not bool(defender_has_four_or_five(cand[idx:idx + 1])[0])
        print(f"    start tempo-safe        : {start_safe}")
        if s >= 0:
            blk = cand[idx:idx + 1].copy()
            blk[0, 1, s // N, s % N] = True                    # defender plays the forced block
            after = bool(defender_has_four_or_five(blk)[0])
            print(f"    tempo after block at s  : {after}   "
                  "(True = block manufactures defender four/five = mid-seq counter-tempo)")


# --------------------------------------------------------------------------- #
# Probe self-test (deterministic, no data dir needed).
# --------------------------------------------------------------------------- #
def _selftest_tempo():
    r = 7 * N + 3  # a row-interior anchor far from edges

    def D(cells):
        bd = np.zeros((2, N, N), bool)
        for c in cells:
            bd[1, c // N, c % N] = True
        return bd

    def A(cells):  # attacker (own) stones, defender empty -> def_tempo must be False
        bd = np.zeros((2, N, N), bool)
        for c in cells:
            bd[0, c // N, c % N] = True
        return bd

    cases = [
        ("empty board", D([]), False),
        ("defender pair .OO..", D([r, r + 1]), False),
        ("defender open-three .OOO.", D([r, r + 1, r + 2]), True),       # four-move
        ("defender split O.OO", D([r, r + 2, r + 3]), True),            # four-move
        ("defender four .OOOO.", D([r, r + 1, r + 2, r + 3]), True),    # five-completion
        ("defender gap-four OO.OO", D([r, r + 1, r + 3, r + 4]), True), # five-completion
        ("defender vertical three", D([r, r + N, r + 2 * N]), True),    # four-move (col)
        ("attacker open-three only", A([r, r + 1, r + 2]), False),      # own, not opp
    ]
    boards = np.stack([b for _, b, _ in cases])
    got = defender_has_four_or_five(boards)
    # bb.probe cross-check: defender five-completion (flipped completion_mask) ⇒ def_tempo
    _, dcomp = bb.probe(boards[:, [1, 0]].copy())
    ok_all = True
    for i, (name, _, exp) in enumerate(cases):
        ok = bool(got[i]) == exp
        ok_all &= ok
        comp = len(cells_from_words(dcomp[i])) > 0
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: def_tempo={bool(got[i])} "
              f"(exp {exp}); flipped completion_mask nonempty={comp}")
        if comp:
            assert bool(got[i]), "five-completion present but def_tempo False"
    print(f"\n{'PASS' if ok_all else 'FAIL'}: def_tempo probe self-test")
    return 0 if ok_all else 1


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Harden the stencil certificate property (#89).")
    ap.add_argument("--pool", type=int, default=4096)
    ap.add_argument("--seeds", type=str, default="0", help="comma list, e.g. 0,1,2")
    ap.add_argument("--max-nodes", type=int, default=500)
    ap.add_argument("--min-ply", type=int, default=6)
    ap.add_argument("--max-ply", type=int, default=60)
    ap.add_argument("--per-shape", type=int, default=8, help="transfer perturbations/shape")
    ap.add_argument("--n-opp", type=int, default=12, help="transfer opp stones bolted on")
    ap.add_argument("--cap", type=int, default=400, help="max self-contained shapes tested")
    ap.add_argument("--tests", type=str, default="transfer,adversarial,edge",
                    help="comma subset of {transfer,adversarial,edge}")
    ap.add_argument("--self-test", action="store_true",
                    help="run the def_tempo probe unit self-test and exit")
    args = ap.parse_args()

    if args.self_test:
        import sys
        sys.exit(_selftest_tempo())

    # late import so --help / --self-test work without the game data dir present
    from scripts.vct_metal.positions import load_position_stack

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    tests = {t.strip() for t in args.tests.split(",") if t.strip()}
    grand = {"transfer": [0, 0], "adversarial": [0, 0], "edge": [0, 0]}  # [refuted/fail, tested]

    for seed in seeds:
        rng = np.random.default_rng(seed)
        pool = load_position_stack(args.pool, seed=seed,
                                   min_ply=args.min_ply, max_ply=args.max_ply)
        wins, sc, supp, carr = mine_self_contained(pool, max_nodes=args.max_nodes)
        rate = len(sc) / max(len(wins), 1) * 100
        print(f"\n=== seed {seed}: pool {args.pool} -> {len(wins)} clean attacker wins; "
              f"self-contained {len(sc)}/{len(wins)} = {rate:.0f}% ===")
        if len(sc) == 0:
            continue
        if "transfer" in tests:
            r = run_transfer(sc, supp, carr, rng=rng, max_nodes=args.max_nodes,
                             per_shape=args.per_shape, n_opp=args.n_opp, cap=args.cap)
            grand["transfer"][0] += r["refuted_tempo"]; grand["transfer"][1] += r["tested_tempo"]
        if "adversarial" in tests:
            r = run_adversarial(sc, supp, carr, rng=rng, max_nodes=args.max_nodes, cap=args.cap)
            grand["adversarial"][0] += r["refuted_tempo"]; grand["adversarial"][1] += r["tested_tempo"]
        if "edge" in tests:
            r = run_edge(sc, supp, carr, max_nodes=args.max_nodes, cap=args.cap)
            grand["edge"][0] += r["fail"]; grand["edge"][1] += r["tested"]

    print("\n=== GRAND TOTALS across seeds", seeds, "===")
    print(f"  transfer    : {grand['transfer'][0]} refuted / {grand['transfer'][1]} tempo-safe tested")
    print(f"  adversarial : {grand['adversarial'][0]} refuted / {grand['adversarial'][1]} tempo-safe tested")
    print(f"  edge        : {grand['edge'][0]} fit-but-lose / {grand['edge'][1]} fitting tested")


if __name__ == "__main__":
    main()
