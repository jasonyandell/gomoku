"""Validate the batched numpy detection (detect_ref) against the CPU oracle gomoku.vcf.

Run directly for a readable report:
    GOMOKU_BOARD_SIZE=15 uv run python scripts/vct_metal/test_detect_ref.py
Also importable as pytest (the asserts fire under `uv run pytest`).
"""
from __future__ import annotations

import numpy as np

from gomoku import vcf
from scripts.vct_metal import detect_ref as D

N = D.N


def gen_clustered(B: int, seed: int, lo: int = 8, hi: int = 36):
    """Grow connected blobs of alternating stones -> dense local structure
    (fours / open threes actually occur), unlike sparse random scatter."""
    rng = np.random.default_rng(seed)
    own = np.zeros((B, N, N), bool)
    opp = np.zeros((B, N, N), bool)
    for b in range(B):
        nstones = int(rng.integers(lo, hi))
        cells = [(int(rng.integers(N)), int(rng.integers(N)))]
        occ = {cells[0]}
        while len(cells) < nstones:
            r0, c0 = cells[int(rng.integers(len(cells)))]
            dr, dc = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
            r, c = r0 + dr, c0 + dc
            if 0 <= r < N and 0 <= c < N and (r, c) not in occ:
                occ.add((r, c))
                cells.append((r, c))
        for i, (r, c) in enumerate(cells):
            (own if i % 2 == 0 else opp)[b, r, c] = True
    return own, opp


def _board(own, opp, b):
    return np.ascontiguousarray(np.stack([own[b], opp[b]]), dtype=bool)


def run(B: int = 300, seed: int = 0):
    own, opp = gen_clustered(B, seed)
    empty = D.empties(own, opp)

    five = D.five_completion_mask(own, opp)
    ncomp, block = D.four_structure(own, opp)
    cand = D.candidate_mask(own, opp)
    tempo = D.defender_can_four_or_five(own, opp)

    stats = dict(five=0, four=0, dbl_four=0, tempo=0, four_skipped=0)
    fail = []

    for b in range(B):
        ob, pb = own[b], opp[b]
        eb = empty[b]
        board = _board(own, opp, b)

        # 1. immediate five
        mine5 = set(np.flatnonzero(five[b].reshape(-1)).tolist())
        cpu5 = set(int(x) for x in vcf._five_completions(ob, eb))
        if mine5 != cpu5:
            fail.append((b, "five", sorted(mine5 ^ cpu5)[:8]))
        stats["five"] += bool(cpu5)

        # 2. four_structure: n_comp + block, every empty cell.
        #    four_structure counts fours CREATED BY m (m in the five-window) -- the
        #    solver-relevant quantity. vcf._completions_through additionally counts
        #    completions of PRE-EXISTING fours collinear with m; those only occur
        #    when an immediate five already exists, which the OR-node handles before
        #    four-enumeration. So validate in that regime: skip boards with an
        #    immediate five (there the two semantics legitimately diverge).
        if cpu5:
            stats["four_skipped"] += 1
        else:
            occ_b = ob | pb
            nc_cpu = np.zeros(N * N, dtype=np.int16)
            for m in np.flatnonzero(eb.reshape(-1)):
                mr, mc = divmod(int(m), N)
                ob[mr, mc] = True
                occ_b[mr, mc] = True
                comps = vcf._completions_through(ob, int(m), occ_b)
                ob[mr, mc] = False
                occ_b[mr, mc] = False
                nc_cpu[m] = len(comps)
                if len(comps) >= 1 and ncomp[b].reshape(-1)[m] == 1:
                    if int(block[b].reshape(-1)[m]) not in set(int(x) for x in comps):
                        fail.append((b, "block", int(m)))
            if not np.array_equal(nc_cpu, ncomp[b].reshape(-1)):
                d = np.flatnonzero(nc_cpu != ncomp[b].reshape(-1))[:8]
                fail.append((b, "n_comp", [(int(i), int(nc_cpu[i]),
                                            int(ncomp[b].reshape(-1)[i])) for i in d]))
        stats["four"] += int((ncomp[b] >= 1).any())
        stats["dbl_four"] += int((ncomp[b] >= 2).any())

        # 3. candidate mask
        mine_c = set(np.flatnonzero(cand[b].reshape(-1)).tolist())
        cpu_c = set(int(x) for x in vcf._candidate_cells(board, vcf._empties(board)))
        if mine_c != cpu_c:
            fail.append((b, "cand", sorted(mine_c ^ cpu_c)[:8]))

        # 4. tempo guard (defender can make four/five)
        cpu_t = vcf._defender_has_four_or_five(pb.copy(), ob.copy(), eb.copy())
        if bool(tempo[b]) != bool(cpu_t):
            fail.append((b, "tempo", (bool(tempo[b]), bool(cpu_t))))
        stats["tempo"] += int(cpu_t)

    print(f"N={N}  boards={B}  seed={seed}")
    print(f"coverage: five={stats['five']}  four={stats['four']}  "
          f"double_four={stats['dbl_four']}  tempo={stats['tempo']}  "
          f"four_cmp_skipped(imm5)={stats['four_skipped']}")
    if fail:
        print(f"FAIL  ({len(fail)} mismatches), first 12:")
        for f in fail[:12]:
            print("   ", f)
    else:
        print("PASS  all detection primitives match gomoku.vcf cell-for-cell")
    return fail


def test_detect_ref_matches_vcf():
    assert run(B=200, seed=1) == []


if __name__ == "__main__":
    bad = []
    for s in (0, 1, 2):
        bad += run(B=300, seed=s)
        print()
    raise SystemExit(1 if bad else 0)
