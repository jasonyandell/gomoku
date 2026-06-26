"""Validate the batched wavefront solver vs gomoku.vcf on REAL Rapfi positions.

    GOMOKU_BOARD_SIZE=15 timeout 120 uv run python -m scripts.vct_metal.test_wavefront

Compare verdict on clean (no-cap) positions. A wavefront WIN where vcf says NOWIN is
re-checked against vcf with its three-branch cap lifted -- if vcf-uncapped then agrees
WIN, the wavefront merely found a win vcf's heuristic cap had dropped (sound). A
NOWIN-vs-WIN, or a win vcf-uncapped still rejects, is a real failure.
"""
from __future__ import annotations

import sys
import time
import numpy as np

from gomoku import vcf
from scripts.vct_metal.wavefront import solve_batch
from scripts.vct_metal.positions import load_position_stack

MAX_DEPTH = 8
MAX_NODES = 20000


def run(B: int = 50, seed: int = 0):
    stack = load_position_stack(B, seed=seed, min_ply=6, max_ply=60)
    t = time.time()
    win, hit = solve_batch(stack, max_depth=MAX_DEPTH, max_nodes=MAX_NODES)
    dt = time.time() - t

    clean_agree = cap = 0
    false_pos = []   # wavefront WIN, truth NOWIN  (the dangerous direction)
    false_neg = []   # wavefront NOWIN, vcf WIN cleanly
    found_extra = 0  # wavefront WIN that vcf-capped missed but vcf-uncapped confirms
    vwins = 0
    for b in range(B):
        print(".", end="", flush=True)
        v = vcf.solve_vct(stack[b], max_depth=MAX_DEPTH, max_nodes=MAX_NODES)
        vwins += int(v.has_forced_win)
        mw, mh = bool(win[b]), bool(hit[b])
        if mw == v.has_forced_win:
            if not (mh or v.hit_cap):
                clean_agree += 1
            continue
        if mh or v.hit_cap:
            cap += 1
            continue
        if mw and not v.has_forced_win:
            # re-check with vcf's three-branch cap lifted
            old = vcf._MAX_THREE_BRANCH
            try:
                vcf._MAX_THREE_BRANCH = 64
                v2 = vcf.solve_vct(stack[b], max_depth=MAX_DEPTH, max_nodes=MAX_NODES * 3)
            finally:
                vcf._MAX_THREE_BRANCH = old
            if v2.has_forced_win:
                found_extra += 1
            else:
                false_pos.append(b)
        else:
            false_neg.append(b)

    print(f"N={solve_batch.__globals__['N']} boards={B} seed={seed} "
          f"solve={dt:.2f}s ({dt/B*1000:.0f} ms/board)")
    print(f"  vcf_wins={vwins} clean_agree={clean_agree} cap_boundary={cap} "
          f"found_extra(vcf-cap-missed)={found_extra}")
    print(f"  FALSE_POSITIVES={len(false_pos)} {false_pos[:6]}  "
          f"FALSE_NEGATIVES={len(false_neg)} {false_neg[:6]}")
    ok = not false_pos and not false_neg
    print("  PASS" if ok else "  FAIL")
    return false_pos, false_neg


def test_wavefront_matches_vcf():
    fp, fn = run(B=80, seed=1)
    assert not fp and not fn


if __name__ == "__main__":
    bad = 0
    for s in (0, 1, 2):
        fp, fn = run(B=120, seed=s)
        bad += len(fp) + len(fn)
        print()
    raise SystemExit(1 if bad else 0)
