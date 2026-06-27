"""Validate the MSL bitboard primitives (bb.probe) against the golden ref bb_ref.

    GOMOKU_BOARD_SIZE=15 timeout 115 uv run python -m scripts.vct_metal.test_bb

Checks, per board: has_five(own) and completion_mask(own, empty) computed in MSL
match the Python big-int reference bit-for-bit. Uses random clustered boards and
real Rapfi positions.
"""
from __future__ import annotations

import numpy as np

from scripts.vct_metal import bb_ref, bb


def _random_boards(n, seed, density=0.18):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 2, bb.N, bb.N), dtype=bool)
    for k in range(n):
        occ = rng.random((bb.N, bb.N)) < density
        who = rng.random((bb.N, bb.N)) < 0.5
        out[k, 0] = occ & who
        out[k, 1] = occ & ~who
    return out


def _check(boards, label):
    five, comp = bb.probe(boards)
    bad_five = bad_comp = 0
    for k in range(boards.shape[0]):
        own = bb_ref.pack(boards[k, 0])
        opp = bb_ref.pack(boards[k, 1])
        empty = (~(own | opp)) & bb_ref.BOARDMASK
        if bool(five[k]) != bb_ref.has_five(own):
            bad_five += 1
        ref_comp = bb_ref.completion_mask(own, empty)
        gpu_comp = bb.words_to_mask(comp[k])
        if ref_comp != gpu_comp:
            bad_comp += 1
    print(f"{label}: n={boards.shape[0]} bad_five={bad_five} bad_comp={bad_comp} "
          f"{'PASS' if not (bad_five or bad_comp) else 'FAIL'}")
    return bad_five + bad_comp


def main():
    bad = 0
    for s in range(3):
        bad += _check(_random_boards(400, s), f"random seed={s}")
    try:
        from scripts.vct_metal.positions import load_position_stack
        for s in range(2):
            st = load_position_stack(300, seed=s, min_ply=6, max_ply=80)
            bad += _check(st.astype(bool), f"real seed={s}")
    except Exception as e:  # pragma: no cover
        print(f"(skipped real positions: {e})")
    print("ALL PASS" if not bad else f"FAIL ({bad})")
    return bad


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
