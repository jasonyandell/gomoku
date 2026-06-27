"""Validate the bitmask defense algebra (threes_ref) against the scalar oracle.

    GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_metal.test_threes_ref
"""
from __future__ import annotations

import numpy as np

from gomoku import vcf
from scripts.vct_metal import threes_ref as T
from scripts.vct_metal.test_detect_ref import gen_clustered

N = T.N


def run(B: int = 300, seed: int = 0):
    own, opp = gen_clustered(B, seed)
    fail = []
    stats = dict(boards_with_threes=0, threes=0, forks=0, max_threats=0)

    for b in range(B):
        threes = T.forcing_threes(own[b], opp[b])
        if threes:
            stats["boards_with_threes"] += 1
        for m, threats in threes:
            stats["threes"] += 1
            stats["max_threats"] = max(stats["max_threats"], len(threats))

            reply_mask, is_fork = T.reply_mask_and_fork(threats)
            mine_reply = set(np.flatnonzero(reply_mask).tolist())

            # scalar oracle: reply-set union and the disjoint-fork test
            cpu_reply = set()
            for f, comps in threats:
                cpu_reply.add(int(f))
                cpu_reply.update(int(c) for c in comps)
            cpu_fork = vcf._has_disjoint_threats(
                [(f, comps) for f, comps in threats])

            if mine_reply != cpu_reply:
                fail.append((b, m, "reply", sorted(mine_reply ^ cpu_reply)[:8]))
            if bool(is_fork) != bool(cpu_fork):
                fail.append((b, m, "fork", (is_fork, cpu_fork)))
            stats["forks"] += int(cpu_fork)

    print(f"N={N}  boards={B}  seed={seed}")
    print(f"coverage: boards_with_threes={stats['boards_with_threes']}  "
          f"threes={stats['threes']}  forks={stats['forks']}  "
          f"max_threats_on_one_three={stats['max_threats']}")
    if fail:
        print(f"FAIL ({len(fail)}), first 12:")
        for f in fail[:12]:
            print("   ", f)
    else:
        print("PASS  bitmask defense (reply-set + fork) matches scalar oracle")
    return fail


def test_threes_ref_defense_matches_oracle():
    assert run(B=200, seed=1) == []


if __name__ == "__main__":
    bad = []
    for s in (0, 1, 2):
        bad += run(B=300, seed=s)
        print()
    raise SystemExit(1 if bad else 0)
