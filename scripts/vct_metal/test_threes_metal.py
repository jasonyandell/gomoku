"""Validate threes_metal (GPU) against threes_ref (which is validated vs gomoku.vcf).

    GOMOKU_BOARD_SIZE=15 timeout 120 uv run python -m scripts.vct_metal.test_threes_metal

threes_ref.forcing_threes is the ground truth (it excludes four-moves + unsound m
+ non-candidates, exactly as vcf does). The kernel is the pure STRUCTURAL detector,
so we check:
  (a) every truth three -> kernel is_three=1 with matching reply-set & fork;
  (b) every kernel is_three=1 NOT in truth is excluded for a legit reason
      (four-move, or defender-immediate-five-after-m, or non-candidate).
Run on own-no-immediate-five boards (the OR-node regime where four_structure ==
vcf._completions_through, so the exclusions line up).
"""
from __future__ import annotations

import numpy as np

from gomoku import vcf
from scripts.vct_metal import detect_ref as D
from scripts.vct_metal import threes_ref as T
from scripts.vct_metal import threes_metal as TM
from scripts.vct_metal.test_detect_ref import gen_clustered

N = D.N


def run(B: int = 80, seed: int = 0):
    own, opp = gen_clustered(B, seed)
    # restrict to the OR-node regime: attacker has no immediate five
    keep = ~D.five_completion_mask(own, opp).reshape(B, -1).any(1)
    own, opp = own[keep], opp[keep]
    B = own.shape[0]

    is_three, is_fork, reply = TM.forcing_threes_gpu(own, opp)
    nc_base, _ = D.four_structure(own, opp)
    cand = D.candidate_mask(own, opp)

    fails = []
    stats = dict(threes=0, forks=0, extra_legit=0)
    for b in range(B):
        truth = {m: (set(np.flatnonzero(T.reply_mask_and_fork(th)[0]).tolist()),
                     T.reply_mask_and_fork(th)[1])
                 for m, th in T.forcing_threes(own[b], opp[b])}
        stats["threes"] += len(truth)

        # (a) every truth three is found, with matching reply + fork
        for m, (rset, fork) in truth.items():
            if not is_three[b].reshape(-1)[m]:
                fails.append((b, m, "missed_three")); continue
            kr = TM.reply_words_to_cells(reply[b].reshape(-1, 8)[m])
            if kr != rset:
                fails.append((b, m, "reply", sorted(kr ^ rset)[:6]))
            if bool(is_fork[b].reshape(-1)[m]) != bool(fork):
                fails.append((b, m, "fork", (bool(is_fork[b].reshape(-1)[m]), bool(fork))))
            stats["forks"] += int(fork)

        # (b) kernel extras must be legitimately excluded
        kernel_m = set(np.flatnonzero(is_three[b].reshape(-1)).tolist())
        for m in kernel_m - set(truth):
            is_four = nc_base[b].reshape(-1)[m] >= 1
            att2 = own[b].copy(); att2.reshape(-1)[m] = True
            unsound = vcf._has_immediate_five(opp[b], ~(att2 | opp[b]))
            non_cand = not cand[b].reshape(-1)[m]
            if is_four or unsound or non_cand:
                stats["extra_legit"] += 1
            else:
                fails.append((b, m, "spurious_three"))

    print(f"N={N} boards={B} seed={seed}")
    print(f"coverage: truth_threes={stats['threes']} forks={stats['forks']} "
          f"kernel_extras_legit={stats['extra_legit']}")
    if fails:
        print(f"FAIL ({len(fails)}), first 10:")
        for f in fails[:10]:
            print("   ", f)
    else:
        print("PASS  threes_metal matches threes_ref (modulo caller-side exclusions)")
    return fails


def test_threes_metal_matches_ref():
    assert run(B=60, seed=1) == []


if __name__ == "__main__":
    bad = []
    for s in (0, 1, 2):
        bad += run(B=80, seed=s)
        print()
    raise SystemExit(1 if bad else 0)
