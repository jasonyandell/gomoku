"""DEEP-tier validation for the GPU VCT megakernel — run ON-DEMAND, NOT in the
fast pytest gate (it is CPU-heavy and uses the retired oracle).

    GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_metal.validate_deep

It sets ``GOMOKU_ALLOW_CPU_SOLVER=1`` and exercises three things the committed
fast fixture deliberately does NOT:

  1. VERDICT vs the live CPU oracle ``gomoku.vcf.solve_vct`` at high budget over a
     larger n (FP/FN must be 0 on boards neither solver capped).
  2. winmask SOUNDNESS: every move in the kernel's ``complete`` winmask is an
     independently verified forced win (play m, solve every defender reply child
     with the kernel; m wins iff all replies lose).
  3. winmask COMPLETENESS: every independently-verified-winning FORCING first move
     is in the winmask. The forcing-move oracle mirrors vcf's exact root candidate
     generation INCLUDING the tempo guard ``_defender_has_four_or_five`` — the
     subtlety that earlier made a sound kernel look "incomplete" (the kernel was
     right; the verifier was missing the guard). A verified-win move that is NOT
     forcing is a free win in an already-won position, NOT a VCT first move, so it
     is correctly absent from the winmask.

Clear PASS/FAIL, time-bounded by the (small) n and budget knobs.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("GOMOKU_ALLOW_CPU_SOLVER", "1")

from gomoku import state_ops, vcf  # noqa: E402
from scripts.vct_metal.mega_vct_bb import (  # noqa: E402
    cells_from_words,
    solve_vct_mega_bb,
)
from scripts.vct_metal.positions import load_position_stack  # noqa: E402

N = state_ops.BOARD_SIZE
NN = N * N


# --------------------------------------------------------------------------- #
# Forcing-move oracle: exact mirror of vcf._vct_attack root candidate generation
# (fours u forcing-threes u immediate-five-completions), INCLUDING the tempo
# guard. Copied/adapted from the reference gold_complete.py.
# --------------------------------------------------------------------------- #
def forcing_moves(attacker0: np.ndarray, defender0: np.ndarray) -> set[int]:
    attacker = attacker0.copy()
    defender = defender0.copy()
    empty_plane = ~(attacker | defender)
    empty_idx = vcf._empties_from_plane(empty_plane)
    out = {int(m) for m in vcf._five_completions(attacker, empty_plane)}
    candidates = vcf._candidate_cells_from_planes(attacker, defender, empty_idx)
    occupied = attacker | defender
    four_set: set[int] = set()
    for m in candidates:
        mr, mc = int(m) // N, int(m) % N
        attacker[mr, mc] = True
        occupied[mr, mc] = True
        comps = vcf._completions_through(attacker, int(m), occupied)
        attacker[mr, mc] = False
        occupied[mr, mc] = False
        if comps:
            four_set.add(int(m))
            out.add(int(m))
    for m in candidates:
        if int(m) in four_set:
            continue
        mr, mc = int(m) // N, int(m) % N
        attacker[mr, mc] = True
        new_empty = ~(attacker | defender)
        if vcf._has_immediate_five(defender, new_empty):
            attacker[mr, mc] = False
            continue
        of_cands = vcf._collinear_empties(int(m), new_empty)
        threats = vcf._open_four_threats(attacker, defender, new_empty, of_cands)
        if not threats:
            attacker[mr, mc] = False
            continue
        # TEMPO GUARD: a three where the defender has a counter four/five is NOT
        # forcing — the defender seizes initiative (vcf._defender_has_four_or_five).
        tempo = vcf._defender_has_four_or_five(defender, attacker, new_empty)
        attacker[mr, mc] = False
        if not tempo:
            out.add(int(m))
    return out


def empties(board: np.ndarray) -> list[int]:
    occ = board[0] | board[1]
    return [i for i in range(NN) if not occ[i // N, i % N]]


def has_five_np(plane: np.ndarray) -> bool:
    return bool(state_ops.has_five_in_a_row(plane))


def verify_moves_for_board(board: np.ndarray, cand_moves, max_nodes: int) -> dict:
    """m -> True/False/None (None = inconclusive cap): play m, solve every defender
    reply with the kernel; m wins iff every reply loses."""
    specs = []
    children = []
    trivial: dict[int, bool] = {}
    for m in cand_moves:
        r, c = divmod(m, N)
        if board[0, r, c] or board[1, r, c]:
            continue
        b2 = board.copy()
        b2[0, r, c] = True
        if has_five_np(b2[0]):
            trivial[m] = True
            continue
        for d in empties(b2):
            dr, dc = divmod(d, N)
            ch = b2.copy()
            ch[1, dr, dc] = True
            specs.append((m, d))
            children.append(ch)
    res = {m: True for m in trivial}
    if children:
        ch = np.stack(children)
        wins = np.empty(len(ch), bool)
        hits = np.empty(len(ch), bool)
        for s in range(0, len(ch), 16000):
            w, h = solve_vct_mega_bb(ch[s:s + 16000], max_nodes=max_nodes)
            wins[s:s + len(w)] = w
            hits[s:s + len(h)] = h
        by_m: dict[int, list] = {}
        for (m, _), w, h in zip(specs, wins, hits):
            by_m.setdefault(m, []).append((bool(w), bool(h)))
        for m, lst in by_m.items():
            if any(h for _, h in lst):
                res[m] = None
            elif all(w for w, _ in lst):
                res[m] = True
            else:
                res[m] = False
    return res


def main(B: int = 256, seed: int = 0, max_nodes: int = 20_000,
         gold_boards: int = 12, gold_nodes: int = 8000) -> int:
    if N != 15:
        raise SystemExit(f"BOARD_SIZE={N}; run with GOMOKU_BOARD_SIZE=15")
    t0 = time.time()
    st = load_position_stack(B, seed=seed, min_ply=6, max_ply=40)
    fails: list[str] = []

    # 1) VERDICT vs the live CPU oracle on clean boards.
    wg, hg = solve_vct_mega_bb(st, max_nodes=max_nodes)
    cpu_w = np.zeros(B, bool)
    cpu_h = np.zeros(B, bool)
    for b in range(B):
        r = vcf.solve_vct(st[b], max_depth=7, max_nodes=max_nodes)
        cpu_w[b] = bool(r.has_forced_win)
        cpu_h[b] = bool(r.hit_cap)
    clean = ~hg & ~cpu_h
    fp = int(((wg & ~cpu_w) & clean).sum())
    fn = int(((~wg & cpu_w) & clean).sum())
    if fp or fn:
        fails.append(f"VERDICT: FP={fp} FN={fn} on {int(clean.sum())} clean boards")
    print(f"[1] verdict vs CPU oracle: clean={int(clean.sum())}/{B} "
          f"FP={fp} FN={fn}")

    # 2+3) winmask soundness + completeness on a few winning boards.
    wc, hc, winmask = solve_vct_mega_bb(st, max_nodes=max_nodes, complete=True)
    gold_idx = [b for b in range(B) if wc[b] and not hc[b]][:gold_boards]
    chk = unsound = missing = nonforce_gap = 0
    for b in gold_idx:
        wm = set(cells_from_words(winmask[b]))
        forc = forcing_moves(st[b, 0], st[b, 1])
        ver = verify_moves_for_board(st[b], empties(st[b]), gold_nodes)
        for m in wm:
            chk += 1
            if ver.get(m) is not True:
                unsound += 1
                fails.append(f"SOUND: board{b} move{m} in winmask but verify={ver.get(m)}")
        for m in forc:
            if ver.get(m) is True and m not in wm:
                missing += 1
                fails.append(f"COMPLETE: board{b} forcing winning move {m} MISSING from winmask")
        gap = {m for m, v in ver.items() if v is True and m not in wm}
        nonforce_gap += len(gap - forc)
    print(f"[2/3] winmask over {len(gold_idx)} boards: checked={chk} "
          f"unsound={unsound} forcing-missing={missing} "
          f"non-forcing-gap(OK)={nonforce_gap}")

    dt = time.time() - t0
    print(f"\nRESULT: {'PASS' if not fails else f'FAIL ({len(fails)})'}  [{dt:.1f}s]")
    for f in fails[:25]:
        print("  ", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
