"""A VCT solver composed from the validated primitives — the search SPEC.

This mirrors ``gomoku.vcf``'s AND/OR proof search (``_vct_attack`` /
``_vct_defend``) but built entirely on this package's primitives:
  * fives + fours from ``detect_ref`` (the GPU-portable detection),
  * forcing-threes + the **bitmask defense** (reply-set, fork) from ``threes_ref``.

Its only purpose is to prove those primitives COMPOSE into a correct solver
before we batch the search and move it on-device. It is recursive and slow (one
detection call per node, B=1) — correctness, not speed. Verdict is validated
against ``vcf.solve_vct`` on clean (no-cap) positions, exactly as the v0 spike
was (the proof value of an uncapped AND/OR tree is order-independent, so clean
verdicts must agree).
"""
from __future__ import annotations

import numpy as np

from gomoku import state_ops
from scripts.vct_metal import detect_ref as D
from scripts.vct_metal import threes_ref as T

N = D.N
MAX_THREE_BRANCH = 6  # matches gomoku.vcf._MAX_THREE_BRANCH


def _or(att: np.ndarray, deff: np.ndarray, depth: int, ctx: dict) -> bool:
    """OR-node: attacker to move. True iff a forced win is proven from here."""
    ctx["nodes"] += 1
    if ctx["nodes"] > ctx["max_nodes"] or depth > ctx["max_depth"]:
        ctx["hit"] = True
        return False

    a1 = att[None]
    d1 = deff[None]

    # 1. immediate five
    if D.five_completion_mask(a1, d1)[0].any():
        return True

    # 2. fours (double-four = win; single four = forced-block recursion)
    nc, blk = D.four_structure(a1, d1)
    nc = nc[0].reshape(-1)
    blk = blk[0].reshape(-1)
    four_cells = np.flatnonzero(nc >= 1)
    # double fours first (best-first, matches vcf ordering)
    four_cells = sorted(four_cells.tolist(), key=lambda m: 0 if nc[m] >= 2 else 1)
    for m in four_cells:
        att2 = att.copy()
        att2.reshape(-1)[m] = True
        # soundness: attacker's four is only forcing if defender has no own 5 now
        if D.five_completion_mask(deff[None], att2[None])[0].any():
            continue
        if nc[m] >= 2:
            return True                       # double four: unstoppable
        block = int(blk[m])
        deff2 = deff.copy()
        deff2.reshape(-1)[block] = True
        if _or(att2, deff2, depth + 1, ctx):
            return True

    # 3. forcing threes (each opens a defender AND-node)
    threes = T.forcing_threes(att, deff)
    threes.sort(key=lambda mt: -len(mt[1]))
    del threes[MAX_THREE_BRANCH:]
    for m, threats in threes:
        att2 = att.copy()
        att2.reshape(-1)[m] = True
        # tempo guard FIRST: a defender counter-four/five refutes even a fork
        if bool(D.defender_can_four_or_five(att2[None], deff[None])[0]):
            continue
        reply_mask, is_fork = T.reply_mask_and_fork(threats)
        if is_fork:
            return True                       # two disjoint threats: win in 2
        # AND-node: attacker must win against EVERY sound defender reply
        ok = True
        for d in np.flatnonzero(reply_mask):
            d = int(d)
            if att2.reshape(-1)[d] or deff.reshape(-1)[d]:
                continue
            deff2 = deff.copy()
            deff2.reshape(-1)[d] = True
            if state_ops.has_five_in_a_row(deff2):
                ok = False                    # defender's block makes its own five
                break
            if not _or(att2, deff2, depth + 1, ctx):
                ok = False
                break
        if ok:
            return True

    return False


def solve_vct_ref(board: np.ndarray, *, max_depth: int = 8,
                  max_nodes: int = 20000):
    """Return (has_forced_win, hit_cap). board: (2,N,N) bool, plane 0 = attacker."""
    att = np.ascontiguousarray(board[0], dtype=bool)
    deff = np.ascontiguousarray(board[1], dtype=bool)
    if state_ops.has_five_in_a_row(deff):
        return False, False
    if state_ops.has_five_in_a_row(att):
        return True, False
    ctx = {"nodes": 0, "max_nodes": max_nodes, "max_depth": max_depth, "hit": False}
    win = _or(att, deff, 1, ctx)
    return win, ctx["hit"]
