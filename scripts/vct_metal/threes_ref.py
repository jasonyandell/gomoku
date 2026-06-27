"""Forcing-threes detection + the BITMASK defense algebra (the v0 70%).

v0's wall was assembling, per forcing-three per node, the open-four threat list +
tempo guard on the Python host as lists of ``(f, comps)`` tuples. The approach's
sharpening (B) is to represent each threat's *defeating set* ``{f} ∪ comps`` as a
cell **bitmask** and do all the defender reasoning as parallel set-algebra:

    reply-set (AND-node) = OR  of the threats' defeating masks
    fork / immediate win = ∃ i,j : maskᵢ AND maskⱼ == 0   (disjoint -> unblockable)

This module:
  * ``forcing_threes`` — the SPEC for a forcing three and its threats, assembled
    from the trusted ``gomoku.vcf`` helpers exactly as ``_vct_attack`` does (so it
    is correct-by-construction; the GPU kernel will later reproduce THIS output
    using on-device detection instead of the scalar helpers).
  * ``defeating_mask`` / ``reply_mask_and_fork`` — the new bitmask algebra (B),
    which ``tests`` then pins against the scalar oracle (``_has_disjoint_threats``
    + the ``_vct_defend`` reply-set union).

Reference masks are plain ``(N*N,)`` bool for clarity; the kernel will pack them
into 8×uint32 words (AND/OR/popcount) — same algebra, different storage.
"""
from __future__ import annotations

import numpy as np

from gomoku import vcf
from scripts.vct_metal import detect_ref as D

N = D.N


def forcing_threes(own: np.ndarray, opp: np.ndarray):
    """List of ``(m, threats)`` for the side to move (``own``), where each
    ``threats`` is a list of ``(f, comps)`` open-four threats the three at ``m``
    creates. Mirrors ``gomoku.vcf._vct_attack`` step 3 (forcing threes only;
    fours are excluded). ``own``/``opp`` are ``(N, N)`` bool. Not mutated.
    """
    att = own.copy()
    deff = opp.copy()
    occupied = att | deff
    empty_idx = vcf._empties_from_plane(~occupied)
    candidates = vcf._candidate_cells_from_planes(att, deff, empty_idx)

    # Four-making moves are handled by the OR-node before threes -> exclude them.
    four_set = set()
    for m in candidates:
        mr, mc = divmod(int(m), N)
        att[mr, mc] = True
        occupied[mr, mc] = True
        comps = vcf._completions_through(att, int(m), occupied)
        att[mr, mc] = False
        occupied[mr, mc] = False
        if comps:
            four_set.add(int(m))

    threes = []
    for m in candidates:
        if int(m) in four_set:
            continue
        mr, mc = divmod(int(m), N)
        att[mr, mc] = True
        new_empty = ~(att | deff)
        if vcf._has_immediate_five(deff, new_empty):   # soundness: no defender 5
            att[mr, mc] = False
            continue
        of_cands = vcf._collinear_empties(int(m), new_empty)
        threats = vcf._open_four_threats(att, deff, new_empty, of_cands)
        att[mr, mc] = False
        if threats:
            threes.append((int(m), [(int(f), [int(c) for c in comps])
                                    for f, comps in threats]))
    return threes


# --------------------------------------------------------------------------- #
# (B) the bitmask defense algebra
# --------------------------------------------------------------------------- #
def defeating_mask(f: int, comps: list[int]) -> np.ndarray:
    """The cells that neutralise threat ``(f, comps)``: ``{f} ∪ comps``."""
    m = np.zeros(N * N, dtype=bool)
    m[f] = True
    if comps:
        m[np.asarray(comps, dtype=np.int64)] = True
    return m


def reply_mask_and_fork(threats):
    """Return ``(reply_mask, is_fork)`` for an AND-node from its threat list.

    reply_mask : (N*N,) bool  -- union of every threat's defeating set (the
                                 bounded, sound defender reply set).
    is_fork    : bool         -- two threats with disjoint defeating sets exist
                                 => no single defender move stops both => win.
    """
    masks = [defeating_mask(f, comps) for f, comps in threats]
    reply = np.zeros(N * N, dtype=bool)
    is_fork = False
    for i, mi in enumerate(masks):
        reply |= mi
        for j in range(i):
            if not (mi & masks[j]).any():        # disjoint defeating sets
                is_fork = True
    return reply, is_fork
