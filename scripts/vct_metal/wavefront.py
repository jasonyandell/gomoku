"""Batched wavefront VCT solver — host orchestration, all-GPU per-node primitives.

Solves B positions at once. Each wave: stack the live OR-node boards, run the GPU
kernels over the whole frontier (detection + forcing-threes + a swapped-detect tempo
pass), expand every node, push children to the next wave. Then a single reverse-order
AND/OR backup. This is the GPU-shaped replacement for search_ref's B=1 recursion.

Soundness:
  * four soundness (defender no five after att+m): conservative def5 test -- never a
    false win (it can only SKIP a sound four), exact whenever the defender has no
    pre-existing five-completion (the common case).
  * three tempo guard (defender no four/five after att+m): EXACT, via a batched
    swapped-detect pass over the materialised att+m boards.
  * forks / double-fours / five = immediate WIN leaves; cap (depth/nodes) = UNKNOWN.

Threes are NOT capped at MAX_THREE_BRANCH here (the node cap bounds the tree); so on
nodes with many threes this can find wins vcf's cap drops -- still sound (no false
win). Verdict validated against gomoku.vcf on clean (no-cap) positions.
"""
from __future__ import annotations

import numpy as np

from gomoku import state_ops
from scripts.vct_metal.detect_metal import detect_or_node
from scripts.vct_metal.threes_metal import forcing_threes_gpu, reply_words_to_cells

N = state_ops.BOARD_SIZE
UNK, WIN, NOWIN = 0, 1, 2
_has5 = state_ops.has_five_in_a_row


def solve_batch(boards: np.ndarray, *, max_depth: int = 8, max_nodes: int = 40000):
    """boards: (B,2,N,N) bool (plane 0 = attacker). Returns (win, hit_cap): (B,) bool."""
    B = boards.shape[0]
    kind: list[str] = []
    par: list[int] = []
    rt: list[int] = []
    dep: list[int] = []
    st: list[int] = []
    ch: list[list[int]] = []
    incomplete: list[bool] = []
    bown: list = []
    bopp: list = []

    def add(k, p, r, d, o=None, op=None, status=UNK):
        i = len(kind)
        kind.append(k); par.append(p); rt.append(r); dep.append(d)
        st.append(status); ch.append([]); incomplete.append(False)
        bown.append(o); bopp.append(op)
        return i

    roots = []
    for b in range(B):
        o = np.ascontiguousarray(boards[b, 0], bool)
        op = np.ascontiguousarray(boards[b, 1], bool)
        i = add('OR', -1, b, 1, o, op)
        roots.append(i)
        if _has5(op):
            st[i] = NOWIN
        elif _has5(o):
            st[i] = WIN
    frontier = [i for i in roots if st[i] == UNK]

    while frontier:
        F = len(frontier)
        os = np.stack([bown[i] for i in frontier])
        ps = np.stack([bopp[i] for i in frontier])
        five, ncomp, blk = detect_or_node(os, ps)
        is_three, is_fork, reply = forcing_threes_gpu(os, ps)
        def5 = detect_or_node(ps, os)[0]                 # defender five-completions
        def5_cnt = def5.reshape(F, -1).sum(1)

        # ---- exact tempo guard: collect three-candidate (att+m) boards, batch-detect
        tjobs = []
        for fi in range(F):
            node = frontier[fi]
            if dep[node] > max_depth or len(kind) >= max_nodes:
                continue
            if five[fi].any():
                continue
            ncf = ncomp[fi].reshape(-1)
            for m in np.flatnonzero(is_three[fi].reshape(-1)):
                if ncf[int(m)] < 1:                       # not a four-move
                    tjobs.append((fi, int(m)))
        tempo = {}
        if tjobs:
            atm = np.stack([_place(bown[frontier[fi]], m) for fi, m in tjobs])
            dfp = np.stack([bopp[frontier[fi]] for fi, _ in tjobs])
            df, dnc, _ = detect_or_node(dfp, atm)         # defender five+four after att+m
            for k, (fi, m) in enumerate(tjobs):
                tempo[(fi, m)] = bool(df[k].any() or (dnc[k] >= 1).any())

        # ---- expand
        nextf = []
        for fi in range(F):
            node = frontier[fi]
            d = dep[node]
            if d > max_depth or len(kind) >= max_nodes:
                st[node] = UNK                            # capped leaf
                continue
            if five[fi].any():
                st[node] = WIN
                continue
            o, op = bown[node], bopp[node]
            ncf = ncomp[fi].reshape(-1)
            bkf = blk[fi].reshape(-1)
            d5 = def5[fi].reshape(-1)
            d5c = def5_cnt[fi]

            def four_sound(m):                            # conservative, never false win
                return (d5c - d5[m]) == 0

            # --- pass 1: immediate wins (sound double-four, or fork three w/ tempo-pass)
            won = False
            for m in np.flatnonzero(ncf >= 2):
                if four_sound(int(m)):
                    won = True
                    break
            if not won:
                for m in np.flatnonzero(is_three[fi].reshape(-1)):
                    m = int(m)
                    if ncf[m] >= 1 or tempo.get((fi, m), True):
                        continue
                    if is_fork[fi].reshape(-1)[m]:
                        won = True
                        break
            if won:
                st[node] = WIN
                continue

            # --- pass 2: branches (single fours -> forced-block OR child; threes -> AND)
            kids = []
            for m in np.flatnonzero(ncf == 1):
                m = int(m)
                if not four_sound(m):
                    continue
                if len(kind) >= max_nodes:
                    incomplete[node] = True
                    break
                a = _place(o, m)
                dd = _place(op, int(bkf[m]))
                cid = add('OR', node, rt[node], d + 1, a, dd)
                kids.append(cid)
                nextf.append(cid)

            for m in np.flatnonzero(is_three[fi].reshape(-1)):
                m = int(m)
                if ncf[m] >= 1 or tempo.get((fi, m), True):
                    continue
                if len(kind) >= max_nodes:
                    incomplete[node] = True
                    break
                a = _place(o, m)
                and_id = add('AND', node, rt[node], d + 1)
                and_kids = []
                for dcell in reply_words_to_cells(reply[fi].reshape(-1, 8)[m]):
                    if a.reshape(-1)[dcell] or op.reshape(-1)[dcell]:
                        continue
                    dd = _place(op, dcell)
                    if _has5(dd):
                        cid = add('OR', and_id, rt[node], d + 2, a.copy(), dd, status=NOWIN)
                        and_kids.append(cid)
                    elif len(kind) >= max_nodes:
                        incomplete[and_id] = True
                        break
                    else:
                        cid = add('OR', and_id, rt[node], d + 2, a.copy(), dd)
                        and_kids.append(cid)
                        nextf.append(cid)
                ch[and_id] = and_kids
                kids.append(and_id)

            ch[node] = kids
            if not kids and not incomplete[node]:
                st[node] = NOWIN

        frontier = nextf

    # ---- backup (reverse index order: children always have higher index than parent)
    for i in range(len(kind) - 1, -1, -1):
        if not ch[i]:
            continue                                      # leaf: status already final
        cs = [st[c] for c in ch[i]]
        if kind[i] == 'OR':
            if any(c == WIN for c in cs):
                st[i] = WIN
            elif all(c == NOWIN for c in cs) and not incomplete[i]:
                st[i] = NOWIN
            else:
                st[i] = UNK
        else:  # AND
            if cs and all(c == WIN for c in cs) and not incomplete[i]:
                st[i] = WIN
            elif any(c == NOWIN for c in cs):
                st[i] = NOWIN
            else:
                st[i] = UNK

    win = np.array([st[roots[b]] == WIN for b in range(B)])
    hit_cap = np.array([st[roots[b]] == UNK for b in range(B)])
    return win, hit_cap


def _place(plane: np.ndarray, cell: int) -> np.ndarray:
    p = plane.copy()
    p.reshape(-1)[cell] = True
    return p
