"""GPU/MPS-batched VCT (Victory-by-Continuous-Threats) solver for 15x15
freestyle gomoku (PERF / FEASIBILITY SPIKE).

This is a batched re-implementation of ``gomoku.vcf.solve_vct``'s *verdict*
(``has_forced_win``). Unlike plain VCF (a pure OR/reachability search the
defender cannot branch in), VCT is a real AND/OR proof search: a forcing THREE
leaves the defender a *choice* of cost squares to block (an AND-node), and the
defender may also seize tempo with a counter-four. So the GPU formulation is the
"wave-batched MCTS" pattern: the irregular AND/OR tree lives on the host, but the
heavy per-node THREAT DETECTION (fours, forcing threes, the defender tempo
guard) is batched on the GPU across the WHOLE live node pool each wave -- pooling
nodes from every independent root AND every internal AND/OR branch (this fuses
the prompt's Layer-1 "batch across positions" and Layer-2 "batch the frontier"
into one pool, which is the natural GPU shape).

WHY THIS MATCHES THE CPU VERDICT
--------------------------------
``solve_vct`` is itself a *bounded* search (depth cap, node cap, top-K three
branching, a bounded AND-node reply set, a 1-ply tempo guard). Given the SAME
bounding rules, the AND/OR value of the tree is well-defined and independent of
visit order; node-visit order only changes WHEN a node/depth cap bites. So this
solver replicates those exact rules and computes the full AND/OR value
bottom-up. On any root where BOTH solvers terminate without hitting a cap, the
verdicts must agree. Cap-boundary roots are reported, never a false positive.

KEY REUSE / CORRECTNESS-BY-CONSTRUCTION
---------------------------------------
* Four detection (n_comp / forced block / soundness) and immediate-five reuse
  the proven directional length-5 conv kernels from ``gpu_vcf_prototype``.
* The GPU does the *bulk filtering*: which empty cells are fours, and which are
  candidate forcing threes (placing the stone yields a collinear open-four
  follow-up). For the FEW survivors, the exact threat list / tempo guard /
  disjoint-fork test are confirmed with the very CPU helpers from
  ``gomoku.vcf`` (``_open_four_threats``, ``_defender_has_four_or_five``,
  ``_has_disjoint_threats``) -- so a confirmed three is bit-identical to what the
  CPU solver would build. The GPU pre-filter is complete (no false negatives:
  same n_comp>=2 condition), so it never hides a three the CPU would explore.

Run:  GOMOKU_BOARD_SIZE=15 uv run python scripts/gpu_vct_prototype.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from gomoku.board_config import BOARD_SIZE as N
from gomoku import state_ops
from gomoku.vcf import (
    _open_four_threats,
    _defender_has_four_or_five,
    _has_disjoint_threats,
    _collinear_empties,
    DEFAULT_VCT_MAX_DEPTH,
    DEFAULT_VCT_MAX_NODES,
    _MAX_THREE_BRANCH,
)

# Reuse the proven VCF detection kernels verbatim.
from gpu_vcf_prototype import (
    DIRS,
    WIN_LEN,
    pick_device,
    _shift,
    _build_shift_cache,
    _comp5,
    _four_structure,
    _has5,
)

WIN, LOSS, UNKNOWN = 1, -1, 0


# --------------------------------------------------------------------------- #
# Batched GPU detection over a wave of OR-node boards
# --------------------------------------------------------------------------- #

def _candidate_mask(own: torch.Tensor, opp: torch.Tensor) -> torch.Tensor:
    """(F,N,N) bool: empty cells within Chebyshev distance 2 of any stone.

    Matches ``vcf._candidate_cells_from_planes`` (dilate occupied by 2, drop
    occupied)."""
    occ = own | opp
    near = torch.zeros_like(occ)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            near |= _shift(occ, dr, dc)
    return near & (~occ)


def _attacker_detect(own, opp, idx_shift):
    """Batched attacker-side detection for a wave of boards.

    Returns dict with:
      atk5_any (F,)   : attacker has an immediate five (win now)
      n_comp   (F,N,N): #distinct five-completions if attacker plays this cell
      block    (F,N,N): forced block square (flat) for a single four
      forced_ok(F,N,N): defender has no immediate-five except possibly at this
                        cell (the four/three soundness gate)
      cand     (F,N,N): near-stone candidate empties
      empty    (F,N,N)
    """
    empty = ~(own | opp)
    own_c = _build_shift_cache(own)
    opp_c = _build_shift_cache(opp)
    empty_c = _build_shift_cache(empty)
    F = own.shape[0]

    atk5_cell = _comp5(own_c, empty)
    atk5_any = atk5_cell.flatten(1).any(1)

    def5_cell = _comp5(opp_c, empty)
    def5_count = def5_cell.flatten(1).sum(1)
    forced_ok = (def5_count.view(F, 1, 1) - def5_cell.to(torch.long)) == 0

    n_comp, block = _four_structure(own_c, empty, empty_c, idx_shift)
    cand = _candidate_mask(own, opp)
    return dict(atk5_any=atk5_any, n_comp=n_comp, block=block,
                forced_ok=forced_ok, cand=cand, empty=empty)


def _defender_tempo(own, opp, idx_shift):
    """(F,) bool: defender (plane 1) can make a four or five on these boards.

    Matches ``vcf._defender_has_four_or_five``: a pre-existing defender
    five-completion (open four) OR any defender four-making move."""
    empty = ~(own | opp)
    opp_c = _build_shift_cache(opp)
    empty_c = _build_shift_cache(empty)
    def5_any = _comp5(opp_c, empty).flatten(1).any(1)
    n_comp_d, _ = _four_structure(opp_c, empty, empty_c, idx_shift)
    four_d_any = (n_comp_d >= 1).flatten(1).any(1)
    return def5_any | four_d_any


def _make_idx_shift(dev):
    idx_grid = (torch.arange(N, device=dev).view(N, 1) * N
                + torch.arange(N, device=dev).view(1, N)).view(1, N, N).long()
    idx_shift = {}
    for (dr, dc) in DIRS:
        for delta in (-4, -3, -2, -1, 1, 2, 3, 4):
            idx_shift[(dr, dc, delta)] = _shift(idx_grid, delta * dr, delta * dc)
    return idx_shift


# offsets (dr,dc,sign,k) for the <=32 collinear follow-up cells of a three
_OFFSETS = [(dr, dc, s, k) for (dr, dc) in DIRS for s in (1, -1)
            for k in range(1, WIN_LEN)]


@torch.no_grad()
def _wave_detect(own_np, opp_np, idx_shift, dev, chunk=200_000):
    """Batched detection for a wave of OR-node boards.

    Inputs are (F,2,N,N)-ish numpy (own_np,opp_np each (F,N,N) bool).
    Returns, per node, the raw material the host needs to build children:
      imm_win (F,) bool                 : immediate WIN leaf (five / sound double four)
      sf_list : list[F] of [(m,block)]  : sound single fours
      three_cand : list[F] of [m]       : GPU-filtered forcing-three candidate cells
    The few three candidates are confirmed exactly on the host afterwards.
    """
    F = own_np.shape[0]
    own = torch.as_tensor(own_np, dtype=torch.bool, device=dev)
    opp = torch.as_tensor(opp_np, dtype=torch.bool, device=dev)
    d = _attacker_detect(own, opp, idx_shift)

    n_comp = d["n_comp"]
    forced_ok = d["forced_ok"]
    cand = d["cand"]
    empty = d["empty"]

    single_four = (n_comp == 1) & forced_ok & cand
    double_four = (n_comp >= 2) & forced_ok & cand
    imm_win = d["atk5_any"] | double_four.flatten(1).any(1)

    # Single fours -> host list of (m, block)
    sf_list = [[] for _ in range(F)]
    block_flat = d["block"].reshape(F, -1)
    nz_n, nz_m = single_four.reshape(F, -1).nonzero(as_tuple=True)
    nz_n = nz_n.tolist()
    nz_m = nz_m.tolist()
    blk_vals = block_flat[single_four.reshape(F, -1)].tolist()
    for node_i, m, blk in zip(nz_n, nz_m, blk_vals):
        if not bool(imm_win[node_i]):
            sf_list[node_i].append((int(m), int(blk)))

    # Three CANDIDATE cells: near-stone empties that are NOT four-moves but are
    # sound (defender no imm5). Expand each (place attacker), batch four_structure,
    # and keep those with a collinear open-four (>=2 comps) follow-up.
    three_seed = (n_comp == 0) & forced_ok & cand
    # don't bother for nodes that already win
    live = (~imm_win).view(F, 1, 1)
    three_seed = three_seed & live
    seed_n, seed_m = three_seed.reshape(F, -1).nonzero(as_tuple=True)
    M = seed_n.shape[0]
    three_cand = [[] for _ in range(F)]
    if M == 0:
        return imm_win.cpu().numpy(), sf_list, three_cand

    seed_n_cpu = seed_n.tolist()
    seed_m_cpu = seed_m.tolist()

    # Process the expansion in chunks to bound memory.
    keep_flags = np.zeros(M, dtype=bool)
    for c0 in range(0, M, chunk):
        c1 = min(M, c0 + chunk)
        cn = seed_n[c0:c1]
        cm = seed_m[c0:c1]
        Mc = c1 - c0
        own_e = own[cn].clone()
        opp_e = opp[cn]
        ar = torch.arange(Mc, device=dev)
        mr = cm // N
        mc = cm % N
        own_e[ar, mr, mc] = True
        de = _attacker_detect(own_e, opp_e, idx_shift)
        n_comp2 = de["n_comp"]            # (Mc,N,N)
        empty2 = de["empty"]              # (Mc,N,N) (excludes m)
        # any collinear-with-m follow-up f (dist<=4) that is empty & n_comp2>=2
        has_threat = torch.zeros(Mc, dtype=torch.bool, device=dev)
        for (dr, dc, s, k) in _OFFSETS:
            fr = mr + s * dr * k
            fc = mc + s * dc * k
            valid = (fr >= 0) & (fr < N) & (fc >= 0) & (fc < N)
            frc = fr.clamp(0, N - 1)
            fcc = fc.clamp(0, N - 1)
            ncv = n_comp2[ar, frc, fcc]
            emv = empty2[ar, frc, fcc]
            has_threat |= valid & emv & (ncv >= 2)
        keep_flags[c0:c1] = has_threat.cpu().numpy()

    for i in range(M):
        if keep_flags[i]:
            three_cand[seed_n_cpu[i]].append(int(seed_m_cpu[i]))

    return imm_win.cpu().numpy(), sf_list, three_cand


# --------------------------------------------------------------------------- #
# Host AND/OR tree orchestration with wave-batched detection
# --------------------------------------------------------------------------- #

class _Tree:
    """Flat node table; backup is a single reverse pass (children always have a
    larger id than their parent because we expand strictly by increasing depth)."""

    def __init__(self):
        self.kind: list[str] = []      # 'OR' | 'AND' | 'LEAF'
        self.val: list[int] = []       # WIN / LOSS / UNKNOWN (LEAF preset)
        self.children: list[list[int]] = []
        self.root: list[int] = []
        self.depth: list[int] = []

    def add(self, kind, root, depth, val=UNKNOWN):
        i = len(self.kind)
        self.kind.append(kind)
        self.val.append(val)
        self.children.append([])
        self.root.append(root)
        self.depth.append(depth)
        return i

    def backup(self):
        kind = self.kind
        val = self.val
        children = self.children
        for i in range(len(kind) - 1, -1, -1):
            k = kind[i]
            if k == 'LEAF':
                continue
            cs = children[i]
            if k == 'OR':
                v = LOSS
                allloss = True
                for c in cs:
                    cv = val[c]
                    if cv == WIN:
                        v = WIN
                        allloss = False
                        break
                    if cv != LOSS:
                        allloss = False
                if v != WIN:
                    v = LOSS if (cs and allloss) else (LOSS if not cs else UNKNOWN)
                val[i] = v
            else:  # AND
                v = WIN
                allwin = True
                for c in cs:
                    cv = val[c]
                    if cv == LOSS:
                        v = LOSS
                        allwin = False
                        break
                    if cv != WIN:
                        allwin = False
                if v != LOSS:
                    v = WIN if allwin else UNKNOWN
                val[i] = v


def solve_vct_batch(boards: np.ndarray, *,
                    max_depth: int = DEFAULT_VCT_MAX_DEPTH,
                    max_nodes: int = DEFAULT_VCT_MAX_NODES,
                    max_three_branch: int = _MAX_THREE_BRANCH,
                    device: str | None = None,
                    max_frontier: int = 300_000):
    """Batched VCT verdict for B positions (plane 0 = attacker/side-to-move).

    boards: (B,2,N,N) bool. Returns (won, hit_cap) numpy (B,).
    ``won[b]`` matches ``gomoku.vcf.solve_vct(boards[b]).has_forced_win`` whenever
    neither solver hit a cap; ``hit_cap[b]`` flags roots whose verdict was bounded
    by a depth / node / frontier cap (a "no win" there is unproven, never a false
    positive)."""
    dev = device or pick_device()
    idx_shift = _make_idx_shift(dev)
    B = boards.shape[0]
    boards = np.ascontiguousarray(boards, dtype=bool)

    won = np.zeros(B, dtype=bool)
    hit_cap = np.zeros(B, dtype=bool)
    root_nodes = np.zeros(B, dtype=np.int64)   # per-root node budget counter

    tree = _Tree()
    root_node_id = np.full(B, -1, dtype=np.int64)

    # Root special cases (mirror solve_vct).
    own0 = torch.as_tensor(boards[:, 0], dtype=torch.bool, device=dev)
    opp0 = torch.as_tensor(boards[:, 1], dtype=torch.bool, device=dev)
    def_has5 = _has5(_build_shift_cache(opp0)).cpu().numpy()
    atk_has5 = _has5(_build_shift_cache(own0)).cpu().numpy()

    # Frontier of OR-nodes to expand: parallel lists.
    f_id: list[int] = []
    f_own: list[np.ndarray] = []
    f_opp: list[np.ndarray] = []

    for b in range(B):
        if def_has5[b]:
            won[b] = False
            continue
        if atk_has5[b]:
            won[b] = True
            continue
        nid = tree.add('OR', b, 1)
        root_node_id[b] = nid
        root_nodes[b] += 1
        f_id.append(nid)
        f_own.append(boards[b, 0].copy())
        f_opp.append(boards[b, 1].copy())

    while f_id:
        F = len(f_id)
        if F > max_frontier:
            # frontier blow-up: truncate, flag dropped roots capped.
            for j in range(max_frontier, F):
                # truncated node becomes an UNKNOWN leaf; backup decides whether
                # its root is actually left unproven (hit_cap) or proven anyway.
                tree.val[f_id[j]] = UNKNOWN
                tree.kind[f_id[j]] = 'LEAF'
            f_id = f_id[:max_frontier]
            f_own = f_own[:max_frontier]
            f_opp = f_opp[:max_frontier]
            F = max_frontier

        own_np = np.stack(f_own)
        opp_np = np.stack(f_opp)
        imm_win, sf_list, three_cand = _wave_detect(own_np, opp_np, idx_shift, dev)

        next_id: list[int] = []
        next_own: list[np.ndarray] = []
        next_opp: list[np.ndarray] = []

        for j in range(F):
            nid = f_id[j]
            r = tree.root[nid]
            depth = tree.depth[nid]
            own = f_own[j]
            opp = f_opp[j]

            if imm_win[j]:
                tree.kind[nid] = 'LEAF'
                tree.val[nid] = WIN
                continue

            child_depth = depth + 1
            over_depth = child_depth > max_depth

            made_child = False

            # ---- single fours: forced single OR-child (place m + block) ----
            for (m, blk) in sf_list[j]:
                if root_nodes[r] >= max_nodes or over_depth:
                    cid = tree.add('LEAF', r, child_depth, UNKNOWN)
                    tree.children[nid].append(cid)
                    made_child = True
                    continue
                no = own.copy(); npp = opp.copy()
                no[m // N, m % N] = True
                npp[blk // N, blk % N] = True
                cid = tree.add('OR', r, child_depth)
                root_nodes[r] += 1
                tree.children[nid].append(cid)
                next_id.append(cid); next_own.append(no); next_opp.append(npp)
                made_child = True

            # ---- forcing threes ----
            # CPU order (solve_vct): collect ALL threes-with-threats, sort
            # most-threats-first, TRUNCATE to top-K, THEN per survivor apply the
            # tempo guard / disjoint-fork test. (Truncating before the tempo guard
            # matters: a tempo-skipped three still consumes a top-K slot.)
            threes_all = []   # (m, threats)
            for m in three_cand[j]:
                no = own.copy()
                no[m // N, m % N] = True
                new_empty = ~(no | opp)
                of_cands = _collinear_empties(m, new_empty)
                threats = _open_four_threats(no, opp, new_empty, of_cands)
                if threats:
                    threes_all.append((m, threats))
            threes_all.sort(key=lambda mt: -len(mt[1]))
            del threes_all[max_three_branch:]

            won_by_three = False
            for (m, threats) in threes_all:
                no = own.copy()
                no[m // N, m % N] = True
                new_empty = ~(no | opp)
                # tempo guard (rule 4) FIRST
                if _defender_has_four_or_five(opp, no, new_empty):
                    continue
                # disjoint fork => immediate WIN at this OR node (depth-2 win)
                if _has_disjoint_threats(threats):
                    cid = tree.add('LEAF', r, child_depth, WIN)
                    tree.children[nid].append(cid)
                    made_child = True
                    won_by_three = True
                    break
                # AND-node: bounded reply set = union of {f} U comps
                reply_set = set()
                for f, comps in threats:
                    reply_set.add(f)
                    reply_set.update(comps)
                replies = sorted(d for d in reply_set
                                 if not (no[d // N, d % N] or opp[d // N, d % N]))
                if not replies:
                    continue
                if root_nodes[r] >= max_nodes or over_depth:
                    cid = tree.add('LEAF', r, child_depth, UNKNOWN)
                    tree.children[nid].append(cid)
                    made_child = True
                    continue
                and_id = tree.add('AND', r, child_depth)
                root_nodes[r] += 1
                tree.children[nid].append(and_id)
                made_child = True
                gchild_depth = child_depth + 1
                gc_over = gchild_depth > max_depth
                for dd in replies:
                    npp = opp.copy()
                    npp[dd // N, dd % N] = True
                    if state_ops.has_five_in_a_row(npp):
                        gid = tree.add('LEAF', r, gchild_depth, LOSS)
                        tree.children[and_id].append(gid)
                        continue
                    if root_nodes[r] >= max_nodes or gc_over:
                        gid = tree.add('LEAF', r, gchild_depth, UNKNOWN)
                        tree.children[and_id].append(gid)
                        continue
                    gid = tree.add('OR', r, gchild_depth)
                    root_nodes[r] += 1
                    tree.children[and_id].append(gid)
                    next_id.append(gid)
                    next_own.append(no.copy())
                    next_opp.append(npp)

            if not made_child and not tree.children[nid]:
                # no fours, no threes -> attacker cannot continue: LOSS leaf
                tree.kind[nid] = 'LEAF'
                tree.val[nid] = LOSS

        f_id, f_own, f_opp = next_id, next_own, next_opp

    tree.backup()

    for b in range(B):
        nid = root_node_id[b]
        if nid < 0:
            continue  # decided by root special case
        v = tree.val[nid]
        if v == WIN:
            won[b] = True
        elif v == UNKNOWN:
            hit_cap[b] = True

    return won, hit_cap


# --------------------------------------------------------------------------- #
# Correctness + throughput harness
# --------------------------------------------------------------------------- #

def _gen_positions(n: int, seed: int = 0, lo: int = 8, hi: int = 30):
    from gomoku.game import GameState
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n):
        s = GameState.initial()
        plies = int(rng.integers(lo, hi))
        for _ in range(plies):
            la = s.legal_actions()
            if len(la) == 0:
                break
            s = s.apply(int(rng.choice(la)))
            done, _ = s.is_terminal()
            if done:
                break
        boards.append(np.asarray(s.board, dtype=bool))
    return np.stack(boards)


def correctness_test(n: int = 300, seed: int = 0,
                     max_depth: int = DEFAULT_VCT_MAX_DEPTH,
                     max_nodes: int = DEFAULT_VCT_MAX_NODES, lo=8, hi=30):
    from gomoku.vcf import solve_vct
    boards = _gen_positions(n, seed, lo, hi)
    cpu = np.zeros(n, dtype=bool)
    cpu_cap = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i in range(n):
        r = solve_vct(boards[i], max_depth=max_depth, max_nodes=max_nodes)
        cpu[i] = r.has_forced_win
        cpu_cap[i] = r.hit_cap
    cpu_dt = time.time() - t0

    t0 = time.time()
    gpu, gpu_cap = solve_vct_batch(boards, max_depth=max_depth, max_nodes=max_nodes)
    gpu_dt = time.time() - t0

    clean = ~(cpu_cap | gpu_cap)
    agree = (cpu == gpu)
    n_clean = int(clean.sum())
    n_agree_clean = int((agree & clean).sum())
    # false positives: gpu says WIN but cpu (clean, no cap) says no-win
    fp = np.flatnonzero(gpu & ~cpu & ~cpu_cap)
    mism_clean = np.flatnonzero(~agree & clean)
    print(f"--- VCT CORRECTNESS (n={n}, depth={max_depth}, plies {lo}-{hi}) ---")
    print(f"CPU forced-wins: {int(cpu.sum())}/{n}  (cpu {cpu_dt:.2f}s, {n/cpu_dt:.1f}/s)")
    print(f"GPU forced-wins: {int(gpu.sum())}/{n}  (gpu {gpu_dt:.2f}s, {n/gpu_dt:.1f}/s)")
    print(f"CPU hit_cap: {int(cpu_cap.sum())}   GPU hit_cap: {int(gpu_cap.sum())}")
    print(f"Agreement (all):   {int(agree.sum())}/{n} = {agree.mean()*100:.2f}%")
    print(f"Agreement (clean both): {n_agree_clean}/{n_clean} = "
          f"{(n_agree_clean/max(1,n_clean))*100:.2f}%")
    print(f"FALSE POSITIVES (gpu win, cpu clean no-win): {len(fp)}")
    if len(mism_clean):
        print(f"Clean mismatches ({len(mism_clean)}): {mism_clean[:15].tolist()}")
        for i in mism_clean[:8]:
            print(f"  pos {i}: cpu={cpu[i]} gpu={gpu[i]} "
                  f"cpu_cap={cpu_cap[i]} gpu_cap={gpu_cap[i]}")
    return boards, cpu, gpu, cpu_cap, gpu_cap


def throughput_test(batch_sizes=(256, 1024, 4096, 16384),
                    max_depth=DEFAULT_VCT_MAX_DEPTH,
                    max_nodes=250, seed=7, lo=8, hi=30, cpu_baseline_n=40):
    """PRELIMINARY throughput. Same node cap on CPU and GPU (apples-to-apples).

    Use a TIGHT max_nodes: CPU solve_vct is ~16ms/node so the production cap
    (20000) makes the CPU baseline take minutes-per-solve. v0 is CPU-bound (the
    GPU path is ~16% of wall) -- see wiki/topics/gpu-vct-feasibility.md."""
    from gomoku.vcf import solve_vct
    print("\n--- VCT THROUGHPUT (PRELIMINARY, GPU batched full solves/sec) ---")
    dev = pick_device()
    print(f"device={dev}, board={N}x{N}, max_depth={max_depth}, max_nodes={max_nodes}")
    pool = _gen_positions(max(batch_sizes), seed, lo, hi)

    # CPU baseline on a subsample (it is very slow).
    t0 = time.time()
    for i in range(cpu_baseline_n):
        solve_vct(pool[i], max_depth=max_depth, max_nodes=max_nodes)
    cpu_sps = cpu_baseline_n / (time.time() - t0)
    print(f"CPU baseline: {cpu_sps:.2f} solves/s (n={cpu_baseline_n})")

    print(f" {'batch':>7} {'solves/s':>11} {'speedup':>9} {'wins%':>6} {'cap%':>6}")
    for B in batch_sizes:
        boards = pool[:B] if B <= pool.shape[0] else \
            np.concatenate([pool] * (B // pool.shape[0] + 1))[:B]
        w, cap = solve_vct_batch(boards, max_depth=max_depth,
                                 max_nodes=max_nodes, device=dev)  # warmup
        if dev == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        w, cap = solve_vct_batch(boards, max_depth=max_depth,
                                 max_nodes=max_nodes, device=dev)
        if dev == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        sps = B / dt
        print(f" {B:>7} {sps:>11.1f} {sps/cpu_sps:>8.1f}x "
              f"{w.mean()*100:>5.1f} {cap.mean()*100:>5.1f}")


if __name__ == "__main__":
    import sys
    if N != 15:
        print(f"WARNING: board size is {N}, expected 15 "
              f"(set GOMOKU_BOARD_SIZE=15)", file=sys.stderr)
    # NOTE: CPU solve_vct is ~16 ms/node on open 15x15 boards, so the baselines
    # below use a TIGHT node cap (250) to stay tractable; same cap on both sides
    # keeps the clean-agreement comparison valid. See
    # wiki/topics/gpu-vct-feasibility.md for the full (preliminary) numbers and
    # the CPU-bound diagnosis -- this v0 is correctness-validated but ~2x CPU.
    correctness_test(n=120, seed=0, max_nodes=250)
    throughput_test(batch_sizes=(256, 1024), max_depth=DEFAULT_VCT_MAX_DEPTH,
                    cpu_baseline_n=15)
