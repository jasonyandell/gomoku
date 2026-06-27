"""GPU-batched VCF solver for 15x15 freestyle gomoku (PERF SPIKE).

This is an EXACT* re-implementation of ``gomoku.vcf.solve_vcf``'s *verdict*
(``has_forced_win``), parallelised across B independent searches so the heavy
per-ply threat detection runs batched on the GPU (MPS) instead of one position
at a time on the CPU.

Why a re-implementation can be exact
------------------------------------
Plain VCF is a pure OR/reachability search: the defender is ALWAYS forced to the
unique completion square of the attacker's four, so a defender AND-node has
exactly one child. Therefore "is there a forced win?" == "from the root attacker
node, can we reach (within ``max_depth`` attacker moves) a node where the
attacker has an immediate five or a sound double-four?". We run that reachability
as a breadth-first frontier expansion: every node in the current frontier is an
attacker-to-move board at the SAME depth, so we can advance the WHOLE frontier in
lockstep with batched tensor ops.

The detection primitives, all expressed as directional shift-products over the
frontier tensor (own/opp/empty are ``(F, N, N)`` bool on MPS):

  * immediate-five completions of a plane (``comp5``)  -> attacker win-now / the
    defender soundness test;
  * four-making moves + how many DISTINCT completions each has (single vs double
    four) + the single-four's forced block square.

Correctness argument for the four-structure (matches ``vcf._five_completions``):
A four-move ``m`` with completion ``c`` corresponds exactly to a length-5 line
window of 3 own + the two empties {m, c} + 0 opp (proof: a completion after
placing m must run through m, giving 5 consecutive = 3 own + m + c). Each
direction d and signed offset delta = c-m identifies a UNIQUE completion cell
(the 4 line-directions through m pairwise intersect only at m, and within a
direction distinct offsets are distinct cells). So the number of distinct
firing (d, delta) pairs == ``len(comps)`` in the CPU solver, and a single four
(count==1) has its block square at ``m + delta*d``.

Soundness/forcing test (matches ``vcf._attack``'s ``_has_immediate_five``): a
four at m is forcing iff, after attacker occupies m, the defender has NO move
that makes its own five. Placing attacker at m can only remove m from the
defender's completion set, so this is: the defender has no five-completion at any
cell other than m.

* "EXACT" caveat: the CPU solver and this one use DIFFERENT node-budget
  accounting (CPU = DFS ``_attack`` calls; here = BFS frontier nodes). When
  neither hits its cap, both compute the true VCF verdict and agree. A capped CPU
  "no win" is "unproven", so the only possible disagreements are cap-boundary
  cases; the bundled test filters/reports those.

Run:  GOMOKU_BOARD_SIZE=15 uv run python scripts/gpu_vcf_prototype.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from gomoku.board_config import BOARD_SIZE as N

DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
WIN_LEN = 5


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _shift(x: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    """Return y with y[..., r, c] = x[..., r+dr, c+dc]; out-of-board -> 0/False.

    Works on (F, H, W) tensors (bool/int/long)."""
    if dr == 0 and dc == 0:
        return x
    H, W = x.shape[-2], x.shape[-1]
    y = torch.zeros_like(x)
    rd0, rd1 = max(0, -dr), min(H, H - dr)
    cd0, cd1 = max(0, -dc), min(W, W - dc)
    if rd0 < rd1 and cd0 < cd1:
        y[..., rd0:rd1, cd0:cd1] = x[..., rd0 + dr:rd1 + dr, cd0 + dc:cd1 + dc]
    return y


def _build_shift_cache(plane: torch.Tensor) -> dict:
    """Cache shift(plane, k*dir) for every direction and k in -4..4."""
    cache: dict = {}
    for (dr, dc) in DIRS:
        for k in range(-(WIN_LEN - 1), WIN_LEN):
            cache[(dr, dc, k)] = _shift(plane, k * dr, k * dc)
    return cache


def _comp5(shift_cache: dict, empty: torch.Tensor) -> torch.Tensor:
    """Empty cells where placing a `plane` stone makes >=5 in a row (any dir).

    shift_cache is the cache of the `plane` whose completions we want.
    Returns (F, N, N) bool."""
    out = torch.zeros_like(empty)
    for (dr, dc) in DIRS:
        for jc in range(WIN_LEN):  # offset of the completion cell within the window
            m = empty.clone()
            for i in range(WIN_LEN):
                if i == jc:
                    continue
                r = i - jc
                m &= shift_cache[(dr, dc, r)]
            out |= m
    return out


def _four_structure(own_c: dict, empty: torch.Tensor, empty_c: dict,
                    idx_shift: dict):
    """Per-cell four-move analysis over the whole frontier.

    Returns (n_comp, block_acc):
      n_comp[F,N,N]  int  : number of DISTINCT five-completions if attacker plays
                            this (empty) cell -> 0 none, 1 single four, >=2 double.
      block_acc[F,N,N] long: for a single four, the flat index of its forced
                            block square (garbage where n_comp != 1).
    """
    F = empty.shape[0]
    n_comp = torch.zeros((F, N, N), dtype=torch.int16, device=empty.device)
    block_acc = torch.zeros((F, N, N), dtype=torch.long, device=empty.device)
    for (dr, dc) in DIRS:
        for delta in (-4, -3, -2, -1, 1, 2, 3, 4):
            jm_lo = max(0, -delta)
            jm_hi = min(WIN_LEN - 1, WIN_LEN - 1 - delta)
            if jm_lo > jm_hi:
                continue
            ec = empty_c[(dr, dc, delta)]          # completion cell empty
            fired = torch.zeros_like(empty)
            for jm in range(jm_lo, jm_hi + 1):
                pat = empty & ec
                for i in range(WIN_LEN):
                    r = i - jm
                    if r == 0 or r == delta:
                        continue
                    pat &= own_c[(dr, dc, r)]
                fired |= pat
            n_comp += fired.to(torch.int16)
            block_acc += fired.to(torch.long) * idx_shift[(dr, dc, delta)]
    return n_comp, block_acc


def _has5(shift_cache: dict) -> torch.Tensor:
    """(F,) bool: plane already contains a 5-in-a-row."""
    F = shift_cache[(0, 1, 0)].shape[0]
    dev = shift_cache[(0, 1, 0)].device
    out = torch.zeros(F, dtype=torch.bool, device=dev)
    for (dr, dc) in DIRS:
        run = shift_cache[(dr, dc, 0)].clone()
        for k in range(1, WIN_LEN):
            run &= shift_cache[(dr, dc, k)]
        out |= run.flatten(1).any(1)
    return out


@torch.no_grad()
def solve_vcf_batch(boards: np.ndarray, *, max_depth: int = 16,
                    max_nodes: int = 200_000, device: str | None = None,
                    max_frontier: int = 4_000_000):
    """Batched VCF verdict for B positions.

    boards: (B, 2, N, N) bool/int  (plane 0 attacker = side to move, plane 1 def).
    Returns (won, hit_cap): each (B,) numpy bool.  ``won[b]`` matches
    ``solve_vcf(boards[b]).has_forced_win`` whenever neither solver hits a cap.
    ``hit_cap[b]`` flags roots whose verdict was bounded by depth/frontier caps
    (a "no win" on such a root is unproven, never a false positive).
    """
    dev = device or pick_device()
    B = boards.shape[0]
    own = torch.as_tensor(boards[:, 0], dtype=torch.bool, device=dev)
    opp = torch.as_tensor(boards[:, 1], dtype=torch.bool, device=dev)
    root = torch.arange(B, dtype=torch.long, device=dev)

    won = torch.zeros(B, dtype=torch.bool, device=dev)
    hit_cap = torch.zeros(B, dtype=torch.bool, device=dev)

    # index grid + its directional shifts (shared across all frontier rows).
    idx_grid = (torch.arange(N, device=dev).view(N, 1) * N
                + torch.arange(N, device=dev).view(1, N)).view(1, N, N).long()
    idx_shift = {}
    for (dr, dc) in DIRS:
        for delta in (-4, -3, -2, -1, 1, 2, 3, 4):
            idx_shift[(dr, dc, delta)] = _shift(idx_grid, delta * dr, delta * dc)

    # Root special cases (mirror solve_vcf): defender already has five -> no win;
    # attacker already has five -> win (cannot happen for a legal side-to-move).
    own_c0 = _build_shift_cache(own)
    opp_c0 = _build_shift_cache(opp)
    def_has5 = _has5(opp_c0)
    atk_has5 = _has5(own_c0)
    won |= atk_has5
    dead_root = def_has5 | atk_has5  # already decided; drop from search

    keep = ~dead_root
    own, opp, root = own[keep], opp[keep], root[keep]

    nodes_total = 0
    depth = 1
    while own.shape[0] > 0 and depth <= max_depth:
        F = own.shape[0]
        nodes_total += F
        empty = ~(own | opp)
        own_c = _build_shift_cache(own)
        opp_c = _build_shift_cache(opp)
        empty_c = _build_shift_cache(empty)

        # attacker immediate-five (win now)
        atk5_cell = _comp5(own_c, empty)
        imm5 = atk5_cell.flatten(1).any(1)

        # defender immediate-five completions (soundness / forcing test)
        def5_cell = _comp5(opp_c, empty)
        def5_count = def5_cell.flatten(1).sum(1)                       # (F,)
        # sound[m] = defender has no five-completion other than m
        forced_ok = (def5_count.view(F, 1, 1) - def5_cell.to(torch.long)) == 0

        # four structure
        n_comp, block_acc = _four_structure(own_c, empty, empty_c, idx_shift)
        is_single = (n_comp == 1) & forced_ok
        is_double = (n_comp >= 2) & forced_ok

        node_win = imm5 | is_double.flatten(1).any(1)                  # (F,)

        # record wins on roots (scatter OR)
        if node_win.any():
            won_roots = root[node_win]
            won[won_roots] = True

        # mark roots still searching that we may have to truncate later
        live_root_mask = ~won[root]               # nodes whose root not yet won
        # children: sound single fours from non-winning nodes whose root is live
        gen_mask = live_root_mask & (~node_win)
        child_cells = is_single & gen_mask.view(F, 1, 1)

        flat = child_cells.reshape(F, -1)
        node_ids, mflat = flat.nonzero(as_tuple=True)   # (C,) host sync here
        C = node_ids.shape[0]
        if C == 0:
            break
        if C > max_frontier:
            # frontier blow-up: truncate. Roots of dropped nodes are unproven.
            hit_cap[root[node_ids[max_frontier:]]] = True
            node_ids = node_ids[:max_frontier]
            mflat = mflat[:max_frontier]
            C = max_frontier
        if nodes_total + C > max_nodes:
            hit_cap[root[node_ids.unique()]] = True
            break

        mr = (mflat // N)
        mc = (mflat % N)
        blk = block_acc.reshape(F, -1)[node_ids, mflat]
        br = blk // N
        bc = blk % N

        new_own = own[node_ids].clone()
        new_opp = opp[node_ids].clone()
        ar = torch.arange(C, device=dev)
        new_own[ar, mr, mc] = True
        new_opp[ar, br, bc] = True
        new_root = root[node_ids]

        # prune children whose root already won this round
        alive = ~won[new_root]
        own, opp, root = new_own[alive], new_opp[alive], new_root[alive]
        depth += 1

    # any roots still with frontier at depth-cap are unproven (no win, flagged)
    if own.shape[0] > 0:
        hit_cap[root.unique()] = True

    return won.cpu().numpy(), hit_cap.cpu().numpy()


# --------------------------------------------------------------------------- #
# Correctness + throughput harness
# --------------------------------------------------------------------------- #

def _gen_positions(n: int, seed: int = 0):
    """Short random games -> midgame boards (mix of forced-win and not)."""
    from gomoku.game import GameState
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n):
        s = GameState.initial()
        plies = int(rng.integers(8, 30))
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


def correctness_test(n: int = 500, seed: int = 0, max_depth: int = 16,
                     max_nodes: int = 200_000):
    from gomoku.vcf import solve_vcf
    boards = _gen_positions(n, seed)
    # CPU reference
    cpu = np.zeros(n, dtype=bool)
    cpu_cap = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i in range(n):
        r = solve_vcf(boards[i], max_depth=max_depth, max_nodes=max_nodes)
        cpu[i] = r.has_forced_win
        cpu_cap[i] = r.hit_cap
    cpu_dt = time.time() - t0

    gpu, gpu_cap = solve_vcf_batch(boards, max_depth=max_depth,
                                   max_nodes=max_nodes)

    clean = ~(cpu_cap | gpu_cap)
    agree = (cpu == gpu)
    n_clean = int(clean.sum())
    n_agree_clean = int((agree & clean).sum())
    mism = np.flatnonzero(~agree)
    print(f"--- CORRECTNESS (n={n}, depth={max_depth}) ---")
    print(f"CPU forced-wins: {int(cpu.sum())} / {n}   (cpu solve {cpu_dt:.2f}s, "
          f"{n / cpu_dt:.0f}/s)")
    print(f"CPU hit_cap: {int(cpu_cap.sum())}   GPU hit_cap: {int(gpu_cap.sum())}")
    print(f"Agreement (all):   {int(agree.sum())}/{n} = {agree.mean() * 100:.2f}%")
    print(f"Agreement (clean): {n_agree_clean}/{n_clean} = "
          f"{(n_agree_clean / max(1, n_clean)) * 100:.2f}%")
    if len(mism):
        print(f"Mismatches ({len(mism)}): showing up to 10")
        for i in mism[:10]:
            print(f"  pos {i}: cpu={cpu[i]} gpu={gpu[i]} "
                  f"cpu_cap={cpu_cap[i]} gpu_cap={gpu_cap[i]}")
    return boards, cpu, gpu, cpu_cap, gpu_cap


def throughput_test(batch_sizes=(256, 1024, 4096, 16384),
                    depths=(8, 16), seed=7):
    print("\n--- THROUGHPUT (GPU batched VCF full solves/sec) ---")
    dev = pick_device()
    print(f"device={dev}, board={N}x{N}")
    # build one big pool of positions, slice per batch
    pool = _gen_positions(max(batch_sizes), seed)
    for depth in depths:
        print(f"\n max_depth={depth}")
        print(f" {'batch':>7} {'solves/s':>12} {'speedup vs 53/s':>16} {'wins%':>7}")
        for B in batch_sizes:
            boards = pool[:B] if B <= pool.shape[0] else \
                np.concatenate([pool] * (B // pool.shape[0] + 1))[:B]
            # warmup
            w, _ = solve_vcf_batch(boards, max_depth=depth, device=dev)
            if dev == "mps":
                torch.mps.synchronize()
            reps = 3
            t0 = time.time()
            for _ in range(reps):
                w, _ = solve_vcf_batch(boards, max_depth=depth, device=dev)
            if dev == "mps":
                torch.mps.synchronize()
            dt = (time.time() - t0) / reps
            sps = B / dt
            print(f" {B:>7} {sps:>12.0f} {sps / 53:>15.0f}x "
                  f"{w.mean() * 100:>6.1f}")


if __name__ == "__main__":
    import sys
    if N != 15:
        print(f"WARNING: board size is {N}, expected 15 "
              f"(set GOMOKU_BOARD_SIZE=15)", file=sys.stderr)
    correctness_test(n=500, seed=0)
    throughput_test()
