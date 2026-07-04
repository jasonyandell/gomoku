"""On-path walk + 1-ply off-path fan over Rapfi games — what the neighborhood of a strong
game reveals about VCT structure. (Synthesis: wiki/topics/vct-reachability-mining.md.)

THE METHOD. Ride each Rapfi-vs-Rapfi game. At KNOWN-non-VCT on-path nodes in the pre-onset
band (onset = first ply the side-to-move has a proven VCT), fan EVERY alternative move the
side did NOT play, and solve VCT on each resulting board. NO recursion (one ply off-path,
re-anchored on the real game each step).

THE FRAMING (the load-bearing subtlety). A VCT belongs to the SIDE TO MOVE. After side S
plays an alternative m, it is the OPPONENT's turn, so a VCT found on the fanned board is the
OPPONENT's forced win — i.e. m is a forced-LOSING move for S. The fan is therefore a
DEFENSE/blunder detector + a VCT-board MINER, never an offense ("S missed a win") detector
(S would need its own next turn). Both seats are strong Rapfi; the losing moves are
counterfactuals WE inject, never moves a player chose. Integrity-checked here: every fanned
NODE must itself be non-VCT (side-to-move had no forced win), and stm stones <= opp stones.

WHAT IT MEASURES (this script reproduces the 2026-06-26 findings):
  * knife-edge sharpness: fraction of a side's alternatives that lose by force, by who's to
    move (VCT-holder vs opponent) x distance-to-onset. (~80% near onset; pre-onset is NOT a
    forgiving region — the seek-VCT thesis update.)
  * triviality split of the VCT-wins via the fast VCF kernel: VCF (four-driven, ~trivial) vs
    non-VCF VCT (needs a three = the combinational GOLD). VCF subseteq VCT, so VCF is run only
    on VCT-wins. The non-VCF gold concentrates on the WINNER's wins (defender-perturbed nodes).

Run (no contention with the CPU rapfi fleet — the kernels are Metal/GPU):
  GOMOKU_BOARD_SIZE=15 nice -n 10 uv run python -m scripts.threat_shapes.vct_fan \
      --max-shards 30 --max-nodes-sampled 2500 --pre-band 6 --mn 500
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import defaultdict

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from scripts.threat_shapes.read_games import iter_games
from scripts.threat_shapes.mine_first_vct import all_boards
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb
from scripts.vct_metal.mega_vcf_bb import solve_vcf_mega_bb

NN = N * N
GPU_BATCH = 16384


def load_verdicts(puzzles, sel, sid_of):
    """win/cap plies per game (sid,idx) for the selected shards (no re-solve)."""
    win, cap = defaultdict(set), defaultdict(set)
    with gzip.open(puzzles, "rt") as f:
        for line in f:
            if not line:
                continue
            r = json.loads(line)
            sh = r["shard"]
            if sh not in sel:
                continue
            k = (sid_of[sh], int(r["idx"])); p = int(r["ply"])
            if r["win"] and not r["cap"]:
                win[k].add(p)
            elif r["cap"]:
                cap[k].add(p)
    return win, cap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles", default=os.path.expanduser("~/data/puzzle_miner/puzzles.jsonl.gz"))
    ap.add_argument("--manifest", default=os.path.expanduser("~/data/puzzle_miner/manifest.txt"))
    ap.add_argument("--games-dir", default=os.path.expanduser("~/data/games_raphi"))
    ap.add_argument("--max-shards", type=int, default=30)
    ap.add_argument("--max-nodes-sampled", type=int, default=2500, help="cap fanned NODES")
    ap.add_argument("--pre-band", type=int, default=6, help="fan nodes within this many plies before onset")
    ap.add_argument("--mn", type=int, default=500, help="solver max_nodes (matches the miner budget)")
    ap.add_argument("--side", choices=["both", "defender", "attacker"], default="both",
                    help="defender=opponent-to-move nodes (mine the winner's VCTs); "
                         "attacker=VCT-holder-to-move nodes")
    args = ap.parse_args()

    manifest = [l.strip() for l in open(args.manifest) if l.strip()][:args.max_shards]
    sel = set(manifest); sid_of = {b: i for i, b in enumerate(manifest)}
    win, cap = load_verdicts(args.puzzles, sel, sid_of)
    t0 = time.time()

    # ---- collect fanned boards from pre-onset non-VCT nodes
    boards_parts, dist_parts, node_boards = [], [], []
    node_count = parity_bad = 0
    stop = False
    for base in manifest:
        if stop:
            break
        sid = sid_of[base]; sp = os.path.join(args.games_dir, base)
        if not os.path.exists(sp):
            continue
        for gi, g in enumerate(iter_games(args.games_dir, [sp])):
            moves = g.get("moves", [])
            if any((not isinstance(m, int)) or m < 0 or m >= NN for m in moves):
                continue
            k = (sid, gi); W = win.get(k, set())
            if not W:
                continue
            onset = min(W); Cg = cap.get(k, set()); bs = all_boards(moves)
            if bs is None:
                continue
            for p in range(max(0, onset - args.pre_band), onset):
                if p in W or p in Cg or p >= len(moves):
                    continue
                holder = (p % 2 == onset % 2)            # this side will own the first VCT
                if args.side == "defender" and holder:    # defender = opponent-to-move node
                    continue
                if args.side == "attacker" and not holder:
                    continue
                b = bs[p]; S = b[0]; opp = b[1]
                if int(S.sum()) > int(opp.sum()):
                    parity_bad += 1
                node_boards.append(b.astype(bool))
                alts = np.flatnonzero((~(S | opp)).reshape(-1)); alts = alts[alts != moves[p]]
                A = len(alts)
                if A == 0:
                    continue
                s1 = np.empty((A, 2, N, N), dtype=bool)
                s1[:, 0] = opp; s1[:, 1] = S
                s1[:, 1].reshape(A, -1)[np.arange(A), alts] = True
                boards_parts.append(s1); dist_parts.append(np.full(A, onset - p, dtype=np.int8))
                node_count += 1
                if node_count >= args.max_nodes_sampled:
                    stop = True; break
            if stop:
                break

    boards = np.concatenate(boards_parts); dist = np.concatenate(dist_parts); n = len(boards)
    print(f"[fan] side={args.side} nodes={node_count} fanned={n} (collect {time.time()-t0:.0f}s)", flush=True)

    # ---- integrity: every NODE must be non-VCT; parity sane
    nb = np.stack(node_boards)
    node_vct = np.zeros(len(nb), bool)
    for s in range(0, len(nb), GPU_BATCH):
        w, _ = solve_vct_mega_bb(nb[s:s + GPU_BATCH], max_nodes=args.mn)
        node_vct[s:s + len(w)] = np.asarray(w, bool)
    print(f"[integrity] NODE-is-VCT rate={100*node_vct.mean():.3f}% (MUST be ~0)  "
          f"parity_violations={parity_bad} (MUST be 0)", flush=True)

    # ---- VCT on all fanned boards
    vct_win = np.zeros(n, bool); vct_cap = np.zeros(n, bool)
    ts = time.time()
    for s in range(0, n, GPU_BATCH):
        w, h = solve_vct_mega_bb(boards[s:s + GPU_BATCH], max_nodes=args.mn)
        vct_win[s:s + len(w)] = np.asarray(w, bool); vct_cap[s:s + len(w)] = np.asarray(h, bool)
    nw = int(vct_win.sum())
    print(f"\n[verdicts] (n={n}, mn={args.mn}, {time.time()-ts:.0f}s)  "
          f"VCT={100*nw/n:.1f}%  cap={100*vct_cap.mean():.1f}%  clear={100*(~vct_win&~vct_cap).mean():.1f}%",
          flush=True)

    print("\n=== knife-edge: fraction of a side's alternatives that lose by force, who x dist ===")
    print(f"{'dist':>5} | {'VCT-holder-to-move':>20} | {'opponent-to-move':>20}")
    for d in range(1, args.pre_band + 1):
        cells = []
        for parity_is_holder in (True, False):
            # who-to-move at the node: holder iff dist is even (p%2==onset%2 <=> (onset-p)%2==0)
            want_even = parity_is_holder
            m = (dist == d) if ((d % 2 == 0) == want_even) else np.zeros(n, bool)
            cells.append(f"{100*(vct_win[m]).mean():5.1f}% (n={int(m.sum())})" if m.any() else "   -   ")
        print(f"{d:>5} | {cells[0]:>20} | {cells[1]:>20}")

    # ---- VCF only on the VCT-wins -> triviality split
    widx = np.flatnonzero(vct_win)
    vcf_win = np.zeros(n, bool); vcf_cap = np.zeros(n, bool)
    ts = time.time()
    for s in range(0, len(widx), GPU_BATCH):
        idx = widx[s:s + GPU_BATCH]
        w, h = solve_vcf_mega_bb(boards[idx], max_nodes=args.mn)
        vcf_win[idx] = np.asarray(w, bool); vcf_cap[idx] = np.asarray(h, bool)
    n_vcf = int(vcf_win.sum())
    nonvcf = vct_win & ~vcf_win & ~vcf_cap
    n_nonvcf = int(nonvcf.sum()); n_unk = int((vct_win & ~vcf_win & vcf_cap).sum())
    print(f"\n=== triviality split of the {nw} VCT-wins (VCF on wins, {time.time()-ts:.0f}s) ===")
    print(f"  VCF (four-driven, trivial-ish):    {n_vcf:>8} ({100*n_vcf/max(nw,1):.1f}%)")
    print(f"  NON-VCF VCT (needs a three, GOLD): {n_nonvcf:>8} ({100*n_nonvcf/max(nw,1):.1f}%)")
    print(f"  VCF-capped (unknown):              {n_unk:>8} ({100*n_unk/max(nw,1):.1f}%)")
    print(f"  non-VCF VCT as share of ALL fanned: {100*n_nonvcf/n:.2f}%")
    print("  non-VCF rate among VCT-wins by dist (odd=winner's win, even=opponent's):")
    for d in range(1, args.pre_band + 1):
        m = (dist == d) & vct_win
        if m.any():
            print(f"    dist {d} ({'opp-to-move' if d % 2 else 'holder-to-move':>14}): "
                  f"{100*nonvcf[m].mean():5.1f}% of {int(m.sum())} wins")
    print(f"[fan] DONE total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
