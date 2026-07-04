"""Build a corpus of NON-TRIVIAL VCT-positive 15x15 positions.

Funnel (uses the M5 generously; VCF is ~free):
  1. GENERATE dense midgame positions with a 'gravity' random policy (moves
     biased to land next to existing stones -> real threat geometry, unlike
     uniform-random scatter which is mostly empty board).
  2. GPU pre-filter every position with ``solve_vcf_batch`` (MPS, ~10k/s).
  3. STREAM A (cheap, plentiful): VCF-positive positions whose mate needs >= 2
     attacker moves (md==1 is a trivial immediate double-four -- discarded; those
     have no chain/shape structure). Tagged 'vcf'.
  4. STREAM B (time-boxed): VCF-NEGATIVE positions that have a forcing three
     (the precondition for a continuous-threes attack) -> run CPU ``solve_vct``;
     keep the wins. Tagged 'vct'. These are the genuinely non-VCF three-mates.

Saves boards + winning moves + tags + mate distances to an .npz, plus a stats
JSON. Run:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.build_corpus \
      --out <dir> --target-a 2500 --b-seconds 120
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from gomoku.game import GameState
from gomoku.vcf import (
    _candidate_cells_from_planes,
    _collinear_empties,
    _completions_through,
    _empties_from_plane,
    _open_four_threats,
    solve_vcf,
    solve_vct,
)
from scripts.gpu_vcf_prototype import solve_vcf_batch

N = GameState.initial().board.shape[-1]


def gen_gravity(n: int, seed: int, lo: int = 12, hi: int = 46,
                near_p: float = 0.85) -> np.ndarray:
    """n dense midgame boards via near-stone-biased random play."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = GameState.initial()
        plies = int(rng.integers(lo, hi))
        for _ in range(plies):
            la = s.legal_actions()
            if len(la) == 0:
                break
            atk, dfd = s.board[0], s.board[1]
            occ = atk | dfd
            pool = la
            if occ.any() and rng.random() < near_p:
                cc = _candidate_cells_from_planes(atk, dfd, _empties_from_plane(~occ))
                pool = cc if len(cc) > 0 else la
            s = s.apply(int(rng.choice(pool)))
            if s.is_terminal()[0]:
                break
        out.append(np.asarray(s.board, dtype=bool))
    return np.stack(out)


def has_forcing_three(board: np.ndarray) -> bool:
    """Cheap test: attacker has a move that makes a forcing three (not a four)."""
    atk = board[0].copy()
    dfd = board[1]
    occ = atk | dfd
    empty = ~occ
    for m in _candidate_cells_from_planes(atk, dfd, _empties_from_plane(empty)):
        mr, mc = int(m) // N, int(m) % N
        atk[mr, mc] = True
        occ2 = atk | dfd
        if _completions_through(atk, int(m), occ2):
            atk[mr, mc] = False
            continue                      # it's a four, not a three
        ne = ~occ2
        th = _open_four_threats(atk, dfd, ne, _collinear_empties(int(m), ne))
        atk[mr, mc] = False
        if th:
            return True
    return False


def build(out_dir: str, target_a: int, b_seconds: float, seed0: int = 1000,
          chunk: int = 2500, max_chunks: int = 60):
    t_start = time.time()
    boards, moves, tags, mds = [], [], [], []
    n_gen = n_vcfpos = 0
    seed = seed0
    # ---- Stream A (GPU-only non-trivial filter) + harvest pool for B ----
    # A VCF win that is NOT a win at attacker-depth 1 needs >= 2 attacker moves:
    # depth-1 wins are exactly the trivial immediate double-fours/fives. So
    #   nontrivial = won(depth16) & ~won(depth1)
    # is computed entirely on the GPU -- no slow CPU solve_vcf in the funnel.
    b_pool = []
    while len(boards) < target_a and (seed - seed0) < max_chunks:
        b = gen_gravity(chunk, seed)
        seed += 1
        n_gen += len(b)
        won16, cap16 = solve_vcf_batch(b, max_depth=16)
        won1, _ = solve_vcf_batch(b, max_depth=1)
        n_vcfpos += int(won16.sum())
        nontrivial = won16 & ~won1
        for i in np.flatnonzero(nontrivial):
            boards.append(b[i]); moves.append(-1)
            tags.append("vcf"); mds.append(-1)
        if len(b_pool) < 4000:
            for i in np.flatnonzero(~won16 & ~cap16):
                if has_forcing_three(b[i]):
                    b_pool.append(b[i])
                    if len(b_pool) >= 4000:
                        break
    n_a = len(boards)
    t_a = time.time() - t_start

    # ---- Stream B: time-boxed VCT over the forcing-three pool ----
    t_b0 = time.time()
    nb = 0
    for bd in b_pool:
        if time.time() - t_b0 > b_seconds:
            break
        r = solve_vct(bd, max_depth=7, max_nodes=12000)
        if r.has_forced_win and r.winning_move is not None:
            boards.append(bd); moves.append(r.winning_move)
            tags.append("vct"); mds.append(r.mate_distance)
            nb += 1
    t_b = time.time() - t_b0

    boards = np.stack(boards).astype(bool)
    moves = np.array(moves, dtype=np.int64)
    tags = np.array(tags)
    mds = np.array([m if m is not None else -1 for m in mds], dtype=np.int64)
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "corpus.npz"),
                        boards=boards, moves=moves, tags=tags, mds=mds)
    stats = {
        "n_total": int(len(boards)), "n_vcf": int(n_a), "n_vct": int(nb),
        "n_generated": int(n_gen), "n_vcfpos_raw_d16": int(n_vcfpos),
        "b_pool": int(len(b_pool)),
        "md_hist": {int(k): int(v) for k, v in
                    zip(*np.unique(mds, return_counts=True))},
        "t_streamA_s": round(t_a, 1), "t_streamB_s": round(t_b, 1),
        "t_total_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "corpus_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-a", type=int, default=2500)
    ap.add_argument("--b-seconds", type=float, default=120)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()
    build(args.out, args.target_a, args.b_seconds, args.seed0)
