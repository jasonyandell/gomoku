"""Falsification harness for the stencil CERTIFICATE property (#88, 2026-06-27).

Claim under test (the "do we have a proof?" question): a VCT stencil that wins **in
isolation** — its carrier stones alone on an empty board, support cells empty — is a
forced win on ANY board where it fits and the defender has no counter-threat, by the
SAME forcing line. This is the soundness of Allis's dependency-based / threat-space
search (1994) made operational on our GPU-mined stencils; here we try to BREAK it.

Pipeline (all bulk-synchronous per the call-cost law):
  1. mine clean attacker VCTs from a pool of real positions (support + carriers).
  2. self-containment: does each win reproduce from carriers ALONE on an empty board
     (support empty, nothing else)?  -> tests that (carriers, support) is a complete
     offensive description.
  3. transfer: bolt random opponent stones (off support & carriers) onto each
     self-contained shape; keep boards where the defender has NO VCT of its own
     (tempo-safe); re-solve.  A refutation = a tempo-safe board where the attacker
     no longer wins = a counterexample that sharpens the exact defender condition.
  Control: boards where the defender DOES have a VCT should refute some -> confirms
  counter-tempo is the (only) breaker, not a vacuous filter.

Run (from repo root):
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python \
      scripts/threat_shapes/certificate_falsification.py --pool 4096 --max-nodes 500
"""

from __future__ import annotations

import argparse

import numpy as np

from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb, cells_from_words, N


def _iso_board(carrier_cells):
    """A board with only the carrier stones as own (attacker to move), else empty."""
    bd = np.zeros((2, N, N), bool)
    for c in carrier_cells:
        bd[0, c // N, c % N] = True
    return bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=500)
    ap.add_argument("--min-ply", type=int, default=6)
    ap.add_argument("--max-ply", type=int, default=60)
    ap.add_argument("--per-shape", type=int, default=8, help="perturbations per shape")
    ap.add_argument("--n-opp", type=int, default=12, help="opponent stones bolted on")
    ap.add_argument("--cap", type=int, default=400, help="max self-contained shapes tested")
    args = ap.parse_args()

    # late import so --help works without the data dir present
    from scripts.vct_metal.positions import load_position_stack
    rng = np.random.default_rng(args.seed)

    pool = load_position_stack(args.pool, seed=args.seed,
                               min_ply=args.min_ply, max_ply=args.max_ply)
    win, hit, move, supp, carr = solve_vct_mega_bb(
        pool, max_nodes=args.max_nodes, return_move=True,
        return_support=True, return_carriers=True)
    wins = np.flatnonzero(win & ~hit)
    print(f"pool {args.pool}: {len(wins)} clean attacker wins")

    # STEP 1 — self-containment
    iso = np.stack([_iso_board(cells_from_words(carr[b])) for b in wins])
    wi, hi = solve_vct_mega_bb(iso, max_nodes=args.max_nodes)
    sc = wins[wi & ~hi]
    print(f"self-contained (carriers alone on empty board still win): "
          f"{len(sc)}/{len(wins)} = {len(sc) / max(len(wins), 1) * 100:.0f}%")
    sc = sc[:args.cap]

    # STEP 2 — bolt random opponent stones off the shape
    pert, owner = [], []
    for b in sc:
        forbidden = set(cells_from_words(supp[b])) | set(cells_from_words(carr[b]))
        base = _iso_board(cells_from_words(carr[b]))
        empties = [c for c in range(N * N) if c not in forbidden]
        for _ in range(args.per_shape):
            bd = base.copy()
            for c in rng.choice(empties, size=min(args.n_opp, len(empties)), replace=False):
                bd[1, c // N, c % N] = True
            pert.append(bd)
            owner.append(int(b))
    if not pert:
        print("no self-contained shapes to perturb")
        return
    pert = np.stack(pert)
    owner = np.array(owner)

    # STEP 3 — classify: defender counter-VCT? ; attacker still wins?
    wf, hf = solve_vct_mega_bb(pert[:, [1, 0]].copy(), max_nodes=args.max_nodes)  # defender
    wa, ha = solve_vct_mega_bb(pert, max_nodes=args.max_nodes)                    # attacker
    safe = ~wf & ~hf      # defender has NO VCT of its own (tempo-safe, conservative)
    clean = ~ha           # attacker verdict not capped
    tested = safe & clean
    preserved = int((wa & tested).sum())
    refuted = int((~wa & tested).sum())
    unsafe = (~safe) & clean

    print(f"\nperturbed boards: {len(pert)}  ({len(sc)} shapes x {args.per_shape})")
    print(f"tempo-safe (defender has no VCT) & clean: {int(tested.sum())}")
    print(f"  attacker STILL WINS : {preserved}")
    print(f"  attacker REFUTED    : {refuted}   <-- counterexamples to the theorem")
    print(f"control (defender HAS a VCT): attacker wins {int((wa & unsafe).sum())}"
          f"/{int(unsafe.sum())}  (expect some losses = counter-tempo is the breaker)")

    if refuted:
        for idx in np.flatnonzero((~wa) & tested)[:3]:
            b = owner[idx]
            opp = [c for c in range(N * N) if pert[idx, 1, c // N, c % N]]
            print(f"\n--- counterexample (shape from board {b}) ---")
            print("  carriers:", cells_from_words(carr[b]))
            print("  support :", cells_from_words(supp[b]))
            print("  added opp:", opp)


if __name__ == "__main__":
    main()
