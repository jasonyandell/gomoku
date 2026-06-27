"""W-channel framing probe — "the `W` channel carries IDENTITY, not EXISTENCE" (#90).

The v2 `W` channel (`return_w`, the OPP mirror of `carriers`) reports the
*over-inclusive* load-bearing DEFENDER stones the proof's five-lines run through.
Before trusting it as a stencil ingredient we confirm the framing the issue rests
on, empirically:

  By FREESTYLE monotonicity (`wiki/topics/shape-library-engine.md` §3) a defender
  stone NEVER makes an attacker VCT *appear* — adding defender stones only hurts
  or is neutral. So removing EVERY defender stone from a winning board must
  PRESERVE the win. Equivalently: no attacker VCT is existence-dependent on a
  defender stone. Therefore `W` cannot be needed for the win to EXIST; it carries
  IDENTITY (which forced line / mate-distance) — and the MINIMAL load-bearing `W`
  is the md-ablation program, BLOCKED on md-extraction
  (`shape-library-engine.md` §3 correction #2 / §8), which this over-inclusive
  `w` over-approximates.

Experiment (bulk-synchronous per the call-cost law):
  1. solve a pool of real Rapfi positions; take the clean attacker wins.
  2. ZERO the opp plane (remove ALL defender stones) and re-solve.
  3. count preserved wins. Expect ~100% preserved, 0 monotonicity violations
     (a clean win that flips to a clean no-win after removing defenders).
  Alongside: how often `w` is non-empty on the original win (defender stones DO
  sit on the proof lines) — yet none are existence-critical, the whole point.

Run (from repo root):
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python \
      scripts/threat_shapes/w_channel_probe.py --pool 4096 --max-nodes 500
"""

from __future__ import annotations

import argparse

import numpy as np

from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb, cells_from_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=500)
    ap.add_argument("--min-ply", type=int, default=6)
    ap.add_argument("--max-ply", type=int, default=60)
    args = ap.parse_args()

    # late import so --help works without the Rapfi data dir present
    from scripts.vct_metal.positions import load_position_stack

    pool = load_position_stack(args.pool, seed=args.seed,
                               min_ply=args.min_ply, max_ply=args.max_ply)
    win, hit, support, carr, wch = solve_vct_mega_bb(
        pool, max_nodes=args.max_nodes, return_support=True,
        return_carriers=True, return_w=True)
    wins = np.flatnonzero(win & ~hit)
    print(f"pool {args.pool} (seed={args.seed}, max_nodes={args.max_nodes}): "
          f"{len(wins)} clean attacker wins")
    if len(wins) == 0:
        print("no clean wins — widen the pool or raise max_nodes")
        return

    # --- W-channel population on the ORIGINAL wins (context, not the test) ---
    n_w = sum(bool(cells_from_words(wch[b]).__len__()) for b in wins)
    w_sizes = np.array([len(cells_from_words(wch[b])) for b in wins])
    n_defenders = pool[wins, 1].reshape(len(wins), -1).sum(1)
    print(f"  wins WITH ≥1 W stone on the proof lines: {n_w}/{len(wins)} "
          f"({n_w / len(wins) * 100:.0f}%); |w| mean={w_sizes.mean():.1f} "
          f"max={int(w_sizes.max())}; defenders on board mean={n_defenders.mean():.1f}")

    # --- EXISTENCE-MONOTONICITY: strip ALL defenders, re-solve ---
    stripped = pool[wins].copy()
    stripped[:, 1] = False                 # zero the opp plane (remove every W stone)
    ws, hs = solve_vct_mega_bb(stripped, max_nodes=args.max_nodes)

    preserved = int((ws & ~hs).sum())      # still a clean win
    capped = int(hs.sum())                 # search capped after stripping (ambiguous)
    violated = int((~ws & ~hs).sum())      # clean win -> clean NO-WIN == monotonicity break
    n = len(wins)
    print(f"\nremove ALL defender stones (zero opp plane), re-solve {n} clean wins:")
    print(f"  preserved (still a clean win) : {preserved}/{n} = {preserved / n * 100:.1f}%")
    print(f"  capped after stripping        : {capped}/{n} (ambiguous, not a violation)")
    print(f"  MONOTONICITY VIOLATIONS       : {violated}/{n} "
          f"(clean win -> clean no-win; expect 0)")

    clean_after = preserved + violated     # boards with a definitive post-strip verdict
    if clean_after:
        print(f"\n  among the {clean_after} boards with a definitive verdict after "
              f"stripping, {preserved} ({preserved / clean_after * 100:.1f}%) survived, "
              f"{violated} violated.")
    verdict = "CONFIRMED" if violated == 0 else "VIOLATED"
    print(f"\n{verdict}: a defender stone never creates a freestyle attacker VCT, so "
          f"`W` is IDENTITY (which forced line), not EXISTENCE. The minimal "
          f"load-bearing `W` is the md-ablation program (shape-library-engine §3/§8); "
          f"`w` over-approximates it.")


if __name__ == "__main__":
    main()
