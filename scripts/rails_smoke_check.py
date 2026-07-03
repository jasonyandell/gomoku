"""Rails-v0 gen-semantics smoke + poison check (issue #116).

Exercises the ACTUAL rails-v0 live config with the real MLX oracle:
  veto ON, attacker-preserve ON, vct-terminus OFF (play-on), board 15, fresh
  small line-planes net. Asserts two invariants:
  (A) VETO poison: no recorded pi carries mass on a proven-blunder cell
      (re-solve every recorded position at full breadth; the #107 guardrail).
  (B) ATTACKER-PRESERVE soundness: for every recorded position whose mover had
      a CLEAN proven VCT, the recorded pi mass sits ONLY on winning first moves
      (the winmask) -- the net's target was restricted to win-preserving moves.

Run:  GOMOKU_BOARD_SIZE=15 uv run python scripts/rails_smoke_check.py [seed] [concurrent]
No <ckpt> needed -- builds a fresh net (the run itself is from-scratch).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from gomoku import self_play as sp
from gomoku.game import N_ACTIONS
from gomoku.mcts import make_torch_evaluator
from gomoku.model import build_model
from scripts.vct_metal.mega_vct_bb import cells_from_words, solve_vct_mega_bb

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
concurrent = int(sys.argv[2]) if len(sys.argv) > 2 else 0
budget = int(os.environ.get("GOMOKU_POISON_BUDGET", "25"))

model = build_model("small", stem_padding=1, line_planes=True, global_pool=True)
model = model.to("mps")
model.eval()
evaluator = make_torch_evaluator(model, "mps")

# The rails-v0 config: NO terminus (play-on), veto + attacker-preserve ON.
sp.configure_vct_terminus(enabled=False, budget=budget)
sp.configure_oracle_veto(enabled=True)
sp.configure_attacker_preserve(enabled=True)
sp.configure_oracle_overlap(enabled=True)

rng = np.random.default_rng(seed)
n_games = 4 * concurrent if concurrent > 0 else 8
records = sp.generate_games(
    n_games, evaluator, n_simulations=100, wave_size=32, rng=rng,
    temperature_moves=30, augment_symmetries=False,
    fixed_openings=True, concurrent_games=concurrent)

print(f"budget={budget} seed={seed} concurrent={concurrent} games={len(records)}")
print(f"plies={[r.plies for r in records]}")
print(f"outcomes={[r.outcome for r in records]}")

# (A) veto poison invariant
bad = checked = 0
worst = None
for rec in records:
    for ex in rec.examples:
        vmaps, _ = sp._vct_defense_solve([np.asarray(ex.planes)], max_cands=0)
        mass = float(np.asarray(ex.pi)[vmaps[0] >= 0.5].sum())
        checked += 1
        if mass > 1e-6:
            bad += 1
            if worst is None or mass > worst[0]:
                worst = (mass, ex.side, ex.ply)
print(f"(A) VETO VIOLATIONS: {bad}/{checked} positions with blunder mass"
      + (f" (worst mass={worst[0]:.3f} side={worst[1]} ply={worst[2]})" if worst else ""))

# (B) attacker-preserve invariant: batch-solve every recorded position's root in
# complete mode; where the mover had a CLEAN win, recorded pi mass must be 100%
# on the winmask. (Off-mass = a move that lets the forced win slip -> a leak.)
all_planes = [np.asarray(ex.planes) for rec in records for ex in rec.examples]
all_pi = [np.asarray(ex.pi) for rec in records for ex in rec.examples]
arrs = [np.asarray(p) for p in all_planes]
boards = sp._terminus_boards(arrs)
win, hit, _mv, winmask = solve_vct_mega_bb(
    boards, max_nodes=budget, complete=True, return_move=True)
win = np.asarray(win).astype(bool)
hit = np.asarray(hit).astype(bool)
winmask = np.asarray(winmask)
clean = win & ~hit
leak = clean_wins = 0
worst_leak = 0.0
for i in np.flatnonzero(clean):
    cells = np.asarray(cells_from_words(winmask[i]), dtype=np.int64)
    m = np.zeros(N_ACTIONS, dtype=bool)
    if cells.size:
        m[cells] = True
    off_mass = float(all_pi[i][~m].sum())
    clean_wins += 1
    if off_mass > 1e-6:
        leak += 1
        worst_leak = max(worst_leak, off_mass)
print(f"(B) PRESERVE: {clean_wins} clean-win positions recorded; "
      f"LEAKS (pi mass off winmask): {leak}"
      + (f" (worst off-mass={worst_leak:.3f})" if leak else ""))
print("SMOKE OK" if (bad == 0 and leak == 0) else "SMOKE FAILED")
