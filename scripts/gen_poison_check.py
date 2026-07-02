"""Gen-semantics poison detector (#107 guardrail — run after ANY change to
generation semantics).
Gen at LIVE config (MPS, sims=100, wave=32, trained weights), then re-solve
every recorded position at full breadth and verify the veto invariant:
recorded pi must have ZERO mass on proven-blunder cells. argv[2]='overlap'
enables the overlap; argv[3]=seed; argv[4]=concurrent width (0 = legacy
lockstep; >0 exercises the continuous-refill path, issue #112, with 4x
that many total games)."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from gomoku import self_play as sp
from gomoku.mcts import make_torch_evaluator
from gomoku.model import load_checkpoint

if len(sys.argv) < 2:
    raise SystemExit("usage: gen_poison_check.py <checkpoint.pt> [overlap] [seed] [concurrent]")
overlap = len(sys.argv) > 2 and sys.argv[2] == "overlap"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
concurrent = int(sys.argv[4]) if len(sys.argv) > 4 else 0
model, _ = load_checkpoint(sys.argv[1], device="mps")
model.eval()
evaluator = make_torch_evaluator(model, "mps")
sp.configure_vct_terminus(enabled=True, budget=50)
sp.configure_oracle_veto(enabled=True)
if hasattr(sp, 'configure_oracle_overlap'):
    sp.configure_oracle_overlap(enabled=overlap)

rng = np.random.default_rng(seed)
n_games = 4 * concurrent if concurrent > 0 else 8
records = sp.generate_games(n_games, evaluator, n_simulations=100, wave_size=32,
                            rng=rng, temperature_moves=30,
                            augment_symmetries=False,
                            concurrent_games=concurrent)
print(f"overlap={overlap} seed={seed} concurrent={concurrent} "
      f"games={len(records)} "
      f"plies={[r.plies for r in records]} outcomes={[r.outcome for r in records]}")

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
print(f"VIOLATIONS: {bad}/{checked} positions with blunder mass"
      + (f" (worst mass={worst[0]:.3f} side={worst[1]} ply={worst[2]})" if worst else ""))
