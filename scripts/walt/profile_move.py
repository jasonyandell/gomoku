"""Profile one walt move: uv run python scripts/walt/profile_move.py K H M"""
import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walt_gomoku2 import VCT, Field, WaltConfig, WaltPlayer  # noqa: E402

from gomoku.game import GameState  # noqa: E402
from gomoku.mcts import make_torch_evaluator  # noqa: E402
from gomoku.model import fuse_model_for_inference, load_checkpoint  # noqa: E402
from gomoku.self_play import _random_opening_state  # noqa: E402
from gomoku.util import pick_device  # noqa: E402

K, H, M = (int(x) for x in sys.argv[1:4])
CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 0
d = pick_device(None)
m, _ = load_checkpoint("/Users/jason/data/sound-world-107b/sweep_runs/sound-world/checkpoints/worker_weights.pt", device=d)
ev = make_torch_evaluator(fuse_model_for_inference(m), d)
ev([GameState.initial()])
vct = VCT(50)
rng = np.random.default_rng(3)
s, _ = _random_opening_state(rng, 8)
w = WaltPlayer(WaltConfig(n_tapes=K, horizon=H, n_cand=M, rollout_cap=CAP), Field("net", ev, 1.0), ev, vct)
pr = cProfile.Profile()
pr.enable()
t = time.perf_counter()
w(s, rng)
dt = time.perf_counter() - t
pr.disable()
L = w.log[-1]
print("secs", round(dt, 2), {k: L[k] for k in ("kind", "leaves", "rollouts", "batch_max", "rollout_plies", "vct_calls",
                                               "vct_secs", "rollout_secs") if k in L})
o = io.StringIO()
pstats.Stats(pr, stream=o).sort_stats("tottime").print_stats(12)
print(o.getvalue()[-2500:])
