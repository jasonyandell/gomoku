"""Bounded walt-on-gomoku matches. One process = one (player, opponent) pairing.

  uv run python scripts/walt/run_walt.py --player walt-net --opp heuristic --games 10 --max-wall-secs 900
Writes one JSON line per game + a summary line to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walt_gomoku import VCT, Field, WaltConfig, WaltPlayer  # noqa: E402

from gomoku.baselines import heuristic_player, lookahead_player  # noqa: E402
from gomoku.eval import mcts_picker  # noqa: E402
from gomoku.game import GameState  # noqa: E402
from gomoku.mcts import make_torch_evaluator  # noqa: E402
from gomoku.model import fuse_model_for_inference, load_checkpoint  # noqa: E402
from gomoku.self_play import _random_opening_state  # noqa: E402
from gomoku.util import pick_device  # noqa: E402

CHAMP = "/Users/jason/data/sound-world-107b/sweep_runs/sound-world/checkpoints/worker_weights.pt"


class Timed:
    """Wrap a picker to record per-move seconds."""

    def __init__(self, picker, label):
        self.picker, self.label, self.secs = picker, label, []

    def __call__(self, s, rng):
        t = time.perf_counter()
        a = self.picker(s, rng)
        self.secs.append(time.perf_counter() - t)
        return a


def build(name, ev, vct, args, seed):
    """Player/opponent factory. Returns (picker, walt_or_None)."""
    if name == "heuristic":
        return heuristic_player, None
    if name == "lookahead4":
        return lookahead_player(depth=4), None
    if name.startswith("mcts"):  # mcts<sims>: the champion product (net + MCTS + cap50 finisher)
        sims = int(name[4:] or 200)
        return mcts_picker(ev, n_simulations=sims, vct_finish_nodes=50, fpu_reduction_c=0.45), None
    if name == "netargmax":  # bare policy + finisher
        f = Field("net", ev, temperature=1e-3)
        def pick(s, rng):
            w, mv, _ = vct([s])
            if w[0] and mv[0] >= 0:
                return int(mv[0])
            return int(np.argmax(f.probs([s])[0]))
        return pick, None
    if name == "field":  # the modeled field itself, as an opponent: net @T + finisher, own rng
        f = Field("net", ev, temperature=args.field_temp)
        def pick(s, rng):
            w, mv, _ = vct([s])
            if w[0] and mv[0] >= 0:
                return int(mv[0])
            p = f.probs([s])[0]
            return int(rng.choice(len(p), p=p))
        return pick, None
    if name.startswith("walt"):
        kind = "uniform" if name == "walt-uniform" else "net"
        cfg = WaltConfig(n_tapes=args.tapes, horizon=args.horizon, n_cand=args.cand,
                         vct_nodes=args.vct_nodes, rollout_vct_every=args.rollout_vct_every, seed=seed)
        w = WaltPlayer(cfg, Field(kind, ev, temperature=args.field_temp), ev, vct, label=name)
        return w, w
    raise SystemExit(f"unknown player {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--max-wall-secs", type=float, default=900)
    ap.add_argument("--opening-moves", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tapes", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--cand", type=int, default=8)
    ap.add_argument("--vct-nodes", type=int, default=50)
    ap.add_argument("--rollout-vct-every", type=int, default=1)
    ap.add_argument("--field-temp", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = pick_device(os.environ.get("GOMOKU_DEVICE"))
    model, _ = load_checkpoint(CHAMP, device=device)
    ev = make_torch_evaluator(fuse_model_for_inference(model), device)
    ev([GameState.initial()])
    vct = VCT(args.vct_nodes)

    pa, walt = build(args.player, ev, vct, args, args.seed)
    pb, _ = build(args.opp, ev, vct, args, args.seed + 1)
    ta, tb = Timed(pa, args.player), Timed(pb, args.opp)

    rng = np.random.default_rng(args.seed)
    t_start = time.perf_counter()
    W = L = D = 0
    capped = False
    with open(args.out, "a") as fo:
        opening = None
        for g in range(args.games):
            if time.perf_counter() - t_start > args.max_wall_secs:
                capped = True
                break
            if g % 2 == 0:  # same opening, colors swapped
                opening, _ = _random_opening_state(rng, args.opening_moves)
            a_black = g % 2 == 0
            state, ply0 = opening, opening.move_count
            log0 = len(walt.log) if walt else 0
            ta.secs.clear(); tb.secs.clear()
            ply = ply0
            while True:
                a_to_move = (ply % 2 == 0) == a_black
                act = (ta if a_to_move else tb)(state, rng)
                state = state.apply(act)
                done, v = state.is_terminal()
                if done:
                    res = ("W" if a_to_move else "L") if v == -1.0 else "D"
                    break
                ply += 1
            W += res == "W"; L += res == "L"; D += res == "D"
            rec = {"game": g, "player": args.player, "opp": args.opp, "a_black": a_black,
                   "result": res, "plies": state.move_count,
                   "a_secs_mean": float(np.mean(ta.secs)) if ta.secs else 0.0,
                   "b_secs_mean": float(np.mean(tb.secs)) if tb.secs else 0.0}
            if walt:
                rec["walt_moves"] = walt.log[log0:]
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"[{args.player} vs {args.opp}] g{g} {'B' if a_black else 'W'} {res} "
                  f"plies={state.move_count} a={rec['a_secs_mean']:.2f}s/mv b={rec['b_secs_mean']:.2f}s/mv "
                  f"W-L-D={W}-{L}-{D} t={time.perf_counter()-t_start:.0f}s", flush=True)
        summ = {"summary": True, "player": args.player, "opp": args.opp, "W": W, "L": L, "D": D,
                "games": W + L + D, "capped": capped, "wall": time.perf_counter() - t_start,
                "args": vars(args)}
        fo.write(json.dumps(summ) + "\n")
        print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
