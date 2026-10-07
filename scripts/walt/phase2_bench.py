"""Phase 2 bench: (1) oracle batch curve on realistic 9x9 boards, (2) hidden mass
(cap50 vs cap2000 on the same positions), (3) walt s/move vs tapes K.

Positions come from field-vs-field games (net@T1 + finisher, the walt field),
started from 4-move random openings and advanced in lockstep.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walt_vec as wv  # noqa: E402
from walt_gomoku2 import VCT, Field, WaltConfig, WaltPlayer  # noqa: E402

from gomoku.game import GameState  # noqa: E402
from gomoku.mcts import make_torch_evaluator  # noqa: E402
from gomoku.model import fuse_model_for_inference, load_checkpoint  # noqa: E402
from gomoku.self_play import _random_opening_state  # noqa: E402
from gomoku.util import pick_device  # noqa: E402

CHAMP = "/Users/jason/data/sound-world-107b/sweep_runs/sound-world/checkpoints/worker_weights.pt"
OUT = "sweep_logs/walt/phase2_bench.jsonl"


def emit(rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def field_positions(ev, vct, n_games, rng):
    """Play field-vs-field games in lockstep; return every non-terminal board seen (+ ply)."""
    st = [_random_opening_state(rng, 4)[0] for _ in range(n_games)]
    b, h = wv.state_arrays(st)
    tapes = rng.integers(1, 2**62, n_games).astype(np.uint64)
    seen_b, seen_ply = [], []
    live = np.arange(n_games)
    ply = 4
    while live.size:
        bb, hh = b[live], h[live]
        seen_b.append(bb.copy()); seen_ply.append(np.full(live.size, ply))
        win, mv, _ = vct.solve_boards(bb)
        p = wv.field_probs("net", ev, bb, hh, 1.0)
        a = wv.draw(p, wv.tape_u(bb, tapes[live]))
        a = np.where(win & (mv >= 0), mv, a)
        r, c = np.divmod(a, wv.N)
        snap = bb.copy(); nb = bb.copy()
        nb[np.arange(live.size), 0, r, c] = True
        done = wv.five(nb[:, 0]) | (nb[:, 0] | nb[:, 1]).all((1, 2))
        nb = nb[:, ::-1].copy()
        nh = np.concatenate([snap[:, None], hh[:, :-1]], 1)
        keep = ~done
        live = live[keep]
        b[live], h[live] = nb[keep], nh[keep]
        ply += 1
    return np.concatenate(seen_b), np.concatenate(seen_ply)


def main():
    device = pick_device(os.environ.get("GOMOKU_DEVICE"))
    model, _ = load_checkpoint(CHAMP, device=device)
    ev = make_torch_evaluator(fuse_model_for_inference(model), device)
    ev([GameState.initial()])
    vct = VCT(50)
    rng = np.random.default_rng(7)

    t = time.perf_counter()
    boards, plies = field_positions(ev, vct, 2048, rng)
    emit({"stage": "positions", "n": int(len(boards)), "secs": time.perf_counter() - t,
          "ply_mean": float(plies.mean())})

    # (1) oracle batch curve at cap50 (and cap2000 for the small end), realistic boards
    for nodes in (50,):
        for B in (16, 64, 256, 1024, 4096, 16384, 32768):
            idx = rng.choice(len(boards), size=min(B, len(boards)), replace=B > len(boards))
            sub = np.ascontiguousarray(boards[idx])
            vct.solve(sub, max_nodes=nodes)  # warm this shape
            ts = []
            for _ in range(3):
                t = time.perf_counter(); vct.solve(sub, max_nodes=nodes); ts.append(time.perf_counter() - t)
            emit({"stage": "oracle_curve", "nodes": nodes, "B": int(B), "secs": float(np.median(ts)),
                  "boards_per_s": float(B / np.median(ts))})

    # (2) hidden mass: same positions at 50 vs 2000 nodes
    idx = rng.choice(len(boards), size=min(16384, len(boards)), replace=False)
    sub = np.ascontiguousarray(boards[idx])
    t = time.perf_counter(); w50, cap50 = vct.solve(sub, max_nodes=50)[:2]; s50 = time.perf_counter() - t
    t = time.perf_counter(); w2k, cap2k = vct.solve(sub, max_nodes=2000)[:2]; s2k = time.perf_counter() - t
    w50, cap50, w2k, cap2k = map(lambda x: np.asarray(x, bool), (w50, cap50, w2k, cap2k))
    emit({"stage": "hidden_mass", "n": int(len(sub)), "secs50": s50, "secs2000": s2k,
          "win50": float(w50.mean()), "hitcap50": float(cap50.mean()),
          "win2000": float(w2k.mean()), "hitcap2000": float(cap2k.mean()),
          "flip_nowin50_to_win2000": float((~w50 & w2k).mean()),
          "flip_given_hitcap50": float((w2k & cap50).sum() / max(cap50.sum(), 1)),
          "flip_given_clean_nowin50": float((w2k & ~w50 & ~cap50).sum() / max((~w50 & ~cap50).sum(), 1)),
          "lost_win": int((w50 & ~w2k).sum())})

    # (3) walt s/move vs K on a fixed set of mid-game positions
    mids = [i for i in range(len(boards)) if 10 <= plies[i] <= 20]
    pick = rng.choice(mids, size=6, replace=False)
    st = []
    for i in pick:  # rebuild a GameState with this board (history omitted -> zero planes, same as start)
        st.append(GameState(board=boards[i].copy(), move_count=int(plies[i]), history=()))
    for K in (32, 256, 1024, 2048, 4096):
        w = WaltPlayer(WaltConfig(n_tapes=K, horizon=2, n_cand=8, vct_nodes=50, seed=K),
                       Field("net", ev, 1.0), ev, vct)
        secs, vsecs, bmax, bmean = [], [], [], []
        for s in st:
            c0, s0 = vct.calls, vct.secs
            t = time.perf_counter(); w(s, rng); secs.append(time.perf_counter() - t)
            m = w.log[-1]
            if m["kind"] == "search":
                vsecs.append(m["vct_secs"]); bmax.append(m["batch_max"]); bmean.append(m["batch_mean"])
        emit({"stage": "walt_K", "K": K, "s_per_move": float(np.mean(secs)),
              "vct_frac": float(np.sum(vsecs) / np.sum(secs)) if vsecs else None,
              "batch_max": float(np.mean(bmax)) if bmax else None,
              "batch_mean": float(np.mean(bmean)) if bmean else None,
              "choices": [m.get("choice") for m in w.log]})


if __name__ == "__main__":
    main()
