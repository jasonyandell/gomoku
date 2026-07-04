"""FAST batched eval-gradient: net @idx-2 vs Rapfi graded by THINK-TIME.

The plain gradient played games SERIALLY with one engine and one-game MCTS — minutes
per pass. This plays EVERY rung's games CONCURRENTLY in one lockstep loop, reusing
the two throughput tricks the rest of the project already relies on:

  * net moves are **batched across all net-to-move games** (every rung at once) in a
    single `run_batched_mcts` call — one GPU pass for the whole field; and
  * Rapfi moves are **parallelized across engines** via `RapfiPool.label_states`
    (the pool's per-move pick, the same engine fan-out the mine drove).

Strength dial = Rapfi **think-time (timeout_ms)** — the real-strength knob that the
plain idx-2 gate reads 0 against at 1000 ms. (max_node was rejected: even 2,000,000
nodes is far weaker than 1000 ms of search, so a node ladder never reaches the wall.)
Running the rungs CONCURRENTLY (one pool per timeout, all live at once) makes the
wall-clock ≈ the slowest rung's single-game move chain, not the sum of rungs — so a
4-tier pass lands in the tens of seconds. White reported separately per rung.

    rapfi@50ms < rapfi@150ms < rapfi@400ms < rapfi@1000ms(~the gate's bar)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gomoku.board_config import BOARD_SIZE
from gomoku.eval_panel import EvaluatorCache, IDX2_OPENING, fixed_opening_state
from gomoku.mcts import MCTSGame, policy_from_visits, run_batched_mcts
from gomoku.rapfi_pool import RapfiPool, rapfi_available


# (label, timeout_ms) — weak → strong, by per-move think-time. Calibrated to the
# net's CURRENT transition: at epoch ~190 it beats rapfi@25ms (100%) but loses to
# 50 ms+ (0%), so the live band is LOW timeouts — which are also inherently FAST
# (≤200 ms/move). As the net climbs, the 0.5-crossing migrates up the ladder; the
# 1000 ms bar stays the plain idx-2 gate's job (too slow to belong in a fast loop).
DEFAULT_RUNGS = [
    ("rapfi@25ms",  25),
    ("rapfi@50ms",  50),
    ("rapfi@100ms", 100),
    ("rapfi@200ms", 200),
]


def _epoch_of(path: str) -> int:
    m = re.search(r"epoch0*([0-9]+)\.pt", path)
    return int(m.group(1)) if m else -1


def run_fast_gradient(checkpoint: str, *, n_games: int, sims: int, seed: int,
                      device=None, rungs=DEFAULT_RUNGS, c_puct: float = 1.5,
                      max_plies: int | None = None):
    """Play all rungs' games concurrently; return per-rung color-split scores.

    One RapfiPool per rung timeout (all live at once) → rungs advance in parallel,
    so wall-clock tracks the slowest rung rather than the sum.
    """
    rng = np.random.default_rng(seed)
    start = fixed_opening_state(IDX2_OPENING)
    start_ply = start.move_count
    max_plies = max_plies or (BOARD_SIZE * BOARD_SIZE)
    net_eval = EvaluatorCache(device=device).evaluator(checkpoint)

    games = []
    for label, tmo in rungs:
        for i in range(n_games):
            games.append({"rung": label, "tmo": tmo, "state": start,
                          "net_black": (i % 2 == 0), "done": False, "winner": None})

    # One pool per distinct timeout; size = max concurrent rapfi-to-move games for
    # that timeout (~half its games per ply). Created together so rungs run in lockstep.
    timeouts = sorted({tmo for _, tmo in rungs})
    psize = max(1, (n_games + 1) // 2)
    pools = {}
    try:
        for tmo in timeouts:
            pools[tmo] = RapfiPool(size=psize, timeout_ms=tmo, board_size=BOARD_SIZE)
            pools[tmo].__enter__()

        ply = start_ply
        while any(not g["done"] for g in games) and ply < max_plies:
            active = [g for g in games if not g["done"]]
            side = ply % 2  # 0=black, 1=white to move
            net_games = [g for g in active if (side == 0) == g["net_black"]]
            rap_games = [g for g in active if (side == 0) != g["net_black"]]

            moves = {}
            # NET: one batched MCTS pass across every net-to-move game (all rungs), greedy.
            if net_games:
                mgs = [MCTSGame(g["state"], c_puct=c_puct, rng=rng) for g in net_games]
                run_batched_mcts(mgs, net_eval, n_simulations=sims, add_root_noise=False)
                for g, mg in zip(net_games, mgs):
                    moves[id(g)] = int(np.argmax(policy_from_visits(mg.root, temperature=0.0)))
            # RAPFI: per-timeout pool, parallel real timed moves. Run the rung
            # pools CONCURRENTLY (each label_states is itself threaded) so a ply
            # costs the slowest rung's move, not the sum across rungs.
            by_tmo = {}
            for g in rap_games:
                by_tmo.setdefault(g["tmo"], []).append(g)
            if by_tmo:
                def _move_group(item):
                    tmo, gs = item
                    return gs, pools[tmo].label_states([g["state"] for g in gs])
                with ThreadPoolExecutor(max_workers=len(by_tmo)) as ex:
                    for gs, acts in ex.map(_move_group, list(by_tmo.items())):
                        for g, a in zip(gs, acts):
                            moves[id(g)] = int(a)

            for g in active:
                a = moves.get(id(g))
                if a is None:
                    continue
                g["state"] = g["state"].apply(a)
                done, v = g["state"].is_terminal()
                if done:
                    g["done"] = True
                    g["winner"] = side if v == -1.0 else None  # else draw
            ply += 1
    finally:
        for p in pools.values():
            try:
                p.__exit__(None, None, None)
            except Exception:
                pass

    out = []
    for label, _tmo in rungs:
        gs = [g for g in games if g["rung"] == label]
        def split(seat_black):
            sub = [g for g in gs if g["net_black"] == seat_black]
            w = sum(1 for g in sub if g["winner"] is not None
                    and g["winner"] == (0 if seat_black else 1))
            d = sum(1 for g in sub if g["winner"] is None and g["done"])
            return w, d, len(sub)
        bw, bd, bn = split(True)
        ww, wd, wn = split(False)
        tot = len(gs)
        score = (bw + ww + 0.5 * (bd + wd)) / max(tot, 1)
        out.append({"rung": label,
                    "score": round(score, 4),
                    "black": round((bw + 0.5 * bd) / max(bn, 1), 4),
                    "white": round((ww + 0.5 * wd) / max(wn, 1), 4),
                    "n_games": tot})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-games", type=int, default=12)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    if BOARD_SIZE != 15:
        print("WARNING: GOMOKU_BOARD_SIZE != 15", file=sys.stderr)
    if not rapfi_available():
        print("ERROR: native Rapfi not available", file=sys.stderr)
        return 2

    ep = _epoch_of(args.checkpoint)
    rows = run_fast_gradient(args.checkpoint, n_games=args.n_games, sims=args.sims,
                             seed=args.seed, device=args.device)
    print(json.dumps({"checkpoint": args.checkpoint, "epoch": ep, "rungs": rows}, indent=2))
    parts = [f"{r['rung']}={r['score']:.2f}/w{r['white']:.2f}" for r in rows]
    print(f"\nGRADIENT epoch={ep} (n={args.n_games} sims={args.sims}): " + "  ".join(parts),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
