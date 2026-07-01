"""VCT-terminus science A/B — finisher eval (issue #100).

Evaluates the two matched 100-epoch nets (vctsci-terminus vs vctsci-control)
through the STANDARD match harness, each in two configs:
  - raw       : the net's own MCTS policy (vct_finish_nodes=0)
  - finisher  : policy to the first cap50 VCT, then the #99 GPU-oracle finisher
                hammers it out to a real five (vct_finish_nodes=50)
vs fixed baselines (heuristic, lookahead:d2, lookahead:d4), balanced colors.

The point: the terminus net learns to REACH VCTs but not CONVERT, so its
raw-policy strength is UNDERRATED; the finisher closes that. The control already
converts, so the finisher should barely fire and never hurt it. A fire-counter
proves both.

Run sequentially (single process) — the #99 guard forbids the finisher under
multi-worker fork (MLX-under-fork / Metal-wedge risk).
"""
import os
os.environ.setdefault("GOMOKU_BOARD_SIZE", "9")
import argparse
import numpy as np
import torch

from gomoku.model import load_checkpoint
from gomoku.mcts import make_torch_evaluator
from gomoku import eval as evmod
from gomoku.eval import mcts_picker, play_match_pickers
from gomoku.baselines import heuristic_player, lookahead_player, random_player


def fmt(r):
    return (f"{r.win_rate:5.1%}  ({r.wins}W-{r.losses}L-{r.draws}D"
            f"  black {r.black_w}-{r.black_l}-{r.black_d}"
            f" / white {r.white_w}-{r.white_l}-{r.white_d})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terminus", default="sweep_runs/vctsci-terminus/checkpoints/epoch0100.pt")
    ap.add_argument("--control", default="sweep_runs/vctsci-control/checkpoints/epoch0100.pt")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--n-games", type=int, default=40)
    ap.add_argument("--budget", type=int, default=50)
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={dev} sims={args.sims} n_games={args.n_games} finisher_budget={args.budget}\n")

    # Fire-counter around the real oracle loader (proves the finisher fires).
    _real_loader = evmod._load_vct_solver
    fires = {"n": 0, "calls": 0}

    def counting_loader():
        solver = _real_loader()

        def wrapped(boards, *, max_nodes, return_move=False):
            out = solver(boards, max_nodes=max_nodes, return_move=return_move)
            fires["calls"] += int(boards.shape[0])
            fires["n"] += int(np.asarray(out[0]).sum())
            return out
        return wrapped
    evmod._load_vct_solver = counting_loader

    baselines = [
        ("random", random_player),
        ("heuristic", heuristic_player),
        ("lookahead:d2", lookahead_player(depth=2)),
        ("lookahead:d4", lookahead_player(depth=4)),
    ]

    nets = [("terminus", args.terminus), ("control", args.control)]

    # net -> config -> baseline -> (winrate, fires)
    results = {}
    for net_name, ckpt in nets:
        model, payload = load_checkpoint(ckpt, device=dev)
        model.eval()
        ev = make_torch_evaluator(model, dev)
        print(f"=== {net_name}  ({ckpt}, epoch={payload.get('epoch')}) ===")
        for cfg_name, vct_finish in [("raw", 0), ("finisher", args.budget)]:
            picker = mcts_picker(ev, n_simulations=args.sims, c_puct=1.5,
                                 vct_finish_nodes=vct_finish)
            for base_name, base in baselines:
                fires["n"] = fires["calls"] = 0
                r = play_match_pickers(picker, base, n_games=args.n_games, seed=0)
                results[(net_name, cfg_name, base_name)] = (r, fires["n"], fires["calls"])
                tag = f"fires={fires['n']}/{fires['calls']}" if vct_finish else ""
                print(f"  {cfg_name:8s} vs {base_name:12s}: {fmt(r)}  {tag}")
        print()

    # Compact deltas: what did the finisher buy each net vs each baseline?
    print("=== finisher lift (finisher_winrate - raw_winrate) ===")
    for net_name, _ in nets:
        for base_name, _ in baselines:
            raw = results[(net_name, "raw", base_name)][0].win_rate
            fin = results[(net_name, "finisher", base_name)][0].win_rate
            print(f"  {net_name:8s} vs {base_name:12s}: {fin - raw:+.1%}  "
                  f"(raw {raw:.0%} -> fin {fin:.0%})")
    print("\nRESULT: OK")


if __name__ == "__main__":
    main()
