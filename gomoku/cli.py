"""Terminal CLI: play against a trained checkpoint."""

from __future__ import annotations

import argparse

import numpy as np

from gomoku.game import GameState, action_to_str, render, str_to_action
from gomoku.mcts import MCTSGame, make_torch_evaluator, policy_from_visits, run_batched_mcts
from gomoku.model import fuse_model_for_inference, load_checkpoint
from gomoku.util import pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n-simulations", type=int, default=400)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--first", type=str, default="ask", choices=["ask", "human", "model"])
    p.add_argument("--show-topk", type=int, default=5)
    return p.parse_args()


def _model_move(state: GameState, evaluator, n_sims: int, c_puct: float, topk: int) -> int:
    g = MCTSGame(state, c_puct=c_puct)
    run_batched_mcts([g], evaluator, n_simulations=n_sims, add_root_noise=False)
    pi = policy_from_visits(g.root, temperature=0.0)
    visits = g.root.N
    if topk > 0:
        order = np.argsort(-visits)[:topk]
        legal_total = visits.sum()
        if legal_total > 0:
            parts = []
            for a in order:
                if visits[a] == 0:
                    continue
                parts.append(f"{action_to_str(int(a))}({visits[a]})")
            if parts:
                print("  model top moves: " + ", ".join(parts))
        # Value estimate from root: Q of best action
        if visits.sum() > 0:
            best = int(np.argmax(visits))
            q = float(g.root.W[best] / max(g.root.N[best], 1))
            print(f"  model eval: {q:+.2f}")
    return int(np.argmax(pi))


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device = {device}")
    model, _payload = load_checkpoint(args.checkpoint, device=device)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, device)

    if args.first == "ask":
        ans = input("Go first? [Y/n] ").strip().lower()
        human_is_black = ans != "n"
    else:
        human_is_black = (args.first == "human")

    state = GameState.initial()
    human_to_move = human_is_black
    last_action: int | None = None

    print()
    print(render(state, last_action=None))
    print()

    ply = 0
    while True:
        if human_to_move:
            while True:
                s = input(f"your move (e.g. e5): ").strip()
                if s in ("q", "quit", "exit"):
                    print("bye")
                    return
                try:
                    action = str_to_action(s)
                except ValueError as e:
                    print(f"  bad input: {e}")
                    continue
                if not state.legal_mask()[action]:
                    print("  illegal — that square is occupied")
                    continue
                break
        else:
            print(f"thinking ({args.n_simulations} sims)...")
            action = _model_move(state, evaluator, args.n_simulations, args.c_puct, args.show_topk)
            print(f"model plays: {action_to_str(action)}")

        side_just_moved = ply % 2
        state = state.apply(action)
        last_action = action
        print()
        print(render(state, last_action=last_action))
        print()

        done, v = state.is_terminal()
        if done:
            if v == -1.0:
                winner_black = (side_just_moved == 0)
                winner_human = (winner_black == human_is_black)
                print("you win!" if winner_human else "model wins")
            else:
                print("draw")
            return
        human_to_move = not human_to_move
        ply += 1


if __name__ == "__main__":
    main()
