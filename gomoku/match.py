"""Ad-hoc matchups between players.

Usage:
    python -m gomoku.match random vs heuristic --n-games 50
    python -m gomoku.match heuristic vs lookahead:depth=4 --n-games 30
    python -m gomoku.match model:checkpoint=checkpoints/latest.pt,sims=200 vs heuristic --n-games 20
    python -m gomoku.match --matrix random heuristic lookahead:depth=4 --n-games 30

Player spec format: `kind[:k=v,k=v,...]`. Supported kinds:
    random
    heuristic
    lookahead   (depth=N, default 4)
    model       (checkpoint=PATH required, sims=N default 100, c_puct=F default 1.5)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable

from gomoku.baselines import (
    Player,
    defensive_player,
    heuristic_player,
    lookahead_player,
    pacifist_blocker,
    random_player,
)
from gomoku.eval import MatchResult, mcts_picker, play_match_pickers


@dataclass
class PlayerSpec:
    raw: str
    kind: str
    kwargs: dict[str, str]

    def label(self) -> str:
        if not self.kwargs:
            return self.kind
        bits = ",".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"{self.kind}:{bits}"


def parse_spec(s: str) -> PlayerSpec:
    if ":" in s:
        kind, rest = s.split(":", 1)
        kwargs: dict[str, str] = {}
        for part in rest.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"bad spec {s!r}: expected k=v, got {part!r}")
            k, v = part.split("=", 1)
            kwargs[k.strip()] = v.strip()
    else:
        kind, kwargs = s.strip(), {}
    kind = kind.strip().lower()
    if kind not in ("random", "heuristic", "defensive", "pacifist", "lookahead", "model"):
        raise SystemExit(f"unknown player kind {kind!r}")
    return PlayerSpec(raw=s, kind=kind, kwargs=kwargs)


# Cache built model pickers so a matrix run doesn't reload the checkpoint per cell.
_model_cache: dict[tuple, Player] = {}


def build_player(spec: PlayerSpec) -> Player:
    if spec.kind == "random":
        return random_player
    if spec.kind == "heuristic":
        return heuristic_player
    if spec.kind == "defensive":
        return defensive_player
    if spec.kind == "pacifist":
        max_own = int(spec.kwargs.get("max_own_line", "3"))
        def _picker(state, rng, *, _m=max_own):
            return pacifist_blocker(state, rng, max_own_line=_m)
        return _picker
    if spec.kind == "lookahead":
        depth = int(spec.kwargs.get("depth", "4"))
        return lookahead_player(depth=depth)
    if spec.kind == "model":
        checkpoint = spec.kwargs.get("checkpoint")
        if not checkpoint:
            raise SystemExit(f"model spec needs checkpoint=PATH: {spec.raw!r}")
        sims = int(spec.kwargs.get("sims", "100"))
        c_puct = float(spec.kwargs.get("c_puct", "1.5"))
        key = (os.path.abspath(checkpoint), sims, c_puct)
        if key not in _model_cache:
            # Lazy torch import — keeps `python -m gomoku.match random vs heuristic`
            # zero-torch.
            from gomoku.mcts import make_torch_evaluator
            from gomoku.model import fuse_model_for_inference, load_checkpoint
            from gomoku.util import pick_device

            device = pick_device(os.environ.get("GOMOKU_DEVICE"))
            model, _ = load_checkpoint(checkpoint, device=device)
            model = fuse_model_for_inference(model)
            evaluator = make_torch_evaluator(model, device)
            _model_cache[key] = mcts_picker(evaluator, n_simulations=sims, c_puct=c_puct)
        return _model_cache[key]
    raise SystemExit(f"unhandled kind {spec.kind!r}")


def _format_result(label_a: str, label_b: str, r: MatchResult) -> str:
    return (
        f"{label_a} vs {label_b}: "
        f"{r.wins}W-{r.losses}L-{r.draws}D over {r.n_games} games "
        f"(win rate {r.win_rate:.2%})"
    )


def _run_single(spec_a: PlayerSpec, spec_b: PlayerSpec, n_games: int, seed: int) -> MatchResult:
    pa = build_player(spec_a)
    pb = build_player(spec_b)
    return play_match_pickers(pa, pb, n_games=n_games, seed=seed)


def _print_matrix(specs: list[PlayerSpec], results: dict[tuple[int, int], MatchResult]) -> None:
    """Print a win-rate grid: rows = A (row player), columns = B (column player).
    Cell (i, j) shows A's win rate when A=spec[i], B=spec[j]."""
    labels = [s.label() for s in specs]
    col_w = max(max((len(lab) for lab in labels), default=0), 12)
    header = " " * (col_w + 2) + " ".join(f"{lab:>{col_w}}" for lab in labels)
    print(header)
    for i, row_lab in enumerate(labels):
        cells = [f"{row_lab:>{col_w}}: "]
        for j in range(len(labels)):
            if i == j:
                cells.append(f"{'-':>{col_w}}")
            elif (i, j) in results:
                r = results[(i, j)]
                cells.append(f"{r.win_rate:>{col_w}.2%}")
            else:
                cells.append(f"{'?':>{col_w}}")
        print(" ".join(cells))
    print()
    print("Raw results:")
    for (i, j), r in sorted(results.items()):
        print("  " + _format_result(labels[i], labels[j], r))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run matches between players.")
    p.add_argument("specs", nargs="+", help="Player specs (and optional 'vs' separator).")
    p.add_argument("--matrix", action="store_true",
                   help="Run all pairs across all given specs.")
    p.add_argument("--n-games", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_specs = [s for s in args.specs if s.strip().lower() != "vs"]

    if args.matrix:
        specs = [parse_spec(s) for s in raw_specs]
        if len(specs) < 2:
            raise SystemExit("--matrix needs at least 2 specs")
        results: dict[tuple[int, int], MatchResult] = {}
        for i, sa in enumerate(specs):
            for j, sb in enumerate(specs):
                if i == j:
                    continue
                print(f"running {sa.label()} vs {sb.label()} ({args.n_games} games)...")
                results[(i, j)] = _run_single(sa, sb, args.n_games, args.seed)
                print("  " + _format_result(sa.label(), sb.label(), results[(i, j)]))
        print()
        _print_matrix(specs, results)
        return

    if len(raw_specs) != 2:
        raise SystemExit("expected exactly two player specs (or use --matrix). "
                         f"got: {raw_specs}")
    spec_a = parse_spec(raw_specs[0])
    spec_b = parse_spec(raw_specs[1])
    print(f"running {spec_a.label()} vs {spec_b.label()} ({args.n_games} games)...")
    r = _run_single(spec_a, spec_b, args.n_games, args.seed)
    print(_format_result(spec_a.label(), spec_b.label(), r))


if __name__ == "__main__":
    main()
