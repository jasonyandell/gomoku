#!/usr/bin/env python3
"""Eval a model checkpoint against the Rapfi external-engine YARDSTICK.

WHY: our anchored elo ladder (random/heuristic/lookahead2/lookahead4) saturates
around ~1700 Elo — we cannot tell "great" from "very good" past it. Rapfi
(Gomocup freestyle Elo 2625) is a rated external engine that gives an honest
yardstick for the v4 ceiling-mover era. This is EVAL-ONLY; Rapfi never touches
self-play training.

Plays `model:checkpoint=...,sims=...` vs `external:cmd=<rapfi>,timeout_ms=T` at
one or more time-control tiers, color-alternated, and writes one JSONL record
per tier with explicit provenance (engine, build ref, timeout, board size,
rule, wrapper version).

Treat the Gomocup Elo as provenance, not a 9x9 label. The local question is
narrow: "what does checkpoint X score against Rapfi at local time control Y on
9x9 freestyle?"

Example:
    PYTHONPATH=$PWD GOMOKU_DEVICE=cpu python scripts/eval_vs_rapfi.py \
        --checkpoint archives/wl5_e10200_seed.pt --sims 100 \
        --rapfi engines/rapfi/pbrain-rapfi \
        --timeouts 100 500 1000 --n-games 20 \
        --out sweep_logs/rapfi_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone


def _build_ref(rapfi_path: str) -> dict:
    """Best-effort build provenance for the Rapfi binary."""
    ref: dict = {"path": os.path.abspath(rapfi_path)}
    side = os.path.join(os.path.dirname(rapfi_path), "BUILD_COMMIT.txt")
    if os.path.exists(side):
        try:
            with open(side) as f:
                ref["build_commit_file"] = f.read().strip()
        except OSError:
            pass
    return ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    ap.add_argument("--sims", type=int, default=100, help="Model MCTS sims per move.")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--rapfi", required=True, help="Path to pbrain-rapfi binary.")
    ap.add_argument("--timeouts", type=int, nargs="+", default=[100, 500, 1000],
                    help="Rapfi per-move timeout tiers in ms (difficulty tiers).")
    ap.add_argument("--n-games", type=int, default=20,
                    help="Games per tier (color-alternated). >=20 to beat single-digit noise.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", type=int, default=0, help="0 = freestyle.")
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--out", default=None, help="JSONL output path (append).")
    args = ap.parse_args()

    # Lazy: only import the heavy bits inside main so --help is cheap.
    from gomoku.eval import play_match_pickers
    from gomoku.external_engine import (
        WRAPPER_VERSION,
        ExternalEngineConfig,
        ExternalEnginePlayer,
    )
    from gomoku.match import build_player, parse_spec

    rapfi_abs = os.path.abspath(args.rapfi)
    if not os.path.exists(rapfi_abs):
        raise SystemExit(f"rapfi binary not found: {rapfi_abs}")

    model_picker = build_player(
        parse_spec(f"model:checkpoint={args.checkpoint},sims={args.sims},c_puct={args.c_puct}")
    )

    build_ref = _build_ref(rapfi_abs)
    records = []
    for timeout_ms in args.timeouts:
        engine = ExternalEnginePlayer(
            ExternalEngineConfig(
                cmd=rapfi_abs,
                timeout_ms=timeout_ms,
                label=f"rapfi{timeout_ms}",
                rule=args.rule,
                board_size=args.size,
            )
        )
        t0 = time.time()
        try:
            res = play_match_pickers(
                model_picker, engine, n_games=args.n_games, seed=args.seed
            )
        finally:
            engine.close()
        dt = time.time() - t0

        rec = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": os.path.abspath(args.checkpoint),
            "model_sims": args.sims,
            "model_c_puct": args.c_puct,
            "engine": f"rapfi{timeout_ms}",
            "engine_source": "https://github.com/dhbloo/rapfi",
            "engine_build_ref": build_ref,
            "timeout_ms": timeout_ms,
            "board_size": args.size,
            "rule": args.rule,  # 0 = freestyle
            "wrapper_version": WRAPPER_VERSION,
            "n_games": res.n_games,
            "wins": res.wins,
            "losses": res.losses,
            "draws": res.draws,
            "win_rate": res.win_rate,  # draws = half, model perspective
            "wall_secs": round(dt, 2),
        }
        records.append(rec)
        print(
            f"rapfi{timeout_ms}: {res.wins}W-{res.losses}L-{res.draws}D / "
            f"{res.n_games} games  win_rate={res.win_rate:.2%}  ({dt:.1f}s)"
        )
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    if args.out:
        print(f"wrote {len(records)} record(s) to {args.out}")


if __name__ == "__main__":
    main()
