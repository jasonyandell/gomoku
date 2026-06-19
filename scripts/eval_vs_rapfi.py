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


def make_eval_record(
    *,
    checkpoint: str,
    sims: int,
    c_puct: float,
    fpu_reduction_c: float,
    reuse_tree: bool,
    proven_prop: bool,
    timeout_ms: int,
    board_size: int,
    rule: int,
    wrapper_version: str,
    build_ref: dict,
    n_games: int,
    wins: int,
    losses: int,
    draws: int,
    win_rate: float,
    wall_secs: float,
    extra_fields: dict | None = None,
) -> dict:
    """Build one JSONL strength record (pure; unit-testable without GPU/engine).

    This is the single source of truth for the eval row schema, used by the
    Rapfi CLI and the paced 15x15 ladder driver alike.
    """
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": os.path.abspath(checkpoint),
        "model_sims": sims,
        "model_c_puct": c_puct,
        "fpu_reduction_c": fpu_reduction_c,
        "reuse_tree": reuse_tree,
        "proven_prop": proven_prop,
        "engine": f"rapfi{timeout_ms}",
        "engine_source": "https://github.com/dhbloo/rapfi",
        "engine_build_ref": build_ref,
        "timeout_ms": timeout_ms,
        "board_size": board_size,
        "rule": rule,  # 0 = freestyle
        "wrapper_version": wrapper_version,
        "n_games": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,  # draws = half, model perspective
        "wall_secs": round(wall_secs, 2),
    }
    if extra_fields:
        rec.update(extra_fields)
    return rec


def append_jsonl(out: str, rec: dict) -> None:
    """Append one record as a JSON line, creating the parent dir if needed.

    Append-only (mode 'a') so repeated calls accumulate rows — never clobbers.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_rapfi_eval(
    *,
    checkpoint: str,
    rapfi: str,
    timeouts,
    sims: int = 100,
    c_puct: float = 1.5,
    n_games: int = 20,
    seed: int = 0,
    rule: int = 0,
    size: int = 9,
    fpu_reduction_c: float = 0.0,
    reuse_tree: bool = False,
    proven_prop: bool = False,
    random_opening_moves: int = 0,
    out: str | None = None,
    extra_fields: dict | None = None,
    print_progress: bool = True,
) -> list[dict]:
    """Score a checkpoint against Rapfi at one or more time-control tiers.

    Shared core used by both this script's CLI and the paced 15x15 ladder
    driver (``scripts/ladder_eval_15x15.py``). Loads the model once, builds the
    eval picker, and for each tier plays ``n_games`` color-alternated games vs
    ``external:cmd=<rapfi>,timeout_ms=T``, appending one JSONL record per tier.

    IMPORTANT: ``size`` MUST equal the board size resolved at import time
    (``gomoku.board_config.BOARD_SIZE``). The external-engine coordinate mapping
    derives from the module-level ``BOARD_SIZE``, so a mismatch is a hard error
    rather than silently mis-mapping moves. Callers running at 15x15 must set
    ``GOMOKU_BOARD_SIZE=15`` (or pass ``--board-size 15``) before any gomoku
    import.

    Returns the list of records written (one per tier). ``extra_fields`` are
    merged into every record (e.g. a driver tag).
    """
    from gomoku.board_config import BOARD_SIZE
    from gomoku.eval import mcts_picker, play_match_pickers
    from gomoku.external_engine import (
        WRAPPER_VERSION,
        ExternalEngineConfig,
        ExternalEnginePlayer,
    )
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint
    from gomoku.util import pick_device

    if int(size) != int(BOARD_SIZE):
        raise SystemExit(
            f"--size {size} conflicts with the active board size {BOARD_SIZE} "
            f"(fixed at import time). Set GOMOKU_BOARD_SIZE={size} (or pass "
            f"--board-size {size}) and start a new process — the external-engine "
            f"coordinate mapping derives from the import-time BOARD_SIZE."
        )

    rapfi_abs = os.path.abspath(rapfi)
    if not os.path.exists(rapfi_abs):
        raise SystemExit(f"rapfi binary not found: {rapfi_abs}")

    # Built directly via mcts_picker (not the match.py spec string) so the
    # eval-config levers (FPU / tree-reuse / proven-prop) can be threaded.
    device = pick_device(os.environ.get("GOMOKU_DEVICE"))
    model, _ = load_checkpoint(checkpoint, device=device)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, device)
    model_picker = mcts_picker(
        evaluator,
        n_simulations=sims,
        c_puct=c_puct,
        fpu_reduction_c=fpu_reduction_c,
        reuse_tree=reuse_tree,
        proven_prop=proven_prop,
    )

    build_ref = _build_ref(rapfi_abs)
    records: list[dict] = []
    for timeout_ms in timeouts:
        engine = ExternalEnginePlayer(
            ExternalEngineConfig(
                cmd=rapfi_abs,
                timeout_ms=timeout_ms,
                label=f"rapfi{timeout_ms}",
                rule=rule,
                board_size=size,
            )
        )
        t0 = time.time()
        try:
            res = play_match_pickers(
                model_picker, engine, n_games=n_games, seed=seed,
                random_opening_moves=random_opening_moves,
            )
        finally:
            engine.close()
        dt = time.time() - t0

        rec = make_eval_record(
            checkpoint=checkpoint,
            sims=sims,
            c_puct=c_puct,
            fpu_reduction_c=fpu_reduction_c,
            reuse_tree=reuse_tree,
            proven_prop=proven_prop,
            timeout_ms=timeout_ms,
            board_size=size,
            rule=rule,
            wrapper_version=WRAPPER_VERSION,
            build_ref=build_ref,
            n_games=res.n_games,
            wins=res.wins,
            losses=res.losses,
            draws=res.draws,
            win_rate=res.win_rate,
            wall_secs=dt,
            extra_fields=extra_fields,
        )
        records.append(rec)
        if print_progress:
            print(
                f"rapfi{timeout_ms}: {res.wins}W-{res.losses}L-{res.draws}D / "
                f"{res.n_games} games  win_rate={res.win_rate:.2%}  ({dt:.1f}s)"
            )
        if out:
            append_jsonl(out, rec)

    return records


# ---------------------------------------------------------------------------
# Parallel (--jobs) path (issue #52): saturate the M5 Max. Rapfi is single-thread,
# so concurrency comes from running many games at once. A persistent spawn-pool
# of `jobs` workers each loads the model ONCE on its own device (MPS is shared
# safely across processes) + builds its own Rapfi subprocess per task, and plays
# a SHARD of color-balanced pairs. We shard at the PAIR level so every shard is
# self-color-balanced; distinct per-shard seeds give independent openings.
# EVAL-ONLY — never touches training. Spawn (macOS default) is required: fork +
# MPS/torch is unsafe.
# ---------------------------------------------------------------------------
_WORKER: dict = {}  # per-process state, populated by _worker_init


def _shard_pairs(total_pairs: int, jobs: int) -> list[int]:
    """Split ``total_pairs`` color-balanced pairs across up to ``jobs`` shards as
    evenly as possible. Returns the pair-count per shard (no zero shards). Pure +
    unit-testable: by construction ``sum(_shard_pairs(p, j)) == p`` for p>=1."""
    shards = max(1, min(jobs, total_pairs))
    base, rem = divmod(total_pairs, shards)
    return [base + (1 if s < rem else 0) for s in range(shards)]


def _worker_init(checkpoint, sims, c_puct, fpu_reduction_c, reuse_tree, proven_prop):
    """Load the model + build the eval picker ONCE per worker process."""
    import os as _os

    from gomoku.eval import mcts_picker
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint
    from gomoku.util import pick_device

    device = pick_device(_os.environ.get("GOMOKU_DEVICE"))
    model, _ = load_checkpoint(checkpoint, device=device)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, device)
    _WORKER["picker"] = mcts_picker(
        evaluator, n_simulations=sims, c_puct=c_puct,
        fpu_reduction_c=fpu_reduction_c, reuse_tree=reuse_tree,
        proven_prop=proven_prop,
    )


def _worker_play(task: dict) -> dict:
    """Play one shard (a slice of color-balanced pairs) at one timeout tier."""
    from gomoku.eval import play_match_pickers
    from gomoku.external_engine import ExternalEngineConfig, ExternalEnginePlayer

    engine = ExternalEnginePlayer(
        ExternalEngineConfig(
            cmd=task["rapfi_abs"], timeout_ms=task["timeout_ms"],
            label=f"rapfi{task['timeout_ms']}", rule=task["rule"],
            board_size=task["size"],
        )
    )
    try:
        res = play_match_pickers(
            _WORKER["picker"], engine, n_games=task["n_games"],
            seed=task["seed"], random_opening_moves=task["random_opening_moves"],
        )
    finally:
        engine.close()
    return {
        "timeout_ms": task["timeout_ms"], "n_games": res.n_games,
        "wins": res.wins, "losses": res.losses, "draws": res.draws,
        "black": [res.black_w, res.black_l, res.black_d],
        "white": [res.white_w, res.white_l, res.white_d],
    }


def run_rapfi_eval_parallel(
    *, checkpoint, rapfi, timeouts, sims=100, c_puct=1.5, n_games=20, seed=0,
    rule=0, size=9, fpu_reduction_c=0.0, reuse_tree=False, proven_prop=False,
    random_opening_moves=0, jobs=4, out=None, extra_fields=None,
    print_progress=True,
) -> list[dict]:
    """Parallel sibling of ``run_rapfi_eval``: same JSONL schema, sharded across
    ``jobs`` worker processes. ``n_games`` must be even (pair color-balance)."""
    import multiprocessing as mp

    from gomoku.board_config import BOARD_SIZE
    from gomoku.external_engine import WRAPPER_VERSION

    if int(size) != int(BOARD_SIZE):
        raise SystemExit(
            f"--size {size} conflicts with active board size {BOARD_SIZE} "
            f"(set GOMOKU_BOARD_SIZE={size} and start a new process)."
        )
    if n_games % 2 != 0:
        raise SystemExit("--n-games must be even for color balance under --jobs.")
    rapfi_abs = os.path.abspath(rapfi)
    if not os.path.exists(rapfi_abs):
        raise SystemExit(f"rapfi binary not found: {rapfi_abs}")

    total_pairs = n_games // 2
    # Build the task list: shard each tier's pairs across up to `jobs` shards.
    tasks: list[dict] = []
    for ti, t in enumerate(timeouts):
        for s, sp in enumerate(_shard_pairs(total_pairs, jobs)):
            if sp == 0:
                continue
            tasks.append({
                "timeout_ms": t, "rapfi_abs": rapfi_abs, "rule": rule, "size": size,
                "n_games": 2 * sp,
                "seed": seed + ti * 100003 + s * 101,  # distinct independent openings
                "random_opening_moves": random_opening_moves,
            })

    ctx = mp.get_context("spawn")
    init_args = (checkpoint, sims, c_puct, fpu_reduction_c, reuse_tree, proven_prop)
    agg: dict = {}
    t0 = time.time()
    with ctx.Pool(processes=jobs, initializer=_worker_init, initargs=init_args) as pool:
        for r in pool.imap_unordered(_worker_play, tasks):
            a = agg.setdefault(r["timeout_ms"], {
                "n_games": 0, "wins": 0, "losses": 0, "draws": 0,
                "black": [0, 0, 0], "white": [0, 0, 0],
            })
            a["n_games"] += r["n_games"]; a["wins"] += r["wins"]
            a["losses"] += r["losses"]; a["draws"] += r["draws"]
            for i in range(3):
                a["black"][i] += r["black"][i]; a["white"][i] += r["white"][i]
    batch_wall = time.time() - t0

    build_ref = _build_ref(rapfi_abs)
    records: list[dict] = []
    for t in timeouts:  # stable, requested order
        a = agg.get(t)
        if not a:
            continue
        wr = (a["wins"] + 0.5 * a["draws"]) / max(a["n_games"], 1)
        ef = dict(extra_fields or {})
        ef.update({"black": a["black"], "white": a["white"], "jobs": jobs,
                   "parallel": True, "batch_wall_secs": round(batch_wall, 2)})
        rec = make_eval_record(
            checkpoint=checkpoint, sims=sims, c_puct=c_puct,
            fpu_reduction_c=fpu_reduction_c, reuse_tree=reuse_tree,
            proven_prop=proven_prop, timeout_ms=t, board_size=size, rule=rule,
            wrapper_version=WRAPPER_VERSION, build_ref=build_ref,
            n_games=a["n_games"], wins=a["wins"], losses=a["losses"],
            draws=a["draws"], win_rate=wr, wall_secs=batch_wall, extra_fields=ef,
        )
        records.append(rec)
        if print_progress:
            bw = a["white"]
            print(
                f"rapfi{t}: {a['wins']}W-{a['losses']}L-{a['draws']}D / "
                f"{a['n_games']} games  win_rate={wr:.2%}  "
                f"| white {bw[0]}-{bw[1]}-{bw[2]} black {a['black'][0]}-{a['black'][1]}-{a['black'][2]}"
            )
        if out:
            append_jsonl(out, rec)
    if print_progress:
        print(f"[parallel] {len(tasks)} shards over {jobs} workers in {batch_wall:.1f}s")
    return records


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
    ap.add_argument("--fpu-reduction-c", type=float, default=0.0,
                    help="KataGo-style FPU reduction (0.0 = OFF, byte-identical). "
                         "0.45 is the verified derby-3w0 eval config.")
    ap.add_argument("--reuse-tree", action="store_true",
                    help="Persistent tree across moves (derby-jmi eval lever).")
    ap.add_argument("--proven-prop", action="store_true",
                    help="Proven win/loss propagation (derby-b3n eval lever).")
    ap.add_argument("--out", default=None, help="JSONL output path (append).")
    ap.add_argument(
        "--jobs", type=int, default=1, metavar="J",
        help=(
            "Parallel worker processes (issue #52). 1 = serial (default). J>1 "
            "shards color-balanced pairs across J spawn-workers (each loads the "
            "model once + its own single-thread Rapfi) to saturate the M5 Max. "
            "Requires even --n-games."
        ),
    )
    ap.add_argument(
        "--random-opening-moves", type=int, default=0, metavar="N",
        help=(
            "Start each game pair from an identical N-stone random opening. "
            "Default 0 (empty board). Recommended: 4-6 for 15x15 freestyle "
            "to neutralise black's first-mover advantage."
        ),
    )
    args = ap.parse_args()

    runner = run_rapfi_eval_parallel if args.jobs > 1 else run_rapfi_eval
    kwargs = dict(
        checkpoint=args.checkpoint,
        rapfi=args.rapfi,
        timeouts=args.timeouts,
        sims=args.sims,
        c_puct=args.c_puct,
        n_games=args.n_games,
        seed=args.seed,
        rule=args.rule,
        size=args.size,
        fpu_reduction_c=args.fpu_reduction_c,
        reuse_tree=args.reuse_tree,
        proven_prop=args.proven_prop,
        random_opening_moves=args.random_opening_moves,
        out=args.out,
    )
    if args.jobs > 1:
        kwargs["jobs"] = args.jobs
    records = runner(**kwargs)

    if args.out:
        print(f"wrote {len(records)} record(s) to {args.out}")


if __name__ == "__main__":
    main()
