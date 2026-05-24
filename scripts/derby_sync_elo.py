#!/usr/bin/env python3
"""Sync derby model_elo from eval_results.jsonl INTO each idea's wandb run summary.

NOTE (corrected 2026-05-24): eval/model_elo + every eval/vs_<anchor>_winrate ARE
already in wandb HISTORY — the trainer tails eval_results.jsonl and forwards each row
to its run (single-writer), even under --no-eval. This script is NOT needed for that;
it only adds the PEAK elo + beat-heuristic as SUMMARY scalars (wandb summary keeps the
LAST value, not the peak), which the dashboard pins as a leaderboard column.

(Original rationale, now partly obsolete:) the derby trainer runs ``--no-eval``, so ``eval/model_elo`` is
NEVER logged to wandb — it's computed by the bundle's separate ``--final-eval``
step and written to ``sweep_runs/<cell>/checkpoints/eval_results.jsonl``. This
script reads that jsonl and writes the latest + peak elo (and the anchor winrates)
into the matching wandb run's **summary** via the public API.

Why summary (not history): wandb history is append-only by increasing step, so
backfilling elo at past epochs would be dropped as out-of-order. Summary is
last-write-wins and — crucially — the trainer never writes ``eval/*`` keys, so
there is ZERO conflict even while a run is actively training. The result is a
per-run scalar you can compare directly (a leaderboard) in the dashboard's runs
table / bar panels.

Re-run any time to refresh (idempotent). Pairs with scripts/derby_dashboard.py.

Usage:
    python scripts/derby_sync_elo.py                      # v3 board
    python scripts/derby_sync_elo.py --board scripts/derby_v2_board.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROJECT = "gomoku"


def api_key() -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    try:
        return subprocess.check_output(
            ["security", "find-generic-password", "-s", "wandb-api-key", "-w"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def read_elo_series(cell_name: str) -> dict | None:
    f = REPO / "sweep_runs" / cell_name / "checkpoints" / "eval_results.jsonl"
    if not f.exists():
        return None
    rows = []
    for line in f.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    elos = [(r.get("eval_worker/epoch_evaluated"), r.get("eval/model_elo")) for r in rows
            if r.get("eval/model_elo") is not None]
    if not elos:
        return None
    last = rows[-1]
    return {
        "eval/model_elo": elos[-1][1],
        "eval/peak_elo": max(e for _, e in elos),
        "eval/last_epoch": elos[-1][0],
        "eval/n_evals": len(elos),
        "eval/vs_heuristic_winrate": last.get("eval/vs_heuristic_winrate"),
        "eval/vs_lookahead2_winrate": last.get("eval/vs_lookahead2_winrate"),
        "eval/vs_lookahead4_winrate": last.get("eval/vs_lookahead4_winrate"),
        "eval/beat_heuristic": 1.0 if max(e for _, e in elos) >= 800 else 0.0,
    }


def newest_run_per_idea(api, ideas):
    """Map cell_name -> newest wandb run (the live/clean one), by run name
    `9x9-sweep-<cell_name>`, newest-created wins."""
    path = f"{api.default_entity}/{PROJECT}"
    want = {f"9x9-sweep-{cell}": cell for _, cell, _ in ideas}
    found: dict[str, object] = {}
    for r in api.runs(path, order="-created_at", per_page=200):
        cell = want.get(r.name)
        if cell and cell not in found:
            found[cell] = r
        if len(found) >= len(want):
            break
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=str(REPO / "scripts" / "derby_v3_board.json"))
    args = ap.parse_args()

    key = api_key()
    if key:
        os.environ.setdefault("WANDB_API_KEY", key)
    import wandb
    api = wandb.Api()

    board = json.loads(Path(args.board).read_text())
    ideas = [(i["name"], i["cell_name"], i.get("lever", "")) for i in board["ideas"]]
    runs = newest_run_per_idea(api, ideas)

    print(f"syncing model_elo → wandb summary ({api.default_entity}/{PROJECT}):")
    for name, cell, _ in ideas:
        e = read_elo_series(cell)
        r = runs.get(cell)
        if e is None:
            print(f"  {name:<12} — no elo yet (no eval_results.jsonl)")
            continue
        if r is None:
            print(f"  {name:<12} elo={e['eval/model_elo']:.0f} — NO wandb run yet")
            continue
        for k, v in e.items():
            if v is not None:
                r.summary[k] = v
        r.summary.update()  # flush to the backend
        print(f"  {name:<12} elo={e['eval/model_elo']:>5.0f}  peak={e['eval/peak_elo']:>5.0f}  "
              f"ep={e['eval/last_epoch']}  → {r.name} ({r.id})")
    print("\ndone. Refresh the dashboard (or its runs table) to see eval/model_elo + eval/peak_elo.")


if __name__ == "__main__":
    main()
