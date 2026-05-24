#!/usr/bin/env python3
"""Watch a Δelo Derby live: wandb training dynamics + the anchored-elo scoreboard.

Why this fuses two sources: the derby logs each idea as a wandb run named
``9x9-sweep-<cell_name>`` carrying TRAINING dynamics (loss / plies / throughput),
but the trainer runs ``--no-eval`` so ``eval/model_elo`` is NOT in wandb — it's
written per wall-slice to ``sweep_runs/<cell_name>/checkpoints/eval_results.jsonl``
by the bundle's ``--final-eval``. The real scoreboard column (elo) therefore comes
from the jsonl; the training-health columns come from wandb. This script merges
them into one board, sorted by current model_elo.

Auth: uses $WANDB_API_KEY if set, else pulls it from the macOS keychain
(``security find-generic-password -s wandb-api-key -w``).

Usage:
    python scripts/watch_derby.py                      # one-shot, v3 board
    python scripts/watch_derby.py --watch 30           # live, refresh every 30s
    python scripts/watch_derby.py --board scripts/derby_v2_board.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROJECT = "gomoku"
ELO_FLOOR = 389.0  # v1/v2 empirical fresh-init floor; Δelo/hr is noise below this


def get_api_key() -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    try:
        return subprocess.check_output(
            ["security", "find-generic-password", "-s", "wandb-api-key", "-w"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def load_board(path: str) -> tuple[dict, list[tuple[str, str, str]]]:
    b = json.loads(Path(path).read_text())
    ideas = [(i["name"], i["cell_name"], i.get("lever", "")) for i in b["ideas"]]
    return b.get("global", {}), ideas


def read_elo(cell_name: str) -> dict:
    """Latest + peak model_elo and the epoch it was evaluated at, from the
    per-cell eval_results.jsonl (the authoritative elo source)."""
    f = REPO / "sweep_runs" / cell_name / "checkpoints" / "eval_results.jsonl"
    if not f.exists():
        return {}
    elos, last = [], None
    for line in f.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        e = d.get("eval/model_elo")
        if e is not None:
            elos.append(e)
            last = d
    if not elos:
        return {}
    return {"elo": elos[-1], "peak": max(elos),
            "ep": (last or {}).get("eval_worker/epoch_evaluated"),
            "n_evals": len(elos)}


def fetch_wandb(run_names: set[str]) -> tuple[dict, str | None, str | None]:
    """Return {run_name: Run} (newest per display_name), the project URL, and an
    error string (None on success)."""
    try:
        import wandb
        api = wandb.Api(timeout=20)
        entity = api.default_entity
        path = f"{entity}/{PROJECT}" if entity else PROJECT
        proj_url = f"https://wandb.ai/{entity}/{PROJECT}" if entity else None
        by_name: dict = {}
        # order desc by creation → first occurrence of each name is the newest run
        # (v3 reuses v2 cell names for the carryovers, so dedupe to the active one).
        for r in api.runs(path, order="-created_at", per_page=200):
            if r.name in run_names and r.name not in by_name:
                by_name[r.name] = r
            if len(by_name) >= len(run_names):
                break
        return by_name, proj_url, None
    except Exception as e:  # offline / auth / network — degrade to elo-only
        return {}, None, str(e)


def _num(summary, key):
    try:
        return float(summary.get(key))
    except Exception:
        return None


def _fmt(v, spec, dash="—"):
    return format(v, spec) if isinstance(v, (int, float)) else dash


def render(board_path: str) -> None:
    g, ideas = load_board(board_path)
    cap_h = g.get("cap_wall_secs", 0) / 3600.0
    run_names = {f"9x9-sweep-{cell}": (name, cell, lever) for name, cell, lever in ideas}
    runs, proj_url, werr = fetch_wandb(set(run_names))

    rows = []
    for rn, (name, cell, lever) in run_names.items():
        e = read_elo(cell)
        r = runs.get(rn)
        s = r.summary if r is not None else {}
        rows.append({
            "idea": name,
            "elo": e.get("elo"), "peak": e.get("peak"), "ep": e.get("ep"),
            "state": (r.state if r is not None else "—"),
            "pl": _num(s, "loss/policy"), "vl": _num(s, "loss/value"),
            "plies": _num(s, "selfplay/plies_mean"),
            "step": _num(s, "_step"),
            "wall_min": (_num(s, "_runtime") or 0) / 60.0 if r is not None else None,
            "url": (r.url if r is not None else None),
            "lever": lever,
        })
    rows.sort(key=lambda x: (x["elo"] is not None, x["elo"] or -1), reverse=True)

    board = Path(board_path).stem.replace("_board", "")
    print(f"\n  Δelo Derby — {board}   (cap {cap_h:.1f}h/idea · elo from eval_results.jsonl, dynamics from wandb)")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}   project: {proj_url or '(wandb unavailable)'}")
    if werr:
        print(f"  ⚠ wandb: {werr.splitlines()[0][:80]} — showing elo-only")
    print("  " + "-" * 96)
    print(f"  {'#':>2} {'idea':<12} {'elo':>6} {'peak':>6} {'ep':>4} {'state':<8} "
          f"{'pl':>6} {'vl':>6} {'plies':>6} {'step':>6} {'wall_m':>6}")
    print("  " + "-" * 96)
    for i, x in enumerate(rows, 1):
        beat = "✓" if (x["peak"] or 0) >= 800 else " "
        print(f"  {i:>2} {x['idea']:<12} {_fmt(x['elo'], '6.0f')} {_fmt(x['peak'], '6.0f')}{beat}"
              f"{_fmt(x['ep'], '4.0f')} {x['state']:<8} "
              f"{_fmt(x['pl'], '6.3f')} {_fmt(x['vl'], '6.3f')} {_fmt(x['plies'], '6.1f')} "
              f"{_fmt(x['step'], '6.0f')} {_fmt(x['wall_min'], '6.1f')}")
    print("  " + "-" * 96)
    lead = next((x for x in rows if x["elo"] is not None), None)
    if lead:
        print(f"  leader: {lead['idea']} @ {lead['elo']:.0f} elo"
              f"{'  (beat-heuristic ✓)' if (lead['peak'] or 0) >= 800 else ''}   "
              f"— {lead['lever'][:60]}")
    print("  elo: latest eval_results.jsonl · peak ✓=≥800 (beat-heuristic) · "
          "pl/vl/plies/step from the live wandb run\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch the Δelo Derby (wandb + elo).")
    ap.add_argument("--board", default=str(REPO / "scripts" / "derby_v3_board.json"))
    ap.add_argument("--watch", type=float, default=0.0,
                    help="refresh interval in seconds (0 = print once and exit)")
    args = ap.parse_args()

    key = get_api_key()
    if key:
        os.environ.setdefault("WANDB_API_KEY", key)

    if args.watch > 0:
        try:
            while True:
                os.system("clear")
                render(args.board)
                print(f"  (refreshing every {args.watch:.0f}s — Ctrl-C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        render(args.board)


if __name__ == "__main__":
    main()
