#!/usr/bin/env python3
"""Watch a Δelo Derby live: the elo scoreboard + the Δelo/HOUR hill-climb signal
+ wandb training dynamics, in one board.

Three sources, each for what it's best at:
  * ``<base_out_dir>/derby_state.json`` (the derby's own state) — authoritative for
    per-idea elo history, chunks, status, and per-chunk wall-time. This is what the
    scheduler reads, so the Δelo/hr column + the "next pick" line below MATCH the
    real priority decision exactly (computed via delo_derby.pick_priority).
  * ``sweep_runs/<cell>/checkpoints/eval_results.jsonl`` — elo fallback if state is
    missing (model_elo is NOT in wandb: the trainer runs ``--no-eval``).
  * wandb run ``9x9-sweep-<cell_name>`` — live TRAINING dynamics (loss / plies / step).

Priority recap (what the Δelo/hr column drives): never-run / entry-fee first (an idea
needs 2 elo points to have a slope → round-0 then round-1 for all), THEN highest
Δelo/HOUR — hill-climb the steepest recent elo gradient.

Usage:
    python scripts/watch_derby.py                  # one-shot, v3 board
    python scripts/watch_derby.py --watch 30       # live, refresh every 30s
    python scripts/watch_derby.py --board scripts/derby_v2_board.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
PROJECT = "gomoku"

# Import the derby's OWN scheduler functions so our Δelo/hr + next-pick match it.
sys.path.insert(0, str(SCRIPTS))
try:
    import delo_derby as _derby  # noqa: E402
except Exception:
    _derby = None


def get_api_key() -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    try:
        return subprocess.check_output(
            ["security", "find-generic-password", "-s", "wandb-api-key", "-w"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_board(path: str) -> tuple[dict, dict, list[tuple[str, str, str]]]:
    b = json.loads(Path(path).read_text())
    g = b.get("global", {})
    ideas = [(i["name"], i["cell_name"], i.get("lever", "")) for i in b["ideas"]]
    return b, g, ideas


def load_state(base_out_dir: str) -> dict:
    f = REPO / base_out_dir / "derby_state.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return {}
    return {}


def elo_from_jsonl(cell_name: str) -> dict:
    """Fallback elo when derby_state.json has no history yet."""
    f = REPO / "sweep_runs" / cell_name / "checkpoints" / "eval_results.jsonl"
    if not f.exists():
        return {}
    elos = []
    for line in f.read_text().splitlines():
        try:
            e = json.loads(line).get("eval/model_elo")
        except Exception:
            continue
        if e is not None:
            elos.append(e)
    return {"elo": elos[-1], "peak": max(elos), "pts": len(elos)} if elos else {}


def idea_metrics(state: dict, name: str, cell: str) -> dict:
    """elo / peak / Δelo-per-hr / chunks / status / pts, from state (jsonl fallback)."""
    st = (state.get("ideas") or {}).get(name)
    if st and st.get("elo_history"):
        hist = st["elo_history"]
        elos = [h[1] for h in hist]
        rate = _derby.delo_per_hr(state, name) if _derby else None
        return {"elo": elos[-1], "peak": max(elos), "pts": len(elos),
                "rate": rate, "chunks": st.get("chunks_done", len(hist)),
                "status": st.get("status", "?"),
                "wall_min": st.get("wall_secs_total", 0.0) / 60.0,
                "ep": hist[-1][0]}
    fb = elo_from_jsonl(cell)
    return {"elo": fb.get("elo"), "peak": fb.get("peak"), "pts": fb.get("pts", 0),
            "rate": None, "chunks": (st or {}).get("chunks_done", 0),
            "status": (st or {}).get("status", "—"),
            "wall_min": (st or {}).get("wall_secs_total", 0.0) / 60.0, "ep": None}


def fetch_wandb(run_names: set[str]) -> tuple[dict, str | None]:
    try:
        import wandb
        api = wandb.Api(timeout=20)
        entity = api.default_entity
        path = f"{entity}/{PROJECT}" if entity else PROJECT
        proj_url = f"https://wandb.ai/{entity}/{PROJECT}" if entity else None
        by_name: dict = {}
        for r in api.runs(path, order="-created_at", per_page=200):
            if r.name in run_names and r.name not in by_name:
                by_name[r.name] = r
            if len(by_name) >= len(run_names):
                break
        return by_name, proj_url
    except Exception:
        return {}, None


def _num(summary, key):
    try:
        return float(summary.get(key))
    except Exception:
        return None


def _fmt(v, spec, dash="—"):
    return format(v, spec) if isinstance(v, (int, float)) else dash


def render(board_path: str) -> None:
    board, g, ideas = load_board(board_path)
    cap_h = g.get("cap_wall_secs", 0) / 3600.0
    state = load_state(g.get("base_out_dir", ""))
    run_names = {f"9x9-sweep-{cell}": (name, cell, lever) for name, cell, lever in ideas}
    runs, proj_url = fetch_wandb(set(run_names))

    rows = []
    for name, cell, lever in ideas:
        m = idea_metrics(state, name, cell)
        r = runs.get(f"9x9-sweep-{cell}")
        s = r.summary if r is not None else {}
        m.update({
            "idea": name, "lever": lever,
            "pl": _num(s, "loss/policy"), "vl": _num(s, "loss/value"),
            "plies": _num(s, "selfplay/plies_mean"), "step": _num(s, "_step"),
        })
        rows.append(m)
    # scoreboard order: by elo desc (None last)
    rows.sort(key=lambda x: (x["elo"] is not None, x["elo"] or -1), reverse=True)

    # the scheduler's actual next pick (matches the real priority exactly)
    next_pick = None
    if _derby is not None and state.get("ideas"):
        try:
            cand = _derby.active_ideas(board, state)
            if cand:
                next_pick = _derby.pick_priority(board, state, cand)
        except Exception:
            next_pick = None

    name_t = Path(board_path).stem.replace("_board", "")
    print(f"\n  Δelo Derby — {name_t}   (cap {cap_h:.1f}h/idea · priority: never-run → Δelo/hr hill-climb)")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}   {proj_url or '(wandb unavailable)'}")
    print("  " + "-" * 104)
    print(f"  {'#':>2} {'idea':<11} {'elo':>5} {'peak':>5} {'Δelo/hr':>8} {'chk':>3} "
          f"{'status':<9} {'pl':>6} {'vl':>6} {'plies':>6} {'wall_m':>6}")
    print("  " + "-" * 104)
    for i, x in enumerate(rows, 1):
        beat = "✓" if (x["peak"] or 0) >= 800 else " "
        run_mark = "►" if x["status"] == "running" else " "
        rate = ("entry" if x["rate"] is None and (x["pts"] or 0) < 2
                else _fmt(x["rate"], "8.1f"))
        print(f"  {i:>2} {run_mark}{x['idea']:<10} {_fmt(x['elo'], '5.0f')} {_fmt(x['peak'], '5.0f')}{beat}"
              f"{rate:>8} {_fmt(x['chunks'], '3.0f')} {x['status']:<9} "
              f"{_fmt(x['pl'], '6.3f')} {_fmt(x['vl'], '6.3f')} {_fmt(x['plies'], '6.1f')} "
              f"{_fmt(x['wall_min'], '6.1f')}")
    print("  " + "-" * 104)
    lead = next((x for x in rows if x["elo"] is not None), None)
    if lead:
        print(f"  leader: {lead['idea']} @ {lead['elo']:.0f} elo"
              f"{'  (beat-heuristic ✓)' if (lead['peak'] or 0) >= 800 else ''}")
    if next_pick:
        nr = _derby.delo_per_hr(state, next_pick) if _derby else None
        why = "entry-fee (needs a 2nd point for a slope)" if nr is None else f"steepest Δelo/hr = {nr:.1f}"
        print(f"  ► next pick (live priority): {next_pick}  — {why}")
    print("  ►=running · Δelo/hr=last-chunk slope (the hill-climb signal) · "
          "'entry'=<2 elo pts · peak ✓=≥800\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch the Δelo Derby (elo + Δelo/hr + wandb).")
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
