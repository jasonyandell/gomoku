#!/usr/bin/env python3
"""Watch a Δelo Derby live: the elo scoreboard + the Δelo/HOUR hill-climb signal
+ wandb training dynamics, in one board.

Three sources, each for what it's best at:
  * ``<base_out_dir>/derby_state.json`` (the derby's own state) — authoritative for
    per-idea elo history, chunks, status, and per-chunk wall-time. This is what the
    scheduler reads, so the Δelo/hr column + the "next pick" line below MATCH the
    real priority decision exactly (computed via delo_derby.pick_priority).
  * ``sweep_runs/<cell>/checkpoints/eval_results.jsonl`` — elo fallback if state is
    missing. (model_elo IS in wandb history too — the trainer forwards the eval
    jsonl to its run — but reading the jsonl/state directly is simpler here.)
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


def _etime_to_secs(et: str) -> float:
    """Parse `ps -o etime` ([[DD-]HH:]MM:SS) into seconds."""
    et = et.strip()
    if not et:
        return 0.0
    days = 0
    if "-" in et:
        d, et = et.split("-", 1)
        days = int(d)
    parts = [int(p) for p in et.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return days * 86400 + h * 3600 + m * 60 + s


def running_slice_secs(cell: str) -> float:
    """Wall elapsed of the in-flight run_sweep slice for this cell (0 if none is
    running) — lets wall_m count the CURRENT chunk live, not just closed slices."""
    try:
        pid = subprocess.check_output(
            ["pgrep", "-f", f"run_sweep.py --cell {cell} "],
            text=True, stderr=subprocess.DEVNULL).split()[0]
        et = subprocess.check_output(
            ["ps", "-o", "etime=", "-p", pid], text=True, stderr=subprocess.DEVNULL)
        return _etime_to_secs(et)
    except Exception:
        return 0.0


LIVE_RATE_WINDOW = 10  # trailing fine-grained eval points for the live elo/hr slope


def read_jsonl(cell_name: str) -> tuple[list, dict]:
    """Return ([(ts, model_elo), ...], latest full eval row) from the cell's
    eval_results.jsonl. The (ts, elo) series feeds both the freshest-strength
    columns AND the LIVE elo/hr slope (so the rate moves mid-slice, not only when
    the coarse per-slice scheduler state updates)."""
    f = REPO / "sweep_runs" / cell_name / "checkpoints" / "eval_results.jsonl"
    if not f.exists():
        return [], {}
    pts, last = [], {}
    for line in f.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        last = d
        if d.get("eval/model_elo") is not None:
            pts.append((d.get("ts"), d["eval/model_elo"]))
    return pts, last


def live_delo_per_hr(pts: list) -> float | None:
    """Live Δelo/HOUR = least-squares slope of model_elo vs wall-clock over the last
    LIVE_RATE_WINDOW fine-grained eval points. Reflects the CURRENT climb even
    mid-slice — unlike the scheduler's slice-close rate (one point per finished
    slice), which lags a full slice behind (e.g. shows '1pt' while an idea is
    actually rocketing up within its running chunk)."""
    series = [(t, e) for t, e in pts if isinstance(t, (int, float))]
    if len(series) < 2:
        return None
    w = series[-LIVE_RATE_WINDOW:]
    t0 = w[0][0]
    xs = [(t - t0) / 3600.0 for t, _ in w]
    ys = [e for _, e in w]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def idea_metrics(state: dict, name: str, cell: str) -> dict:
    """elo/peak = freshest/best fine-grained eval; rate = LIVE elo/hr slope from the
    fine-grained eval series (moves mid-slice); chunks/status/wall from the state."""
    pts, last = read_jsonl(cell)
    st = (state.get("ideas") or {}).get(name) or {}
    ep = last.get("eval_worker/epoch_evaluated")
    if pts:
        elos = [e for _, e in pts]
        elo, peak, rate = elos[-1], max(elos), live_delo_per_hr(pts)
    elif st.get("elo_history"):
        elos = [h[1] for h in st["elo_history"]]
        elo, peak, rate = elos[-1], max(elos), None
    else:
        elo = peak = rate = None
    # wall = closed-slice total + the in-flight slice's live elapsed (running idea only)
    wall = float(st.get("wall_secs_total", 0.0))
    if st.get("status") == "running":
        wall += running_slice_secs(cell)
    return {"elo": elo, "peak": peak, "pts": len(pts), "rate": rate, "ep": ep,
            "chunks": st.get("chunks_done", 0), "status": st.get("status", "—"),
            "wall_min": wall / 60.0}


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
    print("  " + "-" * 108)
    print(f"  {'#':>2} {'idea':<11} {'elo':>5} {'peak':>5} {'e':>4} {'Δelo/hr':>8} {'chk':>3} "
          f"{'status':<9} {'pl':>6} {'vl':>6} {'plies':>6} {'wall_m':>6}")
    print("  " + "-" * 108)
    for i, x in enumerate(rows, 1):
        beat = "✓" if (x["peak"] or 0) >= 800 else " "
        run_mark = "►" if x["status"] == "running" else " "
        # live elo/hr slope from the fine-grained eval series ('—' = <2 eval points)
        rate = _fmt(x["rate"], "8.1f") if x["rate"] is not None else "—"
        print(f"  {i:>2} {run_mark}{x['idea']:<10} {_fmt(x['elo'], '5.0f')} {_fmt(x['peak'], '5.0f')}{beat}"
              f"{_fmt(x['ep'], '4.0f')} {rate:>8} {_fmt(x['chunks'], '3.0f')} {x['status']:<9} "
              f"{_fmt(x['pl'], '6.3f')} {_fmt(x['vl'], '6.3f')} {_fmt(x['plies'], '6.1f')} "
              f"{_fmt(x['wall_min'], '6.1f')}")
    print("  " + "-" * 108)
    # footer: one status line (leader · next pick) + one compact legend line
    bits = []
    lead = next((x for x in rows if x["elo"] is not None), None)
    if lead:
        bits.append(f"leader: {lead['idea']} @ {lead['elo']:.0f}"
                    f"{' ✓' if (lead['peak'] or 0) >= 800 else ''}")
    if next_pick:
        nr = _derby.delo_per_hr(state, next_pick) if _derby else None
        bits.append(f"next: {next_pick} ({'entry-fee' if nr is None else f'Δelo/hr {nr:.0f}'})")
    if bits:
        print("  " + "    ·    ".join(bits))
    print("  e=epoch · elo/Δelo·hr/wall_m are LIVE (mid-slice) · ✓=beat-heuristic · "
          "scheduler ranks on slice-close (next may lag)\n")


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
