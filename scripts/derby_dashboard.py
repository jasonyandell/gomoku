"""Build a native wandb WORKSPACE that compares all Δelo Derby ideas side by side.

A *workspace* is wandb's live, interactive view of a project: a set of PANELS
(charts) grouped into SECTIONS, drawing from a RUN SET (the runs shown + how
they're filtered/colored). Unlike a *report* (a frozen snapshot you write prose
around), a workspace updates live as the runs log more data — so this is the
thing to keep open while the derby races.

What this script does that the picker-by-hand approach doesn't:
  * filters the run set to just the derby runs (regex on run name);
  * finds the ONE clean run per idea (newest — the others are stale dupes from
    earlier launches) and gives each a distinct color, disabling the dupes so the
    overlay isn't cluttered;
  * charts the eval history (model_elo + every anchor winrate, logged to wandb by
    the trainer's jsonl-tail-forward at each eval cycle) + pins eval/peak_elo as a
    table column (the leaderboard — run `derby_sync_elo.py` for the peak scalar,
    which wandb summary doesn't track natively);
  * lays out every comparison stat the trainer logs, grouped by what question it
    answers (losses / game-shape / policy-sharpness / intensity / throughput).

Re-running makes a NEW saved view (the wandb-workspaces Python API can't update
one in place) and prints its URL — open the latest, bookmark it. Old views are
harmless; delete them in the wandb UI if they pile up.

Usage:
    python scripts/derby_dashboard.py
    python scripts/derby_dashboard.py --board scripts/derby_v2_board.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as rpt

REPO = Path(__file__).resolve().parent.parent
ENTITY = "jasonyandell-forge42"
PROJECT = "gomoku"

# A distinct color per idea so the overlay is readable (qualitative palette).
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D",
]


def api_key() -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    try:
        return subprocess.check_output(
            ["security", "find-generic-password", "-s", "wandb-api-key", "-w"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def line(title: str, ys: list[str]) -> rpt.LinePlot:
    # x="Step" = the trainer's epoch axis. Flip to "_runtime" in the UI (per-panel
    # gear, or the workspace x-axis control) to compare wall-clock instead.
    return rpt.LinePlot(title=title, x="Step", y=ys, legend_position="north")


def build_sections() -> list[ws.Section]:
    return [
        # Eval IS in wandb history: the eval_worker writes eval_results.jsonl and the
        # trainer tails+forwards every row to its own run (single-writer), so
        # eval/model_elo + each eval/vs_<anchor>_winrate are logged at every eval
        # cycle (~5 epochs). One LinePlot per eval type → x=Step (iteration), one
        # line per run (experiment). peak_elo bar = the leaderboard (from the summary
        # sync, derby_sync_elo.py). This is "each type of eval at each iteration for
        # each experiment."
        ws.Section(name="1 · Strength — model_elo + every anchor eval (each cycle, each experiment)",
                   panels=[
                       line("model_elo — the headline elo curve (every eval cycle)",
                            ["eval/model_elo"]),
                       rpt.BarPlot(title="peak model_elo (leaderboard)", metrics=["eval/peak_elo"]),
                       line("vs random — winrate", ["eval/vs_random_winrate"]),
                       line("vs heuristic — winrate", ["eval/vs_heuristic_winrate"]),
                       line("vs lookahead:depth=2 — winrate", ["eval/vs_lookahead2_winrate"]),
                       line("vs lookahead:depth=4 — winrate", ["eval/vs_lookahead4_winrate"]),
                   ], is_open=True),
        ws.Section(name="2 · Losses",
                   panels=[
                       line("policy loss", ["loss/policy"]),
                       line("value loss", ["loss/value"]),
                       line("total loss", ["loss/total"]),
                   ], is_open=True),
        ws.Section(name="3 · Game shape — how the lever reshapes self-play",
                   panels=[
                       line("plies — mean / p10 / p90",
                            ["selfplay/plies_mean", "selfplay/plies_p10", "selfplay/plies_p90"]),
                       line("first-mover edge — black vs white wins",
                            ["selfplay/black_wins", "selfplay/white_wins"]),
                       line("draws", ["selfplay/draws"]),
                   ], is_open=True),
        ws.Section(name="4 · Policy quality — target sharpness (the gumbel / forced / playoutcap signal)",
                   panels=[
                       line("entropy — MCTS target vs net",
                            ["train/policy_target_entropy", "train/policy_net_entropy"]),
                       line("policy KL (net ‖ target)", ["train/policy_kl"]),
                       line("policy accuracy (argmax match)", ["train/policy_acc"]),
                   ]),
        ws.Section(name="5 · Training intensity",
                   panels=[
                       line("SGD steps / cycle", ["train/steps_this_cycle"]),
                       line("positions / cycle", ["train/positions_this_cycle"]),
                       line("actual SGD per position", ["train/actual_sgd_per_position"]),
                   ]),
        ws.Section(name="6 · Throughput & generation",
                   panels=[
                       line("new games / cycle", ["selfplay/new_games"]),
                       line("new examples / cycle", ["selfplay/new_examples"]),
                   ]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default=str(REPO / "scripts" / "derby_v3_board.json"))
    ap.add_argument("--name", default="Δelo Derby v3 — lever comparison (live)")
    args = ap.parse_args()

    key = api_key()
    if key:
        os.environ.setdefault("WANDB_API_KEY", key)

    board = json.loads(Path(args.board).read_text())
    ideas = [(i["name"], i["cell_name"]) for i in board["ideas"]]
    want_names = {f"9x9-sweep-{cell}": (name, cell) for name, cell in ideas}

    # Identify the clean (newest) run per idea + the stale dupes to grey out.
    import wandb
    api = wandb.Api()
    path = f"{ENTITY}/{PROJECT}"
    clean: dict[str, str] = {}     # idea_name -> run_id (newest)
    dupes: list[str] = []
    for r in api.runs(path, order="-created_at", per_page=300):
        meta = want_names.get(r.name)
        if not meta:
            continue
        name = meta[0]
        if name not in clean:
            clean[name] = r.id      # first seen (newest) is the live one
        else:
            dupes.append(r.id)      # older same-named run → disable in the overlay

    run_settings: dict[str, ws.RunSettings] = {}
    for i, (name, _) in enumerate(ideas):
        rid = clean.get(name)
        if rid:
            run_settings[rid] = ws.RunSettings(color=PALETTE[i % len(PALETTE)])
    for rid in dupes:
        run_settings[rid] = ws.RunSettings(disabled=True)

    runset = ws.RunsetSettings(
        query="9x9-sweep-derby-",   # the v3 run-name prefix (excludes the v1 derby-* runs)
        regex_query=True,
        run_settings=run_settings,
        # surface the elo leaderboard + key shape columns in the runs table
        pinned_columns=["eval/peak_elo", "eval/model_elo", "eval/beat_heuristic",
                        "selfplay/plies_mean", "loss/policy"],
    )

    workspace = ws.Workspace(
        entity=ENTITY, project=PROJECT, name=args.name,
        sections=build_sections(),
        runset_settings=runset,
    )
    saved = workspace.save()
    print(f"\nDashboard URL (bookmark this):\n  {saved.url}\n")
    print(f"clean runs shown ({len(clean)}/{len(ideas)} ideas have a run so far):")
    for name, _ in ideas:
        rid = clean.get(name)
        print(f"  {name:<12} {'→ ' + rid if rid else '— (not started yet; appears when its round-0 runs)'}")
    if dupes:
        print(f"greyed-out stale dupes: {len(dupes)}")
    print("\nTip: run `python scripts/derby_sync_elo.py` first/again to refresh the elo columns,")
    print("then hit refresh in the browser. Flip any panel's x-axis to _runtime for wall-clock.")


if __name__ == "__main__":
    main()
