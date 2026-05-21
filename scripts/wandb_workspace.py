"""Create or refresh the gomoku wandb workspace for WL1-vs-Z monitoring.

Layout: 6 sections (strength / learning / game shape / buffer / wave / wall),
preconfigured to overlay both runs. Run this once; it prints a URL that
opens the same workspace every time.

Re-running creates a *new* workspace view (wandb-workspaces doesn't update
in place from this Python API as of 0.3.9). The old views stay; clean
them up via the wandb UI if they accumulate.
"""
from __future__ import annotations

import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as rpt

ENTITY = "jasonyandell-forge42"
PROJECT = "gomoku"

# Runs to overlay. Add new run IDs here as they become interesting.
# The view itself doesn't pin runs (workspaces API doesn't support that cleanly
# from Python); these are the run IDs you click in the picker after opening.
RUNS = {
    "WL2 (scale-emulation: EMA + past-mix + jitter + grad-accum)": "9wng4yu9",
    "WL1 (wave-lockstep, native)": "l8mbntcm",
    "Z (continuous, python MCTS)": "sppjo3z5",
}


def line(title: str, ys: list[str], **kwargs) -> rpt.LinePlot:
    return rpt.LinePlot(title=title, x="Step", y=ys, legend_position="north", **kwargs)


def build_sections() -> list[ws.Section]:
    return [
        # 1. Strength vs fixed baselines (headline)
        ws.Section(
            name="1. Strength vs fixed baselines",
            panels=[
                line("vs heuristic", ["eval/vs_heuristic_winrate"]),
                line("vs lookahead2 / lookahead4",
                     ["eval/vs_lookahead2_winrate", "eval/vs_lookahead4_winrate"]),
                line("model_elo", ["model_elo"]),
            ],
        ),
        # 2. Learning dynamics
        ws.Section(
            name="2. Learning dynamics",
            panels=[
                line("policy loss", ["loss/policy"]),
                line("value loss", ["loss/value"]),
            ],
        ),
        # 3. Game shape — Jason flagged plies_p90 as the leading defense indicator
        ws.Section(
            name="3. Game shape (defense regime signal)",
            panels=[
                line("plies — mean / p10 / p90",
                     ["selfplay/plies_mean", "selfplay/plies_p10", "selfplay/plies_p90"]),
                line("first-mover wins (black vs white)",
                     ["selfplay/black_wins", "selfplay/white_wins"]),
            ],
        ),
        # 4. Buffer health
        ws.Section(
            name="4. Buffer health",
            panels=[
                line("buffer age — mean / p50 / p90",
                     ["buffer/age_mean", "buffer/age_p50", "buffer/age_p90"]),
                line("fraction from current version", ["buffer/frac_current"]),
                line("stone-count buckets",
                     ["buffer/stones_00_05", "buffer/stones_05_10",
                      "buffer/stones_10_15", "buffer/stones_15_20",
                      "buffer/stones_20_30", "buffer/stones_30_40",
                      "buffer/stones_40_60", "buffer/stones_60_81"]),
            ],
        ),
        # 5. Wave dynamics — WL1-only; Z runs will show no data here
        ws.Section(
            name="5. Wave dynamics (WL1 only)",
            panels=[
                line("barrier wait — slowest worker / total",
                     ["wave/wait_for_slowest_s", "wave/total_s"]),
                line("games per worker — min / max / mean",
                     ["wave/games_per_worker_min", "wave/games_per_worker_max",
                      "wave/games_per_worker_mean"]),
                line("greedy extras per wave", ["wave/greedy_extras_count"]),
            ],
        ),
        # 6. Wall economy — switch x-axis to _runtime in the UI to see total time savings
        ws.Section(
            name="6. Wall economy",
            panels=[
                line("per-cycle time (gen vs train)",
                     ["time/gen_s", "time/train_s"]),
                line("cumulative games", ["total_games"]),
            ],
        ),
    ]


def main() -> None:
    runset = ws.RunsetSettings(
        # Filter to the two runs we care about. Add more here for follow-up runs.
        filters=[
            ws.FilterExpr(
                expr=" or ".join(f'Name == "9x9-sweep-{label.split()[0]}"' if False else f'ID == "{rid}"'
                                  for label, rid in RUNS.items())
            )
        ] if False else [],  # noqa: simpler: don't filter, let user pick in UI
    )

    workspace = ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name="WL2 vs WL1 vs Z — live monitoring",
        sections=build_sections(),
    )
    saved = workspace.save()
    print(f"workspace URL:\n  {saved.url}")
    print("\nTo overlay all three runs in this workspace:")
    print(f"  - top-left run picker, select: {', '.join(RUNS.values())}")
    print(f"  - or set the filter to: Run.id in [{', '.join(repr(r) for r in RUNS.values())}]")


if __name__ == "__main__":
    main()
