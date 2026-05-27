"""Create the gomoku Derby v9 wandb workspace — net-capacity ladder monitoring.

Overlays the three v9 runs (small / medium / large), all the same champion
recipe, model-size the only difference. Run this once; it prints a URL.

IMPORTANT caveats baked into the layout (learned 2026-05-27):
  - `eval/model_elo` is logged by the DECOUPLED CPU eval-worker, which LAGS the
    trainer — and for the LARGE net it stalls hard (frozen ~epoch 45 while the
    trainer is at epoch 240+). So the strength panel is NOT a fair live readout;
    the authoritative per-chunk elo is sweep_runs/derby_v9/derby_state.json, and
    head-to-head round_robin is the real verdict (anchored elo is a mirage).
  - The training-dynamics panels (loss / plies / buffer / time) ARE logged every
    epoch and are honest live signals — large DOES move on these.
  - Default x-axis is Step (= epoch). Large does ~1/6 the epochs of small per
    300s chunk, so on a Step axis its line is SHORT, not stalled. Switch the
    x-axis to `Relative Time (Wall)` / `_runtime` in the UI for the fair
    Δelo-per-wall comparison.
"""
from __future__ import annotations

import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as rpt

ENTITY = "jasonyandell-forge42"
PROJECT = "gomoku"

# The three v9 lanes (run ids are stable across chunks — embedded in the ckpt).
RUNS = {
    "small (64f/4blk — champion net, fresh control)": "b713kz9l",
    "medium (96f/6blk, ~2.3x params)": "ueo994qq",
    "large (128f/10blk, ~6x params)": "z0i6qs0x",
}


def line(title: str, ys: list[str], **kwargs) -> rpt.LinePlot:
    return rpt.LinePlot(title=title, x="Step", y=ys, legend_position="north", **kwargs)


def build_sections() -> list[ws.Section]:
    return [
        # 1. Strength — CAVEAT: eval-worker lags (large frozen ~e45). Real elo =
        #    derby_state.json + round_robin. Kept here for the lanes that DO track.
        ws.Section(
            name="1. Strength vs baselines (NOTE: eval lags; large frozen ~e45 — see derby_state.json)",
            panels=[
                line("eval/model_elo (LAGGY — real elo is derby scoreboard)", ["eval/model_elo"]),
                line("vs lookahead2 / lookahead4 winrate",
                     ["eval/vs_lookahead2_winrate", "eval/vs_lookahead4_winrate"]),
                line("vs heuristic winrate", ["eval/vs_heuristic_winrate"]),
            ],
        ),
        # 2. Learning dynamics — honest live signal; large moves here.
        ws.Section(
            name="2. Learning dynamics (live, honest — switch x-axis to wall to compare fairly)",
            panels=[
                line("policy loss", ["loss/policy"]),
                line("value loss", ["loss/value"]),
            ],
        ),
        # 3. Game shape — defense regime.
        ws.Section(
            name="3. Game shape (defense regime)",
            panels=[
                line("plies — mean / p10 / p90",
                     ["selfplay/plies_mean", "selfplay/plies_p10", "selfplay/plies_p90"]),
                line("first-mover wins (black vs white)",
                     ["selfplay/black_wins", "selfplay/white_wins"]),
            ],
        ),
        # 4. Buffer health.
        ws.Section(
            name="4. Buffer health",
            panels=[
                line("buffer age — mean / p50 / p90",
                     ["buffer/age_mean", "buffer/age_p50", "buffer/age_p90"]),
                line("fraction from current version", ["buffer/frac_current"]),
                line("z outcomes (win/draw/loss)",
                     ["buffer/z_wins", "buffer/z_draws", "buffer/z_losses"]),
            ],
        ),
        # 5. Wall economy — the heart of the v9 question (Δelo per wall-second).
        ws.Section(
            name="5. Wall economy (per-epoch cost — bigger net = slower epoch; the whole v9 tradeoff)",
            panels=[
                line("per-cycle time (gen vs train)",
                     ["time/gen_s", "time/train_s"]),
                line("cumulative games", ["total_games"]),
                line("epoch", ["epoch"]),
            ],
        ),
    ]


def main() -> None:
    workspace = ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name="Derby v9 — net-capacity ladder (small/medium/large)",
        sections=build_sections(),
    )
    saved = workspace.save()
    print(f"workspace URL:\n  {saved.url}")
    print("\nTo overlay the three v9 lanes:")
    print(f"  - top-left run picker, select ids: {', '.join(RUNS.values())}")
    print(f"  - or set the filter to: Run.id in [{', '.join(repr(r) for r in RUNS.values())}]")
    print("\nReminder: eval/model_elo LAGS (large frozen ~e45). Real strength:")
    print("  - per-chunk anchored: sweep_runs/derby_v9/derby_state.json")
    print("  - head-to-head verdict: scripts/round_robin.py over _peaks (anchored elo is a mirage)")


if __name__ == "__main__":
    main()
