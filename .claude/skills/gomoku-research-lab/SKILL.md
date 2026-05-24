---
name: gomoku-research-lab
description: The gomoku research lab apparatus at ~/code/gomoku — what it IS and where everything lives. The wiki (source of truth), the two-queue model, the scripts (canonical_sweep / lab_train_cell / run_sweep / delo_derby / delta_e_harness / build_archive …), the ops boards (gpu-queue, experiment-ledger, best-cells, baselines, perf-log, research-board), the reference points (R-S* gen aug/s, R-TRAIN-* holistic), and how to list / add / remove work. To OPERATE the lab — pull a lane, dispatch a cell, run the loop, the friction log — use the lab-operator skill; the workers it dispatches are .claude/agents/{gpu-worker,lab-researcher,lab-reviewer,lab-archivist}. Trigger on "what is the research lab", "where's the charter / queue / ledger", "the lab scripts", "lab status", "what's the lab running", "R-S* / R-TRAIN-* reference points", "the gpu queue", "the experiment ledger", "add a lane / Derby recipe / campaign".
---

# gomoku-research-lab — the lab (apparatus)

The lab is the lab — the apparatus. **This skill is the map: what the lab is and where everything lives.** To *drive* it (pull a lane, dispatch a cell, run the loop), use the **`lab-operator`** skill — the operator is the process space the orchestrator runs in, and it dispatches the worker subagents listed at the bottom.

Receipt-driven research on the M5 Max across two areas:
- **Perf research** (R-S\* family) — push measured aug-positions-per-second at a fixed quality pin.
- **Training-recipe research** (R-TRAIN-\* family + the Δelo Derby) — which training loop climbs elo fastest. A training run is a first-class **GPU-required** item: a wall-time-capped, resumable slice via `run_sweep --max-wall-secs --final-eval`, eval inside the bundle, results in `eval_results.jsonl`.

The north-star metric is **Δelo/Δt** — the elo-gain RATE vs a stable anchor over a fixed window. aug/s and epochs/s are gameable *means*, not the goal.

## Read these first (the wiki is the source of truth)

1. **[research-lab-charter.md](/Users/jason/code/gomoku/wiki/topics/research-lab-charter.md)** — mission, success metrics (R-S\* / R-TRAIN-\*), the two-queue scheduler, autonomy boundaries, tiered priority, and the 12-row **stop-gates triage matrix** (CONTINUE / ESCALATE / HALT).
2. **[research-lab-session-runbook.md](/Users/jason/code/gomoku/wiki/topics/research-lab-session-runbook.md)** — per-session mechanics: lock, `--status`, `--retry-failed`, fan-out, resumability contract, cell-design defaults.
3. **[research-lab-reviewer-role.md](/Users/jason/code/gomoku/wiki/topics/research-lab-reviewer-role.md)** — the Reviewer process, now first-class as the `lab-reviewer` subagent: APPROVE / REVISE / BLOCK.
4. **[gpu-queue.md](/Users/jason/code/gomoku/wiki/ops/gpu-queue.md)** — current queue + RESUME STATE. Source of truth for what runs next.
5. **[conventions.md](/Users/jason/code/gomoku/wiki/topics/conventions.md)** — merge-commit (never rebase), deny-list autonomy, memories-also-go-to-wiki.

## The two-queue model

- **GPU-required queue** — serial, **one MPS tenant at a time** (train, gen, or eval; a training slice is a tenant here). Ordered by tier (1 architectural > 2 compound > 3 speculative), then priority `E[Δ]·P / wall_cost`.
- **Everything-else queue** — parallel, Agent fan-out for code/wiki work in worktrees. Never serialize code behind a GPU cell.

The operator keeps both turning. Per-reference winners and the reference points themselves (R-S400/200/100, R-TRAIN-WL5/…) live in `wiki/ops/best-cells.md`.

## The scripts

| Script | Purpose |
|---|---|
| `scripts/canonical_sweep.py` | Canonical perf sweep — pure self-play throughput (R-S\* family); multi-cell from a `cells.csv`. |
| `scripts/lab_train_cell.py` | Live-training cell — trainer + workers, time-capped, resumable (R-TRAIN-\* family). |
| `scripts/run_sweep.py` | Multi-run-safe launcher for training slices / K×buffer sweeps; `--max-wall-secs --final-eval --resume`. |
| `scripts/delo_derby.py` | The Δelo Derby race scheduler — crash-resumable, Δelo-rate priority, failure-isolated per chunk. |
| `scripts/delta_e_harness.py` | Head-to-head Δelo comparison (tight CIs, paired random openings) — the north-star tool. |
| `scripts/calibrate_anchor_elos.py` | Round-robin baseline tournament → `ANCHOR_ELOS` for `rating.py`. |
| `scripts/build_archive.py` · `mine_validation_archive.py` | Build / mine retain-all PositionArchives + curated slices. |
| `scripts/plot_canonical_sweep.py` · `wandb_workspace.py` | Chart a sweep / build the W&B overlay workspace. |
| `scripts/export_onnx.py` | Export a checkpoint to ONNX for browser inference. |

(Plus scouts/probes: `aggressive_engine_scout.py`, `coreml_ane_residency_scout.py`, `bench_lookahead.py`, `l09i_*`.)

## Lab status & managing work

Live state lives in three places: the **GPU-required lane** (what's running now), **`wiki/ops/gpu-queue.md`** (what's next + RESUME STATE), and per-campaign state files under `sweep_runs/<campaign>/`. These recipes read and edit them.

### List what's active

```bash
cd ~/code/gomoku
# 1. GPU-required lane — the serial tenant running NOW (train / gen / eval / derby):
pgrep -fl 'gomoku.train|selfplay_worker|run_sweep|eval_worker|lab_train_cell|delo_derby' || echo "GPU LANE IDLE"
# 2. The queue + what's next (also read the lane table at the top of the file):
grep -A 6 "RESUME STATE" wiki/ops/gpu-queue.md
# 3. Live progress — latest model_elo per cell that has eval results:
for f in sweep_runs/*/checkpoints/eval_results.jsonl; do
  printf '%-32s %s\n' "$(basename "$(dirname "$(dirname "$f")")")" "$(tail -1 "$f" | grep -oE '"eval/model_elo": [0-9.]+')"
done 2>/dev/null
# 4. everything-else lane — background agent worktrees doing parallel CPU work:
git worktree list
# 5. A running Derby — race state + standings:
python -m json.tool sweep_runs/derby_v2/derby_state.json 2>/dev/null | head -30
cat sweep_runs/derby_v2/standings.md 2>/dev/null          # board_md_path from the board json
```
For a single run's per-epoch health: `tail -3 sweep_logs/<CELL>/trainer.log` — watch the `gen=`/`train=` split (gen = worker MPS, train = SGD MPS; a ballooning `train=` is the runaway/contention tell, per the friction log + [[project-concurrent-runs]]).

### Add work

| Want | Concrete steps |
|---|---|
| A **GPU-queue lane** (perf cell or training-slice experiment) | Append a lane to `wiki/ops/gpu-queue.md`: id, tier (1 architectural / 2 compound / 3 speculative), the dispatch command, an `E[Δ]·P / wall_cost` priority note. Bump RESUME STATE. It runs when it reaches the top of the serial lane; then file receipts (the operator's receipt checklist). |
| A **Derby recipe** (race a new training lever) | (1) `scripts/run_sweep.py` → add `Cell("derby-<idea>", …)`, cloning a sibling `derby-*` cell, change **exactly one** lever. (2) `scripts/derby_v2_board.json` → add `{"name","cell","cell_name","lever"}` (set `cell`=`cell_name`=`derby-<idea>`). (3) *optional* title card (hypothesis + expected Δelo signature) in `wiki/ops/research-board.md`. Verify with `python scripts/delo_derby.py --board scripts/derby_v2_board.json --dry-run`. |
| A **new campaign / board** | New `scripts/<name>_board.json` (set `global.engine`, `base_out_dir`, `board_md_path`) + a `research-board.md` section + its cells. Same shape as the derby. |

### Remove / stop work

```bash
# Stop the running GPU tenant CLEANLY — SIGTERM makes the trainer self-save a
# resumable latest.pt (buffer embedded) and run_sweep tear down workers+eval.
# NEVER kill -9 (skips the clean save; wandb won't flush).
pkill -TERM -f 'sweep_runs/<CELL>/'    # one run_sweep training slice
pkill -TERM -f 'delo_derby'            # a Derby race (current chunk finishes its teardown)
```
- **Drop a queue lane:** move it to the Completed section of `wiki/ops/gpu-queue.md` with a one-line reason; recompute RESUME STATE + `consecutive_rejects`. Ledgers are **append-only** — mark it dropped, don't delete history.
- **Remove a Derby recipe:** delete its entry from `scripts/derby_v2_board.json` (and optionally its `Cell`). If a race is mid-flight, also drop its idea from `sweep_runs/derby_v2/derby_state.json`, or `--resume` re-queues it.
- **Resume after a stop/crash:** `python scripts/delo_derby.py --board <board> --resume` (re-queues `running` ideas), or `run_sweep.py --cell <CELL> --resume sweep_runs/<CELL>/checkpoints/latest.pt --max-wall-secs <N> --final-eval`. The clean-stop save means no cold buffer refill ([[project-training-slices]]).

## The operator & the workers

- **`lab-operator`** (skill) — *operates* this lab: the dispatch loop, fan-out, serialization, title cards, receipt discipline, stop-gate triage, and the compounding **friction log**. It is the process space the orchestrator runs in; the conductor lives there because subagents can't spawn subagents.
- **`.claude/agents/`** — the workers the operator dispatches:
  - **gpu-worker** — the single MPS tenant; runs ONE GPU cell, hands back a receipt-ready summary.
  - **lab-researcher** — read-only parallel-queue investigator (logs / wandb / web → synthesis).
  - **lab-reviewer** — the promotion gate (APPROVE / REVISE / BLOCK).
  - **lab-archivist** — curates evidence into the wiki (new archives only; never mutates existing).

## Cross-refs

- Operator (drives this lab): [[lab-operator]]
- Sister skill (training/publish pipeline, not the lab): [[gomoku-train]]
- Charter / runbook / reviewer-role / conventions — `wiki/topics/`; queue / ledger / baselines / best-cells / perf-log / research-board — `wiki/ops/`.
- Memories: [[feedback-lab-scheduler]], [[project-training-slices]], [[feedback-elo-per-wall]], [[project-delo-derby]], [[feedback-know-the-machine]], [[feedback-merge-commits]]
