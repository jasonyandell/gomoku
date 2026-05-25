---
name: gomoku-research-lab
description: Orchestrate the gomoku research lab at ~/code/gomoku — perf research AND training-recipe research on the M5 Max. Two queues: GPU-required (serial; train/gen/eval) + everything-else (parallel via Agent fan-out). Receipt discipline, Reviewer audits, TQ-gated promotion, time-capped training slices (run_sweep --max-wall-secs --final-eval), the Δelo Derby. Designed to run forever autonomously per Jason's 2026-05-23 directive. Trigger on "run the lab", "run the research lab", "orchestrate the lab", "make this mac sing", "keep the lab going", "next lane", "perf cycle", "race training recipes", "the derby", "kick off perf", "what should we run next", or anything about R-S* / R-TRAIN-* references, lane dispatch, sweep_logs/lab-*, the experiment-ledger, or gpu-queue.md.
---

# gomoku-research-lab

Operate the gomoku research lab at `~/code/gomoku/`. The lab runs receipt-driven research on the M5 Max across two areas: **perf research** (push measured aug-positions-per-second at fixed quality) and **training-recipe research** (which training loop climbs elo fastest — e.g. the Δelo Derby). A training run is a first-class **GPU-required** item: a wall-time-capped, resumable slice via `run_sweep --max-wall-secs --final-eval`, with eval inside the bundle and results read from `eval_results.jsonl`. The mission and rules of engagement live in the wiki — this skill is the personalized invocation index.

## Read these first (wiki is the source of truth)

1. **[research-lab-charter.md](/Users/jason/code/gomoku/wiki/topics/research-lab-charter.md)** — mission, success metrics (R-S* and R-TRAIN-*), two-queue scheduler, autonomy boundaries, tiered priority, **stop gates and escalation triage** (the 12-row matrix that tells you when to CONTINUE / ESCALATE / HALT).
2. **[research-lab-session-runbook.md](/Users/jason/code/gomoku/wiki/topics/research-lab-session-runbook.md)** — per-session mechanics: lock, --status, --retry-failed, fan-out pattern, resumability contract, cell-design defaults.
3. **[research-lab-reviewer-role.md](/Users/jason/code/gomoku/wiki/topics/research-lab-reviewer-role.md)** — the Reviewer process; spawn per receipt; APPROVE / REVISE / BLOCK.
4. **[gpu-queue.md](/Users/jason/code/gomoku/wiki/ops/gpu-queue.md)** — current queue state + RESUME STATE. Source of truth for what to run next.
5. **[conventions.md](/Users/jason/code/gomoku/wiki/topics/conventions.md)** — merge-commit (never rebase), default-allow autonomy, memories-also-go-to-wiki.

## Quick mental model

The lab is **not** "run a benchmark, look at a number, done." It's a receipt-driven scientific loop:

- **Two queues, two-queue scheduler.** The **GPU-required** queue is serial (one MPS tenant at a time — train, gen, or eval; a training run is a time-capped slice here). The **everything-else** queue is parallel (Agent fan-out for code/wiki work in worktrees). The orchestrator keeps both turning — never serialize code behind GPU-required cells.
- **Lanes file receipts.** Every measurement is `decision: promote | reject | needs_repeat | blocked`, with 5 surfaces updated (experiment-ledger.md, baselines.md, best-cells.md, perf-log.md, gpu-queue.md). A lane that ran without filing receipts is invisible to the next session — **the receipt is the lane**.
- **Reviewer gates promotion.** No promote without Reviewer APPROVE. Spawn a fresh `general-purpose` agent with the audit prompt from research-lab-reviewer-role.md. Reviewer reads files; it does not edit. APPROVE / REVISE / BLOCK.
- **Tier-1 wins compound across the run; Tier-3 wins are speculative knob tuning.** Tier-1 always runs first when unblocked. Don't leapfrog Tier on score alone.
- **Smoke-first.** Default cell time is 60–90s (NOT 5 min). Escalate to 300s only when the smoke is genuinely ambiguous.

## Keep going — Jason's standing directive (2026-05-23)

**Default to continuing. As long as there is queueable work, keep working — do NOT pause at a clean milestone to ask "should I continue?".** Jason 2026-05-23, after I stopped at a tidy stopping point: he wants the loop to keep pulling lanes autonomously. A finished lane, a filed receipt, a Reviewer APPROVE, even a corrected error are NOT stopping points — they're the loop's normal beat. Pull the next lane.

What "work to do" means concretely:
- Any unblocked lane in gpu-queue.md (any tier). Pull the top one and dispatch.
- Lane needs a code change? That's CPU-queue work — fan out an Agent in a worktree and keep the GPU queue turning. Don't stop.
- Lane needs a cooled chip (thermal-sensitive, e.g. Lpwr2)? Run a different unblocked lane meanwhile, or do CPU-queue prep (code/wiki), or take a short cooldown then proceed. Don't hand back.
- Only genuinely STOP when the charter triage says HALT (queue empty AND no follow-up queueable AND no compound mechanism left) or ESCALATE (box busy, Reviewer BLOCK, production-default flip, Class C). Everything else → CONTINUE.

When you catch yourself about to write "good place to pause" / "clean stopping point" — instead pull the next lane. Report progress as you go; don't block on permission. The loop runs until the work runs out, not until a milestone feels tidy.

## Quick start

```bash
cd ~/code/gomoku
pgrep -fl 'gomoku.train|selfplay_worker|run_sweep|eval_worker|lab_train_cell|delo_derby' || echo "BOX IDLE"
# read the current RESUME STATE
grep -A 4 "RESUME STATE" wiki/ops/gpu-queue.md
```

(Full status + how to add/remove work is the next section.) Then pull the top lane from the queue and dispatch via either:
- `scripts/canonical_sweep.py` — pure self-play (R-S* family). Multi-cell sweeps.
- `scripts/lab_train_cell.py` — live training + workers (R-TRAIN-* family). One cell per invocation; 30s warmup + 60–120s measure default.

Both surface the flags Jason added 2026-05-23: `--compile`, `--fp16-eval`, `--evaluator coreml`, `--coreml-compute-units`, and (canonical_sweep only) the `env` column in cells.csv.

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
| A **GPU-queue lane** (perf cell or training-slice experiment) | Append a lane to `wiki/ops/gpu-queue.md`: id, tier (1 architectural / 2 compound / 3 speculative), the dispatch command, an `E[Δ]·P / wall_cost` priority note. Bump RESUME STATE. It runs when it reaches the top of the serial lane; then file receipts (next section). |
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

## Title card — every run starts with one

**Before any run that lands weights — a fresh production run, a time-capped training slice, or a new Derby idea's first chunk — present a TITLE CARD.** It makes "what is *this specific* run doing?" unambiguous, both in the moment and in the scrollback. We've always done these; they're easy to skip when autonomous — don't.

**Always present the card, then proceed — no ACK gate.** This is an autonomous lab; the card is for clarity, not permission. Print it and launch. (Flipping the production-default WL-release lineage is a separate deliberate ESCALATE per the stop-gates — *that* still surfaces to the user, but the title card itself never blocks.) A Derby idea's card already lives in `wiki/ops/research-board.md` — cite it rather than re-authoring.

Template — tight, fill every line, lead with the lever + a falsifiable expectation:

```
━━━━━━ <RUN / cell name> ━━━━━━
What:    <one line: what this run is>
Lever:   <the ONE change vs parent / baseline>
Parent:  <fresh (--seed 0) | resume <ckpt> @ elo NNNN>
Config:  <cell · sims · buffer · key recipe deltas> · cap <wall-slice or epochs>
Why:     <hypothesis — what we expect to learn>
Expect:  confirm = <signature> · refute = <signature>
Track:   <wandb run | sweep_runs/<cell>/checkpoints/eval_results.jsonl>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Same shape as the Derby title cards in `wiki/ops/research-board.md`. The long-form production-launch flow (smoke → launch → monitor → close-out) lives in [[gomoku-train]]'s launch-sequence-runbook; this card sits at the top of it.

## How to dispatch a cell

For R-S* (pure gen, canonical_sweep):

```bash
# Make a cells.csv (model,workers,games_per_batch,n_simulations,wave_size[,env])
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT="sweep_logs/lab-<LANE_ID>-$TS"
mkdir -p "$OUT"
cat > "$OUT/cells.csv" <<'CSV'
model,workers,games_per_batch,n_simulations,wave_size
small,8,8,400,512
CSV
nohup python scripts/canonical_sweep.py \
  --out-dir "$OUT" --cells-from "$OUT/cells.csv" \
  --lane <LANE_ID> --secs-per-cell 60 [--compile] [--fp16-eval] \
  > "$OUT.driver.log" 2>&1 &
```

For R-TRAIN-* (live training, lab_train_cell.py):

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT="sweep_logs/lab-<LANE_ID>-$TS"
nohup python scripts/lab_train_cell.py \
  --out-dir "$OUT" --lane <LANE_ID> \
  --model small --workers 8 --games-per-batch 8 \
  --n-simulations 400 --wave-size 64 \
  --ema-tau 0.99 --grad-accum-steps 4 \
  [--sgd-per-position 0.001] [--fp16-eval] \
  [--evaluator coreml --coreml-compute-units CPU_AND_NE] \
  --warmup-secs 30 --measurement-secs 120 \
  --device mps \
  > "$OUT.driver.log" 2>&1 &
```

Bash background mode + harness notification on completion. Don't poll.

## Fan-out orchestration (the parallel CPU-queue mode)

Canonical branch/worktree workflow: `wiki/topics/branch-and-worktree-workflow.md`. The rules below are this skill's agent-specific instantiation of it — including the load-bearing rule that **even you (the orchestrator) work in a worktree, never directly in the shared `main` checkout**, since the derby and other sessions share it.

When a lane block has multiple independent sub-lanes (e.g. the LF1-followups: code + analysis + design), run it as a **two-queue fan-out**: dispatch every CPU-queue sub-lane as a background Agent in parallel, while you personally drive the single serial GPU lane. This is the lab's highest-throughput mode and it stays clean if you follow the rules below (validated 2026-05-23 on the LF1-followups block — Jason: "this is smooth").

**Step 1 — route every sub-lane by queue.** Read the lane block, then tag each sub-lane:
- **CPU-queue (fan out, parallel):** code changes, log/wandb analysis, wiki/design docs. No GPU contention → all run at once.
- **GPU-queue (serial, you drive):** any lane that needs an MPS cell (`canonical_sweep` / `lab_train_cell`). One at a time. Don't hand a GPU cell to a subagent — you run it so the box stays a single serial tenant.

**Step 2 — fan out the CPU lanes as background Agents, one message, all at once.** Three agent flavors:
- **code agent → `isolation: "worktree"`** (MANDATORY). This is the fix for the recurring shared-working-tree contamination (see friction log). Each code agent gets its own git worktree; they can't clobber each other or the main tree. They commit to a feature branch and **do NOT merge** — you merge `--no-ff` serially (Step 4).
- **analysis agent → read-only, no worktree.** It reads logs/wandb and **reports findings in its final message** (you integrate into the receipt). Don't let it edit repo files.
- **design/doc agent → `isolation: "worktree"`** (it writes a wiki page; worktree avoids contending on the main tree).

**Step 3 — the non-negotiable guardrails in every code-agent prompt:**
- **"FIRST, sync your worktree to current local `main`: `git merge --no-ff main` (NEVER rebase)."** The Agent worktree-isolation can create your worktree from a STALE base commit (in the LF1 run, two agents landed on `d019e58`, which predated `lab_train_cell.py` entirely). Working against a stale base produces a branch whose diff vs main looks fine but whose code is months old. Merging local main in at startup pins you to current code; then branch `feat/perf-<lane>` off that. (Do NOT pull from a remote — `main` here means the local main HEAD.)
- **Opt-in only; production defaults byte-identical.** New knobs default to current behavior. Changing a production training default (WL5 recipe) is an **ESCALATE**, not a subagent's call. Say this explicitly or an agent will "helpfully" flip a default.
- **"Read `wiki/topics/<the writeup>.md` FIRST."** Give precise file + symbol pointers (grep targets), the deliverable spec, and the verification bar (`--dry-run` + a *short* smoke; **never a long/divergent cell — that's your GPU queue**).
- **"Confirm `python -c "import gomoku; print(gomoku.__file__)"` resolves to YOUR worktree before running — and if it points at the main checkout (the editable install can resolve there), run with `PYTHONPATH=<your worktree>` so your code executes and your edits don't leak into the main checkout."** (friction log).
- **"Branch `feat/perf-<lane>`, commit, do NOT merge to main. Your final message is the only thing I see"** — so demand: branch, commit hash, exact flags/behavior added, smoke output, confirmation that unset flags = unchanged behavior, blockers.

**Step 4 — integrate serially, you in control.** As agents return: merge each code branch `--no-ff` yourself, one at a time (don't let agents merge — that's how the concurrent-merge mess happens). Resolve conflicts, run tests on the merged tree, then file the consolidated receipt batch and spawn Reviewer(s). Note: a worktree agent may have `merge main` into its own worktree first to resolve cross-refs (lane-4 did) — that's fine; you still merge ITS branch back.

**Step 5 — GPU/CPU coordination.** Your GPU cell's `preflight_idle` aborts if it sees a tenant; a code agent's smoke (if it uses `lab_train_cell`, which also preflights) will abort while your cell runs. That's correct — tell code agents to prefer `--dry-run` and that long cells are the orchestrator's job. Keep tenant-detection strings (`selfplay_worker`, `gomoku.train`, …) out of YOUR launch command line so you don't self-collide (friction log).

**Step 6 — don't poll; one health-check is fine.** All Agents and background cells are harness-tracked → you're notified on completion. Don't burn turns polling. ONE early peek at a GPU cell's `trainer.log` to confirm it cleared preflight and is logging epochs (not silently aborted) is good hygiene; after that, wait for the notification.

**Receipts still apply.** Batch the sub-lane findings into one consolidated lane-block receipt across the 5 surfaces rather than thrashing the wiki while agents are mid-flight; each promote still needs a Reviewer.

## Measuring & reading results

**The north-star is Δelo/Δt — measure it, don't eyeball proxies.** `scripts/delta_e_harness.py` is the tool: the elo-gain RATE from a common checkpoint over a fixed window vs a stable anchor. aug/s and epochs/s are gameable *means*, not the goal ([[feedback-elo-per-wall]]).
- **Anchored `eval/model_elo`** (written per cycle by the bundle's eval_worker into `eval_results.jsonl`) is the cheap signal — fine while a model is climbing through the anchor ladder.
- **Head-to-head** is the tiebreaker once candidates beat the anchors (all the strong WL / Derby checkpoints saturate the ladder): `delta_e_harness.py --head-to-head` (or `gomoku.match`) between candidates. This is the correct eval for the Derby v2 top-3 — anchored elo can't separate models that all crush `lookahead:depth=4`.

**Reading / comparing:**
- `scripts/plot_canonical_sweep.py` — chart a sweep's cells.
- `scripts/wandb_workspace.py` — build a saved W&B workspace overlaying a run set (6 pre-tuned sections). Re-run with the new run id prepended when a run joins the comparison; the URL changes each time (bookmark the latest it prints).

## Receipt-filing checklist

After every lane completes, before moving on:

1. Read `<sweep-dir>/summary.tsv` and `<sweep-dir>/cell_*/logs/trainer.log` (for R-TRAIN cells) or `worker-*.log` (for R-S* cells).
2. Append a yaml block to **`wiki/ops/experiment-ledger.md`** (schema at the top of that file; the Training-Quality Promotion Gate fires if you changed any training-behavior knob).
3. Append a row to **`wiki/ops/baselines.md`** with the command + result.
4. Update **`wiki/ops/best-cells.md`** — table row AND promotion-log header (only on promote). Don't flip R-TRAIN-WL5; that's the WL5 production recipe pin.
5. Append a narrative entry to **`wiki/ops/perf-log.md`** — what was tried, what surprised you, what comes next.
6. Update **`wiki/ops/gpu-queue.md`** — move lane to Completed; update reference table at the top if a new best landed; recompute consecutive_rejects; update RESUME STATE for the next session.
7. Commit with a clear `lab <lane_id>: <decision> — <one-line summary>` message.
8. Spawn the Reviewer — a background `general-purpose` Agent with the audit prompt from `wiki/topics/research-lab-reviewer-role.md`, pointed at the receipt + the 5 surfaces. It READS, never edits; verdict **APPROVE / REVISE / BLOCK**. No `promote` is final until APPROVE — then backfill the `Reviewer: APPROVE` field. (REVISE → apply the numbered fixes, re-spawn same lane/receipt. BLOCK → ESCALATE.) **Pair the receipt commit with the Reviewer spawn** — never chain to the next dispatch before the previous lane's Reviewer is at least spawned (its read can run in parallel with the next GPU cell).

## Stop gates — when to CONTINUE / ESCALATE / HALT

**Read [research-lab-charter.md § Stop gates and escalation protocol](/Users/jason/code/gomoku/wiki/topics/research-lab-charter.md#stop-gates-and-escalation-protocol) for the full 12-row triage matrix.** Summary:

- **CONTINUE** (no user attention): code-change bugs (patch in flight), 3-reject streak with any Tier-3 lane still queueable, charter staleness with mechanical fix, Reviewer REVISE, MPS error on a single cell (one retry), TQ-gate needs_repeat, cell hit wall-time-cap mid-warmup (extend window).
- **ESCALATE** (one-line PushNotification, pause loop): box busy with another tenant, Reviewer BLOCK, decision to flip production training default, Class C work surfaces (custom Metal, new model arch, replacing trainer/evaluator backend wholesale), repeated MPS errors after retry.
- **HALT** (clean session-end + perf-log entry): queue empty AND no follow-ups queueable AND no compound mechanism left to test.

**Heuristic:** the lab can autonomously do anything that's reversible at the file/branch level and doesn't change the production training default. When in doubt, lean CONTINUE; the worst case of CONTINUE is a wasted cell wall-time; the worst case of pre-emptive HALT is leaving a +97% headline win on the table (which the 3-reject-halt would have done 2026-05-23).

## Handoffs, hygiene & discipline

**Worktree hygiene (multi-agent fan-out).** Every spawned code/doc agent's FIRST step is `git merge --no-ff main` (NEVER rebase) — `isolation: worktree` can branch from a STALE base commit (hit repeatedly; see the friction log). After merging an agent's branch back: `git worktree list`, `git worktree prune`, `git branch -d feat/<merged>`.

**The janitor, not a remembered procedure (2026-05-25).** `isolation: worktree` locks each agent worktree to the spawning session's PID and the harness removes it *only on graceful session exit*. Our regime is overnight autonomous derbies that get killed / crash / OOM — those leak a worktree + a dead-PID lock + a `feat/*` or `worktree-agent-*` branch every time. By 2026-05-25 this had silently grown to 26 worktrees / 57 branches. The old advice here ("harness auto-cleans — don't force-unlock") was *wrong for the crash case* and, worse, it was a procedure — procedures that must run at session-end fail exactly when the session dies. The fix is a liveness-aware janitor that runs at session **start**:
- **Session start (run every time):** `python scripts/reclaim_worktrees.py` (dry-run preview) → `--apply` to reclaim. It removes ONLY agent worktrees whose lock PID is dead, deletes merged + empty-scratch branches, and never touches live-session worktrees, external (`~/.codex/`) worktrees, or `feat/*` with unmerged work. Safe to run while other sessions / the derby are live (the PID check protects them). Pass `--include-scratch` to also drop `worktree-agent-*` branches that carry only superseded commits (it prints reflog-recoverable hashes first).
- **Per reporting cycle:** the cron narrator emits `python scripts/reclaim_worktrees.py --gauge` — a one-line `repo-hygiene: worktrees=N (orphaned=K) branches=M (merged-undeleted=J)` metric. `orphaned>0` or `merged-undeleted>3` ⇒ run `--apply`. This is the *gauge* that makes slow entropy visible the day it grows.

**Handoffs to [[gomoku-train]] (the machine — cross-ref, don't duplicate its steps):**
- A recipe WON and you want to ship it → gomoku-train's publish flow (HuggingFace push + Cloudflare deploy).
- Starting a real long PRODUCTION run (a new WL-release lineage, not a slice) → gomoku-train's launch-sequence-runbook: mine the validation archive first, 30-epoch smoke, then launch with a title card (no ACK). The research lab schedules *slices*; a new release lineage is a gomoku-train launch.

**Discipline one-liners:**
- **Thermal:** heat-soaked IS the production regime. Absolute aug/s drifts ~5% across a session as the chip warms — for cross-time comparisons do a matched-thermal back-to-back A/B, never a compare-to-a-cold-session reference ([[feedback-heat-soaked-is-production]]).
- **Box-busy:** run the Quick-start idle check before every GPU dispatch — a non-lab tenant is an ESCALATE, not a barge-in.
- **Forward (15×15 / Gomocup era):** rated external-engine baselines (`wiki/topics/external-engine-baselines.md`, the Piskvork wrapper) become the real strength bar; not load-bearing at 9×9 yet.

## Friction-smoothing log

Things that bit us before, with their fixes. **Read this on session start; append after every session.** This is the part of the skill that compounds across runs.

**Meta — what kind of fix to write (this is how the lab learns in new ways).** This log's default sensor is *narrated friction*: something bit us in the moment, we wrote it down. That sensor is blind to **slow entropy that no single moment surfaces** — leaked worktrees, undeleted branches, disk creep, buffer drift, stale crons. Nobody ever had a bad *moment* over "26 worktrees"; the cost was only legible in aggregate, a week later. So when you write a fix here, classify it:
- If the fix is a **procedure** ("remember to run X after Y") — that's a smell. Procedures decay and fail silently exactly when a session crashes. **Upgrade it to a janitor + a gauge:** a *janitor* is idempotent, liveness-aware, and runs at a robust trigger (session **start**, not session-end); a *gauge* is a one-line metric in the cron narrator so you'd notice the day it stops working. (First instance: `scripts/reclaim_worktrees.py` — see Worktree hygiene above and `wiki/topics/worktree-hygiene.md`.)
- **The standing rule: for every class of artifact the lab creates (worktrees, branches, checkpoints, buffers, sweep_logs, crons), there should be a janitor that reclaims it and a gauge that counts it.** When you add a creator, add its janitor+gauge in the same change. Audit the existing creators against this rule when you touch them.

### 2026-05-23 (session that doubled R-S400 and R-TRAIN)

**L12 driver `--save-every=1000000` froze worker_weights publishing.**
- Symptom: workers stayed on v0 forever; trainer waited for v1+ games that never came; epoch 1 logged then silence.
- Root cause: `gomoku/train.py:1220` publishes `worker_weights.pt` inside the `--save-every` block. With save-every set to 1M to "disable mid-run checkpoint IO", the worker-facing weights file never refreshed after the initial publish.
- Fix (1dc4abb): `--save-every=1 --keep-last-n=1` (small per-epoch ~4MB writes, auto-pruned; the 1.4GB latest.pt still gated by save-buffer-every=1M).
- Lesson: in gomoku.train, `--save-every` is not just "save less often" — it's a tight coupling with the worker-version barrier. Don't crank it without understanding the cascade.

**L12 `count_records()` at SIGTERM undercounted by ~30×.**
- Symptom: real games_per_sec was 14 but driver reported 0.53; aug/s reported 108 instead of 3,300.
- Root cause: the trainer **ingests + deletes** worker `game*.pt` files as it goes. By SIGTERM time only the in-flight residuals are on disk.
- Fix (4a825f1): parse the trainer's own epoch line for cumulative `games=N` and `buf=N` counters + per-epoch wall `(Xs:`. The trainer log is the authoritative source for R-TRAIN throughput numbers; the file-system count is the fallback for trainer-less canonical_sweep.
- Lesson: prefer trainer-emitted counters over post-hoc file walks for live-training cells. The trainer-log shape (with cumulative fields) is in `scripts/lab_train_cell.py:parse_trainer_log`.

**`--fp16-eval` × `--evaluator coreml` crashes at startup.**
- Symptom: `RuntimeError: Input type (float) and bias type (c10::Half) should be the same` at first conv inside `export_model_to_coreml`'s `torch.jit.trace`.
- Root cause: `_maybe_half` ran BEFORE `_build_evaluator`; Core ML export's dummy input is fp32; cast model + fp32 dummy = type mismatch. Also semantically redundant — Core ML already runs at `compute_precision=FLOAT16` internally.
- Fix (9cd16e3): `selfplay_worker.parse_args` force-sets `fp16_eval=False` when `evaluator=='coreml'` with a printed audit line. Graceful no-op + auditable.
- Lesson: when adding a perf flag, check it against every other perf flag's pipeline order. The order in selfplay_worker.main is `_load_model → _maybe_half → _maybe_compile → _build_evaluator` — anything that consumes the model (Core ML export, torch.compile, etc.) sees whatever the prior steps did to it.

**Cell wall-time too short for the trainer's epoch.**
- Symptom: `epochs_per_sec=0.0`, `epochs_in_window=1` despite a clean run.
- Root cause: trainer's first epoch is ~12s (warmup includes buffer prefill); subsequent epochs ~10–11s. A 60s measurement window after 30s warmup catches only 1 epoch in the rate-computation window.
- Fix: dispatch with `--measurement-secs 120` for R-TRAIN cells. Charter's smoke-first table now codifies 30s warmup + 60–120s measure as the R-TRAIN-* default.
- Lesson: R-TRAIN cells need a window that spans ≥ 3 of the trainer's actual epochs. Read one trainer log to calibrate, then set the measurement budget. Hard cap 5 min, but 120s is the sweet spot for the default WL5 recipe.

**Multiple agents in parallel + main worktree's working tree.**
- Symptom: my L06 merge silently picked up a charter edit the user was making in his IDE.
- Root cause: `git commit` after a merge-conflict resolution snapshots the entire working tree, including in-progress edits to unrelated files. The user was editing `wiki/topics/research-lab-charter.md` (adding the Vibe footer) while I had L06 mid-merge.
- Mitigation: before resolving a merge conflict, `git status` to see what else is modified; either stash unrelated edits, or note the inclusion in the commit message. (The 2026-05-23 case was benign — the user's edit was intentional — but the next instance might not be.)
- Lesson: parallel-agent workflows share a working tree even when they don't share a branch.

**Reviewer audit skipped (orchestrator memory lapse).**
- Symptom: L06fu-extended landed with 3 promotes and no Reviewer audit; the next Reviewer (L11b') flagged the missed audit incidentally.
- Mitigation: every commit with `lab <lane_id>: promote` in the message should immediately be followed by a Reviewer spawn. The orchestrator should NOT chain to the next dispatch before the previous lane's Reviewer is at least spawned (Reviewer reads can run in parallel with the next GPU lane; that's the point).
- Lesson: a "spawn Reviewer" step in the dispatch loop is non-optional. Treat the receipt commit and the Reviewer spawn as a paired action.

### 2026-05-23 (session-resume: ANE envelope mapping — L09c PROMOTE + 4 envelope-mapping rejects)

**plies_mean is NOT stationary across asymmetric-epoch R-TRAIN cells.**
- Symptom: L09c-V512 candidate's plies_mean = 31.02 vs baseline 33.47 (-7.3%) looked alarming. The L09c Reviewer's prior drift-watch flag fired (would have suggested a 2× repeat). Initial read: possible engine-induced game-shape drift between torch+fp16 and Core ML.
- Root cause: The candidate ran 8 trainer epochs in the 120s window because trainer-side MPS-relief made the trainer epoch ~5× shorter (train= field collapsed from ~18s on torch+fp16 to ~2-3s on ANE). The baseline only ran 1 epoch. Per-epoch plies in the candidate trainer.log: 30.9 → 33.1 → 33.5 → 33.4 → 31.7 → 30.4 → 29.4 → 27.7 — peaks at epoch 3 then descends as the policy improves on a fresh-init small model. Policy loss dropped 4.404 → 3.477 across the 8 epochs, confirming within-window training progress. Aggregate plies_mean is dominated by the late epochs.
- Mitigation: when comparing R-TRAIN cells with asymmetric `epochs_in_window` counts, check **per-epoch plies values in trainer.log**, not just the aggregate `plies_mean`. A material aggregate drift in this case can be the within-window training-progress signature, not engine-induced game-shape drift.
- Lesson: `plies_mean` is NOT stationary across asymmetric-epoch R-TRAIN cells. Future Reviewers' drift-watch should look at the per-epoch progression in `trainer.log` when arms have very different `epochs_in_window`. For a clean stationary plies comparison, the right tools are matched-epoch sub-sampling OR matched-epoch-count windows (extend measurement-secs on the slower arm).

**Session-thermal drift can produce ~5% absolute aug/s drift over a single 90-min session.**
- Symptom: L08 default-heap re-measure (8,937.3 aug/s, canonical_sweep / pure self-play / no trainer contention) came in -4.9% vs R-S400 (9,398.5 aug/s) measured by L06-followup ~90 min earlier in the same session. Initial puzzle: why does the same recipe re-run lower? Within-L08 the 3 cells (default / 2.0 / 0.0 heap-ratio) were only 0.74% apart, so it wasn't a heap-ratio confound.
- Root cause: M5 Max sustained-load throughput drops as the chip warms across ~10 sequential lab cells. The thermal floor shifts over tens of minutes. Back-to-back cells run within ~1% (same thermal state); cross-time comparisons (> ~30 min apart) can drift several percent. Trainer contention is NOT the explanation here (L08 was canonical_sweep, no trainer).
- Mitigation: Within-lane back-to-back A/B is reliable. For cross-time comparisons (against references measured at session-start, or against numbers from a different session entirely), do a re-measure under matched thermal state before drawing conclusions. When in doubt, run an A/B pair back-to-back rather than comparing to a far-away reference.
- Lesson: Absolute aug/s numbers are NOT interchangeable across distant-in-time cells. Comparisons against a session-start reference may have a ~5% thermal-drift confound; matched-shape back-to-back A/B remains the gold standard. The R-S400 = 9,398.5 number is the session-start cool-chip value; mid-session re-measures will sit lower without that being a regression.

**Env-axis lanes: use cells.csv `env` column (L08-driver), not shell-prefix env, so metadata.txt stamps per-cell env_overrides.**
- Symptom: L08 set `PYTORCH_MPS_HIGH_WATERMARK_RATIO` via shell prefix (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=2.0 python scripts/canonical_sweep.py ...`). All 3 cells' `metadata.txt` reported `env_overrides: (none)`. The Reviewer flagged this as a soft artifact-capture gap: there's no on-disk artifact discriminating which env value each cell ran under, even though the env DID propagate to the workers.
- Root cause: `canonical_sweep`'s env-stamping path only records values from the cells.csv `env` column. Shell-prefix env propagates to subprocess workers via Popen's env=None inheritance (canonical_sweep.py:378-381) but isn't captured in metadata.
- Mitigation: For env-axis lanes, populate the cells.csv `env` column with one cell per env value. Per-cell `env_overrides` will then stamp into the cell's metadata.txt automatically. Cells.csv format: `model,workers,games_per_batch,n_simulations,wave_size,env` — the env column is semicolon-separated `KEY=VAL` pairs.
- Lesson: Don't rely on shell-prefix env for env-axis experiments where the env value is part of cell identity. The L08-driver env-column infra (shipped 2026-05-23) is the right tool; using shell-prefix env bypasses the stamping path. (Note: this is a heads-up the lane card itself flagged — "the three cells collapse to one cell_id because env isn't in cell_id_of(); disambiguate via three separate out-dirs or a cell_id suffix when running." The cells.csv approach handles both the stamping AND the disambiguation; shell-prefix env doesn't.)

**ANE-axis findings are a SNAPSHOT, not a verdict — re-measure when stack changes.**
- Symptom: This session-resume mapped a 5-point ANE envelope (L09c PROMOTE at tiny+V=64; L09d/L09c-V512/L09e all reject) and surfaced 3 axis-nulls (V, model-size, routing). The temptation at session-end is to declare ANE-offload "doesn't work for our workload, full stop."
- Mitigation: Frame the envelope as a SNAPSHOT of today's stack (today's Core ML version + today's evaluator pipeline + today's model-arch family). The single-point envelope is the right reading of today's data; it is NOT a structural ANE limit. Document explicit re-measurement triggers in the receipt: (a) Core ML major version updates, (b) new ANE features (e.g., new ANE compute_precision options, new Core ML routing primitives), (c) evaluator-pipeline changes (different export path, different cast strategy), (d) model-arch family changes (different residual block, different stem padding, etc.).
- Lesson: When the engine-fit envelope produces a "single-point win" finding, the right framing is "this is the current shape" not "ANE doesn't pay." Future Core ML / ANE work could (and historically has — see Apple's WWDC pattern of incremental Core ML improvements) shift the envelope meaningfully. Lane cards in queue for downweighted-but-not-deleted ANE follow-ups (L09f / L09g / L09h) should flip back to load-bearing the moment any re-measurement trigger fires.

### 2026-05-23 (session-resume: L09i — ANE residency restored, and a misread I had to retract)

**ANE residency is gated by the INPUT-SHAPE DECLARATION, not the compute-unit hint.**
- Symptom: every L09* "ANE" lane (`--coreml-compute-units CPU_AND_NE`) actually ran on CPU/BNNS, never the ANE (L09e' showed this; L09i found why).
- Root cause: `coreml_evaluator.export_model_to_coreml` declared a symbolic `ct.RangeDim` batch dim (`gomoku/coreml_evaluator.py:267`). The ANE requires fully static input shapes; a symbolic batch dim silently demotes the whole Core ML program to CPU/BNNS regardless of the compute-units flag. The lab export and a known-ANE-resident scout export emit byte-identical MIL op graphs — only the batch-dim flexibility differs.
- Fix: export with a single fixed static batch (L09i-fix, branch `feat/perf-L09i-fix`); the evaluator pads each leaf-batch up to it and slices/chunks back. Restored `sample`-confirmed ANE residency. **A single fixed batch is the ONLY ANE-placeable option — `ct.EnumeratedShapes` also falls back to BNNS.**
- Lesson: for ANE residency, check the `.mlpackage`'s input-shape declaration FIRST (static vs RangeDim/Enumerated). The compute-units flag is a request, not the gate. Verify with `sample <pid>` (hollance no-sudo): hot path `AneInferenceOperationImplUsingAnefAPIs`/`AppleNeuralEngine` = ANE; `BnnsCpuInferenceOperation` = CPU.

**In wave-mode, `aug/s` is TRAINER-GATED — attribute a holistic collapse to the gen-vs-train phase BEFORE concluding. (This bit me hard — a wrong conclusion across 8 surfaces.)**
- Symptom: L09i-fix-load (ANE workers + GPU hog) cratered aug/s 7,878→302. I filed "ANE workers collapse −96%, contention-immunity falsified, strand closed." The Reviewer caught it as wrong; I had to retract across ledger/perf-log/queue/coupling-page/2 memories/index.
- Root cause: trainer.log epoch lines split `(Xs: gen=Ys train=Zs)`. Under the hog, worker `gen` HELD at 5.1s (workers NOT throttled); the trainer's `train` ballooned 2.5→99.5s (MPS-command-queue contention with the hog; per-step `trainer_step_p50` barely moved). Wave-mode synchronizes worker output to the trainer's epoch loop, so a stalled trainer gates generation → aug/s tanks even though the workers are fine. The collapse was trainer-side, masquerading as a worker collapse.
- Mitigation: for ANY R-TRAIN holistic collapse, read the per-epoch `gen=`/`train=` split and attribute it to workers (gen) vs trainer (train) before writing the mechanism. To test a WORKER property (contention-resistance, eval speed), use **pure self-play (no trainer barrier)**, not lab_train_cell.
- Lesson: `aug/s` in wave-mode lab_train_cell reflects whichever of {worker gen, trainer train} is the bottleneck — it is NOT a clean worker metric. A worker-side question needs a trainer-less measurement. Slow down on the attribution step; the Reviewer's job includes catching exactly this, so spawn it before propagating a strong conclusion.

**Running cells from a worktree: subprocess workers import the worktree's package via cwd — but verify.**
- Symptom (potential): the editable `pip install` location pointed at a STALE agent-worktree path, not main; `import gomoku` could have resolved to the wrong tree.
- Why it worked: `lab_train_cell.py` sets `REPO_ROOT = Path(__file__).resolve().parent.parent`, inserts it on sys.path, and launches trainer+workers with `cwd=REPO_ROOT`; `python -m gomoku.X` with cwd on sys.path[0] resolves the worktree's gomoku first.
- Lesson: before running a worktree code-lane's cells, `cd` into the worktree and run `python -c "import gomoku; print(gomoku.__file__)"` to confirm it resolves to the worktree (not an editable-install path). Subprocess workers inherit `cwd=REPO_ROOT` so they get the same copy. Grep the patched symbol in the worktree's file to confirm the edit is live.

**lab_train_cell preflight `pgrep` matches your OWN launcher if the wrapper contains tenant strings.**
- Symptom: L09-reopen-small refused to start — `preflight: another tenant is on the box` pointing at a zsh PID that was my own background launcher.
- Root cause: I embedded a residency-sampler in the launch command line containing the literal string `selfplay_worker` (in a `pgrep -f 'selfplay_worker.*<lane>'`). lab_train_cell's preflight runs `pgrep -fl 'selfplay_worker|gomoku\.train|run_sweep|eval_worker'`, which matched my wrapper shell's command line. Self-collision; the box was actually idle.
- Fix: launch the cell with a CLEAN wrapper (no `selfplay_worker`/`gomoku.train`/etc. strings in the command line). Preflight only runs at startup, so do any residency sampling (which needs those grep patterns) in a SEPARATE call AFTER the cell is running — a new process with those strings then can't abort the already-running cell.
- Lesson: keep tenant-detection strings out of the cell-launch command line. Sample/inspect workers in a follow-up call, not inline in the launcher.

**canonical_sweep.py has NO `--evaluator coreml` passthrough (gap surfaced 2026-05-23; FIXED).**
- Symptom: the L09f/L09g lane cards (pure self-play on Core ML) and L09i-fix-load-v2 (decoupled worker test) all assume canonical_sweep can run coreml workers. It can't — only `lab_train_cell.py` has `--evaluator`/`--coreml-compute-units`.
- Lesson: pure-self-play ANE measurement needs that passthrough added to canonical_sweep first (worker-cmd builder mirrors selfplay_worker args). It's a small CPU-queue code lane that unblocks the whole pure-self-play ANE family. Do it in the L09i-fix worktree so it composes with the static-batch export.
- FIXED 2026-05-23 (commit 850d432 on feat/perf-L09i-fix): canonical_sweep now takes `--evaluator {torch,coreml}` + `--coreml-compute-units`, passed through to each selfplay_worker. Pure-self-play ANE lanes (fix-load-v2, L09f, L09g) are unblocked. Note: canonical_sweep has no `--dry-run` (unlike lab_train_cell).

### 2026-05-23 (LA1 — lookahead-eval perf pass, and a concurrent-agent merge surprise)

**The bottleneck in a "slow" eval/search path may be pure-Python helpers, NOT the state ops — check `USING_NATIVE` first.**
- Symptom: `lookahead_player` (the alpha-beta Elo anchor) was the known-slow eval path ("45s+", train.py:341). Easy assumption: it's the per-node `state.apply`/`is_terminal` board copies.
- Reality: `state_ops.USING_NATIVE` is `True` — `_state_ops_native.so` is built, so apply/terminal/legal_mask are already C-fast (~0.1s in a 10.6s depth-4 profile). The bottleneck was three *pure-numpy* helpers in `baselines.py` running per search node: `_find_immediate_wins` (52% — a per-legal-cell Python loop at every leaf), `_candidate_moves` (29% — a 24-offset neighbor-dilation loop with fancy-index bounds masking), and full-81-cell `_score_all_moves` used only for candidate ordering. Vectorizing all three (dense per-cell max via `_DENSE_WIN_BY_CELL`; a precomputed (81,81) `_NEIGHBOR_MASK` gather; a candidate-restricted `_score_cells`) gave ~6.3× at depth=4 / 6.5× at depth=2, byte-identical.
- Lesson: before optimizing a hot CPU path, `python -c "from gomoku import state_ops; print(state_ops.USING_NATIVE)"` and cProfile FIRST. The native boundary already covers state ops; the remaining cost is whatever pure-numpy/Python still runs per node. `scripts/bench_lookahead.py` (committed) is the reusable harness (per-move ms at depth 2/4 + `--profile`).

**For a behavior-IDENTICAL perf change to an Elo anchor, prove byte-identical move selection — and commit the proof as a test, not an ephemeral harness.**
- The lookahead is an Elo *anchor*: if its move selection shifts, every model's measured Elo shifts silently. The TQ gate is satisfied *by construction* only if outputs are provably unchanged. I proved it across 360 positions (candidate sets, immediate-win sets, per-cell scores, final depth-4 move) — but in a job-dir harness that's now gone. The Reviewer (correctly, non-gating) flagged that a future helper edit could regress silently. Fix: committed `tests/test_baselines_vectorized_equiv.py` pinning each helper against an *independent brute-force reference* (not the deleted old code). Do this in the same lane next time, not as a follow-up.
- Lesson: "behavior-preserving" perf wins on anchor/search/encoding paths need a *committed* equivalence test against an independent reference, or the guarantee evaporates the moment the proof harness is deleted.

**A sibling agent's commit can ride into main on YOUR merge when you share a working tree — verify nothing was clobbered before trusting the merge.**
- Symptom: `git merge --no-ff feat/perf-lookahead-eval` reported 3 files I never touched (`train_replay.py`, `delta_e_harness.py`, `curated-buffer-and-curriculum-design.md`). The session-start gitStatus said main=19bd11f, but by the time I ran `checkout -b` main was at 6afd2fe, and a parallel agent then committed `6d47bbb` "flywheel tidy" onto the shared branch/worktree *between* my checkout and my first commit (HEAD reflog: `checkout … feat/perf-lookahead-eval` → `commit 6d47bbb` → my LA1 commits). So 6d47bbb sat as an ancestor of my commits and rode into main via my merge.
- This extends the prior "parallel agents share a working tree" entry: it's not just uncommitted edits — a sibling agent can land a whole *commit* on the branch you're on. The git status snapshot in the harness prompt is a point-in-time photo and can be stale within seconds.
- Mitigation/verification (this is the load-bearing step): when a merge diffstat shows files outside your lane, do NOT assume corruption and do NOT `reset`. Run `git show --stat <merge>` (note the first parent), then `git diff <first-parent> HEAD -- <surprise-files>` to confirm the sibling's changes are PRESENT and intact (not reverted/clobbered), then `git log --oneline --graph` + `git reflog` to reconstruct what happened. In the LA1 case everything was preserved and main stayed coherent (imports + tests green) — benign. Document the inclusion (I noted 6d47bbb in the receipt + flagged it to the user + had the Reviewer verify integrity as item A).
- Lesson: a clean-looking `--no-ff` merge can silently integrate a sibling agent's work. The integrity check (`diff first-parent..HEAD` on the surprise files + reflog reconstruction) is mandatory before filing the receipt, and the Reviewer should be told to verify it.

### 2026-05-23 (LF1-followups fan-out — worktree agents landed on a STALE base commit + editable-install edit leak)

**Agent worktree-isolation can create a subagent's worktree from a months-old base commit — the branch diff looks clean but the code is stale.**
- Symptom: two fanned-out code agents (lane-1 warmbuf, lane-6 tilecap), both spawned with `isolation: "worktree"`, reported their worktrees started at `d019e58` — a commit that PREDATED `scripts/lab_train_cell.py` and the wave-mode `gomoku/train.py` entirely. An agent that just edits and commits there produces a branch that, when you `git diff main..branch`, can show spurious reverts/deletions of everything main added since `d019e58` — or (if the agent is careful) forces it to merge main in first to even see the current files.
- Why it happens: the worktree's base is whatever the isolation picked, NOT necessarily current main. This repo has heavy concurrent agent activity (many `.codex/worktrees/*` and `.claude/worktrees/*` coexist), so "current" is a moving target and the snapshot the harness branches from can be old.
- Mitigation (now in the fan-out Step-3 guardrails): tell every code agent, as its FIRST action, to `git merge --no-ff main` (local main, never a remote, never rebase) so it's working against current code, THEN branch `feat/perf-<lane>` off that. Both LF1 agents recovered by doing exactly this (lane-6 merge `511ff39`; lane-1 reset its branch onto main HEAD) — but it should be a startup instruction, not a thing they discover after editing the wrong files.
- Orchestrator-side check: before merging ANY returned branch, `git diff main..<branch> --stat` and confirm it shows ONLY the intended files with ONLY additions (no deletions/reverts of main's code). A stale-base branch betrays itself as deletions of files that exist on main. (In the LF1 run the net diffs were clean — lane-1 = `lab_train_cell.py` +226/-2, lane-6 = `train.py` +112/-1 + `lab_train_cell.py` +26 — so the agents' recovery worked.)

**The editable `gomoku` install can resolve to the MAIN checkout even from inside a worktree — edits and runs leak across trees.**
- Symptom: lane-1 reported its Read/Edit calls and `import gomoku` resolved to the *main* worktree's copy, not its own worktree, because the `pip install -e` location points at a specific tree (here, a different agent's worktree path). Edits via that path land in the main checkout; a run imports the wrong code.
- Mitigation: each code agent must `python -c "import gomoku; print(gomoku.__file__)"` and, if it doesn't resolve to its own worktree, run everything with `PYTHONPATH=<its worktree>` (cwd precedence is unreliable because agent bash calls reset cwd). Lane-1 caught it, restored the main worktree to clean (`git checkout --`), and re-applied on the correct base; the main worktree was NOT left modified — but this is a trap that can silently corrupt the main checkout if undetected.
- Lesson: worktree isolation isolates the git index/branch, NOT the Python import resolution. For a repo with an editable install, pin `PYTHONPATH` explicitly in worktree agents.

**Two code agents editing the same file (`lab_train_cell.py`) auto-merged cleanly because they touched non-overlapping regions — but verify, don't assume.**
- lane-1 (warm-buffer flags) and lane-6 (tile-cap flags) both added argparse entries + `build_trainer_cmd` threading to `lab_train_cell.py`. Merging both `--no-ff` serially, the `ort` strategy auto-merged with no conflict. The load-bearing verification was NOT "merge succeeded" but: (a) `--help` shows ALL 7 new flags; (b) a default WL5 `--dry-run` emits NONE of them (production path byte-identical); (c) a combined `--dry-run` threads all of them through; (d) tests green. Always run that 4-point check after merging concurrent edits to a shared file.

### 2026-05-23 (LF1-followups L2 — the runaway hides in steps/wall/age, NOT in per-version tile; and run the missing control)

**A wave-mode runaway does NOT show up in the per-version `tile` — that's barrier-bounded and V-invariant. Watch steps/epoch, wall/epoch, new-positions/epoch, and `age`.**
- Symptom: mapping the V=512 runaway boundary in `lab_train_cell --max-epochs 18`, I watched the `wave[vN tile=X]` field. It sat at ~85 for V=256, V=384, AND V=512 — flat and identical. I nearly filed "lab_train_cell can't reproduce the runaway; the knee is an artifact."
- Root cause: `lab_train_cell` sets `--worker-min-games = workers×games_per_batch` (8×8=64) as the wave barrier, so the per-version tile is structurally bounded at ~64–100 regardless of wave_size. The runaway instead manifests as the **trainer falling behind**: `age` grows (2→3), so it drains *more stale versions per epoch* → steps/epoch, wall/epoch, and new-positions/epoch climb monotonically. At V=512 uncapped: steps 22→154, wall 6.8→19.9s, new 77→630 over 18 epochs. At V≤384: flat. The knee (384,512] is real — visible only in the right columns (`trajectory.tsv`: `steps`, `wall_s`, `new`), not `tile`.
- Lesson: for wave-mode runaway/throughput questions, the load-bearing trajectory columns are steps/epoch + wall/epoch + new-positions/epoch + age. The per-version tile is a barrier artifact in `lab_train_cell` and will mislead you. (This is the same "attribute before concluding" discipline as the wave-mode aug/s-is-trainer-gated entry — wave-mode bookkeeping repeatedly hides the real signal in a column you weren't watching.)

**Run the missing control before importing another harness's result as your data point.**
- Symptom: I had V=256 bounded, V=384 bounded, and "V=512 divergent" — but the V=512-divergent point came from LF1 (a `run_sweep` run), not from a `lab_train_cell` run I'd executed. I almost concluded the knee from a cross-harness comparison. I also jumped straight to the *capped* V=512 cell, so when it came back bounded I couldn't tell if the cap worked or if lab_train_cell just doesn't diverge.
- Fix: run the missing uncapped-V512-in-lab_train_cell control. It diverged (steps→154), which simultaneously (a) confirmed the knee, (b) confirmed lab_train_cell reproduces the runaway, and (c) made the capped run a valid A/B proving the cap tames it. One control closed three open questions.
- Lesson: every comparison arm must be measured in the SAME harness/config; don't borrow a number from a sibling run as a control. And when validating a fix (the cap), always run the un-fixed control in the same harness, or a "bounded" result is uninterpretable.

### 2026-05-23 (delta-e run-1 — the GLOBAL editable install silently pointed the MAIN repo at a stale worktree)

**`python scripts/*.py` in the clean main checkout imported `gomoku` from a 146-commit-old agent worktree — and nothing told me until an ImportError.**
- Symptom: a brand-new `delta_e_harness.py` head-to-head smoke crashed with `ImportError: cannot import name 'fuse_model_for_inference' from 'gomoku.model'` — and the traceback showed the path `/Users/jason/code/gomoku/.claude/worktrees/agent-a7d0c83fc186e0497/gomoku/model.py`. The function exists in main (model.py:114); it did NOT exist in that stale worktree (commit cc6aa4e, 146 commits behind main).
- Root cause: some other agent had run `pip install -e .` from inside its worktree, which rewrote the GLOBAL `__editable___gomoku_0_1_0_finder.py` MAPPING to `{'gomoku': '<that worktree>/gomoku'}`. The editable finder is a meta-path finder, so it wins for ANY process whose `sys.path[0]` isn't the repo root. **`python -m gomoku.X` dodges it** (cwd goes on sys.path first) — which is why the train_replay forks and earlier `-m` invocations worked fine — but **`python scripts/foo.py` does NOT** (sys.path[0] = `scripts/`, so the finder resolves `gomoku` to the stale worktree). This is invisible: the run "works," just against old code. Δelo run-1's entire anchored eval ran on 146-commit-old gomoku (the ceiling verdict was robust to it, but it could just as easily have been a silent correctness bug).
- Detection: `python -c "import gomoku; print(gomoku.__file__)"` from `/tmp` (NOT from the repo root, or cwd-precedence masks it) — if it doesn't print `/Users/jason/code/gomoku/gomoku/__init__.py`, the finder is hijacked. Or read the MAPPING directly: `grep MAPPING <site-packages>/__editable___gomoku_0_1_0_finder.py`.
- Fix: `cd ~/code/gomoku && python -m pip install -e . --no-deps` repoints the finder MAPPING to main. **Add this to session-start: after the BOX IDLE check, verify the editable install points to main before running any `python scripts/*.py` lane** — otherwise every harness invocation is silently running stale `gomoku`.
- Structural fragility (flag, not yet fixed): because ANY agent's `pip install -e .` clobbers the one global finder, and this repo runs many concurrent worktree agents, the main checkout's import target is a shared mutable resource that drifts. Options for a real fix: per-worktree venvs, or never `pip install -e` from a worktree (use `PYTHONPATH` in worktree agents instead — which the entry above already recommends for the agent side). The "merge-local-main-at-startup" guardrail does NOT address this (it fixes the branch base, not the global finder).

### 2026-05-24 (Δelo Derby — built it single-process, ran the whole machine at ~30% GPU for hours)

**A long autonomous training experiment was built on single-process `gomoku.train` (one stream) instead of the production multiprocess recipe — GPU sat at ~30%, ~70% of the M5 Max idle, every wall-time under-counted.**
- Symptom: the derby raced 8 recipes for ~9h, one `gomoku.train` process at a time. Jason noticed the GPU was at ~30% and called it: "we're waiting on training and my GPU is at 30% because we were just not running more processes with available resources." Correct — the chunk engine was a lone in-process self-play stream (no `--wave-mode`, no `selfplay_worker`s), which is generation-bound and can't fill the GPU.
- Why it happened: I reached for single-process `gomoku.train` for *overnight robustness* (no oversubscription, clean isolated per-recipe wall-times, simple crash-resume). That tradeoff quietly bought a 3×+ wall-clock penalty AND made every Δelo/hr number unrepresentative.
- Root lesson (this is literally [[project-perf-bench-lesson]] biting again): **single-process benches/runs under-count production parallelism.** The production recipe (`scripts/run_sweep.py`, wave-mode, 1 trainer + 8 `selfplay_worker`s) saturates the machine; single-process does not. If an experiment's headline metric is wall-clock / Δelo-per-hour, it MUST run in the saturated (multiprocess) config or the numbers are busted.
- Diagnostic worth doing EARLY on any training-speed experiment: `pgrep -fc 'gomoku.train|selfplay_worker'` (how many streams?) + check GPU util. One stream + 30% GPU = you're leaving the machine on the table. Also grep the per-epoch log for `(Xs: gen=Ys train=Zs)` — if `gen >> train`, you're generation-bound and parallel workers will help; if `train >= gen` you're trainer-bound (rare, only at very low sims).
- Fix going forward: for wall-clock-sensitive experiments, drive `run_sweep` wave-mode cells (multiprocess) — OR run K single-process recipes concurrently to fill the GPU (~3 at 30%/stream). Either way, MEASURE in the config you'd actually deploy. Build it into the harness, don't bolt it on.

### 2026-05-24 (stopping the Derby — `pkill -f delo_derby` orphans the chunk, `pgrep -c` is a no-op on macOS, and a 100%-full disk blocks the Bash tool itself)

**`pkill -TERM -f delo_derby` kills ONLY the scheduler — the running chunk (`run_sweep` + trainer + 8 workers + eval_worker) is orphaned and keeps training.**
- Symptom: asked to stop the derby; `pkill -TERM -f delo_derby` reported success and the scheduler died, but the `sims100` chunk (trainer `gomoku.train …`, 8 `selfplay_worker`s, an `eval_worker`) kept running and kept writing game records. The skill's "Remove/stop" section says `pkill -TERM -f delo_derby` → "current chunk finishes its teardown," but that only holds if delo_derby's SIGTERM handler forwards to and reaps its children — in this run it did not (or the child `run_sweep` exited without forwarding), so the grandchildren were reparented to init and ran on.
- Root cause: `pkill -f delo_derby` matches only processes whose command line contains the literal "delo_derby". The chunk's `run_sweep`/`gomoku.train`/`selfplay_worker` command lines do NOT — so they're never signalled by that one pkill.
- Fix / correct stop sequence: (1) delete the watchdog cron FIRST (else it relaunches the workhorse mid-stop). (2) `pkill -TERM -f delo_derby`. (3) Then ALSO SIGTERM the chunk directly, scoped to the cell so you don't hit other tenants: `pkill -TERM -f 'gomoku.train.*<cell>'` + `pkill -TERM -f 'selfplay_worker.*<cell>'` + `pkill -TERM -f 'eval_worker.*<cell>'` (the trainer self-saves `latest.pt` on SIGTERM). (4) Wait for the save (>15s for the 1.5M buffer) and verify `pgrep -fl '<cell>'` is empty. Update the skill's Remove/stop recipe to include the scoped chunk-kill, not just the scheduler kill.

**`pgrep -fc` / `pgrep -c` does NOT work on macOS (BSD pgrep has no count flag) — and the skill's own diagnostics recommend it.**
- Symptom: my wait-loop used `pgrep -fc delo_derby`; it printed `usage: pgrep …` to stderr and returned empty, so `[ "$D" -eq 0 ]` mis-evaluated and the loop exited after one iteration without waiting or nudging the orphaned chunk. (Note: line ~390 of this very friction log recommends `pgrep -fc 'gomoku.train|selfplay_worker'` — that command is broken on this Mac.)
- Fix: count with `pgrep -f '<pat>' | wc -l | tr -d ' '`. Reserve `pgrep -fl` (list) for display. Never rely on `-c`.

**A 100%-full Data volume blocks the Bash tool ITSELF — the harness can't create the command's output file (ENOSPC), so NO command runs until space is freed.**
- Symptom: mid-teardown, every Bash call failed with `ENOSPC: no space left on device, open '/private/tmp/claude-501/.../tasks/<id>.output'` — both foreground and `run_in_background`. The pkills I "sent" never executed (the harness opens the output file before running the command). df later showed `/System/Volumes/Data` at 100%, 2.3–3.3Gi free.
- Why it compounded: the orphaned `selfplay_worker`s kept writing `game*.pt` shards into `_records`, eating the last GB while I couldn't run anything to stop them. Classic deadlock: can't free space without a command, can't run a command without space.
- Fix / what worked: retry Bash a few times (a worker dying on its own ENOSPC frees a few KB — enough for the tiny output file), then immediately `rm -rf sweep_runs/derby-*/checkpoints/_records` (transient, already-ingested game shards; regenerated on resume) — freed ~21G here. `_records` is the safe disposable lever; never delete `latest.pt`/`_peaks` to free space (those are the resumable/best checkpoints).
- Lesson: on a training box, watch free disk as a first-class health metric — a full disk silently corrupts clean-saves AND disables your own tooling. The derby's footprint is large (~160G across cells; `derby_v3/_peaks` alone was 80G of best-checkpoint snapshots, `_records` 21G of disposable shards). Add a disk-free check (`df -h /System/Volumes/Data`) to session-start hygiene alongside the BOX IDLE + editable-install checks, and clear `_records` between long runs.

### < add new friction-smoothing entries here as they appear >

## Self-improvement clause

**Future sessions: this skill gets better when you write it better.**

At the end of every session — successful, halted, or escalated — append:

1. A new entry to the Friction-smoothing log above (any non-trivial bug, surprise, or workflow gap you hit, with the symptom + root cause + fix + lesson).
2. A note in the relevant wiki page if the lesson is project-durable (not just personal-to-Claude). Memories also go to the wiki per [conventions.md § Memories also go to the wiki](/Users/jason/code/gomoku/wiki/topics/conventions.md).
3. If a stop-gate triage was non-obvious during the session, propose a row addition to the charter's stop-gates triage matrix in your perf-log session-end entry. Future-Claude reads the charter first.

Be specific. "Watch out for race conditions" is folk wisdom; "the trainer's `--save-every` couples to worker_weights.pt publication at gomoku/train.py:1220" is the kind of note that saves the next session 20 minutes of debugging.

If you smooth a friction that was already documented here, you can reword the entry for clarity — but **do not delete entries**. The accumulating ledger is the value.

## Cross-refs

- Charter: [/Users/jason/code/gomoku/wiki/topics/research-lab-charter.md](/Users/jason/code/gomoku/wiki/topics/research-lab-charter.md)
- Session runbook: [/Users/jason/code/gomoku/wiki/topics/research-lab-session-runbook.md](/Users/jason/code/gomoku/wiki/topics/research-lab-session-runbook.md)
- Reviewer role: [/Users/jason/code/gomoku/wiki/topics/research-lab-reviewer-role.md](/Users/jason/code/gomoku/wiki/topics/research-lab-reviewer-role.md)
- Conventions: [/Users/jason/code/gomoku/wiki/topics/conventions.md](/Users/jason/code/gomoku/wiki/topics/conventions.md)
- Live queue: [/Users/jason/code/gomoku/wiki/ops/gpu-queue.md](/Users/jason/code/gomoku/wiki/ops/gpu-queue.md)
- Best cells: [/Users/jason/code/gomoku/wiki/ops/best-cells.md](/Users/jason/code/gomoku/wiki/ops/best-cells.md)
- Receipts: [/Users/jason/code/gomoku/wiki/ops/experiment-ledger.md](/Users/jason/code/gomoku/wiki/ops/experiment-ledger.md)
- Baselines: [/Users/jason/code/gomoku/wiki/ops/baselines.md](/Users/jason/code/gomoku/wiki/ops/baselines.md)
- Narrative log: [/Users/jason/code/gomoku/wiki/ops/perf-log.md](/Users/jason/code/gomoku/wiki/ops/perf-log.md)
- Sister skill (training pipeline, not the perf lab): [[gomoku-train]]
- Memories: [[feedback-lab-scheduler]], [[feedback-autonomy-denylist]], [[feedback-know-the-machine]], [[feedback-memories-to-wiki]], [[feedback-merge-commits]], [[project-gomoku]], [[project-coreml-reality]]
