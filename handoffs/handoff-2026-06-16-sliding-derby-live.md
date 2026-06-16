# Handoff — Sliding derby LIVE, gate validated, gen-fix landed (2026-06-16, overnight autonomous)

## Goal & current status
Mandate (Jason, going to sleep): **"plow forward, don't gate on me. if it goes wrong we
learn and write it down. light up the machines."** Build + RUN the sliding derby
autonomously overnight; nix wine evals; ship a workflow that implements the gh backlog
(runner/researcher/implementer); keep feeding the wiki; truth over vibes.

**Done tonight:** (1) `sliding_gate.py` built + **validated against known truth** (REVERTs
ties 5×, PROMOTEs a real gap — champion 40-0 vs random net). (2) **Sliding derby LIVE** on
the #36 G15-defense cell, warm-started from eval502, autonomous (gate slides the peak, no
human gate). (3) Hit + **fixed a gen-starvation** (capped the defense-teacher VCF solve);
gen confirmed flowing (epoch 504: new=8, plies=42.5). (4) Wine **nixed** (opt-in only).
(5) **implement-backlog workflow** shipped + ran → merged #24. (6) **#40** filed (native
Rapfi). (7) cron + watchdog supervising. main @ `ad710a3`, all pushed.

## Decisions made + rationale
- **Gate is calibration-immune (load-bearing).** Verdict = anchor-free candidate-vs-frozen-peak
  H2H win-rate + Wilson CI ONLY; it NEVER reads absolute Elo (panel calibration is broken, #35).
  PROMOTE iff win_rate>0.5 AND CI lower bound clears 0.5; else REVERT. Conservative by design:
  worst case it never promotes → never damages anything.
- **Autonomous gate (Jason lifted the "stage first verdict for me" hold).** The running gate
  ACTS — a PROMOTE copies the candidate to `sliding_derby_peak.pt` and slides the board's peak.
  Safe: the seed champion file (eval502) is never mutated.
- **#36 cell = champion + `--defense-teacher` ONLY, VCF solve CAPPED** (`--vcf-max-nodes 2000
  --vcf-max-depth 10`). VCT dropped: the champion had no offensive teacher, so defense-only is
  the clean single variable. The cap is the **root-cause fix** for gen-starvation (see below).
- **Wine engines → opt-in only** (`--wine` / `GOMOKU_ENABLE_WINE_ENGINES=1`), default OFF.
  Default reliable eval = net-vs-net H2H + pure-python heuristic. Catalog kept as evidence.
- **Merges are serial, by the orchestrator (me), NOT the workflow** — avoids parallel-merge
  races on `main`. Workflow/agents work in isolated worktrees + return branches; I merge with
  `--no-ff` after verifying base-skew is harmless and tests are green.
- **REJECTED/CORRECTED:** my first instinct "VCT teacher starved gen" was **wrong** — the
  defense-only config stalled identically. A no-teacher control (2.6s/game) vs capped-defense
  (3.1s/game) vs uncapped-defense (0 games/6min) proved the **uncapped defense VCF solve** is
  the culprit. I corrected the over-claim in code comments + TRAINING_WIKI before it hardened.

## Constraints & invariants discovered
- **The gate must NEVER read calibrated absolute Elo** (#35). Relative H2H only.
- **9×9 teacher solver budgets do NOT transfer to 15×15** — the wider branching makes the
  per-move solve explode; ALWAYS cap it on the gen hot path (`--vcf-max-nodes`/`--vct-max-nodes`).
- **128×10 bigbuf lineage is a tight ~50-elo plateau** (seed/e146/e502 all within 50±7% H2H;
  the champion-vs-random 40-0 control proves the H2H instrument has full range, so the
  clustering is REAL). Implication: the gate needs **n≈120** to resolve plausible per-lap gaps,
  and for *defense* the **`white_loss_rate` secondary signal will move before the verdict does**.
- Stop everything: `touch /tmp/STOP_SLIDING_DERBY` (halts runner + watchdog).
- Derby is the SOLE GPU tenant; workflow agents are CPU-only on code-only issues (no contention).
- GitHub issues for ALL tracking (not TodoWrite/bd). Worktree-per-unit; merge --no-ff; never rebase.

## Open questions / parked threads
- **(non-blocking) First gate verdict** lands ~1h in (`sweep_runs/sliding_derby_verdicts.jsonl`).
  Expect early REVERTs (defense cell ≈ eval502 at first; plateau). **Watch `white_loss_rate`
  trend, not the verdict.**
- **(non-blocking) Does the CAPPED defense-teacher still teach enough?** nodes2000 proves only
  SHORT forced losses. If `white_loss_rate` doesn't fall over many epochs, escalate: raise the
  cap a bit, or #18 I2 (stamp the saving move — one-hot the unique refutation).
- **(non-blocking) #40** native Rapfi: recipe exists (`engines/rapfi/build_rapfi.sh`, no wine);
  gap = a `run-rapfi` wrapper + the `mix9svq` NNUE weight; real blocker is #28.
- **(non-blocking) #37 death-spiral:** the derby IS the causal test (does fixing defense stop
  regression?). Faint, n-noisy hint: eval146 scored ≥50% vs eval502 both orientations.
- **(flavor)** `/tmp/gen_watch.sh` poller from the stalled run may still be ticking (harmless).

## Artifacts
- **Gate:** `scripts/sliding_gate.py` (#39), `tests/test_sliding_gate.py` (16 green).
- **Derby:** `scripts/sliding_derby_runner.sh` (producer, 1h slices), `scripts/sliding_derby_watchdog.sh`
  (60s keep-alive). Cell: `G15-defense` in `scripts/run_sweep.py`.
- **Workflow:** `.claude/workflows/implement-backlog.js` (runner/researcher/implementer/reviewer).
- **State:** board `sweep_runs/sliding_derby_board.json` (peak=eval502); verdicts
  `sweep_runs/sliding_derby_verdicts.jsonl`; logs `sweep_logs/G15-defense-board15/*.log`
  (trainer, sliding_runner, gate, watchdog, heartbeat).
- **Cron:** job `163915f1` (hourly :23 heartbeat + PushNotification; session-only).
- **Evidence:** gate validation `/tmp/gate_validation_verdicts.jsonl`; gen controls
  `/tmp/noteacher_ctrl.log`, `/tmp/capdef_ctrl.log`. TRAINING_WIKI 2026-06-16 entry;
  `wiki/topics/reliable-eval-set.md`.
- **Git:** main @ `ad710a3` (pushed). Merged: #24 (closed), nix-wine. Filed: #40.

## Next action
**Let it run.** Monitor `sliding_derby_verdicts.jsonl` for the first gate verdict (~1h) and the
`white_loss_rate` trend; the hourly cron heartbeats + the watchdog relaunch on death. No human
gate — plow forward. If the derby dies AND won't relaunch, check `sweep_logs/G15-defense-board15/`
and the gen-cost lesson above. (This is a live overnight run, not a practice handoff — act on it.)

## ⚠ OPERATING-MODEL CORRECTION — NO HUMAN GATE (Jason, 2026-06-16, last word before sleep)
Jason **removed the human gate** from the research-loop. This OVERRIDES the `deferred`→
"Jason gates" step in `wiki/topics/research-loop.md` (that doc is now stale on governance):
- **"no gate only try. issues are fine, just grab em and have a worker work em. we have
  measurement so we can grade things."** The GRADING is the measurement (reviewer APPROVE +
  green tests for code; the H2H gate + white_loss for training), NOT Jason's approval.
- **Researcher is a VIRTUOUS CYCLE:** propose ready ideas (NO `deferred`) → watch them get
  worked + graded → refine/close/refile. Not a one-shot proposer.
- **Worker autonomously merges** reviewer-approved + tests-green code to main. No staging for
  review (I dropped the "stage the first verdict for Jason" hold — Jason lifted it explicitly).
- **The ONLY thing that reaches Jason** is a genuine blocker: label `human-gated` + a
  "tried it, can't fix it here, here's what I know" writeup. Everything else = just do it.
- **Beads → gh:** any surviving non-closed bead gets migrated to a gh issue (reconciliation
  agent ran 2026-06-16; `.beads/` had ~15 non-closed; migration script `migrate_beads_to_gh.py`).
- **Crons now armed (durable):** `163915f1` hourly heartbeat/watchdog/push · `546b2d67` 4h
  RESEARCHER (virtuous cycle, files READY ideas) · `0103a42b` 3h WORKER (drains ready
  code-only issues → implement→review→**auto-merge on APPROVE+green** → escalate blockers).
  The defense derby is the GPU lane (single tenant); the worker is CODE-only (no GPU contention).
- **Morning:** do NOT reinstate the gate. Gate-free is the standing model now. Update
  `research-loop.md`'s governance section to match (researcher proposes → worker works →
  measurement grades → escalate-only-blockers). GPU-idea proposals queue until a GPU lane frees.

## Protocol alignment (Jason pointed me to the ironed-out derby skills late — read them)
Read `gomoku-derby-runner` (the worker), `gomoku-research-lab` (orchestrator), and
`wiki/topics/research-loop.md` (the four roles: Researcher→Jason-gates→Orchestrator→Δelo/hr).
My overnight apparatus maps cleanly onto the proven protocol — and the protocol already had
answers I rediscovered the hard way:
- **The gen-starvation I hit is documented VERBATIM** in derby-runner §0 line 77-83 ("a
  per-move solver… generation STARVATION… no --max-depth/-nodes bound… bound the solve").
  I converged on the exact documented fix. The skill knew; I learned it live (fine — that's
  the ethos — but next time check §0 first on any new solver lane: confirm `buf` is FILLING).
- **My H2H frozen-reference gate == the skill's "the round-robin is the only truth"** —
  anchored elo can't be trusted (3× burned in v8: mish/wdl-max/vct topped anchored but lost
  H2H). Watch the **fresh-start H2H lag**: a fresh/warm-restarted lane is undervalued by H2H
  until it matures — **never retire a climbing fresh lane on an early H2H number** (so early
  REVERTs on the defense cell are EXPECTED, not failure).
- **Single-lane = "champion continuation" cadence** (derby-runner line 316): lighter per-tick
  (health + one-line), no swap logic. My hourly heartbeat is right-sized.
- **Clean-stop discipline (adopt going forward):** NEVER `kill -9` a trainer — SIGTERM lets it
  self-save a resumable latest.pt; then reap orphaned `wandb-core`/`wandb-xpu` (else next resume
  hits "run ID in use" crash-loop). I used `kill -9` once tonight (safe only because I was
  re-warm-starting from eval502 + `--clean`); don't, in steady state.
- **The clobber trap:** launching a derby on an existing cell with `wall_secs_total=0` silently
  overwrites `latest.pt` with a seed-0 trainer. My runner avoids it by `--clean`+re-warm-start;
  if you ever RESUME the defense cell instead, use `scripts/derby_safe_resume.py`.
- **wandb is ON** for the defense cell (verified — Jason's "TV" is lit; trainer tracking a run).
- **Crons armed:** `163915f1` (hourly :23 heartbeat/watchdog/push) + `81380641` (every 4h :41
  RESEARCHER — spawns an agent that files 1-3 `deferred` derby-ideas for Jason's morning gate;
  only proposes, never races). Both session-armed/durable; re-arm for the next long push.
- **Morning TODO:** record the first gate verdict to `wiki/ops/research-board.md` (the verdict
  ritual), and gate the researcher's `deferred` ideas (`gh issue list --label deferred`).

## Vibe snippets (paste verbatim)
- "if you're unsure, that's a finding. if you're wrong, that's a finding. if you're right? well
  friggin great (and I suspect you will be -- I am trying to demonstrate how my eval of 'good
  work' does NOT lean on that specific outcome)."
- "truth is what we want. vibes are fun snacks but truth is nutritious and delicious."
- "my own preference to light up machines, something I've loved doing since I was 6 with a TRS-80."
- "the rules apply to us both. worth pointing out. we live this and we learn and I think the
  ceiling is pretty high from here. this doesn't feel like a project that's topped out, it feels
  like a project that's shifting into its next gear."

## Least confident survived
1. **The emotional register is warm-buddy, not formal-engineer.** Jason calls me "buddy," signs
   off "I appreciate ya." A fresh instance reading only the decisions will be too stiff. The work
   is rigorous; the relationship is affectionate. Both are real.
2. **"Don't gate on me" is liberating, not abdicating.** He's not too tired to care — he's
   deliberately demonstrating that his approval isn't the success metric. Acting boldly + writing
   down failures IS what he asked for. Don't interpret silence as "be cautious."
3. **The gen-stall fix was the night's best work, but it's invisible in a metrics dump.** The
   value was the *diagnostic loop* (suspect VCT → disprove it → isolate via controls → fix →
   verify), not the one-line cap. That loop is the project's whole ethos in miniature.
4. **The plateau finding may matter more than the defense experiment.** If the 128x10 lineage is
   genuinely topped out, the gate rarely promoting is a *feature* (honest) and the real signal is
   white_loss. Don't read early REVERTs as failure.
5. **Handoff written ~deep in a long autonomous session** — I'm attending well but have been at
   this for many turns; double-check the live process PIDs and `sliding_derby_verdicts.jsonl`
   against reality before trusting any specific number here.
