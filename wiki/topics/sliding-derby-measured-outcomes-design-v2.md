# Design RFC v2 — Measured-Outcome Sliding Derby (composite workflow)

**Status:** v2, 2026-06-16. Supersedes the v1 draft (`sliding-derby-design.md`)
after: an adversarial red-pen (`redpen-sliding-derby-design`), a harness-capabilities probe
(`workflow-harness-capabilities.md`), and a friction-log-coordination simplification (Jason).
Written as a **DIFF against what already runs** — not a greenfield design.

## What already exists (keep / change / delete)
- `scripts/sliding_derby_runner.sh` — the running shell loop (train slice → gate → next). **KEEP** the
  shape; it becomes the deterministic mechanical daemon.
- `scripts/sliding_gate.py` — the frozen-reference H2H gate. **KEEP** as a reusable primitive;
  **CHANGE** it to add an AMBIGUOUS verdict + a `--gate-on` metric selector (below).
- `scripts/sliding_derby_watchdog.sh` — keep-alive. **KEEP** (or subsume into the workflow's monitor loop).
- Board + `sliding_derby_verdicts.jsonl` — state files. **KEEP**; these are how a fresh workflow
  invocation re-adopts running work.
- The cron loop (heartbeat/researcher/worker) — **DELETED** 2026-06-16; the composite workflow replaces it.

## What HOLDS from v1 (the red-pen validated these — keep verbatim)
The §0 diagnosis (kill-too-early vs kill-too-late; absorption is expected; *the discriminator is
"failure that stops teaching," and a no-recovery mechanism can fire after one result*); the
title-card-as-pre-committed-kill-switch; the AMBIGUOUS-band-not-threshold instinct; fresh-per-session
for the runner; the gate primitive; naming the ~1-slice eval lag. *"Instincts right end-to-end; the
gaps were all in mechanism and signal-wiring."*

## CHANGE 1 — fix the discriminator signals (red-pen's strongest concern)
v1's refute criteria (`Δelo<-100 OR white_loss rises 2 laps`) were the exact **absorption** dip-signals
to IGNORE. The true death-tell is **value-loss collapse with plies HELD** (verified: #42 died vl
0.16→0.06, plies 40-50). Refute now requires a strength signal AND a learning-dynamics signal to AGREE:
```
refute = (vl < 0.10 AND plies >= 25 sustained)   # value-poisoning death
       OR (plies < 25 sustained)                  # fast-attack collapse
       OR (regression past a per-lever LAP/EPOCH budget)
confirm = the hypothesis's targeted metric clears its CI in the right direction
ambiguous = everything else  → one more lap (bounded), never an auto-kill
```
Raw Δelo / white_loss dips are **ambiguous-band inputs, never refute.** The gate must ingest vl + plies
(it doesn't today).

## CHANGE 2 — gate on the signal that MOVES (red-pen: the dead needle)
On the ~50-elo 128x10 plateau the frozen-peer H2H verdict is ~always REVERT/ambiguous (n≈120 to resolve
~70 elo) — no usable gradient. The one signal that moves on the defense arc, `white_loss_rate`, is
hard-coded "never gates." Fix: a `--gate-on <metric>` selector. For defense hypotheses, gate on
`white_loss` over the **fixed white-defense eval suite (#45)** with ITS OWN CI; demote overall H2H Δelo
to a guardrail ("did not regress past −100"). **Default stays calibration-immune** (white_loss logged-only)
so the primitive isn't broken for the general case.

## CHANGE 3 — build the AMBIGUOUS band for real + symmetric refute-CI
`decide_verdict` is binary today (a CI-straddling win collapses to REVERT). Add a CONTINUE/AMBIGUOUS
verdict; thread it through the runner (which greps only `PROMOTE|REVERT`). Bound AMBIGUOUS with a hard
lap budget (N-more-then-forced-decision) so it can't be a GPU sink, + hysteresis so n=120 eval noise
can't flip a lever lap-to-lap. **Symmetric power:** refute must clear its OWN CI (`ci_hi < 0.5−margin`),
mirroring promote's `ci_lo > 0.5+margin` — else a noisy lap false-refutes a good lever.

## CHANGE 4 — the architecture (replaces v1 §4; PROBE-CONFIRMED buildable)
v1's "in-loop fresh Claude agent spawned by the shell" was unbuildable (no programmatic `claude -p`
spawner; and `claude -p` is going away). The composite is buildable on the harness as-is — all three
load-bearing facts were probed (`workflow-harness-capabilities.md`):
- **Skill tool IS callable inside workflow subagents** + skill MD readable → skills-as-memory works; the
  freshness lives in the *agents*, not the session.
- **A workflow agent launches a detached process; a fresh agent re-adopts it from state** (PID/log files).
- **The detached process SURVIVES the workflow's death** → "forever" = bounded workflow invocations
  chained by a *dumb* re-kicker, each re-adopting running training from state.

Shape (Jason's intuition, fits one-level `workflow()` nesting):
```
TOP workflow (loops; coordinates):
  parallel:
    TRAIN-supervisor   : loop → launch detached slice (nohup) → spawn FRESH gate/score agent →
                          score eval vs the hypothesis's pre-stated outcome → update priority/board.
                          Monitor for child-wedge (dead agent → null → relaunch).
    RESEARCH-supervisor: loop → spawn FRESH researcher agent (Skill-loads latest) → propose/refine
                          hypotheses + their title-card outcomes into the queue.
    WORK-supervisor    : loop → spawn FRESH worker agent → drain ready code-only issues (implement→review→merge).
  (dumb cron re-kicks the TOP workflow when an invocation hits its token/agent cap; state persists in files.)
```
The training process lives **outside** the agent graph (agents are ephemeral LLM reasoners that must never
hold the GPU); agents launch + supervise it from state.

## CHANGE 5 — friction-log coordination: no race, by construction (Jason)
There is no inherent write-race — the workflow IS the coordinator. Pattern: **fan-out agents RETURN their
findings (don't write) → ONE reduce step folds them into SKILL.md + commits once → a verify-flush check**
confirms the lesson actually landed. The verify step isn't race-avoidance, it's the *guarantee* that makes
"skill is the memory, session is cache" reliable (an agent can't silently forget to flush). Drop v1's
"designated writer session." (Read-staleness is a non-issue: the friction log is append-only; an agent that
loaded before a peer appended just gets the new lesson next cycle — eventual consistency is fine.)

## CHANGE 6 — BOOTSTRAP / cold-start (new; the methodology answering its own questions)
The system's purpose is to answer "which axis is worth the compute?" by measurement, not human gut. But it
can't until it can MEASURE and has run something. So:
1. **Instruments first.** No outcome is measurable without its instrument. Build the white-defense eval
   suite (#45) and an external-strength-gradient eval (for plateau-escape #46) *before* the relevant
   hypotheses can be raced. Instruments are the true first prerequisite.
2. **Cold-start seed = COST, not quality.** The first probe-order comes from a build-/run-cost prior (or a
   human naming the cheapest probe) — *never* from a human picking the winner. Measurement always picks the
   winner. This is the line that keeps the methodology honest: the human seeds the *probe order*, the race
   decides the *outcome*.
3. **Staged, earned trust** (mirrors the gate's champion-vs-random validation):
   - (a) Test the loop on a KNOWN-answer case: champion-continuation (no teacher; outcome "stays within
     noise of eval502" → should CONFIRM) alongside a deliberately-broken lever (should REFUTE). Verify the
     loop scores both correctly before trusting it on the unknown.
   - (b) THEN hand it the real open question: **#43 (defense swing) vs #46 (plateau-escape) both go in the
     queue with pre-stated outcomes; the race ranks them.** That is the resolution to "#43 vs pivot" — not a
     human gut-call, a measured outcome. (Champion-continuation doubles as the #37 death-spiral control.)

## Still-open / untested (do NOT assume — probe before relying)
- The full **re-invocation chain** (workflow N ends → dumb kicker → N+1 re-adopts) — validated only by its
  parts; worth one end-to-end probe before the production loop.
- **Concurrency** (3 supervisors share the ~16-agent cap) + steady-state **token burn** of a looping
  composite — unmeasured.
- The **external-strength-gradient instrument** for plateau-escape — undesigned; #46/#40 (native Rapfi) feed it.

## Next actions
1. Build the gate changes (CHANGES 1-3) — code-only, unit-testable (the worker can do most).
2. Build instruments: #45 (white-defense suite). Scope the external-strength eval.
3. Stand up the composite workflow skeleton; run the **known-answer bootstrap test** (CHANGE 6.3a).
4. One end-to-end re-invocation probe.
5. THEN race #43 vs #46 and let the methodology answer the question.
