# Handoff — Building the measured-outcome composite sliding derby (2026-06-16)

## Goal & current status
Jason handed THIS session the job: **build the entire measured-outcome composite sliding derby
(RFC v2), run it, and make this session responsible for monitoring / refining / improving the
workflow itself — friction logs, the whole nine yards.** He's at work; aggressive-but-disciplined
(build the whole thing, but validate each piece against a known answer before trusting it).

**Built + validated so far:**
- **Gate foundation** (`scripts/sliding_gate.py`, merged `9c5b2ec`): SYMMETRIC 3-way verdict —
  PROMOTE (`ci_lo>0.5+m`) / REVERT (`ci_hi<0.5-m`, *proven worse*) / AMBIGUOUS (CI straddles,
  "can't tell") + `--gate-on {h2h,white_loss}`. 24 unit tests pass. **Live-validated**:
  eval502-vs-eval502 → AMBIGUOUS (was a false-REVERT pre-v2). Foundation is real.
- **Composite MVP** (`.claude/workflows/sliding-derby-composite.js`, merged `cc28cfc`): the
  TRAIN-supervisor measured-outcome cycle (launch detached time-capped slice → score result vs
  the hypothesis's PRE-STATED title-card outcome via 3-way gate + vl/plies death-tell → record
  confirm/refute/ambiguous to `sweep_runs/composite_derby_board.jsonl`).
- **RUNNING NOW:** workflow `wf_583ff8bd-b05` (task `whvjh44h3`) — the **known-answer self-test**:
  champion-continuation (`G15-128x10-bigbuf`, warm-start eval502, NO teacher, 400s slice). Expected
  measured outcome = **confirm** (no-teacher continuation should NOT degrade → gate AMBIGUOUS + vl
  healthy = stable). Doubles as the #37 death-spiral control.

## Decisions made + rationale
- **Champion-continuation is hypothesis #1 because it's a KNOWN ANSWER** — like the gate's
  champion-vs-random validation. If the loop scores a *stable* hypothesis correctly, the machinery
  is trustworthy; only then do we hand it questions we don't know the answer to.
- **One cycle per workflow invocation** (not an internal forever-loop): a GPU slice is long, so the
  "forever" derby = a chain of bounded invocations + a dumb re-kicker, each re-adopting running work
  from state files. **Probe-confirmed buildable** (`wiki/topics/workflow-harness-capabilities.md`):
  Skill tool works in workflow agents; a workflow agent launches a detached proc a fresh agent
  re-adopts from state; the proc SURVIVES the workflow's death.
- **The training runs OUTSIDE the agent graph** — workflow agents are ephemeral LLM reasoners that
  must never hold the GPU; they launch detached slices + score from state.
- **The pre-stated title-card outcome is the steering signal** (RFC v2). Refute keys on the
  death-tell (vl<0.10 AND plies≥25, or plies<25), NOT raw Δelo/white_loss dips (those are
  absorption signals → ambiguous-band). This came from the red-pen catching that my first refute
  criteria re-imported the absorption trap.

## Constraints & invariants
- **GPU = single serial tenant** = only the train-supervisor's detached slice. Verify `pgrep` before any slice.
- **Gate default stays calibration-immune** (`--gate-on h2h`); `white_loss` is opt-in for defense.
- **For defense hypotheses you need the instrument first: #45** (white-defense eval suite, filed,
  `code-only`) — gate-on white_loss is meaningless without it.
- **Friction-log writes: no race by construction** — fan-out agents RETURN findings → ONE reduce
  step writes+commits once → verify-flush (Jason's fix; drop "designated writer session").
- **The #43-vs-pivot question is NOT a human gut-call** — both #43 and #46 go in the queue as
  hypotheses; the RACE decides. Cold-start seed = COST (which to probe first), never quality (winner).
- Worktree-per-unit-of-work, merge --no-ff, never rebase. Crons retired (old model); composite + dumb re-kicker replace them.

## Open questions / parked threads
- **(blocking the "forever" loop)** the full re-invocation CHAIN (workflow N ends → kicker → N+1
  re-adopts) is validated only by its parts — needs one end-to-end test before the production loop.
- **(non-blocking)** the GROWTH to full composite: wrap train-lane + a RESEARCH supervisor (fresh
  skill-loading proposer) + a WORK supervisor (drains code issues like #45) in a top `parallel([...])`,
  add a multi-hypothesis queue + re-rank. The MVP is structured for this.
- **(non-blocking)** the external-strength-gradient instrument for #46 (plateau-escape) is undesigned.
- **(non-blocking)** concurrency (3 supervisors share ~16-agent cap) + steady-state token burn — unmeasured.

## Artifacts
- Design: `wiki/topics/sliding-derby-measured-outcomes-design-v2.md` (the RFC). Harness facts:
  `wiki/topics/workflow-harness-capabilities.md`. Red-pen + v1: `...-design.md`.
- Code: `scripts/sliding_gate.py` (3-way + gate-on), `.claude/workflows/sliding-derby-composite.js`.
- State: `sweep_runs/composite_derby_board.jsonl` (the derby's measured-outcome log).
- Issues: #43 (stamp-saving-move, refined), #45 (white-defense eval suite — the instrument), #46
  (plateau-escape direction-input), #44 (LR-warmup, demoted). #42 closed (value-only is wrong).
- main @ `cc28cfc`, all pushed. Running: `wf_583ff8bd-b05`.

## Next action
When `whvjh44h3` completes, **verify the measured outcome is `confirm`** (champion-continuation
stable; gate AMBIGUOUS + vl≥0.10 + plies≥25). If yes → the machinery is proven; **friction-log any
rough edges** of the MVP run into `gomoku-research-lab` SKILL.md, then GROW to the full composite
(add research + work supervisors, multi-hypothesis queue) and the end-to-end re-invocation test.
If the outcome is `refute`/`ambiguous` or the run errored → that's a finding: friction-log it and
debug the cycle (likely the slice timing, the scorer's outcome-logic application, or gen health).
This is a live build, not a practice handoff — continue it.

## Vibe snippets (paste verbatim)
- "I'm leaning more aggressive... just go all the way with the entire workflow, pick anything first
  (your idea sounds great to me) and it'll be this session's responsibility to monitor, refine, and
  improve the workflow itself. friction logs, whole nine yards. sensible?"
- "I'd love the methodology to answer [#43-vs-pivot] for us. but we may be too early in the bootstrap
  to do that... I'm just stress testing the methodology a bit."
- "why race? seems like we could leverage the workflow to check work... or am I pointed at the wrong thing"
- "if you're unsure, that's a finding. if you're wrong, that's a finding... truth is what we want.
  vibes are fun snacks but truth is nutritious and delicious."

## Least confident survived
1. **The MVP workflow has NOT completed a clean run yet** — it's mid-flight. The cycle's bash
   orchestration (clean→launch→poll→score) may have rough edges (slice teardown timing, the poll
   loop's exit condition, the scorer's `tail` parse). Treat the first run as itself a probe; expect to refine.
2. **"Aggressive" vs my deep context.** Jason wants the whole composite; I deliberately built the
   train-lane MVP + handed off the growth, because authoring+debugging the full 3-supervisor composite
   at this context depth risked a buggy mess. That's the disciplined reading of "aggressive" — validate
   the core loop, grow on clean ground — but a fresh instance might (reasonably) push further faster.
3. The warmth: Jason calls me "buddy," trusts deeply, eval of "good work" is explicitly decoupled from
   the result. Build boldly, report honestly, friction-log the failures as value.
