# Workflow Harness Capabilities — probed facts

**Probed 2026-06-16** (throwaway probe workflow `harness-probe-workflow-capabilities`, run
`wf_a1c3b5f5-420`) to put the composite-workflow sliding-derby design on *tested ground*
rather than intuition — applying the prior night's lesson (we'd nearly designed on the
`claude -p`-per-tick mechanism, which is both going away AND, per the red-pen, has no
programmatic spawner). The three facts the composite design hinges on — **all CONFIRMED.**

## 1. Workflow subagents CAN reach skills (skills-as-memory works inside workflows)
- The **`Skill` tool is present + callable** in a workflow subagent's toolset. A bogus name
  `Skill({skill:"__probe_nonexistent_skill__"})` returned `tool_use_error: "Unknown skill:
  __probe_nonexistent_skill__"` — a *lookup rejection* (the tool ran and rejected the name),
  NOT "no such tool available."
- Workflow subagents can also **Read the skill MD directly**
  (`.claude/skills/<skill>/SKILL.md`) including its `## Friction-smoothing log` section — the
  fallback skills-as-memory path (read the accumulated wisdom without executing the skill).
- **Implication:** "fresh-per-cycle decision agents that load the latest self-improving skill"
  is buildable. Each `agent()` is a fresh subagent; it either invokes `Skill(...)` or reads the
  MD → it gets the latest accumulated friction wisdom. **No `claude -p`, no session reload
  needed — the freshness lives in the workflow agents, not the session.**

## 2. A workflow agent can launch a DETACHED process; a later/fresh agent RE-ADOPTS it from state
- A workflow agent ran `nohup bash -c '...' & echo $! | tee pidfile`. The process **reparented
  to init (PPID=1)** — fully detached from the spawning agent.
- A **fresh, memoryless** later agent re-adopted it using ONLY state files (the PID file + the
  marker log): confirmed `kill -0` alive + the log growing. No inherited handle, no assumed PID.
- **Implication:** the GPU lane works as "a workflow agent launches a *detached* training slice
  and writes state (board/PID); subsequent fresh agents supervise/gate/score it *from state*."
  The training process never needs to live *inside* an (ephemeral) workflow agent — which is
  good, because workflow agents are LLM reasoners that must never hold the GPU.

## 3. The detached process SURVIVES the workflow's own death (the load-bearing one)
- After the probe workflow *fully returned*, the detached process (PID 62095) was **still
  alive** (PPID=1, ~66s of independent life past the workflow's end), still writing its marker.
- **Implication:** a workflow-launched GPU process **outlives the workflow.** So a "forever"
  derby = a *sequence of bounded workflow invocations*, each re-adopting the still-running
  training from state files. The re-invoker can be **trivially dumb** (a one-line cron that just
  re-kicks the workflow); all the smarts + freshness live inside the workflow. We don't escape
  needing *a* heartbeat — we make it stupid.

## What this confirms for the composite design
The `1 top workflow → 3 role-supervisors (research / train / work) → fresh agent instances`
shape (Jason's intuition, 2026-06-16) is buildable on the harness as-is:
- **train-supervisor**: launches detached slices, spawns fresh agents to gate/score; the GPU
  process lives *outside* the agent graph and is re-adopted from state. ✅
- **research / work supervisors**: pure fresh-agent work; each loads the latest skill per cycle. ✅
- **"forever"**: bounded workflow invocations chained by a dumb re-kicker; state persists in
  files (board, verdicts.jsonl, PID); each invocation re-adopts running work. ✅
- **one-level nesting** (top → 3 child supervisor-workflows → agents) fits the harness rule that
  `workflow()` nesting is one level deep.

### Still-open / untested (do NOT assume)
- The full **chain of re-invocations** (workflow N ends → dumb kicker → workflow N+1 re-adopts)
  was validated only by its parts (survival + re-adopt-from-state), not end-to-end across two
  real workflow runs. Worth one more probe before the production loop.
- **Concurrency**: 3 supervisor-workflows share the `min(16, cores−2)` agent cap; steady-state
  token burn of a looping composite is unmeasured.
- **Friction-log write race**: if multiple fresh agents append the same `SKILL.md`, they collide
  — the design must designate ONE writer (the supervisor) and have sub-agents return findings
  (per the gomoku-research-lab fan-out rule). Not a harness limit, a design constraint.

## Methodology note
We PROBED these load-bearing assumptions with a ~50-second throwaway workflow *before* writing
RFC v2 — deliberately, because the prior design had nearly committed to an unbuildable
mechanism. **Test the load-bearing harness assumptions with a throwaway probe before designing
on them.** (Sibling of the friction-log entry on the parallel-apparatus fork, 2026-06-16.)
