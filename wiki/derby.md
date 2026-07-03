# The Derby — the Δelo/Δt engine

The autonomous research lab: race training recipes in **time-capped slices**,
score by **Δelo per Δt**, promote on receipts. Three roles keep it honest; a
Reviewer gate audits every lane.

> **Two eras (don't conflate the slice size).** The CLASSIC `delo_derby` v9 ran
> **~300s chunks** (stopped 2026-05-27; see [derby-registration.md](topics/derby-registration.md)).
> The later **autolab** ran **~1-hour slices**. Both are now historical — the
> autonomous derby is **stopped** (status below).

> **← Hubs:** [index](index.md) · sibling hubs: [AlphaZero](alphazero.md) ·
> [Experiments](experiments.md) · [M5-as-Mainframe](m5-mainframe.md) ·
> [Reference](reference.md)

## The three roles

| Role | Job | Skill / page |
|---|---|---|
| **Researcher** | Propose the next lane / cell; register contestants | `gomoku-derby-register`, [idea-pile.md](topics/idea-pile.md) |
| **Trainer** | Run the GPU slice (serial — one GPU tenant) | `gomoku-derby-runner`, [research-lab-session-runbook.md](topics/research-lab-session-runbook.md) |
| **Runner** (issues) | Dispatch code-only work to worktrees, mirror status | `gomoku-bead-runner` |

## Start → Now

- **Started:** the charter + two-queue scheduler (GPU-serial + parallel agent
  fan-out); the autolab went **LIVE 2026-06-19** — ran 6 real 9×9 slices then a
  full 15×15 lane unattended, crowned the first 9×9 + 15×15 champion, 0 failures.
- **Now / where it stopped:** the recorded derby verdicts run to **v9** (concluded
  2026-05-27, on [ops/research-board.md](ops/research-board.md)) — the durable
  levers were a matured champion + `--fpu-reduction-c 0.45` sweeping the
  lookahead-ladder. Work then moved off the 9×9 perf derby to **15×15 training**
  (June `G15-*` runs) and **VCT-science** (June 30 `vctsci-*`), which the board
  hasn't caught up to. The `derby_champ` state file's **final write is 2026-05-28** —
  the last champion snapshot recorded the day after the v9 racing concluded (05-27).
  *(research-board needs a June/15×15 refresh — see the [Ops hub](ops.md).)*

## The pages

| Page | Role |
|---|---|
| [research-lab-charter.md](topics/research-lab-charter.md) | The charter: mission, queues, priority function, tiers, reviewer gate, stop conditions. |
| [autolab-architecture.md](topics/autolab-architecture.md) | The self-driving lab (epic #53): one append-only ledger spine read by trainer/arena/research/worker loops. |
| [autolab-supervisor-and-monitor.md](topics/autolab-supervisor-and-monitor.md) | Unattended overnight operating contract (launchd, monitor digest, `autolab up`/`down`). |
| [cockpit-vs-autopilot.md](topics/cockpit-vs-autopilot.md) | The operating lens: autopilot runs without you, cockpit makes it supervisable. |
| [research-lab-reviewer-role.md](topics/research-lab-reviewer-role.md) | The Reviewer: when it fires, the audit prompts, APPROVE/REVISE/BLOCK. |
| [engine-panel-derby-design.md](topics/engine-panel-derby-design.md) | The calibrated engine-panel derby / eval ladder. |
| [wall-clock-to-elo-metric.md](topics/wall-clock-to-elo-metric.md) | Δelo·Δt⁻¹ as a first-class metric family (MTTE primary). |
| [workflow-orchestration.md](topics/workflow-orchestration.md) | The Workflow feature = the everything-else lane (never the GPU lane). |
| [fleet-management.md](topics/fleet-management.md) | The agent-management north star (roadmap). |

## Live ops surfaces

The surfaces you *touch* to run the lab (GPU queue, bests registry, promotion
gate, benchmark cookbook, derby board) live on the **[Ops hub](ops.md)** —
verified live-vs-archived so you don't trust a stale number. This hub is the
*charter*; the Ops hub is the *surfaces*.

## Full page index — every page in this hub

*Complete map (16 pages); the sections above surface the headline ones.*

| Page | Note |
|---|---|
| [research-lab-charter.md](topics/research-lab-charter.md) | the charter |
| [research-lab-session-runbook.md](topics/research-lab-session-runbook.md) |  |
| [research-lab-reviewer-role.md](topics/research-lab-reviewer-role.md) |  |
| [autolab-architecture.md](topics/autolab-architecture.md) |  |
| [autolab-supervisor-and-monitor.md](topics/autolab-supervisor-and-monitor.md) |  |
| [cockpit-vs-autopilot.md](topics/cockpit-vs-autopilot.md) |  |
| [engine-panel-derby-design.md](topics/engine-panel-derby-design.md) |  |
| [wall-clock-to-elo-metric.md](topics/wall-clock-to-elo-metric.md) |  |
| [fleet-management.md](topics/fleet-management.md) | roadmap |
| [workflow-orchestration.md](topics/workflow-orchestration.md) |  |
| [derby-registration.md](topics/derby-registration.md) | intake mechanics |
| [research-loop.md](topics/research-loop.md) | governance / roles |
| [event-log.md](topics/event-log.md) | lab_log.py event stream |
| [sliding-derby-measured-outcomes-design-v2.md](topics/sliding-derby-measured-outcomes-design-v2.md) | current design of record |
| [sliding-derby-design.md](topics/sliding-derby-design.md) | v1 blueprint; superseded by v2; keep as reuse-ledger with banner _([superseded])_ |
| [workflow-harness-capabilities.md](topics/workflow-harness-capabilities.md) | probe appendix under v2; merge-candidate |
