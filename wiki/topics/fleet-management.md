# Fleet Management — the agent-management toolchain

**Role:** The north-star design for landing the *too-many-sessions* problem.
Jason runs many concurrent Claude sessions across many topics; they meander,
overlap, and stall. The **agent-management toolchain** exists to *land that work*
so he can run Sid Bidasaria's **"stop babysitting your agents"** playbook
(verify → multi-Claude → background loops, attention protected — see
[cockpit-vs-autopilot](cockpit-vs-autopilot.md)). It is the *self-improving agent
for the too-many-sessions problem*, built on the path a Claude Code author laid
out. Attention is the scarce resource; this is cockpit, not more autopilot.

> Status: north-star / design. The scripts named below are the intended surface;
> they are **not** present in `scripts/` as of this writing — treat as roadmap
> until landed.

## Three capabilities, one purpose each

| Capability | Purpose | Intended surface |
|---|---|---|
| **Search** | know *what topics are happening* (recall = a corpus search that reaches dead sessions) | `session_db.py` (SQLite/FTS transcript cache), `topics.json`, `session_mindmap.py` (web mindmap) |
| **Post office** | *talk to* the sessions — fills the no-CLI-send-to-a-live-session gap (a DIY Channel) | `postoffice.py` (append-only bus + cursor) + a `cagent` post-office session |
| **Session control** | *actually interact* — supervise and resume | `agent_fleet.py` (status / gauge / digest + resume/fork copy-paste commands) |

## Load-bearing principle: log-based, append-only

**"Nothing gets lost or destroyed, only added or learned."** Every surface is a
log + cursor, never a mutated state: the post-office log+cursor, the FTS cache
(transcripts remain the source of truth), worktree-sessions `.jsonl`,
`events.jsonl`. Same epistemic stance as `TRAINING_WIKI.md` and the wiki's
append-correction rule. A **watch is the wake, a cursor is the truth** — recovery
is a catch-up read from the last cursor, never a replay of mutable state.

## How to apply

When building fleet / session / coordination tooling: make it **log-backed**
(append + cursor, never mutate), **recoverable by catch-up**, and **fold each new
working-path / friction back into the agent-management skill** so the toolchain
self-improves. Jason spawns persistent `cagent`s himself (supervisor-dispatched);
the toolchain's job is to hand him a paste-able prompt, not to act for him.

## Related
- [cockpit-vs-autopilot](cockpit-vs-autopilot.md) — why this is cockpit (attention layer), not more autopilot.
- [event-log](event-log.md) — the same record-≠-report, log-as-source stance for lab observations.
- Source: [Sid Bidasaria — Stop babysitting your agents](../sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md).
