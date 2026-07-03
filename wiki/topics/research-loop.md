# The continuous research loop

> **Historical / paused.** The autonomous flywheel described here has **stopped**
> (see [derby.md](../derby.md) for status). Kept as the intake + governance design
> record; the roles and rules below still describe how the loop was wired.

The lab ran a never-ending flywheel: **a researcher proposes ideas → Jason gates →
the orchestrator implements + races → Δelo/hr allocates compute → verdicts feed back
into new ideas.** The derby was the engine; this doc is the intake + governance around
it. (Set up 2026-05-25.)

## The four roles

1. **Researcher (proposes).** A periodic agent (cron-spawned, or on-demand) that reads
   the current standings, recent verdicts, the wiki "open candidates", the friction
   log, and the existing backlog (to avoid dupes), then files **1–3 NEW ideas** as
   GitHub issues with a title-card body (lever / hypothesis / expected Δelo signature /
   maps-to-cell / build cost). It **only proposes** — files with the **`deferred` label**
   (excluded from the ready query). It does not implement or race.
2. **Jason (gates).** The intake gate. Jason reviews deferred ideas
   (`gh issue list --label deferred`) and **gates** the ones worth building by removing
   the gate label: `gh issue edit <N> --remove-label deferred` (or just says which).
   Anything still labelled `deferred` is parked and never surfaces in the ready query.
   This is the "researcher proposes, you gate" decision (2026-05-25), enforced by the
   `deferred` LABEL so it can't be bypassed.
3. **Orchestrator (implements + races).** Pulls gated (open, un-deferred) ideas via the
   ready query, vets (one lever?
   byte-identical-off? implementable?), implements the cell + board entry (CPU-queue
   fan-out per the lab skill), `--dry-run`, hot-adds to the live derby board
   (`delo_derby --resume` reconciles a new board entry as a fresh lane). Claims with
   `gh issue edit <N> --add-assignee @me` (→ also add `in-progress`), then closes with
   the verdict (`gh issue close <N>`, or `Closes #N` in the landing merge).
4. **Δelo/hr (allocates).** The gas pedal — peak-progress + patience (see below). The
   derby decides which racing lane gets the next chunk. Reviewer gates promotion.
   Round-robin head-to-head (`scripts/round_robin.py`) is a **field-ranking /
   diagnostic** once the anchored ladder saturates (~1700) — it ranks siblings, it is
   **NOT the promotion gate**. Sibling H2H is non-transitive; the canonical promotion
   gate is H2H vs a **frozen champion** (`scripts/sliding_gate.py`), never sibling-vs-sibling.

## Backlog = GitHub issues (prefix `derby-` in titles, label `derby-idea`)

Task tracking is **GitHub issues** — one remote, visible from every checkout and session
(migrated from beads 2026-05-28; see the `gomoku-bead-runner` skill). **The gate is the
`deferred` LABEL.** GitHub has only open/closed, so the beads-era extra states
(`deferred`/`blocked`/`in_progress`) are modelled as labels the ready query EXCLUDES.
`label:derby-idea` tags the whole backlog; descriptive labels (`researcher`,
`from-verdict`) say where an idea came from. Lifecycle:

| state | meaning | who sets it | in the ready query? |
|---|---|---|:--:|
| open + `deferred` | researcher filed it; **awaiting the gate** | researcher | **no** (label excluded) |
| open, un-deferred, unassigned | Jason **gated** it — approved to build | Jason | **yes** |
| assigned + `in-progress` | orchestrator is implementing / racing it | orchestrator | no |
| closed | verdict filed (promoted / rejected / superseded) | orchestrator | no |

The **ready query** is:
```bash
gh issue list --state open --search 'no:assignee -label:blocked -label:deferred -label:in-progress -label:epic -label:runner-domain -label:human-gated'
```
So filing with the `deferred` label keeps an un-gated idea OUT of the ready query — you
can never accidentally pick up something Jason hasn't approved. Commands:
- backlog: `gh issue list --label derby-idea`; awaiting-gate: `gh issue list --label deferred`
- **gate (Jason):** `gh issue edit <N> --remove-label deferred`  ← the one action that approves an idea
- build (orchestrator): `gh issue edit <N> --add-assignee @me --add-label in-progress`; `gh issue close <N>` (or `Closes #N` in the merge)
- An EPIC (`epic` label) is excluded from the ready query; its subtasks carry `deferred`
  (or `blocked` with a dependency note in the body) until the epic is gated.

**Provenance backlink.** Every idea carries an `external_ref: claude-session:<id>` line
in its issue body — the Claude session that CREATED it (`$CLAUDE_CODE_SESSION_ID` at
create time). To recover the reasoning/context that generated an idea,
`claude --resume <id>`. Stamp it at create by including the line in `--body-file`:
`gh issue create --title … --body-file <body-with-external_ref-line> --label derby-idea`.

## Δelo/hr — peak-progress + patience (the gas pedal)

`scripts/delo_derby.py:delo_per_hr` + `pick_priority` (rebuilt 2026-05-25, replacing
last-chunk slope which starved mid-swing lanes):
- **rate = peak-progress**: (best elo in the last `peak_window` chunks − best elo
  before it) / window-wall. A dip *below* the running peak no longer zeroes the rate
  — only failing to set NEW peaks does.
- **patience tier**: lanes with `chunks_since_new_peak < patience_chunks` (still
  actively peaking, presumed mid-swing) outrank plateaued ones, so a learning-phase
  swing isn't read as "stopped climbing".
- **starvation floor**: a non-capped lane unfed for ≥ `starvation_factor × N` picks is
  force-fed, so a slow-resettling late bloomer (vcf was one) still gets periodic gas.
- Board-configurable: `global.{peak_window, patience_chunks, starvation_factor}`
  (defaults 6 / 4 / 2).

## Gauge + janitor (per the lab's standing rule)

Every artifact class gets a janitor + a gauge:
- **Gauge:** `gh issue list --label derby-idea --state all` counts — surface
  `backlog: deferred=P gated=R racing=I closed=C` (by label/state) in the cron narrator.
- **Janitor:** close `deferred` ideas superseded by a landed verdict or a dup; close
  `derby-idea` issues whose cell lost decisively and won't be revisited. Run on review.

## Running the researcher

On-demand: spawn a `general-purpose` agent with the researcher brief (reads standings
+ verdicts + wiki + backlog, files 1–3 ideas with the `deferred` label). For overnight/away pushes,
arm it as a cron (same pattern as the scoreboard cron) at a modest cadence (~3–4h).
It is **session-armed**; re-arm for each long push. It never implements — the gate is
Jason, the build is the orchestrator.

## See also

This page is the *governance* (who proposes, who gates, the lifecycle). The
*mechanics of intake* — how a gated idea becomes a derby lane (config-only cell-clone vs
code-heavy `derby-idea` issue, cell hygiene, the issue format) — are in
[derby-registration.md](derby-registration.md), operationalized by the
`gomoku-derby-register` skill. The loop that races the result is the `gomoku-derby-runner`
skill.
