# The continuous research loop

The lab runs a never-ending flywheel: **a researcher proposes ideas → Jason gates →
the orchestrator implements + races → Δelo/hr allocates compute → verdicts feed back
into new ideas.** The derby is the engine; this doc is the intake + governance around
it. (Set up 2026-05-25.)

## The four roles

1. **Researcher (proposes).** A periodic agent (cron-spawned, or on-demand) that reads
   the current standings, recent verdicts, the wiki "open candidates", the friction
   log, and the existing backlog (to avoid dupes), then files **1–3 NEW ideas** as
   beads with a title-card description (lever / hypothesis / expected Δelo signature /
   maps-to-cell / build cost). It **only proposes** — files at **status `deferred`**
   (hidden from `bd ready`). It does not implement or race.
2. **Jason (gates).** The intake gate. Jason reviews deferred ideas
   (`bd list --status deferred`) and **gates** the ones worth building with
   `bd update <id> --status open` (or just says which). Anything still `deferred` is
   parked and never surfaces in `bd ready`. This is the "researcher proposes, you
   gate" decision (2026-05-25), enforced by STATUS so it can't be bypassed.
3. **Orchestrator (implements + races).** Pulls gated (`open`) ideas via `bd ready`,
   vets (one lever?
   byte-identical-off? implementable?), implements the cell + board entry (CPU-queue
   fan-out per the lab skill), `--dry-run`, hot-adds to the live derby board
   (`delo_derby --resume` reconciles a new board entry as a fresh lane). Moves the
   bead `in_progress` → `closed` with the verdict.
4. **Δelo/hr (allocates).** The gas pedal — peak-progress + patience (see below). The
   derby decides which racing lane gets the next chunk. Reviewer gates promotion;
   round-robin head-to-head (`scripts/round_robin.py`) is the final verdict once the
   anchored ladder saturates (~1700).

## Backlog = Beads (repo-local, prefix `derby-`)

Initialized in `.beads/` (dolt, embedded). **The gate is the STATUS, not a label**
(see the friction note below — a label gate is invisible to `bd ready`, so un-gated
ideas get picked up). `label:derby-idea` tags the whole backlog; descriptive labels
(`researcher`, `from-verdict`) say where an idea came from. Lifecycle by STATUS:

| status | meaning | who sets it | in `bd ready`? |
|---|---|---|:--:|
| `deferred` | researcher filed it; **awaiting the gate** | researcher | **no** (hidden) |
| `open` | Jason **gated** it — approved to build | Jason | **yes** |
| `in_progress` | orchestrator is implementing / racing it | orchestrator | no |
| `closed` | verdict filed (promoted / rejected / superseded) | orchestrator | no |

`bd ready` = status `open` AND no blocking deps. So filing as `deferred` keeps an
un-gated idea OUT of `bd ready` — you can never accidentally pick up something Jason
hasn't approved. Commands:
- backlog: `bd list --label derby-idea`; awaiting-gate: `bd list --status deferred`
- **gate (Jason):** `bd update <id> --status open`  ← the one action that approves an idea
- build (orchestrator): `bd update <id> --status in_progress`; `bd close <id>`
- An EPIC's subtasks are also `deferred` until the epic is gated; gating an epic =
  open the epic + its subtasks (they then flow via their `blocks` deps).

**Provenance backlink.** Every idea carries `external_ref: claude-session:<id>` — the
Claude session that CREATED it (`$CLAUDE_CODE_SESSION_ID` at create time; shows as
`External:` in `bd show`). To recover the reasoning/context that generated an idea,
`claude --resume <id>`. Stamp it at create: MCP `create(external_ref="claude-session:"+sid)`
or CLI `bd create … --external-ref "claude-session:$CLAUDE_CODE_SESSION_ID"`.
Gotcha: `bd update` reads stdin, so a tight `for` loop of `bd update` eats its own
iteration list — always `bd update … </dev/null` in a loop (else only the first runs).

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
- **Gauge:** `bd list --label derby-idea` counts by label — surface
  `backlog: deferred=P gated=R racing=I closed=C` (by status) in the cron narrator.
- **Janitor:** close `deferred` ideas superseded by a landed verdict or a dup; close
  `derby-idea` beads whose cell lost decisively and won't be revisited. Run on review.

## Running the researcher

On-demand: spawn a `general-purpose` agent with the researcher brief (reads standings
+ verdicts + wiki + backlog, files 1–3 ideas at status `deferred`). For overnight/away pushes,
arm it as a cron (same pattern as the scoreboard cron) at a modest cadence (~3–4h).
It is **session-armed**; re-arm for each long push. It never implements — the gate is
Jason, the build is the orchestrator.

## See also

This page is the *governance* (who proposes, who gates, the status lifecycle). The
*mechanics of intake* — how a gated idea becomes a derby lane (config-only cell-clone vs
code-heavy `derby-idea` bead, cell hygiene, the bead format) — are in
[derby-registration.md](derby-registration.md), operationalized by the
`gomoku-derby-register` skill. The loop that races the result is the `gomoku-derby-runner`
skill.
