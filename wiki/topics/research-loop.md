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
   maps-to-cell / build cost). It **only proposes** — label `proposed`. It does not
   implement or race.
2. **Jason (gates).** The intake gate. Jason reviews `proposed` ideas and relabels the
   ones worth building to `ready` (`bd update <id> --label ready` or just say which).
   Anything still `proposed` is parked, not built. This is the "researcher proposes,
   you gate" decision (2026-05-25).
3. **Orchestrator (implements + races).** Pulls `ready` ideas, vets (one lever?
   byte-identical-off? implementable?), implements the cell + board entry (CPU-queue
   fan-out per the lab skill), `--dry-run`, hot-adds to the live derby board
   (`delo_derby --resume` reconciles a new board entry as a fresh lane). Moves the
   bead `in_progress` → `closed` with the verdict.
4. **Δelo/hr (allocates).** The gas pedal — peak-progress + patience (see below). The
   derby decides which racing lane gets the next chunk. Reviewer gates promotion;
   round-robin head-to-head (`scripts/round_robin.py`) is the final verdict once the
   anchored ladder saturates (~1700).

## Backlog = Beads (repo-local, prefix `derby-`)

Initialized in `.beads/` (dolt, embedded). Idea lifecycle via labels on
`label:derby-idea`:

| label | meaning | who sets it |
|---|---|---|
| `proposed` | researcher filed it; awaiting the gate | researcher |
| `ready` | Jason approved to build | Jason |
| (status `in_progress`) | orchestrator is implementing / racing it | orchestrator |
| (status `closed`) | verdict filed (promoted / rejected / superseded) | orchestrator |

Commands: `bd list --label derby-idea` (whole backlog), `bd ready` / `bd list
--label ready` (gated, buildable), `bd update <id> --label ready`,
`bd close <id> --reason "<verdict>"`.

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
  `backlog: proposed=P ready=R racing=I closed=C` in the cron narrator each cycle.
- **Janitor:** close `proposed` ideas superseded by a landed verdict or a dup; close
  `derby-idea` beads whose cell lost decisively and won't be revisited. Run on review.

## Running the researcher

On-demand: spawn a `general-purpose` agent with the researcher brief (reads standings
+ verdicts + wiki + backlog, files 1–3 `proposed` beads). For overnight/away pushes,
arm it as a cron (same pattern as the scoreboard cron) at a modest cadence (~3–4h).
It is **session-armed**; re-arm for each long push. It never implements — the gate is
Jason, the build is the orchestrator.
