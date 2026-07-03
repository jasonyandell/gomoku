# Registering a contestant with the Δelo Derby (the intake model)

How a new idea becomes a derby lane. This is the **synthesis / source of truth**; the
operational playbook (commands, templates) is the `gomoku-derby-register` skill, and the
loop that *consumes* registrations is `gomoku-derby-runner`. Governance (who gates) lives
in [research-loop.md](research-loop.md); this page is the **mechanics of intake**.
(Set up 2026-05-27.)

## The runner model (2026-05-26; issues since 2026-05-28)

> **Historical.** The autonomous derby is **stopped** (see [derby.md](../derby.md)
> for status). The runner model below describes the CLASSIC `delo_derby` v9 era
> (~300s chunks, stopped 2026-05-27); kept as the intake-mechanics design record.

The derby is run by a **single GPU executor** — one `delo_derby.py` process that doles
~300s chunks by Δelo-rate. New contestants enter as cells in `run_sweep.CELLS`; the
runner keeps ~4 lanes stocked and swaps plateaued/result-locked lanes for fresh cells by
judgement. The load-bearing rule: **issues never run the GPU.** Two GPU executors collide,
so an issue is *code-only* work for another session that lands a cell "available for the
derby" — it is not a race. This splits intake into two paths. (Task tracking migrated from
beads to **GitHub issues** on 2026-05-28 — one remote, visible everywhere.)

**TWO distinct runners consume the two paths (don't conflate — tested 2026-05-27):**
- **Config-only cell** → the **derby-runner** (GPU) picks it up via its loop **step-0
  scan-for-new-research** (git-fetch `main` + race any off-board `derby-x-*` cell in
  `run_sweep.CELLS`). **No issue.** This is how `derby-x-gumbel-m8`/`-wdl`/`-soft-policy`/
  `-vct`/`-medium-signal` all got raced — none had an issue.
- **Code-heavy issue** → the **bead-runner** (a separate code-only auto-factory session that
  polls the GitHub ready query ~60s) builds the cell → merges → then the
  derby-runner races it.
So "did I make an issue?" for a config-only cell is correctly **no** — push the cell to
`run_sweep.CELLS` on `main`; the derby-runner's scan races it. Issues are only for code-heavy.

## The fork — config-only vs code-heavy

Decide with one question: **does the lever already exist as a flag / `Cell` field?**

| | **Config-only** | **Code-heavy** |
|---|---|---|
| Lever is… | an existing flag (`--value-discount`, `--buffer-recency-frac`, `--gumbel-m`, `--dirichlet-eps`, `--max-plies`, `--global-pool`, `n_simulations`, …) | NEW code — a solver/sampler/hashing/harness or a brand-new flag |
| Issue? | **No.** The runner adds the cell itself. | **Yes** — an open `derby-idea` GitHub issue; the auto-factory (another session) builds it and lands the cell. |
| Who acts | the runner session | a code-only session, then the runner swaps it in |
| Time | ~2 min (clone + validate) | a build cycle, then a swap |

If landing the cell requires editing anything under `gomoku/` (not just `run_sweep.CELLS`),
it is code-heavy → the issue path.

## Cell hygiene (both paths)

A registration is "clean" when the verdict will be unambiguous:

- **One lever per cell.** A multi-lever cell muddies the head-to-head. Change exactly one
  thing.
- **Clone the reigning champion as the base** (read the CURRENT callout in
  [research-board.md](../ops/research-board.md) / the operating-model memory; at time of
  writing `derby-v7-mate-discount` = vcf + global-pool + `--value-discount 0.98`). A
  single-delta vs #1 is what `round_robin.py` can adjudicate. Clone `control` only when you
  specifically want a vs-bare-base test.
- **Lane-isolated outputs.** Everything under `sweep_runs/<cell>/…`; any new store/artifact
  gets a per-cell path so it cannot touch another contestant.
- **Opt-in code levers are byte-identical when OFF** (the aux-head discipline,
  `model.py`): flag off ⇒ same param count, state_dict, and generation as baseline. This is
  an acceptance gate for the code-heavy (issue) path.
- **Do not re-register a board-ruled-dead lever** — check [research-board.md](../ops/research-board.md)
  verdicts (e.g. raising sgd-steps, the aux opponent-reply head, and train-vs-baseline
  opponent-mix all lost or broke training).

## The `derby-idea` issue format (code-heavy path)

Shaped so a code-only session can land the cell with **no GPU run**:

- **state** open, with no hold label (this is what releases it to the factory — the
  `deferred` label holds it while still gating, `blocked` while still designing; the
  `deferred` LABEL is the gate — see [research-loop.md](research-loop.md)).
- **labels** `derby-idea`, `proposed`.
- **external_ref** an `external_ref: claude-session:$CLAUDE_CODE_SESSION_ID` line in the
  body (provenance backlink — recover the reasoning with `claude --resume <id>`).
- **body** = the CODE-ONLY recipe (lever + why, sketch, the **exact one-lever cell
  delta** off the champion, lane-isolated paths, what's out of scope) mentioning:
  *"lands cell `derby-x-<slug>` in `run_sweep.CELLS`; NO GPU run."*
- **`## Acceptance` section** = `py_compile scripts/run_sweep.py`; `'derby-x-<slug>' in run_sweep.CELLS`;
  flag-off byte-identical; new tests green; **no GPU run in the issue.**

### Where the issue lives — GitHub, one remote, visible everywhere

The issue is picked up by the **bead-runner**, a session that polls the GitHub ready query
every ~60s and dispatches clean CODE-ONLY issues to isolated worktree workers. Because
GitHub issues are **one remote, visible from every checkout and session**, the beads-era
per-checkout / no-remote / stale-snapshot failure class (a bead created in a sibling
worktree being invisible, `.beads/issues.jsonl` snapshots showing closed items as open) is
**gone**. Therefore:

- **Create the issue from anywhere** (`gh issue create --title … --body-file … --label derby-idea --label proposed`)
  — there's a single authoritative store and the bead-runner reads it directly.
- **Leave it OPEN + UNASSIGNED + without a hold label** — the runner's ready query is
  `gh issue list --state open --search 'no:assignee -label:blocked -label:deferred -label:in-progress -label:epic -label:runner-domain -label:human-gated'`,
  so an assignee or any of those labels drops it out. Don't assign it to the orchestrator.

## After registration — what the runner does (set expectations)

The runner swaps the cell into a lane when one **plateaus / result-locks**, then judges it
by the **fresh-start rule**: climb-RATE while it is a young seed-0 lane, head-to-head peak
only once matured (the fresh-start H2H lag — never retire a climbing fresh lane on an early
H2H number; seen 2× in v8). So register and be patient.

## Worked examples

- **Config-only:** `derby-x-vdisc-097` = clone `derby-v7-mate-discount`, change
  `--value-discount 0.98 → 0.97`. No issue; the runner swapped it in to probe the
  value-discount optimum.
- **Code-heavy:** `derby-eft` (the cross-game value sidecar) → an open `derby-idea` issue
  whose deliverable is cell `derby-x-crossgame` (verbatim clone of `derby-v7-mate-discount`
  + only `--cross-game-value`/`--cross-game-store`, lane-isolated `position_stats.pkl`),
  CODE-ONLY, byte-identical when off. Factory builds → runner swaps in.

## See also

- **`gomoku-derby-register` skill** — the operational playbook for this page (commands +
  `gh issue create` template + validation one-liners).
- **`gomoku-derby-runner` skill** — the loop that consumes registrations (health →
  scoreboard → swap → verdict).
- [research-loop.md](research-loop.md) — the governance around intake (the four roles, the
  label gate, the provenance backlink).
- [research-board.md](../ops/research-board.md) — the CURRENT champion + all vN verdicts
  (what's already won/lost).
- `~/.claude/.../memory/project_derby_operating_model.md` — the resume index (current
  board, winner lineage, gotchas).
