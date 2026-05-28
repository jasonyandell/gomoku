# Registering a contestant with the Δelo Derby (the intake model)

How a new idea becomes a derby lane. This is the **synthesis / source of truth**; the
operational playbook (commands, templates) is the `gomoku-derby-register` skill, and the
loop that *consumes* registrations is `gomoku-derby-runner`. Governance (who gates) lives
in [research-loop.md](research-loop.md); this page is the **mechanics of intake**.
(Set up 2026-05-27.)

## The beads-runner model (2026-05-26)

The derby is run by a **single GPU executor** — one `delo_derby.py` process that doles
~300s chunks by Δelo-rate. New contestants enter as cells in `run_sweep.CELLS`; the
runner keeps ~4 lanes stocked and swaps plateaued/result-locked lanes for fresh cells by
judgement. The load-bearing rule: **beads never run the GPU.** Two GPU executors collide,
so a bead is *code-only* work for another session that lands a cell "available for the
derby" — it is not a race. This splits intake into two paths.

**TWO distinct runners consume the two paths (don't conflate — tested 2026-05-27):**
- **Config-only cell** → the **derby-runner** (GPU) picks it up via its loop **step-0
  scan-for-new-research** (git-fetch `main` + race any off-board `derby-x-*` cell in
  `run_sweep.CELLS`). **No bead.** This is how `derby-x-gumbel-m8`/`-wdl`/`-soft-policy`/
  `-vct`/`-medium-signal` all got raced — none had a bead.
- **Code-heavy bead** → the **bead-runner** (a separate code-only auto-factory session that
  polls `bd ready` ~60s from the main checkout) builds the cell → merges → then the
  derby-runner races it.
So "did I make a bead?" for a config-only cell is correctly **no** — push the cell to
`run_sweep.CELLS` on `main`; the derby-runner's scan races it. Beads are only for code-heavy.

## The fork — config-only vs code-heavy

Decide with one question: **does the lever already exist as a flag / `Cell` field?**

| | **Config-only** | **Code-heavy** |
|---|---|---|
| Lever is… | an existing flag (`--value-discount`, `--buffer-recency-frac`, `--gumbel-m`, `--dirichlet-eps`, `--max-plies`, `--global-pool`, `n_simulations`, …) | NEW code — a solver/sampler/hashing/harness or a brand-new flag |
| Bead? | **No.** The runner adds the cell itself. | **Yes** — an `open`/`derby-idea` bead; the auto-factory (another session) builds it and lands the cell. |
| Who acts | the runner session | a code-only session, then the runner swaps it in |
| Time | ~2 min (clone + validate) | a build cycle, then a swap |

If landing the cell requires editing anything under `gomoku/` (not just `run_sweep.CELLS`),
it is code-heavy → the bead path.

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
  an acceptance gate for the code-heavy path.
- **Do not re-register a board-ruled-dead lever** — check [research-board.md](../ops/research-board.md)
  verdicts (e.g. raising sgd-steps, the aux opponent-reply head, and train-vs-baseline
  opponent-mix all lost or broke training).

## The `derby-idea` bead format (code-heavy path)

Shaped so a code-only session can land the cell with **no GPU run**:

- **status** `open` (this is what releases it to the factory — `deferred`/`blocked` holds it
  while still designing; the STATUS is the gate, not a label — see
  [research-loop.md](research-loop.md)).
- **labels** `derby-idea,proposed`.
- **external-ref** `claude-session:$CLAUDE_CODE_SESSION_ID` (provenance backlink — recover
  the reasoning with `claude --resume <id>`).
- **description** = the CODE-ONLY recipe (lever + why, sketch, the **exact one-lever cell
  delta** off the champion, lane-isolated paths, what's out of scope) ending with:
  *"lands cell `derby-x-<slug>` in `run_sweep.CELLS`; NO GPU run."*
- **acceptance** = `py_compile scripts/run_sweep.py`; `'derby-x-<slug>' in run_sweep.CELLS`;
  flag-off byte-identical; new tests green; **no GPU run in the bead.**

### Where the bead must live — the per-checkout store gotcha

The bead is picked up by the **bead-runner**, a session that polls `bd ready` every ~60s
from the **main checkout `/Users/jason/code/gomoku`** and dispatches clean CODE-ONLY beads
to isolated worktree workers. But `bd` here is **embedded Dolt, per-checkout** — the DB
lives at `/Users/jason/code/gomoku/.beads/embeddeddolt/` with **no remote configured**, so
each checkout is its own island. Therefore:

- **Create the bead from `/Users/jason/code/gomoku`** (`cd ~/code/gomoku && bd create …`) —
  that is the one store the bead-runner polls.
- **Never `bd create` from a sibling worktree** (`gomoku-<slug>`, `gomoku-gpu-broker`, …):
  those carry only a **stale `.beads/issues.jsonl` snapshot** and no live DB, and with no
  remote the bead is **invisible to the bead-runner** (this is why broker subtasks got
  "lost").
- **Leave it `open` + unblocked + UNASSIGNED** — the runner claims unassigned *ready* beads;
  an assigned bead (even assigned to the runner) drops out of `bd ready`. Don't set
  `assignee=orchestrator`.

## After registration — what the runner does (set expectations)

The runner swaps the cell into a lane when one **plateaus / result-locks**, then judges it
by the **fresh-start rule**: climb-RATE while it is a young seed-0 lane, head-to-head peak
only once matured (the fresh-start H2H lag — never retire a climbing fresh lane on an early
H2H number; seen 2× in v8). So register and be patient.

## Worked examples

- **Config-only:** `derby-x-vdisc-097` = clone `derby-v7-mate-discount`, change
  `--value-discount 0.98 → 0.97`. No bead; the runner swapped it in to probe the
  value-discount optimum.
- **Code-heavy:** `derby-eft` (the cross-game value sidecar) → an `open`/`derby-idea` bead
  whose deliverable is cell `derby-x-crossgame` (verbatim clone of `derby-v7-mate-discount`
  + only `--cross-game-value`/`--cross-game-store`, lane-isolated `position_stats.pkl`),
  CODE-ONLY, byte-identical when off. Factory builds → runner swaps in.

## See also

- **`gomoku-derby-register` skill** — the operational playbook for this page (commands +
  `bd create` template + validation one-liners).
- **`gomoku-derby-runner` skill** — the loop that consumes registrations (health →
  scoreboard → swap → verdict).
- [research-loop.md](research-loop.md) — the governance around intake (the four roles, the
  status gate, the provenance backlink).
- [research-board.md](../ops/research-board.md) — the CURRENT champion + all vN verdicts
  (what's already won/lost).
- `~/.claude/.../memory/project_derby_operating_model.md` — the resume index (current
  board, winner lineage, gotchas).
