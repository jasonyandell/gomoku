---
name: gomoku-derby-register
description: Register a new contestant/idea with the Δelo Derby — the INTAKE counterpart to gomoku-derby-runner (which runs the GPU loop and swaps lanes). Routes an idea down the correct path; a CONFIG-ONLY lever (an existing flag) clones a `derby-` cell + flips ONE flag (no bead — the runner swaps it in); a CODE-HEAVY lever (new solver/sampler/hashing/harness/flag) files an `open`/`derby-idea` bead whose deliverable is landing a cell for the auto-factory to build. Enforces one-lever-per-cell, clone-the-reigning-champion, lane-isolation, byte-identical-when-off. Registration NEVER runs the GPU. Trigger on "register a derby contestant", "propose this to the derby/runner", "add a derby cell", "file a derby-idea bead", "make this a derby lane", "new contestant", "propose this recipe", "land a cell".
---

# gomoku-derby-register

The intake playbook: turn an idea into a derby contestant **correctly**, so the runner can race it. This is the *front door*. `gomoku-derby-runner` is the loop that then swaps it in and judges it — do not run the GPU from here.

**Source of truth:** `wiki/topics/derby-registration.md` — the canonical synthesis of the intake model (the config-only vs code-heavy fork, cell hygiene, the bead format). This skill is its operational playbook; keep the *why* and the model there, the *how* (commands) here.

**Read for context:**
- `gomoku-derby-runner` — the loop that consumes what you register (health → scoreboard → swap → verdict).
- `~/.claude/.../memory/project_derby_operating_model.md` — the resume index: **the CURRENT board (`derby_vN`) and the reigning champion cell.** Read it to know what to clone.
- `wiki/ops/research-board.md` — the CURRENT callout + vN verdicts (which levers already won/lost — don't re-register a dead one).

## The operating model (Jason's, non-negotiable)
- **The derby owns the GPU; registration does not.** You only land a cell or file a bead. Two GPU executors collide.
- **"You can't pick wrong as long as you keep things moving."** Everything gets raced eventually. A clean one-lever cell is never a wrong proposal.
- **SUBMIT research; do NOT make RANKING decisions (Jason, 2026-05-27).** The researcher's job is to
  submit *contestants* (cells), surveys, and read-only measurement tools — and then **let the ranking
  handle it from there.** How the derby SCORES / ALLOCATES / RANKS lanes (the success metric, `pick_priority`,
  eval game-counts, what counts as "best") is the **ranking owner's** (the derby-runner's) domain, NOT the
  researcher's. If a metric looks wrong (e.g. anchored elo saturates while the real goal is unmet), submit
  the **observation + a read-only tool** (e.g. `scripts/report_100pct.py`) and let the ranking owner decide
  — do **NOT** file a bead that rewires the success metric / `pick_priority` / scoring. (Anti-pattern that
  triggered this rule: a "rank by distance-to-100% instead of anchored elo" bead — a ranking decision dressed
  as research. Closed; re-filed as a pure signal-submission.)

## THE FORK — config-only vs code-heavy

Ask one question: **does the lever already exist as a flag / `Cell` field?**

| | Path A — CONFIG-ONLY | Path B — CODE-HEAVY |
|---|---|---|
| Lever is… | an existing flag (`--value-discount`, `--buffer-recency-frac`, `--gumbel-m`, `--dirichlet-eps`, `--max-plies`, `--global-pool`, `n_simulations`, …) | NEW code — a solver/sampler/hashing/harness or a brand-new flag |
| Bead? | **NO.** The runner adds the cell itself. | **YES** — `open`/`derby-idea` bead; the auto-factory builds it, lands the cell. |
| Who acts | the runner session | another (code-only) session, then the runner swaps in |

When unsure: if landing the cell requires editing anything under `gomoku/` (not just `run_sweep.CELLS`), it's code-heavy → Path B.

## Cell hygiene (BOTH paths) — what makes a registration clean
- **ONE lever per cell.** A multi-lever cell muddies the verdict. Change exactly one thing.
- **Clone the reigning champion as the base** (currently `derby-v7-mate-discount` = vcf + global-pool + `--value-discount 0.98`; confirm in the resume index). A single-delta vs #1 is what the round-robin can adjudicate. (Clone `control` only when you specifically want a vs-bare-base test.)
- **Lane-isolated outputs.** Everything under `sweep_runs/<cell>/…`; any new store/artifact gets a per-cell path so it can't touch another contestant.
- **Opt-in code levers are byte-identical when OFF** (the aux-head discipline, `model.py:37-47`): flag off ⇒ same param count, state_dict, and generation as baseline. This is an acceptance gate for Path B.
- **Don't register a known-dead lever** (check research-board.md verdicts — e.g. raising sgd-steps, aux opponent-reply head, train-vs-baseline opponent-mix all lost).

## Path A — config-only (no bead)
```bash
cd ~/code/gomoku
# 1. clone a sibling derby- cell in scripts/run_sweep.CELLS; change EXACTLY ONE existing
#    flag (e.g. --value-discount 0.98 -> 0.97). Name it derby-x-<slug>.
# 2. validate:
python -m py_compile scripts/run_sweep.py
python -c "import sys;sys.path.insert(0,'scripts');import run_sweep;print('derby-x-<slug>' in run_sweep.CELLS)"
# 3. (optional) note it in the board JSON _doc "swap-pool candidates" list so the runner sees it.
# 4. commit + push (clean main fast-forward, NOT confirm-gated):
git add scripts/run_sweep.py && git commit -m "derby: register derby-x-<slug> (<one-lever>)" && git push
```
Then it's "available" — the runner swaps it into a freed lane by judgement.

## Path B — code-heavy (the `derby-idea` bead)
The bead must be shaped so a code-only session can land the cell with **no GPU run** (per `gomoku-derby-runner` SKILL.md):
```bash
cd ~/code/gomoku                          # MUST be the main checkout — see store gotcha below
SID="claude-session:$CLAUDE_CODE_SESSION_ID"
# write the CODE-ONLY recipe to a file; it MUST end with:
#   "lands cell derby-x-<slug> in run_sweep.CELLS; NO GPU run."
bd create \
  --title="<lever> -> land cell derby-x-<slug> (CODE-ONLY, no GPU)" \
  --type=feature --priority=2 \
  --labels="derby-idea,proposed" \
  --external-ref="$SID" \
  --description="$(cat $CLAUDE_JOB_DIR/desc.txt)" \
  --acceptance="py_compile scripts/run_sweep.py passes; 'derby-x-<slug>' in run_sweep.CELLS (one-lever clone of derby-v7-mate-discount + only <new flags>, lane-isolated paths); flag OFF = byte-identical baseline; new tests green. NO GPU run in this bead."
# status defaults to OPEN -> the auto-factory picks it up.
# flip to BLOCKED (bd update <id> --status=blocked) while still designing / cooking.
# optional: bd dep add <id> <related-id> --type=related
```
The bead description should spell out: the lever + WHY, the implementation sketch, the **exact one-lever cell delta** off the champion, lane-isolated paths, performance/disk if relevant, and what's explicitly out of scope (e.g. a GPU Phase-2).

### Where Path B beads must live — reaching the bead-runner
The bead is worked by the **bead-runner**: a separate session that polls `bd ready` every **~60s** from the **main checkout `/Users/jason/code/gomoku`**, auto-claims clean CODE-ONLY beads, and dispatches each to an isolated worktree worker (posts `◐ IN PROGRESS` → `✅` in `#gomoku-beads`; lands the cell via worktree → `git merge --no-ff` → push). So the bead only gets picked up if it lands in the store the runner polls:
- **`bd` here is embedded Dolt, per-checkout** — the DB lives at `/Users/jason/code/gomoku/.beads/embeddeddolt/`, with **no remote configured**. Each checkout is its own island.
- **Create the bead from `/Users/jason/code/gomoku`** (`cd ~/code/gomoku && bd create …`). That is the one store the bead-runner polls.
- **Never `bd create` from a sibling worktree** (`gomoku-<slug>`, `gomoku-gpu-broker`, …): those carry only a **stale `.beads/issues.jsonl` snapshot** and no live DB, and with no remote a bead created there is **invisible to the bead-runner** — this is why broker subtasks got "lost."
- **Leave it `open` + unblocked + UNASSIGNED.** The runner claims unassigned **ready** beads; an assigned bead (even assigned to the runner) drops out of `bd ready`. Don't set `assignee=orchestrator`.
- Keep `--labels="derby-idea,proposed"` and the `(CODE-ONLY, no GPU)` title — the runner's high-confidence grab signal. Pickup latency is ~60s.

## After registration — what the runner does (set expectations)
The runner swaps the cell into a lane when one **plateaus / result-locks**, then judges it by the fresh-start rule: **climb-RATE while it's a young seed-0 lane, H2H peak only once matured** (the fresh-start H2H lag — never retire a climbing fresh lane on an early H2H number). So register and be patient; a fresh lane looks underwhelming in round-robin before it ripens.

## Don'ts
- ❌ Run `delo_derby.py` / `run_sweep.py` / any GPU from registration. Front door only.
- ❌ Multi-lever cells. ❌ GPU steps inside a `derby-idea` bead.
- ❌ Over-bead a config-only lever (skip the bead — the runner just adds the cell).
- ❌ Re-register a lever the research board already ruled dead.
- ❌ **File a bead that changes the RANKING** — the success metric, `pick_priority`, scoring, or eval-weighting/allocation. That's the ranking owner's (derby-runner's) call. Submit a read-only tool + the observation and let the ranking handle it.
- ❌ `bd create` from a sibling worktree, or assign the bead — both make it invisible to the bead-runner (per-checkout store + no remote; assigned drops out of `bd ready`). Create from `/Users/jason/code/gomoku`, leave it `open`+UNASSIGNED.

## Worked examples
- **Config-only:** `derby-x-vdisc-097` = clone `derby-v7-mate-discount`, change `--value-discount 0.98 → 0.97`. No bead; runner swapped it in to probe the discount optimum.
- **Code-heavy:** `derby-eft` (cross-game value sidecar) → `open`/`derby-idea` bead; deliverable = land `derby-x-crossgame` (verbatim clone of `derby-v7-mate-discount` + only `--cross-game-value`/`--cross-game-store`, lane-isolated `position_stats.pkl`), CODE-ONLY, byte-identical when off. Factory builds → runner swaps in.
