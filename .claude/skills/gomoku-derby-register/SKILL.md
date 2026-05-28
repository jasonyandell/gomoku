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
| Who picks it up | the **derby-runner** (GPU) — its loop **step-0 scan-for-new-research** git-fetches `main` + races any off-board `derby-x-*` cell in `run_sweep.CELLS` by judgement | the **bead-runner** (code-only auto-factory) polls `bd ready`, builds → lands the cell; THEN the derby-runner races it |

**Two distinct runners — don't confuse them (tested):** config-only **cells** are consumed by the **derby-runner's scan-for-new-research** (no bead, no bead-runner) — that is literally how `derby-x-gumbel-m8`/`-wdl`/`-soft-policy`/`-vct`/`-medium-signal` all got raced. Code-heavy **beads** are consumed by the **bead-runner** (it builds the cell, then the derby-runner races it). So "did you make a bead?" for a config-only cell = **no, correctly** — push the cell to `run_sweep.CELLS` on `main` and the derby-runner's scan picks it up.

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

## Operability gates for long-running scripts (acceptance-criteria addendum)

The principle (Jason, 2026-05-28): **perf as first-class, even above research — "more perf more research."** A script that streams + resumes + self-times is fine even if it's intrinsically long; a script that doesn't have those is a perf bug regardless of wall-time.

If a Path B bead ships a SCRIPT estimated to run **>5 min wall** (probe drivers, sweep orchestrators, multi-cell tools, batch evaluators — anything with multiple cells/epochs/chunks), the bead's `--acceptance` MUST require these three operability properties. They're tablestakes — without them, a long script is a usability bug waiting to fire:

1. **Streaming output** — incremental results land in the output file (or stdout) AS the work happens, not just at end. Per-cell, per-epoch, per-chunk. If the script dies at cell 15/16, the first 14 cells' results survive on disk. Implementation: open in append mode + `f.write(...); f.flush()` after each unit, or an unbuffered `print()` per-cell at the minimum. **No "compute everything → write at end" patterns.**
2. **`--resume` support** — re-running the script with the same `--output` path **skips cells already in the output file** (idempotent). Lets the runner recover from crash, OOM, GPU contention, or a wrong-cwd misfire. Pair with `--no-resume` for explicit re-runs. The trainer does this (`--resume latest.pt`); long scripts should too.
3. **Honest timing self-report** — script prints its OWN estimate up front (*"expect ~X min for N cells at Y sec/cell, scaling with K"*) AND prints `actual / estimate` ratio at end. When the script's estimate is off by **>2×**, that signals a perf-meta bug to investigate; the next session knows to suspect modeling error, contention, or a missed cost.

**Acceptance-criteria template addendum (for any script-shipping Path B bead):**
> `… script streams results incrementally (per-cell flush, not end-of-run); --resume <existing-output> skips already-completed cells; script header prints an estimate (X min for N cells) and end-of-run prints actual / estimate ratio; if est is off >2×, a meta-perf-bead is suggested in the script's exit message.`

**Anti-pattern poster child (2026-05-28):** `probe_100pct.py` shipped via `derby-5xs` + `derby-u8d` without any of the three. The 4-hour la6 matrix run had **no per-cell visibility** (output buffered to end). The 100g re-eval ran 2+ hours with **no observable progress and no resumability** — a crash would have lost it all. The bead acceptance criteria SHOULD have required these properties; it didn't. The next iteration of this script must add them; future probe-driver beads must include them from the start.

**The 5-min wall is the trigger** for adding the gate, NOT a hard limit on script duration. A 4-hour script with all three properties is fine. A 7-min script without them is a bug. Wall-time is a signal that the operational properties matter; the properties themselves are what protect the work.

## After registration — what the runner does (set expectations)
The runner swaps the cell into a lane when one **plateaus / result-locks**, then judges it by the fresh-start rule: **climb-RATE while it's a young seed-0 lane, H2H peak only once matured** (the fresh-start H2H lag — never retire a climbing fresh lane on an early H2H number). So register and be patient; a fresh lane looks underwhelming in round-robin before it ripens.

## Don'ts
- ❌ Run `delo_derby.py` / `run_sweep.py` / any GPU from registration. Front door only.
- ❌ Multi-lever cells. ❌ GPU steps inside a `derby-idea` bead.
- ❌ Over-bead a config-only lever (skip the bead — the runner just adds the cell).
- ❌ Re-register a lever the research board already ruled dead.
- ❌ **File a bead that changes the RANKING** — the success metric, `pick_priority`, scoring, or eval-weighting/allocation. That's the ranking owner's (derby-runner's) call. Submit a read-only tool + the observation and let the ranking handle it.
- ❌ `bd create` from a sibling worktree, or assign the bead — both make it invisible to the bead-runner (per-checkout store + no remote; assigned drops out of `bd ready`). Create from `/Users/jason/code/gomoku`, leave it `open`+UNASSIGNED.
- ❌ **Merge `feat→main` with `.beads/issues.jsonl` dirty (TESTED gotcha).** Any `bd create`/`bd update` (run from the main checkout) stages `.beads/issues.jsonl` on `main`, so a subsequent `git merge --no-ff feat/<slug>` ABORTS with *"Your local changes to the following files would be overwritten by merge: .beads/issues.jsonl."* **Always `git commit -q -m "beads: …" -- .beads/issues.jsonl` FIRST, then merge.** (It's beads' own export; committing it is the sanctioned close-protocol sync, not a content edit.)
- ❌ **Ship a Path B script-bead estimated >5 min wall without the three operability gates** (streaming output, `--resume`, honest timing self-report) in the acceptance criteria. See *Operability gates for long-running scripts* above. Anti-pattern: `derby-5xs`/`probe_100pct.py` shipped without any of them, then a 2-hour 100g re-eval became un-resumable + un-observable. Perf is first-class above research.

## Worked examples
- **Config-only:** `derby-x-vdisc-097` = clone `derby-v7-mate-discount`, change `--value-discount 0.98 → 0.97`. No bead; runner swapped it in to probe the discount optimum.
- **Code-heavy:** `derby-eft` (cross-game value sidecar) → `open`/`derby-idea` bead; deliverable = land `derby-x-crossgame` (verbatim clone of `derby-v7-mate-discount` + only `--cross-game-value`/`--cross-game-store`, lane-isolated `position_stats.pkl`), CODE-ONLY, byte-identical when off. Factory builds → runner swaps in.
- **Capacity-unlock (config-only, a tested pattern):** `derby-x-medium-signal` = clone the bigger-net cell `derby-v9-medium` (96×6) + activate an OLD lever that was *middling on its own* at the small net — the KataGo aux heads (`--record-aux --record-ownership` + `--aux-opponent-reply-weight 0.15 --aux-ownership-weight 0.15`). Hypothesis: a lever that adds representational load (extra heads) washes out at a capacity-bottlenecked small net but may **compound** where a bigger net has spare capacity. Still ONE conceptual lever vs the base, all existing flags, byte-identical-off → config-only, no bead. (Re-activating a known-middling lever at scale is fair game; re-activating a board-*dead* one is not.)
