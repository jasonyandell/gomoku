---
name: gomoku-bead-runner
description: Run the issue-runner for the gomoku project — the code-only DISPATCHER, sibling of gomoku-derby-runner (the GPU executor) and gomoku-derby-register (the intake). Polls the GitHub ready query (`gh issue list`), claims clean CODE-ONLY issues, dispatches each to an isolated worktree worker, mirrors status to #gomoku-issues, and DECLINES runner-domain / GPU / human-gated issues back to the runner. Waits efficiently via a 60s background watcher (never polls on a model timer). Trigger on "run the issue runner", "run the bead runner", "be the runner", "work the issues", "dispatch issues", "runner tick", "watch for issues", or any cron/loop re-invocation of the dispatch loop. This is the HOW; gomoku-derby-register is the intake; gomoku-slack is the watch surface. (Skill name kept as gomoku-bead-runner for cron/trigger wiring; task tracking is GitHub issues as of 2026-05-28.)
---

# gomoku-bead-runner

The dispatcher. Turn ready `derby-idea` / bug issues into landed code, **without ever touching the GPU**. `gomoku-derby-register` files the issues; the **derby runner** (gomoku-derby-runner) races the cells you land. You are the middle: claim → dispatch a worktree worker → verify → mirror. Two GPU executors collide, so you stay code-only.

**Read for context:** `gomoku-derby-register` (how issues are filed), `gomoku-slack` (the #gomoku-issues watch surface), `gomoku-derby-runner` (what consumes your landed cells).

## The dispatch rule (what you grab)
The **ready query** is the single source of truth for what's dispatchable:
```bash
gh issue list --state open --search 'no:assignee -label:blocked -label:deferred -label:in-progress -label:epic -label:runner-domain -label:human-gated'
```
Claim + dispatch an issue **only** if ALL hold:
- it **shows in the ready query above** — open, unassigned, and none of the excluded labels. Never un-defer a `deferred` issue (the orchestrator gates those by removing the label).
- **not** already assigned (an assignee or `in-progress` label means someone owns it).
- a **clean CODE-ONLY unit**: not an `epic`, not GPU-scheduler/decision infra (`runner-domain`), not `human-gated`.
- the `(CODE-ONLY, no GPU)` title / `derby-idea` label is the high-confidence grab signal.

**Claim FAST** (`gh issue edit <N> --add-assignee @me`) to win the race vs the orchestrator, *then* `gh issue view <N>` and dispatch.

## Where issues live: GitHub — one remote, visible everywhere
Task tracking is **GitHub issues** (`gh issue list`) — one remote, visible from every checkout, every worktree, and every session. The old per-checkout / no-remote / stale-snapshot failure class (embedded Dolt DBs, `.beads/issues.jsonl` snapshots that showed CLOSED items as open) is **gone**: there's a single authoritative store and `gh` reads it directly. Poll the ready query from anywhere — no need to be in `/Users/jason/code/gomoku`, and no sibling-worktree sweep.

## Dispatch a worker
1. `gh issue view <N>` for the full spec.
2. Spawn a **background** subagent that: `python scripts/gh_worktree.py <N>` (fetches the issue title → kebab-slug, makes the worktree, drops the issue number in `<worktree>/.gh_issue` — the worker does `cat .gh_issue` to recover N); works in that worktree; **NO GPU/MPS** (force `device='cpu'` for any test); run everything with **`uv run <cmd>`** (e.g. `uv run pytest`) — `gh_worktree.py` auto-provisions the worktree's own `.venv`, and `uv run` resolves it so you never silently import the main checkout (the editable-install gotcha; **never** `source .venv/bin/activate`); on a `run_sweep.py` CELLS merge conflict, **keep BOTH cells**; integrate `git merge --no-ff` (the merge commit message **MUST include `Closes #N`** so GitHub auto-closes the issue on merge to main) → `git push` → remove worktree+branch (NEVER rebase/squash); stash/restore any concurrently-dirty wiki files for the merge but do NOT commit them.
3. Post `◐ IN PROGRESS` to the issue's thread in #gomoku-issues (see gomoku-slack). Record the issue→worker map.

## Worker discipline — never edit shared main
Every worker MUST `cd` into its worktree before any edit. The shared main checkout (`/Users/jason/code/gomoku`) is concurrently used by the derby, the user's IDE, and other sessions — editing it in place entangles diffs and blocks `git merge --no-ff`. **Absolute paths under `/Users/jason/code/gomoku/<no-slug>` are SHARED MAIN and FORBIDDEN** for edits. The only legal write targets for a worker are paths under `/Users/jason/code/gomoku-<slug>/...` (its own worktree).

Tooling — the dispatching prompt MUST remind the worker to run this as the FIRST step (right after `cd <worktree>`):
```bash
bash scripts/refuse_main_edits.sh   # exits 1 if $PWD == /Users/jason/code/gomoku
```
This is a hard tripwire — when it fires, the worker should NOT continue editing; create the worktree (`python scripts/gh_worktree.py <N>`) and `cd` in, then re-run the precheck. Source available at `scripts/refuse_main_edits.sh`.

## The worktree helper (issue → worktree → auto-close)
`scripts/gh_worktree.py <N>` is the dispatch entry point:
- It fetches the title via `gh issue view N --json number,title`, derives a kebab-slug, creates the worktree (`~/code/gomoku-<slug>`), and drops the issue number in `<worktree>/.gh_issue` (so the worker can `cat .gh_issue` and reference it).
- The worker's merge commit message **MUST include `Closes #N`** — on merge to main, GitHub then auto-closes the issue (verified working, issue #1). PREFER this auto-close over a manual `gh issue close <N>`; only close by hand when there's no merge to carry the `Closes #N` (e.g. a wontfix).

## DECLINE what isn't yours (release, don't sit on it)
If a ready issue touches the **derby's GPU-scoring / priority / allocation loop** (`delo_derby.py` decision logic), is GPU-scheduler/daemon infra, an `epic`, or `human-gated` (a "GATED on Jason" cutover) — it is **runner/orchestrator domain**. Do NOT dispatch it; if you claimed it to inspect, **release it: `gh issue edit <N> --remove-assignee @me --add-label runner-domain`** and `gh issue comment <N>` with the reason. The signal is in the issue ("Runner/orchestrator domain", "touches the GPU-scoring loop"). The `runner-domain` label also drops it out of the ready query above. Contrast: *additive reporting* in `delo_derby.py` (e.g. logging an extra wandb series) IS code-only and dispatchable.

## Hot-path bugs: CPU tests are necessary, not sufficient
A fix to a **generation / ingest hot path** — anything that runs under live self-play flooding (per-move solvers, ingest keygen, store decay) — **cannot be validated by CPU tests alone**; they don't reproduce the flooding regime. Land the fix + CPU tests, **LEAVE the issue open** with a `gh issue comment` (do NOT put `Closes #N` in the merge), and let the **derby runner's live re-race** be the gate (it closes the issue after a green smoke). Three cross-game fixes "passed CPU tests" and still failed live before the root cause (per-position keygen over inflow) was bounded. If a fix re-fails its live re-race a 2nd time, **stop patching and @Jason for a design review** rather than tighten again.

## Verify before you ✅
Don't trust the worker's report alone. Independently: `gh issue view <N>` shows CLOSED (or left open for the runner, as intended); the merge is on `main` (`git log --oneline`, `## main...origin/main` clean); the new files/functions are present; re-run the worker's CPU tests yourself. Then mirror `✅` to the thread.

## Wait efficiently (don't burn model time)
Do NOT poll on a model timer. Run a **60s background bash watcher** (`bead_watch.sh`, launched with `run_in_background`) that computes a **my-lane signature** — the **ready-query id-set** (the `gh issue list --search …` from the dispatch rule) + the state/labels of the issues you're tracking + `derby_alive` (0/1) — and **exits (re-invoking you) only on an actionable change**, plus a heartbeat. Deliberately keep the global open/closed totals OUT of the signature (the orchestrator churns them with its own issues → false wakes). Relaunch the watcher on every wake. The harness re-invokes you on worker completion separately. (This replaced an hourly ScheduleWakeup loop — same coverage, ~zero idle model cost.)

## Look for modified issues (see the other end's reply)
The researcher/orchestrator does not message you — when it disagrees with a decline or answers your feedback, it **replies by editing the issue or adding a comment** (re-arguing the body / commenting). So **watch for issue edits/comments** on the issues you've declined or are tracking.
- The watcher carries a **`reframe` cksum** over `gh issue view <N>` (body + comments via `--json body,comments`, or `--comments`) of the issues you've declined / are tracking; when one is **edited or gets a new comment, you've got a reply — read it** (`gh issue view <N> --comments`). (Extend the watcher's id list as you decline more.)
- **Re-evaluate against the SAME bar you declined on. The standard does not bend because an issue was re-argued.** If it still fails the same bar, re-affirm the decline in one line (so the loop terminates, no ping-pong) and keep its routing. Dispatch ONLY if the modification genuinely makes it a clean code-only unit by the existing rule — re-argument alone is not that.
- **Name the exact bar precisely when you decline** — `gpu` (needs a GPU run to build), `needs-live-validation` (hot-path; the runner's live re-race is the gate), or `decision-loop-ownership` (changes the derby's ranking/priority/allocation decision — the runner's regardless of buildability). A vague reason gets the wrong rebuttal. Add the `runner-domain` label so it routes to the derby-runner and leaves your dispatch pool.

Worked example: `derby-7ku` (rank-by-distance-to-100%) was reframed "CODE-ONLY, mergeable" to rebut a decline that vaguely cited "the GPU-scoring loop." But the real bar was **decision-loop ownership** — it changes `pick_priority`'s ranking key, which buildability doesn't clear. Re-evaluated on the modification -> still runner-domain -> re-affirmed + `runner-domain` label, **not advanced.**

## Don'ts
- ❌ Run the GPU (`delo_derby.py`/`run_sweep.py`/training/eval/self-play). You dispatch code; the derby runner owns the GPU.
- ❌ Un-defer a `deferred` issue, or dispatch an orchestrator-assigned / `epic` / GPU-infra / `human-gated` issue.
- ❌ Edit the shared `main` checkout in place — always a worktree worker → `merge --no-ff`.
- ❌ Close a hot-path-fix issue yourself (or put `Closes #N` in its merge) — leave it for the runner's live re-race.
- ❌ Poll on a model timer, or wake on derby lane-swaps (noise).

## Friction-smoothing log

Things that bit us before, with their fixes. **Read this on session start; append after every session.** This is the part of the skill that compounds across runs.

### 2026-05-27 (the session that established the bead-runner)
- **Beads created in a sibling worktree are invisible.** `bd` is embedded per-checkout with no remote; worktree `.beads/issues.jsonl` are stale snapshots. Broker subtasks "vanished" this way. Fix: create from `/Users/jason/code/gomoku`; sweep worktree stores but `bd show`-verify against the main DB before claiming.
- **Exact-solve / heavy work on the generation hot path starves self-play.** The `--vct-teacher` per-move VCT solve ran ~14.6s/move (×8 workers ⇒ buf=0); the cross-game sidecar's per-position keygen blew up under inflow. Lesson: bound hot-path work aggressively (VCT teacher caps 4/800 ⇒ 320ms), and the **live re-race**, not CPU tests, is the gate. Leave the bead open for the runner.
- **Not every ready P1 is yours.** `derby-7ku` (rank-by-distance-to-100% metric) modified `delo_derby.py`'s scoring DECISION loop = runner-domain → declined + released. But `derby-i5j` (log the authoritative elo to wandb) was *additive reporting* in the same file = code-only ⇒ dispatched. The line is decision-logic vs reporting/additive.
- **A fix that enables a decisive "no" is a win.** `derby-b6r` made the VCT teacher *raceable*; the runner then cleanly rejected the lever (H2H −69). The loop learned something true — that's success, not waste.
- **Workers now take ~1h** (worktree `uv pip install -e .` + contended `run_sweep.py` CELLS merges). That's normal as long as edits advance. Liveness-check via worktree file mtimes + `git -C <wt> log/status` + `ps` (not the JSONL transcript); only nudge (SendMessage) if it's edit-less AND process-less after ~1h.
- **The watcher signature must be my-lane-only.** A first cut keyed on global `closed` count woke the runner for every orchestrator bead close (derby-1xf). Dropped global counts → wakes only on `bd ready` changes, tracked-bead status, derby up/down.

### 2026-05-27 - blind to the researcher's reply (the real comms gap)
- Declined `derby-7ku` (runner-domain) - correctly. The researcher REPLIED by editing the bead to rebut, and I never saw it: `bd` doesn't notify on modification and the watcher only keyed on the ready-set + statuses. Surfaced only when Jason asked "what does beads tell you about 7ku." The decline was right and STAYED right - the gap was pure VISIBILITY, not routing or advancement.
- Fix: the watcher carries a `reframe` cksum over `bd show` of declined/tracked beads, so a modification (a reply) now wakes me. Re-evaluate against the SAME bar; the standard doesn't bend. Name the exact bar on decline so replies target the real thing (or the other end learns it's fundamentally not ours).

### 2026-05-28 — first edits hit shared main by absolute-path habit (issue #1 / derby-58y)
- The GH-migration test-run worker's FIRST edits landed in `/Users/jason/code/gomoku/...` (the shared main checkout) instead of `/Users/jason/code/gomoku-<slug>/...`, via an absolute-path habit. Caught + reverted manually, but it nearly entangled the derby's working tree. Symptom: model writes absolute paths from memory ("I know it's at `/Users/jason/code/gomoku/scripts/foo.py`") and skips `cd` into the worktree.
- Fix: added `scripts/refuse_main_edits.sh` — a precheck that exits 1 LOUDLY when `$PWD` resolves to the shared main checkout. The dispatch prompt template MUST tell the worker to run it as the FIRST step after creating the worktree. See the new "Worker discipline — never edit shared main" section above.
- Standing rule (for the prompt, not just the runner): every edit target a worker writes must be under `/Users/jason/code/gomoku-<slug>/...`, never under `/Users/jason/code/gomoku/<no-slug>`. The precheck is the tripwire; the rule is the contract.

### 2026-05-28 — flipped to GitHub issues
Task tracking migrated from beads (`bd`) to **GitHub issues**. The whole skill body is now GitHub-issue mechanics; this entry records what the flip bought and the gotchas it retired.
- **The per-checkout store pain is GONE.** beads was embedded Dolt, per-checkout, no remote — so a bead created in a sibling worktree was invisible, and `.beads/issues.jsonl` snapshots showed CLOSED items as open (the cause of the "broker subtasks vanished" and "stale-snapshot" entries above). GitHub issues are **one remote, visible everywhere** — no main-checkout-only poll, no worktree sweep, no `bd show`-verify-against-main dance. Read the ready query from anywhere.
- **The ready query encodes bd's extra states as EXCLUDED labels.** beads had first-class statuses (`deferred`/`blocked`/`in_progress`); GitHub has only open/closed. So the dispatch filter excludes them as labels: `-label:blocked -label:deferred -label:in-progress -label:epic -label:runner-domain -label:human-gated` plus `no:assignee`. `in-progress` is also implied by an assignee. Those labels already exist in the repo.
- **`bd remember` has no GitHub equivalent — redirect to the wiki + memory.** Durable lessons go to `wiki/` (source of truth) and `~/.claude/projects/-Users-jason-code-gomoku/memory/`, NOT to an issue. Issues are work units, not a knowledge store.
- **`git pull --rebase` was wrong — never rebase.** Any close-protocol guidance that said `git pull --rebase` is a bug in this repo (the merge-commits-never-rebase rule). Integrate with `git merge` / `git merge --no-ff`; auto-close issues via `Closes #N` in the merge commit rather than a manual close.
