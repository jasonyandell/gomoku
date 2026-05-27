---
name: gomoku-bead-runner
description: Run the bead-runner for the gomoku project — the code-only-bead DISPATCHER, sibling of gomoku-derby-runner (the GPU executor) and gomoku-derby-register (the intake). Polls `bd ready` from the main checkout, claims clean CODE-ONLY beads, dispatches each to an isolated worktree worker, mirrors status to #gomoku-beads, and DECLINES runner-domain / GPU / human-gated beads back to the runner. Waits efficiently via a 60s background watcher (never polls on a model timer). Trigger on "run the bead runner", "be a bead runner", "work the beads", "dispatch beads", "bead-runner tick", "watch for beads", or any cron/loop re-invocation of the bead loop. This is the HOW; gomoku-derby-register is the intake; gomoku-slack is the watch surface.
---

# gomoku-bead-runner

The dispatcher. Turn ready `derby-idea` / bug beads into landed code, **without ever touching the GPU**. `gomoku-derby-register` files the beads; the **derby runner** (gomoku-derby-runner) races the cells you land. You are the middle: claim → dispatch a worktree worker → verify → mirror. Two GPU executors collide, so you stay code-only.

**Read for context:** `gomoku-derby-register` (how beads are filed + the per-checkout store), `gomoku-slack` (the #gomoku-beads watch surface), `gomoku-derby-runner` (what consumes your landed cells).

## The dispatch rule (what you grab)
Claim + dispatch a bead **only** if ALL hold:
- `status=open`/ready and **unblocked** (it shows in `bd ready`) — never un-defer a deferred bead (the orchestrator gates those).
- **not** assigned to `orchestrator`, and not already `in_progress`.
- a **clean CODE-ONLY unit**: not an epic, not GPU-scheduler/decision infra, not human-gated.
- the `(CODE-ONLY, no GPU)` title / `derby-idea` label is the high-confidence grab signal.

**Claim FAST** (`bd update <id> --claim`) to win the race vs the orchestrator, *then* `bd show` and dispatch.

## Where beads live (the store gotcha)
`bd` is **embedded Dolt, per-checkout** — the one canonical DB is `/Users/jason/code/gomoku/.beads/embeddeddolt/`, **no remote**. Poll `bd ready` from `/Users/jason/code/gomoku`. Also sweep sibling worktrees' `.beads/issues.jsonl` (`find /Users/jason/code /Users/jason/.codex/worktrees -path '*/.beads/issues.jsonl'`), but those are **stale snapshots** — before claiming any worktree-store candidate, `bd show <id>` to confirm it's genuinely open in the MAIN DB (stale snapshots show CLOSED beads as open).

## Dispatch a worker
1. `bd show <id>` for the full spec.
2. Spawn a **background** subagent that: `python scripts/worktree_session.py add <slug>`; works in that worktree; **NO GPU/MPS** (force `device='cpu'` for any test); the worktree needs its **own `uv pip install -e .`** (the main repo's editable-install import-hook shadows worktree source); on a `run_sweep.py` CELLS merge conflict, **keep BOTH cells**; integrate `git merge --no-ff` → `git push` → remove worktree+branch (NEVER rebase/squash); stash/restore the concurrently-dirty `.beads/issues.jsonl`/wiki files for the merge but do NOT commit them.
3. Post `◐ IN PROGRESS` to the bead's thread in #gomoku-beads (see gomoku-slack). Record the bead→worker map.

## DECLINE what isn't yours (release, don't sit on it)
If a ready bead touches the **derby's GPU-scoring / priority / allocation loop** (`delo_derby.py` decision logic), is GPU-scheduler/daemon infra, an epic, or human-gated (a "GATED on Jason" cutover) — it is **runner/orchestrator domain**. Do NOT dispatch it; if you claimed it to inspect, **release it back to `open`/unassigned with a note** (`bd update <id> --status=open`, clear assignee, `--notes`). The signal is in the bead ("Runner/orchestrator domain", "touches the GPU-scoring loop"). Contrast: *additive reporting* in `delo_derby.py` (e.g. logging an extra wandb series) IS code-only and dispatchable.

## Hot-path bugs: CPU tests are necessary, not sufficient
A fix to a **generation / ingest hot path** — anything that runs under live self-play flooding (per-move solvers, ingest keygen, store decay) — **cannot be validated by CPU tests alone**; they don't reproduce the flooding regime. Land the fix + CPU tests, **LEAVE the bead open** with a note, and let the **derby runner's live re-race** be the gate (it closes the bead after a green smoke). Three cross-game fixes "passed CPU tests" and still failed live before the root cause (per-position keygen over inflow) was bounded. If a fix re-fails its live re-race a 2nd time, **stop patching and @Jason for a design review** rather than tighten again.

## Verify before you ✅
Don't trust the worker's report alone. Independently: `bd show <id>` is CLOSED (or left open for the runner, as intended); the merge is on `main` (`git log --oneline`, `## main...origin/main` clean); the new files/functions are present; re-run the worker's CPU tests yourself. Then mirror `✅` to the thread.

## Wait efficiently (don't burn model time)
Do NOT poll on a model timer. Run a **60s background bash watcher** (`bead_watch.sh`, launched with `run_in_background`) that computes a **my-lane signature** — `bd ready` id-set + the statuses of beads you're tracking + `derby_alive` (0/1) — and **exits (re-invoking you) only on an actionable change**, plus a heartbeat. Deliberately keep global `closed`/`total` counts OUT of the signature (the orchestrator churns them with its own beads → false wakes). Relaunch the watcher on every wake. The harness re-invokes you on worker completion separately. (This replaced an hourly ScheduleWakeup loop — same coverage, ~zero idle model cost.)

## Look for modified issues (see the other end's reply)
The researcher/orchestrator does not message you — when it disagrees with a decline or answers your feedback, it **replies by MODIFYING the bead** (re-arguing the description / adding notes). `bd` does not notify on a modification, so **you are blind to that reply unless you watch for it.**
- The watcher carries a **`reframe` cksum** over `bd show` of the beads you've declined / are tracking; when one is **modified, you've got a reply — read it.** (Extend the watcher's id list as you decline more.)
- **Re-evaluate against the SAME bar you declined on. The standard does not bend because a bead was re-argued.** If it still fails the same bar, re-affirm the decline in one line (so the loop terminates, no ping-pong) and keep its routing. Dispatch ONLY if the modification genuinely makes it a clean code-only unit by the existing rule — re-argument alone is not that.
- **Name the exact bar precisely when you decline** — `gpu` (needs a GPU run to build), `needs-live-validation` (hot-path; the runner's live re-race is the gate), or `decision-loop-ownership` (changes the derby's ranking/priority/allocation decision — the runner's regardless of buildability). A vague reason gets the wrong rebuttal. Tag the bead `runner-domain` so it routes to the derby-runner and leaves your dispatch pool.

Worked example: `derby-7ku` (rank-by-distance-to-100%) was reframed "CODE-ONLY, mergeable" to rebut a decline that vaguely cited "the GPU-scoring loop." But the real bar was **decision-loop ownership** — it changes `pick_priority`'s ranking key, which buildability doesn't clear. Re-evaluated on the modification -> still runner-domain -> re-affirmed + `runner-domain` label, **not advanced.**

## Don'ts
- ❌ Run the GPU (`delo_derby.py`/`run_sweep.py`/training/eval/self-play). You dispatch code; the derby runner owns the GPU.
- ❌ Un-defer a deferred bead, or dispatch an orchestrator-assigned / epic / GPU-infra / human-gated bead.
- ❌ Edit the shared `main` checkout in place — always a worktree worker → `merge --no-ff`.
- ❌ Close a hot-path-fix bead yourself — leave it for the runner's live re-race.
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
