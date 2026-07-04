# Gomoku Agent Orientation

Standalone orientation for any coding agent in this repo. This file is a
**native twin of `CLAUDE.md`** — the two are kept deliberately duplicated rather
than linked by an `@import` (models read imports unreliably). **If you change a
shared rule here, change it in `CLAUDE.md` too**, and vice versa. See
`wiki/topics/conventions.md` § Two native agent-instruction files.

Start with the wiki index: `wiki/index.md`. This repo follows the LLM-wiki
pattern: raw evidence stays stable, the wiki is the maintained synthesis layer,
and this file is the schema for how to work. Treat the wiki as a compounding
artifact that makes each session smarter than the last — not a prettier
transcript.

After the index, read `TRAINING_WIKI.md` — the chronological training notebook
(training story, failed hypotheses, W&B run IDs, checkpoint meanings, perf
findings, current learning-dynamics interpretation). Read it before claiming why
training works or fails. It is append-oriented: don't rewrite old conclusions;
add a dated entry that explains the correction and points to the evidence.

## Wiki architecture
**Remembering / curating anything into the wiki?** `wiki/curation.md` is the
whole instruction: the routing table (where each input class lands) + the
query rule (answers synthesized from 2+ pages get filed back) + rotation and
lint. Don't improvise structure — read it first.
- `wiki/index.md` — content entry point; keep current when durable pages appear.
- `wiki/log.md` — chronological maintenance log; append on structure/synthesis changes.
- `wiki/sources/` — stable source records for external references/evidence.
- `wiki/topics/` — maintained synthesis pages too reusable to leave in chat/logs.
- `wiki/ops/` — operational ledgers: experiment-ledger, perf-log, baselines,
  best-cells, gpu-queue, and the event log `events.jsonl`.
- `TRAINING_WIKI.md` — the training lab notebook.

## Shape of the repo
- `gomoku/` — the Python training stack:
  - `game.py` 9×9 state, terminal detection, history planes, D4 augmentation.
  - `model.py` residual policy/value net + checkpoint format.
  - `mcts.py` PUCT search, batched + wave-batched evaluation.
  - `self_play.py` / `selfplay_worker.py` self-play record generation (workers are file-based).
  - `train.py` trainer, replay-buffer ingest, W&B logging, checkpointing.
  - `baselines.py`, `eval.py`, `match.py` fixed opponents + match probes.
- `tests/` smoke + correctness. `scripts/` utilities. `web/`, `app/` playable surfaces.
- `checkpoints*/`, `sweep_logs/`, `sweep_runs/`, `wandb/` — artifacts; treat as
  evidence, don't clean or overwrite unless the user asks.

## Working rules

**Read order:** wiki index → `TRAINING_WIKI.md` → W&B/logs/checkpoints → code.
Training dynamics are subtle; code inspection alone is usually misleading.
`README.md` is the setup/command surface.

**Git workflow (canonical, load-bearing)** — `wiki/topics/branch-and-worktree-workflow.md`.
Every unit of work happens in its own git worktree off `main`, lands via `git
merge --no-ff`, is **pushed once merged** (`git push` `main` — a clean
fast-forward of your own work is *encouraged*, not confirm-gated; only
force-pushes and pushes to shared/long-lived branches still ask first), and is
torn down after (`git worktree remove` + `git branch -d`)
— **you do not edit the shared `main` checkout** (the derby, the user's IDE, and
other agent sessions share it concurrently; working there entangles diffs and
blocks clean merges). Never rebase, fast-forward, or squash. Start a worktree
with `python scripts/worktree_session.py add <slug>` — it creates
`~/code/gomoku-<slug>` on `feat/<slug>` and records the owning session so its
logs are findable later via `claude --resume <id>` (`worktree_session.py log`
survives teardown). **Then actively `EnterWorktree` (path = the worktree)** —
subagents and shell calls inherit the SESSION's cwd, not the conversational
"current worktree"; a subagent spawned without absolute paths will silently act
on the main checkout (confirmed 2026-07-04). EnterWorktree pins the session so
everything inherits the worktree by default — mechanism, not vigilance.

**Worktree cleanup is MANUAL and careful.** The auto-janitor
(`reclaim_worktrees.py`) is retired (2026-07-01): it removed a worktree a LIVE
training run was executing from — "clean + merged" says nothing about live
processes. Before removing any worktree, `ps aux | grep <path>` first
(`wiki/topics/worktree-hygiene.md`).

**Fan out to preserve context:** the orchestrator's context window is the
scarcest resource. Delegate context-heavy or parallelizable work — broad
searches, log/transcript trawls, many-file reads, independent parallel tasks —
to subagents (background when async) and keep only their distilled findings, not
file dumps. Pair edits with worktree isolation. (`wiki/topics/conventions.md` §
Fan out to preserve context.)

**Don't compete with live tenants:** a non-lab process on the box (or a running
derby) means wait/escalate, not barge in. Check before any GPU dispatch.

**Evidence vs synthesis:** keep them separate. The wiki is canonical synthesis;
`TRAINING_WIKI.md` is append-only (add dated corrections, don't rewrite). Don't
overwrite raw artifacts or clear local run evidence unless asked. File reusable
answers back into the wiki so the next session doesn't re-derive them.

**ML judgment:**
- Prefer fixed external baselines (heuristic/lookahead) for strength claims;
  sibling head-to-head is non-transitive and can just measure mutual specialization.
- Treat short evals as noisy — small-n is a hint, not proof; verify with a larger
  match or cite uncertainty.
- Watch game length: falling `selfplay/plies_mean` + concave buffer-fill usually
  means fast-attack collapse; stable/growing plies beat policy-loss alone.
- Keep experiment notes evidence-backed: command/config, run ID, checkpoint path,
  key metrics, what changed in the working theory. Preserve user work (large
  local artifacts, unfinished experimental state).

## Commands
```bash
uv sync --extra dev                # per-worktree env (auto-run at worktree creation); uv.lock-pinned
uv run pytest                      # run before claiming a change works
uv run gomoku-train --help         # training loop (latest.pt embeds buffer for resume)
uv run gomoku-play --checkpoint checkpoints/latest.pt
uv run gomoku-web                  # FastAPI UI around a checkpoint
```
**`uv run <cmd>` — never `source .venv/bin/activate`.** Each worktree has its OWN
`.venv` (uv, editable `gomoku` → that worktree); `uv run` resolves it from cwd, so
you can never silently import the main checkout (the editable-install gotcha,
`wiki/topics/worktree-hygiene.md`).
Native hot-path extensions toggle off for A/B: `GOMOKU_DISABLE_NATIVE_MCTS=1`,
`GOMOKU_DISABLE_NATIVE_STATE_OPS=1`. Prefer MPS over CPU paths. W&B project:
`gomoku` — pull exact run histories rather than guessing from summaries.

## Claude Code specifics
If you are Claude Code, you additionally get the auto-loaded twin `CLAUDE.md`,
auto-memory at `~/.claude/projects/-Users-jason-code-gomoku/memory/` (indexed by
`MEMORY.md`), and project skills — `gomoku-train` (run/resume/tune the loop + web
UI, play a checkpoint) and `gomoku-research-lab` (two-queue scheduler: GPU-serial
+ parallel agent fan-out, receipts + Reviewer audits, time-capped training
slices, the Δelo Derby; north-star metric **Δelo/Δt**).

**Persistent memory — scope:** memory holds **only** (a) local-machine facts (this
Mac's hardware, paths, keychain, MCP/tool setup, thermal/contention behavior) and
(b) working-with-Jason facts (background, preferences, autonomy boundaries, how he
wants to be worked with). **Everything about the project, ML/training judgment,
lab operation, or roadmap lives in the wiki, NOT memory** — do not mirror
project/process knowledge into memory ("memories compete with the wiki", Jason
2026-06-16). See `wiki/topics/conventions.md` § What belongs in memory vs the wiki.

## GitHub Issues — task tracking

This project tracks ALL work in **GitHub issues** (`jasonyandell/gomoku`) via the `gh` CLI.
`scripts/gh_prime.sh` prints the live ready queue + this workflow on session start.
(Beads `bd` is **retired** as of 2026-05-28; `.beads/` is dormant history.)

### Quick reference

```bash
# Ready work — open, unblocked, code-only (excludes epics / in-progress / runner-domain / human-gated):
gh issue list --state open --search 'no:assignee -label:blocked -label:deferred -label:in-progress -label:epic -label:runner-domain -label:human-gated'
gh issue view <N>                      # issue details
gh issue edit <N> --add-assignee @me   # claim
gh issue close <N>                     # complete (or auto-close via `Closes #N` in the merge commit)
```

### Rules

- Use **GitHub issues** for ALL task tracking — do NOT use TodoWrite, TaskCreate, beads (`bd`), or markdown TODO lists.
- GitHub has only open/closed; bd's extra states are encoded as labels **excluded from the ready query**:
  `deferred` (awaiting Jason's gate), `blocked` (unmet dependency), `in-progress` (mirror of an assignee).
  Routing labels: `derby-idea`, `runner-domain`, `code-only`, `gpu`, `needs-live-validation`,
  `decision-loop-ownership`, `proposed`, `human-gated`.
- One worktree per issue: `python scripts/gh_worktree.py <N>` (title→slug, drops `.gh_issue`); put
  `Closes #N` in the merge commit so GitHub auto-closes on merge to `main`.
- Persistent knowledge → the wiki + `~/.claude/.../memory/` (NOT issues; the old `bd remember` is retired).

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** — `gh issue create` for anything that needs follow-up.
2. **Run quality gates** (if code changed) — tests, linters, builds.
3. **Update issue status** — close finished work (`gh issue close`, or `Closes #N` in the merge commit); update in-progress items.
4. **PUSH TO REMOTE** — this is MANDATORY (this repo integrates with `merge --no-ff`; **never rebase**):
   ```bash
   git pull --no-rebase   # merge, never rebase
   git push
   git status             # MUST show "up to date with origin"
   ```
5. **Clean up** — clear stashes, remove the worktree + branch, prune remote branches.
6. **Verify** — all changes committed AND pushed.
7. **Hand off** — provide context for next session.

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds.
- NEVER stop before pushing — that leaves work stranded locally.
- NEVER say "ready to push when you are" — YOU must push.
- If push fails, resolve and retry until it succeeds.
