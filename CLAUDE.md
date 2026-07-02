# CLAUDE.md

AlphaZero for 9×9 free-style gomoku on Apple Silicon (PyTorch + MPS), W&B from
day one. This is the Claude Code entry map and a **native twin of `AGENTS.md`** —
the two are kept deliberately duplicated rather than linked by an `@import`
(models read imports unreliably). **Change a shared rule here? Change it in
`AGENTS.md` too.** See `wiki/topics/conventions.md` § Two native
agent-instruction files.

## Read first
1. **`wiki/index.md`** — the maintained synthesis layer and source of truth; pick
   the doorway for your task. (LLM-wiki pattern: raw evidence stays stable, the
   wiki compounds learning so each session is smarter than the last.)
2. **`TRAINING_WIKI.md`** — append-only chronological training notebook (W&B run
   IDs, checkpoint meanings, failed hypotheses, learning-dynamics
   interpretation). Read before claiming why training works/fails; add dated
   corrections, don't rewrite old conclusions.
3. **`README.md`** — setup + the user-facing command surface.

General read order: wiki → `TRAINING_WIKI.md` → W&B/logs/checkpoints → code
(dynamics are subtle; code inspection alone usually misleads).

## Repo shape
- `gomoku/`: `game.py` (9×9 state, D4 aug), `model.py` (residual policy/value +
  checkpoint format), `mcts.py` (PUCT, wave-batched eval), `self_play.py` /
  `selfplay_worker.py` (record gen), `train.py` (trainer, replay buffer, W&B,
  checkpointing), `baselines.py`/`eval.py`/`match.py` (fixed opponents + probes).
- `tests/`, `scripts/`, `web/`+`app/` (playable). `wiki/` (index, log, sources,
  topics, ops/). `checkpoints*/`, `sweep_logs/`, `sweep_runs/`, `wandb/` =
  artifacts — evidence; don't clean or overwrite unless asked.

## Commands
```bash
uv sync --extra dev                # per-worktree env (auto-run at worktree creation); uv.lock-pinned
uv run pytest                      # run before claiming a change works
uv run gomoku-train --help         # latest.pt embeds the buffer for resume
uv run gomoku-play --checkpoint checkpoints/latest.pt
uv run gomoku-web                  # FastAPI UI around a checkpoint
```
**`uv run <cmd>` — never `source .venv/bin/activate`.** Each worktree has its OWN
`.venv` (uv, editable `gomoku` → that worktree); `uv run` resolves it from cwd, so
you can never silently import the main checkout (the editable-install gotcha,
`wiki/topics/worktree-hygiene.md`). Native ext A/B:
`GOMOKU_DISABLE_NATIVE_MCTS=1`, `GOMOKU_DISABLE_NATIVE_STATE_OPS=1`.
Prefer MPS over CPU. W&B project: `gomoku` (pull exact run histories, don't guess).

## Conventions that override default behavior
- **One worktree per unit of work — never edit the shared `main` checkout.**
  Lifecycle: worktree off `main` → `feat/<slug>` → `git merge --no-ff` →
  **`git push`** → remove worktree + branch
  (`wiki/topics/branch-and-worktree-workflow.md`). Pushing `main` once merged is
  *encouraged*, not confirm-gated (it's a clean fast-forward of your own work);
  only force-pushes / shared-branch pushes still ask first. The derby,
  the user's IDE, and other sessions share `main` concurrently; working there
  entangles diffs and blocks clean merges. **Never rebase, fast-forward, squash.**
  Start with `python scripts/worktree_session.py add <slug>` — records the owning
  session for `claude --resume <id>` (`worktree_session.py log` survives teardown).
- **Worktree cleanup is MANUAL and careful.** The auto-janitor
  (`reclaim_worktrees.py`) is retired (2026-07-01): it removed a worktree a
  LIVE training run was executing from — "clean + merged" says nothing about
  live processes. Before removing any worktree, `ps aux | grep <path>` first
  (`wiki/topics/worktree-hygiene.md`).
- **Fan out to preserve context.** Context is the scarcest resource; delegate
  context-heavy/parallel work to subagents (`run_in_background: true` when async)
  — broad searches, log trawls, many-file reads, independent tasks — and keep the
  findings, not the file dumps. Pair edits with `isolation: worktree`.
  (`wiki/topics/conventions.md` § Fan out to preserve context.)
- **Don't compete with live GPU/CPU tenants.** A non-lab process / running derby
  → wait or escalate, not barge in. Check before any GPU dispatch.
- **Evidence vs synthesis.** Wiki = canonical synthesis; `TRAINING_WIKI.md`
  append-only (dated corrections). Don't clean/overwrite `checkpoints*/`,
  `sweep_logs/`, `wandb/` unless asked. File reusable answers back to the wiki.
- **ML judgment.** Fixed baselines (heuristic/lookahead) for strength, not
  sibling head-to-head (non-transitive). Short evals are noisy (small-n = hint).
  Watch `selfplay/plies_mean`: falling + concave buffer-fill = fast-attack
  collapse. Experiment notes evidence-backed (cmd/config, run ID, checkpoint,
  metrics). Preserve user work.

## Skills (invoke when the task matches)
- **`gomoku-train`** — start/resume/stop/tune the loop, the web UI, play a checkpoint.
- **`gomoku-research-lab`** — two-queue scheduler (GPU-serial + parallel agent
  fan-out), receipts + Reviewer audits, time-capped training slices, the Δelo
  Derby. North-star: **Δelo/Δt**.

## Persistent memory
`~/.claude/projects/-Users-jason-code-gomoku/memory/` (indexed by `MEMORY.md`),
auto-loaded each session. Memory holds **only** (a) local-machine facts (this
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
