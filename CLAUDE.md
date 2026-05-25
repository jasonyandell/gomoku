# CLAUDE.md

AlphaZero for 9×9 free-style gomoku on Apple Silicon (PyTorch + MPS), with
W&B from day one. This file is the Claude Code entry map. Claude Code auto-loads
**only `CLAUDE.md`** (not `AGENTS.md`), so the line below imports the shared
agent orientation; the rest adds Claude-Code-specific operating rules.

@AGENTS.md

## Read first (in order)

1. **[wiki/index.md](wiki/index.md)** — the maintained synthesis layer; pick the
   doorway matching your task. The wiki is the **source of truth**; this repo
   compounds what we learn instead of rediscovering it each session.
2. **[README.md](README.md)** — setup and the user-facing command surface.

## Commands

```bash
source .venv/bin/activate          # uv venv; uv pip install -e ".[dev]"
pytest                              # tests/ — run before claiming a change works
gomoku-train --help                # training loop (checkpoints/latest.pt embeds buffer for resume)
gomoku-play --checkpoint checkpoints/latest.pt
gomoku-web                         # FastAPI UI around a checkpoint
```

Native hot-path extensions can be disabled for A/B: `GOMOKU_DISABLE_NATIVE_MCTS=1`,
`GOMOKU_DISABLE_NATIVE_STATE_OPS=1`. Prefer MPS over CPU paths on this machine.

## Conventions that override default behavior

- **One worktree per unit of work — never edit the shared `main` checkout.**
  The canonical lifecycle (worktree off `main` → `feat/<slug>` → `git merge
  --no-ff` → remove worktree + branch) is in
  [wiki/topics/branch-and-worktree-workflow.md](wiki/topics/branch-and-worktree-workflow.md).
  This is load-bearing: the overnight derby, the user's IDE, and other agent
  sessions routinely share the `main` checkout concurrently. Working there
  entangles diffs and blocks clean merges. **Never rebase, fast-forward, or squash.**
  Start worktrees with `python scripts/worktree_session.py add <slug>` — it
  records the owning Claude session so the branch's logs are findable later via
  `claude --resume <id>` (`worktree_session.py log` survives teardown).
- **Run the janitor at session start:** `python scripts/reclaim_worktrees.py --apply`
  reclaims worktrees/branches leaked by crashed sessions; `--gauge` prints the
  repo-hygiene metric. Cleanup is a janitor + gauge, not a remembered
  procedure ([wiki/topics/worktree-hygiene.md](wiki/topics/worktree-hygiene.md)).
- **Don't compete with live GPU/CPU tenants.** A non-lab process on the box (or
  a running derby) is a reason to wait/escalate, not to barge in. Check before
  any GPU dispatch.
- **Wiki/evidence discipline:** the wiki is canonical synthesis; `TRAINING_WIKI.md`
  is append-only chronological evidence (don't rewrite old conclusions — add
  dated corrections). Don't clean or overwrite `checkpoints*/`, `sweep_logs/`,
  `wandb/` unless asked.

## Skills (invoke when the task matches)

- **`gomoku-train`** — start/resume/stop/tune the training loop, the web UI,
  play against a checkpoint.
- **`gomoku-research-lab`** — the autonomous research lab: two-queue scheduler
  (GPU-serial + parallel agent fan-out), receipts + Reviewer audits, time-capped
  training slices, the Δelo Derby. The north-star metric is **Δelo/Δt**.

## Persistent memory

Cross-session memory lives at
`~/.claude/projects/-Users-jason-code-gomoku/memory/` (indexed by `MEMORY.md`).
Every durable lesson also gets a wiki section — memory points back to the wiki.
