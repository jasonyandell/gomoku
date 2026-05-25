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
source .venv/bin/activate          # uv venv; uv pip install -e ".[dev]"
pytest                              # run before claiming a change works
gomoku-train --help                # latest.pt embeds the buffer for resume
gomoku-play --checkpoint checkpoints/latest.pt
gomoku-web                         # FastAPI UI around a checkpoint
```
Native ext A/B: `GOMOKU_DISABLE_NATIVE_MCTS=1`, `GOMOKU_DISABLE_NATIVE_STATE_OPS=1`.
Prefer MPS over CPU. W&B project: `gomoku` (pull exact run histories, don't guess).

## Conventions that override default behavior
- **One worktree per unit of work — never edit the shared `main` checkout.**
  Lifecycle: worktree off `main` → `feat/<slug>` → `git merge --no-ff` → remove
  worktree + branch (`wiki/topics/branch-and-worktree-workflow.md`). The derby,
  the user's IDE, and other sessions share `main` concurrently; working there
  entangles diffs and blocks clean merges. **Never rebase, fast-forward, squash.**
  Start with `python scripts/worktree_session.py add <slug>` — records the owning
  session for `claude --resume <id>` (`worktree_session.py log` survives teardown).
- **Janitor at session start:** `python scripts/reclaim_worktrees.py --apply`
  reclaims what crashed sessions leak; `--gauge` prints the repo-hygiene metric.
  Janitor + gauge, not a remembered procedure (`wiki/topics/worktree-hygiene.md`).
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
auto-loaded each session. Every durable lesson also gets a wiki section — memory
points back to the wiki (the source of truth).
