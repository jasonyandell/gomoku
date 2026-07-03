# Lab event log — record what happened + what we observed; view it separately

Two layers, deliberately separate (the AGENTS.md model: raw evidence stays
stable, synthesis is built on top):

- **Record (write side):** `scripts/lab_log.py` appends one JSON line per
  **event** (something happened) or **observation** (something we noticed/
  measured) to the tracked log **`wiki/ops/events.jsonl`**. Git is the
  durability — it lives with `experiment-ledger.md` / `perf-log.md` and is
  reviewable in diffs. **Log only meaningful entries**: every append is a
  commit, so this is not a debug firehose.
- **View (report side):** `scripts/lab_log.py view` is a filterable log viewer
  over that file — the "dashboard" is just this read side. It never mutates.

## Usage

```bash
# record
python scripts/lab_log.py event       --scope hygiene  "reclaimed 21 orphaned worktrees -> 15/17"
python scripts/lab_log.py observation --scope derby-v4 "vcf wins head-to-head" --data '{"ref":"@9735444"}'
#   --data may carry structured payload; data.ref (a commit/run id) is shown inline.
#   --at <ISO> backfills a known past event.

# view (the dashboard)
python scripts/lab_log.py view --since 2d
python scripts/lab_log.py view --kind observation --scope derby-v4 --tail 20
python scripts/lab_log.py view --session <id>      # everything one session logged
```

## Entry schema (`wiki/ops/events.jsonl`)

```json
{"ts":"<ISO8601>","kind":"event|observation","scope":"<lane/area>","session":"<$CLAUDE_CODE_SESSION_ID>","summary":"<one line>","data":{}}
```

## How it fits

- Append a line as part of the **receipt-filing** step (alongside the
  experiment-ledger), not mid-flight. The viewer then answers "what happened
  and what did we see?" without trawling chat/logs.
- The worktree→session registry ([branch-and-worktree-workflow.md](branch-and-worktree-workflow.md))
  is a sibling event stream; reclaim runs, derby verdicts, and promote/reject
  decisions are natural entries here.
- Next steps (not yet wired): instrument `worktree_session.py` to auto-emit its
  events, and add a `lab_log.py event` line to the skill's receipt checklist so
  logging is structural, not a remembered procedure ([[feedback-janitor-not-procedure]]).
  (The old `reclaim_worktrees.py` auto-emit idea is **moot** — the reclaimer is
  **retired** as of 2026-07-01; worktree cleanup is manual now.)
