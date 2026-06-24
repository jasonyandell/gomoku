#!/usr/bin/env bash
# session_janitor.sh — session-start / pre-compact worktree janitor.
#
# Wires the "Janitor, not a procedure" doctrine (MEMORY.md;
# wiki/topics/worktree-hygiene.md) into the SessionStart + PreCompact hooks,
# alongside gh_prime.sh. It (1) reclaims what crashed sessions leaked
# (`reclaim_worktrees.py --apply` — liveness-aware: dead-PID worktrees and
# clean+merged manual siblings only, never live/external/unmerged trees) and
# (2) prints the one-line `repo-hygiene:` gauge so the hygiene number is
# surfaced at every session start (the gauge's home now that the cron narrator
# is retired — issue #48).
#
# ROBUST + NON-FATAL BY DESIGN: a janitor hiccup must NEVER block session start.
# No `set -e`; every line is guarded with `|| true`; the script always exit 0.
# Mirrors the up-front-guard, fast, non-hanging pattern of gh_prime.sh.

# CLAUDE_PROJECT_DIR is set by the Claude Code hook runtime; fall back to the
# script's own repo root (two dirs up from scripts/) so it also works when run
# by hand or from a test harness.
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$PROJECT_DIR" ]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd || true)"
fi
cd "$PROJECT_DIR" 2>/dev/null || true

# Pick the python interpreter: prefer the project venv, fall back to python3.
# GOMOKU_JANITOR_PY overrides both (lets tests force the fallback branch and
# inject a failing interpreter to prove the non-fatal guard).
pick_python() {
  if [ -n "${GOMOKU_JANITOR_PY:-}" ]; then
    echo "$GOMOKU_JANITOR_PY"
  elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    echo "$PROJECT_DIR/.venv/bin/python"
  else
    echo "python3"
  fi
}
PY="$(pick_python || true)"

# Ensure the worktree-aware import shim is installed in the shared venv so
# `python scripts/x.py` / pytest from ANY worktree pick up THAT worktree's
# source, not the editable-mapped main checkout (wiki/topics/worktree-hygiene.md
# § editable-install gotcha). Idempotent + non-fatal; absent file is fine.
"$PY" "$PROJECT_DIR/scripts/worktree_sitecustomize.py" >/dev/null 2>&1 || true

# Reclaim leaked/stale worktrees. --apply is safe by design (liveness-aware).
# Errors are swallowed so a janitor failure never blocks the session.
"$PY" "$PROJECT_DIR/scripts/reclaim_worktrees.py" --apply >/dev/null 2>&1 || true

# Surface the hygiene gauge (one line, `repo-hygiene: ...`). If the gauge
# itself fails, emit a sentinel line so the regex still has something to match
# and the human still sees that the janitor ran.
"$PY" "$PROJECT_DIR/scripts/reclaim_worktrees.py" --gauge 2>/dev/null \
  || echo "repo-hygiene: gauge unavailable (janitor non-fatal — session continues)"

exit 0
