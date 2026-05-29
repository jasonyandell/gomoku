#!/usr/bin/env bash
# gh_prime.sh — session primer for the GitHub-issues task tracker.
#
# Replaces `bd prime` (beads retired 2026-05-28, issue #3 / epic #2). Wired into
# the SessionStart + PreCompact hooks. Fast + non-hanging: bails in <1s if `gh`
# isn't installed or authenticated (there is no `timeout` binary on this box, so
# we guard up front rather than wrapping the calls).
set -u

if ! command -v gh >/dev/null 2>&1; then
  echo "# Task tracker: GitHub issues — gh CLI not found (install + 'gh auth login' to see the queue)"
  exit 0
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "# Task tracker: GitHub issues — gh not authenticated (run: gh auth login)"
  exit 0
fi

READY_Q='no:assignee -label:blocked -label:deferred -label:in-progress -label:epic -label:runner-domain -label:human-gated'

cat <<'EOF'
# GitHub Issues — task tracking (beads retired 2026-05-28; see CLAUDE.md)

Tracker: github.com/jasonyandell/gomoku/issues. Use `gh`, NOT bd/TodoWrite/markdown.
  gh issue view <N>                      # details
  gh issue edit <N> --add-assignee @me   # claim
  python scripts/gh_worktree.py <N>      # worktree for an issue (put `Closes #N` in the merge)
Session close: file follow-ups, run gates, then `git pull --no-rebase && git push` (NEVER rebase).
EOF

echo
echo "## Ready (open, unblocked, code-only):"
gh issue list --state open --search "$READY_Q" --limit 20 \
  --json number,title,labels \
  -q '.[] | "  #\(.number) [\(.labels|map(.name)|join(","))] \(.title)"' 2>/dev/null \
  || echo "  (query failed)"

echo
echo "## In progress:"
gh issue list --state open --label in-progress --limit 20 \
  --json number,title -q '.[] | "  #\(.number) \(.title)"' 2>/dev/null \
  || echo "  (none)"

echo
echo "## Epics:"
gh issue list --state open --label epic --limit 10 \
  --json number,title -q '.[] | "  #\(.number) \(.title)"' 2>/dev/null \
  || echo "  (none)"
