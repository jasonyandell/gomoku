#!/usr/bin/env bash
# refuse_main_edits.sh — worker discipline precheck.
#
# Fail LOUDLY if the current working directory resolves to the shared main
# checkout (/Users/jason/code/gomoku). Pass when invoked from a sibling
# worktree like /Users/jason/code/gomoku-<slug>.
#
# Per derby-58y (2026-05-28): an absolute-path habit caused the first edits
# of issue #1's worker to land in shared main. The shared main checkout is
# concurrently used by the derby, the user's IDE, and other sessions —
# editing it in place entangles diffs and blocks clean `git merge --no-ff`.
#
# Usage (from any worker, as the FIRST step before edits):
#   bash scripts/refuse_main_edits.sh   # exits 1 on failure
# or sourced:
#   source scripts/refuse_main_edits.sh
#
# Env override (for tests / unusual layouts):
#   GOMOKU_MAIN_PATH=/path/to/main  bash scripts/refuse_main_edits.sh

set -u

MAIN_PATH="${GOMOKU_MAIN_PATH:-/Users/jason/code/gomoku}"

# Resolve both sides to absolute, symlink-free paths for an honest compare.
# Prefer `realpath`; fall back to a portable shell-only resolution.
resolve() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1" 2>/dev/null || printf '%s\n' "$1"
    else
        # Portable: cd in a subshell, print pwd -P.
        ( cd "$1" 2>/dev/null && pwd -P ) || printf '%s\n' "$1"
    fi
}

CUR="$(resolve "$PWD")"
MAIN="$(resolve "$MAIN_PATH")"

if [ "$CUR" = "$MAIN" ]; then
    cat >&2 <<EOF
============================================================
REFUSING TO EDIT SHARED MAIN — \$PWD is the shared checkout:
    $CUR

This is the gomoku shared main; the derby, the user's IDE, and
concurrent sessions share it. Editing here entangles diffs and
blocks \`git merge --no-ff\`.

Fix: create + cd into a worktree off main:
    python scripts/worktree_session.py add <slug>
    cd /Users/jason/code/gomoku-<slug>

Then re-run this precheck. See wiki/topics/branch-and-worktree-workflow.md.
============================================================
EOF
    exit 1
fi

# Soft-friendly success line (only if running interactively, not when sourced
# silently; keep it short so it doesn't spam logs).
printf 'refuse_main_edits: OK — working in %s\n' "$CUR"
exit 0
