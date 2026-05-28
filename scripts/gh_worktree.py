#!/usr/bin/env python3
"""gh_worktree.py — GH-issue → worktree helper (for the migration flip).

Given a GitHub issue number N, fetch its title via `gh issue view`, derive a
kebab-slug, call `scripts/worktree_session.py add <slug>` to create the
worktree, and write the issue number to `<worktree>/.gh_issue` so the
worker can read it and put `Closes #N` in its commit.

Per derby-58y (2026-05-28): this is PAUSED pending Jason's flip from beads
to GH issues. When the flip is on, workers replace
`scripts/worktree_session.py add <slug>` with `scripts/gh_worktree.py N`.
The commit message MUST include `Closes #N`; on merge to main, GH
auto-closes the issue (verified via test-run issue #1).

Usage:
    python scripts/gh_worktree.py <issue-number>

Env:
    GH_WORKTREE_DRY_RUN=1   skip the actual `worktree_session.py add` call
                            (prints what would happen).

Smoke-test the slug derivation without the network:
    python scripts/gh_worktree.py --slug-from "Fix the foo bar baz quux"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys


def kebab_slug(title: str, max_words: int = 3) -> str:
    """Derive a kebab-case slug from a GH issue title.

    Pure function — testable without the network.
    - lowercases
    - strips non-alphanumeric (split on word boundaries)
    - keeps the first `max_words` non-trivial words
    - drops common prefix noise ('fix:', '[bug]', 'feat:', etc.)
    """
    # Strip common prefix tags like "fix:", "[bug]", "feat -".
    cleaned = re.sub(r"^\s*(\[[^\]]+\]|fix|bug|feat|chore|docs|task)\s*[:\-]\s*",
                     "", title, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9]+", cleaned.lower())
    # Drop tiny stopwords if we have plenty of words.
    stop = {"a", "an", "the", "to", "of", "for", "in", "on", "and", "or"}
    rich = [w for w in words if w not in stop]
    use = (rich if len(rich) >= 1 else words)[:max_words]
    if not use:
        return "issue"
    return "-".join(use)


def fetch_issue_title(n: int) -> str:
    """Call `gh issue view N --json number,title` and return the title."""
    r = subprocess.run(
        ["gh", "issue", "view", str(n), "--json", "number,title"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.stderr.write(f"gh issue view {n} failed:\n{r.stderr}\n")
        sys.exit(2)
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"could not parse `gh issue view` JSON: {e}\n")
        sys.exit(2)
    return data["title"]


def add_worktree(slug: str) -> str:
    """Call `scripts/worktree_session.py add <slug>` and return the worktree path.

    Path follows the same default as worktree_session.py: ~/code/gomoku-<slug>.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    wts = os.path.join(here, "worktree_session.py")
    r = subprocess.run([sys.executable, wts, "add", slug])
    if r.returncode != 0:
        sys.stderr.write(f"worktree_session.py add {slug} failed (rc={r.returncode})\n")
        sys.exit(r.returncode)
    return os.path.expanduser(f"~/code/gomoku-{slug}")


def write_issue_marker(worktree: str, n: int) -> str:
    """Drop the issue number into <worktree>/.gh_issue for the worker to read."""
    p = os.path.join(worktree, ".gh_issue")
    with open(p, "w") as f:
        f.write(f"{n}\n")
    return p


def cmd_main(n: int, dry: bool) -> int:
    title = fetch_issue_title(n)
    slug = kebab_slug(title)
    print(f"issue #{n}: {title!r} → slug {slug!r}")
    if dry:
        print(f"DRY_RUN: would create ~/code/gomoku-{slug} and write .gh_issue={n}")
        return 0
    worktree = add_worktree(slug)
    marker = write_issue_marker(worktree, n)
    print(f"  wrote {marker}")
    print(f"\nworktree: {worktree}")
    print(f"REMINDER: include `Closes #{n}` in your merge commit so GH auto-closes the issue.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("issue", nargs="?", type=int, help="GH issue number")
    ap.add_argument("--slug-from", metavar="TITLE",
                    help="(test) derive a slug from this title and print it; no network")
    args = ap.parse_args(argv)

    if args.slug_from is not None:
        print(kebab_slug(args.slug_from))
        return 0
    if args.issue is None:
        ap.error("issue number is required (or use --slug-from for a smoke test)")
    dry = os.environ.get("GH_WORKTREE_DRY_RUN") == "1"
    return cmd_main(args.issue, dry)


if __name__ == "__main__":
    sys.exit(main())
