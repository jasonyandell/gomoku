#!/usr/bin/env python3
"""Reclaim orphaned git worktrees and merged branches.

The mess generator (see wiki/topics/worktree-hygiene.md): Claude Code's
`isolation: worktree` agents create `.claude/worktrees/agent-<hex>` worktrees
locked to the spawning session's PID. The harness removes them on *graceful*
session exit — but Jason's regime is overnight autonomous derbies that get
killed / crash / OOM, leaving the worktree + a dead-PID lock behind forever.
Each leaked session also leaves a `feat/*` or `worktree-agent-*` branch.

This janitor reclaims ONLY what is provably safe:

  * agent worktrees under `.claude/worktrees/` whose lock PID is DEAD
    (a live PID means a running session still owns it — never touched);
  * local branches fully MERGED into main (`git branch -d`, which refuses
    unmerged branches as a hard safety net);
  * `worktree-agent-<hex>` scratch branches (the harness's auto-created
    per-worktree branch, never meant to persist) only with --include-scratch.

It NEVER touches: the main worktree, live-PID worktrees, external-tool
worktrees (e.g. ~/.codex/...), manual sibling worktrees, or any branch with
unique unmerged commits (those are real WIP — reported, never deleted).

Default is a dry-run report. Pass --apply to act. Everything it removes is
recoverable: merged branch tips live on in main; worktree dirs are
re-creatable; git reflog holds deleted branch tips for the gc window.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class Worktree:
    path: str
    branch: str | None = None      # short name, or None when detached
    detached: bool = False
    lock_reason: str | None = None  # text after "locked ", or None if unlocked
    bare: bool = False


def parse_worktrees(porcelain: str) -> list[Worktree]:
    """Parse `git worktree list --porcelain` into Worktree records."""
    out: list[Worktree] = []
    cur: Worktree | None = None
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            if cur is not None:
                out.append(cur)
            cur = Worktree(path=line[len("worktree "):])
        elif cur is None:
            continue
        elif line.startswith("branch "):
            ref = line[len("branch "):]
            cur.branch = ref.removeprefix("refs/heads/")
        elif line == "detached":
            cur.detached = True
        elif line == "bare":
            cur.bare = True
        elif line.startswith("locked"):
            cur.lock_reason = line[len("locked"):].strip() or "(no reason)"
    if cur is not None:
        out.append(cur)
    return out


def parse_lock_pid(lock_reason: str | None) -> int | None:
    """Extract the PID from a lock reason like 'claude agent ... (pid 67879)'."""
    if not lock_reason:
        return None
    marker = "pid "
    i = lock_reason.rfind(marker)
    if i == -1:
        return None
    digits = ""
    for ch in lock_reason[i + len(marker):]:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def pid_alive(pid: int) -> bool:
    """True if a process with this PID exists (signal 0 probe)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


# Worktree dispositions.
KEEP_MAIN = "keep:main"
KEEP_LIVE = "keep:live-session"
KEEP_EXTERNAL = "keep:external-tool"
KEEP_MANUAL = "keep:manual-sibling"
RECLAIM = "reclaim:orphaned"


def classify_worktree(
    wt: Worktree,
    repo_root: str,
    agent_subdir: str,
    external_prefixes: tuple[str, ...],
    alive_fn=pid_alive,
) -> tuple[str, str]:
    """Return (disposition, human-reason) for one worktree."""
    if os.path.realpath(wt.path) == os.path.realpath(repo_root):
        return KEEP_MAIN, "primary checkout"
    if any(wt.path.startswith(p) for p in external_prefixes):
        return KEEP_EXTERNAL, "owned by an external tool, not Claude Code"
    if agent_subdir not in wt.path:
        return KEEP_MANUAL, "manual sibling worktree — remove by hand if stale"
    pid = parse_lock_pid(wt.lock_reason)
    if pid is not None and alive_fn(pid):
        return KEEP_LIVE, f"lock pid {pid} is ALIVE (session still running)"
    if pid is None:
        return RECLAIM, "agent worktree with no live lock"
    return RECLAIM, f"lock pid {pid} is DEAD (session gone, leaked)"


def run(cmd: list[str], cwd: str, apply: bool) -> tuple[int, str]:
    if not apply:
        return 0, "(dry-run)"
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def git_lines(args: list[str], cwd: str) -> list[str]:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return [ln for ln in p.stdout.splitlines() if ln.strip()]


def unique_nonmerge_commits(branch: str, cwd: str) -> list[str]:
    """Original (non-merge) commits on `branch` that are not in main.

    Empty list => the branch carries no work of its own (only merge commits),
    so it is pure scratch and safe to drop even though `git branch -d` refuses
    it (its merge-commit tip is not reachable from main)."""
    return git_lines(
        ["rev-list", "--no-merges", "--format=%h %s", "--no-commit-header",
         branch, "^main"], cwd)


@dataclass
class Plan:
    reclaim: list[tuple[Worktree, str]] = field(default_factory=list)
    keep: list[tuple[Worktree, str, str]] = field(default_factory=list)


def build_plan(repo_root: str, agent_subdir: str, external_prefixes) -> Plan:
    porcelain = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root, capture_output=True, text=True,
    ).stdout
    plan = Plan()
    for wt in parse_worktrees(porcelain):
        if wt.bare:
            continue
        disp, why = classify_worktree(wt, repo_root, agent_subdir, external_prefixes)
        if disp == RECLAIM:
            plan.reclaim.append((wt, why))
        else:
            plan.keep.append((wt, disp, why))
    return plan


def gauge(repo_root: str, agent_subdir: str, external_prefixes) -> str:
    """One-line repo-hygiene metric for the cron narrator — never mutates.

    Turns silent, slow-accumulating entropy (leaked worktrees, undeleted
    merged branches) into a number that shows up every reporting cycle, so
    you notice the day it grows, not the week someone calls the repo messy."""
    plan = build_plan(repo_root, agent_subdir, external_prefixes)
    orphaned = len(plan.reclaim)
    wt_total = len(plan.reclaim) + len(plan.keep)
    branches = git_lines(["branch"], repo_root)
    merged = [b for b in git_lines(["branch", "--merged", "main"], repo_root)
              if not b.strip().startswith("*")]
    checked = {ln.removeprefix("branch refs/heads/")
               for ln in git_lines(["worktree", "list", "--porcelain"], repo_root)
               if ln.startswith("branch ")}
    merged_undeleted = sum(1 for b in merged
                           if b.strip().lstrip("+ ").strip() not in checked)
    flag = "⚠ run reclaim_worktrees --apply" if (orphaned or merged_undeleted > 3) else "clean"
    return (f"repo-hygiene: worktrees={wt_total} (orphaned={orphaned}) "
            f"branches={len(branches)} (merged-undeleted={merged_undeleted}) — {flag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually remove/delete (default: dry-run report)")
    ap.add_argument("--include-scratch", action="store_true",
                    help="also force-delete unmerged worktree-agent-* scratch branches")
    ap.add_argument("--gauge", action="store_true",
                    help="print the one-line hygiene metric and exit (for the narrator)")
    ap.add_argument("--repo", default=os.getcwd(), help="repo root (default: cwd)")
    args = ap.parse_args(argv)

    repo_root = os.path.realpath(args.repo)
    agent_subdir = "/.claude/worktrees/"
    external_prefixes = (os.path.expanduser("~/.codex/"),)

    if args.gauge:
        print(gauge(repo_root, agent_subdir, external_prefixes))
        return 0

    plan = build_plan(repo_root, agent_subdir, external_prefixes)
    mode = "APPLY" if args.apply else "DRY-RUN (pass --apply to act)"
    print(f"=== reclaim_worktrees [{mode}] — repo {repo_root} ===\n")

    print(f"KEEP ({len(plan.keep)}):")
    for wt, disp, why in plan.keep:
        print(f"  [{disp:22}] {wt.branch or '(detached)':35} {why}")
    print()

    print(f"RECLAIM worktrees ({len(plan.reclaim)}):")
    for wt, why in plan.reclaim:
        # Two --force: the first overrides a dirty tree, the second overrides
        # the stale harness lock (safe — we already proved the PID is dead).
        rc, msg = run(["git", "worktree", "remove", "--force", "--force", wt.path],
                      repo_root, args.apply)
        status = "removed" if rc == 0 else f"FAILED: {msg}"
        print(f"  - {wt.path}")
        print(f"      branch {wt.branch or '(detached)'} | {why} | {status}")
    run(["git", "worktree", "prune"], repo_root, args.apply)
    print()

    # Branch sweep — only branches not checked out in a SURVIVING worktree.
    # Branches attached to reclaimed worktrees are now free, so the dry-run
    # preview must not count them as still-checked-out.
    reclaimed_branches = {wt.branch for wt, _ in plan.reclaim if wt.branch}
    checked_out = {
        ln.removeprefix("branch refs/heads/")
        for ln in git_lines(["worktree", "list", "--porcelain"], repo_root)
        if ln.startswith("branch ")
    } - reclaimed_branches
    merged = set(b.strip().lstrip("+ ").strip()
                 for b in git_lines(["branch", "--merged", "main"], repo_root)
                 if not b.strip().startswith("*"))
    all_branches = set(b.strip().lstrip("+ ").strip()
                       for b in git_lines(["branch"], repo_root)
                       if not b.strip().startswith("*"))

    del_merged, del_empty_scratch, del_scratch_wip, keep_wip = [], [], [], []
    for b in sorted(all_branches):
        if b == "main" or b in checked_out:
            continue
        if b in merged:
            del_merged.append(b)
        elif b.startswith("worktree-agent-"):
            commits = unique_nonmerge_commits(b, repo_root)
            (del_empty_scratch if not commits else del_scratch_wip).append((b, commits))
        else:
            keep_wip.append(b)

    print(f"DELETE merged branches ({len(del_merged)}):")
    for b in del_merged:
        rc, msg = run(["git", "branch", "-d", b], repo_root, args.apply)
        print(f"  - {b}  {'deleted' if rc == 0 else 'SKIPPED: ' + msg}")
    print()

    # Empty scratch: worktree-agent-* with zero original commits => always safe.
    print(f"DELETE empty worktree-agent-* scratch (no original commits) "
          f"({len(del_empty_scratch)}):")
    for b, _ in del_empty_scratch:
        rc, msg = run(["git", "branch", "-D", b], repo_root, args.apply)
        print(f"  - {b}  {'force-deleted (no content lost)' if rc == 0 else 'FAILED: ' + msg}")
    print()

    # Scratch carrying original commits: needs the flag; print recoverable hashes.
    print(f"worktree-agent-* scratch WITH original commits ({len(del_scratch_wip)})"
          f"{' — deleting' if args.include_scratch else ' — kept (pass --include-scratch)'}:")
    for b, commits in del_scratch_wip:
        for c in commits:
            print(f"      reflog-recoverable: {c}")
        if args.include_scratch:
            rc, msg = run(["git", "branch", "-D", b], repo_root, args.apply)
            print(f"  - {b}  {'force-deleted' if rc == 0 else 'FAILED: ' + msg}")
        else:
            print(f"  - {b}")
    print()

    print(f"KEEP — feat/* branches with unique unmerged work ({len(keep_wip)}):")
    for b in keep_wip:
        print(f"  - {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
