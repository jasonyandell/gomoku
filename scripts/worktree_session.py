#!/usr/bin/env python3
"""Record which Claude Code session owns a worktree, so its logs are findable later.

When you start a worktree (the canonical workflow,
wiki/topics/branch-and-worktree-workflow.md), the session that created it is the
one whose transcript holds the reasoning behind that branch. This records the
mapping so weeks later you can `claude --resume <session-id>` or read the
transcript `.jsonl` — even after the worktree itself has been torn down.

Two records are written:
  * <worktree>/.claude-session — a gitignored per-worktree file. `cat` it from
    inside a worktree to see who owns it and the resume command.
  * <git-common-dir>/worktree-sessions.jsonl — an append-only registry in the
    SHARED git dir. Git never tracks `.git/` contents and the file survives
    `git worktree remove`, so the mapping outlives the worktree (the whole point
    of being able to find logs *later*).

Session id comes from $CLAUDE_CODE_SESSION_ID (set by Claude Code). Transcripts
live at ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl; since the id is
globally unique we record a glob (`.../*/<id>.jsonl`) that finds it regardless
of which project dir the session ran in.

Usage:
  python scripts/worktree_session.py add <slug>   # git worktree add ~/code/gomoku-<slug> -b feat/<slug>, then record
  python scripts/worktree_session.py record         # record the CURRENT worktree (already created)
  python scripts/worktree_session.py log             # print the registry with resume commands
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys

SESSION_FILE = ".claude-session"
REGISTRY_NAME = "worktree-sessions.jsonl"


def session_id() -> str:
    return os.environ.get("CLAUDE_CODE_SESSION_ID", "unknown")


def transcript_glob(sid: str) -> str:
    """A glob that finds the session transcript regardless of project dir."""
    return os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl")


def resume_cmd(sid: str) -> str:
    return f"claude --resume {sid}"


def build_record(worktree: str, branch: str, sid: str, now: str) -> dict:
    """Pure: assemble the registry/session record. Testable without git or env."""
    return {
        "ts": now,
        "worktree": worktree,
        "branch": branch,
        "session_id": sid,
        "resume": resume_cmd(sid),
        "transcript": transcript_glob(sid),
    }


def render_session_file(rec: dict) -> str:
    """Pure: the human-readable .claude-session body."""
    return (
        f"# Claude Code session that owns this worktree (gitignored).\n"
        f"session_id={rec['session_id']}\n"
        f"branch={rec['branch']}\n"
        f"created_at={rec['ts']}\n"
        f"worktree={rec['worktree']}\n"
        f"transcript={rec['transcript']}\n"
        f"# resume / attach to read its logs:\n"
        f"resume={rec['resume']}\n"
    )


def git(args: list[str], cwd: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True).stdout.strip()


def common_git_dir(cwd: str) -> str:
    d = git(["rev-parse", "--git-common-dir"], cwd)
    return os.path.abspath(os.path.join(cwd, d))


def registry_path(cwd: str) -> str:
    return os.path.join(common_git_dir(cwd), REGISTRY_NAME)


def write_records(worktree: str, branch: str) -> dict:
    sid = session_id()
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    rec = build_record(os.path.abspath(worktree), branch, sid, now)
    # 1) gitignored per-worktree file
    with open(os.path.join(worktree, SESSION_FILE), "w") as f:
        f.write(render_session_file(rec))
    # 2) durable shared registry (survives worktree teardown)
    reg = registry_path(worktree)
    with open(reg, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if sid == "unknown":
        print("  ⚠ $CLAUDE_CODE_SESSION_ID not set — recorded session_id=unknown",
              file=sys.stderr)
    return rec


def provision_venv(worktree: str) -> None:
    """Give the worktree its OWN uv-managed `.venv` (gomoku editable -> itself).

    This is what makes `uv run` / a relative activate correct BY CONSTRUCTION
    instead of silently importing the main checkout (the editable-install gotcha,
    wiki/topics/worktree-hygiene.md). On APFS uv clones from cache: ~1-4s, ~0
    incremental disk. Non-fatal: a provisioning hiccup must not strand the
    worktree — the dev can re-run `uv sync --extra dev` by hand.
    """
    uv = shutil.which("uv")
    if uv is None:
        print("  ⚠ uv not found on PATH — skipping venv; run `uv sync --extra dev` "
              "in the worktree by hand", file=sys.stderr)
        return
    print("  provisioning per-worktree env (uv sync --extra dev) ...")
    r = subprocess.run([uv, "sync", "--extra", "dev"], cwd=worktree)
    if r.returncode != 0:
        print("  ⚠ `uv sync` failed — run it by hand in the worktree before working",
              file=sys.stderr)
    else:
        print("  env ready: `cd` in and use `uv run <cmd>` (no activation needed)")


def cmd_add(args) -> int:
    repo = os.getcwd()
    path = args.path or os.path.expanduser(f"~/code/gomoku-{args.slug}")
    branch = args.branch or f"feat/{args.slug}"
    rc = subprocess.run(["git", "worktree", "add", path, "-b", branch], cwd=repo)
    if rc.returncode != 0:
        return rc.returncode
    rec = write_records(path, branch)
    if not args.no_venv:
        provision_venv(path)
    print(f"worktree {path} [{branch}] — owned by session {rec['session_id']}")
    print(f"  resume later: {rec['resume']}")
    return 0


def cmd_record(args) -> int:
    worktree = args.worktree or git(["rev-parse", "--show-toplevel"], os.getcwd())
    branch = args.branch or git(["rev-parse", "--abbrev-ref", "HEAD"], worktree)
    rec = write_records(worktree, branch)
    print(f"recorded {worktree} [{branch}] — session {rec['session_id']}")
    print(f"  wrote {os.path.join(worktree, SESSION_FILE)} + {registry_path(worktree)}")
    print(f"  resume later: {rec['resume']}")
    return 0


def cmd_log(args) -> int:
    reg = registry_path(os.getcwd())
    if not os.path.exists(reg):
        print(f"(no registry yet at {reg})")
        return 0
    print(f"=== worktree → session registry ({reg}) ===")
    with open(reg) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            print(f"{r['ts']}  {r['branch']:24}  {r['session_id']}")
            print(f"    {r['worktree']}")
            print(f"    {r['resume']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="create a worktree and record its session")
    p_add.add_argument("slug")
    p_add.add_argument("--path", help="worktree path (default ~/code/gomoku-<slug>)")
    p_add.add_argument("--branch", help="branch name (default feat/<slug>)")
    p_add.add_argument("--no-venv", action="store_true",
                       help="skip provisioning the per-worktree uv env")
    p_add.set_defaults(func=cmd_add)

    p_rec = sub.add_parser("record", help="record the current (already-created) worktree")
    p_rec.add_argument("--worktree", help="worktree path (default: cwd's toplevel)")
    p_rec.add_argument("--branch", help="branch (default: current)")
    p_rec.set_defaults(func=cmd_record)

    p_log = sub.add_parser("log", help="print the registry with resume commands")
    p_log.set_defaults(func=cmd_log)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
