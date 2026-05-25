#!/usr/bin/env python3
"""agent_fleet.py — inspect and cross-correlate the `claude agents` fleet.

`claude agents` (agent view) dispatches background Claude sessions. This is the
READ-ONLY gauge + inspector for that fleet: it joins the live roster, the durable
per-agent job state, and the repo's worktree↔session registry into one board, and
emits the slow-entropy gauges (agents sharing the shared `main` checkout, leaked
locked subagent worktrees) that no single narrated moment would surface.

It never stops, kills, or attaches a session — those affect *other* live agents,
so they stay manual (see the skill). This tool only observes and reports.

State surfaces (authoritative > derived):
  ~/.claude/daemon/roster.json        live workers + dispatch.source (fleet/spare/slash)
  ~/.claude/jobs/<ai-id>/state.json   DURABLE per-agent record, keyed by the 8-char ai-id
  `claude agents --json`              live process list (the alive? signal)
  <repo>/.git/worktree-sessions.jsonl session→worktree registry (survives teardown)
  `git worktree list --porcelain`     on-disk worktrees + lock state

The "ai-id" is the first 8 hex chars of the session UUID — the handle agent view
shows and `~/.claude/jobs/<ai-id>/` is keyed by.

Subcommands:
  status [--cwd PATH]   one board: ai-id | name | source | state | alive | worktree/branch
  gauge  [--json]       slow-entropy metrics (for a cron narrator / one-glance check)
  digest <ai-id>        compact summary of one session's transcript (asks/results/tail)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

HOME = Path.home()
ROSTER = HOME / ".claude" / "daemon" / "roster.json"
JOBS = HOME / ".claude" / "jobs"
PROJECTS = HOME / ".claude" / "projects"


# ---------- pure helpers (unit-tested; no I/O) ----------

def short_id(session_id: str) -> str:
    """The 8-char ai-id: first hex group of the session UUID."""
    return (session_id or "")[:8]


def index_roster(roster: dict) -> dict:
    """ai-id -> {'source', 'intent', 'name'} from roster.workers."""
    out = {}
    for short, w in (roster.get("workers") or {}).items():
        d = w.get("dispatch") or {}
        seed = d.get("seed") or {}
        out[short] = {
            "source": d.get("source", "?"),
            "intent": seed.get("intent", ""),
            "name": seed.get("name", ""),
        }
    return out


def latest_worktree_by_session(records: list[dict]) -> dict:
    """session_id -> (worktree, branch) using the LAST registry entry per session.

    The registry is append-only, so the final entry is the current ownership."""
    out = {}
    for r in records:
        sid = r.get("session_id")
        if sid:
            out[sid] = (r.get("worktree", ""), r.get("branch", ""))
    return out


def parse_git_worktrees(porcelain: str) -> list[dict]:
    """Parse `git worktree list --porcelain` into {path, branch, locked}."""
    out, cur = [], {}
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            if cur:
                out.append(cur)
            cur = {"path": line[len("worktree "):], "branch": "", "locked": False}
        elif line.startswith("branch "):
            cur["branch"] = line[len("branch "):].replace("refs/heads/", "")
        elif line.strip() == "locked" or line.startswith("locked "):
            cur["locked"] = True
    if cur:
        out.append(cur)
    return out


def build_board(fleet: list[dict], roster_idx: dict, jobs: dict,
                wt_by_session: dict, repo_root: str) -> list[dict]:
    """Union of durable job state and the live fleet, keyed by ai-id.

    `jobs` is {ai_id: state.json dict}; `fleet` is `claude agents --json`."""
    alive_sids = {f.get("sessionId") for f in fleet}
    rows: dict[str, dict] = {}

    def ensure(ai_id: str) -> dict:
        return rows.setdefault(ai_id, {
            "ai_id": ai_id, "name": "", "source": "", "state": "",
            "alive": False, "kind": "", "cwd": "", "session_id": "",
            "worktree": "", "branch": "", "on_main": False,
        })

    for ai_id, st in jobs.items():
        r = ensure(ai_id)
        sid = st.get("sessionId", "")
        r.update(name=st.get("name", "") or r["name"],
                 state=st.get("state", "") or r["state"],
                 session_id=sid or r["session_id"],
                 cwd=st.get("cwd", "") or r["cwd"])
        if sid in alive_sids:
            r["alive"] = True

    for f in fleet:
        sid = f.get("sessionId", "")
        r = ensure(short_id(sid))
        r["alive"] = True
        r["session_id"] = sid or r["session_id"]
        r["kind"] = f.get("kind", "") or r["kind"]
        r["cwd"] = f.get("cwd", "") or r["cwd"]
        if not r["name"]:
            r["name"] = f.get("name", "")
        if not r["state"]:
            r["state"] = f.get("status", "")

    for ai_id, r in rows.items():
        meta = roster_idx.get(ai_id)
        if meta:
            r["source"] = meta["source"]
            if not r["name"]:
                r["name"] = meta["name"] or meta["intent"][:38]
        wt, br = wt_by_session.get(r["session_id"], ("", ""))
        r["worktree"], r["branch"] = wt, br
        r["on_main"] = bool(wt) and os.path.realpath(wt) == os.path.realpath(repo_root)

    return sorted(rows.values(), key=lambda r: (not r["alive"], r["name"] or r["ai_id"]))


def compute_gauges(board: list[dict], git_worktrees: list[dict], repo_root: str) -> dict:
    """Slow-entropy metrics. Thresholds chosen so 'healthy' reads as 0/low."""
    real = os.path.realpath(repo_root)
    agents_on_main = sorted({r["ai_id"] for r in board if r["on_main"]})
    leaked_locked = [w for w in git_worktrees
                     if "/.claude/worktrees/agent-" in w["path"] and w["locked"]]
    return {
        "live_fleet": sum(1 for r in board if r["alive"]),
        "jobs_board_total": len(board),
        "needs_input": sum(1 for r in board if r["state"] in ("needs-input", "blocked")),
        "agents_sharing_main": len(agents_on_main),
        "agents_sharing_main_ids": agents_on_main,
        "leaked_locked_subagent_worktrees": len(leaked_locked),
        "leaked_locked_worktree_branches": [w["branch"] for w in leaked_locked],
    }


# ---------- transcript digest ----------

def _text_blocks(msg) -> list[str]:
    c = (msg or {}).get("content")
    if isinstance(c, str):
        return [c]
    if isinstance(c, list):
        return [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"]
    return []


def digest_session(jsonl_path: str, tail: int = 30) -> str:
    title, asks, results, narrative, tools = "", [], [], [], Counter()
    with open(jsonl_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "ai-title":
                title = ev.get("aiTitle", title)
            elif t == "last-prompt":
                p = (ev.get("lastPrompt") or "").strip().replace("\n", " ")
                if p and (not asks or asks[-1] != p):  # collapse repeats
                    asks.append(p[:300])
            elif t == "assistant":
                for txt in _text_blocks(ev.get("message")):
                    low = txt.lstrip().lower()
                    if low.startswith(("result:", "needs input:", "failed:")):
                        results.append(txt.strip()[:300])
                    if txt.strip():
                        narrative.append(txt.strip().replace("\n", " ")[:280])
                for b in (ev.get("message") or {}).get("content", []) or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tools[b.get("name", "?")] += 1
    out = [f"TITLE: {title or '<none>'}", "", "HUMAN ASKS:"]
    out += [f"  {i+1}. {a}" for i, a in enumerate(asks)] or ["  <none>"]
    out += ["", "RESULT / STATUS LINES:"]
    out += [f"  {r}" for r in results[-10:]] or ["  <none>"]
    out += ["", f"NARRATIVE TAIL (last {tail}):"]
    out += [f"  - {n}" for n in narrative[-tail:]] or ["  <none>"]
    out += ["", "TOOL USE: " + (", ".join(f"{n}×{c}" for n, c in tools.most_common()) or "<none>")]
    return "\n".join(out)


# ---------- I/O glue ----------

def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def load_fleet(cwd: str | None) -> list[dict]:
    cmd = ["claude", "agents", "--json"]
    if cwd:
        cmd += ["--cwd", cwd]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError):
        return []


def load_jobs() -> dict:
    out = {}
    if JOBS.is_dir():
        for d in JOBS.iterdir():
            st = d / "state.json"
            if st.is_file():
                out[d.name] = _read_json(st, {})
    return out


def load_wt_records(repo_root: str) -> list[dict]:
    try:
        common = subprocess.run(["git", "-C", repo_root, "rev-parse", "--git-common-dir"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return []
    path = Path(common)
    if not path.is_absolute():
        path = Path(repo_root) / path
    reg = path / "worktree-sessions.jsonl"
    recs = []
    if reg.is_file():
        for line in reg.read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def load_git_worktrees(repo_root: str) -> list[dict]:
    try:
        out = subprocess.run(["git", "-C", repo_root, "worktree", "list", "--porcelain"],
                             capture_output=True, text=True, timeout=10).stdout
        return parse_git_worktrees(out)
    except (OSError, subprocess.SubprocessError):
        return []


def find_transcript(ai_id: str) -> str | None:
    hits = sorted(glob.glob(str(PROJECTS / "*" / f"{ai_id}*.jsonl")), key=os.path.getmtime)
    return hits[-1] if hits else None


def _gather(repo_root: str, cwd: str | None):
    fleet = load_fleet(cwd)
    roster_idx = index_roster(_read_json(ROSTER, {}))
    jobs = load_jobs()
    wt_by_session = latest_worktree_by_session(load_wt_records(repo_root))
    git_wts = load_git_worktrees(repo_root)
    board = build_board(fleet, roster_idx, jobs, wt_by_session, repo_root)
    return board, git_wts


def cmd_status(args):
    repo = os.path.realpath(args.repo)
    board, _ = _gather(repo, args.cwd)
    print(f"{'AI-ID':9} {'ALIVE':5} {'SRC':6} {'STATE':9} {'NAME':32} WORKTREE/BRANCH")
    for r in board:
        live = "live" if r["alive"] else "dead"
        wt = "⚠ shared-main" if r["on_main"] else (r["branch"] or r["worktree"] or "-")
        print(f"{r['ai_id']:9} {live:5} {r['source'] or '-':6} {r['state'] or '-':9} "
              f"{(r['name'] or '-')[:32]:32} {wt}")
    return 0


def cmd_gauge(args):
    repo = os.path.realpath(args.repo)
    board, git_wts = _gather(repo, args.cwd)
    g = compute_gauges(board, git_wts, repo)
    if args.json:
        print(json.dumps(g, indent=2))
        return 0
    print(f"fleet: {g['live_fleet']} live / {g['jobs_board_total']} on board"
          f" | needs-input: {g['needs_input']}"
          f" | sharing main: {g['agents_sharing_main']}"
          f" | leaked locked agent-worktrees: {g['leaked_locked_subagent_worktrees']}")
    if g["agents_sharing_main"] > 1:
        print(f"  ⚠ {g['agents_sharing_main']} agents recorded against shared main: "
              f"{', '.join(g['agents_sharing_main_ids'])}")
    if g["leaked_locked_subagent_worktrees"]:
        print(f"  ⚠ leaked locked worktree branches: "
              f"{', '.join(g['leaked_locked_worktree_branches'])}")
    return 0


def cmd_digest(args):
    path = find_transcript(args.ai_id)
    if not path:
        print(f"no transcript found for ai-id {args.ai_id}", file=sys.stderr)
        return 1
    print(f"# {args.ai_id}  ({path})\n")
    print(digest_session(path, tail=args.tail))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Inspect the `claude agents` fleet (read-only).")
    p.add_argument("--repo", default=os.getcwd(), help="repo root (default: cwd)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="cross-correlated fleet board")
    s.add_argument("--cwd", help="filter to sessions started under this path")
    s.set_defaults(func=cmd_status)

    g = sub.add_parser("gauge", help="slow-entropy metrics")
    g.add_argument("--cwd")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=cmd_gauge)

    d = sub.add_parser("digest", help="summarize one session transcript")
    d.add_argument("ai_id")
    d.add_argument("--tail", type=int, default=30)
    d.set_defaults(func=cmd_digest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
