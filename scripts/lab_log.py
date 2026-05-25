#!/usr/bin/env python3
"""The lab event log: write down what happened + what we observed, then view it.

Two layers, kept separate on purpose (the model in AGENTS.md — raw evidence
stays stable, synthesis is built on top):

  * RECORD (write side): append one JSON line per event/observation to the
    tracked log `wiki/ops/events.jsonl`. Git is the durability; it sits with
    experiment-ledger.md / perf-log.md. Log only *meaningful* entries — every
    append is a commit, so this is not a debug firehose.
  * VIEW (report side): `view` is a filterable log viewer over that file — the
    "dashboard" is just this read side. It never mutates the log.

Examples:
  python scripts/lab_log.py event       --scope worktree  "created feat/wt-session"
  python scripts/lab_log.py observation --scope derby-v4  "vcf wins head-to-head" --data '{"verdict":"vcf"}'
  python scripts/lab_log.py view --since 2d
  python scripts/lab_log.py view --kind observation --scope derby-v4 --tail 20
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys

LOG_REL = os.path.join("wiki", "ops", "events.jsonl")
KIND_GLYPH = {"event": "⬤", "observation": "◇"}
_DUR = re.compile(r"^(\d+)([smhd])$")
_UNIT = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def session_id() -> str:
    return os.environ.get("CLAUDE_CODE_SESSION_ID", "unknown")


def repo_root(cwd: str) -> str:
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd,
                       capture_output=True, text=True).stdout.strip()
    return r or cwd


def log_path(cwd: str) -> str:
    return os.path.join(repo_root(cwd), LOG_REL)


# ---- pure logic (unit-tested) -------------------------------------------

def build_entry(kind: str, scope: str, summary: str, session: str, now: str,
                data: dict | None) -> dict:
    return {
        "ts": now,
        "kind": kind,
        "scope": scope,
        "session": session,
        "summary": summary,
        "data": data or {},
    }


def parse_entry(line: str) -> dict | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    return json.loads(line)


def parse_since(spec: str) -> _dt.timedelta:
    m = _DUR.match(spec)
    if not m:
        raise ValueError(f"bad --since {spec!r}; use e.g. 30m, 12h, 2d")
    return _dt.timedelta(**{_UNIT[m.group(2)]: int(m.group(1))})


def filter_entries(entries, *, kind=None, scope=None, session=None, cutoff=None):
    out = []
    for e in entries:
        if kind and e.get("kind") != kind:
            continue
        if scope and e.get("scope") != scope:
            continue
        if session and e.get("session") != session:
            continue
        if cutoff is not None:
            ts = _dt.datetime.fromisoformat(e["ts"])
            if ts < cutoff:
                continue
        out.append(e)
    return out


def format_row(e: dict) -> str:
    when = _dt.datetime.fromisoformat(e["ts"]).strftime("%Y-%m-%d %H:%M")
    glyph = KIND_GLYPH.get(e.get("kind"), "·")
    ref = e.get("data", {}).get("ref", "")
    tail = f"  {ref}" if ref else ""
    return f"{when}  {glyph} {e.get('kind',''):<11} {e.get('scope',''):<11} {e.get('summary','')}{tail}"


# ---- I/O wrappers -------------------------------------------------------

def cmd_record(args) -> int:
    data = json.loads(args.data) if args.data else {}
    now = (args.at or _dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    entry = build_entry(args._kind, args.scope, args.summary, session_id(), now, data)
    path = log_path(os.getcwd())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"logged {args._kind} [{args.scope}] {args.summary}")
    print(f"  -> {path} (commit it with your receipt)")
    return 0


def cmd_view(args) -> int:
    path = log_path(os.getcwd())
    if not os.path.exists(path):
        print(f"(no event log yet at {path})")
        return 0
    with open(path) as f:
        entries = [e for e in (parse_entry(ln) for ln in f) if e]
    cutoff = None
    if args.since:
        cutoff = _dt.datetime.now().astimezone() - parse_since(args.since)
    rows = filter_entries(entries, kind=args.kind, scope=args.scope,
                          session=args.session, cutoff=cutoff)
    rows.sort(key=lambda e: e["ts"])
    if args.tail:
        rows = rows[-args.tail:]
    hdr = "=== lab event log"
    if args.since:
        hdr += f" (since {args.since})"
    print(hdr + f" — {len(rows)} entr{'y' if len(rows)==1 else 'ies'} ===")
    for e in rows:
        print(format_row(e))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for kind in ("event", "observation"):
        p = sub.add_parser(kind, help=f"record an {kind}")
        p.add_argument("summary")
        p.add_argument("--scope", required=True, help="lane/area, e.g. derby-v4, hygiene, worktree")
        p.add_argument("--data", help="optional JSON payload (use data.ref for a commit/run id)")
        p.add_argument("--at", help="ISO timestamp override (for backfilling known past events)")
        p.set_defaults(func=cmd_record, _kind=kind)

    pv = sub.add_parser("view", help="filterable log viewer (the dashboard)")
    pv.add_argument("--since", help="window, e.g. 30m, 12h, 2d")
    pv.add_argument("--kind", choices=["event", "observation"])
    pv.add_argument("--scope")
    pv.add_argument("--session")
    pv.add_argument("--tail", type=int)
    pv.set_defaults(func=cmd_view)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
