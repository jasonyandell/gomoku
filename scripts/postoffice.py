#!/usr/bin/env python3
"""postoffice.py — an append-only message bus for the `claude agents` fleet.

The piece that lets you *talk to* fleet sessions. A "cagent" (post-office session you
spawn in agent view) subscribes to a mailbox and reacts to posts. Everything is
**log-based: nothing is ever deleted or rewritten — only appended, only acked.**

Durability + missed-watch recovery, by design:
  - Each mailbox is an append-only `<mailbox>.log` (one JSON post per line). Senders
    only ever append; a crash mid-send loses at most that one line, never the history.
  - A separate `<mailbox>.cursor` records how many posts have been processed. The cagent
    advances it; it never edits the log. So progress and history are decoupled.
  - **Catch-up, not just last-event:** `pending` returns *every* post after the cursor.
    If the cagent was busy/down/missed a watch, the next scan still picks all of them up
    (at-least-once). The `wait` block has a timeout so a periodic re-scan happens even if
    an event wake is ever missed — belt and suspenders.

Low-resource event loop (run by the cagent — see `postoffice.py prompt`):
  catch up (pending → handle → ack) → `wait` (blocks asleep until a post or timeout,
  run as a background command so the harness wakes the session on exit) → repeat.

Subcommands:
  send    --to <mb> [--from] [--subject] <body>   append a post (any producer)
  pending --mailbox <mb> [--json]                 posts after the cursor (don't advance)
  ack     --mailbox <mb> [--through N | --all]    advance the cursor
  wait    --mailbox <mb> [--timeout 600]          block until a new post or timeout
  prompt  --mailbox <mb>                          print the paste-able cagent spawn prompt
  log     --mailbox <mb>                           print the whole append-only log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path.home() / ".claude" / "agent-fleet" / "postoffice"
SCRIPTS = Path(__file__).resolve().parent


def _paths(root, mailbox: str):
    d = Path(root)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{mailbox}.log", d / f"{mailbox}.cursor"


# ---------- append-only core (unit-tested) ----------

def append_post(root, mailbox: str, body: str, sender: str = "user", subject: str = "") -> dict:
    log, _ = _paths(root, mailbox)
    post = {"id": uuid.uuid4().hex[:8],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "to": mailbox, "from": sender, "subject": subject, "body": body}
    with open(log, "a", encoding="utf-8") as f:           # O_APPEND: atomic per line
        f.write(json.dumps(post) + "\n")
    return post


def read_log(root, mailbox: str) -> list[dict]:
    log, _ = _paths(root, mailbox)
    if not log.exists():
        return []
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def get_cursor(root, mailbox: str) -> int:
    _, c = _paths(root, mailbox)
    try:
        return int(c.read_text().strip())
    except (OSError, ValueError):
        return 0


def set_cursor(root, mailbox: str, n: int) -> None:
    _, c = _paths(root, mailbox)
    c.write_text(str(max(0, int(n))))


def pending(root, mailbox: str):
    """-> (cursor, [(seq, post)...], total). seq is the 1-based line number."""
    posts = read_log(root, mailbox)
    cur = get_cursor(root, mailbox)
    return cur, [(i + 1, p) for i, p in enumerate(posts) if i + 1 > cur], len(posts)


def ack(root, mailbox: str, through: int | None) -> int:
    total = len(read_log(root, mailbox))
    n = total if through is None else max(0, min(int(through), total))
    set_cursor(root, mailbox, n)
    return n


def wait(root, mailbox: str, timeout: int) -> str:
    """Block (≈0 CPU) until a new post is appended, or `timeout` seconds. Run in background."""
    log, _ = _paths(root, mailbox)
    log.touch()
    try:
        subprocess.run(["sh", "-c", f"tail -n0 -F {str(log)!r} | head -n1"],
                       timeout=timeout, capture_output=True, start_new_session=True)
        return "post"
    except subprocess.TimeoutExpired:
        return "timeout"


# ---------- cagent spawn prompt ----------

def spawn_prompt(mailbox: str) -> str:
    po = f"python {SCRIPTS}/postoffice.py"
    return f"""\
You are `cagent` — a low-resource POST OFFICE for the agent fleet, mailbox `{mailbox}`.
Your inbox is an append-only log: nothing is ever deleted, only added or acked. Run this
loop FOREVER and never stop it:

1. CATCH UP (this recovers anything that arrived while you were busy or asleep):
     {po} pending --mailbox {mailbox}
   For each pending post, do what it asks — handle it directly, or ROUTE it by posting to
   the named mailbox, or REPLY with `{po} send --to <mailbox> --from {mailbox} "<text>"`.
   Then mark them all processed (advance the cursor; never edit the log):
     {po} ack --mailbox {mailbox} --all

2. BLOCK until the next post (sleeps at ~0% CPU; 600s safety re-scan). Run this as a
   BACKGROUND command (run_in_background: true) so the harness wakes you when it returns:
     {po} wait --mailbox {mailbox} --timeout 600

3. When the wait returns (a post arrived, or the 10-min timer fired), GO BACK TO STEP 1.

Keep each handling terse to conserve context. If your context grows large, post a note to
`{mailbox}` recording the current cursor and ask to be re-spawned fresh — the log + cursor
make that seamless (a fresh cagent just catches up from the cursor). Anyone reaches you with
`{po} send --to {mailbox} "..."`."""


# ---------- CLI ----------

def cmd_send(a):
    p = append_post(a.root, a.to, a.body, sender=a.sender, subject=a.subject)
    print(f"posted {p['id']} -> {a.to}")
    return 0


def cmd_pending(a):
    cur, items, total = pending(a.root, a.mailbox)
    if a.json:
        print(json.dumps([{"seq": s, **p} for s, p in items], indent=2))
        return 0
    print(f"mailbox {a.mailbox}: cursor={cur} total={total} pending={len(items)}")
    for s, p in items:
        subj = f" [{p['subject']}]" if p.get("subject") else ""
        print(f"  #{s} {p['id']} from {p.get('from','?')}{subj}: {p.get('body','')[:160]}")
    return 0


def cmd_ack(a):
    n = ack(a.root, a.mailbox, None if a.all else a.through)
    print(f"cursor -> {n}")
    return 0


def cmd_wait(a):
    print(wait(a.root, a.mailbox, a.timeout))
    return 0


def cmd_prompt(a):
    print(spawn_prompt(a.mailbox))
    return 0


def cmd_log(a):
    for i, p in enumerate(read_log(a.root, a.mailbox), 1):
        print(f"#{i} {p.get('ts','')} {p['id']} {p.get('from','?')}->{p.get('to','?')}: {p.get('body','')[:160]}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Append-only message bus for the agents fleet.")
    p.add_argument("--root", default=str(ROOT), help=f"postoffice dir (default {ROOT})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("send"); s.add_argument("--to", required=True); s.add_argument("--from", dest="sender", default="user")
    s.add_argument("--subject", default=""); s.add_argument("body"); s.set_defaults(func=cmd_send)

    pe = sub.add_parser("pending"); pe.add_argument("--mailbox", required=True); pe.add_argument("--json", action="store_true")
    pe.set_defaults(func=cmd_pending)

    ak = sub.add_parser("ack"); ak.add_argument("--mailbox", required=True)
    ak.add_argument("--through", type=int); ak.add_argument("--all", action="store_true"); ak.set_defaults(func=cmd_ack)

    w = sub.add_parser("wait"); w.add_argument("--mailbox", required=True); w.add_argument("--timeout", type=int, default=600)
    w.set_defaults(func=cmd_wait)

    pr = sub.add_parser("prompt"); pr.add_argument("--mailbox", default="cagent"); pr.set_defaults(func=cmd_prompt)
    lg = sub.add_parser("log"); lg.add_argument("--mailbox", required=True); lg.set_defaults(func=cmd_log)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
