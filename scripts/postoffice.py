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
    """Block (≈0 CPU) until a new post is appended, or `timeout` seconds. Run in background.

    Pending-aware (closes the race a cagent caught 2026-05-25): `tail -n0 -F` only fires on
    posts appended AFTER it starts, so a post that landed between the caller's last `pending`
    check and this `wait` would be invisible until the timeout. So if unhandled mail already
    sits in the log (`total > cursor`), return immediately — don't block on a watch that has
    already missed it. The cursor stays the source of truth; the watch is only the wake."""
    if len(read_log(root, mailbox)) > get_cursor(root, mailbox):
        return "pending"
    log, _ = _paths(root, mailbox)
    log.touch()
    try:
        subprocess.run(["sh", "-c", f"tail -n0 -F {str(log)!r} | head -n1"],
                       timeout=timeout, capture_output=True, start_new_session=True)
        return "post"
    except subprocess.TimeoutExpired:
        return "timeout"


def list_mailboxes(root) -> list[str]:
    d = Path(root)
    return sorted(p.stem for p in d.glob("*.log")) if d.exists() else []


def feed(root, n: int = 20) -> list[tuple]:
    """Recent posts across ALL mailboxes, chronological — the cross-mailbox 'what's been said'."""
    rows = []
    for mb in list_mailboxes(root):
        for p in read_log(root, mb):
            rows.append((p.get("ts", ""), mb, p))
    rows.sort(key=lambda r: r[0])
    return rows[-n:]


# ---------- self-improvement: append-only lessons + a notes runbook ----------

def _aux_paths(root, mailbox: str):
    d = Path(root)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{mailbox}.lessons.jsonl", d / f"{mailbox}.notes.md"


def append_lesson(root, mailbox: str, text: str, tags: str = "") -> dict:
    """Append-only friction ledger — the raw record of everything a cagent learned."""
    lessons, _ = _aux_paths(root, mailbox)
    rec = {"id": uuid.uuid4().hex[:8], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "tags": tags, "text": text}
    with open(lessons, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def read_lessons(root, mailbox: str) -> list[dict]:
    lessons, _ = _aux_paths(root, mailbox)
    if not lessons.exists():
        return []
    out = []
    for line in lessons.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def read_notes(root, mailbox: str) -> str:
    """The curated operating runbook a cagent reads on startup (the durable rules)."""
    _, notes = _aux_paths(root, mailbox)
    return notes.read_text(encoding="utf-8") if notes.exists() else ""


def append_note(root, mailbox: str, rule: str) -> None:
    _, notes = _aux_paths(root, mailbox)
    if not notes.exists():
        notes.write_text(f"# {mailbox} — operating notes (read on startup; append only, never delete)\n\n")
    with open(notes, "a", encoding="utf-8") as f:
        f.write(f"- ({time.strftime('%Y-%m-%d')}) {rule}\n")


# ---------- cagent spawn prompt ----------

def spawn_prompt(mailbox: str) -> str:
    po = f"python {SCRIPTS}/postoffice.py"
    return f"""\
You are `cagent` — a low-resource, SELF-IMPROVING POST OFFICE for the agent fleet, mailbox
`{mailbox}`. Your inbox AND your lessons are append-only logs: nothing is ever deleted, only
added, acked, or learned. Run this loop FOREVER and never stop it:

0. READ YOUR NOTES — the operating runbook your predecessors accumulated. Follow it:
     {po} notes --mailbox {mailbox}

1. CATCH UP (recovers anything that arrived while you were busy or asleep):
     {po} pending --mailbox {mailbox}
   For each pending post, do what it asks — handle it directly, or ROUTE it by posting to the
   named mailbox, or REPLY with `{po} send --to <mailbox> --from {mailbox} "<text>"`.
   Then mark them all processed (advance the cursor; never edit the log):
     {po} ack --mailbox {mailbox} --all

2. BLOCK until the next post (sleeps at ~0% CPU; 600s safety re-scan). Run this as a
   BACKGROUND command (run_in_background: true) so the harness wakes you when it returns:
     {po} wait --mailbox {mailbox} --timeout 600
   `wait` is pending-aware: it returns `pending` IMMEDIATELY if mail already sits unhandled
   (so it can't miss a post that arrived during step 1). Still, after launching it, do NOT
   trust the watch alone — IMMEDIATELY re-run `pending` once to close any gap; pending is the
   source of truth, the wake is just a nudge. (A past cagent lost a post to this race.)

3. When the wait returns (a post, the timer, or `pending`), GO BACK TO STEP 1.

SELF-IMPROVEMENT — this is how you and your successors get smarter; the log compounds:
- Hit friction (couldn't route a post, an ambiguous request, a recurring pattern, context got
  tight)? Record it: `{po} lesson --mailbox {mailbox} "symptom -> what you did -> fix"`.
- If it's a DURABLE operating rule the next cagent should START with, also add it to the notes
  that step 0 reads: `{po} learn --mailbox {mailbox} "the rule"`.
- Never delete notes or lessons; only add or refine. When your context grows large, post a note
  to `{mailbox}` with the current cursor and ask to be re-spawned fresh — your successor reads
  your notes (step 0) and catches up from the cursor, so it starts SMARTER than you, losing
  nothing. Anyone reaches you with `{po} send --to {mailbox} "..."`."""


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


def cmd_mailboxes(a):
    mbs = list_mailboxes(a.root)
    print("\n".join(mbs) if mbs else "(no mailboxes yet)")
    return 0


def cmd_feed(a):
    rows = feed(a.root, a.n)
    if not rows:
        print("(no posts in any mailbox yet)")
    for ts, mb, p in rows:
        print(f"{ts}  {p.get('from','?')}->{mb}: {p.get('body','')[:150]}")
    return 0


def cmd_lesson(a):
    r = append_lesson(a.root, a.mailbox, a.text, tags=a.tags)
    print(f"lesson {r['id']} recorded")
    return 0


def cmd_lessons(a):
    rows = read_lessons(a.root, a.mailbox)
    if not rows:
        print(f"(no lessons yet for {a.mailbox})")
    for r in rows:
        tg = f" [{r['tags']}]" if r.get("tags") else ""
        print(f"{r.get('ts','')} {r['id']}{tg} {r['text']}")
    return 0


def cmd_learn(a):
    append_note(a.root, a.mailbox, a.rule)
    print("note added to runbook")
    return 0


def cmd_notes(a):
    n = read_notes(a.root, a.mailbox)
    print(n if n.strip() else
          f"(no notes yet — add a rule with: postoffice.py learn --mailbox {a.mailbox} \"<rule>\")")
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
    mb = sub.add_parser("mailboxes", help="list all mailboxes"); mb.set_defaults(func=cmd_mailboxes)
    fd = sub.add_parser("feed", help="recent posts across ALL mailboxes (what's been said)")
    fd.add_argument("--n", type=int, default=20); fd.set_defaults(func=cmd_feed)

    le = sub.add_parser("lesson", help="append a friction/lesson (self-improvement ledger)")
    le.add_argument("--mailbox", required=True); le.add_argument("--tags", default=""); le.add_argument("text")
    le.set_defaults(func=cmd_lesson)
    les = sub.add_parser("lessons", help="list recorded lessons")
    les.add_argument("--mailbox", required=True); les.set_defaults(func=cmd_lessons)
    ln = sub.add_parser("learn", help="add a durable rule to the cagent runbook (read on startup)")
    ln.add_argument("--mailbox", required=True); ln.add_argument("rule"); ln.set_defaults(func=cmd_learn)
    no = sub.add_parser("notes", help="print the cagent operating runbook")
    no.add_argument("--mailbox", required=True); no.set_defaults(func=cmd_notes)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
