"""Tests for postoffice.py — the append-only fleet message bus.

Focus on the durability + catch-up guarantees: the log is append-only, the cursor
recovers missed posts (at-least-once), and ack never mutates history.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import postoffice as po  # noqa: E402


def test_append_and_read(tmp_path):
    a = po.append_post(tmp_path, "cagent", "hello", sender="jason", subject="hi")
    assert len(a["id"]) == 8 and a["to"] == "cagent" and a["from"] == "jason"
    log = po.read_log(tmp_path, "cagent")
    assert len(log) == 1 and log[0]["body"] == "hello"


def test_append_is_additive_only(tmp_path):
    po.append_post(tmp_path, "cagent", "one")
    raw1 = (tmp_path / "cagent.log").read_text()
    po.append_post(tmp_path, "cagent", "two")
    raw2 = (tmp_path / "cagent.log").read_text()
    assert raw2.startswith(raw1)                 # nothing rewritten, only added
    assert len(po.read_log(tmp_path, "cagent")) == 2


def test_pending_then_ack(tmp_path):
    for b in ("a", "b", "c"):
        po.append_post(tmp_path, "cagent", b)
    cur, items, total = po.pending(tmp_path, "cagent")
    assert cur == 0 and total == 3 and [p["body"] for _, p in items] == ["a", "b", "c"]
    assert items[0][0] == 1 and items[2][0] == 3            # 1-based seq
    po.ack(tmp_path, "cagent", through=2)
    cur, items, _ = po.pending(tmp_path, "cagent")
    assert cur == 2 and [p["body"] for _, p in items] == ["c"]


def test_missed_watch_catch_up(tmp_path):
    # cagent processed nothing, then THREE posts pile up while it was asleep/busy.
    for b in ("p1", "p2", "p3"):
        po.append_post(tmp_path, "cagent", b)
    # one scan recovers ALL of them (not just the latest) — at-least-once.
    _, items, _ = po.pending(tmp_path, "cagent")
    assert len(items) == 3
    n = po.ack(tmp_path, "cagent", through=None)            # --all
    assert n == 3
    assert po.pending(tmp_path, "cagent")[1] == []          # drained


def test_ack_idempotent_and_clamped(tmp_path):
    po.append_post(tmp_path, "cagent", "only")
    assert po.ack(tmp_path, "cagent", through=99) == 1      # clamped to total
    assert po.ack(tmp_path, "cagent", through=None) == 1    # re-ack is a no-op move
    assert po.get_cursor(tmp_path, "cagent") == 1


def test_read_tolerates_corrupt_line(tmp_path):
    po.append_post(tmp_path, "cagent", "good")
    with open(tmp_path / "cagent.log", "a") as f:
        f.write("not-json-garbage\n")
    assert len(po.read_log(tmp_path, "cagent")) == 1        # bad line skipped, good kept


def test_spawn_prompt_is_pasteable():
    pr = po.spawn_prompt("cagent")
    assert "postoffice.py pending --mailbox cagent" in pr
    assert "postoffice.py wait --mailbox cagent" in pr
    assert "run_in_background" in pr and "FOREVER" in pr


def test_lessons_are_append_only(tmp_path):
    po.append_lesson(tmp_path, "cagent", "ambiguous post -> asked for clarification")
    raw1 = (tmp_path / "cagent.lessons.jsonl").read_text()
    r2 = po.append_lesson(tmp_path, "cagent", "perf post -> route to lab", tags="routing")
    raw2 = (tmp_path / "cagent.lessons.jsonl").read_text()
    assert raw2.startswith(raw1)                          # additive only
    L = po.read_lessons(tmp_path, "cagent")
    assert len(L) == 2 and L[1]["tags"] == "routing" and L[1]["id"] == r2["id"]


def test_notes_runbook_accumulates(tmp_path):
    assert po.read_notes(tmp_path, "cagent") == ""        # empty until a rule is learned
    po.append_note(tmp_path, "cagent", "route perf posts to the lab mailbox")
    po.append_note(tmp_path, "cagent", "ack in batches, not per-post")
    n = po.read_notes(tmp_path, "cagent")
    assert "operating notes" in n                          # header seeded once
    assert "route perf posts" in n and "ack in batches" in n
    assert n.count("\n- (") == 2                          # two dated bullets


def test_wait_is_pending_aware(tmp_path):
    # a cagent (a4c2c77e) caught this race: a post already in the log must not be missed.
    po.append_post(tmp_path, "cagent", "arrived before the watch armed")
    assert po.wait(tmp_path, "cagent", timeout=30) == "pending"   # returns at once, no block
    po.ack(tmp_path, "cagent", through=None)
    assert po.wait(tmp_path, "cagent", timeout=1) == "timeout"     # nothing pending → blocks to timeout


def test_feed_and_mailboxes_cross_mailbox(tmp_path):
    po.append_post(tmp_path, "cagent", "ping", sender="operator")
    po.append_post(tmp_path, "reply", "pong", sender="cagent")
    assert po.list_mailboxes(tmp_path) == ["cagent", "reply"]
    rows = po.feed(tmp_path, n=20)
    assert {mb for _, mb, _ in rows} == {"cagent", "reply"}        # both mailboxes in one view
    assert [p["body"] for _, _, p in rows][-1] == "pong"           # chronological


def test_spawn_prompt_has_self_improvement():
    pr = po.spawn_prompt("cagent")
    assert "SELF-IMPROVEMENT" in pr
    assert "notes --mailbox cagent" in pr      # reads runbook on startup (step 0)
    assert "lesson --mailbox cagent" in pr     # records friction
    assert "learn --mailbox cagent" in pr      # promotes durable rule into runbook
