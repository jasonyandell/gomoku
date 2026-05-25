"""Tests for scripts/agent_fleet.py — the claude-agents fleet gauge/inspector.

Covers the pure join/gauge logic with fixtures (no daemon needed) plus the
transcript digest against a tiny synthetic JSONL.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import agent_fleet as af  # noqa: E402

REPO = "/Users/jason/code/gomoku"


def test_short_id():
    assert af.short_id("55038670-b85a-4bb6-b41f-002828592c5e") == "55038670"
    assert af.short_id("") == ""


def test_index_roster_extracts_source_and_intent():
    roster = {"workers": {
        "55038670": {"dispatch": {"source": "fleet", "seed": {"intent": "race recipes"}}},
        "60c8a964": {"dispatch": {"source": "spare", "seed": {}}},
        "4cc0ac98": {"dispatch": {"source": "slash", "seed": {"name": "rsi-bro"}}},
    }}
    idx = af.index_roster(roster)
    assert idx["55038670"]["source"] == "fleet"
    assert idx["55038670"]["intent"] == "race recipes"
    assert idx["60c8a964"]["source"] == "spare"
    assert idx["4cc0ac98"]["name"] == "rsi-bro"


def test_index_roster_empty():
    assert af.index_roster({}) == {}


def test_latest_worktree_wins():
    records = [
        {"session_id": "abc", "worktree": "/a", "branch": "feat/x"},
        {"session_id": "abc", "worktree": "/b", "branch": "feat/y"},  # later → wins
        {"session_id": "def", "worktree": "/c", "branch": "main"},
    ]
    m = af.latest_worktree_by_session(records)
    assert m["abc"] == ("/b", "feat/y")
    assert m["def"] == ("/c", "main")


def test_parse_git_worktrees_locked_and_branch():
    porc = (
        "worktree /Users/jason/code/gomoku\n"
        "HEAD 2e4c398\n"
        "branch refs/heads/main\n"
        "\n"
        "worktree /Users/jason/code/gomoku/.claude/worktrees/agent-aaa\n"
        "HEAD 8fec4d4\n"
        "branch refs/heads/feat/research-lab-rename\n"
        "locked\n"
    )
    wts = af.parse_git_worktrees(porc)
    assert len(wts) == 2
    assert wts[0]["branch"] == "main" and wts[0]["locked"] is False
    assert wts[1]["locked"] is True
    assert wts[1]["branch"] == "feat/research-lab-rename"


def _board():
    fleet = [
        {"sessionId": "55038670-aaaa", "kind": "background", "name": "lab", "status": "idle"},
        {"sessionId": "cfbdadf1-bbbb", "kind": "background", "name": "me", "status": "busy"},
    ]
    roster_idx = {"55038670": {"source": "fleet", "intent": "", "name": ""},
                  "cfbdadf1": {"source": "fleet", "intent": "", "name": ""}}
    jobs = {
        "55038670": {"sessionId": "55038670-aaaa", "name": "lab", "state": "done"},
        "cfbdadf1": {"sessionId": "cfbdadf1-bbbb", "name": "me", "state": "working"},
        "64d53d62": {"sessionId": "64d53d62-cccc", "name": "wiki", "state": "done"},  # dead
    }
    wt_by_session = {
        "55038670-aaaa": (REPO, "main"),       # on main
        "64d53d62-cccc": (REPO, "main"),       # on main, dead
        "cfbdadf1-bbbb": ("/Users/jason/code/gomoku-x", "feat/x"),
    }
    return af.build_board(fleet, roster_idx, jobs, wt_by_session, REPO)


def test_build_board_alive_and_on_main_flags():
    board = _board()
    by = {r["ai_id"]: r for r in board}
    assert by["55038670"]["alive"] is True and by["55038670"]["on_main"] is True
    assert by["64d53d62"]["alive"] is False and by["64d53d62"]["on_main"] is True
    assert by["cfbdadf1"]["on_main"] is False
    assert by["55038670"]["source"] == "fleet"
    # alive sorted before dead
    assert board[-1]["ai_id"] == "64d53d62"


def test_compute_gauges_flags_main_contention():
    g = af.compute_gauges(_board(), [], REPO)
    assert g["agents_sharing_main"] == 2
    assert set(g["agents_sharing_main_ids"]) == {"55038670", "64d53d62"}
    assert g["live_fleet"] == 2


def test_compute_gauges_counts_leaked_locked():
    wts = [
        {"path": REPO, "branch": "main", "locked": False},
        {"path": f"{REPO}/.claude/worktrees/agent-aaa", "branch": "feat/a", "locked": True},
        {"path": f"{REPO}/.claude/worktrees/agent-bbb", "branch": "feat/b", "locked": False},
    ]
    g = af.compute_gauges([], wts, REPO)
    assert g["leaked_locked_subagent_worktrees"] == 1
    assert g["leaked_locked_worktree_branches"] == ["feat/a"]


def test_digest_session(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [
        {"type": "ai-title", "aiTitle": "demo session"},
        {"type": "last-prompt", "lastPrompt": "do the thing"},
        {"type": "last-prompt", "lastPrompt": "do the thing"},   # dup → collapsed
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Working on it."},
            {"type": "tool_use", "name": "Bash"},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "result: did the thing"},
        ]}},
        "not json",  # malformed line tolerated
    ]
    p.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows))
    out = af.digest_session(str(p))
    assert "TITLE: demo session" in out
    assert out.count("do the thing") == 1          # collapsed repeat
    assert "result: did the thing" in out
    assert "Bash×1" in out
