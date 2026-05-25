"""Unit tests for the worktree→session recorder's pure logic."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import worktree_session as ws  # noqa: E402

SID = "8acd3c77-0502-4236-a51e-d05b96438a55"


def test_resume_and_transcript_glob():
    assert ws.resume_cmd(SID) == f"claude --resume {SID}"
    g = ws.transcript_glob(SID)
    assert g.endswith(f"/{SID}.jsonl")
    assert "/.claude/projects/*/" in g


def test_build_record_shape():
    rec = ws.build_record("/Users/jason/code/gomoku-foo", "feat/foo", SID,
                          "2026-05-25T11:00:00-05:00")
    assert rec["worktree"] == "/Users/jason/code/gomoku-foo"
    assert rec["branch"] == "feat/foo"
    assert rec["session_id"] == SID
    assert rec["resume"] == f"claude --resume {SID}"
    # round-trips as one registry line
    assert json.loads(json.dumps(rec)) == rec


def test_render_session_file_is_greppable():
    rec = ws.build_record("/w", "feat/x", SID, "2026-05-25T11:00:00-05:00")
    body = ws.render_session_file(rec)
    assert f"session_id={SID}" in body
    assert "branch=feat/x" in body
    assert f"resume=claude --resume {SID}" in body
    # every non-comment line is key=value (machine-parseable)
    for line in body.splitlines():
        if line and not line.startswith("#"):
            assert "=" in line


def test_session_id_defaults_to_unknown(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    assert ws.session_id() == "unknown"
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", SID)
    assert ws.session_id() == SID
