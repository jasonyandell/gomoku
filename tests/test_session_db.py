"""Tests for session_db.py (SQLite cache + FTS search) and session_mindmap.build_payload.

Uses a temp DB + fake transcripts — no real ~/.claude data, no daemon.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import session_db as sdb        # noqa: E402
import session_mindmap as smap  # noqa: E402

SID = "410251ca-9352-4e11-80fd-c0acb006aed7"


def _write(proj: Path, sid: str, rows):
    p = proj / f"{sid}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def _sample():
    return [
        {"type": "ai-title", "aiTitle": "alpha zero strategy analysis"},
        {"type": "last-prompt", "lastPrompt": "what can we learn and improve about alpha zero"},
        {"type": "last-prompt", "lastPrompt": "what can we learn and improve about alpha zero"},  # dup
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Here is the plan."}, {"type": "tool_use", "name": "Bash"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "result: shipped the strategy"}]}},
        "garbage-not-json",
    ]


def test_cmd_strings():
    assert sdb.resume_cmd(SID) == f"claude --resume {SID}"
    assert sdb.fork_cmd(SID) == f"claude --resume {SID} --fork-session"


def test_parse_session_dedup_and_kinds(tmp_path):
    p = _write(tmp_path, SID, _sample())
    title, msgs, nh, na = sdb.parse_session(str(p))
    assert title == "alpha zero strategy analysis"
    assert nh == 1 and na == 2                       # dup prompt collapsed
    kinds = {(r, k) for (_, r, k, _) in msgs}
    assert ("human", "prompt") in kinds and ("assistant", "result") in kinds


def test_build_fts_query_quotes_phrases():
    assert sdb.build_fts_query(["alpha", "neural engine"]) == '"alpha" OR "neural engine"'
    assert sdb.build_fts_query(["self-play", "run_sweep"]) == '"self-play" OR "run_sweep"'
    assert sdb.build_fts_query([]) == ""


def _fresh(tmp_path):
    con = sdb.connect(tmp_path / "t.db")
    proj = tmp_path / "proj"; proj.mkdir()
    return con, proj


def test_sync_cache_hit_miss_and_prune(tmp_path):
    con, proj = _fresh(tmp_path)
    p = _write(proj, SID, _sample())
    assert sdb.sync(con, str(proj)) == {"imported": 1, "skipped": 0, "pruned": 0, "total": 1}
    # unchanged → cache hit
    assert sdb.sync(con, str(proj))["skipped"] == 1
    # touch mtime forward → cache miss, re-import
    future = time.time() + 100
    os.utime(p, (future, future))
    assert sdb.sync(con, str(proj))["imported"] == 1
    # delete → pruned
    p.unlink()
    r = sdb.sync(con, str(proj))
    assert r["pruned"] == 1 and r["total"] == 0
    assert con.execute("SELECT count(*) c FROM sessions").fetchone()["c"] == 0


def test_search_scope_and_safety(tmp_path):
    con, proj = _fresh(tmp_path)
    _write(proj, SID, _sample())
    sdb.sync(con, str(proj))
    assert [r["ai_id"] for r in sdb.search(con, "alpha", scope="human")] == ["410251ca"]
    assert sdb.search(con, "alpha", scope="assistant") == []     # "alpha" only in the prompt
    assert sdb.search(con, "strategy", scope="assistant")        # in the result line
    assert sdb.search(con, "zzznotpresent", scope="all") == []
    assert isinstance(sdb.search(con, "alpha AND", scope="all"), list)  # bad syntax -> no crash


def test_topics_and_export(tmp_path):
    con, proj = _fresh(tmp_path)
    _write(proj, SID, _sample())
    sdb.sync(con, str(proj))
    n = sdb.recompute_topics(con, [{"id": "az", "label": "AlphaZero", "keywords": ["alpha", "strategy"]}])
    assert n == 1
    g = sdb.export_graph(con)
    assert g["topics"] == [{"id": "az", "label": "AlphaZero"}]
    assert g["edges"][0]["session_id"] == SID and g["edges"][0]["weight"] >= 1
    s = g["sessions"][0]
    assert s["ai_id"] == "410251ca" and s["fork"].endswith("--fork-session") and s["prompts"]


def test_mindmap_payload_only_linked_sessions():
    graph = {
        "topics": [{"id": "az", "label": "AlphaZero"}, {"id": "buf", "label": "Buffer"}],
        "sessions": [
            {"session_id": SID, "ai_id": "410251ca", "title": "strategy", "n_human": 1,
             "n_assistant": 2, "prompts": ["x"], "resume": "r", "fork": "f"},
            {"session_id": "deadbeef-0000", "ai_id": "deadbeef", "title": "orphan", "n_human": 0,
             "n_assistant": 0, "prompts": [], "resume": "r", "fork": "f"},
        ],
        "edges": [{"session_id": SID, "topic_id": "az", "weight": 3}],
    }
    pl = smap.build_payload(graph)
    ids = {n["id"] for n in pl["nodes"]}
    assert "topic:az" in ids and f"sess:{SID}" in ids
    assert "sess:deadbeef-0000" not in ids          # orphan (no edge) excluded
    assert f"sess:{SID}" in pl["details"]
    assert len(pl["edges"]) == 1


def test_render_html_embeds_data():
    html = smap.render_html({"nodes": [], "edges": [], "details": {}})
    assert "vis-network" in html and "__DATA__" not in html and "fork" in html.lower()
