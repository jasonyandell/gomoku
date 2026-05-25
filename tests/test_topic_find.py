"""Tests for topic_find.build_corpus (the pure, LLM-free part of haiku-as-RAG).

The Haiku call itself isn't unit-tested (it shells out to `claude -p`); we test that the
corpus is assembled correctly: most-recent-first, user prompts as the fingerprint, resume
commands present, and a sensible fallback for prompt-less sessions.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import session_db as sdb     # noqa: E402
import topic_find as tf      # noqa: E402


def _seed(tmp_path):
    con = sdb.connect(tmp_path / "t.db")
    proj = tmp_path / "proj"; proj.mkdir()
    return con, proj


def _write(proj, sid, title, prompts, mtime):
    import json
    rows = [{"type": "ai-title", "aiTitle": title}]
    for q in prompts:
        rows.append({"type": "last-prompt", "lastPrompt": q})
    p = proj / f"{sid}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    import os
    os.utime(p, (mtime, mtime))


def test_corpus_recent_first_with_prompts_and_resume(tmp_path):
    con, proj = _seed(tmp_path)
    old = time.time() - 86400
    new = time.time()
    _write(proj, "11111111-aaaa", "old derby session", ["race the recipes"], old)
    _write(proj, "22222222-bbbb", "ml direction chat", ["what is next for the model"], new)
    sdb.sync(con, str(proj))
    corpus = tf.build_corpus(con)
    lines = corpus.splitlines()
    # most-recent session card appears first
    assert lines[0].startswith("[22222222]")
    assert "[11111111]" in corpus
    # user's words are the fingerprint, and resume commands are present
    assert "what is next for the model" in corpus
    assert "claude --resume 22222222-bbbb" in corpus


def test_corpus_fallback_for_promptless_session(tmp_path):
    import json
    con, proj = _seed(tmp_path)
    p = proj / "33333333-cccc.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"type": "ai-title", "aiTitle": "silent session"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "did some analysis"}]}},
    ]))
    sdb.sync(con, str(proj))
    corpus = tf.build_corpus(con)
    assert "silent session" in corpus
    assert "did some analysis" in corpus or "(no captured text)" in corpus
