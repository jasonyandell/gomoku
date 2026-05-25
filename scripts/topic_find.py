#!/usr/bin/env python3
"""topic_find.py — find the most recent session where a TOPIC came up (semantic, not keyword).

FTS `search` matches words; this matches MEANING. The trick: with a few dozen sessions the
entire corpus fits in one Haiku context, so we skip embeddings/vector-stores entirely — we
hand Haiku a compact card per session (date + title + what the USER said) and let it pick the
matches, ranked by recency. "Haiku-as-RAG": the cards are the corpus, Haiku is the retriever.

The semantic fingerprint is the user's own prompts (already cached by session_db) — phrased
in their words, which is exactly what a topic query should match against.

  python scripts/topic_find.py "what's next for the ML"
  python scripts/topic_find.py "comparing our model against an external engine" --max-sessions 50
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_db as sdb  # noqa: E402


def build_corpus(con, max_sessions: int = 60, max_prompts: int = 6) -> str:
    """One compact card per session (most-recent first): date, title, resume id, user prompts.

    Pure over the DB — testable without an LLM."""
    cards = []
    rows = con.execute(
        "SELECT session_id, ai_id, title, file_mtime FROM sessions "
        "ORDER BY file_mtime DESC LIMIT ?", (max_sessions,)).fetchall()
    for r in rows:
        date = dt.datetime.fromtimestamp(r["file_mtime"]).strftime("%Y-%m-%d")
        said = [m["text"] for m in con.execute(
            "SELECT text FROM messages WHERE session_id=? AND kind='prompt' ORDER BY seq LIMIT ?",
            (r["session_id"], max_prompts)).fetchall()]
        if not said:  # fall back to assistant lines for prompt-less sessions
            said = [m["text"] for m in con.execute(
                "SELECT text FROM messages WHERE session_id=? AND role='assistant' ORDER BY seq LIMIT 2",
                (r["session_id"],)).fetchall()]
        said_s = " | ".join(s.replace("\n", " ")[:160] for s in said) or "(no captured text)"
        cards.append(f"[{r['ai_id']}] {date} \"{r['title'] or 'untitled'}\" "
                     f"(resume: claude --resume {r['session_id']})\n    said: {said_s}")
    return "\n".join(cards)


def ask_haiku(corpus: str, topic: str, model: str = "haiku") -> str:
    prompt = (
        "You are a semantic search over a user's past Claude Code sessions. Below are sessions "
        "(MOST RECENT FIRST) with date, title, resume command, and what the USER said in them.\n\n"
        f'The user wants the most recent session where this TOPIC came up: "{topic}"\n\n'
        "Match by MEANING, not keywords — the topic may be phrased differently or only implied. "
        "Return the 1–4 genuinely-matching sessions RANKED BY RECENCY (most recent first; use the "
        "dates). For each: date, title, ai-id, a one-line why-it-matches, and the resume command. "
        "If nothing matches, say so plainly. Be concise.\n\n"
        f"SESSIONS:\n{corpus}\n")
    try:
        out = subprocess.run(["claude", "-p", "--model", model, prompt],
                             capture_output=True, text=True, timeout=180)
        return out.stdout.strip() or (out.stderr.strip() or "(no output from haiku)")
    except (OSError, subprocess.SubprocessError) as e:
        return f"(haiku call failed: {e})"


def main(argv=None):
    p = argparse.ArgumentParser(description="Semantic, recency-ranked topic search (haiku-as-RAG).")
    p.add_argument("topic", help="free-text topic, e.g. \"what's next for the ML\"")
    p.add_argument("--repo", default=os.getcwd())
    p.add_argument("--db")
    p.add_argument("--model", default="haiku")
    p.add_argument("--max-sessions", type=int, default=60)
    p.add_argument("--max-prompts", type=int, default=6)
    p.add_argument("--show-corpus", action="store_true", help="print the assembled corpus and exit")
    args = p.parse_args(argv)

    con = sdb.connect(args.db or sdb.db_path_for(args.repo))
    sdb.sync(con, str(sdb.project_dir_for(args.repo)))           # keep cache fresh (cheap)
    corpus = build_corpus(con, args.max_sessions, args.max_prompts)
    if args.show_corpus:
        print(corpus)
        return 0
    print(f"# topic: {args.topic}\n")
    print(ask_haiku(corpus, args.topic, args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
