#!/usr/bin/env python3
"""session_db.py — a SQLite cache over Claude session transcripts, for fast search.

The transcripts (`~/.claude/projects/<proj>/<uuid>.jsonl`) are the source of truth;
this builds an FTS5 cache so "which session did I discuss X in?" is a millisecond
query instead of a re-scan of every multi-MB JSONL.

Cache model (cache hit / miss = the whole point):
  - `sync` walks the project's transcripts. For each, it compares the file's mtime to
    what's stored. **Hit** (mtime unchanged) → skip. **Miss** (new or modified) →
    re-import that one session from its JSONL. Deleted transcripts are pruned.
  - `search`, `topics`, and `export` run an incremental `sync` first (cheap — only
    changed files re-import), so the cache is always fresh without a manual step.

Subcommands:
  sync            import/refresh changed sessions (incremental by mtime)
  search <query>  FTS5 search; prints rank, title, resume + fork commands  [--scope]
  topics          (re)compute topic<->session edges from topics.json
  export          dump JSON (sessions, topics, edges, prompts) for the mindmap
  resume|fork <id> print the copy-paste CLI command for a session

Fork semantics (verified against code.claude.com/docs/en/sessions.md + `claude --help`):
  `claude --resume <uuid> --fork-session` copies the conversation history into a NEW
  session id and leaves the original untouched. Forks always branch from the latest
  message; previously-approved permissions do not carry over to the fork.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

HOME = Path.home()
PROJECTS = HOME / ".claude" / "projects"
CACHE_DIR = HOME / ".claude" / "agent-fleet"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TOPICS = SCRIPT_DIR / "topics.json"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  session_id TEXT PRIMARY KEY, ai_id TEXT, title TEXT, path TEXT,
  size_bytes INTEGER, file_mtime REAL, n_human INTEGER, n_assistant INTEGER,
  imported_at REAL);
CREATE TABLE IF NOT EXISTS messages(
  session_id TEXT, seq INTEGER, role TEXT, kind TEXT, text TEXT);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(text, session_id UNINDEXED, role UNINDEXED);
CREATE TABLE IF NOT EXISTS topics(id TEXT PRIMARY KEY, label TEXT, keywords TEXT);
CREATE TABLE IF NOT EXISTS session_topics(session_id TEXT, topic_id TEXT, weight INTEGER,
  PRIMARY KEY(session_id, topic_id));
"""


# ---------- copy-paste commands ----------

def resume_cmd(session_id: str) -> str:
    return f"claude --resume {session_id}"


def fork_cmd(session_id: str) -> str:
    return f"claude --resume {session_id} --fork-session"


# ---------- pure parsing ----------

def _text_blocks(msg) -> list[str]:
    c = (msg or {}).get("content")
    if isinstance(c, str):
        return [c]
    if isinstance(c, list):
        return [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"]
    return []


def parse_session(path: str):
    """-> (title, [(seq, role, kind, text)], n_human, n_assistant). Tolerates junk lines."""
    title, msgs, seq, last_human = "", [], 0, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            t = ev.get("type")
            if t == "ai-title":
                title = ev.get("aiTitle", title)
            elif t == "last-prompt":
                q = (ev.get("lastPrompt") or "").strip()
                if q and q != last_human:           # collapse the re-emitted prompt
                    msgs.append((seq, "human", "prompt", q)); seq += 1; last_human = q
            elif t == "assistant":
                for b in _text_blocks(ev.get("message")):
                    b = b.strip()
                    if not b:
                        continue
                    low = b.lower()
                    kind = "result" if low.startswith(("result:", "needs input:", "failed:")) else "text"
                    msgs.append((seq, "assistant", kind, b)); seq += 1
    n_human = sum(1 for m in msgs if m[1] == "human")
    return title, msgs, n_human, len(msgs) - n_human


def build_fts_query(keywords: list[str]) -> str:
    """OR of keywords, each quoted as an FTS5 phrase.

    Quoting every keyword (not just multi-word ones) keeps terms with hyphens or
    underscores — 'self-play', 'run_sweep', 'self-improving' — from being parsed as
    FTS operators, which would otherwise error and silently drop the whole topic."""
    parts = []
    for kw in keywords:
        kw = kw.strip().replace('"', "")
        if kw:
            parts.append(f'"{kw}"')
    return " OR ".join(parts)


# ---------- db glue ----------

def db_path_for(repo_root: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    enc = os.path.realpath(repo_root).replace("/", "-")
    return CACHE_DIR / f"{enc}.db"


def connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def project_dir_for(repo_root: str) -> Path:
    return PROJECTS / os.path.realpath(repo_root).replace("/", "-")


def import_session(con: sqlite3.Connection, path: str) -> None:
    sid = os.path.splitext(os.path.basename(path))[0]
    st = os.stat(path)
    title, msgs, nh, na = parse_session(path)
    con.execute("DELETE FROM messages WHERE session_id=?", (sid,))
    con.execute("DELETE FROM search_fts WHERE session_id=?", (sid,))
    con.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
    con.execute(
        "INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?,?)",
        (sid, sid[:8], title, path, st.st_size, st.st_mtime, nh, na, time.time()))
    con.executemany("INSERT INTO messages VALUES(?,?,?,?,?)",
                    [(sid, s, r, k, t) for (s, r, k, t) in msgs])
    con.executemany("INSERT INTO search_fts(text, session_id, role) VALUES(?,?,?)",
                    [(t, sid, r) for (_, r, _, t) in msgs])


def sync(con: sqlite3.Connection, project_dir: str, force: bool = False) -> dict:
    files = glob.glob(os.path.join(project_dir, "*.jsonl"))
    on_disk = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    cached = {r["session_id"]: r["file_mtime"]
              for r in con.execute("SELECT session_id, file_mtime FROM sessions")}
    imported = skipped = 0
    for sid, path in on_disk.items():
        mtime = os.stat(path).st_mtime
        if not force and sid in cached and abs(cached[sid] - mtime) < 1e-6:
            skipped += 1
            continue
        import_session(con, path)
        imported += 1
    pruned = 0
    for sid in set(cached) - set(on_disk):
        con.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        con.execute("DELETE FROM search_fts WHERE session_id=?", (sid,))
        con.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
        pruned += 1
    con.commit()
    return {"imported": imported, "skipped": skipped, "pruned": pruned, "total": len(on_disk)}


def search(con: sqlite3.Connection, query: str, scope: str = "human", limit: int = 10) -> list[dict]:
    role_clause, params = "", [query]
    if scope in ("human", "assistant"):
        role_clause = " AND role=?"
        params.append(scope)
    sql = (f"SELECT session_id, count(*) AS hits FROM search_fts "
           f"WHERE search_fts MATCH ?{role_clause} GROUP BY session_id "
           f"ORDER BY hits DESC LIMIT ?")
    try:
        rows = con.execute(sql, params + [limit]).fetchall()
    except sqlite3.OperationalError:                      # bad FTS syntax -> phrase search
        params[0] = '"' + query.replace('"', "") + '"'
        rows = con.execute(sql, params + [limit]).fetchall()
    out = []
    for r in rows:
        meta = con.execute("SELECT ai_id, title FROM sessions WHERE session_id=?",
                            (r["session_id"],)).fetchone()
        snip = con.execute(
            "SELECT snippet(search_fts,0,'[',']','…',12) s FROM search_fts "
            "WHERE search_fts MATCH ? AND session_id=? LIMIT 1",
            [params[0], r["session_id"]]).fetchone()
        out.append({"session_id": r["session_id"], "ai_id": meta["ai_id"] if meta else r["session_id"][:8],
                    "title": (meta["title"] if meta else "") or "", "hits": r["hits"],
                    "snippet": (snip["s"] if snip else "").replace("\n", " ")})
    return out


def load_topics(topics_path: str) -> list[dict]:
    try:
        return json.loads(Path(topics_path).read_text())
    except (OSError, json.JSONDecodeError):
        return []


def recompute_topics(con: sqlite3.Connection, topics: list[dict]) -> int:
    con.execute("DELETE FROM topics")
    con.execute("DELETE FROM session_topics")
    edges = 0
    for tp in topics:
        con.execute("INSERT INTO topics VALUES(?,?,?)",
                    (tp["id"], tp.get("label", tp["id"]), json.dumps(tp.get("keywords", []))))
        q = build_fts_query(tp.get("keywords", []))
        if not q:
            continue
        try:
            rows = con.execute("SELECT session_id, count(*) w FROM search_fts "
                               "WHERE search_fts MATCH ? GROUP BY session_id", (q,)).fetchall()
        except sqlite3.OperationalError:
            continue
        for r in rows:
            con.execute("INSERT OR REPLACE INTO session_topics VALUES(?,?,?)",
                        (r["session_id"], tp["id"], r["w"]))
            edges += 1
    con.commit()
    return edges


def export_graph(con: sqlite3.Connection, max_prompts: int = 5) -> dict:
    sessions = []
    for s in con.execute("SELECT session_id, ai_id, title, n_human, n_assistant FROM sessions"):
        prompts = [r["text"] for r in con.execute(
            "SELECT text FROM messages WHERE session_id=? AND kind='prompt' ORDER BY seq LIMIT ?",
            (s["session_id"], max_prompts))]
        sessions.append({"session_id": s["session_id"], "ai_id": s["ai_id"],
                         "title": s["title"] or "", "n_human": s["n_human"],
                         "n_assistant": s["n_assistant"], "prompts": prompts,
                         "resume": resume_cmd(s["session_id"]), "fork": fork_cmd(s["session_id"])})
    topics = [{"id": t["id"], "label": t["label"]}
              for t in con.execute("SELECT id, label FROM topics")]
    edges = [{"session_id": e["session_id"], "topic_id": e["topic_id"], "weight": e["weight"]}
             for e in con.execute("SELECT session_id, topic_id, weight FROM session_topics")]
    return {"sessions": sessions, "topics": topics, "edges": edges}


# ---------- CLI ----------

def _con(args):
    db = args.db or db_path_for(args.repo)
    return connect(db), db


def cmd_sync(args):
    con, db = _con(args)
    r = sync(con, str(project_dir_for(args.repo)), force=args.force)
    print(f"db: {db}")
    print(f"synced: {r['imported']} imported, {r['skipped']} cache-hit, {r['pruned']} pruned "
          f"({r['total']} transcripts)")
    return 0


def cmd_search(args):
    con, _ = _con(args)
    if not args.no_sync:
        sync(con, str(project_dir_for(args.repo)))
    rows = search(con, args.query, args.scope, args.limit)
    if not rows:
        print("no sessions matched", file=sys.stderr)
        return 1
    for r in rows:
        print(f"{r['ai_id']}  hits={r['hits']:<3} {(r['title'] or '<untitled>')[:36]:36}")
        print(f"           ↳ {r['snippet'][:140]}")
        print(f"           resume: {resume_cmd(r['session_id'])}")
        print(f"           fork:   {fork_cmd(r['session_id'])}")
    return 0


def cmd_topics(args):
    con, _ = _con(args)
    sync(con, str(project_dir_for(args.repo)))
    topics = load_topics(args.topics)
    n = recompute_topics(con, topics)
    print(f"{len(topics)} topics, {n} session↔topic edges computed")
    for t in topics:
        c = con.execute("SELECT count(*) c FROM session_topics WHERE topic_id=?", (t["id"],)).fetchone()["c"]
        print(f"  {t['id']:16} {c} sessions")
    return 0


def cmd_export(args):
    con, _ = _con(args)
    sync(con, str(project_dir_for(args.repo)))
    if not con.execute("SELECT count(*) c FROM topics").fetchone()["c"]:
        recompute_topics(con, load_topics(args.topics))
    print(json.dumps(export_graph(con), indent=2))
    return 0


def cmd_cmd(args):
    print((fork_cmd if args.kind == "fork" else resume_cmd)(args.session_id))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="SQLite cache + FTS search over Claude session transcripts.")
    p.add_argument("--repo", default=os.getcwd())
    p.add_argument("--db", help="sqlite path (default ~/.claude/agent-fleet/<proj>.db)")
    p.add_argument("--topics", default=str(DEFAULT_TOPICS))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="import/refresh changed sessions")
    s.add_argument("--force", action="store_true", help="re-import all, ignore cache")
    s.set_defaults(func=cmd_sync)

    se = sub.add_parser("search", help="FTS search; prints resume + fork commands")
    se.add_argument("query")
    se.add_argument("--scope", choices=["human", "assistant", "all"], default="human")
    se.add_argument("--limit", type=int, default=10)
    se.add_argument("--no-sync", action="store_true")
    se.set_defaults(func=cmd_search)

    tp = sub.add_parser("topics", help="recompute topic<->session edges")
    tp.set_defaults(func=cmd_topics)

    ex = sub.add_parser("export", help="dump graph JSON for the mindmap")
    ex.set_defaults(func=cmd_export)

    cc = sub.add_parser("cmd", help="print resume/fork command for a session id")
    cc.add_argument("kind", choices=["resume", "fork"])
    cc.add_argument("session_id")
    cc.set_defaults(func=cmd_cmd)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
