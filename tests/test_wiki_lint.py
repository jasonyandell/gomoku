"""Tests for scripts/wiki_lint.py — synthetic wiki fixtures, one pass/fail
case per check in wiki/curation.md § Lint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import wiki_lint  # noqa: E402


def make_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    (wiki / "topics").mkdir(parents=True)
    (wiki / "ops").mkdir()
    (wiki / "_archive").mkdir()
    return wiki


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# 1. Banners
# --------------------------------------------------------------------------

def test_banner_pass_variants(tmp_path):
    wiki = make_wiki(tmp_path)
    write(wiki / "topics" / "a.md", "# A\n\n> **Status: LIVE** *(2026-07-04)* — fine.\n\nBody.\n")
    write(wiki / "topics" / "b.md", "# B\n\n**HISTORICAL (2026-06-01 -> 06-02).** Closed era.\n")
    write(
        wiki / "topics" / "c.md",
        "# C\n\n> **Status (2026-07-04): SUPERSEDED-BY([a.md](a.md)); trimmed.**\n",
    )
    write(wiki / "index.md", "# Index\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    banner_findings = result.by_check().get("banners", [])
    assert banner_findings == []


def test_banner_missing_marker_flagged(tmp_path):
    wiki = make_wiki(tmp_path)
    write(wiki / "topics" / "no_banner.md", "# No Banner\n\nJust prose, no status line at all.\n")
    write(wiki / "index.md", "# Index\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    banner_findings = result.by_check().get("banners", [])
    assert len(banner_findings) == 1
    assert banner_findings[0].severity == "error"
    assert banner_findings[0].file == "topics/no_banner.md"


def test_banner_missing_date_flagged(tmp_path):
    wiki = make_wiki(tmp_path)
    write(wiki / "topics" / "no_date.md", "# No Date\n\n**Status: LIVE.** No date anywhere nearby.\n")
    write(wiki / "index.md", "# Index\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    banner_findings = result.by_check().get("banners", [])
    assert len(banner_findings) == 1
    assert "no YYYY-MM-DD date" in banner_findings[0].message


# --------------------------------------------------------------------------
# 2. Links
# --------------------------------------------------------------------------

def test_links_pass(tmp_path):
    wiki = make_wiki(tmp_path)
    write(wiki / "topics" / "a.md", "# A\n\n> **Status: LIVE** *(2026-07-04)*\n\nSee [b](b.md#section).\n")
    write(wiki / "topics" / "b.md", "# B\n\n> **Status: LIVE** *(2026-07-04)*\n")
    write(wiki / "index.md", "# Index\n\n[a](topics/a.md)\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    assert result.by_check().get("links", []) == []


def test_links_broken_flagged(tmp_path):
    wiki = make_wiki(tmp_path)
    write(
        wiki / "topics" / "a.md",
        "# A\n\n> **Status: LIVE** *(2026-07-04)*\n\nSee [missing](does-not-exist.md).\n",
    )
    write(wiki / "index.md", "# Index\n\n[a](topics/a.md)\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    link_findings = result.by_check().get("links", [])
    assert len(link_findings) == 1
    assert link_findings[0].severity == "error"
    assert "does-not-exist.md" in link_findings[0].message


def test_links_ignore_archive(tmp_path):
    wiki = make_wiki(tmp_path)
    write(
        wiki / "_archive" / "old.md",
        "# Old\n\nSee [gone](nowhere.md).\n",
    )
    write(wiki / "index.md", "# Index\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    assert result.by_check().get("links", []) == []


# --------------------------------------------------------------------------
# 3. Orphans
# --------------------------------------------------------------------------

def test_orphan_pass_hub_link(tmp_path):
    wiki = make_wiki(tmp_path)
    write(wiki / "topics" / "a.md", "# A\n\n> **Status: LIVE** *(2026-07-04)*\n")
    write(wiki / "hub.md", "# Hub\n\n[a](topics/a.md)\n")
    write(wiki / "index.md", "# Index\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    assert result.by_check().get("orphans", []) == []


def test_orphan_zero_inbound_flagged(tmp_path):
    wiki = make_wiki(tmp_path)
    write(wiki / "topics" / "lonely.md", "# Lonely\n\n> **Status: LIVE** *(2026-07-04)*\n")
    write(wiki / "index.md", "# Index\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    orphan_findings = result.by_check().get("orphans", [])
    assert len(orphan_findings) == 1
    assert orphan_findings[0].severity == "warning"
    assert "no inbound links" in orphan_findings[0].message


def test_orphan_only_from_topic_warns(tmp_path):
    wiki = make_wiki(tmp_path)
    write(
        wiki / "topics" / "a.md",
        "# A\n\n> **Status: LIVE** *(2026-07-04)*\n\n[b](b.md)\n",
    )
    write(wiki / "topics" / "b.md", "# B\n\n> **Status: LIVE** *(2026-07-04)*\n")
    write(wiki / "index.md", "# Index\n\n[a](topics/a.md)\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    orphan_findings = result.by_check().get("orphans", [])
    assert len(orphan_findings) == 1
    assert orphan_findings[0].file == "topics/b.md"
    assert "only via other topic pages" in orphan_findings[0].message


# --------------------------------------------------------------------------
# 4. Size / rotation
# --------------------------------------------------------------------------

def test_rotation_pass_small_file(tmp_path):
    wiki = make_wiki(tmp_path)
    write(wiki / "log.md", "# Log\n\nsmall\n")
    write(wiki / "index.md", "# Index\n")
    result = wiki_lint.run(wiki)
    assert result.by_check().get("rotation", []) == []


def test_rotation_smell_flagged(tmp_path):
    wiki = make_wiki(tmp_path)
    write(wiki / "log.md", "# Log\n\n" + ("x" * (70 * 1024)))
    write(wiki / "index.md", "# Index\n")
    result = wiki_lint.run(wiki)
    rotation_findings = result.by_check().get("rotation", [])
    assert len(rotation_findings) == 1
    assert rotation_findings[0].severity == "warning"
    assert rotation_findings[0].file == "log.md"


def test_rotation_skips_archived_marker(tmp_path):
    wiki = make_wiki(tmp_path)
    write(
        wiki / "ops" / "big.md",
        "# Big\n\n> ARCHIVED-IN-PLACE 2026-07-04 — frozen.\n\n" + ("x" * (70 * 1024)),
    )
    write(wiki / "index.md", "# Index\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    assert result.by_check().get("rotation", []) == []


def test_chronicle_itis_notice(tmp_path):
    wiki = make_wiki(tmp_path)
    write(
        wiki / "topics" / "huge.md",
        "# Huge\n\n> **Status: LIVE** *(2026-07-04)*\n\n" + ("x" * (30 * 1024)),
    )
    write(wiki / "index.md", "# Index\n")
    write(wiki / "log.md", "# Log\n")
    result = wiki_lint.run(wiki)
    findings = result.by_check().get("chronicle-itis", [])
    assert len(findings) == 1
    assert findings[0].severity == "notice"


# --------------------------------------------------------------------------
# 5. Index honesty
# --------------------------------------------------------------------------

def test_index_honesty_pass_recent(tmp_path):
    wiki = make_wiki(tmp_path)
    write(
        wiki / "log.md",
        "# Log\n\n## [2026-07-04] latest\nstuff\n\n## [2026-06-20] older\nstuff\n",
    )
    write(
        wiki / "index.md",
        "# Index\n\n**You are here.**\n- Now (2026-07-01): still fresh.\n\n## Next section\nother\n",
    )
    result = wiki_lint.run(wiki)
    assert result.by_check().get("index-honesty", []) == []


def test_index_honesty_stale_flagged(tmp_path):
    wiki = make_wiki(tmp_path)
    write(
        wiki / "log.md",
        "# Log\n\n## [2026-07-04] latest\nstuff\n\n## [2026-05-01] older\nstuff\n",
    )
    write(
        wiki / "index.md",
        "# Index\n\n**You are here.**\n- Now (2026-06-01): stale.\n\n## Next section\nother\n",
    )
    result = wiki_lint.run(wiki)
    findings = result.by_check().get("index-honesty", [])
    assert len(findings) == 1
    assert findings[0].severity == "notice"
    assert "stale" in findings[0].message


# --------------------------------------------------------------------------
# Exit code / summary
# --------------------------------------------------------------------------

def test_exit_code_nonzero_on_error(tmp_path, capsys):
    wiki = make_wiki(tmp_path)
    write(wiki / "topics" / "no_banner.md", "# No Banner\n\nProse.\n")
    write(wiki / "index.md", "# Index\n")
    write(wiki / "log.md", "# Log\n")
    code = wiki_lint.main(["--wiki", str(wiki)])
    assert code == 1
    out = capsys.readouterr().out
    assert "errors" in out


def test_exit_code_zero_when_clean(tmp_path, capsys):
    wiki = make_wiki(tmp_path)
    write(wiki / "topics" / "a.md", "# A\n\n> **Status: LIVE** *(2026-07-04)*\n")
    write(wiki / "index.md", "# Index\n\n[a](topics/a.md)\n")
    write(wiki / "log.md", "# Log\n")
    code = wiki_lint.main(["--wiki", str(wiki)])
    assert code == 0


def test_json_output(tmp_path, capsys):
    wiki = make_wiki(tmp_path)
    write(wiki / "topics" / "a.md", "# A\n\n> **Status: LIVE** *(2026-07-04)*\n")
    write(wiki / "index.md", "# Index\n\n[a](topics/a.md)\n")
    write(wiki / "log.md", "# Log\n")
    code = wiki_lint.main(["--wiki", str(wiki), "--json"])
    assert code == 0
    out = capsys.readouterr().out
    import json

    payload = json.loads(out)
    assert "summary" in payload
    assert "findings" in payload
