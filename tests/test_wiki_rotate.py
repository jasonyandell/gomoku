"""Tests for scripts/wiki_rotate.py — the hardened journal-rotation tool.

Covers: basic split, reconcile-failure refusal, dry-run (no writes),
duplicate-archive refusal, malformed-heading refusal, preamble+pointer
preservation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "wiki_rotate.py"
spec = importlib.util.spec_from_file_location("wiki_rotate", SCRIPT_PATH)
wiki_rotate = importlib.util.module_from_spec(spec)
sys.modules["wiki_rotate"] = wiki_rotate
spec.loader.exec_module(wiki_rotate)


JOURNAL_TEXT = """# Test Journal

Some intro text describing this journal.

## [2026-05-10] First entry

Body of the first entry, from May.

## [2026-06-01] Second entry

Body of the second entry, from June.
Multiple lines here.

## [2026-07-04] Third entry

Body of the third entry, from July.
"""


def write_journal(tmp_path: Path, text: str = JOURNAL_TEXT) -> Path:
    p = tmp_path / "log.md"
    p.write_text(text, encoding="utf-8")
    return p


def test_basic_split(tmp_path):
    journal = write_journal(tmp_path)
    archive = tmp_path / "_archive" / "log-2026-06.md"

    rc = wiki_rotate.main(
        [str(journal), "--before", "2026-07", "--archive", str(archive), "--date", "2026-07-04"]
    )
    assert rc == 0

    live_text = journal.read_text(encoding="utf-8")
    archive_text = archive.read_text(encoding="utf-8")

    assert "First entry" not in live_text
    assert "Second entry" not in live_text
    assert "Third entry" in live_text

    assert "First entry" in archive_text
    assert "Second entry" in archive_text
    assert "Third entry" not in archive_text

    # Pointer line added to live preamble.
    assert "Older eras:" in live_text
    assert "log-2026-06.md" in live_text
    assert "2026-07-04" in live_text

    # Reconciliation: re-parse both and check counts/bytes match original.
    _, orig_entries, _ = wiki_rotate.split_preamble_and_entries(JOURNAL_TEXT)
    _, live_entries, _ = wiki_rotate.split_preamble_and_entries(live_text)
    _, archive_entries, _ = wiki_rotate.split_preamble_and_entries(archive_text)

    orig_totals = wiki_rotate.ReconcileTotals.of(orig_entries)
    live_totals = wiki_rotate.ReconcileTotals.of(live_entries)
    archive_totals = wiki_rotate.ReconcileTotals.of(archive_entries)

    assert live_totals.count + archive_totals.count == orig_totals.count
    assert live_totals.bytes_ + archive_totals.bytes_ == orig_totals.bytes_


def test_dry_run_writes_nothing(tmp_path):
    journal = write_journal(tmp_path)
    archive = tmp_path / "_archive" / "log-2026-06.md"
    original_text = journal.read_text(encoding="utf-8")

    rc = wiki_rotate.main(
        [str(journal), "--before", "2026-07", "--archive", str(archive), "--dry-run"]
    )
    assert rc == 0
    assert not archive.exists()
    assert journal.read_text(encoding="utf-8") == original_text


def test_duplicate_archive_refusal(tmp_path):
    journal = write_journal(tmp_path)
    archive = tmp_path / "_archive" / "log-2026-06.md"
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Pre-seed the archive with an entry that will collide with a rotation.
    archive.write_text(
        "# Archive\n\nRotated out already.\n\n"
        "## [2026-05-10] First entry\n\nBody of the first entry, from May.\n",
        encoding="utf-8",
    )
    original_archive_text = archive.read_text(encoding="utf-8")
    original_journal_text = journal.read_text(encoding="utf-8")

    rc = wiki_rotate.main(
        [str(journal), "--before", "2026-07", "--archive", str(archive)]
    )
    assert rc != 0
    # Refused: nothing written.
    assert archive.read_text(encoding="utf-8") == original_archive_text
    assert journal.read_text(encoding="utf-8") == original_journal_text


def test_malformed_heading_refusal(tmp_path):
    bad_text = JOURNAL_TEXT + "\n## [not-a-date] Oops\n\nMalformed heading body.\n"
    journal = write_journal(tmp_path, bad_text)
    archive = tmp_path / "_archive" / "log-2026-06.md"

    rc = wiki_rotate.main(
        [str(journal), "--before", "2026-07", "--archive", str(archive)]
    )
    assert rc != 0
    assert not archive.exists()
    # Original journal untouched.
    assert journal.read_text(encoding="utf-8") == bad_text


def test_preamble_and_pointer_preserved(tmp_path):
    text_with_pointer = (
        "# Test Journal\n\n"
        "Intro text.\n\n"
        "Older eras: [2026-04 archive](_archive/log-2026-04.md) (rotated out 2026-06-01).\n\n"
        "## [2026-05-10] First entry\n\nBody one.\n\n"
        "## [2026-07-04] Second entry\n\nBody two.\n"
    )
    journal = write_journal(tmp_path, text_with_pointer)
    archive = tmp_path / "_archive" / "log-2026-06.md"

    rc = wiki_rotate.main(
        [str(journal), "--before", "2026-07", "--archive", str(archive), "--date", "2026-07-04"]
    )
    assert rc == 0

    live_text = journal.read_text(encoding="utf-8")
    assert "Intro text." in live_text
    # Existing pointer extended, not clobbered.
    assert "log-2026-04.md" in live_text
    assert "log-2026-06.md" in live_text
    assert "Second entry" in live_text
    assert "First entry" not in live_text


def test_reconcile_failure_refusal(tmp_path, monkeypatch):
    """Simulate a reconciliation bug by corrupting the kept-entries list after
    the split so live+archive no longer equals the original — the tool must
    detect this and refuse to write."""
    journal = write_journal(tmp_path)
    archive = tmp_path / "_archive" / "log-2026-06.md"

    original_of = wiki_rotate.ReconcileTotals.of

    call_count = {"n": 0}

    def corrupting_of(entries):
        call_count["n"] += 1
        totals = original_of(entries)
        # Corrupt only the "kept" totals computation (the 2nd call in main())
        # so the reconciliation arithmetic sees a mismatch, without affecting
        # the initial "original_totals" computation (the 1st call).
        if call_count["n"] == 2:
            return wiki_rotate.ReconcileTotals(count=totals.count, bytes_=totals.bytes_ + 1)
        return totals

    monkeypatch.setattr(wiki_rotate, "ReconcileTotals", wiki_rotate.ReconcileTotals)
    monkeypatch.setattr(wiki_rotate.ReconcileTotals, "of", staticmethod(corrupting_of))

    original_journal_text = journal.read_text(encoding="utf-8")

    rc = wiki_rotate.main(
        [str(journal), "--before", "2026-07", "--archive", str(archive)]
    )
    assert rc != 0
    assert not archive.exists()
    assert journal.read_text(encoding="utf-8") == original_journal_text


def test_no_entries_before_cutoff_is_noop(tmp_path):
    journal = write_journal(tmp_path)
    archive = tmp_path / "_archive" / "log-1999.md"
    original_text = journal.read_text(encoding="utf-8")

    rc = wiki_rotate.main(
        [str(journal), "--before", "2000-01", "--archive", str(archive)]
    )
    assert rc == 0
    assert not archive.exists()
    assert journal.read_text(encoding="utf-8") == original_text


def test_day_level_cutoff(tmp_path):
    journal = write_journal(tmp_path)
    archive = tmp_path / "_archive" / "log-day.md"

    rc = wiki_rotate.main(
        [str(journal), "--before", "2026-07-04", "--archive", str(archive)]
    )
    assert rc == 0
    live_text = journal.read_text(encoding="utf-8")
    archive_text = archive.read_text(encoding="utf-8")
    assert "Third entry" in live_text  # 2026-07-04 not strictly before 2026-07-04
    assert "First entry" in archive_text
    assert "Second entry" in archive_text
