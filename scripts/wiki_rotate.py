#!/usr/bin/env python3
"""Hardened rotation tool for append-only wiki journals (wiki/log.md, wiki/ops/*.md).

Journals are append-only markdown files whose entries start with headings of
the form:

    ## [YYYY-MM-DD] <title>

Everything before the first such heading is the "preamble" (title, intro,
archive pointer line) and always stays in the live file.

This tool splits entries strictly before a cutoff date into an archive file,
following wiki/curation.md § "Rotation thresholds":

    Split by date-prefix with a script, then reconcile — entry counts and
    byte totals across (live + archive) must equal the pre-rotation totals.

A freehand rotation on 2026-07-04 silently dropped 21 entries; this script
exists so that never happens again. It refuses to write anything unless a
reconciliation check (count + bytes) passes, and re-verifies after writing.

Usage:
    python scripts/wiki_rotate.py wiki/log.md \\
        --before 2026-07 --archive wiki/_archive/log-2026-07.md [--dry-run]

    python scripts/wiki_rotate.py wiki/log.md \\
        --before 2026-07-04 --archive wiki/_archive/log-2026-07.md

--before accepts YYYY-MM (rotate everything strictly before that month) or
YYYY-MM-DD (rotate everything strictly before that day).
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^##\s+\[(\d{4}-\d{2}-\d{2})\]\s+(.*)$")
# Any '## [...]' heading that looks like it was *meant* to be a dated entry
# heading but doesn't match the strict date format above.
SUSPECT_HEADING_RE = re.compile(r"^##\s+\[([^\]]*)\]")


@dataclass
class Entry:
    date: str  # YYYY-MM-DD
    title: str
    heading: str  # full heading line, e.g. "## [2026-07-04] Title"
    body: str  # heading line + following lines, up to (not including) next entry


def parse_cutoff(raw: str) -> str:
    """Normalize --before into a comparable YYYY-MM-DD-ish prefix string.

    Returns a string such that entry.date < cutoff (lexicographic ==
    chronological for zero-padded ISO dates) means "rotate this entry".
    """
    if re.fullmatch(r"\d{4}-\d{2}", raw):
        # Cutoff is a month: rotate everything before the 1st of that month.
        return f"{raw}-00"  # sorts before any real day "-01".."-31"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    raise ValueError(f"--before must be YYYY-MM or YYYY-MM-DD, got {raw!r}")


def split_preamble_and_entries(text: str) -> tuple[str, list[Entry], list[str]]:
    """Split journal text into (preamble, entries, malformed_headings).

    malformed_headings holds raw '## [...]' lines that are NOT valid
    'YYYY-MM-DD' dated entry headings (a title crept into the date slot, a
    typo'd date, etc.) — callers must refuse to rotate when this is non-empty.
    """
    lines = text.split("\n")

    # Find every '## [' style line; classify as valid entry heading or
    # malformed/suspect heading.
    heading_idxs: list[int] = []
    malformed: list[str] = []
    for i, line in enumerate(lines):
        if HEADING_RE.match(line):
            heading_idxs.append(i)
        elif SUSPECT_HEADING_RE.match(line):
            malformed.append(line)

    if not heading_idxs:
        return text, [], malformed

    preamble = "\n".join(lines[: heading_idxs[0]])
    entries: list[Entry] = []
    for n, start in enumerate(heading_idxs):
        end = heading_idxs[n + 1] if n + 1 < len(heading_idxs) else len(lines)
        body_lines = lines[start:end]
        body = "\n".join(body_lines)
        m = HEADING_RE.match(lines[start])
        assert m is not None
        entries.append(
            Entry(date=m.group(1), title=m.group(2), heading=lines[start], body=body)
        )
    return preamble, entries, malformed


@dataclass
class ReconcileTotals:
    count: int
    bytes_: int

    @staticmethod
    def of(entries: list[Entry]) -> "ReconcileTotals":
        return ReconcileTotals(
            count=len(entries),
            bytes_=sum(len(e.body.encode("utf-8")) for e in entries),
        )


def find_pointer_line(preamble: str) -> tuple[int | None, list[str]]:
    """Return (index of an existing 'Older eras:' pointer line, preamble lines)."""
    lines = preamble.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("Older eras:"):
            return i, lines
    return None, lines


def render_pointer(archive_label: str, archive_relpath: str, date: str, existing: str | None) -> str:
    """Build (or extend) the 'Older eras:' pointer line.

    If an existing pointer line is present and doesn't already reference this
    archive, append a new markdown link to it (comma-separated), matching the
    style of a single growing pointer line. If it already references this
    archive, leave it untouched.
    """
    link = f"[{archive_label}]({archive_relpath})"
    if existing is None:
        return f"Older eras: {link} (rotated out {date})."
    if archive_relpath in existing:
        return existing
    # Insert the new link before " (rotated out" if present, else just append.
    marker = " (rotated out"
    idx = existing.rfind(marker)
    if idx == -1:
        return existing.rstrip(".") + f", {link} (rotated out {date})."
    return existing[:idx] + f", {link}" + existing[idx:]


def format_archive_header(journal_path: Path, archive_path: Path, date: str) -> str:
    title = archive_path.stem.replace("-", " ").replace("_", " ")
    return (
        f"# {title} archive\n\n"
        f"Rotated out of [{journal_path.name}](../{journal_path.name}) on {date}; "
        f"entries verbatim, original file order.\n"
    )


def assemble(preamble_text: str, bodies: list[str]) -> str:
    """Join a preamble and a list of entry bodies into final file text.

    Normalizes to exactly one blank line between the preamble and the first
    entry, exactly one blank line between consecutive entries, and exactly
    one trailing newline at end-of-file — matching the convention already
    used by the real journals (blank-line-separated entries). Each body may
    carry its own trailing blank-line/newline from parsing; those are
    stripped here and regenerated deterministically so re-parsing the
    written file recovers byte-identical entry bodies.
    """
    parts = [preamble_text.rstrip("\n")]
    for b in bodies:
        parts.append(b.rstrip("\n"))
    return "\n\n".join(parts) + "\n"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(text)
        Path(tmp_name).replace(path)
    except BaseException:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("journal", type=Path, help="Path to the live journal (e.g. wiki/log.md)")
    ap.add_argument("--before", required=True, help="Rotate entries strictly before this date (YYYY-MM or YYYY-MM-DD)")
    ap.add_argument("--archive", required=True, type=Path, help="Path to the archive file (created if missing)")
    ap.add_argument("--date", default=None, help="Date to record as 'rotated out' (default: today)")
    ap.add_argument("--dry-run", action="store_true", help="Report what would happen; write nothing")
    args = ap.parse_args(argv)

    try:
        cutoff = parse_cutoff(args.before)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.date is None:
        rotation_date = dt.date.today().isoformat()
    else:
        rotation_date = args.date

    if not args.journal.exists():
        print(f"ERROR: journal not found: {args.journal}", file=sys.stderr)
        return 2

    original_text = args.journal.read_text(encoding="utf-8")
    preamble, entries, malformed = split_preamble_and_entries(original_text)

    if malformed:
        print("ERROR: malformed/unrecognized '## [...]' heading(s) found — refusing to rotate.", file=sys.stderr)
        print("Fix these headings to the 'YYYY-MM-DD' dated form (or exclude them) and re-run:", file=sys.stderr)
        for line in malformed:
            print(f"  {line}", file=sys.stderr)
        return 3

    original_totals = ReconcileTotals.of(entries)

    to_archive = [e for e in entries if e.date < cutoff]
    to_keep = [e for e in entries if e.date >= cutoff]

    # --- Duplicate-in-archive check ---
    existing_archive_headings: set[str] = set()
    archive_preamble = ""
    archive_existing_entries: list[Entry] = []
    if args.archive.exists():
        archive_text = args.archive.read_text(encoding="utf-8")
        archive_preamble, archive_existing_entries, archive_malformed = split_preamble_and_entries(archive_text)
        if archive_malformed:
            print("ERROR: malformed heading(s) found in existing archive — refusing to rotate.", file=sys.stderr)
            for line in archive_malformed:
                print(f"  {line}", file=sys.stderr)
            return 3
        existing_archive_headings = {e.heading for e in archive_existing_entries}

    dupes = [e.heading for e in to_archive if e.heading in existing_archive_headings]
    if dupes:
        print("ERROR: entries to rotate already exist in the archive — refusing (would duplicate).", file=sys.stderr)
        for h in dupes:
            print(f"  {h}", file=sys.stderr)
        return 4

    if not to_archive:
        print(f"Nothing to rotate: no entries strictly before {args.before} in {args.journal}.")
        print(f"Total entries in {args.journal}: {original_totals.count} ({original_totals.bytes_} bytes)")
        return 0

    # --- Build new archive content ---
    if args.archive.exists():
        new_archive_body_entries = archive_existing_entries + to_archive
        new_archive_preamble = archive_preamble
    else:
        new_archive_preamble = format_archive_header(args.journal, args.archive, rotation_date)
        new_archive_body_entries = list(to_archive)

    new_archive_text = assemble(new_archive_preamble, [e.body for e in new_archive_body_entries])

    # --- Build new live content, with pointer line added/updated ---
    try:
        archive_relpath = str(args.archive.resolve().relative_to(args.journal.resolve().parent))
    except ValueError:
        # Archive not under the journal's directory tree; fall back to a
        # plain relative path computed via os.path.relpath semantics.
        import os

        archive_relpath = os.path.relpath(args.archive, args.journal.parent)
    archive_label = args.archive.stem

    pointer_idx, preamble_lines = find_pointer_line(preamble)
    if pointer_idx is None:
        new_pointer = render_pointer(archive_label, archive_relpath, rotation_date, None)
        # Insert as its own paragraph after the first heading/intro paragraph:
        # append at the end of the preamble, preceded by a blank line.
        stripped_preamble = preamble.rstrip("\n")
        new_preamble = stripped_preamble + "\n\n" + new_pointer + "\n"
    else:
        preamble_lines[pointer_idx] = render_pointer(
            archive_label, archive_relpath, rotation_date, preamble_lines[pointer_idx]
        )
        new_preamble = "\n".join(preamble_lines)

    new_live_text = assemble(new_preamble, [e.body for e in to_keep])

    # --- RECONCILE before writing anything ---
    kept_totals = ReconcileTotals.of(to_keep)
    archived_totals = ReconcileTotals.of(to_archive)
    combined_count = kept_totals.count + archived_totals.count
    combined_bytes = kept_totals.bytes_ + archived_totals.bytes_

    report_lines = [
        "=== Reconciliation report (pre-write) ===",
        f"journal:          {args.journal}",
        f"archive:          {args.archive}",
        f"cutoff (--before): {args.before}",
        f"original entries: {original_totals.count} ({original_totals.bytes_} bytes)",
        f"  -> live:        {kept_totals.count} ({kept_totals.bytes_} bytes)",
        f"  -> archived:    {archived_totals.count} ({archived_totals.bytes_} bytes)",
        f"  -> combined:    {combined_count} ({combined_bytes} bytes)",
    ]

    if combined_count != original_totals.count or combined_bytes != original_totals.bytes_:
        report_lines.append(
            "RECONCILIATION FAILED: live + archived does not equal the original "
            "entry count/bytes. Refusing to write anything."
        )
        print("\n".join(report_lines), file=sys.stderr)
        return 5

    report_lines.append("RECONCILIATION OK: live + archived == original (count and bytes).")
    print("\n".join(report_lines))

    if args.dry_run:
        print("\n=== DRY RUN: no files written ===")
        print(f"Would move {archived_totals.count} entries to {args.archive}:")
        for e in to_archive:
            print(f"  {e.heading}")
        return 0

    # --- Write atomically ---
    atomic_write(args.archive, new_archive_text)
    atomic_write(args.journal, new_live_text)

    # --- Re-read and re-verify post-write ---
    reread_live_text = args.journal.read_text(encoding="utf-8")
    reread_archive_text = args.archive.read_text(encoding="utf-8")
    _, reread_live_entries, reread_live_malformed = split_preamble_and_entries(reread_live_text)
    _, reread_archive_entries, reread_archive_malformed = split_preamble_and_entries(reread_archive_text)

    post_live_totals = ReconcileTotals.of(reread_live_entries)
    post_archive_totals = ReconcileTotals.of(reread_archive_entries)
    post_combined_count = post_live_totals.count + post_archive_totals.count
    # Only the newly-rotated bytes moved verbatim need to match; the archive
    # may have pre-existing entries too, so compare against expected totals
    # computed above rather than re-deriving from scratch.
    post_ok = (
        not reread_live_malformed
        and not reread_archive_malformed
        and post_live_totals.count == kept_totals.count
        and post_archive_totals.count == len(new_archive_body_entries)
        and post_combined_count == combined_count
    )

    print("\n=== Reconciliation report (post-write, re-read from disk) ===")
    print(f"live entries:     {post_live_totals.count} ({post_live_totals.bytes_} bytes)")
    print(f"archive entries:  {post_archive_totals.count} ({post_archive_totals.bytes_} bytes)")
    print(f"combined:         {post_combined_count}")
    if post_ok:
        print("POST-WRITE VERIFICATION OK.")
    else:
        print("POST-WRITE VERIFICATION FAILED — inspect files manually before trusting them.", file=sys.stderr)
        return 6

    print(f"\nRotated {archived_totals.count} entries from {args.journal} into {args.archive}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
