"""Mechanized half of the lint list in wiki/curation.md § Lint.

Runs the checks that can be fully automated:

  1. BANNERS      — every wiki/topics/*.md carries a dated status marker
                    (LIVE / HISTORICAL / SUPERSEDED-BY(...) / DORMANT /
                    DESIGN-NEVER-BUILT / DEAD-END) near the top of the page.
  2. LINKS        — every relative *.md link from a live page resolves to a
                    real file (archived pages under wiki/_archive/ are exempt
                    as sources, but a live page's link INTO the archive is
                    still checked for existence).
  3. ORPHANS      — every topics/ page has at least one inbound link from
                    somewhere outside wiki/_archive/ (errors: none at all;
                    warnings: only reachable via other topic pages, never a
                    hub/index/ops/workflow page).
  4. SIZE/ROTATION — log.md / ops/*.md over the ~60 KB smell threshold
                    (skipping anything marked ARCHIVED or FROZEN in its head);
                    topics/*.md over the ~25 KB chronicle-itis trigger.
  5. INDEX HONESTY — the newest `## [YYYY-MM-DD]` entry in log.md compared
                    against the newest date mentioned in index.md's
                    "You are here" section; >14 days apart is a staleness
                    warning.

NOT mechanized here (see wiki/curation.md § Lint for these — they need a
reader's judgment, not a regex):
  - Staleness of VOICE: pages presenting a stopped system in the present
    tense (the janitor/worktree-hygiene incident). A banner can be dated and
    present and the prose can still read as if the system is running.
  - Duplication: the same finding narrated in 2+ pages instead of one
    canonical home + pointers (curation.md rule 3).
  - Query gaps: questions that recurred at >1 fetch because the page that
    would answer them in one fetch doesn't exist yet.

Usage:
    uv run python scripts/wiki_lint.py [--wiki PATH] [--json]

Exit code: 1 if any errors were found, else 0 (warnings/notices don't fail).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Thresholds (wiki/curation.md § Rotation thresholds)
# --------------------------------------------------------------------------

LOG_OPS_SMELL_BYTES = 60 * 1024  # "~60 KB is the smell threshold"
TOPIC_CHRONICLE_BYTES = 25 * 1024  # "> ~25 KB ... run the chronicle-itis check"
INDEX_STALE_DAYS = 14

# The five hard-rule banner markers (curation.md rule 2), plus the
# any-form SUPERSEDED-BY the task calls out explicitly. Word-boundaried so
# "LIVE" doesn't fire inside unrelated words.
BANNER_MARKER_RE = re.compile(
    r"\b(LIVE|HISTORICAL|SUPERSEDED[- ]BY|DORMANT|DESIGN-NEVER-BUILT|DEAD-END)\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
BANNER_WINDOW_LINES = 8  # "within its first ~8 lines"

ARCHIVED_MARKER_RE = re.compile(r"\b(ARCHIVED|FROZEN)\b", re.IGNORECASE)
ARCHIVED_HEAD_LINES = 10

MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+\.md)(#[^)]*)?\)")

LOG_DATE_HEADER_RE = re.compile(r"^##\s*\[(\d{4}-\d{2}-\d{2})\]", re.MULTILINE)


@dataclass
class Finding:
    check: str
    severity: str  # "error" | "warning" | "notice"
    message: str
    file: str | None = None
    line: int | None = None

    def location(self) -> str:
        if self.file is None:
            return ""
        if self.line is None:
            return f"{self.file}: "
        return f"{self.file}:{self.line}: "

    def render(self) -> str:
        return f"[{self.severity.upper():7}] {self.location()}{self.message}"


@dataclass
class LintResult:
    findings: list[Finding] = field(default_factory=list)

    def add(self, check: str, severity: str, message: str, file: str | None = None, line: int | None = None) -> None:
        self.findings.append(Finding(check=check, severity=severity, message=message, file=file, line=line))

    def by_check(self) -> dict[str, list[Finding]]:
        out: dict[str, list[Finding]] = {}
        for f in self.findings:
            out.setdefault(f.check, []).append(f)
        return out

    def counts(self) -> dict[str, int]:
        counts = {"error": 0, "warning": 0, "notice": 0}
        for f in self.findings:
            counts[f.severity] += 1
        return counts


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# 1. Banners
# --------------------------------------------------------------------------

def check_banners(wiki: Path, result: LintResult) -> None:
    for f in sorted((wiki / "topics").glob("*.md")):
        lines = f.read_text(encoding="utf-8").splitlines()
        window = lines[:BANNER_WINDOW_LINES]
        block = "\n".join(window)
        marker = BANNER_MARKER_RE.search(block)
        label = rel(f, wiki)
        if not marker:
            result.add(
                "banners", "error",
                "no status banner found in first "
                f"{BANNER_WINDOW_LINES} lines (expected LIVE / HISTORICAL / "
                "SUPERSEDED-BY(...) / DORMANT / DESIGN-NEVER-BUILT / DEAD-END)",
                file=label, line=1,
            )
            continue
        if not DATE_RE.search(block):
            marker_line = block[: marker.start()].count("\n") + 1
            result.add(
                "banners", "error",
                f"status banner ({marker.group(0)!r}) has no YYYY-MM-DD date "
                f"in the first {BANNER_WINDOW_LINES} lines",
                file=label, line=marker_line,
            )


# --------------------------------------------------------------------------
# 2. Links
# --------------------------------------------------------------------------

def iter_live_markdown_files(wiki: Path):
    for f in sorted(wiki.rglob("*.md")):
        if "_archive" in f.relative_to(wiki).parts:
            continue
        yield f


def check_links(wiki: Path, result: LintResult) -> dict[Path, set[Path]]:
    """Returns a map of live file -> set of resolved .md targets it links to
    (used later by the orphan check)."""
    outbound: dict[Path, set[Path]] = {}
    for f in iter_live_markdown_files(wiki):
        targets: set[Path] = set()
        text = f.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in MD_LINK_RE.finditer(line):
                target_str = m.group(1)
                if target_str.startswith(("http://", "https://")):
                    continue
                resolved = (f.parent / target_str).resolve()
                targets.add(resolved)
                if not resolved.exists():
                    result.add(
                        "links", "error",
                        f"broken link -> {target_str}",
                        file=rel(f, wiki), line=lineno,
                    )
        outbound[f] = targets
    return outbound


# --------------------------------------------------------------------------
# 3. Orphans
# --------------------------------------------------------------------------

HUB_LIKE_SUFFIXES = {"topics"}  # a page is "topic-only" if its parent dir is topics/


def check_orphans(wiki: Path, outbound: dict[Path, set[Path]], result: LintResult) -> None:
    topics_dir = (wiki / "topics").resolve()
    all_topics = {p.resolve() for p in topics_dir.glob("*.md")}

    inbound_from_nontopic: dict[Path, int] = {t: 0 for t in all_topics}
    inbound_from_topic: dict[Path, int] = {t: 0 for t in all_topics}

    for src, targets in outbound.items():
        src_is_topic = src.resolve().parent == topics_dir
        for t in targets:
            if t not in all_topics or t == src.resolve():
                continue
            if src_is_topic:
                inbound_from_topic[t] += 1
            else:
                inbound_from_nontopic[t] += 1

    for t in sorted(all_topics):
        label = rel(t, wiki)
        has_nontopic = inbound_from_nontopic[t] > 0
        has_topic = inbound_from_topic[t] > 0
        if not has_nontopic and not has_topic:
            result.add(
                "orphans", "warning",
                "no inbound links from any live page (hub, index, ops, "
                "workflow, or sibling topic) — unreachable except by direct path",
                file=label,
            )
        elif not has_nontopic:
            result.add(
                "orphans", "warning",
                "reachable only via other topic pages — no hub/index/ops/"
                "workflow page links to it",
                file=label,
            )


# --------------------------------------------------------------------------
# 4. Size / rotation triggers
# --------------------------------------------------------------------------

def has_archived_marker(f: Path) -> bool:
    import itertools

    with f.open(encoding="utf-8") as fh:
        head = "".join(itertools.islice(fh, ARCHIVED_HEAD_LINES))
    return bool(ARCHIVED_MARKER_RE.search(head))


def check_sizes(wiki: Path, result: LintResult) -> None:
    rotation_candidates = [wiki / "log.md", *sorted((wiki / "ops").glob("*.md"))]
    for f in rotation_candidates:
        if not f.exists():
            continue
        size = f.stat().st_size
        if size <= LOG_OPS_SMELL_BYTES:
            continue
        if has_archived_marker(f):
            continue
        result.add(
            "rotation", "warning",
            f"{size / 1024:.1f} KB > ~60 KB smell threshold — consider rotating "
            "a closed era to _archive/ (curation.md § Rotation thresholds)",
            file=rel(f, wiki),
        )

    for f in sorted((wiki / "topics").glob("*.md")):
        size = f.stat().st_size
        if size > TOPIC_CHRONICLE_BYTES:
            result.add(
                "chronicle-itis", "notice",
                f"{size / 1024:.1f} KB > ~25 KB — chronicle-itis check is due "
                "(a trigger to review, not a violation; legitimate reference/"
                "verdict-first pages can stay big)",
                file=rel(f, wiki),
            )


# --------------------------------------------------------------------------
# 5. Index honesty
# --------------------------------------------------------------------------

def newest_log_date(wiki: Path) -> str | None:
    log = wiki / "log.md"
    if not log.exists():
        return None
    dates = LOG_DATE_HEADER_RE.findall(log.read_text(encoding="utf-8"))
    return max(dates) if dates else None


def newest_index_you_are_here_date(wiki: Path) -> str | None:
    index = wiki / "index.md"
    if not index.exists():
        return None
    text = index.read_text(encoding="utf-8")
    marker = re.search(r"you are here", text, re.IGNORECASE)
    if not marker:
        return None
    # Take the section from the marker to the next '##' heading (or EOF).
    rest = text[marker.start():]
    next_heading = re.search(r"\n##\s", rest)
    section = rest[: next_heading.start()] if next_heading else rest
    dates = DATE_RE.findall(section)
    return max(dates) if dates else None


def check_index_honesty(wiki: Path, result: LintResult) -> None:
    log_date = newest_log_date(wiki)
    index_date = newest_index_you_are_here_date(wiki)
    if log_date is None or index_date is None:
        return
    from datetime import date

    log_d = date.fromisoformat(log_date)
    index_d = date.fromisoformat(index_date)
    gap = (log_d - index_d).days
    if gap > INDEX_STALE_DAYS:
        result.add(
            "index-honesty", "notice",
            f"index.md § You are here newest date is {index_date}, but log.md's "
            f"newest entry is {log_date} ({gap} days newer) — index may be stale",
            file=rel(wiki / "index.md", wiki),
        )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

CHECK_TITLES = {
    "banners": "BANNERS",
    "links": "LINKS",
    "orphans": "ORPHANS",
    "rotation": "SIZE / ROTATION (rotation smell)",
    "chronicle-itis": "SIZE / ROTATION (chronicle-itis check)",
    "index-honesty": "INDEX HONESTY",
}


def render_report(result: LintResult) -> str:
    lines = []
    by_check = result.by_check()
    for key, title in CHECK_TITLES.items():
        findings = by_check.get(key, [])
        lines.append(f"## {title}")
        if not findings:
            lines.append("  (clean)")
        else:
            for f in findings:
                lines.append("  " + f.render())
        lines.append("")

    counts = result.counts()
    lines.append(
        f"SUMMARY: {counts['error']} errors, {counts['warning']} warnings, "
        f"{counts['notice']} notices"
    )
    return "\n".join(lines)


def render_json(result: LintResult) -> str:
    counts = result.counts()
    payload = {
        "summary": counts,
        "findings": [
            {
                "check": f.check,
                "severity": f.severity,
                "file": f.file,
                "line": f.line,
                "message": f.message,
            }
            for f in result.findings
        ],
    }
    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run(wiki: Path) -> LintResult:
    result = LintResult()
    check_banners(wiki, result)
    outbound = check_links(wiki, result)
    check_orphans(wiki, outbound, result)
    check_sizes(wiki, result)
    check_index_honesty(wiki, result)
    return result


def default_wiki_path() -> Path:
    # scripts/wiki_lint.py -> repo root -> wiki/
    return Path(__file__).resolve().parent.parent / "wiki"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--wiki", type=Path, default=None,
        help="path to the wiki/ directory (default: wiki/ next to this repo)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    wiki = (args.wiki or default_wiki_path()).resolve()
    if not wiki.is_dir():
        print(f"error: wiki directory not found: {wiki}", file=sys.stderr)
        return 2

    result = run(wiki)

    if args.json:
        print(render_json(result))
    else:
        print(render_report(result))

    return 1 if result.counts()["error"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
