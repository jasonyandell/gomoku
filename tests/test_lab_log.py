"""Unit tests for the lab event log's pure logic."""
import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import lab_log as ll  # noqa: E402


def test_build_entry_shape():
    e = ll.build_entry("observation", "derby-v4", "vcf wins", "sid-1",
                       "2026-05-25T11:08:00-05:00", {"verdict": "vcf"})
    assert e["kind"] == "observation"
    assert e["scope"] == "derby-v4"
    assert e["session"] == "sid-1"
    assert e["data"] == {"verdict": "vcf"}


def test_parse_entry_skips_blank_and_comments():
    assert ll.parse_entry("") is None
    assert ll.parse_entry("  # a note") is None
    assert ll.parse_entry('{"kind":"event","ts":"x"}')["kind"] == "event"


def test_parse_since_units():
    assert ll.parse_since("30m") == dt.timedelta(minutes=30)
    assert ll.parse_since("12h") == dt.timedelta(hours=12)
    assert ll.parse_since("2d") == dt.timedelta(days=2)
    with pytest.raises(ValueError):
        ll.parse_since("soon")


def _mk(ts, kind="event", scope="hygiene", session="s"):
    return ll.build_entry(kind, scope, "x", session, ts, {})


def test_filter_by_kind_scope_session():
    es = [_mk("2026-05-25T10:00:00-05:00", kind="event", scope="hygiene"),
          _mk("2026-05-25T10:01:00-05:00", kind="observation", scope="derby-v4"),
          _mk("2026-05-25T10:02:00-05:00", kind="event", scope="derby-v4", session="z")]
    assert len(ll.filter_entries(es, kind="event")) == 2
    assert len(ll.filter_entries(es, scope="derby-v4")) == 2
    assert len(ll.filter_entries(es, kind="observation")) == 1
    assert len(ll.filter_entries(es, session="z")) == 1


def test_filter_by_cutoff():
    now = dt.datetime.now().astimezone()
    old = (now - dt.timedelta(days=3)).isoformat()
    recent = (now - dt.timedelta(hours=1)).isoformat()
    es = [_mk(old), _mk(recent)]
    cutoff = now - dt.timedelta(days=1)
    kept = ll.filter_entries(es, cutoff=cutoff)
    assert len(kept) == 1 and kept[0]["ts"] == recent


def test_format_row_has_glyph_and_ref():
    e = ll.build_entry("event", "workflow", "merged", "s",
                       "2026-05-25T11:08:00-05:00", {"ref": "@048b7e8"})
    row = ll.format_row(e)
    assert "⬤" in row and "workflow" in row and "merged" in row and "@048b7e8" in row
