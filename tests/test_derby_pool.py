"""Unit tests for derby_pool — the open-entry derby candidate pool.

GPU-free: every test uses a throwaway tmp_path pool dir (no real cells, no
training). Covers register (+ duplicate-name error), list + status filter,
claim (named + oldest-available), the DOUBLE-CLAIM guard, claim-when-empty,
retire, and on-disk atomicity (every candidate file is always valid JSON).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import derby_pool as dp  # noqa: E402


# ---- helpers --------------------------------------------------------------
def _assert_all_valid_json(pool_dir):
    """No candidate file is ever left half-written: each parses as a dict with
    the required keys, and no stray .tmp / .claiming markers linger."""
    cdir = pool_dir / "candidates"
    if not cdir.exists():
        return
    for p in cdir.iterdir():
        # Only canonical <name>.json files should remain after an op completes.
        assert p.name.endswith(".json"), f"stray marker left behind: {p.name}"
        spec = json.loads(p.read_text())  # raises if half-written
        assert isinstance(spec, dict)
        for key in ("name", "cell", "lever", "status", "submitted_at"):
            assert key in spec


# ---- register -------------------------------------------------------------
def test_register_creates_available(tmp_path):
    spec = dp.register("vct-teacher", "derby-x-vct", "+ VCT teacher signal",
                       by="researcher:42", pool_dir=tmp_path)
    assert spec["name"] == "vct-teacher"
    assert spec["cell"] == "derby-x-vct"
    assert spec["lever"] == "+ VCT teacher signal"
    assert spec["status"] == "available"
    assert spec["submitted_by"] == "researcher:42"
    assert isinstance(spec["submitted_at"], int)
    assert spec["claimed_at"] is None
    assert spec["retired_at"] is None
    assert spec["retire_reason"] is None
    _assert_all_valid_json(tmp_path)


def test_register_default_by(tmp_path):
    spec = dp.register("a", "cell-a", "lever a", pool_dir=tmp_path)
    assert spec["submitted_by"] == "unknown"


def test_register_duplicate_name_raises(tmp_path):
    dp.register("dup", "cell-a", "lever", pool_dir=tmp_path)
    try:
        dp.register("dup", "cell-b", "other lever", pool_dir=tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on duplicate name")
    # The original record must be untouched by the failed re-register.
    rows = dp.list_candidates(pool_dir=tmp_path)
    assert len(rows) == 1
    assert rows[0]["cell"] == "cell-a"
    _assert_all_valid_json(tmp_path)


# ---- list + status filter -------------------------------------------------
def test_list_and_status_filter(tmp_path):
    assert dp.list_candidates(pool_dir=tmp_path) == []
    dp.register("a", "cell-a", "la", pool_dir=tmp_path)
    dp.register("b", "cell-b", "lb", pool_dir=tmp_path)
    dp.register("c", "cell-c", "lc", pool_dir=tmp_path)

    all_rows = dp.list_candidates(pool_dir=tmp_path)
    assert {r["name"] for r in all_rows} == {"a", "b", "c"}
    assert len(dp.list_candidates(status="available", pool_dir=tmp_path)) == 3
    assert dp.list_candidates(status="running", pool_dir=tmp_path) == []

    dp.claim(name="b", pool_dir=tmp_path)
    dp.retire("c", "obsolete", pool_dir=tmp_path)
    assert {r["name"] for r in dp.list_candidates(status="available", pool_dir=tmp_path)} == {"a"}
    assert {r["name"] for r in dp.list_candidates(status="running", pool_dir=tmp_path)} == {"b"}
    assert {r["name"] for r in dp.list_candidates(status="retired", pool_dir=tmp_path)} == {"c"}
    _assert_all_valid_json(tmp_path)


# ---- claim ----------------------------------------------------------------
def test_claim_named(tmp_path):
    dp.register("a", "cell-a", "la", pool_dir=tmp_path)
    dp.register("b", "cell-b", "lb", pool_dir=tmp_path)
    spec = dp.claim(name="b", pool_dir=tmp_path)
    assert spec is not None
    assert spec["name"] == "b"
    assert spec["status"] == "running"
    assert isinstance(spec["claimed_at"], int)
    # 'a' stays available
    assert dp.list_candidates(status="available", pool_dir=tmp_path)[0]["name"] == "a"
    _assert_all_valid_json(tmp_path)


def test_claim_oldest_available(tmp_path):
    # Register with explicitly increasing submitted_at so "oldest" is deterministic.
    import time as _t
    s_a = dp.register("a", "cell-a", "la", pool_dir=tmp_path)
    s_b = dp.register("b", "cell-b", "lb", pool_dir=tmp_path)
    # If the clock didn't tick between registers, force a strict ordering on disk.
    if s_a["submitted_at"] == s_b["submitted_at"]:
        path_b = tmp_path / "candidates" / "b.json"
        rec = json.loads(path_b.read_text())
        rec["submitted_at"] = s_a["submitted_at"] + 5
        dp.atomic_write_json(path_b, rec)
    _t.sleep(0)  # no-op; ordering above is now explicit

    spec = dp.claim(pool_dir=tmp_path)  # name omitted -> oldest available
    assert spec is not None
    assert spec["name"] == "a"
    assert spec["status"] == "running"
    # The next oldest is now 'b'
    spec2 = dp.claim(pool_dir=tmp_path)
    assert spec2["name"] == "b"
    _assert_all_valid_json(tmp_path)


def test_claim_empty_returns_none(tmp_path):
    assert dp.claim(pool_dir=tmp_path) is None
    assert dp.claim(name="nope", pool_dir=tmp_path) is None


def test_claim_unknown_name_returns_none(tmp_path):
    dp.register("a", "cell-a", "la", pool_dir=tmp_path)
    assert dp.claim(name="ghost", pool_dir=tmp_path) is None


def test_double_claim_guard(tmp_path):
    """Two brokers race the same candidate -> exactly one wins."""
    dp.register("a", "cell-a", "la", pool_dir=tmp_path)
    first = dp.claim(name="a", pool_dir=tmp_path)
    assert first is not None and first["status"] == "running"
    second = dp.claim(name="a", pool_dir=tmp_path)
    assert second is None, "second claim of an already-running candidate must be None"
    # Still exactly one running 'a', and the file is intact.
    running = dp.list_candidates(status="running", pool_dir=tmp_path)
    assert [r["name"] for r in running] == ["a"]
    _assert_all_valid_json(tmp_path)


def test_oldest_claim_skips_already_running(tmp_path):
    """Anonymous claim must not re-hand-out a running candidate."""
    dp.register("a", "cell-a", "la", pool_dir=tmp_path)
    dp.register("b", "cell-b", "lb", pool_dir=tmp_path)
    dp.claim(name="a", pool_dir=tmp_path)  # a -> running
    spec = dp.claim(pool_dir=tmp_path)     # oldest available is now b
    assert spec["name"] == "b"
    # Claim again: nothing available -> None
    assert dp.claim(pool_dir=tmp_path) is None
    _assert_all_valid_json(tmp_path)


# ---- retire ---------------------------------------------------------------
def test_retire_from_available(tmp_path):
    dp.register("a", "cell-a", "la", pool_dir=tmp_path)
    spec = dp.retire("a", "superseded", pool_dir=tmp_path)
    assert spec["status"] == "retired"
    assert spec["retire_reason"] == "superseded"
    assert isinstance(spec["retired_at"], int)
    _assert_all_valid_json(tmp_path)


def test_retire_from_running(tmp_path):
    dp.register("a", "cell-a", "la", pool_dir=tmp_path)
    dp.claim(name="a", pool_dir=tmp_path)
    spec = dp.retire("a", "broker killed it", pool_dir=tmp_path)
    assert spec["status"] == "retired"
    assert spec["claimed_at"] is not None  # claim stamp preserved through retire
    _assert_all_valid_json(tmp_path)


def test_retire_unknown_raises(tmp_path):
    try:
        dp.retire("ghost", "n/a", pool_dir=tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError retiring an unknown candidate")


# ---- atomicity ------------------------------------------------------------
def test_files_valid_json_after_every_op(tmp_path):
    dp.register("a", "cell-a", "la", pool_dir=tmp_path)
    _assert_all_valid_json(tmp_path)
    dp.register("b", "cell-b", "lb", pool_dir=tmp_path)
    _assert_all_valid_json(tmp_path)
    dp.claim(name="a", pool_dir=tmp_path)
    _assert_all_valid_json(tmp_path)
    dp.claim(pool_dir=tmp_path)  # b
    _assert_all_valid_json(tmp_path)
    dp.retire("b", "done", pool_dir=tmp_path)
    _assert_all_valid_json(tmp_path)
    # No leftover tmp/claiming markers anywhere in the pool tree.
    leftovers = [p.name for p in (tmp_path / "candidates").iterdir()
                 if not p.name.endswith(".json")]
    assert leftovers == [], f"unexpected non-.json files: {leftovers}"
