"""Rapfi artifact resolution: local fast-path vs the pinned HF snapshot fallback.

All HF calls are mocked, so these run anywhere with no network and no real repo.
"""

from __future__ import annotations

import hashlib
import os

import pytest

import gomoku.rapfi_pool as rp
from gomoku.rapfi_pool import (
    RapfiUnavailable,
    default_rapfi_cmd,
    rapfi_artifacts,
    rapfi_available,
    rapfi_obtainable,
)


def _make_local_build(tmp_path, exe=True):
    """A fake local engines/rapfi build under a fake repo root."""
    rdir = tmp_path / "engines" / "rapfi"
    rdir.mkdir(parents=True)
    binp = rdir / "pbrain-rapfi"
    binp.write_bytes(b"#!/bin/sh\necho 7,7\n")
    (rdir / "config.toml").write_text("[model]\n")
    if exe:
        os.chmod(binp, 0o755)
    return str(tmp_path), str(rdir)


def _fake_snapshot(tmp_path, *, exe=False, body=b"FAKE-RAPFI-BINARY"):
    """A fake hub snapshot dir with a (non-exec) binary + config."""
    sdir = tmp_path / "snap"
    sdir.mkdir()
    (sdir / "pbrain-rapfi").write_bytes(body)
    (sdir / "config.toml").write_text("[model]\n")
    if not exe:
        os.chmod(sdir / "pbrain-rapfi", 0o644)
    return str(sdir)


# --- local build is the fast path (no HF call) ----------------------------
def test_local_build_resolved_without_touching_hf(tmp_path, monkeypatch):
    root, rdir = _make_local_build(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("snapshot_download must NOT be called when local build exists")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _boom)
    assert rapfi_artifacts(repo=root) == rdir
    assert default_rapfi_cmd(repo=root) == f"{rdir}/pbrain-rapfi --config {rdir}/config.toml gomocup"
    assert rapfi_available(repo=root) is True


# --- HF fallback: fetch, chmod +x, return the snapshot dir -----------------
def test_hf_fallback_fetches_and_makes_executable(tmp_path, monkeypatch):
    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root)
    snap = _fake_snapshot(tmp_path, exe=False)

    monkeypatch.setattr(rp, "RAPFI_HF_REVISION", "deadbeefsha")
    monkeypatch.setattr(rp, "RAPFI_BINARY_SHA256", None)
    seen = {}

    def _fake_dl(repo_id, **kw):
        seen.update(kw, repo_id=repo_id)
        return snap

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_dl)

    got = rapfi_artifacts(repo=empty_root)
    assert got == snap
    assert os.access(os.path.join(snap, "pbrain-rapfi"), os.X_OK)  # chmod +x happened
    assert seen["revision"] == "deadbeefsha"
    assert seen["local_files_only"] is False  # allow_fetch default


# --- sha256 gate -----------------------------------------------------------
def test_sha256_mismatch_refuses(tmp_path, monkeypatch):
    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root)
    snap = _fake_snapshot(tmp_path, body=b"WRONG-BINARY")
    monkeypatch.setattr(rp, "RAPFI_HF_REVISION", "sha")
    monkeypatch.setattr(rp, "RAPFI_BINARY_SHA256", "0" * 64)  # deliberately wrong
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: snap)
    with pytest.raises(RapfiUnavailable, match="sha256 mismatch"):
        rapfi_artifacts(repo=empty_root)


def test_sha256_match_passes(tmp_path, monkeypatch):
    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root)
    body = b"CORRECT-BINARY"
    snap = _fake_snapshot(tmp_path, body=body)
    monkeypatch.setattr(rp, "RAPFI_HF_REVISION", "sha")
    monkeypatch.setattr(rp, "RAPFI_BINARY_SHA256", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: snap)
    assert rapfi_artifacts(repo=empty_root) == snap


# --- no local + no pin -> clear error -------------------------------------
def test_no_local_no_revision_raises(tmp_path, monkeypatch):
    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root)
    monkeypatch.setattr(rp, "RAPFI_HF_REVISION", None)
    with pytest.raises(RapfiUnavailable, match="no pinned HF revision"):
        rapfi_artifacts(repo=empty_root)


def test_hf_not_installed_raises(tmp_path, monkeypatch):
    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root)
    monkeypatch.setattr(rp, "RAPFI_HF_REVISION", "sha")
    monkeypatch.setattr(rp, "_hf_importable", lambda: False)
    with pytest.raises(RapfiUnavailable, match="huggingface_hub not installed"):
        rapfi_artifacts(repo=empty_root)


# --- rapfi_available never hits the network -------------------------------
def test_rapfi_available_is_cache_only(tmp_path, monkeypatch):
    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root)
    monkeypatch.setattr(rp, "RAPFI_HF_REVISION", "sha")
    calls = {}

    def _fake_dl(repo_id, **kw):
        calls["local_files_only"] = kw.get("local_files_only")
        raise FileNotFoundError("not cached")  # simulates cache miss, no network

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_dl)
    assert rapfi_available(repo=empty_root) is False
    assert calls["local_files_only"] is True  # cache-only, never downloads


def test_rapfi_obtainable_optimistic_on_arm64_mac(tmp_path, monkeypatch):
    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root)
    monkeypatch.setattr(rp, "RAPFI_HF_REVISION", "sha")
    monkeypatch.setattr(rp, "_hf_importable", lambda: True)
    # Not cached -> available() is False, but obtainable() is True on arm64 mac.
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(rp, "_is_arm64_mac", lambda: True)
    assert rapfi_available(repo=empty_root) is False
    assert rapfi_obtainable(repo=empty_root) is True
    monkeypatch.setattr(rp, "_is_arm64_mac", lambda: False)
    assert rapfi_obtainable(repo=empty_root) is False  # wrong arch -> not obtainable
