"""Arena artifact-ref resolution (issue #67).

The trainer's ``_deliver`` returns a BARE ``repo_id@revision`` (no scheme), e.g.
``jasonyandell/gomoku-9x9@9x9-champ-recipe-9x9-champ-recipe@0``. The first live
arena gate (2026-06-19) crashed with ``FileNotFoundError`` because the old
``_resolve_model`` only understood ``hf://`` / ``local://`` and treated the bare
ref as a local path. These tests pin the tolerant parsing.
"""
from __future__ import annotations

import pytest

from gomoku.lab.arena import ArenaRole


# the exact ref that failed in production
REAL = "jasonyandell/gomoku-9x9@9x9-champ-recipe-9x9-champ-recipe@0"


@pytest.mark.parametrize("ref,expected", [
    ("owner/repo@rev", ("owner/repo", "rev")),
    ("hf://owner/repo@rev", ("owner/repo", "rev")),
    ("owner/repo", ("owner/repo", None)),
    ("hf://owner/repo", ("owner/repo", None)),
    (REAL, ("jasonyandell/gomoku-9x9", "9x9-champ-recipe-9x9-champ-recipe@0")),
    ("local:///tmp/cand.pt", None),
    (None, None),
    ("model.pt", None),            # bare word, no org/ -> not an HF ref
])
def test_parse_hf_ref(ref, expected):
    assert ArenaRole._parse_hf_ref(ref) == expected


def test_parse_hf_ref_existing_local_path(tmp_path):
    p = tmp_path / "latest.pt"
    p.write_text("x")
    # an existing local path must NOT be mistaken for an HF repo
    assert ArenaRole._parse_hf_ref(str(p)) is None


def test_resolve_model_local_scheme():
    assert ArenaRole()._resolve_model("local:///tmp/cand.pt") == "/tmp/cand.pt"


def test_resolve_model_bare_repo_rev_downloads(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    calls = {}

    def fake_dl(repo, filename, revision=None):
        calls["args"] = (repo, filename, revision)
        return f"/cache/{repo}/{revision}/{filename}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)

    out = ArenaRole()._resolve_model(REAL)
    assert calls["args"] == (
        "jasonyandell/gomoku-9x9", "model.pt",
        "9x9-champ-recipe-9x9-champ-recipe@0")
    assert out.endswith("/model.pt")
