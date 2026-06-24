"""Tests for gomoku.hf — the slim_checkpoint helper.

The HF push path is not exercised here (no network in tests). We verify
that a slimmed checkpoint:
  * drops optimizer state + replay buffer
  * round-trips through ``gomoku.model.load_checkpoint``
  * produces a forward pass identical to the original (no weight loss)
  * is meaningfully smaller than the source
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

# gomoku.hf imports huggingface_hub at module load, so importing slim_checkpoint
# (below) ERRORs on a minimal install that lacks the optional dep. Skip the whole
# module cleanly instead — these tests exercise the HF push/slim path, which is
# meaningless without huggingface_hub.
pytest.importorskip("huggingface_hub")

from gomoku.game import GameState
from gomoku.hf import slim_checkpoint
from gomoku.model import build_model, load_checkpoint, save_checkpoint


def _make_fat_checkpoint(path):
    """Save a tiny model + optimizer state + fake replay buffer."""
    model = build_model("tiny")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Step the optimizer so it has nontrivial state to drop.
    x = torch.from_numpy(GameState.initial().to_planes()).unsqueeze(0)
    p, v = model(x)
    (p.sum() + v.sum()).backward()
    opt.step()

    # save_checkpoint already wires up the standard keys.
    save_checkpoint(str(path), model, opt, epoch=12, total_games=345, wandb_run_id="run-xyz")

    # Inject a fake replay buffer to mimic the real training checkpoints.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["replay_buffer"] = [torch.randn(3, 9, 9) for _ in range(64)]
    torch.save(payload, path)
    return model


def test_slim_strips_training_state(tmp_path):
    src = tmp_path / "fat.pt"
    dst = tmp_path / "slim.pt"
    _make_fat_checkpoint(src)

    snap = slim_checkpoint(src, dst)

    # Returned dict + saved file should both be slim.
    for key in ("model_state_dict", "model_config", "epoch", "total_games", "wandb_run_id"):
        assert key in snap, f"snapshot missing {key}"
    assert "optimizer_state_dict" not in snap
    assert "replay_buffer" not in snap

    on_disk = torch.load(dst, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" not in on_disk
    assert "replay_buffer" not in on_disk
    assert on_disk["epoch"] == 12
    assert on_disk["total_games"] == 345
    assert on_disk["wandb_run_id"] == "run-xyz"


def test_slim_roundtrips_through_load_checkpoint(tmp_path):
    src = tmp_path / "fat.pt"
    dst = tmp_path / "slim.pt"
    original = _make_fat_checkpoint(src)
    original.eval()

    slim_checkpoint(src, dst)
    loaded, payload = load_checkpoint(str(dst), device="cpu")
    loaded.eval()

    assert payload["epoch"] == 12
    assert payload["total_games"] == 345

    # Forward pass on the empty board should match exactly (cpu, eval).
    x = torch.from_numpy(GameState.initial().to_planes()).unsqueeze(0)
    with torch.no_grad():
        p1, v1 = original(x)
        p2, v2 = loaded(x)
    assert torch.allclose(p1, p2, atol=1e-6)
    assert torch.allclose(v1, v2, atol=1e-6)


def test_slim_is_smaller(tmp_path):
    src = tmp_path / "fat.pt"
    dst = tmp_path / "slim.pt"
    _make_fat_checkpoint(src)
    slim_checkpoint(src, dst)
    assert dst.stat().st_size < src.stat().st_size


def test_slim_rejects_missing_required_keys(tmp_path):
    src = tmp_path / "bad.pt"
    torch.save({"epoch": 1}, src)  # no model_state_dict / model_config
    try:
        slim_checkpoint(src, tmp_path / "out.pt")
    except KeyError:
        return
    raise AssertionError("expected KeyError for malformed checkpoint")


# --- push_slice: per-slice revision + provenance + champion tag -------------


class _FakeApi:
    def __init__(self, rec):
        self._rec = rec

    def create_repo(self, **k):
        self._rec.setdefault("repo", []).append(k)

    def create_branch(self, **k):
        self._rec.setdefault("branch", []).append(k)

    def create_tag(self, **k):
        self._rec.setdefault("tag", []).append(k)

    def upload_folder(self, **k):
        state = json.loads((Path(k["folder_path"]) / "training_state.json").read_text())
        self._rec.setdefault("upload", []).append({"revision": k.get("revision"),
                                                   "state": state})


def test_push_slice_creates_revision_with_provenance_and_tag(tmp_path, monkeypatch):
    import huggingface_hub
    from gomoku import hf

    src = tmp_path / "fat.pt"
    _make_fat_checkpoint(src)
    rec = {}
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: _FakeApi(rec))

    ref = hf.push_slice(src, repo_id="acme/gomoku", revision="mvp-mvp_0",
                        provenance={"git_sha": "abc123", "model_elo": 1500},
                        tag="champion")

    assert ref == "acme/gomoku@mvp-mvp_0"
    assert rec["branch"][0]["branch"] == "mvp-mvp_0"
    assert rec["upload"][0]["revision"] == "mvp-mvp_0"
    assert rec["upload"][0]["state"]["provenance"]["git_sha"] == "abc123"
    assert rec["tag"][0]["tag"] == "champion" and rec["tag"][0]["revision"] == "mvp-mvp_0"


def test_push_slice_without_tag_does_not_tag(tmp_path, monkeypatch):
    import huggingface_hub
    from gomoku import hf

    src = tmp_path / "fat.pt"
    _make_fat_checkpoint(src)
    rec = {}
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: _FakeApi(rec))

    hf.push_slice(src, repo_id="acme/gomoku", revision="r1")
    assert "tag" not in rec
    assert rec["upload"][0]["state"].get("provenance") is None
