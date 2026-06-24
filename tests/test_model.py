import numpy as np
import torch

from gomoku.game import GameState
from gomoku.model import (
    build_model,
    fuse_model_for_inference,
    load_checkpoint,
    n_params,
    save_checkpoint,
)


def test_model_forward_shape():
    m = build_model("tiny")
    m.eval()
    s = GameState.initial()
    x = torch.from_numpy(s.to_planes()).unsqueeze(0)
    with torch.no_grad():
        p, v = m(x)
    assert p.shape == (1, 81)
    assert v.shape == (1,)
    assert torch.all(v >= -1) and torch.all(v <= 1)


def test_model_sizes_distinct():
    sizes = {s: n_params(build_model(s)) for s in ["tiny", "small", "medium", "large"]}
    vals = list(sizes.values())
    assert vals == sorted(vals)
    assert len(set(vals)) == 4


def test_fuse_model_for_inference_preserves_outputs():
    torch.manual_seed(0)
    m = build_model("tiny", stem_padding=1)
    m.eval()
    x = torch.randn(4, 17, 9, 9)
    with torch.no_grad():
        p0, v0 = m(x)
    fused = fuse_model_for_inference(m)
    with torch.no_grad():
        p1, v1 = fused(x)
    assert torch.allclose(p0, p1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(v0, v1, atol=1e-5, rtol=1e-5)
    assert isinstance(fused.stem[1], torch.nn.Identity)


def test_checkpoint_roundtrip(tmp_path):
    m = build_model("tiny")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    p = tmp_path / "ckpt.pt"
    save_checkpoint(str(p), m, opt, epoch=7, total_games=42, wandb_run_id="abc123")
    m2, payload = load_checkpoint(str(p))
    assert payload["epoch"] == 7
    assert payload["total_games"] == 42
    assert payload["wandb_run_id"] == "abc123"
    # Same params
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_save_checkpoint_is_atomic(tmp_path, monkeypatch):
    """save_checkpoint must never leave a partial file observable at `path`
    (closes #76): a concurrent reader watching latest.pt must see either the old
    file or the fully-written new one, never a torn mid-write file.

    We prove the write goes through a sibling `.tmp` + os.replace by spying on
    both: torch.save is asserted to target a `.tmp` path (NOT `path` directly),
    and the bytes only land on `path` via os.replace AFTER torch.save returns. So
    at no point does an incomplete file exist at the observable destination.
    """
    import os as _os

    import gomoku.model as model_mod

    p = tmp_path / "latest.pt"

    # Seed an existing (old) checkpoint at the destination; the atomic replace
    # must overwrite it in one step, never truncate-then-fill it.
    save_checkpoint(str(p), build_model("tiny"), epoch=1)
    old_bytes = p.read_bytes()

    events: list[str] = []
    real_save = torch.save
    real_replace = _os.replace

    def spy_save(obj, f, *a, **k):
        # torch.save must target the sibling tmp, never the final path.
        assert str(f) == str(p) + ".tmp", f"torch.save wrote to {f!r}, not the .tmp"
        # The observable destination still holds the OLD file (no partial) while
        # the new bytes are being streamed into the tmp.
        assert p.read_bytes() == old_bytes, "destination changed before os.replace"
        events.append("save")
        return real_save(obj, f, *a, **k)

    def spy_replace(src, dst, *a, **k):
        assert str(src) == str(p) + ".tmp" and str(dst) == str(p)
        # The replace lands AFTER the tmp is fully written.
        assert events == ["save"], "os.replace ran before torch.save completed"
        events.append("replace")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(model_mod.torch, "save", spy_save)
    monkeypatch.setattr(model_mod.os, "replace", spy_replace)

    save_checkpoint(str(p), build_model("tiny"), epoch=2, total_games=99)

    assert events == ["save", "replace"]
    # No leftover tmp, and the final file is the fully-written new checkpoint.
    assert not (tmp_path / "latest.pt.tmp").exists()
    _m, payload = load_checkpoint(str(p))
    assert payload["epoch"] == 2 and payload["total_games"] == 99


def test_checkpoint_embeds_board_size_and_legacy_defaults_to_9(tmp_path):
    # 15x15-era contract: every new checkpoint embeds its board size, and a
    # legacy checkpoint without the field loads as 9x9 (all pre-era nets were
    # 9x9). At the default board size both paths must load cleanly.
    m = build_model("tiny")
    p = tmp_path / "ckpt.pt"
    save_checkpoint(str(p), m)
    _, payload = load_checkpoint(str(p))
    assert payload["model_config"]["board_size"] == 9

    # Strip the field to simulate a pre-board-size checkpoint.
    raw = torch.load(str(p), weights_only=False)
    del raw["model_config"]["board_size"]
    torch.save(raw, str(p))
    m2, payload2 = load_checkpoint(str(p))
    assert m2.cfg.board_size == 9
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_to_planes_dtype():
    from gomoku.game import N_INPUT_PLANES, BOARD_SIZE
    s = GameState.initial()
    x = s.to_planes()
    assert x.dtype == np.float32
    assert x.shape == (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    # Last plane is the side-to-move indicator (all ones).
    assert (x[-1] == 1.0).all()
