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
