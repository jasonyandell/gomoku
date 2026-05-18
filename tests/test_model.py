import numpy as np
import torch

from gomoku.game import GameState
from gomoku.model import build_model, n_params, save_checkpoint, load_checkpoint


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


def test_to_planes_dtype():
    s = GameState.initial()
    x = s.to_planes()
    assert x.dtype == np.float32
    assert x.shape == (3, 9, 9)
    # Last plane is the side-to-move indicator (all ones).
    assert (x[2] == 1.0).all()
