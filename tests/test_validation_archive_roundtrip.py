"""Save+load roundtrip for the WL5 validation archive format."""

from __future__ import annotations

import numpy as np
import torch

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES


def _make_pi(n: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.random((n, N_ACTIONS)).astype(np.float32) + 1e-3
    return raw / raw.sum(axis=-1, keepdims=True)


def test_archive_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n = 32
    planes = rng.random((n, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    pi = _make_pi(n, rng)
    z = (rng.random(n).astype(np.float32) * 2.0 - 1.0)
    side = rng.integers(0, 2, size=n).astype(np.int8)
    ply = rng.integers(0, 60, size=n).astype(np.int16)
    provenance = ["heuristic_loss"] * 10 + ["high_kl"] * 12 + ["long_defense"] * 10
    archive = {
        "planes": torch.from_numpy(planes),
        "pi_mcts": torch.from_numpy(pi),
        "z": torch.from_numpy(z),
        "provenance": provenance,
        "side": torch.from_numpy(side),
        "ply": torch.from_numpy(ply),
    }
    path = tmp_path / "archive.pt"
    torch.save(archive, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.equal(loaded["planes"], archive["planes"])
    assert torch.equal(loaded["pi_mcts"], archive["pi_mcts"])
    assert torch.equal(loaded["z"], archive["z"])
    assert torch.equal(loaded["side"], archive["side"])
    assert torch.equal(loaded["ply"], archive["ply"])
    assert loaded["provenance"] == provenance


def test_archive_score_runs_against_real_model(tmp_path):
    """Build a tiny model + tiny archive and ensure _score_validation_archive
    returns the expected per-bucket keys with finite values."""
    from gomoku.model import build_model
    from gomoku.train import _score_validation_archive

    rng = np.random.default_rng(7)
    n = 16
    planes = rng.random((n, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    pi = _make_pi(n, rng)
    z = (rng.random(n).astype(np.float32) * 2.0 - 1.0)
    side = np.zeros(n, dtype=np.int8)
    ply = np.zeros(n, dtype=np.int16)
    provenance = ["heuristic_loss"] * 8 + ["long_defense"] * 8
    archive = {
        "planes": torch.from_numpy(planes),
        "pi_mcts": torch.from_numpy(pi),
        "z": torch.from_numpy(z),
        "provenance": provenance,
        "side": torch.from_numpy(side),
        "ply": torch.from_numpy(ply),
    }
    model = build_model("tiny")
    model.eval()
    out = _score_validation_archive(model, archive, device="cpu")
    for key in ("val/policy_ce", "val/policy_kl", "val/value_mse", "val/policy_acc"):
        assert key in out
        assert np.isfinite(out[key])
    for tag in ("heuristic_loss", "long_defense"):
        for stem in ("policy_ce", "policy_kl", "value_mse"):
            key = f"val/{stem}/{tag}"
            assert key in out
            assert np.isfinite(out[key])
    # KL is non-negative up to fp32 roundoff.
    assert out["val/policy_kl"] >= -1e-4
