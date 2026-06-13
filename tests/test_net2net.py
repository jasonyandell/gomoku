"""Function-preservation gate for scripts/net2net_grow.py (Net2Net grow).

The CRITICAL assertion: a net grown from a strong small net computes an
(almost) IDENTICAL function — same policy logits + value on random board inputs
to within ~1e-4 — so it starts at the source's strength (no cold-start retrain).

Board size is a process-level constant, so this file runs at the ACTIVE board
size of the pytest process. Run it BOTH ways:

    pytest tests/test_net2net.py                       # 9x9
    GOMOKU_BOARD_SIZE=15 pytest tests/test_net2net.py  # 15x15

Both must pass: the grow is board-size-agnostic and preserves the source board.
"""

from __future__ import annotations

import dataclasses

import torch

from gomoku import game
from gomoku.model import (
    GomokuNet,
    ModelConfig,
    build_model,
    n_params,
    save_checkpoint,
)

# net2net_grow is a script (no package); import it by path.
import importlib.util
import os

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "net2net_grow.py",
)
_spec = importlib.util.spec_from_file_location("net2net_grow", _SCRIPT)
net2net_grow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(net2net_grow)


N = game.BOARD_SIZE


def _small_source_cfg(**overrides) -> ModelConfig:
    """A tiny but structurally-faithful 'small-like' source: a few channels, a
    couple blocks, value_hidden small. Defaults mirror the live arch shape
    (scalar head, stem_padding=3, global_pool latter-half) so the grow path
    exercised by the test matches production."""
    base = ModelConfig(
        board_size=N,
        n_filters=8,
        n_blocks=2,
        value_hidden=6,
        stem_padding=3,
        global_pool=True,   # latter half -> 1 global-pool block
    )
    return dataclasses.replace(base, **overrides)


def _make_source_checkpoint(tmp_path, cfg: ModelConfig, *, seed: int = 0) -> str:
    """Build a source net with NON-TRIVIAL (randomly perturbed) BN + weights and
    save it as a trainer checkpoint. Random BN stats matter: a fresh-init net has
    BN running_mean=0 / running_var=1 / gamma=1 / beta=0, which can mask channel-
    map bugs. We jitter them so the equivalence test has teeth."""
    torch.manual_seed(seed)
    m = GomokuNet(cfg)
    m.eval()
    # Perturb every parameter AND BN running stats so the function is non-degenerate.
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.add_(0.3 * torch.randn_like(mod.running_mean))
                mod.running_var.abs_().add_(0.5)  # keep > 0
    path = str(tmp_path / "src.pt")
    save_checkpoint(path, m, epoch=11, total_games=999)
    return path


def _register_target_preset(name: str, cfg: ModelConfig) -> None:
    """Register a small TARGET preset in the model's SIZE_PRESETS so grow() can
    grow into it (grow() looks the target up by preset name). Idempotent."""
    from gomoku import model as model_mod

    model_mod.SIZE_PRESETS[name] = cfg


# ----------------------------------------------------------------------------
# The equivalence gate.
# ----------------------------------------------------------------------------


def _grow_and_check(tmp_path, src_cfg: ModelConfig, tgt_cfg: ModelConfig,
                    *, atol: float = 1e-4):
    src_path = _make_source_checkpoint(tmp_path, src_cfg)
    _register_target_preset("net2net_test_target", tgt_cfg)
    out_path = str(tmp_path / "grown.pt")

    summary = net2net_grow.grow(
        src_path, out_path, target_size="net2net_test_target", seed=0
    )

    # 1) Function preservation: the reported max deviation is tiny.
    assert summary["max_dev"] < atol, (
        f"grown net deviates from source by {summary['max_dev']:.3e} "
        f"(policy={summary['max_dev_policy']:.3e}, value={summary['max_dev_value']:.3e})"
    )

    # 2) Bigger param count (real capacity added).
    assert summary["tgt_params"] > summary["src_params"]

    # 3) The saved seed loads + forwards at the SOURCE board size, and matches the
    #    source function there too (independent re-check, fresh nets from disk).
    src_payload = torch.load(src_path, map_location="cpu", weights_only=False)
    src_net = GomokuNet(ModelConfig(**src_payload["model_config"]))
    src_net.load_state_dict(src_payload["model_state_dict"])
    src_net.eval()

    grown_payload = torch.load(out_path, map_location="cpu", weights_only=False)
    grown_cfg = ModelConfig(**grown_payload["model_config"])
    assert grown_cfg.board_size == src_cfg.board_size == N
    grown_net = GomokuNet(grown_cfg)
    grown_net.load_state_dict(grown_payload["model_state_dict"])
    grown_net.eval()

    # Trainer --resume format: fresh timeline, no optimizer/buffer/wandb/ema.
    assert grown_payload["epoch"] == 0
    assert grown_payload["total_games"] == 0
    assert "optimizer_state_dict" not in grown_payload
    assert "replay_buffer" not in grown_payload
    assert "wandb_run_id" not in grown_payload
    assert "ema_model_state_dict" not in grown_payload

    torch.manual_seed(99)
    x = torch.randn(5, src_cfg.n_input_planes, N, N)
    with torch.no_grad():
        p0, v0 = src_net(x)
        p1, v1 = grown_net(x)
    assert torch.allclose(p0, p1, atol=atol), \
        f"reloaded policy dev {float((p0 - p1).abs().max()):.3e}"
    assert torch.allclose(v0, v1, atol=atol), \
        f"reloaded value dev {float((v0 - v1).abs().max()):.3e}"
    return summary


def test_grow_widen_and_deepen_preserves_function(tmp_path):
    """The headline case: widen channels + value_hidden AND deepen (more blocks),
    with global pooling on — mirrors small(64x4) -> 96x8 structurally."""
    src = _small_source_cfg(n_filters=8, n_blocks=2, value_hidden=6, global_pool=True)
    tgt = _small_source_cfg(n_filters=12, n_blocks=4, value_hidden=10, global_pool=True)
    s = _grow_and_check(tmp_path, src, tgt)
    assert s["n_inserted_blocks"] == 2
    assert s["c_src"] == 8 and s["c_tgt"] == 12
    assert s["h_src"] == 6 and s["h_tgt"] == 10


def test_grow_widen_only_no_gp_scalar(tmp_path):
    """Widen-only (no new blocks), no global pool — isolates the widen math."""
    src = _small_source_cfg(n_filters=8, n_blocks=3, value_hidden=6, global_pool=False)
    tgt = _small_source_cfg(n_filters=16, n_blocks=3, value_hidden=12, global_pool=False)
    s = _grow_and_check(tmp_path, src, tgt)
    assert s["n_inserted_blocks"] == 0


def test_grow_deepen_only_preserves_function(tmp_path):
    """Deepen-only (same width, more blocks) — isolates the identity-block math.
    The inserted zero-residual blocks must be EXACT identity (relu, x>=0)."""
    src = _small_source_cfg(n_filters=10, n_blocks=2, value_hidden=8, global_pool=False)
    tgt = _small_source_cfg(n_filters=10, n_blocks=5, value_hidden=8, global_pool=False)
    s = _grow_and_check(tmp_path, src, tgt, atol=1e-5)  # identity is exact
    assert s["n_inserted_blocks"] == 3
    assert s["c_src"] == s["c_tgt"]


def test_grow_wdl_value_head(tmp_path):
    """A non-scalar value head (wdl): the final value FC is value_wdl_fc, whose
    input (value_hidden) must be divided by g_h. Exercises the value-head branch."""
    src = _small_source_cfg(n_filters=8, n_blocks=2, value_hidden=6,
                            global_pool=True, value_head="wdl")
    tgt = _small_source_cfg(n_filters=12, n_blocks=4, value_hidden=10,
                            global_pool=True, value_head="wdl")
    _grow_and_check(tmp_path, src, tgt)


def test_grow_global_pool_only_blocks(tmp_path):
    """global_pool=int so EVERY block is a GlobalPoolResBlock — stresses the
    pool_fc widen (mean/max halves + per-channel output bias) in every block."""
    src = _small_source_cfg(n_filters=8, n_blocks=2, value_hidden=6, global_pool=2)
    tgt = _small_source_cfg(n_filters=14, n_blocks=2, value_hidden=11, global_pool=2)
    _grow_and_check(tmp_path, src, tgt)


def test_grown_param_count_matches_fresh_target(tmp_path):
    """Sanity: the grown net has EXACTLY the same param count as a freshly built
    target preset of the same shape (no stray/missing tensors)."""
    src = _small_source_cfg(n_filters=8, n_blocks=2, value_hidden=6, global_pool=True)
    tgt = _small_source_cfg(n_filters=12, n_blocks=4, value_hidden=10, global_pool=True)
    src_path = _make_source_checkpoint(tmp_path, src)
    _register_target_preset("net2net_test_target", tgt)
    out_path = str(tmp_path / "grown.pt")
    net2net_grow.grow(src_path, out_path, target_size="net2net_test_target", seed=0)

    grown_payload = torch.load(out_path, map_location="cpu", weights_only=False)
    grown_net = GomokuNet(ModelConfig(**grown_payload["model_config"]))
    fresh_target = GomokuNet(tgt)
    assert n_params(grown_net) == n_params(fresh_target)
