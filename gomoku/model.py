"""Small AlphaZero-style ResNet for 9x9 gomoku.

Input:  (B, 3, 9, 9) float32 — see GameState.to_planes()
Output: policy logits (B, 81), value (B,) in [-1, 1] after tanh.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES


@dataclass
class ModelConfig:
    n_input_planes: int = N_INPUT_PLANES
    n_filters: int = 64
    n_blocks: int = 4
    policy_filters: int = 2
    value_filters: int = 1
    value_hidden: int = 64
    # Stem-conv padding. michaelnny/alpha_zero uses padding=3 with a 3x3 kernel
    # for gomoku ("agent fails to block on edge cases" fix). That expands the
    # feature map from BOARD_SIZE to BOARD_SIZE+4 after the stem, giving the
    # network a "virtual padding zone" around the real board.
    stem_padding: int = 3
    # KataGo-style global pooling (Derby v4 "Whole-board" lever). When False
    # (default), the residual tower is the exact current arch and the
    # state_dict is byte-identical. When True, the LATTER HALF of residual
    # blocks become GlobalPoolResBlocks that inject a board-global (mean+max
    # over spatial dims -> FC -> per-channel bias) signal into every cell. An
    # int instead of a bool sets how many of the trailing blocks get pooling.
    # Rationale: a tiny 3x3-conv tower's receptive field barely spans 9x9, so
    # it cannot cheaply represent board-global facts ("is there a live-four
    # ANYWHERE?"); global pooling injects that context directly.
    global_pool: bool | int = False


SIZE_PRESETS: dict[str, ModelConfig] = {
    "tiny":   ModelConfig(n_filters=32, n_blocks=2, value_hidden=32),
    "small":  ModelConfig(n_filters=64, n_blocks=4, value_hidden=64),
    "medium": ModelConfig(n_filters=96, n_blocks=6, value_hidden=128),
    "large":  ModelConfig(n_filters=128, n_blocks=10, value_hidden=256),
}


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class GlobalPoolResBlock(nn.Module):
    """ResBlock with a KataGo-style global-pooling bias.

    Structure (mirrors KataGo's "global pooling bias" in its residual blocks):
      x -> conv1 -> bn1 -> relu = h          (local features)
      g = [mean_HW(h) ; max_HW(h)]           (2*C global summary, per sample)
      b = pool_fc(g)                         (C per-channel biases)
      h = h + b[:, :, None, None]            (broadcast bias to every cell)
      h -> conv2 -> bn2
      out = relu(x + h)

    The bias is the same for every spatial location of a sample but is
    computed from the WHOLE board, so each cell sees a board-global signal
    (e.g. "a live-four exists somewhere") that a 3x3-conv stack of this depth
    cannot otherwise represent. Params added: pool_fc only = 2*C*C + C.

    Note: the first two ops (conv1, bn1) keep the same module names as
    ResBlock, so fuse_model_for_inference's conv1/bn1/conv2/bn2 fusion still
    applies. The pool_fc has no BatchNorm and is untouched by fusion.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        # mean + max pooled -> 2*C inputs; one bias per channel out.
        self.pool_fc = nn.Linear(2 * channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        mean = h.mean(dim=(2, 3))                 # (B, C)
        mx = h.amax(dim=(2, 3))                    # (B, C)
        bias = self.pool_fc(torch.cat([mean, mx], dim=1))  # (B, C)
        h = h + bias[:, :, None, None]             # broadcast over H, W
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


def _global_pool_block_flags(cfg: ModelConfig) -> list[bool]:
    """Return, per residual block, whether it uses global pooling.

    global_pool == False -> all False (byte-identical to current arch).
    global_pool == True  -> the latter half of blocks (n_blocks // 2 trailing).
    global_pool == int k -> the trailing k blocks (clamped to [0, n_blocks]).
    """
    n = cfg.n_blocks
    gp = cfg.global_pool
    if gp is False or gp == 0:
        return [False] * n
    if gp is True:
        n_pool = n // 2
    else:
        n_pool = max(0, min(int(gp), n))
    return [i >= n - n_pool for i in range(n)]


class GomokuNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.n_filters

        # 3x3 conv with padding=cfg.stem_padding. Output H/W after the stem is
        # BOARD_SIZE + 2*stem_padding - 2. The residual tower preserves that shape.
        kernel = 3
        spatial = BOARD_SIZE + 2 * cfg.stem_padding - kernel + 1

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.n_input_planes, c, kernel, padding=cfg.stem_padding, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        gp_flags = _global_pool_block_flags(cfg)
        self.tower = nn.Sequential(*[
            GlobalPoolResBlock(c) if use_gp else ResBlock(c)
            for use_gp in gp_flags
        ])

        # Policy head: 1x1 conv -> flatten -> linear to BOARD_SIZE^2.
        # Note the FC input uses the post-stem spatial size, not BOARD_SIZE.
        self.policy_conv = nn.Conv2d(c, cfg.policy_filters, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(cfg.policy_filters)
        self.policy_fc = nn.Linear(cfg.policy_filters * spatial * spatial, N_ACTIONS)

        # Value head: 1x1 conv -> flatten -> linear -> linear(1) -> tanh
        self.value_conv = nn.Conv2d(c, cfg.value_filters, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(cfg.value_filters)
        self.value_fc1 = nn.Linear(cfg.value_filters * spatial * spatial, cfg.value_hidden)
        self.value_fc2 = nn.Linear(cfg.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return p, v


def build_model(
    size: str = "small",
    *,
    stem_padding: int | None = None,
    global_pool: bool | int | None = None,
) -> GomokuNet:
    if size not in SIZE_PRESETS:
        raise ValueError(f"unknown size {size!r}; options: {list(SIZE_PRESETS)}")
    cfg = SIZE_PRESETS[size]
    overrides = {}
    if stem_padding is not None:
        overrides["stem_padding"] = stem_padding
    if global_pool is not None:
        overrides["global_pool"] = global_pool
    if overrides:
        from dataclasses import replace
        cfg = replace(cfg, **overrides)
    return GomokuNet(cfg)


def _fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.Module) -> tuple[nn.Conv2d, nn.Module]:
    if isinstance(bn, nn.BatchNorm2d):
        return fuse_conv_bn_eval(conv, bn), nn.Identity()
    return conv, bn


def fuse_model_for_inference(model: GomokuNet) -> GomokuNet:
    """Fuse Conv+BatchNorm pairs for eval-only inference.

    This mutates and returns `model`. Call only after loading checkpoint weights
    into a model that will not be trained or saved back as a normal checkpoint.
    """
    model.eval()
    model.stem[0], model.stem[1] = _fuse_conv_bn_pair(model.stem[0], model.stem[1])
    for block in model.tower:
        block.conv1, block.bn1 = _fuse_conv_bn_pair(block.conv1, block.bn1)
        block.conv2, block.bn2 = _fuse_conv_bn_pair(block.conv2, block.bn2)
    model.policy_conv, model.policy_bn = _fuse_conv_bn_pair(
        model.policy_conv, model.policy_bn
    )
    model.value_conv, model.value_bn = _fuse_conv_bn_pair(
        model.value_conv, model.value_bn
    )
    return model


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    path: str,
    model: GomokuNet,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    epoch: int = 0,
    total_games: int = 0,
    wandb_run_id: str | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.cfg),
        "epoch": epoch,
        "total_games": total_games,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if wandb_run_id is not None:
        payload["wandb_run_id"] = wandb_run_id
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
) -> tuple[GomokuNet, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_cfg = dict(payload["model_config"])
    # Pre-AZ-recipe checkpoints predate stem_padding; default to 1 so they load.
    saved_cfg.setdefault("stem_padding", 1)
    cfg = ModelConfig(**saved_cfg)
    model = GomokuNet(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
