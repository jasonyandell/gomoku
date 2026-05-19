"""Small AlphaZero-style ResNet for 9x9 gomoku.

Input:  (B, 3, 9, 9) float32 — see GameState.to_planes()
Output: policy logits (B, 81), value (B,) in [-1, 1] after tanh.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES


@dataclass
class ModelConfig:
    n_input_planes: int = N_INPUT_PLANES
    n_filters: int = 64
    n_blocks: int = 4
    policy_filters: int = 2
    value_filters: int = 1
    value_hidden: int = 64


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


class GomokuNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.n_filters

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.n_input_planes, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(c) for _ in range(cfg.n_blocks)])

        # Policy head: 1x1 conv -> flatten -> linear to 81
        self.policy_conv = nn.Conv2d(c, cfg.policy_filters, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(cfg.policy_filters)
        self.policy_fc = nn.Linear(cfg.policy_filters * BOARD_SIZE * BOARD_SIZE, N_ACTIONS)

        # Value head: 1x1 conv -> flatten -> linear -> linear(1) -> tanh
        self.value_conv = nn.Conv2d(c, cfg.value_filters, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(cfg.value_filters)
        self.value_fc1 = nn.Linear(cfg.value_filters * BOARD_SIZE * BOARD_SIZE, cfg.value_hidden)
        self.value_fc2 = nn.Linear(cfg.value_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return p, v


def build_model(size: str = "small") -> GomokuNet:
    if size not in SIZE_PRESETS:
        raise ValueError(f"unknown size {size!r}; options: {list(SIZE_PRESETS)}")
    return GomokuNet(SIZE_PRESETS[size])


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
    cfg = ModelConfig(**payload["model_config"])
    model = GomokuNet(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload
