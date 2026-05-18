"""Shared utilities."""

from __future__ import annotations

import os
import subprocess

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Pick the best available torch device for this Mac.

    Order: prefer arg → MPS → CPU. (No CUDA on Apple Silicon.)
    """
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_wandb_key_from_keychain() -> str | None:
    """Pull WANDB_API_KEY from macOS Keychain (service `wandb-api-key`). Falls back to env."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "wandb-api-key", "-w"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            key = result.stdout.strip()
            if key:
                os.environ["WANDB_API_KEY"] = key
                return key
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return os.environ.get("WANDB_API_KEY")
