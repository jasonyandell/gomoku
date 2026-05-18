"""Ring buffer of self-play examples backed by preallocated tensors."""

from __future__ import annotations

import numpy as np
import torch

from gomoku.game import BOARD_SIZE, N_ACTIONS
from gomoku.self_play import SelfPlayExample


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device | str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.planes = torch.zeros((capacity, 3, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32, device=self.device)
        self.pi = torch.zeros((capacity, N_ACTIONS), dtype=torch.float32, device=self.device)
        self.z = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
        self.head = 0
        self.size = 0

    def add(self, examples: list[SelfPlayExample]) -> None:
        if not examples:
            return
        n = len(examples)
        planes = torch.from_numpy(np.stack([e.planes for e in examples]))
        pi = torch.from_numpy(np.stack([e.pi for e in examples]))
        z = torch.from_numpy(np.array([e.z for e in examples], dtype=np.float32))
        # Write in chunks if we wrap around the ring.
        i = 0
        while i < n:
            end = min(self.head + (n - i), self.capacity)
            chunk = end - self.head
            self.planes[self.head:end].copy_(planes[i:i + chunk].to(self.device))
            self.pi[self.head:end].copy_(pi[i:i + chunk].to(self.device))
            self.z[self.head:end].copy_(z[i:i + chunk].to(self.device))
            i += chunk
            self.head = end % self.capacity
            self.size = min(self.size + chunk, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise ValueError("empty replay buffer")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.planes[idx], self.pi[idx], self.z[idx]

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "head": self.head,
            "size": self.size,
            "planes": self.planes[:self.size].cpu(),
            "pi": self.pi[:self.size].cpu(),
            "z": self.z[:self.size].cpu(),
        }

    def load_state_dict(self, sd: dict) -> None:
        if sd["capacity"] != self.capacity:
            # Tolerate capacity change — just load what fits.
            pass
        n = min(int(sd["size"]), self.capacity)
        if n > 0:
            self.planes[:n].copy_(sd["planes"][:n].to(self.device))
            self.pi[:n].copy_(sd["pi"][:n].to(self.device))
            self.z[:n].copy_(sd["z"][:n].to(self.device))
        self.size = n
        self.head = n % self.capacity
