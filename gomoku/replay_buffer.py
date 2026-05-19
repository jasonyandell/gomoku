"""Ring buffer of self-play examples backed by preallocated tensors."""

from __future__ import annotations

import numpy as np
import torch

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES
from gomoku.self_play import SelfPlayExample


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device | str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.planes = torch.zeros(
            (capacity, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float32, device=self.device,
        )
        self.pi = torch.zeros((capacity, N_ACTIONS), dtype=torch.float32, device=self.device)
        self.z = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
        # Weight-version tag: which "brain" generated the position in each slot.
        # Used for stratified sampling and shape diagnostics. Auto-overwritten
        # when the ring buffer reuses a slot — no separate deletion needed.
        self.weight_version = torch.zeros((capacity,), dtype=torch.int64, device=self.device)
        self.current_weight_version: int = 0
        self.head = 0
        self.size = 0

    def set_weight_version(self, version: int) -> None:
        """Set the weight-version tag applied to subsequent add() calls."""
        self.current_weight_version = int(version)

    def add(self, examples: list[SelfPlayExample]) -> None:
        if not examples:
            return
        n = len(examples)
        planes = torch.from_numpy(np.stack([e.planes for e in examples]))
        pi = torch.from_numpy(np.stack([e.pi for e in examples]))
        z = torch.from_numpy(np.array([e.z for e in examples], dtype=np.float32))
        ver = int(self.current_weight_version)
        # Write in chunks if we wrap around the ring.
        i = 0
        while i < n:
            end = min(self.head + (n - i), self.capacity)
            chunk = end - self.head
            self.planes[self.head:end].copy_(planes[i:i + chunk].to(self.device))
            self.pi[self.head:end].copy_(pi[i:i + chunk].to(self.device))
            self.z[self.head:end].copy_(z[i:i + chunk].to(self.device))
            self.weight_version[self.head:end] = ver
            i += chunk
            self.head = end % self.capacity
            self.size = min(self.size + chunk, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise ValueError("empty replay buffer")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.planes[idx], self.pi[idx], self.z[idx]

    def shape_stats(self, stone_buckets: tuple[int, ...] = (0, 5, 10, 15, 20, 30, 40, 60, 81)
                    ) -> dict[str, float]:
        """Snapshot of the buffer's *shape* — distribution over position types.

        Returns a dict of named scalars for wandb. Includes:

          buffer/n            : current buffer size
          buffer/stones_bucket_{lo}_{hi}: fraction of positions whose total
                                          stone count is in [lo, hi)
          buffer/z_wins       : fraction with z == +1
          buffer/z_losses     : fraction with z == -1
          buffer/z_draws      : fraction with z == 0

        n_stones is computed as the sum of planes 0 (current side) and the
        opponent-current-frame plane (index = N_INPUT_PLANES//2). That gives
        the total stones on the board for each position in the buffer.
        """
        if self.size == 0:
            return {}
        from gomoku.game import N_INPUT_PLANES
        # current-side me-plane = 0, current-side opp-plane = N_INPUT_PLANES // 2
        opp_idx = N_INPUT_PLANES // 2
        view = self.planes[:self.size]
        # sum stones per position: (n,) tensor of ints in [0, BOARD_SIZE^2]
        stones = (view[:, 0].sum(dim=(-2, -1)) + view[:, opp_idx].sum(dim=(-2, -1)))
        stones_cpu = stones.detach().cpu().numpy()
        z = self.z[:self.size].detach().cpu().numpy()
        out: dict[str, float] = {
            "buffer/n": float(self.size),
            "buffer/z_wins": float((z > 0).mean()),
            "buffer/z_losses": float((z < 0).mean()),
            "buffer/z_draws": float((z == 0).mean()),
            "buffer/stones_mean": float(stones_cpu.mean()),
        }
        # Bucket histogram
        edges = list(stone_buckets) + [10**9]
        for lo, hi in zip(edges[:-1], edges[1:]):
            label = f"buffer/stones_{lo:02d}_{min(hi, 81):02d}"
            mask = (stones_cpu >= lo) & (stones_cpu < hi)
            out[label] = float(mask.mean())
        # Weight-version freshness: how stale the buffer is relative to the
        # current "brain." 0 = all positions tagged with the latest version.
        if self.current_weight_version > 0:
            ver = self.weight_version[:self.size].detach().cpu().numpy()
            age = self.current_weight_version - ver
            out["buffer/age_mean"] = float(age.mean())
            out["buffer/age_p50"] = float(np.percentile(age, 50))
            out["buffer/age_p90"] = float(np.percentile(age, 90))
            out["buffer/frac_current"] = float((age == 0).mean())
        return out

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "head": self.head,
            "size": self.size,
            "planes": self.planes[:self.size].cpu(),
            "pi": self.pi[:self.size].cpu(),
            "z": self.z[:self.size].cpu(),
            "weight_version": self.weight_version[:self.size].cpu(),
            "current_weight_version": int(self.current_weight_version),
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
            if "weight_version" in sd:
                self.weight_version[:n].copy_(sd["weight_version"][:n].to(self.device))
        if "current_weight_version" in sd:
            self.current_weight_version = int(sd["current_weight_version"])
        self.size = n
        self.head = n % self.capacity
