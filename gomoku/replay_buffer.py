"""Ring buffer of self-play examples backed by preallocated tensors."""

from __future__ import annotations

import numpy as np
import torch

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES
from gomoku.self_play import SelfPlayExample


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device | str = "cpu",
        *,
        aux_opponent_reply: bool = False,
    ):
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
        # Side-to-move (0=black, 1=white) and ply count at the position. Used by
        # the WL5 diagnostics layer for per-color and per-ply-bucket metrics.
        self.side = torch.zeros((capacity,), dtype=torch.int8, device=self.device)
        self.ply = torch.zeros((capacity,), dtype=torch.int16, device=self.device)
        # V3 aux opponent-reply target + validity mask. LAZY-ALLOCATED: when the
        # lever is off (default) these tensors are NEVER allocated, so off-case
        # buffer RAM is byte-identical to a buffer that never knew about aux.
        # `aux_pi` holds the opponent's next-ply policy (zeros where undefined);
        # `aux_mask` is True for rows that carry a valid aux target (last-ply
        # positions and aux-off positions are False → excluded from the aux loss).
        self.aux_enabled = bool(aux_opponent_reply)
        if self.aux_enabled:
            self.aux_pi = torch.zeros(
                (capacity, N_ACTIONS), dtype=torch.float32, device=self.device
            )
            self.aux_mask = torch.zeros((capacity,), dtype=torch.bool, device=self.device)
        else:
            self.aux_pi = None
            self.aux_mask = None
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
        side = torch.from_numpy(np.array([getattr(e, "side", 0) for e in examples], dtype=np.int8))
        ply = torch.from_numpy(np.array([getattr(e, "ply", 0) for e in examples], dtype=np.int16))
        if self.aux_enabled:
            # Build a dense aux-target tensor + validity mask for this add.
            # examples whose aux_pi is None (last ply / aux-undefined) get a
            # zero target and mask=False, so they never contribute aux gradient.
            aux_np = np.zeros((n, N_ACTIONS), dtype=np.float32)
            aux_mask_np = np.zeros((n,), dtype=bool)
            for j, e in enumerate(examples):
                ap = getattr(e, "aux_pi", None)
                if ap is not None:
                    aux_np[j] = ap
                    aux_mask_np[j] = True
            aux_pi_t = torch.from_numpy(aux_np)
            aux_mask_t = torch.from_numpy(aux_mask_np)
        ver = int(self.current_weight_version)
        i = 0
        while i < n:
            end = min(self.head + (n - i), self.capacity)
            chunk = end - self.head
            self.planes[self.head:end].copy_(planes[i:i + chunk].to(self.device))
            self.pi[self.head:end].copy_(pi[i:i + chunk].to(self.device))
            self.z[self.head:end].copy_(z[i:i + chunk].to(self.device))
            self.weight_version[self.head:end] = ver
            self.side[self.head:end].copy_(side[i:i + chunk].to(self.device))
            self.ply[self.head:end].copy_(ply[i:i + chunk].to(self.device))
            if self.aux_enabled:
                self.aux_pi[self.head:end].copy_(aux_pi_t[i:i + chunk].to(self.device))
                self.aux_mask[self.head:end].copy_(aux_mask_t[i:i + chunk].to(self.device))
            i += chunk
            self.head = end % self.capacity
            self.size = min(self.size + chunk, self.capacity)

    def sample(self, batch_size: int, *, return_aux: bool = False) -> tuple[torch.Tensor, ...]:
        if self.size == 0:
            raise ValueError("empty replay buffer")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        base = (self.planes[idx], self.pi[idx], self.z[idx], self.side[idx], self.ply[idx])
        if not return_aux:
            # Default 5-tuple — byte-identical to the pre-aux signature so every
            # existing caller is untouched.
            return base
        # Aux path: append (aux_pi, aux_mask). If the buffer was built without
        # the aux tensors (lever off), return a zero target + all-False mask so
        # the trainer's masked aux loss is a no-op and the call never errors.
        if self.aux_enabled:
            return base + (self.aux_pi[idx], self.aux_mask[idx])
        zero_aux = torch.zeros((batch_size, N_ACTIONS), dtype=torch.float32, device=self.device)
        false_mask = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        return base + (zero_aux, false_mask)

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
        sd = {
            "capacity": self.capacity,
            "head": self.head,
            "size": self.size,
            "planes": self.planes[:self.size].cpu(),
            "pi": self.pi[:self.size].cpu(),
            "z": self.z[:self.size].cpu(),
            "weight_version": self.weight_version[:self.size].cpu(),
            "side": self.side[:self.size].cpu(),
            "ply": self.ply[:self.size].cpu(),
            "current_weight_version": int(self.current_weight_version),
        }
        # Only emit aux tensors when the lever is on, so an off-buffer's
        # checkpoint is byte-identical to the pre-aux schema.
        if self.aux_enabled:
            sd["aux_pi"] = self.aux_pi[:self.size].cpu()
            sd["aux_mask"] = self.aux_mask[:self.size].cpu()
        return sd

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
            if "side" in sd:
                self.side[:n].copy_(sd["side"][:n].to(self.device))
            else:
                self.side[:n].zero_()
                print(f"replay buffer: side tag missing from checkpoint, zero-filled {n} slots")
            if "ply" in sd:
                self.ply[:n].copy_(sd["ply"][:n].to(self.device))
            else:
                self.ply[:n].zero_()
                print(f"replay buffer: ply tag missing from checkpoint, zero-filled {n} slots")
            # Aux target tolerate-missing: when this buffer has the aux head on
            # but the checkpoint predates it (or was off), zero-fill the target
            # and set mask=False so old positions contribute no aux gradient
            # until they evict — mirrors the side/ply missing-tag handling above.
            if self.aux_enabled:
                if "aux_pi" in sd and "aux_mask" in sd:
                    self.aux_pi[:n].copy_(sd["aux_pi"][:n].to(self.device))
                    self.aux_mask[:n].copy_(sd["aux_mask"][:n].to(self.device))
                else:
                    self.aux_pi[:n].zero_()
                    self.aux_mask[:n].zero_()
                    print(f"replay buffer: aux target missing from checkpoint, "
                          f"zero-filled + masked-off {n} slots")
        if "current_weight_version" in sd:
            self.current_weight_version = int(sd["current_weight_version"])
        self.size = n
        self.head = n % self.capacity
