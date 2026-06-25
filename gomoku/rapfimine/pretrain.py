"""Pretrain a 15x15 net on the mined idx-2 Rapfi teacher dataset (the Bruce-Lee
one-position experiment): supervised distillation of Rapfi's soft policy + value,
producing a standard checkpoint that ``run_sweep --resume`` can warm-start from.

Uses ONLY standard project pieces — ``build_model`` / ``save_checkpoint`` and the
trainer's ``policy_loss`` (soft-target CE) / ``value_loss`` (MSE) — so the
pretrained net is byte-compatible with the normal AlphaZero loop. No reinvention.

Targets per position:
  * policy: masked temperature-softmax of Rapfi's per-move winrate map (same
    transform ``TeacherDataset`` uses for the --teacher arm; teacher_temp).
  * value (side-to-move): 2 * best-move-winrate - 1, in [-1, 1]. Auxiliary — the
    policy carries the load (#18/#44), value is a calibrated warm bonus for MCTS.

Memory: shards are held CPU-resident as float16 (1.2M positions ~= 9 GB) and a
batch is moved to MPS as float32 per step, so we never put the whole 18 GB
float32 set on the device. D4 augment is applied per batch (planes rotated/
flipped, the dense winrate target permuted by the SAME symmetry).

    GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.pretrain \\
        --shards mined/idx2_15x15 --out checkpoints/idx2_pretrain.pt \\
        --size large --epochs 40 --batch-size 1024 --lr 3e-4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from gomoku.board_config import BOARD_SIZE, N_ACTIONS
from gomoku.model import build_model, n_params, save_checkpoint
from gomoku.rapfimine.store import iter_shard_paths
from gomoku.teacher import _action_perms


def load_shards(shards_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all shards into (planes f16 [M,17,N,N], soft f16 [M,N_ACTIONS]).

    Preallocates from the shard row-counts so peak RAM is ~1× the dataset, not 2×.
    """
    paths = iter_shard_paths(shards_dir)
    if not paths:
        raise SystemExit(f"no shards in {shards_dir!r}")
    counts, shapes = [], None
    for p in paths:
        with np.load(p) as z:
            counts.append(int(z["moves"].shape[0]))
            if shapes is None:
                shapes = z["planes"].shape[1:]
    total = sum(counts)
    planes = np.empty((total, *shapes), dtype=np.float16)
    soft = np.zeros((total, N_ACTIONS), dtype=np.float16)
    i = 0
    for p in paths:
        with np.load(p) as z:
            m = int(z["moves"].shape[0])
            planes[i:i + m] = z["planes"]
            soft[i:i + m] = z["soft_policy"]
            i += m
    print(f"[pretrain] loaded {total} positions from {len(paths)} shards "
          f"({planes.nbytes / 1e9:.1f} GB f16 planes)", flush=True)
    return planes, soft


class PretrainData:
    """CPU-resident teacher set; per-batch D4 augment + device transfer."""

    def __init__(self, planes: np.ndarray, soft: np.ndarray, *, device,
                 teacher_temp: float = 0.10, augment: bool = True):
        import torch

        self.device = device
        self.teacher_temp = float(teacher_temp)
        self.augment = bool(augment)
        self.planes = planes                       # CPU f16 (M,17,N,N)
        self.soft = soft                           # CPU f16 (M,N_ACTIONS)
        self.n = planes.shape[0]
        self._perm = torch.as_tensor(
            _action_perms(BOARD_SIZE, N_ACTIONS), dtype=torch.long, device=device)

    def _softmax_target(self, soft_rows):
        import torch
        temp = max(self.teacher_temp, 1e-6)
        support = soft_rows > 0.0
        logits = (soft_rows / temp).masked_fill(~support, float("-inf"))
        empty = ~support.any(dim=-1, keepdim=True)
        logits = torch.where(empty, torch.zeros_like(logits), logits)
        return torch.softmax(logits, dim=-1)

    def sample(self, batch_size: int):
        import torch
        idx = np.random.randint(0, self.n, size=batch_size)
        planes = torch.as_tensor(
            self.planes[idx].astype(np.float32), device=self.device)
        soft = torch.as_tensor(
            self.soft[idx].astype(np.float32), device=self.device)
        # value (side-to-move) = 2*best-winrate-1 ; computed before augment (scalar,
        # D4-invariant).
        best_wr = soft.max(dim=-1).values
        v = (2.0 * best_wr - 1.0).clamp(-1.0, 1.0)
        if self.augment:
            s = int(np.random.randint(0, 8))
            rot, flip = s % 4, s // 4
            if rot:
                planes = torch.rot90(planes, rot, dims=(-2, -1))
            if flip:
                planes = torch.flip(planes, dims=(-1,))
            permuted = torch.zeros_like(soft)
            permuted[:, self._perm[s]] = soft
            soft = permuted
        return planes.contiguous(), self._softmax_target(soft), v


def pretrain(*, shards_dir: str, out_path: str, size: str = "large",
             epochs: int = 40, batch_size: int = 1024, lr: float = 3e-4,
             value_weight: float = 0.5, teacher_temp: float = 0.10,
             steps_per_epoch: int | None = None, device: str | None = None,
             save_every: int = 5, global_pool: bool | int | None = True,
             stem_padding: int | None = 1) -> str:
    """``global_pool`` / ``stem_padding`` MUST match the warm-start cell's model
    or run_sweep --resume can't load the weights. Defaults match the
    G15-fixed-openings cell (global_pool=True, stem_padding=1, size large)."""
    import torch
    from gomoku.train import policy_loss, value_loss

    dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[pretrain] device={dev} size={size} global_pool={global_pool} "
          f"stem_padding={stem_padding} board={BOARD_SIZE}", flush=True)

    planes, soft = load_shards(shards_dir)
    data = PretrainData(planes, soft, device=dev, teacher_temp=teacher_temp)
    steps = steps_per_epoch or max(1, data.n // batch_size)

    model = build_model(size, global_pool=global_pool,
                        stem_padding=stem_padding).to(dev)
    print(f"[pretrain] model {size}: {n_params(model):,} params, "
          f"{steps} steps/epoch x {epochs} epochs", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)

    model.train()
    t0 = time.time()
    for ep in range(epochs):
        # Accumulate losses ON-DEVICE and sync ONCE per epoch. A per-step
        # float(loss) forced an MPS->CPU sync every step, flushing the command
        # buffer and serializing the GPU (host sat ~7% CPU blocked on syncs).
        pl_acc = torch.zeros((), device=dev)
        vl_acc = torch.zeros((), device=dev)
        for _ in range(steps):
            p, pi, z = data.sample(batch_size)
            logits, value = model(p)
            pl = policy_loss(logits, pi)
            vl = value_loss(value, z)
            loss = pl + value_weight * vl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            pl_acc += pl.detach()
            vl_acc += vl.detach()
        pl_sum = float(pl_acc)  # single sync per epoch
        vl_sum = float(vl_acc)
        print(f"[pretrain] epoch {ep+1:>3d}/{epochs}  policy_ce={pl_sum/steps:.4f}  "
              f"value_mse={vl_sum/steps:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
              f"{time.time()-t0:.0f}s", flush=True)
        if (ep + 1) % save_every == 0 or ep + 1 == epochs:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            save_checkpoint(out_path, model, opt, epoch=ep + 1, total_games=0,
                            extra={"pretrain": {"source": shards_dir, "size": size,
                                                "positions": int(data.n),
                                                "value_weight": value_weight,
                                                "teacher_temp": teacher_temp}})
            print(f"[pretrain] saved {out_path} @ epoch {ep+1}", flush=True)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="mine output dir (sharded npz)")
    ap.add_argument("--out", required=True, help="checkpoint path to write")
    ap.add_argument("--size", default="large")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--teacher-temp", type=float, default=0.10)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-every", type=int, default=5)
    # Architecture — MUST match the warm-start cell (defaults = G15-fixed-openings).
    ap.add_argument("--global-pool", type=int, default=1,
                    help="1=on (G15-fixed-openings), 0=off, or trailing-K int")
    ap.add_argument("--stem-padding", type=int, default=1)
    args = ap.parse_args(argv)
    if BOARD_SIZE != 15:
        print("WARNING: GOMOKU_BOARD_SIZE != 15", file=sys.stderr)
    gp: bool | int = bool(args.global_pool) if args.global_pool in (0, 1) else args.global_pool
    pretrain(shards_dir=args.shards, out_path=args.out, size=args.size,
             epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             value_weight=args.value_weight, teacher_temp=args.teacher_temp,
             steps_per_epoch=args.steps_per_epoch, device=args.device,
             save_every=args.save_every, global_pool=gp, stem_padding=args.stem_padding)
    return 0


if __name__ == "__main__":
    sys.exit(main())
