"""SECONDARY: the residual-won corpus + a first-look MI bond map.

We bucket midgame positions (side-to-move = plane 0) into:
  * FORCING-WON : a sound VCF proof exists (GPU ``solve_vcf_batch``). The win is
                  a continuous-fours line -- collinear by construction.
  * RESIDUAL-WON: the side to move is JUDGED won by a strong checkpoint's value
                  head (>= a high threshold) BUT no forcing (VCF) proof exists.
                  This is the non-line frontier: won, but not by a forcing line.

The "won" label here is APPROXIMATE -- a value-head estimate, not an oracle. We
flag it as such. (VCT proofs would shrink the residual further; we note this.)

We save the residual-won corpus to .npz (it is the input for the next stage's
DCA / shape-clustering) and run a cheap first-look MI bond map: attacker-stone
pairwise mutual information by RELATIVE OFFSET (shift-invariant), residual vs
forcing, against a within-position permutation null -- to ask whether any
NON-COLLINEAR offset couples above chance in the residual set.

Run:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.residual \
      --checkpoint checkpoints/dagger_r1.pt --out <dir> --n 12000 --thr 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from gomoku.board_config import BOARD_SIZE as N
from gomoku.state_ops import HISTORY_PLY
from gomoku.game import N_INPUT_PLANES
from gomoku.model import load_checkpoint
from scripts.gpu_vcf_prototype import solve_vcf_batch, pick_device
from scripts.threat_shapes.build_corpus import gen_gravity


def boards_to_planes(boards: np.ndarray) -> np.ndarray:
    """(B,2,N,N) static boards -> (B,P,N,N) network input with EMPTY history.

    Approximation: no move history (zeros) -- fine for a value estimate, flagged.
    """
    B = boards.shape[0]
    out = np.zeros((B, N_INPUT_PLANES, N, N), dtype=np.float32)
    out[:, 0] = boards[:, 0]
    out[:, HISTORY_PLY] = boards[:, 1]
    out[:, 2 * HISTORY_PLY] = 1.0
    return out


@torch.no_grad()
def value_head(model, boards, device, batch=4096):
    vals = []
    for i in range(0, len(boards), batch):
        x = torch.as_tensor(boards_to_planes(boards[i:i + batch]), device=device)
        _, v = model(x)[:2]
        vals.append(v.float().cpu().numpy())
    return np.concatenate(vals)


def mi_bond_map(boards, max_off=4, n_perm=200, seed=0):
    """Attacker-stone MI by relative offset vs a within-position label-shuffle null.

    For each offset (dr,dc) we measure the normalized co-occurrence lift of an
    attacker stone at i AND at i+offset over the product of marginals, averaged
    over positions; the null shuffles each board's attacker stones to random
    empty cells (preserving stone count) so any line/structure is destroyed.
    Returns a dict offset -> {lift, z, collinear}.
    """
    rng = np.random.default_rng(seed)
    atk = boards[:, 0].astype(np.float32)            # (B,N,N)
    B = atk.shape[0]
    offs = [(dr, dc) for dr in range(-max_off, max_off + 1)
            for dc in range(0, max_off + 1) if (dr, dc) != (0, 0) and not (dr < 0 and dc == 0)]

    def cooc(a):
        out = {}
        for (dr, dc) in offs:
            sh = np.zeros_like(a)
            r0, r1 = max(0, -dr), min(N, N - dr)
            c0, c1 = max(0, -dc), min(N, N - dc)
            sh[:, r0:r1, c0:c1] = a[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            both = (a * sh).sum((1, 2))
            out[(dr, dc)] = both.mean()
        return out

    obs = cooc(atk)
    # null: per board, scatter the same #stones to random cells
    null_samples = {o: [] for o in offs}
    counts = atk.reshape(B, -1).sum(1).astype(int)
    for _ in range(n_perm):
        a2 = np.zeros_like(atk)
        for b in range(B):
            idx = rng.choice(N * N, size=counts[b], replace=False)
            a2[b].reshape(-1)[idx] = 1.0
        c = cooc(a2)
        for o in offs:
            null_samples[o].append(c[o])
    res = {}
    for o in offs:
        arr = np.array(null_samples[o])
        mu, sd = arr.mean(), arr.std()
        z = (obs[o] - mu) / sd if sd > 1e-9 else 0.0
        dr, dc = o
        collinear = (dr == 0 or dc == 0 or abs(dr) == abs(dc))
        res[o] = {"obs": float(obs[o]), "null_mean": float(mu),
                  "z": float(z), "collinear": bool(collinear)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/dagger_r1.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = pick_device()
    t0 = time.time()
    model, _ = load_checkpoint(args.checkpoint, device=dev)
    model.eval()

    boards = gen_gravity(args.n, args.seed)
    won, cap = solve_vcf_batch(boards, max_depth=16)
    vals = value_head(model, boards, dev)

    forcing = won & ~cap
    residual = (vals >= args.thr) & ~won & ~cap         # judged won, no VCF proof
    res_boards = boards[residual]
    np.savez_compressed(os.path.join(args.out, "residual_corpus.npz"),
                        boards=res_boards, values=vals[residual])
    np.savez_compressed(os.path.join(args.out, "forcing_corpus.npz"),
                        boards=boards[forcing], values=vals[forcing])

    # first-look MI bond map on residual vs forcing (cap sizes for speed)
    rb = res_boards[:1500]
    fb = boards[forcing][:1500]
    mi_res = mi_bond_map(rb) if len(rb) >= 100 else {}
    mi_for = mi_bond_map(fb) if len(fb) >= 100 else {}

    def top_noncollinear(mi):
        items = [(o, d) for o, d in mi.items() if not d["collinear"]]
        items.sort(key=lambda kv: -kv[1]["z"])
        return [{"offset": list(o), "z": round(d["z"], 2),
                 "obs": round(d["obs"], 4)} for o, d in items[:8]]

    def top_any(mi):
        items = sorted(mi.items(), key=lambda kv: -kv[1]["z"])
        return [{"offset": list(o), "z": round(d["z"], 2),
                 "collinear": d["collinear"]} for o, d in items[:8]]

    stats = {
        "checkpoint": args.checkpoint, "value_threshold": args.thr,
        "n_generated": int(args.n),
        "n_forcing_won": int(forcing.sum()),
        "n_residual_won_approx": int(residual.sum()),
        "value_won_flag": "APPROXIMATE (checkpoint value head, not an oracle)",
        "mi_residual_top_any": top_any(mi_res),
        "mi_residual_top_noncollinear": top_noncollinear(mi_res),
        "mi_forcing_top_any": top_any(mi_for),
        "t_s": round(time.time() - t0, 1),
        "residual_corpus": os.path.join(args.out, "residual_corpus.npz"),
    }
    with open(os.path.join(args.out, "residual_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
