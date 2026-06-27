"""Seeker policy — can a net IMITATE the quiet-phase steering of the side that reaches a VCT,
on UNSEEN games? (Phase A of the seek-VCT thesis: the offline behavioral-cloning go/no-go.)

Consumes ``gen_seeker_dataset.py``'s shard-split dataset. One question: given a side-to-move
board in the QUIET (pre-onset) phase, predict the move the seeker S actually played on the way
to its own forced VCT — and does that prediction GENERALIZE to shards (games) never trained on?

This is a per-cell POLICY (NN-way classification with legal-move masking), the steering analog
of the is-VCT recognizer. It is a WEAK-BUT-HONEST proxy: top-1 move-match is not "good play"
(many quiet moves are reasonable, and exact-match to a strong engine is not required to steer),
but BEATING the dumb spatial priors by a margin, on unseen shards, is real evidence that there
is learnable, generalizing structure in HOW winners steer toward a forced win. The decisive test
is the hybrid-play eval (Phase C); this is the cheap green/red light before paying for it.

MODELS (small by design; the isvct result says CNN's locality prior should win — the seeker is
attention's claimed home turf, so we run both):
  * uniform   -- random legal move (the trivial floor: ~k/#legal).
  * adjacency -- play an empty cell 8-adjacent to a stone (the strong dumb gomoku prior).
  * cnn       -- residual conv tower + 1x1 policy head.
  * attention -- per-cell self-attention encoder + linear policy head (global receptive field).

SHARD-DISJOINT, identical split rule to the isvct experiment (md5(shard)%10), so the two are
directly comparable; a further md5%10==1 val carve (out of train) drives early stopping. The
train/test shard-id intersection is printed (must be 0).

Run (after gen_seeker_dataset.py):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.train_seeker \
      --data ~/data/puzzle_miner/seeker_exp --out ~/data/puzzle_miner/seeker_exp
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gomoku.board_config import BOARD_SIZE as N

NN = N * N
NEG = -1e9


# --------------------------------------------------------------------------- data
def _unpack(X_packed: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(X_packed, axis=1)[:, : 2 * NN]
    return bits.reshape(-1, 2, N, N).astype(bool)


def _tokens(boards: np.ndarray) -> np.ndarray:
    """(M,2,N,N) bool -> (M,NN) int64: 0 empty, 1 own(mover/seeker), 2 opp."""
    own = boards[:, 0].reshape(len(boards), -1)
    opp = boards[:, 1].reshape(len(boards), -1)
    tok = np.zeros((len(boards), NN), dtype=np.int64)
    tok[own] = 1
    tok[opp] = 2
    return tok


def _legal(boards: np.ndarray) -> np.ndarray:
    """(M,2,N,N) bool -> (M,NN) bool: empty cells (legal moves)."""
    occ = (boards[:, 0] | boards[:, 1]).reshape(len(boards), -1)
    return ~occ


def _adjacency_scores(boards: np.ndarray) -> np.ndarray:
    """(M,NN) float32: 1.0 if an empty cell is 8-adjacent to any stone, else 0."""
    stones = (boards[:, 0] | boards[:, 1])
    M = len(boards)
    pad = np.zeros((M, N + 2, N + 2), dtype=bool)
    pad[:, 1:-1, 1:-1] = stones
    near = np.zeros((M, N, N), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            near |= pad[:, 1 + dr:N + 1 + dr, 1 + dc:N + 1 + dc]
    adj = near & (~stones)
    return adj.reshape(M, -1).astype(np.float32)


def _val_shard(name: str, val_mod: int = 10) -> bool:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % val_mod == 1


def load_split(data_dir, split):
    d = np.load(os.path.join(data_dir, f"seeker_{split}.npz"))
    return d["X"], d["mv"].astype(np.int64), d["shard_id"].astype(np.int64), d["onset"].astype(np.int64)


# --------------------------------------------------------------------------- models
class CNNPolicy(nn.Module):
    def __init__(self, ch=64, blocks=5):
        super().__init__()
        self.stem = nn.Conv2d(2, ch, 3, padding=1)
        self.body = nn.ModuleList()
        for _ in range(blocks):
            self.body.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch)))
        self.relu = nn.ReLU()
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x):  # x: (B,2,N,N) float -> (B,NN) logits
        h = self.relu(self.stem(x))
        for blk in self.body:
            h = self.relu(h + blk(h))
        return self.head(h).reshape(x.size(0), NN)


class AttnPolicy(nn.Module):
    def __init__(self, dim=96, heads=4, layers=3, ff_mult=4, drop=0.1):
        super().__init__()
        self.tok = nn.Embedding(3, dim)
        self.row = nn.Embedding(N, dim)
        self.col = nn.Embedding(N, dim)
        rr = torch.arange(NN) // N
        cc = torch.arange(NN) % N
        self.register_buffer("rr", rr)
        self.register_buffer("cc", cc)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult,
            dropout=drop, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, tok):  # tok: (B,NN) int64 -> (B,NN) logits
        x = self.tok(tok) + self.row(self.rr)[None] + self.col(self.cc)[None]
        x = self.enc(x)
        return self.head(self.norm(x)).squeeze(-1)


# --------------------------------------------------------------------------- metrics
def topk_acc(scores: np.ndarray, mask: np.ndarray, target: np.ndarray, ks=(1, 3, 5)) -> dict:
    """Top-k legal-move-match. scores/mask (M,NN); target (M,)."""
    s = np.where(mask, scores, -np.inf)
    maxk = max(ks)
    kth = min(maxk, s.shape[1] - 1)
    part = np.argpartition(-s, kth=kth, axis=1)[:, :maxk]
    rows = np.arange(len(s))[:, None]
    order = np.argsort(-s[rows, part], axis=1)
    top = part[rows, order]  # (M,maxk) sorted desc by score
    out = {}
    for k in ks:
        out[k] = float((top[:, :k] == target[:, None]).any(axis=1).mean())
    return out


def masked_ce(scores: np.ndarray, mask: np.ndarray, target: np.ndarray) -> float:
    s = np.where(mask, scores.astype(np.float64), -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(s).sum(axis=1))
    ll = s[np.arange(len(s)), target] - logZ
    return float(-ll.mean())


# --------------------------------------------------------------------------- train loop
def train_torch(model, X, tok, ytarget, mask, tr, va, device, is_cnn, args, name):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    yt = torch.from_numpy(ytarget)
    mt = torch.from_numpy(mask)
    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        t0 = time.time(); tot = 0.0
        for s in range(0, len(tr), args.batch):
            bi = tr[perm[s:s + args.batch]]
            xb = (torch.from_numpy(X[bi]).to(device) if is_cnn
                  else torch.from_numpy(tok[bi]).to(device))
            logits = model(xb)
            logits = logits.masked_fill(~mt[bi].to(device), NEG)
            loss = F.cross_entropy(logits, yt[bi].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        sc_va = predict(model, X, tok, va, device, is_cnn, args.batch)
        acc = topk_acc(sc_va, mask[va], ytarget[va])[1]
        print(f"  [{name}] ep{ep} loss={tot/len(tr):.4f} val_top1={acc:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if acc > best_acc + 1e-4:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  [{name}] early stop (best val_top1={best_acc:.4f})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


@torch.no_grad()
def predict(model, X, tok, idx, device, is_cnn, batch):
    model.eval()
    out = []
    for s in range(0, len(idx), batch):
        bi = idx[s:s + batch]
        xb = (torch.from_numpy(X[bi]).to(device) if is_cnn
              else torch.from_numpy(tok[bi]).to(device))
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--cnn-ch", type=int, default=64)
    ap.add_argument("--cnn-blocks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cap-train", type=int, default=200000, help="cap train examples (minutes)")
    ap.add_argument("--cap-val", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.backends.mps.is_available()
                          or args.device != "mps" else "cpu")
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    # ---- load + decode
    Xtr_p, ytr, sid_tr, onset_tr = load_split(args.data, "train")
    Xte_p, yte, sid_te, onset_te = load_split(args.data, "test")
    manifest = json.load(open(os.path.join(args.data, "seeker_shards.json")))
    id2name = {s["id"]: s["name"] for s in manifest["shards"]}

    tr_shards = set(np.unique(sid_tr).tolist()); te_shards = set(np.unique(sid_te).tolist())
    overlap = tr_shards & te_shards
    print(f"[split] train_shards={len(tr_shards)} test_shards={len(te_shards)} "
          f"overlap={len(overlap)} (MUST be 0)", flush=True)

    boards_tr = _unpack(Xtr_p); boards_te = _unpack(Xte_p)
    Xtr = boards_tr.astype(np.float32); Xte = boards_te.astype(np.float32)
    tok_tr = _tokens(boards_tr); tok_te = _tokens(boards_te)
    mask_tr = _legal(boards_tr); mask_te = _legal(boards_te)
    print(f"[data] train_ex={len(ytr)} test_ex={len(yte)} "
          f"mean_legal_train={mask_tr.sum(1).mean():.1f} mean_legal_test={mask_te.sum(1).mean():.1f}",
          flush=True)

    # ---- shard-disjoint val carve out of train; cap train/val for minutes
    is_val = np.array([_val_shard(id2name[s]) for s in sid_tr])
    tr_idx = np.flatnonzero(~is_val); va_idx = np.flatnonzero(is_val)
    n_val_shards = len({s for s in tr_shards if _val_shard(id2name[s])})
    if len(tr_idx) > args.cap_train:
        tr_idx = rng.permutation(tr_idx)[:args.cap_train]
    if len(va_idx) > args.cap_val:
        va_idx = rng.permutation(va_idx)[:args.cap_val]
    print(f"[split] train_idx={len(tr_idx)} val_idx={len(va_idx)} (val from {n_val_shards} shards) "
          f"test={len(yte)}", flush=True)

    results = {}
    te_all = np.arange(len(yte))

    def report(name, scores_te):
        tk = topk_acc(scores_te, mask_te, yte)
        ce = masked_ce(scores_te, mask_te, yte) if np.isfinite(scores_te).any() else float("nan")
        results[name] = {"top1": tk[1], "top3": tk[3], "top5": tk[5], "ce": ce, "n": int(len(yte))}
        print(f"=== {name:10s} top1={tk[1]:.4f} top3={tk[3]:.4f} top5={tk[5]:.4f} ce={ce:.3f}",
              flush=True)

    # ---- dumb baselines (analytic floors, through the SAME topk)
    print("\n[baselines]", flush=True)
    rng_b = np.random.default_rng(args.seed + 1)
    report("uniform", rng_b.random((len(yte), NN)).astype(np.float32))
    adj = _adjacency_scores(boards_te)
    adj = adj + 0.5 * rng_b.random((len(yte), NN)).astype(np.float32)  # random tiebreak within groups
    report("adjacency", adj)

    # ---- CNN policy
    print("\n[cnn] training...", flush=True)
    cnn = CNNPolicy(ch=args.cnn_ch, blocks=args.cnn_blocks)
    print(f"[cnn] params={sum(p.numel() for p in cnn.parameters()):,}", flush=True)
    cnn, cnn_va = train_torch(cnn, Xtr, tok_tr, ytr, mask_tr, tr_idx, va_idx, device, True, args, "cnn")
    report("cnn", predict(cnn, Xte, tok_te, te_all, device, True, args.batch))
    torch.save(cnn.state_dict(), os.path.join(args.out, "seeker_cnn.pt"))

    # ---- Attention policy (the audition)
    print("\n[attn] training...", flush=True)
    attn = AttnPolicy(dim=args.dim, heads=args.heads, layers=args.layers)
    nparam = sum(p.numel() for p in attn.parameters())
    print(f"[attn] params={nparam:,}", flush=True)
    attn, attn_va = train_torch(attn, Xtr, tok_tr, ytr, mask_tr, tr_idx, va_idx, device, False, args, "attn")
    report("attention", predict(attn, Xte, tok_te, te_all, device, False, args.batch))
    torch.save(attn.state_dict(), os.path.join(args.out, "seeker_attn.pt"))

    out = {
        "config": vars(args), "device": str(device), "board_size": N,
        "shard_disjoint": {"train_shards": len(tr_shards), "test_shards": len(te_shards),
                           "overlap": len(overlap), "val_shards": n_val_shards},
        "dataset": {"train_ex": int(len(ytr)), "test_ex": int(len(yte)),
                    "train_used": int(len(tr_idx)), "val_used": int(len(va_idx)),
                    "mean_legal_test": float(mask_te.sum(1).mean())},
        "cnn_best_val_top1": float(cnn_va), "attn_best_val_top1": float(attn_va),
        "attn_params": int(nparam),
        "results": results,
        "wall_secs": round(time.time() - t_start, 1),
    }
    mpath = os.path.join(args.out, "seeker_metrics.json")
    with open(mpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] metrics -> {mpath}  wall={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
