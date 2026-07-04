"""Phi distance-to-VCT field — can a net REGRESS closeness-to-a-forced-win, and does the
GLOBAL target finally give attention its edge over a locality-biased CNN?

Consumes `gen_phi_dataset.py`'s shard-split dataset. Two scalar regressions per board:
  phi_off = gamma**(mover-moves to its nearest future VCT)   -- the OFFENSE potential
  phi_def = gamma**(opp-moves to the opponent's next VCT)     -- the DEFENSE potential

WHY THIS EXPERIMENT (vct-reachability-mining.md §1; vct-recognition / seeker docs §4):
recognition and BC-steering are *local* targets and a CNN beat attention on both. Phi is
"closeness to a fork", a WHOLE-BOARD fact -> "global by construction" -> the fair arena to
re-audition attention vs a CNN. This is also the first real L2 (the AlphaZero/steering layer)
model: a verifiable, non-bootstrapped potential (labels are oracle distances, not self-play
value, so no overestimation).

METRICS on the HELD-OUT shard-disjoint test set, per head, per model:
  * R2            -- 1 - MSE/Var  (variance explained)
  * Spearman rho  -- rank correlation pred vs true (does it ORDER positions by closeness?
                     this is what a steering gradient actually needs)
  * reach AUROC   -- treat (phi>0) i.e. "a VCT is reachable at all" as a label; can the net
                     separate reachable from floor? (coverage half of the signal)
  * calibration   -- mean true phi in each predicted-phi decile (printed)
Models: mean-predictor (floor), linear-on-flattened-board (is it trivially linear?), CNN
(GAP regression head), attention (token encoder + mean-pool regression head). Same
md5(shard)%10 split as the sibling experiments; md5%10==1 val carve drives early stopping.

Run (after gen_phi_dataset.py):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.train_phi \
      --data ~/data/puzzle_miner/phi_exp --out ~/data/puzzle_miner/phi_exp
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


# --------------------------------------------------------------------------- data
def _unpack(X_packed: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(X_packed, axis=1)[:, : 2 * NN]
    return bits.reshape(-1, 2, N, N).astype(bool)


def _tokens(boards: np.ndarray) -> np.ndarray:
    """(M,2,N,N) bool -> (M,NN) int64: 0 empty, 1 own(mover), 2 opp."""
    own = boards[:, 0].reshape(len(boards), -1)
    opp = boards[:, 1].reshape(len(boards), -1)
    tok = np.zeros((len(boards), NN), dtype=np.int64)
    tok[own] = 1
    tok[opp] = 2
    return tok


def _val_shard(name: str, val_mod: int = 10) -> bool:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % val_mod == 1


def load_split(data_dir, split):
    d = np.load(os.path.join(data_dir, f"phi_{split}.npz"))
    y = np.stack([d["phi_off"], d["phi_def"]], axis=1).astype(np.float32)  # (M,2)
    return (d["X"], y, d["shard_id"].astype(np.int64),
            d["d_off"].astype(np.int64), d["par"].astype(np.int64))


# --------------------------------------------------------------------------- models
class CNNReg(nn.Module):
    """Residual conv tower + global-average-pool -> 2 scalar potentials in [0,1]."""
    def __init__(self, ch=64, blocks=5):
        super().__init__()
        self.stem = nn.Conv2d(2, ch, 3, padding=1)
        self.body = nn.ModuleList()
        for _ in range(blocks):
            self.body.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch)))
        self.relu = nn.ReLU()
        self.head = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, 2))

    def forward(self, x):  # (B,2,N,N) -> (B,2)
        h = self.relu(self.stem(x))
        for blk in self.body:
            h = self.relu(h + blk(h))
        h = h.mean(dim=(2, 3))            # global average pool
        return torch.sigmoid(self.head(h))


class AttnReg(nn.Module):
    """Per-cell self-attention encoder + mean-pool -> 2 scalar potentials in [0,1].

    A learned CLS-free global summary: the global receptive field that locality-biased convs
    lack. If 'closeness to a fork' is genuinely global, this is where attention should win."""
    def __init__(self, dim=96, heads=4, layers=3, ff_mult=4, drop=0.1):
        super().__init__()
        self.tok = nn.Embedding(3, dim)
        self.row = nn.Embedding(N, dim)
        self.col = nn.Embedding(N, dim)
        self.register_buffer("rr", torch.arange(NN) // N)
        self.register_buffer("cc", torch.arange(NN) % N)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult,
            dropout=drop, batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 2))

    def forward(self, tok):  # (B,NN) int64 -> (B,2)
        x = self.tok(tok) + self.row(self.rr)[None] + self.col(self.cc)[None]
        x = self.norm(self.enc(x)).mean(dim=1)   # mean-pool tokens -> board embedding
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------- metrics
def _spearman(pred: np.ndarray, true: np.ndarray) -> float:
    # rank-transform both, Pearson on ranks (ties -> ordinal ranks, fine at this scale)
    rp = np.argsort(np.argsort(pred)).astype(np.float64)
    rt = np.argsort(np.argsort(true)).astype(np.float64)
    rp -= rp.mean(); rt -= rt.mean()
    den = np.sqrt((rp * rp).sum() * (rt * rt).sum())
    return float((rp * rt).sum() / den) if den > 0 else 0.0


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    var = ((true - true.mean()) ** 2).mean()
    mse = ((pred - true) ** 2).mean()
    return float(1.0 - mse / var) if var > 0 else 0.0


def _auroc(score: np.ndarray, label: np.ndarray) -> float:
    # rank-based AUROC; label in {0,1}
    n_pos = int(label.sum()); n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _calib(pred: np.ndarray, true: np.ndarray, bins=10) -> list:
    q = np.quantile(pred, np.linspace(0, 1, bins + 1))
    q[-1] += 1e-6
    out = []
    for i in range(bins):
        m = (pred >= q[i]) & (pred < q[i + 1])
        if m.any():
            out.append([round(float(pred[m].mean()), 3), round(float(true[m].mean()), 3),
                        int(m.sum())])
    return out


def evaluate(name, pred, Y, d_off, results, calib_store):
    """pred,Y: (M,2) off/def. Report per head."""
    rec = {}
    for h, tag in ((0, "off"), (1, "def")):
        p, t = pred[:, h], Y[:, h]
        reach = (t > 0).astype(np.int64)
        rec[tag] = {
            "r2": round(_r2(p, t), 4),
            "spearman": round(_spearman(p, t), 4),
            "reach_auroc": round(_auroc(p, reach), 4),
            "pred_mean": round(float(p.mean()), 4),
            "true_mean": round(float(t.mean()), 4),
        }
        calib_store[f"{name}_{tag}"] = _calib(p, t)
    results[name] = rec
    o, d = rec["off"], rec["def"]
    print(f"=== {name:10s} | OFF r2={o['r2']:+.3f} rho={o['spearman']:+.3f} "
          f"reachAUC={o['reach_auroc']:.3f} | DEF r2={d['r2']:+.3f} rho={d['spearman']:+.3f} "
          f"reachAUC={d['reach_auroc']:.3f}", flush=True)


# --------------------------------------------------------------------------- train loop
def train_torch(model, get_x, Y, tr, va, device, args, name):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    yt = torch.from_numpy(Y)
    best_rho, best_state, bad = -2.0, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        t0 = time.time(); tot = 0.0
        for s in range(0, len(tr), args.batch):
            bi = tr[perm[s:s + args.batch]]
            xb = get_x(bi).to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yt[bi].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        pv = predict(model, get_x, va, device, args.batch)
        rho = 0.5 * (_spearman(pv[:, 0], Y[va, 0]) + _spearman(pv[:, 1], Y[va, 1]))
        print(f"  [{name}] ep{ep} mse={tot/len(tr):.4f} val_rho={rho:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if rho > best_rho + 1e-4:
            best_rho, bad = rho, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  [{name}] early stop (best val_rho={best_rho:.4f})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_rho


@torch.no_grad()
def predict(model, get_x, idx, device, batch):
    model.eval()
    out = []
    for s in range(0, len(idx), batch):
        bi = idx[s:s + batch]
        out.append(model(get_x(bi).to(device)).float().cpu().numpy())
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
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cap-train", type=int, default=300000)
    ap.add_argument("--cap-val", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available())
                          else "cpu")
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    Xtr_p, Ytr, sid_tr, doff_tr, par_tr = load_split(args.data, "train")
    Xte_p, Yte, sid_te, doff_te, par_te = load_split(args.data, "test")
    manifest = json.load(open(os.path.join(args.data, "phi_shards.json")))
    id2name = {s["id"]: s["name"] for s in manifest["shards"]}

    tr_shards = set(np.unique(sid_tr).tolist()); te_shards = set(np.unique(sid_te).tolist())
    overlap = tr_shards & te_shards
    print(f"[split] train_shards={len(tr_shards)} test_shards={len(te_shards)} "
          f"overlap={len(overlap)} (MUST be 0)", flush=True)

    boards_tr = _unpack(Xtr_p); boards_te = _unpack(Xte_p)
    Xtr = boards_tr.astype(np.float32); Xte = boards_te.astype(np.float32)
    tok_tr = _tokens(boards_tr); tok_te = _tokens(boards_te)
    flat_tr = boards_tr.reshape(len(boards_tr), -1).astype(np.float32)
    flat_te = boards_te.reshape(len(boards_te), -1).astype(np.float32)
    print(f"[data] train_ex={len(Ytr)} test_ex={len(Yte)} "
          f"off_floor_tr={(doff_tr<0).mean():.3f} off_floor_te={(doff_te<0).mean():.3f} "
          f"phi_off_mean_te={Yte[:,0].mean():.3f} phi_def_mean_te={Yte[:,1].mean():.3f}",
          flush=True)

    is_val = np.array([_val_shard(id2name[s]) for s in sid_tr])
    tr_idx = np.flatnonzero(~is_val); va_idx = np.flatnonzero(is_val)
    n_val_shards = len({s for s in tr_shards if _val_shard(id2name[s])})
    if len(tr_idx) > args.cap_train:
        tr_idx = rng.permutation(tr_idx)[:args.cap_train]
    if len(va_idx) > args.cap_val:
        va_idx = rng.permutation(va_idx)[:args.cap_val]
    print(f"[split] train_idx={len(tr_idx)} val_idx={len(va_idx)} (val from {n_val_shards} shards) "
          f"test={len(Yte)}", flush=True)

    results, calib = {}, {}
    te_all = np.arange(len(Yte))

    # ---- floors
    print("\n[baselines]", flush=True)
    mean_pred = np.tile(Ytr.mean(axis=0, keepdims=True), (len(Yte), 1))
    evaluate("mean", mean_pred, Yte, doff_te, results, calib)

    # linear-on-flattened-board (closed-form ridge, per head) — is phi trivially linear?
    lam = 10.0
    A = flat_tr[tr_idx]
    G = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32)
    Ginv = np.linalg.inv(G)
    Wlin = Ginv @ (A.T @ Ytr[tr_idx])           # (2NN,2)
    lin_pred = np.clip(flat_te @ Wlin, 0.0, 1.0)
    evaluate("linear", lin_pred, Yte, doff_te, results, calib)

    # closures bound to the right array (train vs test handled by passing arrays)
    def make_getx(arr_cnn=None, arr_tok=None, cnn=True):
        src = arr_cnn if cnn else arr_tok
        return lambda idx: torch.from_numpy(src[idx])

    # ---- CNN
    print("\n[cnn] training...", flush=True)
    cnn = CNNReg(ch=args.cnn_ch, blocks=args.cnn_blocks)
    print(f"[cnn] params={sum(p.numel() for p in cnn.parameters()):,}", flush=True)
    cnn, cnn_rho = train_torch(cnn, make_getx(arr_cnn=Xtr, cnn=True), Ytr, tr_idx, va_idx,
                               device, args, "cnn")
    evaluate("cnn", predict(cnn, make_getx(arr_cnn=Xte, cnn=True), te_all, device, args.batch),
             Yte, doff_te, results, calib)
    torch.save(cnn.state_dict(), os.path.join(args.out, "phi_cnn.pt"))

    # ---- Attention (the audition)
    print("\n[attn] training...", flush=True)
    attn = AttnReg(dim=args.dim, heads=args.heads, layers=args.layers)
    nparam = sum(p.numel() for p in attn.parameters())
    print(f"[attn] params={nparam:,}", flush=True)
    attn, attn_rho = train_torch(attn, make_getx(arr_tok=tok_tr, cnn=False), Ytr, tr_idx, va_idx,
                                 device, args, "attn")
    evaluate("attention", predict(attn, make_getx(arr_tok=tok_te, cnn=False), te_all, device,
                                  args.batch), Yte, doff_te, results, calib)
    torch.save(attn.state_dict(), os.path.join(args.out, "phi_attn.pt"))

    out = {
        "config": vars(args), "device": str(device), "board_size": N,
        "gamma": manifest.get("gamma"),
        "shard_disjoint": {"train_shards": len(tr_shards), "test_shards": len(te_shards),
                           "overlap": len(overlap), "val_shards": n_val_shards},
        "dataset": {"train_ex": int(len(Ytr)), "test_ex": int(len(Yte)),
                    "train_used": int(len(tr_idx)), "val_used": int(len(va_idx)),
                    "off_floor_frac_test": float((doff_te < 0).mean()),
                    "phi_off_mean_test": float(Yte[:, 0].mean()),
                    "phi_def_mean_test": float(Yte[:, 1].mean())},
        "cnn_best_val_rho": float(cnn_rho), "attn_best_val_rho": float(attn_rho),
        "attn_params": int(nparam),
        "results": results, "calibration": calib,
        "wall_secs": round(time.time() - t_start, 1),
    }
    mpath = os.path.join(args.out, "phi_metrics.json")
    with open(mpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] metrics -> {mpath}  wall={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
