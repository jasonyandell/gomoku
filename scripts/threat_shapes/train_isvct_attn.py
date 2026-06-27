"""is-VCT classifier — can ATTENTION read forced-VCT off a 15x15 board, on UNSEEN games?

A go/no-go feasibility test (NOT a tuned model). Consumes the shard-split dataset from
``gen_isvct_dataset.py`` and asks one question: does a small self-attention encoder, given
a side-to-move-relative 15x15 board, classify "does the mover have a forced VCT?" -- and
does it GENERALIZE to shards (games) it never trained on?

WHY SHARD SPLIT MATTERS. Train and test come from disjoint shards (the generator's
``md5(shard)%10`` rule). Consecutive plies of one game differ by a single stone, so a
position-level split would leak a near-duplicate of every test board into train and report
a fake-high score. We print the train/test shard-id intersection (must be empty) to prove it.

MODEL (small by design, < ~1M params): each of the 225 cells -> a token = embed{empty,own,opp}
+ a learned 2D (row,col) positional embedding; a few TransformerEncoder layers; a [CLS] token
-> one logit. BCE-with-logits, Adam, early-stop on a shard-disjoint val split by AUROC.

BASELINES for context (is attention earning its keep?):
  * majority      -- always predict the natural-majority class (trivial floor).
  * cnn           -- small conv net (gomoku threats are local; a strong/fair learned rival).
  * logreg        -- logistic regression on cheap hand counts (open-threes / fours / stones).

TRAINING is class-BALANCED (subsample the majority to 50/50); the held-out TEST is reported
BOTH balanced AND at the natural base rate (imbalance not hidden). Metrics per model: base
rate, accuracy, precision, recall, F1, AUROC, confusion matrix.

Run (after gen_isvct_dataset.py):
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.train_isvct_attn \
      --data ~/data/puzzle_miner/isvct_exp --out ~/data/puzzle_miner/isvct_exp
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

from gomoku.board_config import BOARD_SIZE as N

NN = N * N


# --------------------------------------------------------------------------- data
def _unpack(X_packed: np.ndarray) -> np.ndarray:
    """(M,PB) uint8 packed -> (M,2,N,N) bool."""
    bits = np.unpackbits(X_packed, axis=1)[:, : 2 * NN]
    return bits.reshape(-1, 2, N, N).astype(bool)


def _tokens(boards: np.ndarray) -> np.ndarray:
    """(M,2,N,N) bool -> (M,NN) int64 token ids: 0 empty, 1 own(mover), 2 opp."""
    own = boards[:, 0].reshape(len(boards), -1)
    opp = boards[:, 1].reshape(len(boards), -1)
    tok = np.zeros((len(boards), NN), dtype=np.int64)
    tok[own] = 1
    tok[opp] = 2
    return tok


def _val_shard(name: str, val_mod: int = 10) -> bool:
    """Among TRAIN shards, carve a shard-disjoint val split (md5%10==1)."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % val_mod == 1


def load_split(data_dir, split):
    d = np.load(os.path.join(data_dir, f"isvct_{split}.npz"))
    return d["X"], d["y"].astype(np.int64), d["shard_id"].astype(np.int64)


def balance(idx, y, rng, cap_per_class=0):
    """Subsample to equal pos/neg over the given index set."""
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    k = min(len(pos), len(neg))
    if cap_per_class:
        k = min(k, cap_per_class)
    pos = rng.permutation(pos)[:k]
    neg = rng.permutation(neg)[:k]
    out = np.concatenate([pos, neg])
    return rng.permutation(out)


# --------------------------------------------------------------------------- models
class AttnClassifier(nn.Module):
    def __init__(self, dim=96, heads=4, layers=3, ff_mult=4, drop=0.1):
        super().__init__()
        self.tok = nn.Embedding(3, dim)
        self.row = nn.Embedding(N, dim)
        self.col = nn.Embedding(N, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)
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

    def forward(self, tok):  # tok: (B, NN) int64
        x = self.tok(tok) + self.row(self.rr)[None] + self.col(self.cc)[None]
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.enc(x)
        return self.head(self.norm(x[:, 0])).squeeze(-1)


class CNNClassifier(nn.Module):
    def __init__(self, ch=48, blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(2, ch, 3, padding=1)
        self.body = nn.ModuleList()
        for _ in range(blocks):
            self.body.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch)))
        self.relu = nn.ReLU()
        self.head = nn.Linear(ch, 1)

    def forward(self, x):  # x: (B,2,N,N) float
        h = self.relu(self.stem(x))
        for blk in self.body:
            h = self.relu(h + blk(h))
        h = h.mean(dim=(2, 3))
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- metrics
def auroc(y, p):
    """Rank-based AUROC (Mann-Whitney U); no sklearn dependency."""
    y = np.asarray(y); p = np.asarray(p)
    npos = int(y.sum()); nneg = int(len(y) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def metrics_at(y, p, thr=0.5):
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"base_rate": float(y.mean()), "n": int(len(y)), "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1, "auroc": auroc(y, p),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


# --------------------------------------------------------------------------- counts feature (logreg baseline)
def _line_features(boards: np.ndarray) -> np.ndarray:
    """Cheap per-board threat counts on all 4 axes, for own(mover) and opp:
    #stones, #open-twos, #threes(any), #open-threes, #fours(any). 10 features total.
    A deliberately simple hand baseline (no learning of shape, just counts)."""
    M = len(boards)
    feats = np.zeros((M, 10), dtype=np.float32)
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    own = boards[:, 0]; opp = boards[:, 1]
    empty = ~(own | opp)
    for side, plane in ((0, own), (1, opp)):
        stones = plane.reshape(M, -1).sum(1)
        feats[:, side * 5 + 0] = stones
        # slide windows of length 5; count patterns by #own and #empty in window
        for dr, dc in dirs:
            for r0 in range(N):
                for c0 in range(N):
                    rs = [r0 + dr * k for k in range(5)]
                    cs = [c0 + dc * k for k in range(5)]
                    if not (0 <= rs[-1] < N and 0 <= cs[-1] < N):
                        continue
                    win_own = np.zeros(M, dtype=np.int32)
                    win_emp = np.zeros(M, dtype=np.int32)
                    for r, c in zip(rs, cs):
                        win_own += plane[:, r, c].astype(np.int32)
                        win_emp += empty[:, r, c].astype(np.int32)
                    # window with no opp stones (win_own+win_emp==5)
                    clean = (win_own + win_emp == 5)
                    feats[:, side * 5 + 1] += clean & (win_own == 2)
                    feats[:, side * 5 + 2] += clean & (win_own == 3)
                    feats[:, side * 5 + 3] += clean & (win_own == 3) & (win_emp == 2)
                    feats[:, side * 5 + 4] += clean & (win_own == 4)
    return feats


# --------------------------------------------------------------------------- train loop
def train_torch(model, Xtr_tok, ytr, Xva_tok, yva, device, is_cnn, args, name):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    n = len(ytr)
    best_auc, best_state, bad = -1.0, None, 0
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(n)
        t0 = time.time()
        tot = 0.0
        for s in range(0, n, args.batch):
            bi = perm[s:s + args.batch]
            xb = torch.from_numpy(Xtr_tok[bi]).to(device)
            yb = ytr_t[bi].to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        # val AUROC
        pva = predict(model, Xva_tok, device, args.batch)
        va = auroc(yva, pva)
        print(f"  [{name}] ep{ep} loss={tot/n:.4f} val_auroc={va:.4f} ({time.time()-t0:.1f}s)",
              flush=True)
        if va > best_auc + 1e-4:
            best_auc = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  [{name}] early stop (best val_auroc={best_auc:.4f})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


@torch.no_grad()
def predict(model, X_tok, device, batch):
    model.eval()
    out = []
    for s in range(0, len(X_tok), batch):
        xb = torch.from_numpy(X_tok[s:s + batch]).to(device)
        out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dir with isvct_{train,test}.npz + shards.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--cnn-ch", type=int, default=48)
    ap.add_argument("--cnn-blocks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cap-per-class", type=int, default=40000,
                    help="cap balanced train per class (keeps training to minutes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.backends.mps.is_available()
                          or args.device != "mps" else "cpu")
    t_start = time.time()

    # ---- load
    Xtr_p, ytr_all, sid_tr = load_split(args.data, "train")
    Xte_p, yte_all, sid_te = load_split(args.data, "test")
    manifest = json.load(open(os.path.join(args.data, "shards.json")))
    id2name = {s["id"]: s["name"] for s in manifest["shards"]}

    # ---- prove shard disjointness
    tr_shards = set(np.unique(sid_tr).tolist())
    te_shards = set(np.unique(sid_te).tolist())
    overlap = tr_shards & te_shards
    print(f"[split] train_shards={len(tr_shards)} test_shards={len(te_shards)} "
          f"overlap={len(overlap)} (MUST be 0)", flush=True)

    # ---- carve shard-disjoint val out of train
    is_val = np.array([_val_shard(id2name[s]) for s in sid_tr])
    tr_idx = np.flatnonzero(~is_val)
    va_idx = np.flatnonzero(is_val)
    n_val_shards = len({s for s in tr_shards if _val_shard(id2name[s])})
    print(f"[split] train positions={len(tr_idx)} val positions={len(va_idx)} "
          f"(val from {n_val_shards} shards) test positions={len(yte_all)}", flush=True)

    # ---- balance train + val
    tr_bal = balance(tr_idx, ytr_all, rng, cap_per_class=args.cap_per_class)
    va_bal = balance(va_idx, ytr_all, rng, cap_per_class=6000)
    print(f"[data] balanced train={len(tr_bal)} (pos={int(ytr_all[tr_bal].sum())}) "
          f"balanced val={len(va_bal)}", flush=True)

    # ---- decode boards/tokens once
    boards_tr_all = _unpack(Xtr_p)
    boards_te_all = _unpack(Xte_p)
    tok_tr_all = _tokens(boards_tr_all)
    tok_te_all = _tokens(boards_te_all)

    Xtr_tok = tok_tr_all[tr_bal]; ytr = ytr_all[tr_bal]
    Xva_tok = tok_tr_all[va_bal]; yva = ytr_all[va_bal]

    # natural + balanced test index sets
    te_nat = np.arange(len(yte_all))
    te_bal = balance(te_nat, yte_all, rng)
    print(f"[data] test natural n={len(te_nat)} pos_rate={yte_all.mean():.4f} | "
          f"balanced n={len(te_bal)}", flush=True)

    results = {}

    def report(name, p_nat):
        m_nat = metrics_at(yte_all, p_nat)
        m_bal = metrics_at(yte_all[te_bal], p_nat[te_bal])
        results[name] = {"test_natural": m_nat, "test_balanced": m_bal}
        print(f"\n=== {name} ===", flush=True)
        for tag, m in (("natural", m_nat), ("balanced", m_bal)):
            c = m["confusion"]
            print(f"  [{tag}] base={m['base_rate']:.3f} n={m['n']} acc={m['accuracy']:.3f} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                  f"AUROC={m['auroc']:.3f} | TP={c['tp']} FP={c['fp']} TN={c['tn']} FN={c['fn']}",
                  flush=True)

    # ---- majority baseline (always predict natural-majority class)
    maj = 1 if ytr.mean() > 0.5 else 0  # balanced train -> ~0.5; natural majority is NEG
    # use natural-train majority for the trivial floor:
    nat_majority = 1 if ytr_all[tr_idx].mean() > 0.5 else 0
    p_maj = np.full(len(yte_all), float(nat_majority))
    report(f"majority(predict={nat_majority})", p_maj)

    # ---- logreg on counts
    print("\n[logreg] computing line features...", flush=True)
    tf0 = time.time()
    Ftr = _line_features(boards_tr_all[tr_bal])
    Fte = _line_features(boards_te_all)
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-6
    Ftr = (Ftr - mu) / sd; Fte = (Fte - mu) / sd
    lr = nn.Linear(Ftr.shape[1], 1)
    optlr = torch.optim.Adam(lr.parameters(), lr=1e-2, weight_decay=1e-4)
    lf = nn.BCEWithLogitsLoss()
    Ftr_t = torch.from_numpy(Ftr); ytr_t = torch.from_numpy(ytr.astype(np.float32))
    for _ in range(300):
        optlr.zero_grad(); loss = lf(lr(Ftr_t).squeeze(-1), ytr_t); loss.backward(); optlr.step()
    with torch.no_grad():
        p_lr = torch.sigmoid(lr(torch.from_numpy(Fte)).squeeze(-1)).numpy()
    print(f"[logreg] done ({time.time()-tf0:.1f}s)", flush=True)
    report("logreg_counts", p_lr)

    # ---- CNN baseline
    print("\n[cnn] training...", flush=True)
    Xtr_cnn = boards_tr_all[tr_bal].astype(np.float32)
    Xva_cnn = boards_tr_all[va_bal].astype(np.float32)
    Xte_cnn = boards_te_all.astype(np.float32)
    cnn = CNNClassifier(ch=args.cnn_ch, blocks=args.cnn_blocks)
    print(f"[cnn] params={sum(p.numel() for p in cnn.parameters()):,}", flush=True)
    cnn, _ = train_torch(cnn, Xtr_cnn, ytr, Xva_cnn, yva, device, True, args, "cnn")
    p_cnn = predict(cnn, Xte_cnn, device, args.batch)
    report("cnn", p_cnn)
    torch.save(cnn.state_dict(), os.path.join(args.out, "isvct_cnn.pt"))

    # ---- Attention model (the headline)
    print("\n[attn] training...", flush=True)
    attn = AttnClassifier(dim=args.dim, heads=args.heads, layers=args.layers)
    nparam = sum(p.numel() for p in attn.parameters())
    print(f"[attn] params={nparam:,}", flush=True)
    attn, best_va = train_torch(attn, Xtr_tok, ytr, Xva_tok, yva, device, False, args, "attn")
    p_attn = predict(attn, tok_te_all, device, args.batch)
    report("attention", p_attn)
    torch.save(attn.state_dict(), os.path.join(args.out, "isvct_attn.pt"))

    # ---- write metrics
    out = {
        "config": vars(args),
        "device": str(device),
        "board_size": N,
        "shard_disjoint": {"train_shards": len(tr_shards), "test_shards": len(te_shards),
                           "overlap": len(overlap), "val_shards": n_val_shards},
        "dataset": {"train_positions_clean": int(len(ytr_all)),
                    "train_pos_rate": float(ytr_all.mean()),
                    "test_positions_clean": int(len(yte_all)),
                    "test_pos_rate": float(yte_all.mean()),
                    "balanced_train_n": int(len(tr_bal)),
                    "balanced_test_n": int(len(te_bal))},
        "attn_params": int(nparam),
        "attn_best_val_auroc": float(best_va),
        "results": results,
        "wall_secs": round(time.time() - t_start, 1),
    }
    mpath = os.path.join(args.out, "isvct_metrics.json")
    with open(mpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] metrics -> {mpath}  wall={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
