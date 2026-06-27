"""Rediscover-the-claw v0 probe.

THESIS (see wiki/topics/the-claw.md, idea-pile.md "rediscover the claw"):
A RELATION-GENERAL representation (self-attention + periodic positional
encodings) can rediscover the claw lattice L = {(x,y): 2x+y == 0 (mod 5)} -- a
density-1/5 perfect blocker of every 5-in-a-row, with ZERO line content -- purely
from a learning objective, where a LINE/CNN representation (translation-equivariant
local conv, NO absolute position) provably cannot represent the mod-5 field.

We hand-code only the TASK (blocking score), never the claw or mod-5.

TASK (GPU-resident, cheap; reuses the length-5 directional conv kernels from
scratchpad/bench_gpu_vcf.py):
  blocking_score(stones) = #(5-windows, all 4 dirs) that contain >=1 stone.
  The claw maximizes this at minimal density 1/5 (perfect-blocker theorem).

SIGNALS reported:
  0. Sanity: claw blocks ALL windows; random same-count scatter blocks fewer.
  1. Gradient-ascent: maximize each trained model's predicted blocking minus a
     density penalty -> does the relation-general model lay down a mod-5 lattice?
     (claw-ness score + 2D-FFT period-5 peak). CNN control.
  2. Linear probe of cell residue (2x+y)%5 from hidden activations (chance=20%).
  3. Claw-vs-scatter discriminability of each model's predicted score.

Run:
  GOMOKU_BOARD_SIZE=15 uv run python scripts/claw_rediscovery.py
Outputs go to the session scratchpad (PNGs + ASCII).
"""
import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N = int(os.environ.get("GOMOKU_BOARD_SIZE", "15"))
SCRATCH = "/private/tmp/claude-501/-Users-jason-code-gomoku/edbcef19-6597-4d42-8cd9-3d6b59ac2d25/scratchpad"
os.makedirs(SCRATCH, exist_ok=True)

# ------------------------------------------------------------------ kernels --
# 4 directional length-5 kernels (H, V, main-diag, anti-diag). From bench_gpu_vcf.py.
def _kernels():
    ks = []
    h = torch.zeros(1, 1, 1, 5); h[...] = 1; ks.append(h)
    v = torch.zeros(1, 1, 5, 1); v[...] = 1; ks.append(v)
    d = torch.zeros(1, 1, 5, 5); a = torch.zeros(1, 1, 5, 5)
    for i in range(5):
        d[0, 0, i, i] = 1
        a[0, 0, i, 4 - i] = 1
    ks.append(d); ks.append(a)
    return [k.to(DEV) for k in ks]
KS = _kernels()

def window_sums(own):
    """own: (B,1,N,N). Returns list of (B,1,h,w) window stone-counts per dir (VALID conv)."""
    return [F.conv2d(own, k, padding=0) for k in KS]

def blocking_score_hard(own):
    """own: (B,1,N,N) float {0,1}. Returns (B,) int count of blocked 5-windows (all dirs)."""
    tot = torch.zeros(own.shape[0], device=own.device)
    for ws in window_sums(own):
        tot += (ws >= 1).float().flatten(1).sum(1)
    return tot

def num_windows():
    return sum(int((F.conv2d(torch.ones(1,1,N,N,device=DEV), k, padding=0)).numel()) for k in KS)

NW = num_windows()  # total 5-windows over all 4 directions

def blocking_score_soft(own, beta=4.0):
    """Differentiable surrogate: window blocked-ness = 1-exp(-beta*windowsum)."""
    tot = own.new_zeros(own.shape[0])
    for ws in window_sums(own):
        tot = tot + (1.0 - torch.exp(-beta * ws.clamp(min=0))).flatten(1).sum(1)
    return tot

# --------------------------------------------------------------- the claw L --
def claw_lattice(family=0):
    """L = {(x,y): 2x+y == 0 (mod 5)} (family 0) or {x+2y==0} (family 1). Returns (N,N) float."""
    xs = np.arange(N)[:, None]; ys = np.arange(N)[None, :]
    f = (2 * xs + ys) if family == 0 else (xs + 2 * ys)
    return (f % 5 == 0).astype(np.float32)

def residue_field():
    xs = np.arange(N)[:, None]; ys = np.arange(N)[None, :]
    return ((2 * xs + ys) % 5).astype(np.int64)  # (N,N) in 0..4

# --------------------------------------------------------------- dataset -----
def make_dataset(B, seed=0):
    """Random stone configs over a spread of densities. Returns own(B,1,N,N), label(B,)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dens = torch.rand(B, generator=g) * 0.55 + 0.03  # 3%..58%
    r = torch.rand(B, N * N, generator=g)
    own = (r < dens[:, None]).float().view(B, 1, N, N).to(DEV)
    label = blocking_score_hard(own) / NW  # normalized [0,1]
    return own, label

# --------------------------------------------------------------- models ------
class CNNBaseline(nn.Module):
    """Translation-equivariant local conv (line/shape representation). NO absolute
    position. Generous: kernel-5 first layer so it CAN see full 5-windows."""
    def __init__(self, ch=40):
        super().__init__()
        self.c1 = nn.Conv2d(1, ch, 5, padding=2)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.head = nn.Conv2d(ch, 1, 1)
    def features(self, x):
        h = F.relu(self.c1(x)); h = F.relu(self.c2(h)); h = F.relu(self.c3(h))
        return h  # (B,ch,N,N) per-cell features
    def forward(self, x):
        h = self.features(x)
        y = self.head(h)            # (B,1,N,N) per-cell contribution
        return y.flatten(1).mean(1) # global pooled scalar

class FourierPE(nn.Module):
    """General multi-frequency sinusoidal PE over absolute (x,y). Spread of
    frequencies (1..K cycles/board) -- NOT period-5-privileged. Capacity for
    periodicity; the model must combine frequencies to discover mod-5."""
    def __init__(self, dim, n_freq=8):
        super().__init__()
        freqs = torch.arange(1, n_freq + 1).float()  # 1..n_freq cycles across board
        self.register_buffer("freqs", freqs)
        xs = torch.arange(N).float() / N
        ys = torch.arange(N).float() / N
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        feats = []
        for f in freqs:
            for g in (gx, gy):
                feats.append(torch.sin(2 * math.pi * f * g))
                feats.append(torch.cos(2 * math.pi * f * g))
        pe = torch.stack(feats, -1).view(N * N, -1)  # (N*N, 4*n_freq)
        self.proj = nn.Linear(pe.shape[1], dim)
        self.register_buffer("pe_raw", pe)
    def forward(self):
        return self.proj(self.pe_raw)  # (N*N, dim)

class RelationGeneral(nn.Module):
    """Small self-attention over cell tokens + Fourier positional encoding."""
    def __init__(self, dim=48, heads=4, layers=2, n_freq=8):
        super().__init__()
        self.tok = nn.Linear(1, dim)
        self.pe = FourierPE(dim, n_freq)
        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                attn=nn.MultiheadAttention(dim, heads, batch_first=True),
                ln1=nn.LayerNorm(dim), ln2=nn.LayerNorm(dim),
                mlp=nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)),
            )) for _ in range(layers)
        ])
        self.head = nn.Linear(dim, 1)
    def features_pe(self, x):
        B = x.shape[0]
        toks = self.tok(x.view(B, 1, N * N).transpose(1, 2)) + self.pe()[None]
        return toks  # post-PE, pre-attention (the positional SUBSTRATE)
    def features(self, x):
        B = x.shape[0]
        toks = self.tok(x.view(B, 1, N * N).transpose(1, 2))  # (B,N*N,dim)
        toks = toks + self.pe()[None]
        for L in self.layers:
            a, _ = L["attn"](L["ln1"](toks), L["ln1"](toks), L["ln1"](toks))
            toks = toks + a
            toks = toks + L["mlp"](L["ln2"](toks))
        return toks  # (B,N*N,dim)
    def forward(self, x):
        toks = self.features(x)
        return self.head(toks).mean(1).squeeze(-1)  # pooled scalar

def nparams(m): return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def batched_predict(model, X, chunk=512):
    return torch.cat([model(X[i:i+chunk]) for i in range(0, X.shape[0], chunk)])

@torch.no_grad()
def batched_feats(get_feats, X, chunk=128):
    return torch.cat([get_feats(X[i:i+chunk]) for i in range(0, X.shape[0], chunk)])

# --------------------------------------------------------------- train -------
def train(model, steps=2500, bs=256, lr=2e-3, tag=""):
    Xtr, Ytr = make_dataset(20000, seed=1)
    Xte, Yte = make_dataset(4000, seed=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, Xtr.shape[0], (bs,), device=DEV)
        pred = model(Xtr[idx])
        loss = F.mse_loss(pred, Ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p = batched_predict(model, Xte)
        mse = F.mse_loss(p, Yte).item()
        ss_res = ((p - Yte) ** 2).sum().item()
        ss_tot = ((Yte - Yte.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / ss_tot
    print(f"  [{tag}] params={nparams(model):,} train {steps} steps in "
          f"{time.time()-t0:.1f}s  test MSE={mse:.5f}  R2={r2:.4f}", flush=True)
    return r2, mse

# ----------------------------------------------------- gradient ascent -------
def grad_ascent(scorer_units, K, mu=2.0, gmax=10.0, steps=1200, lr=0.08, seed=0):
    """Maximize scorer_units(o) - mu*(sum-K)^2 - gamma_t*sum(o*(1-o))  [FIXED-BUDGET +
    annealed binarization: 'commit K discrete stones to maximize predicted blocking'].
    The binarization term forces the optimizer to CHOOSE which cells (otherwise a uniform
    soft 0.2 density trivially soft-blocks every window). scorer_units -> window-units."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = (-1.4 + torch.randn(1, 1, N, N, generator=g) * 0.5).to(DEV).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    for t in range(steps):
        gamma = gmax * max(0.0, (t / steps - 0.3) / 0.7)  # ramp on after 30% of run
        o = torch.sigmoid(logits)
        obj = scorer_units(o) - mu * (o.sum() - K) ** 2 - gamma * (o * (1 - o)).sum()
        opt.zero_grad(); (-obj).backward(); opt.step()
    return torch.sigmoid(logits).detach().squeeze().cpu().numpy()

def model_scorer(model):
    def s(o):  # o:(1,1,N,N)->window-units (model predicts normalized, *NW)
        return model(o) * NW
    return s

# ----------------------------------------------------- structure metrics -----
def clawness(binary):
    """Fraction of stones lying on the single best residue class, maximized over
    both lattice families {2x+y,x+2y} mod 5 and all 5 phases. Pure claw -> 1.0;
    uniform random -> ~0.2."""
    ys, xs = np.where(binary > 0.5)
    if len(xs) == 0:
        return 0.0, None
    best = 0.0; bestlab = None
    for fam, (a, b) in enumerate([(2, 1), (1, 2)]):
        res = (a * ys + b * xs) % 5  # note ys=row=x-axis here; symmetric over fam anyway
        for ph in range(5):
            frac = float((res == ph).mean())
            if frac > best:
                best = frac; bestlab = (fam, ph)
    return best, bestlab

def fft_period5(field):
    """2D FFT magnitude; return ratio of energy at the period-5 ring vs mean energy.
    Period-5 on an N board ~ N/5 cycles."""
    f = field - field.mean()
    F2 = np.abs(np.fft.fftshift(np.fft.fft2(f)))
    c = N // 2
    fx = np.fft.fftshift(np.fft.fftfreq(N))  # cycles/cell
    gx, gy = np.meshgrid(fx, fx, indexing="ij")
    radial = np.sqrt(gx ** 2 + gy ** 2)
    target = 1.0 / 5.0  # period-5 -> 0.2 cycles/cell
    ring = (np.abs(radial - target) < 0.04) & (radial > 0.01)
    if ring.sum() == 0:
        return 0.0
    return float(F2[ring].max() / (F2[radial > 0.01].mean() + 1e-9))

def ascii_board(binary, title):
    lines = [title]
    for r in range(N):
        lines.append("".join("#" if binary[r, c] > 0.5 else "." for c in range(N)))
    return "\n".join(lines)

# ----------------------------------------------------- linear probe ----------
def probe_residue(get_feats, n_boards=300, seed=7):
    """Decode (2x+y)%5 per cell from hidden features. Multinomial logistic
    regression (torch). Train/test split over BOARDS. chance=20%."""
    Xb, _ = make_dataset(n_boards, seed=seed)
    feats = batched_feats(get_feats, Xb)            # (n_boards, N*N, C)
    C = feats.shape[-1]
    res = torch.tensor(residue_field().reshape(-1), device=DEV)  # (N*N,)
    res_all = res[None].expand(feats.shape[0], -1).reshape(-1)   # (n_boards*N*N,)
    X = feats.reshape(-1, C)
    # standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    nb = feats.shape[0]
    ntr = int(nb * 0.7)
    tr = torch.arange(nb * N * N, device=DEV).view(nb, N * N)[:ntr].reshape(-1)
    te = torch.arange(nb * N * N, device=DEV).view(nb, N * N)[ntr:].reshape(-1)
    clf = nn.Linear(C, 5).to(DEV)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    Xtr, ytr = X[tr], res_all[tr]
    for _ in range(300):
        idx = torch.randint(0, Xtr.shape[0], (4096,), device=DEV)
        loss = F.cross_entropy(clf(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (clf(X[te]).argmax(1) == res_all[te]).float().mean().item()
    return acc

# ============================================================ main ===========
def main():
    print(f"=== Rediscover-the-claw v0  (board N={N}, dev={DEV}) ===", flush=True)
    print(f"total 5-windows (all 4 dirs) = {NW}", flush=True)

    # ---- SIGNAL 0: ground-truth sanity ----
    print("\n[0] GROUND-TRUTH SANITY (verify the claw is the optimum before trusting models)")
    claw = torch.tensor(claw_lattice(0)).view(1, 1, N, N).to(DEV)
    n_claw = int(claw.sum().item())
    bs_claw = int(blocking_score_hard(claw).item())
    # random scatters with the SAME stone count
    rng = np.random.default_rng(0)
    scat_scores = []
    for _ in range(2000):
        b = np.zeros(N * N, np.float32)
        b[rng.choice(N * N, n_claw, replace=False)] = 1
        scat_scores.append(int(blocking_score_hard(
            torch.tensor(b).view(1, 1, N, N).to(DEV)).item()))
    scat_scores = np.array(scat_scores)
    print(f"  claw: {n_claw} stones (density {n_claw/(N*N):.3f}) blocks {bs_claw}/{NW} windows "
          f"({bs_claw/NW:.3f})  -> {'PERFECT' if bs_claw==NW else 'NOT PERFECT'}")
    print(f"  random same-count ({n_claw} stones): blocks {scat_scores.mean():.0f}+-{scat_scores.std():.0f} "
          f"(max over 2000 tries = {scat_scores.max()}, claw beats {(scat_scores<bs_claw).mean()*100:.1f}% of them)")
    # also confirm minimality: any config with fewer stones cannot block all
    print(f"  claw line-content: a 5-in-a-row needs sum==5 in a window; claw max window-sum = "
          f"{int(max(ws.max().item() for ws in window_sums(claw)))} (==1 => ZERO line content)")

    # ---- TRAIN both models ----
    print("\n[ TRAIN both representations on the blocking task ]")
    cnn = CNNBaseline().to(DEV)
    rel = RelationGeneral().to(DEV)
    r2_cnn, _ = train(cnn, tag="CNN-baseline")
    r2_rel, _ = train(rel, tag="Relation-general")

    # ---- SIGNAL 3: claw-vs-scatter discriminability of trained models ----
    print("\n[3] CLAW-vs-SCATTER discriminability (trained-model predicted score)")
    with torch.no_grad():
        pred_claw_cnn = cnn(claw).item(); pred_claw_rel = rel(claw).item()
    scat_boards = []
    for _ in range(500):
        b = np.zeros(N * N, np.float32); b[rng.choice(N * N, n_claw, replace=False)] = 1
        scat_boards.append(b)
    SB = torch.tensor(np.stack(scat_boards)).view(-1, 1, N, N).to(DEV)
    ps_cnn = batched_predict(cnn, SB).cpu().numpy(); ps_rel = batched_predict(rel, SB).cpu().numpy()
    for name, pc, ps in [("CNN", pred_claw_cnn, ps_cnn), ("Relation", pred_claw_rel, ps_rel)]:
        z = (pc - ps.mean()) / (ps.std() + 1e-9)
        pct = (ps < pc).mean() * 100
        print(f"  {name:9s}: pred(claw)={pc:.4f}  scatter={ps.mean():.4f}+-{ps.std():.4f}  "
              f"-> claw is +{z:.2f} SD, beats {pct:.1f}% of same-count scatters")

    # ---- SIGNAL 2: linear probe of residue class ----
    print("\n[2] LINEAR PROBE of cell residue (2x+y)%5 from hidden activations (chance=20%)")
    acc_cnn = probe_residue(lambda X: cnn.features(X).permute(0, 2, 3, 1).reshape(X.shape[0], N*N, -1))
    acc_rel = probe_residue(lambda X: rel.features(X))
    acc_rel_pe = probe_residue(lambda X: rel.features_pe(X))
    print(f"  CNN-baseline (final conv feats) probe acc = {acc_cnn*100:.1f}%   "
          f"(translation-equivariant: NO absolute position -> must be ~chance)")
    print(f"  Relation post-PE substrate      probe acc = {acc_rel_pe*100:.1f}%   "
          f"(Fourier PE -> residue IS in the substrate)")
    print(f"  Relation learned (final layer)  probe acc = {acc_rel*100:.1f}%   "
          f"(what the TRAINED rep actually keeps)")

    # ---- SIGNAL 1: gradient ascent (the emergence headline) ----
    print(f"[1] GRADIENT ASCENT (FIXED BUDGET K={n_claw} stones): 'where do you want "
          f"{n_claw} stones to maximize predicted blocking?'  -> does the claw emerge?")
    # TRUE-objective GA validates the landscape contains the claw & that GA can find it.
    def true_scorer(o): return blocking_score_soft(o, beta=1.5)
    targets = {}
    for name, scorer in [("TRUE-objective", true_scorer),
                         ("CNN", model_scorer(cnn)), ("Relation", model_scorer(rel))]:
        best = None; max_cn = 0.0  # best-by-actual-blocks; also track best-ever claw-ness
        for seed in range(8):
            o = grad_ascent(scorer, K=n_claw, seed=seed)
            b = (o > 0.5).astype(np.float32)
            ab = int(blocking_score_hard(torch.tensor(b).view(1,1,N,N).to(DEV)).item())
            cn_s, _ = clawness(b)
            max_cn = max(max_cn, cn_s)
            if best is None or ab > best[0]:
                best = (ab, b, o)
        ab, b, o = best
        cn, lab = clawness(b)
        fftr = fft_period5(o)
        targets[name] = (b, o)
        print(f"  {name:14s}: nstones={int(b.sum())}  claw-ness={cn:.3f} (best-ever {max_cn:.3f}; "
              f"chance .20)  ACTUAL blocks={ab}/{NW} ({ab/NW:.3f})  FFT p5 peak/mean={fftr:.2f}")

    # ---- save visualizations ----
    print("\n[viz] saving ASCII + PNG to scratchpad")
    txt = []
    txt.append(ascii_board(claw_lattice(0), f"GROUND-TRUTH claw L (2x+y==0 mod5), {n_claw} stones:"))
    for name in ["TRUE-objective", "CNN", "Relation"]:
        b, o = targets[name]
        txt.append("\n" + ascii_board(b, f"GRAD-ASCENT of {name} (thresholded):"))
    blob = "\n\n".join(txt)
    with open(os.path.join(SCRATCH, "claw_ascii.txt"), "w") as f:
        f.write(blob)
    print(blob)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(claw_lattice(0), cmap="gray_r"); ax[0].set_title("ground-truth claw")
        for i, name in enumerate(["TRUE-objective", "CNN", "Relation"]):
            ax[i+1].imshow(targets[name][1], cmap="viridis")
            ax[i+1].set_title(f"GA: {name}")
        for a in ax: a.set_xticks([]); a.set_yticks([])
        p = os.path.join(SCRATCH, "claw_gradascent.png")
        plt.tight_layout(); plt.savefig(p, dpi=110); print(f"  wrote {p}")
    except Exception as e:
        print(f"  (matplotlib unavailable: {e})")

    print("\n=== SUMMARY ===")
    print(f"  task R2:   CNN={r2_cnn:.3f}   Relation={r2_rel:.3f}")
    print(f"  residue probe: CNN={acc_cnn*100:.1f}%   Relation={acc_rel*100:.1f}%   (chance 20%)")

if __name__ == "__main__":
    main()
