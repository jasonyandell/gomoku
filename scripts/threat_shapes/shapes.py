"""Shape extraction + D4-aware class averaging (cryo-EM-style) for threat patches.

Two patch sets are cropped from the replayed forcing lines:
  * THREAT patches    -- a (2R+1)x(2R+1) window centred on each threat's GAIN
                         square. Isolated line-threats are collinear by
                         construction, so these mostly recover Allis's line
                         vocabulary (the validation case).
  * JUNCTION patches  -- windows centred on squares where >= 2 threats meet
                         (shared gain/cost = the fork/combination geometry). This
                         is where genuine 2-D (non-collinear) structure can show.

Each patch is 3 channels (attacker, defender, empty) one-hot per cell, in the
ATTACKER-centric frame (channel 0 = the side that owns the threat).

Clustering = cryo-EM 2-D class averaging with the finite D4 pose group (8
elements) instead of continuous SO(2): k-means where each patch is assigned to
the centroid minimising distance over its 8 D4 transforms (best-of-8 hard
argmax), and each centroid is updated as the mean of its members in their chosen
(aligned) pose. Centroids become soft 3-symbol stencils; we render them as
information-weighted opacity "threat logos".
"""

from __future__ import annotations

import numpy as np

# ---- D4 group on (C, H, W) patches -------------------------------------------

def d4_transforms(patch: np.ndarray) -> list[np.ndarray]:
    """All 8 dihedral images of a (C, H, W) patch (4 rotations x optional flip)."""
    out = []
    for flip in (False, True):
        p = patch[:, :, ::-1] if flip else patch
        for k in range(4):
            out.append(np.rot90(p, k, axes=(1, 2)).copy())
    return out


def crop_patch(board: np.ndarray, center: int, radius: int) -> np.ndarray:
    """(3, 2R+1, 2R+1) one-hot patch centred on flat cell ``center``.

    board is (2, N, N): plane0 attacker, plane1 defender. Channels:
    0=attacker, 1=defender, 2=empty. Off-board cells are encoded as empty.
    """
    N = board.shape[-1]
    R = radius
    S = 2 * R + 1
    cr, cc = center // N, center % N
    atk = np.zeros((S, S), bool)
    dfd = np.zeros((S, S), bool)
    for i in range(S):
        rr = cr - R + i
        if not (0 <= rr < N):
            continue
        for j in range(S):
            ccx = cc - R + j
            if not (0 <= ccx < N):
                continue
            atk[i, j] = board[0, rr, ccx]
            dfd[i, j] = board[1, rr, ccx]
    emp = ~(atk | dfd)
    return np.stack([atk, dfd, emp]).astype(np.float32)


# ---- D4-aware k-means (cryo-EM class averaging) ------------------------------

def _best_align(patch_t: np.ndarray, centroids: np.ndarray):
    """Return (best_centroid_idx, best_transform_idx, best_dist) for one patch.

    patch_t: (8, C, H, W) the 8 D4 images. centroids: (K, C, H, W).
    """
    # dist[t,k] = ||patch_t[t] - centroids[k]||^2
    pt = patch_t.reshape(8, -1)               # (8, D)
    cf = centroids.reshape(centroids.shape[0], -1)  # (K, D)
    # (8,K) = ||p||^2 + ||c||^2 - 2 p.c
    d = (pt * pt).sum(1)[:, None] + (cf * cf).sum(1)[None, :] - 2 * pt @ cf.T
    t, k = np.unravel_index(np.argmin(d), d.shape)
    return int(k), int(t), float(d[t, k])


def d4_kmeans(patches: np.ndarray, k: int, *, iters: int = 25, seed: int = 0,
              restarts: int = 3):
    """D4-aware k-means class averaging (vectorised).

    patches: (M, C, H, W) one-hot float. Returns (centroids (K,C,H,W),
    labels (M,), aligned (M,C,H,W) the chosen-pose patches, inertia, counts).
    Each E-step assigns every patch to the centroid minimising distance over its
    8 D4 poses (hard best-of-8 argmax) in one batched matmul; the M-step averages
    members in their chosen pose. Runs ``restarts`` inits, keeps lowest inertia.
    """
    M = patches.shape[0]
    C, H, W = patches.shape[1:]
    D = C * H * W
    # Pre-compute the 8 D4 images of every patch once -> (M*8, D).
    T = np.stack([np.stack(d4_transforms(p)) for p in patches])  # (M,8,C,H,W)
    Tf = T.reshape(M * 8, D)
    Tsq = (Tf * Tf).sum(1)                                       # (M*8,)

    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        cen = patches[rng.choice(M, size=k, replace=False)].copy()
        inertia = np.inf
        labels = np.zeros(M, int)
        tsel = np.zeros(M, int)
        counts = np.zeros(k, int)
        for _ in range(iters):
            cf = cen.reshape(k, D)
            # dist^2 (M*8, k) = ||T||^2 + ||c||^2 - 2 T.c
            d = Tsq[:, None] + (cf * cf).sum(1)[None, :] - 2.0 * Tf @ cf.T
            d = d.reshape(M, 8, k)
            # best over poses for each centroid, then best centroid
            best_pose = d.argmin(1)                              # (M,k)
            best_pose_d = np.take_along_axis(d, best_pose[:, None, :], 1)[:, 0, :]
            labels = best_pose_d.argmin(1)                       # (M,)
            tsel = best_pose[np.arange(M), labels]               # (M,)
            new_inertia = float(best_pose_d[np.arange(M), labels].sum())
            # M-step
            aligned = Tf.reshape(M, 8, D)[np.arange(M), tsel]    # (M,D)
            newcen = np.zeros((k, D))
            counts = np.zeros(k, int)
            np.add.at(newcen, labels, aligned)
            np.add.at(counts, labels, 1)
            for c in range(k):
                if counts[c] > 0:
                    newcen[c] /= counts[c]
                else:
                    newcen[c] = patches[rng.integers(M)].reshape(D)
            cen = newcen.reshape(k, C, H, W)
            if abs(inertia - new_inertia) < 1e-4:
                inertia = new_inertia
                break
            inertia = new_inertia
        if best is None or inertia < best[3]:
            aligned_full = Tf.reshape(M, 8, D)[np.arange(M), tsel].reshape(M, C, H, W)
            best = (cen, labels.copy(), aligned_full, inertia, counts.copy())
    return best


def cell_information(centroid: np.ndarray) -> np.ndarray:
    """Per-cell information in bits (max log2(3)) from a 3-symbol stencil.

    centroid: (3, H, W) probabilities summing ~1 per cell. Returns (H, W) info.
    """
    p = np.clip(centroid, 1e-9, 1.0)
    p = p / p.sum(0, keepdims=True)
    ent = -(p * np.log2(p)).sum(0)
    return np.log2(3) - ent


def match_centroids(cen_a: np.ndarray, cen_b: np.ndarray) -> tuple[float, list]:
    """Reproducibility across two corpus halves (Einstein-from-noise guard).

    For each centroid in A, find the nearest centroid in B over D4 alignment;
    return mean best-match distance + the per-cluster nearest distances. A small,
    tight matching means the dictionary is reproducible (not fit to noise).
    """
    dists = []
    for ca in cen_a:
        T = np.stack(d4_transforms(ca))
        best = min(
            float(((T - cb[None]) ** 2).reshape(8, -1).sum(1).min())
            for cb in cen_b
        )
        dists.append(best)
    return float(np.mean(dists)), dists
