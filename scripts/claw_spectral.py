"""Training-free spectral (Fourier / reciprocal-lattice) detector for "the claw".

The claw is the mod-5 knight's-move lattice  L = {(x,y) : 2x + y ≡ 0 (mod 5)}
(see wiki/topics/the-claw.md). Its defining plane-wave is e^{2πi(2x+y)/5}, i.e.
normalized 2-D spatial frequency (f_x, f_y) = (2/5, 1/5) plus its reciprocal-
lattice partners. On a 15x15 board these frequencies land EXACTLY on the DFT
grid (1/5 = 3/15), so there is no off-grid leakage and no zero-padding needed.

Recipe (wiki/topics/molecule-discovery-toolkit.md, "#3 spectral claw detector"):
  occupancy -> mean-subtract -> P = |FFT2|^2 -> sum power in the claw Bragg bins,
  normalize by mean power (dimensionless enrichment), test vs a permutation null.

Run:  GOMOKU_BOARD_SIZE=15 uv run --with matplotlib python scripts/claw_spectral.py
This file does NOT touch training, the GPU, or any checkpoint. Seconds of CPU.
"""

from __future__ import annotations

import os

import numpy as np

N = int(os.environ.get("GOMOKU_BOARD_SIZE", "15"))  # honour the env knob; default 15


# --------------------------------------------------------------------------- #
# The reciprocal lattice: which DFT bins carry the claw's signature.
# --------------------------------------------------------------------------- #
def claw_bins(n: int = N) -> list[tuple[int, int]]:
    """Return the non-DC DFT bins where the perfect claw concentrates power.

    Derivation (exact, no approximation):
        1[2x+y ≡ 0 (mod 5)] = (1/5) * sum_{k=0..4} exp(2πi * k(2x+y)/5)
    so the k-th plane wave has frequency (2k/5, k/5).  Scaled onto an n-grid the
    bin is ( (2k mod 5)*n/5 , (k mod 5)*n/5 ).  k=0 is DC (dropped after
    mean-subtraction); k=1..4 give the four Bragg peaks.  For n=15 (n/5=3):
        k=1 -> (6, 3)    k=2 -> (12, 6)    k=3 -> (3, 9)    k=4 -> (9, 12)
    The set is closed under (a,b) -> (n-a, n-b) (conjugate symmetry of a real
    image), i.e. two conjugate pairs {(6,3),(9,12)} and {(12,6),(3,9)}.
    """
    assert n % 5 == 0, f"board size {n} must be a multiple of 5 for on-grid bins"
    s = n // 5
    bins = []
    for k in range(1, 5):
        bins.append(((2 * k % 5) * s, (k % 5) * s))
    # sanity: closed under negation mod n
    sset = set(bins)
    for a, b in bins:
        assert ((n - a) % n, (n - b) % n) in sset, "bins not conjugate-closed"
    return bins


CLAW_BINS = claw_bins()


# --------------------------------------------------------------------------- #
# Core detector
# --------------------------------------------------------------------------- #
def power_spectrum(occ: np.ndarray) -> np.ndarray:
    """Mean-subtracted power spectrum |FFT2|^2 (so DC does not dominate)."""
    occ = occ.astype(np.float64)
    occ = occ - occ.mean()
    return np.abs(np.fft.fft2(occ)) ** 2


def claw_score(occ: np.ndarray, bins=CLAW_BINS) -> float:
    """Dimensionless enrichment: mean power in claw bins / mean power overall.

    1.0 = the claw bins are no louder than a typical bin; >>1 = sharp peaks
    sitting exactly on the reciprocal lattice.  We exclude the (0,0) DC bin from
    the background mean (it is ~0 after mean-subtraction anyway).
    """
    P = power_spectrum(occ)
    claw_power = np.mean([P[i, j] for (i, j) in bins])
    mask = np.ones_like(P, dtype=bool)
    mask[0, 0] = False
    background = P[mask].mean()
    if background <= 0:
        return 0.0
    return float(claw_power / background)


def permutation_test(occ: np.ndarray, n_perm: int = 200, seed: int = 0):
    """Permutation null: keep the stone COUNT, shuffle WHERE they sit.

    Returns (real_score, z, percentile, null_mean, null_std).  Random scatters
    have no reason to favour the claw bins, so this calibrates 'how surprising'
    the real enrichment is.
    """
    rng = np.random.default_rng(seed)
    n_stones = int(occ.sum())
    flat_n = occ.size
    real = claw_score(occ)
    null = np.empty(n_perm)
    for t in range(n_perm):
        flat = np.zeros(flat_n)
        flat[rng.choice(flat_n, size=n_stones, replace=False)] = 1.0
        null[t] = claw_score(flat.reshape(occ.shape))
    mu, sd = null.mean(), null.std()
    z = (real - mu) / sd if sd > 0 else float("inf")
    pct = 100.0 * (null < real).mean()
    return real, z, pct, mu, sd


# --------------------------------------------------------------------------- #
# Synthetic inputs (the teaching set)
# --------------------------------------------------------------------------- #
def make_perfect_claw(n: int = N) -> np.ndarray:
    """(a) Positive control: every cell with (2x+y) % 5 == 0 filled."""
    occ = np.zeros((n, n))
    for x in range(n):
        for y in range(n):
            if (2 * x + y) % 5 == 0:
                occ[x, y] = 1.0
    return occ


def make_lines(n: int = N) -> np.ndarray:
    """(b) Lines of 5: a horizontal, a vertical, and a diagonal 5-in-a-row.

    The teaching point: lines are loud on AXIS / DIAGONAL frequencies, NOT on
    the claw's off-axis Bragg bins.
    """
    occ = np.zeros((n, n))
    occ[2, 3:8] = 1.0  # horizontal 5-in-a-row (row 2, cols 3..7)
    occ[5:10, 11] = 1.0  # vertical 5-in-a-row (col 11, rows 5..9)
    for i in range(5):  # main-diagonal 5-in-a-row
        occ[8 + i, 2 + i] = 1.0
    return occ


def make_random(n: int = N, density: float = 0.2, seed: int = 1) -> np.ndarray:
    """(c) Negative control: random scatter at density ~1/5 (same as the claw)."""
    rng = np.random.default_rng(seed)
    n_stones = round(density * n * n)
    occ = np.zeros(n * n)
    occ[rng.choice(n * n, size=n_stones, replace=False)] = 1.0
    return occ.reshape(n, n)


def random_game_stones(n: int = N, plies: int = 50, seed: int = 0) -> np.ndarray:
    """Optional 'real-ish' position: one color's stones after a random game.

    A cheap proxy for a real defensive position (no engine / no GPU). Alternate
    random legal placements for `plies` half-moves, return the side-to-move-0
    occupancy. Honestly labelled: random play is not skilled defense.
    """
    rng = np.random.default_rng(seed)
    empties = list(range(n * n))
    rng.shuffle(empties)
    chosen = empties[:plies]
    occ = np.zeros(n * n)
    occ[chosen[0::2]] = 1.0  # "black" stones only (one color)
    return occ.reshape(n, n)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(cases, out_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fftshift maps frequency index k -> array index (k + n//2) % n
    half = N // 2
    shifted_bins = [((kx + half) % N, (ky + half) % N) for (kx, ky) in CLAW_BINS]

    nrows = len(cases)
    fig, axes = plt.subplots(nrows, 2, figsize=(8.5, 3.6 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    for r, (label, occ) in enumerate(cases):
        ax_b, ax_s = axes[r]

        # --- board ---
        ax_b.imshow(occ, cmap="Greys", origin="upper", vmin=0, vmax=1)
        ax_b.set_title(f"{label}\n(board: occupancy, {int(occ.sum())} stones)", fontsize=9)
        ax_b.set_xticks([]); ax_b.set_yticks([])

        # --- spectrum (log, fftshifted) ---
        P = power_spectrum(occ)
        Psh = np.fft.fftshift(P)
        ax_s.imshow(np.log1p(Psh), cmap="magma", origin="upper")
        # mark the claw Bragg bins
        ys = [b[0] for b in shifted_bins]
        xs = [b[1] for b in shifted_bins]
        ax_s.scatter(xs, ys, s=140, facecolors="none", edgecolors="cyan",
                     linewidths=1.8, label="claw Bragg bins")
        ax_s.scatter([half], [half], s=40, marker="+", c="lime")  # DC
        sc = claw_score(occ)
        ax_s.set_title(f"power spectrum (log, DC centered)\nclaw_score = {sc:.1f}", fontsize=9)
        ax_s.set_xticks([]); ax_s.set_yticks([])
        if r == 0:
            ax_s.legend(loc="upper right", fontsize=7, framealpha=0.85)

    fig.suptitle(
        "Spectral claw detector — claw=sharp OFF-axis peaks, line=ON-axis peaks, random=mush\n"
        f"claw Bragg bins (n={N}): {CLAW_BINS}  (cyan rings); green + = DC",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    print(f"\nFigure saved -> {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print(f"Board size n = {N}")
    print(f"Claw Bragg bins (non-DC): {CLAW_BINS}")
    print("These are the reciprocal-lattice peaks of L = {(x,y): 2x+y ≡ 0 mod 5}.\n")

    cases = [
        ("(a) perfect claw", make_perfect_claw()),
        ("(b) lines of 5", make_lines()),
        ("(c) random density-1/5", make_random()),
    ]

    print(f"{'case':<26}{'stones':>7}{'claw_score':>12}{'z-score':>10}{'percentile':>12}")
    print("-" * 67)
    for label, occ in cases:
        real, z, pct, mu, sd = permutation_test(occ, n_perm=200, seed=7)
        print(f"{label:<26}{int(occ.sum()):>7}{real:>12.2f}{z:>10.2f}{pct:>11.1f}%")
    print(f"\n(permutation null: 200 reshuffles of the same stone count; "
          f"null claw_score ~ 1.0 by construction)")

    out = ("/private/tmp/claude-501/-Users-jason-code-gomoku/"
           "edbcef19-6597-4d42-8cd9-3d6b59ac2d25/scratchpad/claw_spectrum.png")
    make_figure(cases, out)

    # ---- optional: a handful of 'real-ish' random-game positions ----
    print("\nOptional: random-game positions (one color) vs the null "
          "(cheap proxy, NOT skilled defense):")
    print(f"{'game seed':<26}{'stones':>7}{'claw_score':>12}{'z-score':>10}{'percentile':>12}")
    print("-" * 67)
    for s in range(6):
        occ = random_game_stones(plies=50, seed=s)
        real, z, pct, mu, sd = permutation_test(occ, n_perm=200, seed=100 + s)
        print(f"{'random game #' + str(s):<26}{int(occ.sum()):>7}"
              f"{real:>12.2f}{z:>10.2f}{pct:>11.1f}%")


if __name__ == "__main__":
    main()
