"""D4-invariance sanity checks for the threat-shape pipeline.

The molecule thesis demands D4 invariance ("a fork is a fork anywhere / in any
orientation"). Two cheap checks:

  1. classify_threat is D4-EQUIVARIANT: rotating/reflecting the board and the
     move yields the same threat KIND and the gain/cost/rest squares map under
     the same transform.
  2. d4_kmeans assignment is D4-INVARIANT: a patch and any of its 8 D4 images get
     the same best-of-8 distance to any centroid (so they cluster together).

Run: GOMOKU_BOARD_SIZE=15 uv run python -m scripts.threat_shapes.test_d4
"""

from __future__ import annotations

import numpy as np

from gomoku import state_ops
from scripts.threat_shapes.threats import classify_threat, rc
from scripts.threat_shapes import shapes as S

N = state_ops.BOARD_SIZE


def _sym_board(board, sym):
    rot, flip = sym % 4, sym // 4
    b = board[:, :, ::-1] if flip else board
    return np.rot90(b, rot, axes=(1, 2)).copy()


def _sym_cell(m, sym):
    """Map flat cell index under the same D4 transform as _sym_board."""
    grid = np.arange(N * N).reshape(N, N)
    g = grid[:, ::-1] if (sym // 4) else grid
    g = np.rot90(g, sym % 4)
    # g[r,c] now holds the ORIGINAL index that lands at (r,c); invert it
    inv = np.empty(N * N, dtype=int)
    inv[g.reshape(-1)] = np.arange(N * N)
    return int(inv[m])


def test_classify_equivariant():
    rng = np.random.default_rng(0)
    # build a board with an attacker open-three-ish line and a defender stone
    board = np.zeros((2, N, N), bool)
    board[0, 7, 4] = board[0, 7, 5] = board[0, 7, 6] = True   # three on a row
    board[1, 2, 2] = True
    m = 7 * N + 7   # extend the line -> a four-ish move
    base = classify_threat(board[0], board[1], m)
    ok = True
    for sym in range(8):
        sb = _sym_board(board, sym)
        sm = _sym_cell(m, sym)
        t = classify_threat(sb[0], sb[1], sm)
        if t.kind != base.kind:
            ok = False
            print(f"  sym {sym}: kind {t.kind} != {base.kind}")
        # gain/cost/rest map under the same transform
        exp_cost = sorted(_sym_cell(c, sym) for c in base.cost)
        if sorted(t.cost) != exp_cost:
            ok = False
            print(f"  sym {sym}: cost {sorted(t.cost)} != {exp_cost}")
    print(f"classify_threat D4-equivariant: {'PASS' if ok else 'FAIL'} "
          f"(base kind={base.kind}, cost={base.cost})")
    return ok


def test_kmeans_invariant():
    rng = np.random.default_rng(1)
    patch = (rng.random((3, 9, 9)) < 0.2).astype(np.float32)
    patch[2] = 1 - (patch[0].astype(bool) | patch[1].astype(bool))
    centroids = rng.random((5, 3, 9, 9)).astype(np.float32)
    T = np.stack(S.d4_transforms(patch))
    dists = []
    for img in T:
        Ti = np.stack(S.d4_transforms(img))
        _, _, d = S._best_align(Ti, centroids)
        dists.append(round(d, 4))
    ok = len(set(dists)) == 1
    print(f"d4_kmeans best-of-8 distance D4-invariant: {'PASS' if ok else 'FAIL'} "
          f"(dists={dists})")
    return ok


if __name__ == "__main__":
    a = test_classify_equivariant()
    b = test_kmeans_invariant()
    print("ALL PASS" if (a and b) else "SOME FAILED")
