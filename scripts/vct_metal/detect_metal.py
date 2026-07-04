"""First real MSL kernel: batched OR-node detection on the Apple GPU.

One GPU thread per (board, cell) computes, for an empty cell m, exactly what
``detect_ref`` computes in numpy:
  * five[m]    : 1 if placing ``own`` at m makes >= 5 in a row
  * n_comp[m]  : # distinct five-completions of the four a play at m would create
  * block[m]   : a completion cell index (the forced block) when n_comp>=1, else -1

This is a direct port of ``detect_ref.five_completion_mask`` + ``four_structure``
into Metal Shading Language, run via MLX's runtime-compiled custom kernel. It is
the per-node primitive the search will call; here it is validated standalone
against the numpy spec (which is itself validated against ``gomoku.vcf``).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from scripts.vct_metal import detect_ref as D

N = D.N

_SRC = """
    uint gid = thread_position_in_grid.x;
    const int N = __N__;
    const int NN = N * N;
    uint b = gid / NN;
    int cell = int(gid % NN);
    int r = cell / N;
    int c = cell % N;
    uint base = b * NN;

    if (own[gid] != 0 || opp[gid] != 0) {
        five[gid] = 0; n_comp[gid] = 0; block[gid] = -1; return;
    }

    const int dr4[4] = {0, 1, 1, 1};
    const int dc4[4] = {1, 0, 1, -1};

    // --- five: placing own at (r,c) makes >= 5 along some axis ---
    bool mk5 = false;
    for (int d = 0; d < 4; d++) {
        int dr = dr4[d], dc = dc4[d];
        int cnt = 1;
        int rr = r + dr, cc = c + dc;
        while (rr>=0 && rr<N && cc>=0 && cc<N && own[base + rr*N + cc] != 0) { cnt++; rr+=dr; cc+=dc; }
        rr = r - dr; cc = c - dc;
        while (rr>=0 && rr<N && cc>=0 && cc<N && own[base + rr*N + cc] != 0) { cnt++; rr-=dr; cc-=dc; }
        if (cnt >= 5) mk5 = true;
    }
    five[gid] = mk5 ? 1 : 0;

    // --- four structure: count completions of the four a play at (r,c) makes ---
    int ncomp = 0;
    int blk = -1;
    const int deltas[8] = {-4, -3, -2, -1, 1, 2, 3, 4};
    for (int d = 0; d < 4; d++) {
        int dr = dr4[d], dc = dc4[d];
        for (int di = 0; di < 8; di++) {
            int delta = deltas[di];
            int cr = r + delta*dr;
            int cc2 = c + delta*dc;
            if (cr<0 || cr>=N || cc2<0 || cc2>=N) continue;       // completion on-board
            uint cidx = base + cr*N + cc2;
            if (own[cidx] != 0 || opp[cidx] != 0) continue;       // completion empty
            int jm_lo = (-delta > 0) ? -delta : 0;
            int jm_hi = (4 - delta < 4) ? (4 - delta) : 4;
            bool fired = false;
            for (int jm = jm_lo; jm <= jm_hi && !fired; jm++) {
                bool ok = true;
                for (int i = 0; i < 5; i++) {
                    int off = i - jm;
                    if (off == 0 || off == delta) continue;       // skip m and c
                    int orr = r + off*dr;
                    int occ = c + off*dc;
                    if (orr<0 || orr>=N || occ<0 || occ>=N) { ok = false; break; }
                    if (own[base + orr*N + occ] == 0) { ok = false; break; }
                }
                if (ok) fired = true;
            }
            if (fired) { ncomp++; if (blk < 0) blk = cr*N + cc2; }
        }
    }
    n_comp[gid] = ncomp;
    block[gid] = blk;
"""

_KERNEL = mx.fast.metal_kernel(
    name="or_node_detect",
    input_names=["own", "opp"],
    output_names=["five", "n_comp", "block"],
    source=_SRC.replace("__N__", str(N)),
)


def detect_or_node(own: np.ndarray, opp: np.ndarray):
    """(B,N,N) bool own/opp -> (five uint8, n_comp int32, block int32), each (B,N,N).

    Runs on the GPU. Mirrors detect_ref.five_completion_mask / four_structure.
    """
    B = own.shape[0]
    own_m = mx.array(np.ascontiguousarray(own).reshape(-1).astype(np.uint8))
    opp_m = mx.array(np.ascontiguousarray(opp).reshape(-1).astype(np.uint8))
    total = B * N * N
    five, n_comp, block = _KERNEL(
        inputs=[own_m, opp_m],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(total,), (total,), (total,)],
        output_dtypes=[mx.uint8, mx.int32, mx.int32],
    )
    mx.eval(five, n_comp, block)
    return (np.array(five).reshape(B, N, N),
            np.array(n_comp).reshape(B, N, N),
            np.array(block).reshape(B, N, N))


if __name__ == "__main__":
    import time
    from scripts.vct_metal.test_detect_ref import gen_clustered

    print("MLX device:", mx.default_device())
    bad = 0
    for seed in (0, 1, 2):
        own, opp = gen_clustered(300, seed)
        g_five, g_nc, g_blk = detect_or_node(own, opp)
        r_five = D.five_completion_mask(own, opp)
        r_nc, r_blk = D.four_structure(own, opp)

        # five and n_comp must match exactly; block only needs to be *a* valid
        # completion, so compare block only where n_comp==1 (unique completion).
        five_ok = np.array_equal(g_five.astype(bool), r_five)
        nc_ok = np.array_equal(g_nc.astype(np.int16), r_nc)
        single = r_nc == 1
        blk_ok = np.array_equal(g_blk[single], r_blk[single])
        ok = five_ok and nc_ok and blk_ok
        bad += not ok
        print(f"seed={seed}: five={five_ok} n_comp={nc_ok} block(single)={blk_ok} "
              f"-> {'PASS' if ok else 'FAIL'}")

    # rough throughput
    own, opp = gen_clustered(4096, 7)
    detect_or_node(own, opp)  # warm compile
    t = time.time()
    for _ in range(5):
        detect_or_node(own, opp)
    dt = (time.time() - t) / 5
    print(f"\nthroughput: {4096} boards/call, {dt*1000:.1f} ms/call "
          f"-> {4096/dt:,.0f} boards/s (incl. host<->GPU transfer)")
    raise SystemExit(1 if bad else 0)
