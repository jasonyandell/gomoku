"""GPU forcing-threes + bitmask defense (the v0 70%, lifted onto Metal).

One thread per (board, cell m). For an empty m it does the 2-ply threat assembly
that v0 did on the Python host per-three-per-node:
  * enumerate follow-ups f collinear with m (treating m as own),
  * for each f, compute its open-four completions (treating m,f as own) -> if >=2,
    f is an open-four threat with defeating set {f} ∪ comps (a 256-bit mask),
  * emit  is_three (>=1 threat),  reply_mask = OR of the defeating sets (the sound
    AND-node reply set),  is_fork = a disjoint defeating-set pair exists (win-in-2).

Soundness (defender-no-five-after-m), four-move exclusion, and the tempo guard are
applied by the CALLER (cheaply, from detect output) — see search/wavefront — so this
kernel is the pure structural three-detector. Reply_mask ORs ALL threats (the union
must be complete for AND-node soundness); fork only inspects the first K (a missed
fork is conservative-safe, a false fork would be a false win).

Validated against ``threes_ref`` (itself validated against ``gomoku.vcf``).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from scripts.vct_metal import detect_ref as D

N = D.N

_HEADER = """
inline bool mk5(device const uchar* own, int N, uint base,
                int r0, int c0, int dr, int dc, int mcell, int fcell) {
    // run of own (incl. hypothetical m,f) through (r0,c0) placed there, >=5 along (dr,dc)?
    int cnt = 1;
    int rr = r0 + dr, cc = c0 + dc;
    while (rr>=0&&rr<N&&cc>=0&&cc<N) { int idx=rr*N+cc;
        if (own[base+idx]!=0||idx==mcell||idx==fcell){cnt++;rr+=dr;cc+=dc;} else break; }
    rr = r0 - dr; cc = c0 - dc;
    while (rr>=0&&rr<N&&cc>=0&&cc<N) { int idx=rr*N+cc;
        if (own[base+idx]!=0||idx==mcell||idx==fcell){cnt++;rr-=dr;cc-=dc;} else break; }
    return cnt >= 5;
}

inline int comps_of_f(device const uchar* own, device const uchar* opp, int N, uint base,
                      int fcell, int mcell, thread uint* dmask) {
    // OR the five-completion cells of an own play at f (with m,f as own) into dmask;
    // return their count. Mirrors vcf._completions_through (break on opp/oob, walk own).
    int fr = fcell / N, fc = fcell % N;
    const int dr4[4] = {0,1,1,1};
    const int dc4[4] = {1,0,1,-1};
    int cnt = 0;
    for (int d=0; d<4; d++) {
        int dr=dr4[d], dc=dc4[d];
        for (int s=-1; s<=1; s+=2) {
            for (int k=1; k<5; k++) {
                int rr=fr+s*dr*k, cc=fc+s*dc*k;
                if (rr<0||rr>=N||cc<0||cc>=N) break;          // off-board: stop ray
                int idx = rr*N+cc;
                bool own_here = (own[base+idx]!=0)||idx==mcell||idx==fcell;
                if (own_here) continue;                        // our stone: keep walking
                if (opp[base+idx]!=0) break;                   // opponent: stop ray
                if (mk5(own,N,base,rr,cc,dr,dc,mcell,fcell)) { // empty completion
                    int w=idx>>5, bb=idx&31; uint bit=(1u<<bb);
                    if ((dmask[w]&bit)==0){ dmask[w]|=bit; cnt++; }
                }
            }
        }
    }
    return cnt;
}
"""

_SRC = """
    uint gid = thread_position_in_grid.x;
    const int N = __N__;
    const int NN = N*N;
    uint b = gid / NN;
    int m = int(gid % NN);
    uint base = b * NN;

    for (int w=0; w<8; w++) reply[gid*8+w] = 0;
    is_three[gid] = 0;
    is_fork[gid] = 0;
    if (own[gid]!=0 || opp[gid]!=0) return;     // m must be empty

    int mr = m/N, mc = m%N;
    const int dr4[4] = {0,1,1,1};
    const int dc4[4] = {1,0,1,-1};

    uint rmask[8] = {0,0,0,0,0,0,0,0};
    const int K = 12;
    uint tmasks[K*8];
    int ntm = 0;
    int nthreats = 0;

    // follow-ups f collinear with m within dist 4 (skip occupied, stop ray off-board)
    for (int d=0; d<4; d++) {
        int dr=dr4[d], dc=dc4[d];
        for (int s=-1; s<=1; s+=2) {
            for (int k=1; k<5; k++) {
                int fr=mr+s*dr*k, fc=mc+s*dc*k;
                if (fr<0||fr>=N||fc<0||fc>=N) break;
                int fcell = fr*N+fc;
                if (own[base+fcell]!=0 || opp[base+fcell]!=0) continue;  // f empty
                uint dmask[8] = {0,0,0,0,0,0,0,0};
                int comps = comps_of_f(own, opp, N, base, fcell, m, dmask);
                if (comps >= 2) {                          // open-four threat
                    int w=fcell>>5, bb=fcell&31; dmask[w]|=(1u<<bb);  // add f itself
                    for (int i=0;i<8;i++) rmask[i] |= dmask[i];
                    if (ntm < K) { for (int i=0;i<8;i++) tmasks[ntm*8+i]=dmask[i]; ntm++; }
                    nthreats++;
                }
            }
        }
    }

    if (nthreats == 0) return;
    is_three[gid] = 1;
    for (int w=0; w<8; w++) reply[gid*8+w] = rmask[w];

    bool fork = false;
    for (int i=0; i<ntm && !fork; i++)
        for (int j=i+1; j<ntm && !fork; j++) {
            bool disj = true;
            for (int w=0; w<8; w++) if ((tmasks[i*8+w] & tmasks[j*8+w]) != 0) { disj=false; break; }
            if (disj) fork = true;
        }
    is_fork[gid] = fork ? 1 : 0;
"""

_KERNEL = mx.fast.metal_kernel(
    name="forcing_threes",
    input_names=["own", "opp"],
    output_names=["is_three", "is_fork", "reply"],
    source=_SRC.replace("__N__", str(N)),
    header=_HEADER.replace("__N__", str(N)),
)

NWORDS = (N * N + 31) // 32  # 8 for 15x15


def forcing_threes_gpu(own: np.ndarray, opp: np.ndarray):
    """(B,N,N) bool own/opp -> per (board,cell):
        is_three (B,N,N) uint8, is_fork (B,N,N) uint8, reply (B,N,N,8) uint32 bitmask.
    Structural (no soundness/four-exclusion/tempo — caller applies those)."""
    B = own.shape[0]
    own_m = mx.array(np.ascontiguousarray(own).reshape(-1).astype(np.uint8))
    opp_m = mx.array(np.ascontiguousarray(opp).reshape(-1).astype(np.uint8))
    total = B * N * N
    is_three, is_fork, reply = _KERNEL(
        inputs=[own_m, opp_m],
        grid=(total, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(total,), (total,), (total * 8,)],
        output_dtypes=[mx.uint8, mx.uint8, mx.uint32],
    )
    mx.eval(is_three, is_fork, reply)
    return (np.array(is_three).reshape(B, N, N),
            np.array(is_fork).reshape(B, N, N),
            np.array(reply).reshape(B, N, N, 8))


def reply_words_to_cells(reply_row: np.ndarray) -> set[int]:
    """Decode an 8-word uint32 bitmask (reply[b,m]) into the set of cell indices."""
    cells = set()
    for w in range(8):
        word = int(reply_row[w])
        while word:
            bit = word & (-word)
            cells.add(w * 32 + bit.bit_length() - 1)
            word ^= bit
    return cells
