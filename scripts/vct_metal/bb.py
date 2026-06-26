"""Bitboard detection as Metal (MSL) helpers — ported from bb_ref, validated vs it.

Board packed as 4 x ulong (256 bits >= N*N=225), little-endian: cell (r,c) -> bit
r*N+c -> word (idx>>6), bit (idx&63). These helpers (`shr256`, `shl256`,
`has_five`, `completion_mask`) are the device-side primitives the bitboard
megakernels are built from. This module also exposes a thin probe kernel used by
test_bb.py to validate the MSL against bb_ref bit-for-bit.

Shifts are needed only up to 4*max_step = 4*(N+1) = 64 bits, so the word shift is
always 0 or 1 — shr256/shl256 are specialised to that range.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from scripts.vct_metal import bb_ref

N = bb_ref.N
WIN = bb_ref.WIN
NN = bb_ref.NN
NWORDS = 4
TOPMASK = (1 << (NN - 192)) - 1   # valid bits in word 3 (cells 192..224)


# --------------------------------------------------------------------------- #
# pack (B,2,N,N) bool boards -> (B,4) uint64 own / opp
# --------------------------------------------------------------------------- #
def pack_words(boards: np.ndarray):
    """boards: (B,2,N,N) -> (own, opp): each (B,4) uint64."""
    B = boards.shape[0]
    fo = np.ascontiguousarray(boards[:, 0]).reshape(B, NN).astype(np.uint64)
    fp = np.ascontiguousarray(boards[:, 1]).reshape(B, NN).astype(np.uint64)
    own = np.zeros((B, NWORDS), np.uint64)
    opp = np.zeros((B, NWORDS), np.uint64)
    for idx in range(NN):
        w, bit = idx >> 6, np.uint64(idx & 63)
        own[:, w] |= fo[:, idx] << bit
        opp[:, w] |= fp[:, idx] << bit
    return own, opp


def words_to_mask(words) -> int:
    """(4,) uint64 -> Python big-int bitset (inverse of bb_ref pack via words)."""
    m = 0
    for w in range(NWORDS):
        m |= int(words[w]) << (64 * w)
    return m & bb_ref.BOARDMASK


def _dir_consts():
    """Emit MSL for the per-direction step + startmask[4] tables."""
    steps = [s for _n, s, _m in bb_ref.DIRS]
    lines = []
    lines.append("constant uint DSTEP[4] = {%s};" % ", ".join(str(s) for s in steps))
    rows = []
    for _n, _s, mask in bb_ref.DIRS:
        words = [(mask >> (64 * w)) & ((1 << 64) - 1) for w in range(NWORDS)]
        rows.append("{%s}" % ", ".join("%dul" % w for w in words))
    lines.append("constant ulong DMASK[4][4] = {%s};" % ", ".join(rows))
    return "\n".join(lines)


_HEADER = """
constant ulong TOPMASK = __TOPMASK__ul;   // valid bits in word 3
__DIRCONSTS__

inline void cpy4(thread const ulong* a, thread ulong* o){ o[0]=a[0];o[1]=a[1];o[2]=a[2];o[3]=a[3]; }
inline void and4(thread ulong* a, thread const ulong* b){ a[0]&=b[0];a[1]&=b[1];a[2]&=b[2];a[3]&=b[3]; }
inline bool any4(thread const ulong* a){ return (a[0]|a[1]|a[2]|a[3]) != 0ul; }

// logical right shift of a 256-bit little-endian bitset by s in [0,64].
inline void shr256(thread const ulong* x, uint s, thread ulong* o){
    if (s==0){ cpy4(x,o); return; }
    uint w = s>>6; uint b = s&63;
    for (uint i=0;i<4;i++){
        ulong lo = (i+w   < 4u) ? x[i+w]   : 0ul;
        ulong hi = (i+w+1u < 4u) ? x[i+w+1u] : 0ul;
        o[i] = (b==0u) ? lo : ((lo >> b) | (hi << (64u-b)));
    }
}
// logical left shift by t in [0,64], masked to the board.
inline void shl256(thread const ulong* x, uint t, thread ulong* o){
    uint w = t>>6; uint b = t&63;
    for (int i=3;i>=0;i--){
        int j = i - (int)w;
        ulong lo = (j   >= 0) ? x[j]   : 0ul;
        ulong hi = (j-1 >= 0) ? x[j-1] : 0ul;
        o[i] = (b==0u) ? lo : ((lo << b) | (hi >> (64u-b)));
    }
    o[3] &= TOPMASK;
}

// any >= WIN in a row for bitset P?
inline bool has_five(thread const ulong* P){
    for (int d=0; d<4; d++){
        uint s = DSTEP[d];
        ulong dm[4]; dm[0]=DMASK[d][0];dm[1]=DMASK[d][1];dm[2]=DMASK[d][2];dm[3]=DMASK[d][3];
        ulong t[4]; cpy4(P,t);
        ulong sh[4];
        shr256(P, s,   sh); and4(t,sh);
        shr256(P, 2u*s, sh); and4(t,sh);
        shr256(P, 3u*s, sh); and4(t,sh);
        shr256(P, 4u*s, sh); and4(t,sh);
        and4(t, dm);
        if (any4(t)) return true;
    }
    return false;
}

// empty cells (subset of `empty`) where placing a P-stone makes >= WIN.
inline void completion_mask(thread const ulong* P, thread const ulong* empty, thread ulong* comp){
    comp[0]=0ul; comp[1]=0ul; comp[2]=0ul; comp[3]=0ul;
    for (int d=0; d<4; d++){
        uint s = DSTEP[d];
        // precompute the 5 shifts of P and empty once; reuse across the 5 holes.
        ulong Ps[5][4]; ulong Es[5][4];
        for (int i=0;i<5;i++){ shr256(P,(uint)i*s,Ps[i]); shr256(empty,(uint)i*s,Es[i]); }
        for (int j=0; j<5; j++){
            ulong m[4]; m[0]=DMASK[d][0];m[1]=DMASK[d][1];m[2]=DMASK[d][2];m[3]=DMASK[d][3];
            for (int i=0; i<5; i++){
                if (i==j){ m[0]&=Es[i][0];m[1]&=Es[i][1];m[2]&=Es[i][2];m[3]&=Es[i][3]; }
                else     { m[0]&=Ps[i][0];m[1]&=Ps[i][1];m[2]&=Ps[i][2];m[3]&=Ps[i][3]; }
            }
            ulong shifted[4];
            shl256(m, (uint)(j)*s, shifted);
            comp[0]|=shifted[0]; comp[1]|=shifted[1]; comp[2]|=shifted[2]; comp[3]|=shifted[3];
        }
    }
    and4(comp, empty);
    comp[3] &= TOPMASK;
}
"""


def _header():
    return (_HEADER
            .replace("__TOPMASK__", str(TOPMASK))
            .replace("__DIRCONSTS__", _dir_consts()))


_PROBE_SRC = """
    uint gid = thread_position_in_grid.x;
    uint base = gid * 4u;
    thread ulong own[4]; thread ulong opp[4]; thread ulong empty[4];
    for (uint w=0; w<4u; w++){ own[w]=own_in[base+w]; opp[w]=opp_in[base+w]; }
    for (uint w=0; w<4u; w++){ empty[w] = (~(own[w]|opp[w])); }
    empty[3] &= TOPMASK;
    five_out[gid] = has_five(own) ? 1 : 0;
    thread ulong comp[4];
    completion_mask(own, empty, comp);
    for (uint w=0; w<4u; w++){ comp_out[base+w] = comp[w]; }
"""

_PROBE = mx.fast.metal_kernel(
    name="bb_probe",
    input_names=["own_in", "opp_in"],
    output_names=["five_out", "comp_out"],
    source=_PROBE_SRC,
    header=_header(),
)


def probe(boards: np.ndarray):
    """boards (B,2,N,N) -> (five (B,) bool, comp (B,4) uint64)."""
    own, opp = pack_words(boards)
    B = boards.shape[0]
    o = mx.array(own.reshape(-1))
    p = mx.array(opp.reshape(-1))
    five, comp = _PROBE(
        inputs=[o, p],
        grid=(B, 1, 1),
        threadgroup=(min(B, 64), 1, 1),
        output_shapes=[(B,), (B * 4,)],
        output_dtypes=[mx.uint8, mx.uint64],
    )
    mx.eval(five, comp)
    return np.array(five).astype(bool), np.array(comp).reshape(B, 4)
