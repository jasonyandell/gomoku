"""Bitboard VCF megakernel — same search as mega_vcf, bitboard detection.

One GPU thread per position runs the whole OR-only forced-four search with
make/unmake on a *thread-local bitboard* (own/opp/empty as ulong[4]). Versus
`mega_vcf` (uchar[225] board, cell-scan detection — O(N²)/node and O(N⁴) for the
per-move soundness check), this:

  * generates all forcing four-moves with one set-algebra pass (hole-pair
    shift-AND over the board), no per-cell loop;
  * does soundness (defender immediate-five) and double-four detection with
    `completion_mask` — O(board-words), not a per-move rescan.

Control flow is a line-for-line port of mega_vcf's validated iterative DFS, so a
verdict mismatch would be a detection bug, isolated by test_bb / bb_ref.

Validated vs gomoku.vcf.solve_vcf on real positions in test_mega_vcf_bb.py.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from scripts.vct_metal import bb

N = bb.N
MAXD = 40

_HEADER = bb._header() + """
inline int popcount4(thread const ulong* a){
    return popcount(a[0])+popcount(a[1])+popcount(a[2])+popcount(a[3]);
}
inline int lowbit4(thread const ulong* a){
    for(int w=0;w<4;w++){ if(a[w]!=0ul){ ulong low=a[w]&(~a[w]+1ul); return w*64 + (63 - (int)clz(low)); } }
    return -1;
}
inline void setbit4(thread ulong* a, int idx){ a[idx>>6] |= (1ul << (idx&63)); }
inline void clrbit4(thread ulong* a, int idx){ a[idx>>6] &= ~(1ul << (idx&63)); }

// all forcing four-moves: union of the two empties of every (3 own,2 empty,0 opp)
// length-5 window. 4 directions x C(5,2)=10 hole-pairs of shift-AND.
constant int HP[10][2] = {{0,1},{0,2},{0,3},{0,4},{1,2},{1,3},{1,4},{2,3},{2,4},{3,4}};
inline void gen_forcing(thread const ulong* own, thread const ulong* empty, thread ulong* forcing){
    forcing[0]=0ul;forcing[1]=0ul;forcing[2]=0ul;forcing[3]=0ul;
    for(int d=0; d<4; d++){
        uint s = DSTEP[d];
        // precompute the 5 shifts of own and empty once; reuse across 10 hole-pairs.
        ulong Os[5][4]; ulong Es[5][4];
        for(int i=0;i<5;i++){ shr256(own,(uint)i*s,Os[i]); shr256(empty,(uint)i*s,Es[i]); }
        for(int h=0; h<10; h++){
            int j=HP[h][0], k=HP[h][1];
            ulong m[4]; m[0]=DMASK[d][0];m[1]=DMASK[d][1];m[2]=DMASK[d][2];m[3]=DMASK[d][3];
            for(int i=0;i<5;i++){
                if(i==j||i==k){ m[0]&=Es[i][0];m[1]&=Es[i][1];m[2]&=Es[i][2];m[3]&=Es[i][3]; }
                else          { m[0]&=Os[i][0];m[1]&=Os[i][1];m[2]&=Os[i][2];m[3]&=Os[i][3]; }
            }
            ulong a[4]; shl256(m, (uint)j*s, a);
            forcing[0]|=a[0];forcing[1]|=a[1];forcing[2]|=a[2];forcing[3]|=a[3];
            shl256(m, (uint)k*s, a);
            forcing[0]|=a[0];forcing[1]|=a[1];forcing[2]|=a[2];forcing[3]|=a[3];
        }
    }
    and4(forcing, empty);
}
"""

_SRC = """
    uint gid = thread_position_in_grid.x;
    uint base = gid * 4u;
    const int MAXD = __MAXD__;

    thread ulong own[4]; thread ulong opp[4]; thread ulong empty[4];
    for (uint w=0; w<4u; w++){ own[w]=own_in[base+w]; opp[w]=opp_in[base+w]; }
    for (uint w=0; w<4u; w++){ empty[w] = ~(own[w]|opp[w]); }
    empty[3] &= TOPMASK;

    ulong fmask[__MAXD__][4];     // remaining sound single-four moves per frame
    int mm[__MAXD__]; int mb[__MAXD__];

    int maxnodes = max_nodes[0];
    int nodes = 0; bool hitcap=false;
    int sp = 0; bool entering = true; int ret = 0;

    while (true) {
        if (entering) {
            nodes++;
            if (sp >= MAXD || nodes > maxnodes) { hitcap=true; ret=0; entering=false; continue; }
            ulong cown[4]; completion_mask(own, empty, cown);
            if (any4(cown)) { ret=1; entering=false; continue; }      // immediate five
            ulong forcing[4]; gen_forcing(own, empty, forcing);
            ulong sound[4]={0ul,0ul,0ul,0ul};
            bool dwin=false;
            ulong f[4]; cpy4(forcing, f);
            while (true) {
                int mc = lowbit4(f); if (mc<0) break; clrbit4(f, mc);
                ulong own2[4]; cpy4(own, own2); setbit4(own2, mc);
                ulong empty2[4]; cpy4(empty, empty2); clrbit4(empty2, mc);
                ulong oc[4]; completion_mask(opp, empty2, oc);
                if (any4(oc)) continue;                               // unsound (defender fives)
                ulong cc[4]; completion_mask(own2, empty2, cc);
                int nc = popcount4(cc);
                if (nc>=2) { dwin=true; break; }                     // sound double four = win
                setbit4(sound, mc);
            }
            if (dwin) { ret=1; entering=false; continue; }
            if (!any4(sound)) { ret=0; entering=false; continue; }
            int m = lowbit4(sound); clrbit4(sound, m);
            ulong own2[4]; cpy4(own, own2); setbit4(own2, m);
            ulong empty2[4]; cpy4(empty, empty2); clrbit4(empty2, m);
            ulong cc[4]; completion_mask(own2, empty2, cc);
            int block = lowbit4(cc);
            fmask[sp][0]=sound[0];fmask[sp][1]=sound[1];fmask[sp][2]=sound[2];fmask[sp][3]=sound[3];
            mm[sp]=m; mb[sp]=block;
            setbit4(own, m); setbit4(opp, block); clrbit4(empty, m); clrbit4(empty, block);
            sp++; entering=true; continue;
        } else {
            if (sp==0) break;
            sp--;
            int m=mm[sp], block=mb[sp];
            clrbit4(own, m); clrbit4(opp, block); setbit4(empty, m); setbit4(empty, block);
            if (ret==1) { entering=false; continue; }                // child won -> bubble
            ulong rem[4]; cpy4(&fmask[sp][0], rem);
            int m2 = lowbit4(rem);
            if (m2>=0) {
                clrbit4(rem, m2);
                fmask[sp][0]=rem[0];fmask[sp][1]=rem[1];fmask[sp][2]=rem[2];fmask[sp][3]=rem[3];
                ulong own2[4]; cpy4(own, own2); setbit4(own2, m2);
                ulong empty2[4]; cpy4(empty, empty2); clrbit4(empty2, m2);
                ulong cc[4]; completion_mask(own2, empty2, cc);
                int block2 = lowbit4(cc);
                mm[sp]=m2; mb[sp]=block2;
                setbit4(own, m2); setbit4(opp, block2); clrbit4(empty, m2); clrbit4(empty, block2);
                sp++; entering=true; continue;
            }
            ret=0; entering=false; continue;                         // exhausted
        }
    }
    win[gid] = (uchar)ret;
    hit[gid] = (uchar)(hitcap ? 1 : 0);
"""


def _src():
    return _SRC.replace("__MAXD__", str(MAXD))


_KERNEL = mx.fast.metal_kernel(
    name="mega_vcf_bb",
    input_names=["own_in", "opp_in", "max_nodes"],
    output_names=["win", "hit"],
    source=_src(),
    header=_HEADER,
)


def solve_vcf_mega_bb(boards: np.ndarray, *, max_nodes: int = 20000):
    """boards: (B,2,N,N) bool. Returns (win, hit_cap): (B,) bool. Fully on-device."""
    B = boards.shape[0]
    own, opp = bb.pack_words(boards)
    o = mx.array(own.reshape(-1))
    p = mx.array(opp.reshape(-1))
    mn = mx.array(np.array([max_nodes], dtype=np.int32))
    win, hit = _KERNEL(
        inputs=[o, p, mn],
        grid=(B, 1, 1),
        threadgroup=(min(B, 64), 1, 1),
        output_shapes=[(B,), (B,)],
        output_dtypes=[mx.uint8, mx.uint8],
    )
    mx.eval(win, hit)
    return np.array(win).astype(bool), np.array(hit).astype(bool)


if __name__ == "__main__":
    import time
    from gomoku import vcf
    from scripts.vct_metal.positions import load_position_stack

    print("device:", mx.default_device())
    for seed in (0, 1):
        st = load_position_stack(120, seed=seed, min_ply=6, max_ply=60)
        t = time.time(); win, hit = solve_vcf_mega_bb(st, max_nodes=20000); dt = time.time() - t
        agree = cap = fp = fn = vw = 0
        for b in range(st.shape[0]):
            v = vcf.solve_vcf(st[b], max_nodes=20000)
            vw += int(v.has_forced_win)
            if bool(win[b]) == v.has_forced_win:
                if not (bool(hit[b]) or v.hit_cap): agree += 1
            elif bool(hit[b]) or v.hit_cap:
                cap += 1
            elif bool(win[b]):
                fp += 1
            else:
                fn += 1
        print(f"seed={seed} {st.shape[0]} boards {dt*1000:.0f}ms ({dt/st.shape[0]*1e3:.2f} ms/board) "
              f"vcf_wins={vw} clean_agree={agree} cap={cap} FP={fp} FN={fn} "
              f"{'PASS' if not fp and not fn else 'FAIL'}")
