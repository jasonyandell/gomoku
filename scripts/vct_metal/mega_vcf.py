"""Megakernel proof-of-concept: a fully on-device VCF solver (one thread per position).

This is the (C) direction de-risked on VCF (OR-only): each GPU thread runs an
iterative depth-first proof search with make/unmake on a *thread-local* board — no
host orchestration per node, the whole search lives in the kernel. If this is correct
and fast it validates the on-device search machine; VCT (AND-nodes) is the extension.

VCF shape: every attacker four forces the unique completion block, so the search is
OR-only with a forced single reply. A double four (>=2 completions, sound) wins now.
Soundness (vcf rule 1): a four is forcing only if the defender has no immediate five
after it — checked exactly (make own@m, test opp five, unmake).

Validated against gomoku.vcf.solve_vcf on real Rapfi positions (clean/no-cap cases).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from gomoku import state_ops

N = state_ops.BOARD_SIZE
MAXD = 40  # max search depth (frames)

_HEADER = """
inline bool mk5(thread uchar* own, int N, int r0, int c0, int dr, int dc, int mcell) {
    int cnt = 1;
    int rr=r0+dr, cc=c0+dc;
    while (rr>=0&&rr<N&&cc>=0&&cc<N){ int idx=rr*N+cc; if(own[idx]!=0||idx==mcell){cnt++;rr+=dr;cc+=dc;} else break; }
    rr=r0-dr; cc=c0-dc;
    while (rr>=0&&rr<N&&cc>=0&&cc<N){ int idx=rr*N+cc; if(own[idx]!=0||idx==mcell){cnt++;rr-=dr;cc-=dc;} else break; }
    return cnt >= 5;
}
// any empty cell where placing `a` (with opponent `b`) makes five?
inline bool five_any(thread uchar* a, thread uchar* b, int N) {
    const int dr4[4]={0,1,1,1}; const int dc4[4]={1,0,1,-1};
    for (int c=0;c<N*N;c++){ if(a[c]!=0||b[c]!=0) continue; int r=c/N, cc=c%N;
        for(int d=0;d<4;d++) if(mk5(a,N,r,cc,dr4[d],dc4[d],-1)) return true; }
    return false;
}
// completions of an own play at m (m hypothetical own): count + first block cell.
inline int four_at(thread uchar* own, thread uchar* opp, int N, int m, thread int* block) {
    const int dr4[4]={0,1,1,1}; const int dc4[4]={1,0,1,-1};
    int mr=m/N, mc=m%N; int cnt=0; *block=-1;
    for(int d=0;d<4;d++){ int dr=dr4[d],dc=dc4[d];
        for(int delta=-4;delta<=4;delta++){ if(delta==0) continue;
            int cr=mr+delta*dr, cc=mc+delta*dc; if(cr<0||cr>=N||cc<0||cc>=N) continue;
            int c2=cr*N+cc; if(own[c2]!=0||opp[c2]!=0) continue;
            if(mk5(own,N,cr,cc,dr,dc,m)){ cnt++; if(*block<0)*block=c2; } } }
    return cnt;
}
inline int popbit(thread uint* mask) {
    for(int w=0;w<8;w++){ if(mask[w]!=0){ uint low=mask[w]&(~mask[w]+1u); int bit=31-clz(low);
        mask[w]&=~low; return w*32+bit; } }
    return -1;
}
"""

_SRC = """
    uint gid = thread_position_in_grid.x;
    const int N = __N__;
    const int NN = N*N;
    const int MAXD = __MAXD__;
    uint base = gid * NN;

    thread uchar own[__NN__];
    thread uchar opp[__NN__];
    for (int i=0;i<NN;i++){ own[i]=own_in[base+i]; opp[i]=opp_in[base+i]; }

    uint fmask[__MAXD__][8];
    int mm[__MAXD__];
    int mb[__MAXD__];

    int maxnodes = max_nodes[0];
    int nodes = 0;
    bool hitcap = false;
    int sp = 0;
    bool entering = true;
    int ret = 0;   // 0 = NOWIN, 1 = WIN

    while (true) {
        if (entering) {
            nodes++;
            if (sp >= MAXD || nodes > maxnodes) { hitcap = true; ret = 0; entering = false; continue; }
            if (five_any(own, opp, N)) { ret = 1; entering = false; continue; }
            // build single-four mask; detect a sound double-four win
            for (int w=0;w<8;w++) fmask[sp][w]=0;
            bool dwin = false;
            for (int m=0; m<NN && !dwin; m++) {
                if (own[m]!=0 || opp[m]!=0) continue;
                int block; int nc = four_at(own, opp, N, m, &block);
                if (nc < 1) continue;
                // soundness: defender has no immediate five after own@m
                own[m]=1; bool bad = five_any(opp, own, N); own[m]=0;
                if (bad) continue;
                if (nc >= 2) { dwin = true; break; }
                fmask[sp][m>>5] |= (1u << (m & 31));
            }
            if (dwin) { ret = 1; entering = false; continue; }
            bool any = false; for (int w=0;w<8;w++) if (fmask[sp][w]) { any=true; break; }
            if (!any) { ret = 0; entering = false; continue; }
            int m = popbit(&fmask[sp][0]);
            int block; four_at(own, opp, N, m, &block);
            mm[sp]=m; mb[sp]=block; own[m]=1; opp[block]=1;
            sp++; entering = true; continue;
        } else {
            if (sp == 0) break;
            sp--;
            own[mm[sp]] = 0; opp[mb[sp]] = 0;       // unmake parent's move
            if (ret == 1) { entering = false; continue; }   // parent (OR) wins -> bubble
            int m = popbit(&fmask[sp][0]);
            if (m >= 0) {
                int block; four_at(own, opp, N, m, &block);
                mm[sp]=m; mb[sp]=block; own[m]=1; opp[block]=1;
                sp++; entering = true; continue;
            }
            ret = 0; entering = false; continue;     // exhausted -> NOWIN
        }
    }
    win[gid] = (uchar)ret;
    hit[gid] = (uchar)(hitcap ? 1 : 0);
"""


def _src():
    return _SRC.replace("__NN__", str(N * N)).replace("__N__", str(N)).replace("__MAXD__", str(MAXD))


_KERNEL = mx.fast.metal_kernel(
    name="mega_vcf",
    input_names=["own_in", "opp_in", "max_nodes"],
    output_names=["win", "hit"],
    source=_src(),
    header=_HEADER,
)


def solve_vcf_mega(boards: np.ndarray, *, max_nodes: int = 20000):
    """boards: (B,2,N,N) bool. Returns (win, hit_cap): (B,) bool. Fully on-device."""
    B = boards.shape[0]
    own = mx.array(np.ascontiguousarray(boards[:, 0]).reshape(-1).astype(np.uint8))
    opp = mx.array(np.ascontiguousarray(boards[:, 1]).reshape(-1).astype(np.uint8))
    mn = mx.array(np.array([max_nodes], dtype=np.int32))
    win, hit = _KERNEL(
        inputs=[own, opp, mn],
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
        t = time.time(); win, hit = solve_vcf_mega(st, max_nodes=20000); dt = time.time() - t
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
