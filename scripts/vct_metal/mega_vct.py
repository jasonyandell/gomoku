"""Fully on-device VCT megakernel (one thread per position) — the (C) vision in PoC.

Extends mega_vcf to real AND/OR: each thread runs an iterative VCT proof search with
make/unmake on a thread-local board, OR-frames (fours + forcing threes) and AND-frames
(defender replies), the WHOLE search in-kernel. This realises "VCT fully on the GPU".

It is deliberately UNoptimised: per-node detection (fours + threes + tempo) is recomputed
from scratch (O(N^3)-ish per node), and there is no work-stealing, so it is heavily
tail-bound — the throughput levers (incremental/bitboard detection + work-stealing) are
future work (see wiki gpu-vct-feasibility §5/§7). Correctness is the point here: validated
against gomoku.vcf.solve_vct on real positions.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from gomoku import state_ops

N = state_ops.BOARD_SIZE
MAXD = 48

_HEADER = """
inline bool mk5(thread uchar* a, int N, int r0, int c0, int dr, int dc, int m1, int m2) {
    int cnt=1; int rr=r0+dr, cc=c0+dc;
    while(rr>=0&&rr<N&&cc>=0&&cc<N){int i=rr*N+cc; if(a[i]!=0||i==m1||i==m2){cnt++;rr+=dr;cc+=dc;}else break;}
    rr=r0-dr; cc=c0-dc;
    while(rr>=0&&rr<N&&cc>=0&&cc<N){int i=rr*N+cc; if(a[i]!=0||i==m1||i==m2){cnt++;rr-=dr;cc-=dc;}else break;}
    return cnt>=5;
}
inline bool five_any(thread uchar* a, thread uchar* b, int N) {
    const int dr4[4]={0,1,1,1},dc4[4]={1,0,1,-1};
    for(int c=0;c<N*N;c++){ if(a[c]!=0||b[c]!=0)continue; int r=c/N,cc=c%N;
        for(int d=0;d<4;d++) if(mk5(a,N,r,cc,dr4[d],dc4[d],-1,-1)) return true; }
    return false;
}
inline bool has5(thread uchar* a, int N) {
    const int dr4[4]={0,1,1,1},dc4[4]={1,0,1,-1};
    for(int c=0;c<N*N;c++){ if(a[c]==0)continue; int r=c/N,cc=c%N;
        for(int d=0;d<4;d++) if(mk5(a,N,r,cc,dr4[d],dc4[d],-1,-1)) return true; }
    return false;
}
inline int four_at(thread uchar* own, thread uchar* opp, int N, int m, thread int* block) {
    const int dr4[4]={0,1,1,1},dc4[4]={1,0,1,-1}; int mr=m/N,mc=m%N; int cnt=0; *block=-1;
    for(int d=0;d<4;d++){int dr=dr4[d],dc=dc4[d];
      for(int delta=-4;delta<=4;delta++){ if(delta==0)continue;
        int cr=mr+delta*dr, cc=mc+delta*dc; if(cr<0||cr>=N||cc<0||cc>=N)continue;
        int c2=cr*N+cc; if(own[c2]!=0||opp[c2]!=0)continue;
        if(mk5(own,N,cr,cc,dr,dc,m,-1)){cnt++; if(*block<0)*block=c2;} } }
    return cnt;
}
inline int comps_of_f(thread uchar* own, thread uchar* opp, int N, int fcell, int mcell, thread uint* dmask){
    int fr=fcell/N,fc=fcell%N; const int dr4[4]={0,1,1,1},dc4[4]={1,0,1,-1}; int cnt=0;
    for(int d=0;d<4;d++){int dr=dr4[d],dc=dc4[d];
      for(int s=-1;s<=1;s+=2){ for(int k=1;k<5;k++){
        int rr=fr+s*dr*k,cc=fc+s*dc*k; if(rr<0||rr>=N||cc<0||cc>=N)break; int idx=rr*N+cc;
        bool oh=(own[idx]!=0)||idx==mcell||idx==fcell; if(oh)continue; if(opp[idx]!=0)break;
        if(mk5(own,N,rr,cc,dr,dc,mcell,fcell)){int w=idx>>5,bb=idx&31; uint bit=(1u<<bb);
            if((dmask[w]&bit)==0){dmask[w]|=bit;cnt++;}} } } }
    return cnt;
}
// forcing three at m? fill rmask (reply set), forkout; return #threats.
inline int threes_for_m(thread uchar* own, thread uchar* opp, int N, int m, thread uint* rmask, thread bool* forkout){
    int mr=m/N,mc=m%N; const int dr4[4]={0,1,1,1},dc4[4]={1,0,1,-1};
    for(int w=0;w<8;w++) rmask[w]=0;
    uint tmasks[96]; int ntm=0; int nthreats=0;
    for(int d=0;d<4;d++){int dr=dr4[d],dc=dc4[d];
      for(int s=-1;s<=1;s+=2){ for(int k=1;k<5;k++){
        int fr=mr+s*dr*k,fc=mc+s*dc*k; if(fr<0||fr>=N||fc<0||fc>=N)break; int fcell=fr*N+fc;
        if(own[fcell]!=0||opp[fcell]!=0)continue;
        uint dmask[8]={0,0,0,0,0,0,0,0};
        int comps=comps_of_f(own,opp,N,fcell,m,dmask);
        if(comps>=2){ int w=fcell>>5,bb=fcell&31; dmask[w]|=(1u<<bb);
          for(int i=0;i<8;i++) rmask[i]|=dmask[i];
          if(ntm<12){for(int i=0;i<8;i++)tmasks[ntm*8+i]=dmask[i];ntm++;}
          nthreats++; } } } }
    bool fork=false;
    for(int i=0;i<ntm&&!fork;i++)for(int j=i+1;j<ntm&&!fork;j++){bool dj=true;
      for(int w=0;w<8;w++)if((tmasks[i*8+w]&tmasks[j*8+w])!=0){dj=false;break;} if(dj)fork=true;}
    *forkout=fork; return nthreats;
}
// defender four/five available (board already has own@m placed)?
inline bool def_tempo(thread uchar* own, thread uchar* opp, int N){
    if(five_any(opp,own,N)) return true;
    for(int c=0;c<N*N;c++){ if(own[c]!=0||opp[c]!=0)continue; int b; if(four_at(opp,own,N,c,&b)>=1) return true; }
    return false;
}
inline int popbit(thread uint* mask){
    for(int w=0;w<8;w++){ if(mask[w]!=0){ uint low=mask[w]&(~mask[w]+1u); int bit=31-clz(low);
        mask[w]&=~low; return w*32+bit; } }
    return -1;
}
"""

_SRC = """
    uint gid = thread_position_in_grid.x;
    const int N=__N__; const int NN=N*N; const int MAXD=__MAXD__;
    uint base=gid*NN;
    thread uchar own[__NN__]; thread uchar opp[__NN__];
    for(int i=0;i<NN;i++){ own[i]=own_in[base+i]; opp[i]=opp_in[base+i]; }

    uchar typ[__MAXD__]; uint fmask[__MAXD__][8]; uint tmask[__MAXD__][8]; uint rmask[__MAXD__][8];
    int mm[__MAXD__]; int mb[__MAXD__]; int ar[__MAXD__]; uchar phase[__MAXD__];

    int maxnodes=max_nodes[0]; int nodes=0; bool hitcap=false;
    int sp=0; typ[0]=0; bool entering=true; int ret=0;

    while(true){
      if(entering){
        nodes++;
        if(sp>=MAXD-1 || nodes>maxnodes){ hitcap=true; ret=0; entering=false; continue; }
        if(typ[sp]==0){ // OR
          if(five_any(own,opp,N)){ ret=1; entering=false; continue; }
          for(int w=0;w<8;w++){ fmask[sp][w]=0; tmask[sp][w]=0; }
          bool dwin=false;
          for(int m=0;m<NN && !dwin;m++){ if(own[m]!=0||opp[m]!=0) continue;
            int blk; int nc=four_at(own,opp,N,m,&blk);
            if(nc>=1){ own[m]=1; bool bad=five_any(opp,own,N); own[m]=0; if(bad) continue;
              if(nc>=2){ dwin=true; break; } fmask[sp][m>>5]|=(1u<<(m&31)); continue; }
            uint rm[8]; bool fk; int nt=threes_for_m(own,opp,N,m,rm,&fk);
            if(nt>=1){ own[m]=1; bool tp=def_tempo(own,opp,N); own[m]=0; if(tp) continue;
              if(fk){ dwin=true; break; } tmask[sp][m>>5]|=(1u<<(m&31)); }
          }
          if(dwin){ ret=1; entering=false; continue; }
          bool any=false; for(int w=0;w<8;w++) if(fmask[sp][w]||tmask[sp][w]){ any=true; break; }
          if(!any){ ret=0; entering=false; continue; }
          phase[sp]=0;
          int m=popbit(&fmask[sp][0]);
          if(m>=0){ int blk; four_at(own,opp,N,m,&blk); mm[sp]=m; mb[sp]=blk; own[m]=1; opp[blk]=1;
            typ[sp+1]=0; sp++; entering=true; continue; }
          phase[sp]=1; int t=popbit(&tmask[sp][0]);
          uint rm[8]; bool fk; threes_for_m(own,opp,N,t,rm,&fk);
          mm[sp]=t; mb[sp]=-1; own[t]=1;
          for(int w=0;w<8;w++) rmask[sp+1][w]=rm[w]; typ[sp+1]=1; sp++; entering=true; continue;
        } else { // AND
          int r=popbit(&rmask[sp][0]);
          while(r>=0 && (own[r]!=0||opp[r]!=0)) r=popbit(&rmask[sp][0]);
          if(r<0){ ret=0; entering=false; continue; }
          opp[r]=1; ar[sp]=r;
          if(has5(opp,N)){ opp[r]=0; ret=0; entering=false; continue; }
          typ[sp+1]=0; sp++; entering=true; continue;
        }
      } else { // returning with ret
        if(sp==0) break;
        sp--;
        if(typ[sp]==0){ // OR parent
          if(mb[sp]>=0){ own[mm[sp]]=0; opp[mb[sp]]=0; } else { own[mm[sp]]=0; }
          if(ret==1){ entering=false; continue; }
          if(phase[sp]==0){
            int m=popbit(&fmask[sp][0]);
            if(m>=0){ int blk; four_at(own,opp,N,m,&blk); mm[sp]=m; mb[sp]=blk; own[m]=1; opp[blk]=1;
              typ[sp+1]=0; sp++; entering=true; continue; }
            phase[sp]=1;
          }
          int t=popbit(&tmask[sp][0]);
          if(t>=0){ uint rm[8]; bool fk; threes_for_m(own,opp,N,t,rm,&fk);
            mm[sp]=t; mb[sp]=-1; own[t]=1;
            for(int w=0;w<8;w++) rmask[sp+1][w]=rm[w]; typ[sp+1]=1; sp++; entering=true; continue; }
          ret=0; entering=false; continue;
        } else { // AND parent
          opp[ar[sp]]=0;
          if(ret==0){ entering=false; continue; }       // a reply failed -> AND NOWIN
          int r=popbit(&rmask[sp][0]);
          while(r>=0 && (own[r]!=0||opp[r]!=0)) r=popbit(&rmask[sp][0]);
          if(r<0){ ret=1; entering=false; continue; }    // all replies win -> AND WIN
          opp[r]=1; ar[sp]=r;
          if(has5(opp,N)){ opp[r]=0; ret=0; entering=false; continue; }
          typ[sp+1]=0; sp++; entering=true; continue;
        }
      }
    }
    win[gid]=(uchar)ret; hit[gid]=(uchar)(hitcap?1:0);
"""


def _src():
    return (_SRC.replace("__NN__", str(N * N)).replace("__N__", str(N))
            .replace("__MAXD__", str(MAXD)))


_KERNEL = mx.fast.metal_kernel(
    name="mega_vct", input_names=["own_in", "opp_in", "max_nodes"],
    output_names=["win", "hit"], source=_src(), header=_HEADER)


def solve_vct_mega(boards: np.ndarray, *, max_nodes: int = 20000):
    """boards: (B,2,N,N) bool. Returns (win, hit_cap): (B,) bool. Fully on-device."""
    B = boards.shape[0]
    own = mx.array(np.ascontiguousarray(boards[:, 0]).reshape(-1).astype(np.uint8))
    opp = mx.array(np.ascontiguousarray(boards[:, 1]).reshape(-1).astype(np.uint8))
    mn = mx.array(np.array([max_nodes], dtype=np.int32))
    win, hit = _KERNEL(inputs=[own, opp, mn], grid=(B, 1, 1),
                       threadgroup=(min(B, 32), 1, 1),
                       output_shapes=[(B,), (B,)], output_dtypes=[mx.uint8, mx.uint8])
    mx.eval(win, hit)
    return np.array(win).astype(bool), np.array(hit).astype(bool)


if __name__ == "__main__":
    import time
    from gomoku import vcf
    from scripts.vct_metal.positions import load_position_stack
    print("device:", mx.default_device())
    st = load_position_stack(24, seed=0, min_ply=6, max_ply=40)
    t = time.time(); win, hit = solve_vct_mega(st, max_nodes=4000); dt = time.time() - t
    ag = cap = fp = fn = vw = 0
    for b in range(st.shape[0]):
        v = vcf.solve_vct(st[b], max_depth=MAXD - 2, max_nodes=4000)
        vw += int(v.has_forced_win)
        if bool(win[b]) == v.has_forced_win:
            if not (bool(hit[b]) or v.hit_cap): ag += 1
        elif bool(hit[b]) or v.hit_cap: cap += 1
        elif bool(win[b]): fp += 1
        else: fn += 1
    print(f"{st.shape[0]} boards {dt:.1f}s vcf_wins={vw} clean_agree={ag} cap={cap} "
          f"FP={fp} FN={fn} -> {'PASS' if not fp and not fn else 'FAIL'}")
