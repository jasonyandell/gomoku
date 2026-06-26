"""Bitboard VCT megakernel — the (C) vision, made fast.

A port of mega_vct's validated AND/OR proof search with the board as a thread-local
bitboard (own/opp as ulong[4]) and ALL detection by bitboard set-algebra:

  * OR-node fours: gen_forcing(own) computes EVERY four-move in one set-algebra
    pass (no per-cell loop); each is checked for soundness (defender five) and
    double-four via completion_mask.  (as in mega_vcf_bb — validated 14x, 0 disagree)
  * OR-node forcing threes: only cells in the candidate mask (Chebyshev-2 of any
    stone — the SAME set gomoku.vcf considers, so this is exact, not heuristic) are
    trial-played; for each, gen_forcing(own∪m) gives the follow-up four-cells f and
    f is an open-four threat iff completion_mask(own∪m∪f) has >=2 bits; the threat's
    defeating set is {f}∪comps, the AND reply mask is their union, a fork is a
    disjoint pair (the (B) algebra, validated in threes_ref/threes_metal).
  * defender-tempo guard: completion_mask(opp) (five) or gen_forcing(opp) (four).

`empty` is maintained incrementally alongside own/opp.

Validated vs gomoku.vcf.solve_vct and the cell-scan mega_vct in test_mega_vct_bb.py.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

from scripts.vct_metal import bb, bb_ref, mega_vcf_bb

N = bb.N
MAXD = 32
_BOARDMASK = bb_ref.BOARDMASK


def _mask_words(m: int) -> str:
    return "{%s}" % ", ".join("%dul" % ((m >> (64 * w)) & ((1 << 64) - 1)) for w in range(4))


def _collinear_table() -> str:
    """MSL constant: COLLIN[m] = empties-mask within distance 4 of m on the 4 axes
    (== gomoku.vcf._collinear_empties domain — the follow-up cells f a forcing
    three at m may use)."""
    rows = []
    for m in range(N * N):
        mr, mc = divmod(m, N)
        mask = 0
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            for sign in (1, -1):
                for k in range(1, 5):
                    rr, cc = mr + sign * dr * k, mc + sign * dc * k
                    if 0 <= rr < N and 0 <= cc < N:
                        mask |= 1 << (rr * N + cc)
        rows.append(_mask_words(mask))
    return "constant ulong COLLIN[%d][4] = {%s};" % (N * N, ", ".join(rows))


# column-edge masks for wrap-safe king dilation
_COL0 = sum(1 << (r * N) for r in range(N))
_COLL = sum(1 << (r * N + N - 1) for r in range(N))
_NOT0 = (~_COL0) & _BOARDMASK
_NOTL = (~_COLL) & _BOARDMASK

_VCT_HELPERS = ("""
constant ulong NOT0[4] = __NOT0__;
constant ulong NOTL[4] = __NOTL__;
__COLLIN__
inline void or4(thread ulong* a, thread const ulong* b){ a[0]|=b[0];a[1]|=b[1];a[2]|=b[2];a[3]|=b[3]; }
// dilate bitset x by one king step (8-neighbourhood), column-wrap safe.
inline void king(thread const ulong* x, thread ulong* o){
    cpy4(x,o);
    ulong xe[4]; ulong xw[4]; ulong s[4];
    for(int w=0;w<4;w++){ xe[w]=x[w]&NOTL[w]; xw[w]=x[w]&NOT0[w]; }
    shl256(xe, 1u, s);     or4(o,s);     // east  c+1
    shr256(xw, 1u, s);     or4(o,s);     // west  c-1
    shl256(x,  __N__u, s); or4(o,s);     // south r+1
    shr256(x,  __N__u, s); or4(o,s);     // north r-1
    shl256(xe, __N1__u, s);  or4(o,s);   // SE  r+1,c+1
    shr256(xw, __N1__u, s);  or4(o,s);   // NW  r-1,c-1
    shl256(xw, __Nm1__u, s); or4(o,s);   // SW  r+1,c-1
    shr256(xe, __Nm1__u, s); or4(o,s);   // NE  r-1,c+1
}
// empties within Chebyshev-2 of any stone (== gomoku.vcf candidate set).
inline void candidate_mask(thread const ulong* own, thread const ulong* opp,
                           thread const ulong* empty, thread ulong* cand){
    ulong occ[4]; for(int w=0;w<4;w++) occ[w]=own[w]|opp[w];
    ulong d1[4]; king(occ, d1);
    ulong d2[4]; king(d1, d2);
    for(int w=0;w<4;w++) cand[w]=d2[w]&empty[w];
}
// Forcing-three candidates: empties within Chebyshev-2 of an OWN stone. A three
// is own's threat, so its move sits within radius 2 of the own stones forming it
// (vcf's radius-2 argument, per side) — a tighter superset than candidate_mask.
inline void candidate_own(thread const ulong* own, thread const ulong* empty, thread ulong* cand){
    ulong d1[4]; king(own, d1);
    ulong d2[4]; king(d1, d2);
    for(int w=0;w<4;w++) cand[w]=d2[w]&empty[w];
}
inline bool def_tempo(thread const ulong* opp, thread const ulong* empty){
    ulong c[4]; completion_mask(opp, empty, c); if (any4(c)) return true;  // defender five
    ulong f[4]; gen_forcing(opp, empty, f);     if (any4(f)) return true;  // defender four
    return false;
}
// Forcing threes for the board own2=own|{m}, empty2=empty\\{m}. Fills `reply` (OR of
// {f}∪comps over open-four threats), sets *forkout if two threats are disjoint,
// returns #threats.
inline int gen_threes(thread const ulong* own2, thread const ulong* empty2, int m,
                      thread ulong* reply, thread bool* forkout){
    reply[0]=0ul;reply[1]=0ul;reply[2]=0ul;reply[3]=0ul;
    *forkout=false;
    ulong forcing[4]; gen_forcing(own2, empty2, forcing);
    // restrict follow-up f to cells collinear with m (dist<=4) — matches
    // gomoku.vcf._collinear_empties; without it far pre-existing fours over-generate.
    ulong cl[4]; cl[0]=COLLIN[m][0];cl[1]=COLLIN[m][1];cl[2]=COLLIN[m][2];cl[3]=COLLIN[m][3];
    and4(forcing, cl);
    ulong tmasks[12][4]; int ntm=0; int nthreats=0;
    ulong f[4]; cpy4(forcing, f);
    while (true){
        int fc = lowbit4(f); if (fc<0) break; clrbit4(f, fc);
        ulong own3[4]; cpy4(own2, own3); setbit4(own3, fc);
        ulong empty3[4]; cpy4(empty2, empty3); clrbit4(empty3, fc);
        ulong comps[4]; completion_mask(own3, empty3, comps);
        if (popcount4(comps) >= 2){                       // open-four threat
            ulong dm[4]; cpy4(comps, dm); setbit4(dm, fc); // {f} ∪ comps
            for (int t=0; t<ntm && !(*forkout); t++){
                ulong it[4]; it[0]=dm[0]&tmasks[t][0];it[1]=dm[1]&tmasks[t][1];
                it[2]=dm[2]&tmasks[t][2];it[3]=dm[3]&tmasks[t][3];
                if (!any4(it)) *forkout=true;              // disjoint -> fork
            }
            if (ntm<12){ cpy4(dm, tmasks[ntm]); ntm++; }
            reply[0]|=dm[0];reply[1]|=dm[1];reply[2]|=dm[2];reply[3]|=dm[3];
            nthreats++;
        }
    }
    return nthreats;
}
""".replace("__COLLIN__", _collinear_table())
   .replace("__NOT0__", _mask_words(_NOT0)).replace("__NOTL__", _mask_words(_NOTL))
   .replace("__N1__", str(N + 1)).replace("__Nm1__", str(N - 1)).replace("__N__", str(N)))

_HEADER = mega_vcf_bb._HEADER + _VCT_HELPERS

_SRC = """
    uint gid = thread_position_in_grid.x;
    const int MAXD = __MAXD__;
    uint base = gid * 4u;

    thread ulong own[4]; thread ulong opp[4]; thread ulong empty[4];
    for (uint w=0; w<4u; w++){ own[w]=own_in[base+w]; opp[w]=opp_in[base+w]; }
    for (uint w=0; w<4u; w++){ empty[w] = ~(own[w]|opp[w]); }
    empty[3] &= TOPMASK;

    uchar typ[__MAXD__]; uchar phase[__MAXD__];
    // fmask doubles as the AND-node reply set: a frame is OR (uses fmask=fours +
    // tmask=threes) or AND (uses fmask=replies), never both, so they can alias.
    ulong fmask[__MAXD__][4]; ulong tmask[__MAXD__][4];
    int mm[__MAXD__]; int mb[__MAXD__]; int ar[__MAXD__];

    int maxnodes=max_nodes[0]; int nodes=0; bool hitcap=false;
    int sp=0; typ[0]=0; bool entering=true; int ret=0;

    while (true){
      if (entering){
        nodes++;
        if (sp>=MAXD-1 || nodes>maxnodes){ hitcap=true; ret=0; entering=false; continue; }
        if (typ[sp]==0){ // ---- OR node (attacker) ----
          ulong c0[4]; completion_mask(own, empty, c0);
          if (any4(c0)){ ret=1; entering=false; continue; }       // immediate five
          fmask[sp][0]=0ul;fmask[sp][1]=0ul;fmask[sp][2]=0ul;fmask[sp][3]=0ul;
          tmask[sp][0]=0ul;tmask[sp][1]=0ul;tmask[sp][2]=0ul;tmask[sp][3]=0ul;
          bool dwin=false;
          // defender threats once: per-move checks are monotone (own@m only removes
          // defender options), so skip them whenever the threat is globally absent.
          ulong c_opp[4]; completion_mask(opp, empty, c_opp); bool gfive = any4(c_opp);
          ulong f_opp[4]; gen_forcing(opp, empty, f_opp); bool gtempo = gfive || any4(f_opp);
          // -- fours: one set-algebra pass over the whole board --
          ulong fourc[4]; gen_forcing(own, empty, fourc);
          ulong ff[4]; cpy4(fourc, ff);
          while (true){
            int m = lowbit4(ff); if (m<0) break; clrbit4(ff, m);
            ulong own2[4]; cpy4(own, own2); setbit4(own2, m);
            ulong empty2[4]; cpy4(empty, empty2); clrbit4(empty2, m);
            if (gfive){ ulong oc[4]; completion_mask(opp, empty2, oc); if (any4(oc)) continue; } // unsound
            ulong cm[4]; completion_mask(own2, empty2, cm);
            int nc = popcount4(cm);
            if (nc>=2){ dwin=true; break; }                        // sound double four
            setbit4(&fmask[sp][0], m);
          }
          // -- threes: only candidate cells, excluding four-moves --
          if (!dwin){
            ulong cand[4]; candidate_own(own, empty, cand);
            ulong tt[4]; for(int w=0;w<4;w++) tt[w]=cand[w]&(~fourc[w]);
            while (true){
              int m = lowbit4(tt); if (m<0) break; clrbit4(tt, m);
              ulong own2[4]; cpy4(own, own2); setbit4(own2, m);
              ulong empty2[4]; cpy4(empty, empty2); clrbit4(empty2, m);
              ulong rm[4]; bool fk;
              int nt = gen_threes(own2, empty2, m, rm, &fk);
              if (nt>=1){
                if (gtempo && def_tempo(opp, empty2)) continue;    // defender tempo
                if (fk){ dwin=true; break; }                       // fork three
                setbit4(&tmask[sp][0], m);
              }
            }
          }
          if (dwin){ ret=1; entering=false; continue; }
          if (!any4(&fmask[sp][0]) && !any4(&tmask[sp][0])){ ret=0; entering=false; continue; }
          phase[sp]=0;
          int m = lowbit4(&fmask[sp][0]);
          if (m>=0){
            clrbit4(&fmask[sp][0], m);
            ulong own2[4]; cpy4(own, own2); setbit4(own2, m);
            ulong empty2[4]; cpy4(empty, empty2); clrbit4(empty2, m);
            ulong cm[4]; completion_mask(own2, empty2, cm);
            int blk = lowbit4(cm);
            mm[sp]=m; mb[sp]=blk;
            setbit4(own,m); setbit4(opp,blk); clrbit4(empty,m); clrbit4(empty,blk);
            typ[sp+1]=0; sp++; entering=true; continue;
          }
          phase[sp]=1;
          int t = lowbit4(&tmask[sp][0]); clrbit4(&tmask[sp][0], t);
          ulong own2[4]; cpy4(own, own2); setbit4(own2, t);
          ulong empty2[4]; cpy4(empty, empty2); clrbit4(empty2, t);
          ulong rm[4]; bool fk; gen_threes(own2, empty2, t, rm, &fk);
          mm[sp]=t; mb[sp]=-1;
          setbit4(own,t); clrbit4(empty,t);
          fmask[sp+1][0]=rm[0];fmask[sp+1][1]=rm[1];fmask[sp+1][2]=rm[2];fmask[sp+1][3]=rm[3];
          typ[sp+1]=1; sp++; entering=true; continue;
        } else { // ---- AND node (defender) ----
          int r = lowbit4(&fmask[sp][0]);
          while (r>=0 && (((empty[r>>6]>>(r&63))&1ul)==0ul)){ clrbit4(&fmask[sp][0], r); r=lowbit4(&fmask[sp][0]); }
          if (r<0){ ret=0; entering=false; continue; }             // no legal reply
          clrbit4(&fmask[sp][0], r);
          setbit4(opp,r); clrbit4(empty,r); ar[sp]=r;
          if (has_five(opp)){ clrbit4(opp,r); setbit4(empty,r); ret=0; entering=false; continue; }
          typ[sp+1]=0; sp++; entering=true; continue;
        }
      } else { // ---- returning with ret ----
        if (sp==0) break;
        sp--;
        if (typ[sp]==0){ // OR parent
          if (mb[sp]>=0){ clrbit4(own,mm[sp]); clrbit4(opp,mb[sp]); setbit4(empty,mm[sp]); setbit4(empty,mb[sp]); }
          else          { clrbit4(own,mm[sp]); setbit4(empty,mm[sp]); }
          if (ret==1){ entering=false; continue; }                 // OR child won -> bubble
          if (phase[sp]==0){
            int m = lowbit4(&fmask[sp][0]);
            if (m>=0){
              clrbit4(&fmask[sp][0], m);
              ulong own2[4]; cpy4(own, own2); setbit4(own2, m);
              ulong empty2[4]; cpy4(empty, empty2); clrbit4(empty2, m);
              ulong cm[4]; completion_mask(own2, empty2, cm);
              int blk = lowbit4(cm);
              mm[sp]=m; mb[sp]=blk;
              setbit4(own,m); setbit4(opp,blk); clrbit4(empty,m); clrbit4(empty,blk);
              typ[sp+1]=0; sp++; entering=true; continue;
            }
            phase[sp]=1;
          }
          int t = lowbit4(&tmask[sp][0]);
          if (t>=0){
            clrbit4(&tmask[sp][0], t);
            ulong own2[4]; cpy4(own, own2); setbit4(own2, t);
            ulong empty2[4]; cpy4(empty, empty2); clrbit4(empty2, t);
            ulong rm[4]; bool fk; gen_threes(own2, empty2, t, rm, &fk);
            mm[sp]=t; mb[sp]=-1;
            setbit4(own,t); clrbit4(empty,t);
            fmask[sp+1][0]=rm[0];fmask[sp+1][1]=rm[1];fmask[sp+1][2]=rm[2];fmask[sp+1][3]=rm[3];
            typ[sp+1]=1; sp++; entering=true; continue;
          }
          ret=0; entering=false; continue;
        } else { // AND parent
          clrbit4(opp, ar[sp]); setbit4(empty, ar[sp]);
          if (ret==0){ entering=false; continue; }                 // a reply refutes -> AND NOWIN
          int r = lowbit4(&fmask[sp][0]);
          while (r>=0 && (((empty[r>>6]>>(r&63))&1ul)==0ul)){ clrbit4(&fmask[sp][0], r); r=lowbit4(&fmask[sp][0]); }
          if (r<0){ ret=1; entering=false; continue; }              // all replies win -> AND WIN
          clrbit4(&fmask[sp][0], r);
          setbit4(opp,r); clrbit4(empty,r); ar[sp]=r;
          if (has_five(opp)){ clrbit4(opp,r); setbit4(empty,r); ret=0; entering=false; continue; }
          typ[sp+1]=0; sp++; entering=true; continue;
        }
      }
    }
    win[gid]=(uchar)ret; hit[gid]=(uchar)(hitcap?1:0);
"""


def _src():
    return _SRC.replace("__MAXD__", str(MAXD))


_KERNEL = mx.fast.metal_kernel(
    name="mega_vct_bb", input_names=["own_in", "opp_in", "max_nodes"],
    output_names=["win", "hit"], source=_src(), header=_HEADER)


def solve_vct_mega_bb(boards: np.ndarray, *, max_nodes: int = 20000, tg: int = 32):
    """boards: (B,2,N,N) bool. Returns (win, hit_cap): (B,) bool. Fully on-device."""
    B = boards.shape[0]
    own, opp = bb.pack_words(boards)
    o = mx.array(own.reshape(-1))
    p = mx.array(opp.reshape(-1))
    mn = mx.array(np.array([max_nodes], dtype=np.int32))
    win, hit = _KERNEL(inputs=[o, p, mn], grid=(B, 1, 1),
                       threadgroup=(min(B, tg), 1, 1),
                       output_shapes=[(B,), (B,)], output_dtypes=[mx.uint8, mx.uint8])
    mx.eval(win, hit)
    return np.array(win).astype(bool), np.array(hit).astype(bool)


if __name__ == "__main__":
    import time
    from gomoku import vcf
    from scripts.vct_metal.positions import load_position_stack

    print("device:", mx.default_device())
    st = load_position_stack(64, seed=0, min_ply=6, max_ply=40)
    t = time.time(); win, hit = solve_vct_mega_bb(st, max_nodes=8000); dt = time.time() - t
    ag = cap = fp = fn = vw = 0
    for b in range(st.shape[0]):
        v = vcf.solve_vct(st[b], max_depth=MAXD - 2, max_nodes=8000)
        vw += int(v.has_forced_win)
        if bool(win[b]) == v.has_forced_win:
            if not (bool(hit[b]) or v.hit_cap): ag += 1
        elif bool(hit[b]) or v.hit_cap: cap += 1
        elif bool(win[b]): fp += 1
        else: fn += 1
    print(f"{st.shape[0]} boards {dt:.2f}s ({dt/st.shape[0]*1e3:.1f} ms/board) "
          f"vcf_wins={vw} clean_agree={ag} cap={cap} FP={fp} FN={fn} "
          f"-> {'PASS' if not fp and not fn else 'FAIL'}")
