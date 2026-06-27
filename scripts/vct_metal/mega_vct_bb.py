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
    // winmove = a valid VCT first move at the ROOT (sp==0). For recursive wins it
    // is mm[0] (the root's committed move); for the root's own inline wins
    // (immediate five / sound double-four / fork-three) it is captured here, since
    // those set ret=1 without ever pushing a child / setting mm[0].
    int winmove=-1;

    while (true){
      if (entering){
        nodes++;
        if (sp>=MAXD-1 || nodes>maxnodes){ hitcap=true; ret=0; entering=false; continue; }
        if (typ[sp]==0){ // ---- OR node (attacker) ----
          ulong c0[4]; completion_mask(own, empty, c0);
          if (any4(c0)){ if(sp==0) winmove=lowbit4(c0); ret=1; entering=false; continue; }  // immediate five
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
            if (nc>=2){ if(sp==0) winmove=m; dwin=true; break; }    // sound double four
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
                if (fk){ if(sp==0) winmove=m; dwin=true; break; }  // fork three
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
    move[gid]=(int)((ret==1) ? (winmove>=0 ? winmove : mm[0]) : -1);
"""


def _src():
    return _SRC.replace("__MAXD__", str(MAXD))


# ----------------------------------------------------------------------------
# Source-variant builder.  The DEFAULT (support=False, complete=False) returns
# _src() byte-for-byte, so the default kernel — and its (win, hit, move) verdict
# and throughput — is literally unchanged.  Two optional, independent features
# are injected only into the requested variant (each behind its own compiled
# kernel, never costing the fast path a single instruction):
#
#  * SUPPORT (return_support=True): a per-frame `fsupp[MAXD][4]` accumulator that
#    merges a child's support into its parent ONLY on a winning return (ret==1).
#    Because the merge happens on the win-bubbling return path, abandoned/refuted
#    branches never pollute it.  An OR node that wins contributes its move (+ the
#    forced four-block); an AND node that wins contributes every defender reply;
#    the three inline OR wins (immediate-five / sound double-four / fork-three)
#    contribute the move + its completion/threat cells.  Output: `support` as
#    4×uint64 per board = the union of cells the found proof line touches (an
#    over-inclusive stencil seed, NOT a minimal stencil), 0 on a non-win.
#
#  * COMPLETE (complete=True, slower): the ROOT OR node no longer short-circuits
#    at the first winning move — it records EVERY winning first move into a
#    `winmask` (4×uint64) and keeps enumerating.  Non-root nodes are untouched
#    (they still short-circuit, which is exactly the per-root-move verdict we
#    want).  "All solutions" == all winning FIRST MOVES (sound: every reported
#    move starts a real forced win; complete: no winning first move is missed),
#    NOT all winning lines/trees (combinatorially larger — AND nodes branch).
#    With support also on, `support` is the UNION over all winning first moves
#    (per-move support would need an (B,N*N,4) output — left as a non-goal).
# ----------------------------------------------------------------------------
def _build_src(support: bool, complete: bool, carriers: bool = False,
               w: bool = False) -> str:
    src = _src()
    if not support and not complete and not carriers and not w:
        return src
    # carriers / w are DERIVED from the proof-support mask, so the support
    # accumulation runs whenever ANY of support / carriers / w is requested; the
    # `support` flag still controls only whether that mask is itself OUTPUT.
    # Consequence: _build_src(s, c, carriers, w=False) is byte-identical to the
    # pre-W source for every (s, c, carriers) — existing variants never change.
    acc = support or carriers or w

    # -- declarations (after `int winmove=-1;`) --
    decls = ""
    if acc:
        decls += "\n    ulong fsupp[%d][4]; ulong supp[4]={0ul,0ul,0ul,0ul};" % MAXD
    if complete:
        decls += "\n    ulong wm[4]={0ul,0ul,0ul,0ul};"
    src = src.replace("    int winmove=-1;\n",
                      "    int winmove=-1;" + decls + "\n", 1)

    # -- zero this frame's support on entry (once per node visit) --
    if acc:
        anchor = ("        if (sp>=MAXD-1 || nodes>maxnodes){ hitcap=true; ret=0; "
                  "entering=false; continue; }\n")
        src = src.replace(anchor, anchor +
                          "        fsupp[sp][0]=0ul;fsupp[sp][1]=0ul;"
                          "fsupp[sp][2]=0ul;fsupp[sp][3]=0ul;\n", 1)

    # -- immediate five (OR inline win) --
    five_supp = "cpy4(c0, fsupp[sp]); " if acc else ""
    if complete:
        five_u = "or4(supp, c0); " if acc else ""
        five_new = ("          if (any4(c0)){ if(sp==0){ or4(wm, c0); " + five_u +
                    "} else { winmove=lowbit4(c0); " + five_supp +
                    "ret=1; entering=false; continue; } }  // immediate five")
    else:
        five_new = ("          if (any4(c0)){ if(sp==0) winmove=lowbit4(c0); " + five_supp +
                    "ret=1; entering=false; continue; }  // immediate five")
    src = src.replace(
        "          if (any4(c0)){ if(sp==0) winmove=lowbit4(c0); ret=1; "
        "entering=false; continue; }  // immediate five", five_new, 1)

    # -- sound double-four (OR inline win) --
    df_supp = ("setbit4(&fsupp[sp][0], m); or4(fsupp[sp], cm); " if acc else "")
    if complete:
        df_u = "setbit4(&supp[0], m); or4(supp, cm); " if acc else ""
        df_new = ("            if (nc>=2){ if(sp==0){ setbit4(&wm[0], m); " + df_u +
                  "continue; } " + df_supp + "dwin=true; break; }    // sound double four")
    else:
        df_new = ("            if (nc>=2){ if(sp==0) winmove=m; " + df_supp +
                  "dwin=true; break; }    // sound double four")
    src = src.replace(
        "            if (nc>=2){ if(sp==0) winmove=m; dwin=true; break; }    "
        "// sound double four", df_new, 1)

    # -- fork three (OR inline win); rm = defender reply mask from gen_threes --
    fork_supp = ("setbit4(&fsupp[sp][0], m); or4(fsupp[sp], rm); " if acc else "")
    if complete:
        fork_u = "setbit4(&supp[0], m); or4(supp, rm); " if acc else ""
        fork_new = ("              if (fk){ if(sp==0){ setbit4(&wm[0], m); " + fork_u +
                    "continue; } " + fork_supp + "dwin=true; break; }  // fork three")
    else:
        fork_new = ("              if (fk){ if(sp==0) winmove=m; " + fork_supp +
                    "dwin=true; break; }  // fork three")
    src = src.replace(
        "              if (fk){ if(sp==0) winmove=m; dwin=true; break; }  // fork three",
        fork_new, 1)

    # -- OR parent, winning child returns (ret==1) --
    orret_nonroot = (("or4(fsupp[sp], fsupp[sp+1]); setbit4(&fsupp[sp][0], mm[sp]); "
                      "if(mb[sp]>=0) setbit4(&fsupp[sp][0], mb[sp]); ") if acc else "")
    if complete:
        root_u = (("or4(supp, fsupp[1]); setbit4(&supp[0], mm[0]); "
                   "if(mb[0]>=0) setbit4(&supp[0], mb[0]); ") if acc else "")
        # at the root: record the winning move, then FALL THROUGH to try the next
        # root candidate (no bubble).  Below the root: bubble as usual.
        orret_new = ("          if (ret==1){ if(sp==0){ setbit4(&wm[0], mm[0]); " + root_u +
                     "} else { " + orret_nonroot +
                     "entering=false; continue; } }                 // OR child won")
    else:
        orret_new = ("          if (ret==1){ " + orret_nonroot +
                     "entering=false; continue; }                 // OR child won -> bubble")
    src = src.replace(
        "          if (ret==1){ entering=false; continue; }                 "
        "// OR child won -> bubble", orret_new, 1)

    # -- AND parent: a reply just won; fold it into this frame's support --
    if acc:
        and_anchor = ("          if (ret==0){ entering=false; continue; }                 "
                      "// a reply refutes -> AND NOWIN")
        src = src.replace(and_anchor, and_anchor +
                          "\n          or4(fsupp[sp], fsupp[sp+1]); "
                          "setbit4(&fsupp[sp][0], ar[sp]);", 1)

    # -- outputs --
    win_line = ("    win[gid]=(uchar)(any4(wm)?1:0); hit[gid]=(uchar)(hitcap?1:0);"
                if complete else
                "    win[gid]=(uchar)ret; hit[gid]=(uchar)(hitcap?1:0);")
    move_line = ("    move[gid]=(int)(any4(wm)? lowbit4(wm) : -1);" if complete else
                 "    move[gid]=(int)((ret==1) ? (winmove>=0 ? winmove : mm[0]) : -1);")
    # the proof-support mask per board: in complete mode the union over winning
    # first moves (`supp`), else the root frame's accumulator on a win (else 0).
    support_expr = "supp[w]" if complete else "(ret==1?fsupp[0][w]:0ul)"
    extra = ""
    if support:
        extra += "\n    for(uint w=0;w<4u;w++){ support[base+w] = " + support_expr + "; }"
    if complete:
        extra += "\n    for(uint w=0;w<4u;w++){ winmask[base+w] = wm[w]; }"
    if carriers:
        # `own` here == the ROOT board: every proof move is unmade on the way back
        # to sp==0 before the loop breaks, so `own` == own_in. carriers = root-own
        # stones collinear-within-4 (COLLIN) of any support cell = the load-bearing
        # stones the proof's five-lines run through (the `B` channel to support's
        # `./p`). Over-inclusive, mirror of support: carriers ⊆ own, disjoint from
        # support, all-zero on a non-win (support_expr is 0 there).
        extra += (
            "\n    ulong _sfin[4]; for(uint w=0;w<4u;w++){ _sfin[w] = " + support_expr + "; }"
            "\n    ulong _carr[4]={0ul,0ul,0ul,0ul}; ulong _ct[4]; cpy4(_sfin,_ct);"
            "\n    while(true){ int _c=lowbit4(_ct); if(_c<0) break; clrbit4(_ct,_c);"
            " _carr[0]|=COLLIN[_c][0];_carr[1]|=COLLIN[_c][1];"
            "_carr[2]|=COLLIN[_c][2];_carr[3]|=COLLIN[_c][3]; }"
            "\n    and4(_carr, own);"
            "\n    for(uint w=0;w<4u;w++){ carriers[base+w]=_carr[w]; }")
    if w:
        # The OPP MIRROR of carriers (issue #90). `opp` here == the ROOT opp board
        # too: every defender reply (AND-node `ar[sp]`) and forced four-block
        # (`mb[sp]`) is unmade on the way back to sp==0 before the loop breaks —
        # the OR-parent unmake clears `opp` of `mb[sp]`, the AND-parent unmake
        # clears `opp` of `ar[sp]`, and the three root inline wins never touch
        # `opp` — so `opp` == opp_in, exactly as `own` == own_in for carriers.
        # w = root-opp stones collinear-within-4 (COLLIN) of any support cell =
        # the OVER-INCLUSIVE load-bearing DEFENDER stones the proof's five-lines
        # run through (the `W` channel; mirror of carriers but intersecting OPP).
        # w ⊆ opp, all-zero on a non-win (support_expr is 0 there). The output
        # array is itself named `w`, so its final write uses a non-`w` loop index
        # (`_wi`) to avoid shadowing it; `_wsfin` may keep `w` as its loop var (it
        # never references the output array).
        extra += (
            "\n    ulong _wsfin[4]; for(uint w=0;w<4u;w++){ _wsfin[w] = " + support_expr + "; }"
            "\n    ulong _wcarr[4]={0ul,0ul,0ul,0ul}; ulong _wct[4]; cpy4(_wsfin,_wct);"
            "\n    while(true){ int _wc=lowbit4(_wct); if(_wc<0) break; clrbit4(_wct,_wc);"
            " _wcarr[0]|=COLLIN[_wc][0];_wcarr[1]|=COLLIN[_wc][1];"
            "_wcarr[2]|=COLLIN[_wc][2];_wcarr[3]|=COLLIN[_wc][3]; }"
            "\n    and4(_wcarr, opp);"
            "\n    for(uint _wi=0;_wi<4u;_wi++){ w[base+_wi]=_wcarr[_wi]; }")
    src = src.replace(
        "    win[gid]=(uchar)ret; hit[gid]=(uchar)(hitcap?1:0);\n"
        "    move[gid]=(int)((ret==1) ? (winmove>=0 ? winmove : mm[0]) : -1);",
        win_line + "\n" + move_line + extra, 1)
    return src


_KERNEL_CACHE: dict = {}


def _get_kernel(support: bool, complete: bool, carriers: bool = False,
                w: bool = False):
    key = (bool(support), bool(complete), bool(carriers), bool(w))
    k = _KERNEL_CACHE.get(key)
    if k is None:
        names = ["win", "hit", "move"]
        if support:
            names.append("support")
        if complete:
            names.append("winmask")
        if carriers:
            names.append("carriers")
        if w:
            names.append("w")
        k = mx.fast.metal_kernel(
            name="mega_vct_bb_%d%d%d%d" % (
                int(support), int(complete), int(carriers), int(w)),
            input_names=["own_in", "opp_in", "max_nodes"],
            output_names=names,
            source=_build_src(support, complete, carriers, w), header=_HEADER)
        _KERNEL_CACHE[key] = k
    return k


# eager default-kernel compile preserves prior import-time behavior (~0.1 s)
_KERNEL = _get_kernel(False, False)


def cells_from_words(words) -> list:
    """Unpack a 4×uint64 cell mask into a sorted list of flat row-major cell
    indices in [0, N*N).  Inverse of the kernel's bit packing (bit r*N+c)."""
    out = []
    for w in range(4):
        x = int(words[w]) & ((1 << 64) - 1)
        base = 64 * w
        while x:
            b = (x & -x).bit_length() - 1
            c = base + b
            if c < N * N:
                out.append(c)
            x &= x - 1
    return sorted(out)


def solve_vct_mega_bb(boards: np.ndarray, *, max_nodes: int = 20000, tg: int = 32,
                      return_move: bool = False, return_support: bool = False,
                      complete: bool = False, return_carriers: bool = False,
                      return_w: bool = False):
    """boards: (B,2,N,N) bool. Solves VCT for the side to move (board[0]=attacker).

    Returns (win, hit_cap): (B,) bool by default. Optional outputs are appended in
    a FIXED order — move, support, winmask, carriers, w — so existing callers never
    break:

    * ``return_move=True`` appends ``move`` (B,) int32 — a VALID (sound, not
      necessarily shortest) VCT first move (flat cell index), -1 where no win was
      proved. In ``complete`` mode it is one winning move (lowest cell of winmask).
    * ``return_support=True`` appends ``support`` (B,4) uint64 — the union of cells
      the found proof line touches (relevance window / stencil seed; over-inclusive
      vs a minimal stencil), all-zero on a non-win. Unpack with ``cells_from_words``.
      NB: support ⊆ EMPTY-at-root — it is the *required-openings* (the ``.``/``p``
      cells the forcing line plays into), NOT the carrier stones. See ``carriers``.
    * ``complete=True`` (slower) changes ``win`` to "∃ a winning first move" and
      appends ``winmask`` (B,4) uint64 — the bitmask of ALL winning first moves
      (sound + complete). With ``return_support`` too, ``support`` is the union
      over every winning first move.
    * ``return_carriers=True`` appends ``carriers`` (B,4) uint64 — the load-bearing
      OWN stones the proof's five-lines run through: every root-own stone
      collinear-within-4 of a support cell. The COMPLEMENT of ``support``
      (carriers ⊆ occupied-own, support ⊆ empty) — the ``B`` channel of a typed
      stencil to support's ``./p``. Over-inclusive (mirror of support), all-zero on
      a non-win. Unpack with ``cells_from_words``. Computed from the support mask,
      so it is available without ``return_support`` (the mask is accumulated either
      way; ``return_support`` only controls whether that mask is also output).
    * ``return_w=True`` appends ``w`` (B,4) uint64 — the OPP MIRROR of ``carriers``:
      the (over-inclusive) load-bearing DEFENDER stones the proof's five-lines run
      through, i.e. every root-opp stone collinear-within-4 of a support cell
      (``w = opp ∩ ⋃_support COLLIN``, vs ``carriers = own ∩ ⋃_support COLLIN``).
      The ``W`` channel of a typed stencil. ``w`` ⊆ occupied-opp-at-root, all-zero
      on a non-win. Unpack with ``cells_from_words``. Like ``carriers`` it is
      derived from the support mask, so it needs neither ``return_support`` nor
      ``return_carriers``. NB by FREESTYLE monotonicity a defender stone never makes
      an attacker VCT *appear*, so ``w`` carries IDENTITY (which forced line), not
      EXISTENCE; the MINIMAL load-bearing ``W`` is the md-ablation program (see
      ``wiki/topics/shape-library-engine.md`` §3/§8), which ``w`` over-approximates.

    So e.g. ``solve_vct_mega_bb(b)`` -> (win, hit); ``return_move=True`` ->
    (win, hit, move); ``complete=True, return_move=True, return_support=True`` ->
    (win, hit, move, support, winmask); adding ``return_carriers=True`` then
    ``return_w=True`` appends ``carriers`` then ``w`` after all of the above. Fully
    on-device.
    """
    B = boards.shape[0]
    own, opp = bb.pack_words(boards)
    o = mx.array(own.reshape(-1))
    p = mx.array(opp.reshape(-1))
    mn = mx.array(np.array([max_nodes], dtype=np.int32))
    kernel = _get_kernel(return_support, complete, return_carriers, return_w)
    out_shapes = [(B,), (B,), (B,)]
    out_dtypes = [mx.uint8, mx.uint8, mx.int32]
    if return_support:
        out_shapes.append((B, 4))
        out_dtypes.append(mx.uint64)
    if complete:
        out_shapes.append((B, 4))
        out_dtypes.append(mx.uint64)
    if return_carriers:
        out_shapes.append((B, 4))
        out_dtypes.append(mx.uint64)
    if return_w:
        out_shapes.append((B, 4))
        out_dtypes.append(mx.uint64)
    outs = kernel(
        inputs=[o, p, mn], grid=(B, 1, 1),
        threadgroup=(min(B, tg), 1, 1),
        output_shapes=out_shapes, output_dtypes=out_dtypes)
    mx.eval(*outs)
    w = np.array(outs[0]).astype(bool)
    h = np.array(outs[1]).astype(bool)
    move = np.array(outs[2]).astype(np.int32)
    idx = 3
    support = winmask = carriers = w_out = None
    if return_support:
        support = np.array(outs[idx]).astype(np.uint64).reshape(B, 4)
        idx += 1
    if complete:
        winmask = np.array(outs[idx]).astype(np.uint64).reshape(B, 4)
        idx += 1
    if return_carriers:
        carriers = np.array(outs[idx]).astype(np.uint64).reshape(B, 4)
        idx += 1
    if return_w:
        w_out = np.array(outs[idx]).astype(np.uint64).reshape(B, 4)
        idx += 1
    res = [w, h]
    if return_move:
        res.append(move)
    if return_support:
        res.append(support)
    if complete:
        res.append(winmask)
    if return_carriers:
        res.append(carriers)
    if return_w:
        res.append(w_out)
    return tuple(res)


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
