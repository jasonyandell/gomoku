# `mega_vct_bb` — the on-device GPU VCT solver (canonical API reference)

**One-line.** `scripts/vct_metal/mega_vct_bb.py :: solve_vct_mega_bb` is the
production VCT (Victory-by-Continuous-Threats) solver: one GPU thread per board
runs the *whole* AND/OR proof search on-device as a bitboard (`own`/`opp`/`empty`
= `ulong[4]`), make/unmake on a thread-local board, **all detection by bitboard
set-algebra**. It is the single solver every threat-shape/mining/labeling
consumer calls, and the self-play oracle veto. **~1600× CPU throughput**,
**0 FP / 0 FN** vs `gomoku.vcf.solve_vct` over 320 VCT + 360 VCF real positions.

This page is the **API + invariants contract + performance reference**. For the
build *narrative* (the CPU-bound v0 spike, the bitboard levers, the throughput
derivation) read [gpu-vct-feasibility.md](gpu-vct-feasibility.md) — that page is
the feasibility record; **this page is the contract.**

---

## 1. How it works — the on-device AND/OR bitboard search

**One GPU thread solves one whole board.** The base kernel is `grid=(B,1,1)`:
each thread runs an iterative AND/OR depth-first proof search over its own board,
in a thread-local frame stack, with make/unmake — **no host orchestration per
node**. This is what makes it fast; it is also what shapes its cost (§2).

**Bitboard set-algebra, not cell scans.** Each colour is a 256-bit board packed
as `ulong[4]` (N²≤256; N=15 here). All threat detection is shift-AND on those
words — `has_five`, `completion_mask` (five-completion cells), forcing-move
generation — done whole-board at once rather than by O(N²) per-cell scan. This
bitboard rewrite was the ~10–14× step over the naive cell-scan megakernel (it
mirrors the VCF result; details in [gpu-vct-feasibility.md](gpu-vct-feasibility.md)
§8).

**The proof tree.** A VCT is a forced win by *continuous threats* — every
attacker move is a four or a forcing three, so the defender is always answering
a threat and never gets a free tempo:

- **OR node (attacker to move):** try each candidate move; the node wins if
  **any** move leads to a forced win. Candidates are the fours (generated once by
  set-algebra) plus the forcing threes (restricted to Chebyshev-2 of own stones —
  a three is own's threat, so its move sits within radius-2 of the stones forming
  it). An **inline win** — immediate five, sound double-four, fork-three — sets
  `ret=1` at the detecting OR frame *without descending*.
- **AND node (defender to move):** the defender must answer the threat; the node
  wins for the attacker only if **every** defender reply still leads to a forced
  win. The bounded reply set is the union of the threats' `{f}∪comps`
  defeating-cell bitmasks (this is the v0 "host tempo-guard + open-four assembly"
  reduced to in-kernel parallel set-algebra).

A **four** advances OR→OR (+1 frame); a **forcing three** advances OR→AND→OR
(+2 frames); an inline win collapses (+0). The search is bounded by `max_nodes`
(a node budget — see §2) and a frame ceiling `MAXD=32` (never binding on real
VCTs — §4).

**Soundness.** The kernel never reports a false win: verdicts match the CPU
oracle `gomoku.vcf.solve_vct` with **0 false-positive / 0 false-negative** clean
disagreements over 320 real VCT positions (258 clean agreements, 8 seeds) and
360 VCF positions. A `hit_cap` verdict means "couldn't prove within budget," NOT
"no win" — always treat it conservatively.

---

## 2. The call-cost law — why every caller must be bulk-synchronous

> ⚡ **One call costs one _tail_: the wall is set by the single hardest board in
> the batch, and it is nearly flat in batch size.** An easy board is
> milliseconds; a hard board (a deep *negative* proof — proving "no VCT" exhausts
> the search) is the floor, and that floor scales with `max_nodes`. Passing
> thousands more boards alongside it is nearly free.

Measured (mn=2000, random midgame boards): **B=16 → 24.6 s, B=256 → 28.5 s,
B=4096 → 40.5 s, B=16384 → 71.7 s** — 1000× the boards for ~3× the wall.
Per-board cost collapses ~350× (1.5 s/board at B=16 → 0.0044 s/board at
B=16384). Compile/import is ~0.1 s, so a fresh `uv run` per query is fine.

**The mechanism:** the GPU scheduler already backfills at threadgroup-dispatch
granularity, so retired lanes' slots get pending boards — *throughput is free,
latency is fixed by the deepest board's serial single-thread grind.*

**Binding rules for every caller:**
1. **Be bulk-synchronous. Never solve-in-a-loop on a small batch** — a 25-board
   call wastes ~99% of its wall. Gather every board you need (up to ~16 k) into
   *one* call; sequential consumers (e.g. stencil ablation) march all items in
   lockstep, one candidate per item per call.
2. **`max_nodes` is the tail knob** — lower it to shrink the floor. It is a
   **GLOBAL pool across the batch**, so for corpus labeling scale `max_nodes`
   with B. A CAP verdict is fail-safe wherever "couldn't prove it" can be treated
   conservatively.
3. The floor itself only moves with a kernel that lets threads cooperate on the
   *single deepest* board — that is the `lanes=K` lever (§ Performance).

Full derivation + numbers: [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8.

---

## 3. API (the contract)

```python
solve_vct_mega_bb(boards, *, max_nodes=20000, tg=32,
                  return_move=False, return_support=False, complete=False,
                  return_carriers=False, return_w=False, max_depth=None,
                  work_steal=False, resident=8192, lanes=1)
solve_md_min(boards, *, max_nodes=20000, lo=1, hi=None, tg=32)  # -> (md, capped)
solve_vct_streaming(boards, *, budgets=(250,1000,4000,20000),    # -> (win, hit[, move])
                    work_steal=False, resident=8192, tg=32, return_move=False, log=None)
```

`boards`: `(B, 2, N, N)` bool, **side-to-move-relative** — `board[0]` is the
**attacker** (the side to move whose forced win we solve), `board[1]` the
defender. **No swap**: at any winner-to-move position `GameState.board` is already
in this frame for **black AND white** alike (verified — getting it wrong silently
solves the wrong side; see [vct-backward-mining.md](vct-backward-mining.md) §2).
Free-style rules (overlines win). `N = GOMOKU_BOARD_SIZE` (15 here); the kernel is
a 256-bit `ulong[4]` board so N²≤256.

Returns a tuple. The default is `(win, hit_cap)`; optional outputs append in a
**FIXED order — `move`, `support`, `winmask`, `carriers`, `w`** — so existing callers
never break:

| flag | appends | type | meaning |
|---|---|---|---|
| *(default)* | `win` | `(B,)` bool | a forced VCT win exists for the attacker |
| *(default)* | `hit_cap` | `(B,)` bool | search hit `max_nodes`/depth → verdict inconclusive (not a no-win) |
| `return_move=True` | `move` | `(B,)` int32 | a VALID (sound, not necessarily shortest) VCT first move, flat cell index; `-1` on a non-win |
| `return_support=True` | `support` | `(B,4)` uint64 | union of cells the found proof line touches (relevance window / stencil seed); all-zero on a non-win |
| `complete=True` | `winmask` | `(B,4)` uint64 | bitmask of **ALL** winning first moves; also flips `win` to "∃ a winning first move" |
| `return_carriers=True` | `carriers` | `(B,4)` uint64 | the load-bearing OWN **stones** the proof's five-lines run through (the `B` channel complementing `support`'s `./p`); all-zero on a non-win |
| `return_w=True` | `w` | `(B,4)` uint64 | the OPP mirror of `carriers`: the (over-inclusive) load-bearing DEFENDER **stones** on the proof's lines (the `W` channel) — `w = opp ∩ ⋃_support COLLIN`; all-zero on a non-win |

There is also **one optional INPUT** (issue #91): `max_depth` (int or `(B,)` int32)
caps the proof search at a per-board **frame depth** and adds NO output — it only
restricts `win` to "∃ a forced VCT within `max_depth` frames" (cut branch → clean
no-win, never `hit_cap`). It gates its own compiled variant (default verdict
byte-identical). The wrapper **`solve_md_min(boards) -> (md, capped)`** binary-searches
it to the **order-independent** mate distance md_min (see the `max_depth` / `md_min`
section below).

Unpack a `(4,)` uint64 mask to flat cell indices with
`mega_vct_bb.cells_from_words(words)` (inverse of the kernel's bit packing, bit
`r*N+c`).

Examples: `solve_vct_mega_bb(b)` → `(win, hit)`; `return_move=True` →
`(win, hit, move)`; `complete=True, return_move=True, return_support=True` →
`(win, hit, move, support, winmask)`; adding `return_carriers=True` then
`return_w=True` appends `carriers` then `w` after all of the above.

**Each flag compiles its own kernel variant** (memoized in `_KERNEL_CACHE`). The
default `(support=False, complete=False)` variant is built from a source string
**byte-identical** to the original — so the default verdict and throughput are
provably unchanged. Variants cost compute, not the fast path.

---

## 4. The optional outputs, precisely

### `move` (passive winning move) — `return_move=True`
A *passive* read of the move the search already commits to: the root's chosen
move `mm[0]`, or the captured cell for an inline root win (immediate five / sound
double-four / fork-three). **No extra nodes, no extra search.** Sound because the
kernel never reports a false win, so the root of the line it proves always starts
a real forced win. It is *a* valid VCT move, not necessarily the shortest mate's.
Verified independently on 400 puzzles (play the move, attacker wins vs every
defender reply). Shipped 2026-06-26; see [vct-backward-mining.md](vct-backward-mining.md) §5.

### `support` (proof relevance window) — `return_support=True`
The set of cells the found proof line **touches** — a stencil seed / relevance
window, deliberately **over-inclusive vs a minimal stencil** (the downstream
shape-library engine ablates it down). Built by **return-path accumulation**: a
per-frame `fsupp[]` that merges a child's support into its parent **only on a
winning return** (`ret==1`), so abandoned/refuted branches **never pollute it**.
Contributions: an OR node that wins adds its move (+ the forced four-block); an AND
node that wins adds **every** defender reply; the three inline OR wins add the move
+ its completion/threat cells. **`support` ⊆ the root's EMPTY cells** — it is the
*played* cells of the proof (moves, blocks, replies, completions), NOT the
pre-existing threat-carrier stones (those stay on the board for the ablation pass).
All-zero on a non-win. Validated: cells empty-at-root, contain the move on every
win, empty on every loss. **`support` is the OPENINGS half of the proof shape; the
complementary stone half is `carriers` (below).**

### `carriers` (load-bearing stones) — `return_carriers=True`
The complement of `support`: the **OWN stones the proof's five-lines run through** —
the `B` channel to `support`'s `./p`. Where `support` answers *"which empty cells
must the forcing line fill,"* `carriers` answers *"which already-placed stones make
those lines win."* Defined as every **root-own** stone **collinear-within-4** (the
`COLLIN` table — the same ≤4-along-axis domain a five spans) of any `support` cell,
computed once at output time from the support mask (`own` is back to the root board
there — every proof move is unmade before the search returns to `sp==0`).
**`carriers` ⊆ occupied-own-at-root** (mirror of `support` ⊆ empty), disjoint from
`support`, all-zero on a non-win. A typed stencil is `support ∪ carriers` — e.g.
`.BBBB.` → carriers = the four `B`, support = the two ends. **Over-inclusive** (the
relevance window's stones, not the minimal load-bearing set — that needs the L1
ablation); defender (`W`) load-bearing stones are **not** included (a v2). Derived
from the support mask, so available **without** `return_support` (the mask is
accumulated regardless; `return_support` only controls whether it is *also* output).
Added 2026-06-27 (issue #88) — `support` had been read as "which stones formed the
VCT" when it is the required-openings; `carriers` is the stones half.

### `w` (load-bearing defender stones) — `return_w=True`
The **OPP mirror of `carriers`**: where `carriers` is `own ∩ ⋃_{support} COLLIN`,
`w` is `opp ∩ ⋃_{support} COLLIN` — every **root-opp** stone **collinear-within-4**
of a support cell, i.e. the **defender stones sitting on the proof's five-lines**
(the `W` channel of a typed stencil). Computed at output time from the same support
mask as `carriers`; `opp` is back to the root board there (every defender reply
`ar[sp]` and forced four-block `mb[sp]` is unmade before the search returns to
`sp==0` — verified in the source comment, exactly as `own==own_in` for `carriers`).
**`w` ⊆ occupied-opp-at-root**, disjoint from `support` (opp vs empty) and from
`carriers` (opp vs own), all-zero on a non-win. Derived from the support mask, so
available without `return_support` **or** `return_carriers` (the mask is accumulated
whenever any of support/carriers/w is requested).

**IDENTITY, not EXISTENCE — and OVER-INCLUSIVE.** By **freestyle** monotonicity
([shape-library-engine.md](shape-library-engine.md) §3) a defender stone **never
makes an attacker VCT appear** — adding defender stones only hurts or is neutral.
So `W` is **never needed for the win to EXIST**; it carries **identity** (which
forced line / mate-distance). Measured (`scripts/threat_shapes/w_channel_probe.py`,
pool 4096, `max_nodes=500`): of **660** clean attacker VCTs, **660/660 (100%)** still
win after removing **every** defender stone (zero monotonicity violations; seed 1:
697/698, the one miss a post-strip cap, not a violation) — yet **96%** had ≥1 `w`
stone on their proof lines (mean `|w|`≈6). Defenders *sit on* the lines; none are
*load-bearing for existence*. Like `carriers`, `w` is the **relevance window's**
defender stones (collinear-near support), **not** the minimal set: the MINIMAL
load-bearing `W` is the **md-ablation** program (a stone whose removal *shortens*
the mate), which is **BLOCKED on md-extraction**
([shape-library-engine.md](shape-library-engine.md) §3 correction #2 / §8) — `w`
is the cheap, available-today over-approximation of it. This is the **v2 `W`
channel** the certificate program flagged as needed for *defense-flavored* shapes
that do **not** win in isolation. Added 2026-06-27 (issue #90).

### `winmask` (all winning first moves) — `complete=True`
The default search short-circuits at the root OR node (first winning move wins).
Complete mode instead tries **every** root candidate and records each winning
first move in `winmask`; non-root nodes are untouched (they still short-circuit —
which is exactly the per-root-move verdict). **"All solutions" == all winning
FIRST MOVES**, not all winning lines/trees (combinatorially larger — AND nodes
branch). It is **slower** (no root short-circuit ⇒ strictly more work) and shares
the same global `max_nodes` pool. `winmask` is the winning **forcing** first moves
(fours + forcing-threes passing the tempo guard) = exactly the VCT first moves; a
non-forcing "free win" in an already-won position is NOT a VCT first move and is
correctly excluded. With `return_support` too, `support` is the **union** over all
winning first moves (per-move support would need a `(B,N²,4)` output — left a
non-goal). `move` in complete mode is one winning move (lowest cell of `winmask`).

**Validated sound + complete** (`tests/.../test_mega_vct_bb.py` invariants +
`gold_complete.py`, 2026-06-27): over the gold boards, **0 unsound** winmask moves
(each independently verifies as a forced win) and **0 winning forcing moves
missing**. The completeness oracle is vcf's exact root candidate generation
(fours ∪ forcing-threes) **including the tempo guard** `_defender_has_four_or_five`
— omitting that guard was a *verifier* bug that produced phantom "misses"; the
solver was right.

### `max_depth` (per-board frame cap) + `solve_md_min` (mate distance) — issue #91
The **input** `max_depth` caps the search at a per-board **frame depth**: a branch
that reaches frame `sp == max_depth` returns a **clean `ret=0`** (a definitive "no
forced win within `sp < max_depth` frames") **without** descending and **without**
setting `hit_cap`. The cut is placed *after* the node/structural cap, so an
out-of-`max_nodes` board still latches `hit_cap` (inconclusive) and wins the race —
the depth cut and the node cap are kept semantically distinct (one is "no win within
this depth", the other is "couldn't afford to decide"). It makes **no move before
cutting**, so the `own==own_in` / `opp==opp_in` at-break invariant carriers/`w` rely
on is preserved. Its own compiled kernel variant; the **default verdict is
byte-identical** and every prior variant is unchanged (invariant #9).

**`md_min(b)` = `min{ d : solve_vct_mega_bb(b, max_depth=d).win }`** — read from the
boolean verdict alone, so it is **ORDER-INDEPENDENT** (no move ordering / OR
short-circuit can move a True/False threshold), **monotone** (`win(d) ⟹ win(d+1)`),
and **minimax-correct** (OR = ∃ over attacker moves, AND = ∀ over the bounded reply
set). `solve_md_min(boards) -> (md, capped)` finds it by a **per-board binary search**
over `[lo, hi]` — every board marches its own bracket in one bulk call, so the whole
corpus resolves in `⌈log2(hi−lo+1)⌉ ≈ 5` flat tails (the call-cost law). `md` is `-1`
on a clean no-win or where `capped` (a probe hit `max_nodes`; re-run at higher budget
— md is never silently wrong, only **withheld**).

**FRAME unit — NOT attacker-plies-to-five.** A **four** advances OR→OR = **+1 frame**;
a **forcing three** advances OR→AND→OR = **+2 frames**; an **inline win** (immediate
five / sound double-four / fork-three) sets `ret=1` at its detecting OR frame
*without descending* = **collapses (+0)**. So `md_min = F + 2T + 1` for a
defender-maximised line with `F` fours and `T` threes before the collapsed leaf.
md_min is an affine, **coarser-at-the-leaf** measure than the CPU `mate_distance`
(which *expands* the leaf the GPU collapses). The unit (`md = F + 2T + 1`) is
**source-traced, not independently measured.** Two honest, **length-over-estimating /
over-keeping** (never unsound — L0 re-verifies) bounds: (i) the `def_tempo` veto can
inflate three-opening lines; (ii) the inline collapse can hide a ≤2–3-ply shortening
at constant `sp` (a future fix = emit `md = sp + leaf_offset`, additive).

**Validation SCOPE (don't oversell it).** What is checked: byte-identical default vs
HEAD (16/16 flag combos), depth-**monotonicity** (`win(d)` never True→False — also true
by construction: more budget can't delete a shallower proof), the **bracket**
(`win@md` clean, `win@md−1` clean no-win), and `md_min == an independent **linear scan**`.
The linear scan uses the *same* kernel, so this validates **internal consistency + the
depth-cap mechanism**, NOT md_min's *absolute value* against an independent oracle. None
is well-calibrated: a *live* CPU md searches a **different fragment** (the kernel's
`candidate_own` own-only Cheb-2 is narrower than CPU's any-stone candidate set ⇒
`md_gpu > md_cpu` with no bug), and using it re-summons the retired solver. The verdict
half (`md≥1 ⟺ win`) is cross-checked against the committed `vct_golden.npz` labels. **One
sound external check is open:** on the **VCF (four-only) subset** the two solvers'
fragments coincide (the `candidate_own` gap only affects *threes*), so a gated CPU
`mate_distance` cross-check there *would* independently pin the absolute md — a durable
follow-up, not yet run. Drives the L1 **md-invariant stencil minimizer**
([shape-library-engine.md](shape-library-engine.md) §3/§8).

---

## 5. Performance

The runtime map of the mega-VCT solver, in five facts. Tools:
`scripts/vct_metal/{bench_throughput,sweep_throughput,analyze_sweep,n_sweep,maxd_study,bench_lanes13,bench_gen_refill}.py`;
raw logs + `REPORT.md` under `~/data/idx2_solve/sweep/` (out-of-git).

### 5.1 Peak throughput and the width knob
Fully on-device, RSS ~0.3 GB, ~1600× aggregate CPU (`vcf.solve_vct` = 0.64
solves/s). Solves/s saturate with batch size as the wall stays ~flat (call-cost
law): mn=1500 real positions → B=8192 → 526, 16384 → **891**, 32768 → 1020
(≈ GPU concurrency ceiling). At tight screening budgets the *screening* rate is
enormous — quiet base@10 = **7.9M evals/min** (see §5.4 on why that resolves ~71%
and no more).

**Width is king.** Refill/streaming throughput climbs ~linearly with the number
of lanes in flight (`resident`) up to saturation:

| resident R | 1024 | 2048 | 4096 | 8192 | 16384(=N) | oneshot |
|---|---|---|---|---|---|---|
| vs oneshot | 0.12× | 0.22× | 0.40× | 0.66× | **0.96×** | 1.00× |

**If you can gather a big batch, run it wide in one dispatch** — that beats any
narrow strategy by 3–8×. Going narrow costs ~0.34–0.40× *however* you handle the
tail. Never chunk (`chunked@16384` = 0.63–0.69× — a dispatch tail *per chunk*).

### 5.2 Streaming / work-stealing — `work_steal` + `solve_vct_streaming` (#93/#94/#96)
Two levers for pools you **cannot** gather up front (streamed, memory-bound, or
forced-narrow waves). Both are verdict-invariant (invariant #10).

- **`work_steal=True`** launches `resident` persistent lanes that pull the next
  board from a shared atomic cursor, so a lane that finishes an easy board grabs
  another instead of idling. On a **single in-memory pool it is a no-op-to-loss**
  (measured 0.93–0.97× at full width; 0.72–0.79× across pool shapes) — the
  hardware already backfills one dispatch, so the cursor is pure overhead. Its
  **only** justification: a pool larger than one dispatch can hold, or arriving
  incrementally. There it beats **relaunch-per-wave** by **1.20–1.29×** (finer
  forced slicing → bigger win) because the cursor erases the per-wave tail. But
  the tail is the *small* cost — width dominates (§5.1), so narrowing to gain
  refill is a net loss unless you were forced narrow already.
- **`solve_vct_streaming` (iterative deepening)** solves the whole pool at
  `budgets[0]`, re-solves only the still-capped survivors at `budgets[1]`, etc. A
  clean (non-capped) verdict **latches** — it is budget-independent — so only the
  shrinking hard tail pays the deep budgets. **Deepens on the base kernel by
  default**; `work_steal` deepening loses (its big round-0 handicap eats the win).
- **Hard caveat:** refill only helps if the next wave's boards are *already in
  hand*. A **gated** frontier (wave K+1 depends on K's verdicts) cannot be
  refilled — there the run-dry tail is unavoidable, and the only lever left is
  parallelizing one proof across lanes (§5.3).

### 5.3 The `lanes=K` multi-thread-per-board kernel (#114)
The call-cost floor is one weak thread grinding the single deepest board; at
13×13 half the batch is capped (§5.4), so *every* simdgroup is saturated with
hard lanes and all the dispatch-level levers (ladder, oracle-sort, tg) have
nothing to grab. `lanes=K` (K ∈ {2,4,8,16,32}) attacks the floor directly:

**K simd lanes cooperate on one board.** All K replicate the board and run the
AND/OR DFS in lockstep (identical values, never diverge, nothing shared); only
the two OR-node candidate scans (fours / forcing-threes — the per-node hot cost)
partition work: lane j takes set-bit ranks j, j+K, …, and partials merge with
intra-cluster simd reductions (OR for masks, MIN for the winning cell). Min ==
lowbit order, so the verdict is **BIT-IDENTICAL** to base — same move, node count,
`hit_cap` (invariant #11). This cuts both per-node latency (scan work ÷ K) and
intra-simdgroup divergence (32/K boards per simdgroup). Cost: K× threads, so past
saturation (B×K ≳ ~25 k) a smaller K wins.

**Measured (2026-07-02, `bench_lanes13.py` — real gen batches replayed per
variant, verdict-equality asserted).** 132 real 13×13 merged-veto batches
(360,925 boards, cap50): K=2/4/8/16 → **1.09 / 1.23 / 1.34 / 1.36×** solve wall.
15×15 narrow batches show the mechanism's ceiling (K=8 = **1.72×** @B=150, 1.46×
@B=1500; at B=6000 K=4 wins at 1.33×). End-to-end gen (48 games @32 concurrent,
13×13): wall **67.7→52.6 s (1.29×)**, aug-pos/s 297.5→383.2, join stall
25.8→12.5 s, game stream identical. The prediction was 2–4×; measured 1.34× on
real 13×13 batches — the K× thread inflation eats half the divergence win at real
widths (the 1.72× narrow number is the mechanism at its best).

`GOMOKU_VCT_LANES` (env, default 0=off; read by `gomoku/self_play.py`) turns it
on for the gen oracle path — off = the byte-identical base kernel. `tg` must be a
multiple of 32; v1 is base-verdict-only (optional outputs / work_steal raise).

### 5.4 Hardness is bimodal; MAXD=32 is sufficient (#95)
Resolution across the mixed population is **bimodal**: 70–85% of boards resolve at
budget *10*, then a steep plateau, then a **near-bottomless hard tail** that 80×
the budget barely touches. Cumulative % resolved:

| pool | @10 | @250 | @1000 | @20000 | still capped@20k |
|---|---|---|---|---|---|
| quiet (frontier) | 71% | 76% | 77% | 78% | **22%** |
| capped-only | 0% | 0% | 3% | 6% | **93.7%** |
| random | 72% | 82% | 84% | 85% | 15% |
| deep (high ply) | 83% | 92% | 94% | 95% | 5% |

Consequences: **(a)** budget beyond ~250 buys almost nothing on the easy mass, so
for max *resolved/min* **screen cheap (~10–100) then batch the hard survivors as
ONE dense high-budget dispatch** (a dense all-hard batch saturates the GPU at
~0.9M nodes/s vs ~0.35–0.49M mixed). **(b)** The capped tail is genuinely hard,
not a budget away — `max_nodes` can't fill it (6.5% resolved at 80× budget) and
neither can the frame ceiling: the **MAXD 32→64 study finds *exactly zero* new
wins** (`gained_by_64 = 0`), so real VCTs essentially never exceed 32 forcing
frames. **Keep MAXD=32** (MAXD=64 costs ~5% throughput for nothing); the
`GOMOKU_VCT_MAXD` knob stays for probes.

**Deepening's win is conditional on scale.** deepen `(250,1000,4000)` vs
base@4000 = 0.69× / 0.79× / 1.19× / 1.63× at N = 10k / 20k / 40k / 84k — crossover
≈ **30–35k** boards. Below that a single base@ceiling dispatch wins; above it the
deep rounds stay dense with hard boards and saturate the GPU. So the recipe is:
**screen-cheap → batch-hard → deepen only if the survivor set is ≳30k.**

### 5.5 The oracle dominates self-play gen — optimize the SOLVER, not the gen loop (2026-07-02)
The sound-world gen recipe (`--vct-terminus --oracle-veto`, cap50) makes the VCT
solver the gen bottleneck, and at 13×13 it is overwhelming. Perf isolation
(`bench_gen_refill`, same small 64×4 net, sims=100, net size held constant):

| config | games/min | aug_pos/s | oracle_s / wall_s |
|---|---|---|---|
| oracle ON, lockstep | 215.6 | 449 | 16.2 / 17.8 (**91%**) |
| oracle ON, streaming (concurrent 64→256) | 216.4 | 451 | 16.2 / 17.7 |
| oracle OFF, lockstep | 638 | 2497 | 0 |
| oracle OFF, streaming | 644 | 2520 | 0 |

Two load-bearing findings:

1. **The VCT oracle veto is 91% of the gen wall at 13×13** — turning it on cuts
   throughput ~3× (638→216 games/min). This is the `lanes=K` kernel's domain
   (§5.3), NOT the gen loop's.
2. **Streaming ≈ lockstep in a single process** (216 vs 216 games/min, oracle on
   or off) — the refill loop adds ~0 single-process throughput. **#112's
   3.4–4.6× was an 8-proc FLEET → 1 wide-proc comparison** (a fleet-consolidation
   win, see [mcts-perf-ceiling.md](mcts-perf-ceiling.md)), NOT a gen-path
   overhaul. Do not cite it as a single-process refill speedup.

**PERF LEVER PRIORITY:** because the oracle dominates, the lever for faster
sound-world gen is the **VCT solver/kernel** (`lanes=K`, cap50→cap25 recall study,
tighter per-node work), **not** the gen loop. Prior 9×9-era levers on
`gomoku/self_play.py` (merged per-ply solve = 1.06–1.07× byte-identical default-on;
null-board precheck **refuted** at both 9×9 and 13×13; oracle/search overlap =
1.18× via GIL-released MLX under MPS contention) are second-order next to the
kernel floor. Full gen-loop lever history + the bimodal 13×13 census:
[mcts-perf-ceiling.md](mcts-perf-ceiling.md) (2026-07-01 / 2026-07-02 sections).

---

## 6. Invariants (the regression contract)

1. `return_support` leaves `(win, hit, move)` **byte-identical** to default.
2. On boards neither solve capped, **complete `win` == default `win`**.
3. On a clean win, the default `move` is a member of `winmask`.
4. `support` cells are empty at root, contain the move on a win, empty on a loss.
5. `return_carriers` leaves `(win, hit, move, support)` **byte-identical**; indeed
   `_build_src(s, c, carriers=False)` is byte-identical to the pre-carriers source
   for every `(s, c)`, so **no existing variant changes** (carriers gates a brand
   new compiled kernel only).
6. `carriers` ⊆ occupied-own-at-root, disjoint from `support`, empty on a non-win;
   on the `.BBBB.` / `BB.BB` golden boards `carriers` == exactly the four `B` stones.
7. `return_w` leaves `(win, hit, move, support, carriers)` **byte-identical**;
   `_build_src(s, c, carriers, w=False)` is byte-identical to the pre-W source for
   every `(s, c, carriers)` (verified against a pre-edit snapshot of all 8 existing
   variants), so **no existing variant changes** (`w` gates a brand new compiled
   kernel only).
8. `w` ⊆ occupied-opp-at-root, disjoint from `support` and `carriers`, empty on a
   non-win; on the `.BBBB.` golden board with defenders at COLLIN distances 1/4/5
   from a support cell, `w` == exactly the within-4 (distance 1 and 4) defenders.
9. **`max_depth` / depth_cap (#91) is purely additive.** `_build_src(s,c,cr,w,depth_cap=False)`
   is **byte-identical** to today's `_build_src(s,c,cr,w)` for all 16 `(s,c,cr,w)`
   (verified against `git HEAD`); `depth_cap=True` adds **exactly** one input
   (`max_depth`) and the two injected strings, **zero outputs** — so no existing
   compiled variant changes. Runtime: `max_depth=MAXD-1` reproduces the default
   `(win,hit,move)` exactly (the structural cap fires first → the cut is dead code
   there); `win(d)` is monotone non-decreasing; `solve_md_min` brackets it
   (`win@md` clean, `win@md-1` clean no-win) and `md>=1 ⟺` the default clean win.
10. **`work_steal` (#93) is verdict-invariant.** `_build_src(False, False)` is unchanged
    (`== _src()`); `work_steal=True` only wraps the body in the cursor-pull loop and
    swaps the three base output writes for atomic stores — so `(win, hit, move)` is
    **byte-identical** to the base kernel for every seed/budget and every
    `resident ∈ {<B, ≈B, ≫B}`, down to sub-threadgroup `B`. Combining it with any
    optional output raises. `solve_vct_streaming` latches budget-independent clean
    verdicts ⇒ agrees with a deepest-budget base call on all mutually-clean boards.
11. **`lanes=K` (#114) is verdict-invariant.** `_build_src(False, False)` is
    unchanged (`== _src()`); `lanes>1` only lane-partitions the two OR-node
    candidate scans and merges with simd reductions whose MIN commit == the
    sequential scan's lowbit4 order — so `(win, hit, move)` is **bit-identical**
    to base for every K ∈ {2,4,8,16,32}, every budget, down to sub-threadgroup
    and B=1. Combining with any optional output / work_steal raises; non-power
    K raises. Covered by `test_lanes_source_untouched_default`,
    `test_lanes_rejects_optional_outputs`, `test_lanes_byte_identical`.

Covered by `scripts/vct_metal/test_mega_vct_bb.py` — `test_support_and_complete_invariants`,
`test_carriers_golden_shapes`, `test_carriers_invariants`, `test_w_golden_shapes`,
`test_w_invariants`, `test_depth_cap_byte_identical_and_composes`,
`test_depth_cap_ceiling_equals_default`, `test_depth_monotonic`, `test_md_min_bracket`,
`test_md_matches_golden`, `test_work_steal_source_untouched_default`,
`test_work_steal_rejects_optional_outputs`, `test_work_steal_byte_identical`,
`test_streaming_consistent_with_base`
(run via `GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_metal.test_mega_vct_bb`;
**budget `max_nodes=500` captures ~90% of VCTs and runs in seconds** — Jason's test
default). The default-verdict cross-check vs the cell-scan `mega_vct` (0
disagreements) is the same module's `test_mega_vct_bb_matches_mega_vct`.

---

## 7. Gotchas

- **Plane convention is side-to-move-relative, `board[0]`=attacker, no swap** — the
  #1 silent-wrong-answer trap.
- **Free-style** (overlines win); VCT-win monotonicity holds under free-style.
- **256-bit board** (`ulong[4]`, N²≤256). `uint[8]` was tried and reverted (~20%
  slower in situ — the kernel is bookkeeping-heavy; see
  [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 "word width").
- **Tail-bound, bulk-synchronous only** — see the call-cost law (§2). Keep callers
  bulk-synchronous; never solve-in-a-loop.
- **`hit_cap` is not a no-win** — it's "couldn't prove within budget". Treat
  conservatively.
- Adding the `support`/`winmask` outputs costs nodes/compute, not the fast path
  (separate compiled variants; default is byte-identical).

---

## 8. CPU solver retired (2026-06-27) — gate, env override, fast/deep test tiers

The CPU oracle `gomoku/vcf.py` (`solve_vcf` / `solve_vct` + the `*_from_planes`
wrappers) is **retired as a runtime dependency**. It is kept **intact** as a
bootstrap/reference oracle (open source, every internal helper preserved), but
every public entry point now **throws `gomoku.vcf.CpuSolverRetired`** at the top
unless the deliberate-use override is set:

```bash
GOMOKU_ALLOW_CPU_SOLVER=1   # the ONLY sanctioned bypass (fixture-gen / deep validation)
```

The message names the replacement (`solve_vct_mega_bb`), the slowness reason
(~0.65 ms/node, the ~90 s deep-search tail on hard 15×15 boards — the wall this
retirement removes), and the override. **Why gate, not delete:** runtime reaches
(MCTS leaf-VCF, eval overlay, self-play teachers, web/play) should *surface*
themselves by throwing so they can be triaged to the GPU solver place-by-place —
no silent CPU feature-parity is wanted. Internals stay untouched: the kernel/ref
scaffolding and the fixture-gen / deep-validation paths import helpers
(`_five_completions`, `_defender_has_four_or_five`, …) directly, which never trip
the gate.

**Test tiering** (the fix for the slow test walls — the walls were the *oracle* in
the loop, not the kernel, which is 4 s flat at B=16384):

| tier | file | oracle | gate? | wall |
|------|------|--------|-------|------|
| FAST | `scripts/vct_metal/test_mega_vct_bb.py` | committed golden npz (no vcf) | no | <15 s |
| GATE | `tests/` (`uv run pytest`) | live vcf, `conftest.py` sets override | sanctioned | — |
| DEEP | `scripts/vct_metal/validate_deep.py` | live vcf, high budget, larger n | sets override | on-demand |

- **Reusable fast-test pattern — commit a golden fixture, never re-derive truth at
  test time.** `regen_vct_fixture.py` (one-shot, sets the override) solves a fixed
  seeded position stack with the CPU oracle at a bounded budget, keeps only
  **clean / non-capped** boards (definitive truth), and banks
  `(boards, win, move, winmask)` + seed/budget into a small compressed
  `scripts/vct_metal/fixtures/vct_golden.npz`. The fast test loads the npz and
  diffs `solve_vct_mega_bb` at `max_nodes=500` against it (a non-capped verdict is
  budget-independent, so a tight budget must match the high-budget truth;
  `hit_cap` → skip). The slow cell-scan `mega_vct` cross-check is **gone** from the
  fast path; the support/complete **structural-invariants** test stays (the kernel
  as its own oracle — fast).
- **DEEP completeness needs vcf's tempo guard.** `validate_deep.py`'s forcing-move
  oracle mirrors `_vct_attack` root candidate generation *including*
  `_defender_has_four_or_five` — the subtlety that earlier made a *sound* kernel
  look "incomplete" (the kernel was right; the verifier was missing the guard). A
  verified-win move that is NOT forcing is a free win in an already-won position,
  correctly absent from the winmask.
- `tests/test_cpu_solver_gate.py` covers the gate itself (entry points raise when
  the env is cleared; the override unblocks them).

## 9. Consumers (who calls this)

`scripts/threat_shapes/`: `mine_vct_gpu_flat.py`, `mine_first_vct.py`,
`solve_puzzles.py`, `mine_puzzles.py`, `harvest_molecules.py`, `vct_fan.py`,
`probe_timing.py`, `certificate_falsification.py` (the `carriers` certificate),
`w_channel_probe.py` (the `w` identity-not-existence probe, #90),
`md_minimize.py` (the `max_depth`/`solve_md_min` md-invariant stencil minimizer, #91).
`gomoku/self_play.py` is the self-play **oracle veto** consumer (`GOMOKU_VCT_LANES`
gates `lanes=K`). Feeds [shape-library-engine.md](shape-library-engine.md) (L1
stencils), [vct-backward-mining.md](vct-backward-mining.md),
[vct-reachability-mining.md](vct-reachability-mining.md), and
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md).

**Cross-links:** [gpu-vct-feasibility.md](gpu-vct-feasibility.md) (build narrative
+ throughput derivation) · [mcts-perf-ceiling.md](mcts-perf-ceiling.md) (the gen
loop; oracle-dominates finding) · [vct-mining-research.md](vct-mining-research.md)
(the mining programs this feeds) · [vct-backward-mining.md](vct-backward-mining.md)
§5 (the move output) · [allis-threat-theory.md](allis-threat-theory.md) (VCF/VCT
formalism) · `gomoku/vcf.py` (the CPU oracle `solve_vct`).
