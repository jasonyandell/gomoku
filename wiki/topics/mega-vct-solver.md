# `mega_vct_bb` — the on-device GPU VCT solver (canonical API reference)

**One-line.** `scripts/vct_metal/mega_vct_bb.py :: solve_vct_mega_bb` is the
production VCT (Victory-by-Continuous-Threats) solver: one GPU thread per board
runs the *whole* AND/OR proof search on-device as a bitboard (`own`/`opp`/`empty`
= `ulong[4]`), make/unmake on a thread-local board, **all detection by bitboard
set-algebra**. It is the single solver every threat-shape/mining/labeling
consumer calls. **~1600× CPU throughput**, **0 FP / 0 FN** vs `gomoku.vcf.solve_vct`
over 320 VCT + 360 VCF real positions.

This page is the **API + invariants reference**. For *why* it is built this way
(the CPU-bound v0 spike, the bitboard levers, the throughput characterization)
read [gpu-vct-feasibility.md](gpu-vct-feasibility.md) — that page is the
narrative/feasibility record; **this page is the contract.**

---

## ⚡ THE CALL-COST LAW (the binding constraint — internalize before calling)

**One call costs one _tail_: the wall is set by the single hardest board in the
batch and is nearly flat in batch size.** B=16 → 24.6 s, B=16384 → 71.7 s — 1000×
the boards for ~3× the wall; per-board cost swings ~350×. Compile is ~0.1 s (a
fresh `uv run` per query is fine). Therefore **every caller MUST be
bulk-synchronous**: gather every board you need (up to ~16 k) into *one* call;
never solve-in-a-loop on a small batch. `max_nodes` is the tail knob (it bounds
the single deepest board's search) and is a **GLOBAL pool across the batch** — for
corpus labeling, scale `max_nodes` with B. A CAP verdict is fail-safe wherever
"couldn't prove it" can be treated conservatively. Full numbers + derivation:
[gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 banner.

---

## API

```python
solve_vct_mega_bb(boards, *, max_nodes=20000, tg=32,
                  return_move=False, return_support=False, complete=False,
                  return_carriers=False, return_w=False, max_depth=None,
                  work_steal=False, resident=8192)
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

## The three outputs, precisely

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
forced win within `sp < max_depth` frames") **without** descending and **without
setting `hit_cap`**. The cut is placed *after* the node/structural cap, so an
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

## Streaming / work-stealing — `work_steal` + `solve_vct_streaming` (issue #93)

**The problem.** The base kernel is `grid=(B,1,1)` — one GPU thread per board, each
running the whole AND/OR DFS in a thread-local stack until it wins, loses, or hits
`max_nodes`. So in the long tail, the wall is the *single hardest board* grinding to
`max_nodes` while the lanes that drew easy boards have retired. That's the call-cost
law restated: 1000× boards for ~3× wall ⇒ there's spare lane capacity during the tail.

**`work_steal=True` (Option A) — a persistent dispatch.** Launch `resident` lanes
(not `B`) and have each pull the next board index from a **shared atomic cursor**,
looping until the pool drains. A lane that finishes an easy board immediately grabs
another instead of idling. The per-board search is **byte-identical** to the base
kernel — only the gid source (cursor vs thread id) and the output store change. The
verdict is therefore bit-identical (regression-tested); `work_steal` is a *dispatch*
optimization, not a search change.

Mechanics that made it clean (all confirmed on MLX 0.31.2):
- `mx.fast.metal_kernel(..., atomic_outputs=True)` emits every output as
  `device atomic<T>*`; the cursor uses `atomic_fetch_add_explicit(&cursor[0], 1u,
  memory_order_relaxed)`. Metal has **no `atomic<uchar>`**, so `win`/`hit` are widened
  to `uint32` and `move` stays `int32`; the Python wrapper narrows them back, so the
  returned tuple is dtype-identical to the base call.
- The cursor is zeroed across all threadgroups by passing **`init_value=0`** on the
  call (MLX zeroes the output buffer before the kernel runs) — no `threadgroup_barrier`
  dance, so `resident` can span many threadgroups and saturate the GPU.
- Fail-loud invariant: after the run, `cursor == B + resident` (B successful fetches
  + one terminal overshoot per lane). A mismatch means the cursor wasn't zeroed (would
  silently skip or over-run boards) and asserts rather than returns wrong data.
- v1 is **base verdict only** — combining `work_steal` with
  `return_support`/`complete`/`return_carriers`/`return_w`/`max_depth` raises
  (the optional-output accumulators aren't wired through the loop yet).

**Where it pays off — and where it doesn't.** The GPU hardware *already* backfills at
*threadgroup-dispatch* granularity (a retired threadgroup's slot gets a pending one),
which is *why* the call-cost law is flat. So for a single in-memory pool the marginal win
over the base kernel is **none — measured 0.93–0.97× at best** (see the runtime table
below); the cursor is pure overhead when all boards already fit one dispatch.
`work_steal`'s *only* structural justification is a board pool **larger than one dispatch
can hold or arriving incrementally** (so you cannot gather it up front): one persistent
dispatch streams **millions** of boards at full occupancy with a single tail at the very
end, instead of N sequential launches each paying a terminal tail + launch overhead. The
intra-simdgroup divergence tax remains (a hard lane stalls its ~31 lock-step mates —
measured at ~6% of lanes for a few hard boards in the spike); the cursor fixes
inter-threadgroup idleness, not divergence.

**`solve_vct_streaming` (Option B) — iterative deepening over the pool.** Round 0 solves
**every** board at `budgets[0]`; the subset still `hit_cap` is re-solved at `budgets[1]`,
and so on. A board's verdict **latches** the first round it returns clean — sound because
a non-capped verdict is **budget-independent** (a deeper search never flips a clean
win/no-win; same property `solve_md_min` relies on). Only boards still capped at
`budgets[-1]` stay `hit=True`. This attacks the hard tail directly — most "hard" boards
resolve at a slightly higher budget, so the expensive deep budgets run on only the
shrinking survivor set, not the whole pool.

### Measured runtime properties (#94) — the benchmark that mapped the niche

84k real idx-2 frontier boards (replayed from `run-a` move-sequences, **no Rapfi**;
24% capped@250), solver-only, `scripts/vct_metal/bench_throughput.py`:

| regime | result | takeaway |
|---|---|---|
| **work_steal vs base, single pool** (budget 250/500/1000) | best **0.93 / 0.95 / 0.97×** (resident=16384); 0.34× @ resident=4096 | work_steal **never beats** base on a single in-memory pool — the hardware already backfills one dispatch, the cursor is overhead. Too-small `resident` underutilizes badly. |
| **base chunked@16384** (the frontier's old wave pattern) | **0.66–0.69×** | a dispatch tail **per chunk**; the real lever is *one big dispatch*, not many small ones. |
| **deepening on BASE** `(250,1000,4000)` vs single `@4000` | **1.60×**, identical verdicts (64539/64539) | ✅ **the niche.** The deep budget only touches the survivor tail. |
| deepening ladder `(250,500,1000,2000,4000)` | 1.10× | coarse ladder wins — each round **re-solves survivors from scratch**, so extra rungs pay that redundancy. |
| deepening on **work_steal** `(250,1000,4000)` | **0.87× (a LOSS)** | work_steal's big-round-0 handicap eats the deepening win ⇒ **`solve_vct_streaming` deepens on the base kernel by default**; `work_steal=True` is an opt-in for pools too large to gather into one dispatch. |

> **[2026-06-29 correction — #95]** The **1.60×** above is **conditional on N**, not general:
> it was measured at N=83814. The N-sweep finds deepen-vs-base@4000 = **0.69× / 0.79× / 1.19×
> / 1.63×** at N = 10k / 20k / 40k / 84k (crossover ≈30–35k boards). Deepening wins only when
> the hard-survivor batch is dense enough to saturate the GPU. For pools below ~30k boards,
> a single base@ceiling dispatch is faster. See § Throughput characterization sweep.

Net map of the mega-VCT solver's runtime: **for an in-memory pool, one big base
dispatch is optimal; deepening with a coarse ladder buys ~1.6× at high effective budget;
work_steal is a no-op-to-slight-loss whose only justification is genuinely-streamed pools
you cannot gather up front.** The earlier `~1,750 nodes/s` frontier figure was
*whole-pipeline* (Rapfi + solve + bookkeeping); the solver **alone** does ~6,900 boards/s
@ budget 250 — ~3.9× the frontier rate, i.e. the frontier was **Rapfi-bound, not
solver-bound** (so work_steal could never have sped it up — a prediction confirmed).

**Validated:** `work_steal` verdict is byte-identical to base across seeds × budgets ×
`resident ∈ {256,1024,8192}` (i.e. resident `<`, `≈`, `≫` B) and `B ∈ {1,31,33}` (sub-
threadgroup); `solve_vct_streaming` agrees with a single deepest-budget base call on
every mutually-clean board and resolves ≥ as many. See
`scripts/vct_metal/test_mega_vct_bb.py` (`test_work_steal_*`, `test_streaming_*`).

### Throughput characterization sweep (#95) — IN PROGRESS (overnight run)

Goal: map board-evals/min across the **whole mixed population**, which decomposes into
differently-shaped subproblems (easies / long-tails / deeps) we can't know a-priori in
general — but *can* know for already-computed boards via the append-only logs.
`scripts/vct_metal/sweep_throughput.py` (append-only, resumable) sweeps **4 pools**
(quiet-heavy frontier, capped-only, random, deep) × **strategies** (base @ 11 budgets,
chunked, work_steal × 3 residents, deepening over 4 ladders) and builds a **per-pool
resolution profile** (each board's min-resolving budget → hardness histogram → an oracle
bound). `analyze_sweep.py` turns the logs into the map.

**Two metrics, and the distinction matters:** *evals/min* = boards **screened**/min (high
even at budget 10), vs *resolved/min* = boards given a **verdict**/min. Screening fast
while leaving caps is not progress — the decision metric is resolved/min and the
wall-to-resolve-to-ceiling.

**Hardness profiles (done, N=30000/pool) — resolution is BIMODAL, and the hard tail is
near-bottomless.** Cumulative % resolved by budget:

| pool | @10 | @250 | @1000 | @20000 | still capped@20k |
|---|---|---|---|---|---|
| quiet (frontier) | 71% | 76% | 77% | 78% | **22%** |
| capped-only | 0% | 0% | 3% | **6%** | **93.7%** |
| random | 72% | 82% | 84% | 85% | 15% |
| deep (high ply) | 83% | 92% | 94% | 95% | 5% |

Two facts jump out:
1. **Budget beyond ~250 buys almost nothing.** 71–83% resolve by budget *10*; then a steep
   plateau (quiet 76%→78% from 250→20000). The population is bimodal — "easy" (≤250 nodes)
   vs a hard tail that 80× the budget barely touches. There is very little *in between*.
2. **The capped regime is near-bottomless: 93.7% of capped boards stay capped even at 20,000
   nodes** (80× the frontier's 250). So escalating `max_nodes` to "fill in the caps" is
   largely futile — these need ≫20k nodes **or are unprovable within `MAXD=32` frames**.
   That distinction is exactly what the **MAXD 32→64 study** resolves: if more caps fall at
   64 frames, the cap was *frame-depth-bound*, not node-bound. (Consistent with the earlier
   cap-probe: <1% flipped to a *win* at 16× budget; here 6% reach *any* verdict at 80×.)

Peak solver node-throughput shows on the all-hard pool (capped ~0.9M nodes/s — every lane
stays deep; mixed pools lower). A full fine ascending ladder pays ~1.8× the node-work of
oracle routing (re-solve tax) — which is *why* coarse ladders beat fine.

**Throughput map (partial: quiet/capped/random done, N=20000/config):**
- **Screening rate is enormous at low budget** — quiet base@10 = **7.9M ev/min** resolving
  71%, vs base@250 = 358k ev/min resolving 76%. Because hardness is bimodal, pushing the
  budget up resolves *few* more boards at cratering throughput. For max *resolved/min*, screen
  low and accept the hard tail as deferred.
- **work_steal loses on EVERY pool** (capped 0.72×, quiet 0.79× vs base@250) — the no-op
  generalizes across board shapes, as predicted.
- **chunked@16384 loses on every pool** (0.63–0.66×) — the per-chunk-tail penalty is robust.

**Deepening's win is CONDITIONAL ON SCALE (resolves the #94 contradiction).** At N=20000
deepening *loses* to base@ceiling on every pool (quiet coarse 0.80×, full_low 0.52×; capped
coarse 0.78×), yet #94 measured **1.60×** at N=83814. The N-sweep (deepen `(250,1000,4000)`
vs base@4000 on #94's *exact* pool, `n_sweep.py`) shows the speedup is **monotonic in N**:

| N | 10000 | 20000 | 40000 | 83814 |
|---|---|---|---|---|
| deepen vs base@4000 | 0.69× | 0.79× | **1.19×** | **1.63×** |

Crossover ≈ **30–35k boards**; at 83814 it reproduces #94's 1.60× exactly. Mechanism:
deepening's value is keeping the *deep* rounds **dense with hard boards** — the all-hard
capped pool saturates the GPU at **~0.9M nodes/s** vs ~0.35–0.49M mixed, so a deep round on a
dense survivor set is ~2× faster per node, beating the re-solve tax *only* when there are
enough hard survivors to saturate. Below ~30k the survivor batch is too sparse and the tax
dominates. **So #94's 1.60× is real but conditional on large N / high hard-density — not a
general win** (dated-corrected in the #94 table below).

---

## Invariants (the regression contract)

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

## Gotchas

- **Plane convention is side-to-move-relative, `board[0]`=attacker, no swap** — the
  #1 silent-wrong-answer trap.
- **Free-style** (overlines win); VCT-win monotonicity holds under free-style.
- **256-bit board** (`ulong[4]`, N²≤256). `uint[8]` was tried and reverted (~20%
  slower in situ — the kernel is bookkeeping-heavy; see §8 "word width").
- **Tail-bound, bulk-synchronous only** — see the call-cost law. Keep callers
  bulk-synchronous; never solve-in-a-loop.
- **`hit_cap` is not a no-win** — it's "couldn't prove within budget". Treat
  conservatively.
- Adding the `support`/`winmask` outputs costs nodes/compute, not the fast path
  (separate compiled variants; default is byte-identical).

---

## CPU solver retired (2026-06-27) — gate, env override, fast/deep test tiers

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

## Consumers (who calls this)

`scripts/threat_shapes/`: `mine_vct_gpu_flat.py`, `mine_first_vct.py`,
`solve_puzzles.py`, `mine_puzzles.py`, `harvest_molecules.py`, `vct_fan.py`,
`probe_timing.py`, `certificate_falsification.py` (the `carriers` certificate),
`w_channel_probe.py` (the `w` identity-not-existence probe, #90),
`md_minimize.py` (the `max_depth`/`solve_md_min` md-invariant stencil minimizer, #91).
Feeds [shape-library-engine.md](shape-library-engine.md) (L1
stencils), [vct-backward-mining.md](vct-backward-mining.md),
[vct-reachability-mining.md](vct-reachability-mining.md), and
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md).

**Cross-links:** [gpu-vct-feasibility.md](gpu-vct-feasibility.md) (build narrative
+ throughput) · [vct-backward-mining.md](vct-backward-mining.md) §5 (the move
output) · [allis-threat-theory.md](allis-threat-theory.md) (VCF/VCT formalism) ·
`gomoku/vcf.py` (the CPU oracle `solve_vct`).
