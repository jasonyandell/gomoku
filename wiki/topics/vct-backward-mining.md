# VCT-backward enabling-shape mining — walk won games back to the "first true VCT move"

**One-line idea.** Generate strong games fast (Rapfi-vs-Rapfi), then walk each won
game **BACKWARD** to its **enabling shape** — the *earliest* winner-to-move position
from which a forced VCT win already exists, i.e. the **first true VCT move**. We mine
the **SETUP**, not the kill: the forcing line is line-bound by construction, but the
shape that *enables* it need not be. Cheap, deliberately incomplete, a substrate for
non-line "molecule" discovery.

**Code:** `scripts/threat_shapes/mine_vct_serial.py` (CPU reference/oracle) ·
`scripts/threat_shapes/mine_vct_gpu.py` (level-synchronized GPU walk) ·
`scripts/threat_shapes/mine_vct_gpu_flat.py` (**the flat-batch winner**) ·
`scripts/threat_shapes/mine_vct_backward.py` (`mine_game`, the validated walk-back) ·
`scripts/threat_shapes/collect_rapfi.py` (stage-1 RapfiPool game generation).
**Builds on:** the on-device VCT megakernel `scripts/vct_metal/mega_vct_bb.py`
(`solve_vct_mega_bb`) — see [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8.
**Feeds:** [shape-library-engine.md](shape-library-engine.md) — these 63k enabling shapes
are the **raw material for L1**: the engine reduces each to its minimal full-board prime
implicant (the "exact minimum stones that make the VCT inevitable") and builds the library.
Also feeds [molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) /
[idea-pile.md](idea-pile.md) #10 (the molecule ⊋ line program); the setup is exactly
the *zero-line-content residual* that program hunts. **Threat theory:**
[allis-threat-theory.md](allis-threat-theory.md) (VCF OR-only vs VCT AND/OR). **Non-line
structure:** [the-claw.md](the-claw.md).

Date: 2026-06-26. Hardware: M5 Max, 48 GB, MPS/MLX-Metal. All throughput numbers from a
single overnight session; the science is the run-length distribution, not the wall clock.

---

## 1. The two-stage plan (Jason's framing)

- **Stage 1 — bank strong games fast.** Rapfi-vs-Rapfi via `RapfiPool`
  (`collect_rapfi.py`) at **~9 games/s**; ~300k games banked to `~/data/games_raphi/`.
  Strong games = dense, real forcing structure to mine.
- **Stage 2 — mine the enabling shape.** For each *won* game, anchor at the END and
  walk back over the **winner-to-move** positions while a forced VCT win still exists;
  stop at the boundary (the first NON-forced position); the **earliest still-forced**
  position is the enabling shape.

We mine the **setup**, not the forcing line. The kill is line-bound by construction (a
VCT is a chain of fours/threes); the *shape that first makes it forced* is where the
non-line structure can live — the discovery target for idea #10.

## 2. Algorithm + correctness

Anchored at game end; walk back over winner-to-move positions `p = L, L-2, L-4, …`
while `solve_vct(p)` is a proven forced win; the earliest still-forced `p` = enabling
shape. **Min-run filter ≥3** (require a chain of ≥3 forced winner-to-move positions to
emit, dropping trivial one-move mates).

**Plane convention — VERIFIED (the load-bearing correctness fact).** `GameState.board`
is **side-to-move-relative**, so at a winner-to-move position `board[0]` is *already* the
winner/attacker frame `solve_vct` wants — **NO swap**, and this holds for **black AND
white** winners alike. Getting this wrong silently solves the wrong side; it was checked
explicitly.

**End-to-end validation: 0 FP / 0 FN / 0 extra** over **258 clean GPU-vs-CPU comparisons**
(megakernel verdict vs `gomoku.vcf.solve_vct`), consistent with the megakernel's own
0-FP/0-FN gate in [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8. Validation log:
`~/.claude/jobs/9aac67d6/tmp/vct_final.txt`.

## 3. CPU serial miner — what it taught us (`mine_vct_serial.py`, depth-10 / max_nodes=100k)

The serial miner (committed `806151e`) is the correctness oracle and the source of the
counterintuitive cost findings:

- **THE COST SINK IS THE BOUNDARY NEGATIVE PROOF.** Proving "no forced win exists *yet*"
  (the boundary, first non-forced position) **exhausts the tree**; POSITIVE proofs
  (finding a win) return fast. So walking deep back is cheap — **the single boundary
  solve is the monster.** The slow games are NOT the deepest walk-backs.
- **Per-game distribution is bimodal:** most games <0.1 s; a heavy slow tail of **4–16+
  min** "monster" games (the boundary proof churning). Serial ⇒ one monster freezes the
  whole pipeline.
- **DEPTH is the exponential lever, NOT `max_nodes`.** `solve_vct` ≈ **0.65 ms/node**; the
  node cap barely moves the emit rate. Depth knee: depth-6 caps chains at run≈5; depth-10
  reaches run≈7–8; **depth-20 is WORSE** — it spreads the node budget too thin, hits the
  cap before proving deep wins, and walks back LESS (run≈6). **Shallower-with-more-
  nodes-per-proof walks back FURTHER.** depth-10 was the CPU sweet spot.
- **Yield: 184 move-labeled shapes** over ~10 hours, run-lengths 3–8. depth-10 physically
  cannot see run>~8. This is the **reference/oracle set** (`scratchpad/vct_serial/`) — the
  only one with catalyst moves attached.

## 4. GPU breakthrough — flat-batch (`mine_vct_gpu_flat.py`)

The megakernel `solve_vct_mega_bb` (MAXD=32 ⇒ depth-30; validated 0 FP/FN vs CPU) is
**TAIL-BOUND**: a batch of up to ~16k boards costs ~the same ~16 s as one — throughput
∝ batch size. It returns ONLY `(win, hit_cap)` verdicts.

**The sloppy win (Jason: "just be sloppy, solve everything from the back").** The
level-synchronized walk (`mine_vct_gpu.py`) early-stops cleverly, but serializes into
**~25 sequential GPU calls** (one per depth-step, gated by the deepest game) — a
*pessimization* given the kernel is tail-bound. Instead, **flatten EVERY winner-to-move
position of a chunk of games into ONE big GPU batch**, solve all at once, then per game
read off the **contiguous VCT-won suffix from the end** = the enabling shape. One ~16 s
call replaces the whole walk; first flush in ~20 s instead of ~8 min. The "wasted" GPU
work on obviously-lost early-game positions rides free in the tail-bound batch.

**Result: ~20 games/s (~2500× the CPU serial).** The first `--once` pass completed the
whole corpus: **200,242 shapes over 2,631 shards (263,100 games)**, run-lengths reaching
**run-17** — the long forcing chains the depth-10 CPU miner physically cannot reach.
**Canonical location: `~/data/vct_shapes/`** (`enable_serial.jsonl.gz` + `manifest.txt` +
`README.md`; durable). The `manifest.txt` lets a re-run resume incrementally without
recompute. (Working copy: `scratchpad/vct_gpu_flat/`.)

**Run-length histogram, full 200k corpus (the science — a steep ~1.6× decay per step):**

| run | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 98396 | 61203 | 25096 | 9339 | 3746 | 1517 | 579 | 217 | 84 | 44 | 13 | 5 | 2 | 1 |

## 5. The verdict-vs-move gap (OPEN — next step)

The megakernel gives only **win/no-win** — enough for the walk-back, but NOT the catalyst
move or mate distance. The GPU's 200k shapes currently carry **`move=-1`** (extraction off
for speed). Recovering the move needs a CPU `solve_vct` per shape — a *fast positive proof*
individually, but per-shape serial ⇒ the extraction bottleneck, and at the wrong budget
(depth-16 / 200k) it re-becomes the §3 CPU monster.

**Options on the table (NOT YET DECIDED):**
1. cheap **PARALLEL CPU extraction** (fan the positive proofs across cores); or
2. add a one-`uint` **GPU root-move output** to the megakernel (cheap bandwidth, kills the
   CPU step entirely).

Jason is taking node/move extraction next and has ideas; record as the open next step.

## 6. Lessons (path-dependent, worth keeping)

- **First-flush latency hid behind THREE stacked serial costs, each masking the next:**
  level-walk depth (~8 min/chunk) → MLX first-call kernel compile → and the real wall,
  per-shape CPU move-extraction. A **single direct timing probe** (reconstruct = 0.00 s/game,
  GPU solve = 12.5 s/315 pos, extract = the wall) cut through ~30 min of black-box poking.
  **MEASURE, don't guess.**
- **Don't interrupt a structurally-slow job to "retune."** Repeatedly killing it just
  resets its clock — **instrument instead.** (Mirrors the megakernel tail-bound lesson:
  the cost is one deep position; let the batch run.)
- **The tail-bound kernel inverts the usual instinct.** Clever early-stopping is a
  pessimization when the batch wall is set by the single deepest board; *sloppy and
  exhaustive* is faster. (See [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8.)

## 7. Artifacts

| Path | What |
|---|---|
| `~/data/games_raphi/` | Stage-1 banked Rapfi-vs-Rapfi games (~300k) |
| `~/data/vct_shapes/` | **CANONICAL** GPU flat-batch corpus — **200,242** shapes, unlabeled (`move=-1`) + `manifest.txt` + `README.md` |
| `scratchpad/vct_serial/` | CPU miner output — **184 move-labeled** shapes, the reference/oracle set |
| `scratchpad/vct_gpu_flat/` | GPU flat-batch working copy (source of the canonical landing) |
| `~/.claude/jobs/9aac67d6/tmp/vct_final.txt` | GPU-vs-CPU validation (0 FP / 0 FN / 0 extra, 258 clean) |

**Cross-links:** [gpu-vct-feasibility.md](gpu-vct-feasibility.md) (the megakernel this builds
on) · [allis-threat-theory.md](allis-threat-theory.md) (VCF/VCT formalism) ·
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) ·
[idea-pile.md](idea-pile.md) #10 (the molecule ⊋ line program this feeds) ·
[the-claw.md](the-claw.md) (non-line structure).
