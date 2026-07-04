# VCT cascade labeler — corpus-scale verdict ledger + throughput knees (#97)

> **Status: LIVE** *(2026-07-04)* — architecture + throughput-knee reference (the run itself concluded).

A corpus-agnostic, resumable pipeline that labels **every position** of the rapfi
game corpus with an exact GPU-VCT verdict, deepening the hard tail through a
node-budget ladder. Built to push `solve_vct_mega_bb` to its sustained-throughput
limit on the M5 Max and to record the deepening curve ("X% solved by N nodes, Z%
still cap at the ceiling"). Code: `scripts/vct_cascade/` (branch
`feat/vct-cascade-labeler`). Sister of the in-RAM `solve_vct_streaming` — same
iterative-deepening algorithm, but survivors spill to Parquet so it resumes and
scales to billions.

## Architecture (the reducer-over-a-log shape)
- **`positions/`** — immutable, content-addressed by **D4-canonical blake2b-16**
  (`extract.py`; reuses `gomoku.game._sym_board`, color-fixed since board[0] is
  always the attacker). Identical-under-symmetry positions collapse to one work
  item. Built once, then corpus-agnostic input to everything downstream.
- **Cascade** (`cascade.py`) — ladder `[50,100,250,500,1000,2000,4000,10000]`.
  Each rung solves the prior rung's still-capped survivors at one `max_nodes`,
  writes a FULL per-board verdict ledger row (`win`/`no_win`/`cap` + budget) to
  `results/cap<N>/`, and spills the capped tail to `survivors/cap<N>/` as the next
  rung's input. **Every result recorded explicitly — no "absence = state."**
  Resume = ledger row-count offset (no global in-RAM set). Crash-loss ≤ one
  dispatch (results written last, atomic temp+rename).
- **Outputs ON:** `move + support + carriers + w` (proof extraction, no extra
  search). `complete=OFF` (greedy first proof — the fast path).
- **`perf/`** — one row per dispatch (cap, width, boards/s, verdict counts) →
  knees are a DuckDB query (`stats.py`).
- **Format = what you'd pull off HF:** sharded zstd-Parquet, glob-queryable by
  DuckDB/Polars/`datasets`.

## Corpus stats (2026-06-30, `~/data/games_raphi` → `~/data/raphi_vct`)
- 11,955 game shards · **62,739,675 plies** → **56,121,658 unique D4-canonical
  positions** (89.5% unique — most mid/late 15×15 boards are genuinely distinct;
  symmetry+cross-game collapse only buys ~10%).

## THROUGHPUT KNEES — the gold (measured, all proof-outputs ON, 15×15)
Single-shot `solve_vct_mega_bb` over the real rapfi position distribution. Best
sustained boards/s and the width at which it peaks:

| node cap | best boards/s | knee width | notes |
|---------:|--------------:|-----------:|-------|
|   50 | **43,397** | 2,097,152 (2M) | plateau ~43k from ~2M up to 4M |
|  100 | **23,822** | 2,097,152 (2M) | |
|  250 | **10,107** |   524,288 (512k) | |
|  500 |  **3,134** |    65,536 (64k) | |
| 1000 |  **1,290** |    65,536 (64k) | |
| ≥2000 | not captured | — | >150s per dispatch even at small width; needs a longer per-dispatch window |

**Shape (the law):** width is king — throughput climbs with batch width until the
GPU saturates (~2M wide at low budget), then flat. Throughput **roughly halves per
ladder step** as the budget rises. The **knee width drops as budget rises** (2M @
cap50 → 64k @ cap1000): hard boards saturate the GPU at far smaller batches, so a
fixed width is wrong — tune width per depth. cap50 ≈ **2.6M boards/min**, so the
56.1M-position first rung is ~22 min; deep rungs run only on the shrinking
survivor set.

## Verdict mix (single-shot at budget B, natural rapfi distribution)
Front-of-pool sample (NOT the cascade deepening curve — that needs the full run):

| cap | win% | cap% | no_win% |
|----:|-----:|-----:|--------:|
|   50 | 48.0 | 12.9 | 39.2 |
|  100 | 48.1 | 11.9 | 40.0 |
|  250 | 47.8 | 11.2 | 41.1 |
|  500 | 46.2 | 11.1 | 42.7 |
| 1000 | 46.3 | 10.7 | 43.0 |

Note the **cap% barely falls (12.9→10.7) as budget goes 50→1000** — the capped tail
is hard, not slow: extra budget converts very few caps to definitive. Consistent
with the "near-bottomless tail" prior (gpu-vct-feasibility.md / #95). The deep-win
gold therefore lives in the high-budget rungs run on that stubborn ~11% tail — the
whole point of the cascade. (~48% of all plies having a side-to-move VCT is high
but plausible for tactical rapfi self-play, which is dense with near-terminal
positions; revisit against the cascade's exact curve.)

## Post-mortem: the throughput sweep crashed WindowServer (2026-06-30 ~01:10)
Running the deep-cap throughput sweep wrapped in `timeout` (which SIGKILLs the MLX
process **mid-Metal-compile**) wedged the Metal compiler service system-wide
(`MTLCompilerService ... Reentrancy avoided`) — and we believe it also **took down
WindowServer** (the macOS graphics server), i.e. crashed the whole GUI session.

Evidence (process start times, `ps -o lstart`):
- launchd / kernel (pid 1): original boot **Jun 24 20:21** — NO kernel reboot
  (uptime stayed 5 days).
- **WindowServer + loginwindow: relaunched 01:10** — ≈ exactly when the sweep
  wedged Metal (`sweep_all` finished 01:11:49).
- Symptoms reported: all apps closed, browser gone, next login required a
  **password not Touch ID** (the signature of a fresh loginwindow / GUI teardown).

So a tail-bounded GPU job killed mid-compile didn't just fail itself — it appears
to have crashed the desktop, and because the kernel never rebooted the wedge
**persisted** (8h+ idle and the WindowServer restart both failed to clear it; only
a real reboot does). Operational takeaways, now load-bearing for this lab:
1. **Bound GPU runs by WORK (board count / `max_nodes`), never by an external
   `timeout` SIGKILL.** The cascade already does this (compiles its kernel once,
   then runs `max_nodes`-bounded dispatches), so it is the safe execution path.
2. The `sweep.py` deep-cap probe must not be `timeout`-wrapped — give it a longer
   in-process per-dispatch budget instead, or skip caps ≥2000 in the wide sweep.
3. Distinguish a GUI crash from a reboot via `ps -o lstart= -p 1` (kernel) vs
   WindowServer's start time. (Full machine-fact note lives in session memory:
   `this-machine-metal-compiler-wedge`.)

## Status / how to resume
Extract+dedup DONE; throughput characterized; **labeling cascade RAN 2026-06-30
(06:58→14:12), stopped at cap2000 by decision** — 90.2% of the corpus resolved
(48.9% win / 41.3% no_win), 9.8% (5.51M) deep tail preserved in
`survivors/cap2000/` for a future targeted deep run. Full per-rung record +
closing ledger: [vct-cascade-run-2026-06-30.md](vct-cascade-run-2026-06-30.md).
(The night-of 2026-06-29→30 Metal wedge that first blocked the GPU was cleared by
a reboot, see [[this-machine-metal-compiler-wedge]] in memory.) To run the
deferred deep tail (caps 4000→100000 on `survivors/cap2000/`), or re-run from
scratch:
```bash
cd ~/code/gomoku-vct-cascade-labeler
GOMOKU_BOARD_SIZE=15 bash scripts/vct_cascade/watchdog.sh ~/data/raphi_vct
GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_cascade.stats --out ~/data/raphi_vct
```

**Cross-links:** [mega-vct-solver.md](mega-vct-solver.md) (the solver API + flags) ·
[gpu-vct-feasibility.md](gpu-vct-feasibility.md) (call-cost law, tail hardness) ·
[m5-max-fp16-and-throughput-regimes.md](m5-max-fp16-and-throughput-regimes.md).
