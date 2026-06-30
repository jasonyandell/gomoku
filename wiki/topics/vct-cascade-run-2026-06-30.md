# VCT cascade — at-scale run record (2026-06-30, full rapfi corpus) (#97)

Live, append-only record of the **actual full-corpus labeling run** — what each
ladder rung does *at scale* on the real 56.1M-position rapfi corpus: where width
settles, sustained boards/s, wall time, and the verdict split (how many resolve
vs. how many cap and fall through to the next rung). This is the deepening curve
the throughput-knee sweep ([[vct-cascade-labeler]]) could only sample.

- **Corpus:** `~/data/raphi_vct/positions/` — 56,121,658 unique D4-canonical positions.
- **Run root:** `~/data/raphi_vct/` (`results/cap<N>/`, `survivors/cap<N>/`, `perf/`).
- **Launched:** 2026-06-30 06:58 (after the 06:53 reboot cleared the Metal wedge).
- **Execution:** `watchdog.sh` → `cascade.py`, all proof-outputs ON
  (move+support+carriers+w), `complete=OFF`, work-bounded (no `timeout` kills).
- **Ladder:** planned `50…100000`; **STOPPED AT CAP2000** (Jason's call, 2026-06-30
  14:12) once the deep-ladder projection showed cap4000→100k as a ~10-day grind.

## ✅ Closing summary — run complete through cap2000 (2026-06-30)
The ≤2000 ladder finished cleanly (`[cascade] complete`, no wedge) in **~7.2 h
wall** (06:58→14:12). Final ledger over the whole 56,121,658-position corpus:

| outcome | positions | % of corpus |
|---|---:|---:|
| **win** (side-to-move has a VCT) | 27,428,327 | **48.9%** |
| **no_win** (proven none ≤ its budget) | 23,178,975 | **41.3%** |
| **resolved subtotal** | **50,607,302** | **90.2%** |
| **deep tail** (still cap at 2000 nodes) | 5,514,356 | **9.8%** |

Ledger is exact and complete: 27,428,327 + 23,178,975 + 5,514,356 = 56,121,658 ✓
(every position has a row; nothing is "absence = state").

**The deep tail — `survivors/cap2000/` (5,514,356 positions)** — is the preserved,
content-addressed seed for a future targeted deep run (caps 4000→100000, or a
sampled probe of the depth limits). It was deferred, not abandoned: the cascade is
resume-by-row-offset, so a later run just adds `4000,10000,…` to the ladder and
picks up exactly here. Projection said the full deep ladder ≈ **~10 days** of GPU
on these ~5.5M stubborn boards (each rung resolves only ~3–4%); that's a separate
deliberate run, not part of this one.

**Net:** 90.2% of the rapfi corpus now carries an exact VCT verdict + proof outputs
(move/support/carriers/w), and the hard 9.8% is cleanly isolated for later.

## 🎯 The first VCT arrives at median ply 19 — games are decided FAR earlier than they end
Joining every game's per-ply positions (`positions_raw`: shard/game_idx/ply → id)
back to the win-ledger, for the **first ply at which the side-to-move has a VCT**:

- **Mean 21.6 · median 19** plies to first VCT (min 6, max 119; p10–p90 = **12–31**).
- **96.4%** of the 1,194,662 games contain a VCT at all (3.6% never do = draws /
  cut-short). Distribution: 2.0% by ply 9, **49.6% in plies 10–19**, 36.5% in 20–29,
  11.9% at 30+.
- **First-move advantage is visible:** of games with a VCT, **68%** get their first
  VCT on a *first-player*-to-move position vs 32% second-player (both at ~ply 21.5).
  Consistent with 15×15 free-style being a first-player win (Allis).

**This explains the "shocking" 48.9% win rate.** A VCT appears around move ~20 (≈10
stones/side) and rapfi games run ~40–60 plies, so the **entire back half of nearly
every game is VCT-saturated** — once the board fills past ~20 stones, a forced
threat sequence almost always exists for *someone*. The win rate isn't a sampling
artifact; it's the natural density of forced wins in mid/late tactical positions.

### ⭐ Headline implication — play self-play games TO the first VCT, not to five
A VCT is a *proven forced win* the GPU oracle can both **detect and terminally
value** (exact win + the winning move via `return_move`). So self-play does not need
to play the game out to an actual five-in-a-row: **terminate at the first VCT and
take the oracle's verdict as the game result.** That cuts the searched/played
trajectory from ~40–60 plies to a **median of 19** — *less than half* the plies to a
**labeled** winner, with an **exact** terminal value instead of a bootstrap. Far
fewer MCTS-expanded plies per game → cheaper, higher-quality self-play data. This is
the actionable lever from this run; seeded as [idea-pile.md](idea-pile.md) #11.

## ⭐⭐ 50 nodes is a near-complete first-VCT detector — the wall sits BEHIND cap50
Tagging each VCT position with the budget that resolved it (the rung it won in;
a win leaves the survivor stream so its budget is unique) and joining to game order
answers "how cheap is the first forced win, and what does deeper budget buy?"

- **98.8% of all 27.4M VCT positions resolve at cap50** (27,105,046); only 1.2% need
  more. The cascade's deeper rungs barely add wins — they mostly chase no-wins.
- **A 50-node seeker finds a forced win in 96.1% of ALL games** (99.7% of the
  1.15M games that have any VCT). Only **0.3%** of vct-games (3,987) have a VCT but
  none catchable at 50 nodes.
- **Deeper budget buys almost no earliness.** First VCT a seeker can reach, by ceiling:

  | budget | % games w/ a ≤B VCT | median first-VCT ply | mean |
  |---:|---:|---:|---:|
  | **cap50** | **99.7%** | **20** | 22.2 |
  | cap100 | 99.8% | 20 | 22.0 |
  | cap250 | 99.9% | 19 | 21.8 |
  | cap1000 | 100.0% | 19 | 21.6 |
  | cap2000 | 100.0% | 19 | 21.6 |

  50 → 2000 nodes (**40×** compute) moves the first forced win **~0.6 ply earlier
  (mean), ~1 ply (median)**. Effectively nothing.
- Even the game's **true-earliest** VCT is usually cheap: **86.7%** of games have
  their absolute-first forced win visible at cap50 (cumulative 90.8% by cap100,
  94.7% by cap250, 100% by cap2000).

**Design implication (settles the direction):** build the AI around **cap50 VCT
seeking**. It's a practically-complete mate detector (96% of games, ~as early as an
infinitely-patient solver, within ~1 ply) at **40–850× less compute per position**
than the deep rungs. The deep budgets matter only for the stubborn ~1.2% — and
those are dominated by hard-to-confirm **no-wins**, not earlier wins (the deep tail
is a wall whose far side is mostly empty of cheap gold). Shallow-fast beats
deep-slow for a mate-seeking player: brutal iteration time, gives up only 0.3% of
decidable games and ~1 ply of earliness. Feeds [idea-pile.md](idea-pile.md) #11 —
the seek-VCT target is **cheap**, so VCT-terminated self-play is cheap to run.

## First VCT per game — resolution by budget (first VCTs are DEEP-ENRICHED)
The "50 nodes sees everything" claim above is about the *total* VCT population. Look
only at each game's **first** VCT (the earliest-ply forced win) and the picture
sharpens — the first forced win is by definition the hardest to reach, so it is far
more likely to be a deep one. Tagging each game's first VCT with the budget that
resolved it (`scripts/vct_cascade/first_vct_resolution.py`), over the 1,151,498
games with a confirmed VCT:

| budget that resolved the FIRST VCT | games | % | cumulative |
|---|---:|---:|---:|
| **cap50** (shallowest) | 997,874 | **86.7%** | 86.7% |
| cap100 | 47,374 | 4.1% | 90.8% |
| cap250 | 45,587 | 4.0% | 94.7% |
| cap500 | 24,310 | 2.1% | 96.8% |
| cap1000 | 19,490 | 1.7% | 98.5% |
| cap2000 | 16,863 | 1.5% | 100% |

- **13.3% of games (153,624) had their first VCT go cap→win only by looking deeper
  than 50 nodes** — vs only **1.2%** of the total VCT population. **First VCTs are
  ~11× deep-enriched** relative to a random VCT. The population's 98.8%-cap50 number
  is diluted by the abundant easy *late*-game VCTs; the first forced win lives in the
  contested early zone where the hard positions cluster.
- **The deeper-needed tail does NOT die out at cap2000.** Of the 13.3%: cap100 30.8%
  → cap250 29.7% → cap500 15.8% → cap1000 12.7% → **cap2000 still 11.0%**. Heavy,
  unterminated tail ⇒ pushing past 2000 nodes would keep converting more first VCTs.
  For first VCTs specifically the wall is **softer** than the population implied.
- **Censoring caveat (the honest asterisk):** **91.4% of vct-games have an
  *unresolved* deep-tail position at a ply *earlier* than their first confirmed VCT.**
  So every first-VCT ply here is an **upper bound** — the true first VCT could be
  earlier *and* deeper, hiding in the cap2000 tail. (Most early caps are likely
  no-wins — then the bound is tight — but we can't know without resolving them; this
  is the game-side view of the "no earlier VCT" deep-tail subset.) Also 39,525 games
  have *no* confirmed VCT but *do* have a deep-tail position — a first VCT for them,
  if any, lives entirely past cap2000.
- **What it means for the shallow seeker:** doesn't overturn "build shallow" — a
  cap50 seeker still finds mate in 96% of games — but it sharpens the trade: a
  *deeper*-searching opponent could beat the cap50 seeker **to the punch** (find the
  mate several plies earlier) in ~13%+ of games. Fine for a concept-prover; a known
  lever if/when we want to strengthen it.

## Sustained throughput vs budget — the cap50 tradeoff point (confirmed)
"Sustained" = median boards/s across steady-state dispatches (NOT the peak knee).
As actually run by the cascade:

| step | sustained b/s | ≈ /min | width | ran on |
|---|---:|---:|---:|---|
| **cap50** | **42,496** | ~2.55M | 524k | **natural full corpus** ✅ |
| cap100 | 9,176 | ~0.55M | 262k | hard survivors ⚠️ |
| cap250 | 3,792 | ~0.23M | 262k | hard survivors ⚠️ |

- **Only cap50 is a clean natural-mix rate** (it ran on the whole corpus). cap100/250
  ran only on the hard *survivors*, so those are worst-case rates, not what 100/250
  would do on a normal position stream (the single-shot sweep on a natural mix hit
  ~23.8k / ~10.1k at the knee — ~2.6× the survivor rates).
- **Sanity check (Jason 2026-06-30):** the survivor rungs track ~`rate ∝ 1/budget`
  (cap250→cap100: budget ×0.4, rate ×2.42). Extrapolating that to cap50 (×2) predicts
  **~18–20k b/s** — the "if every board were tail-hard" floor. The *measured*
  natural-mix cap50 is **42,496** — ~2× that floor, because real game positions are
  mostly easy. (Strict-linear extrapolation gives ~11k but is the wrong model; the
  inverse-budget fit is much better.)
- **Verdict — cap50 is the tradeoff sweet spot:** ~18–20k b/s pessimistic floor,
  ~42.5k realistic, while keeping 86.7% of first VCTs and a forced win in 96% of
  games. The VCT terminal test is **not** the bottleneck (tens of thousands/s);
  MCTS tree search is. Operating point confirmed with ~2× headroom over the floor.

## Per-rung results at scale (the deepening curve)
Each rung runs only on the prior rung's `cap` survivors. `cap` count = input to the
next rung. Throughput = cascade-only perf rows (last night's sweep filtered out by ts).

| rung | input boards | steady width | peak b/s | median b/s | GPU wall | win | no_win | cap (→next) | win% | cap% |
|-----:|-------------:|-------------:|---------:|-----------:|---------:|----:|-------:|------------:|-----:|-----:|
| cap50 | 56,121,658 | 524,288 | 43,089 | 42,496 | 22.5 min | 26,837,059 | 21,605,369 | 7,130,052 | 48.3 | 12.8 |
| cap100 | 7,200,627 | 262,144 | 9,257 | 9,176 | 13.2 min | 108,070 | 450,445 | 6,642,112 | 1.5 | 92.2 |
| cap250 | 6,642,112 | 262,144 | 3,867 | 3,792 | 29.4 min | 96,424 | 364,780 | 6,180,908 | 1.5 | 93.1 |
| cap500 | 6,180,908 | 131,072 | 1,926 | 1,889 | 55.0 min | 49,770 | 205,883 | 5,925,255 | 0.8 | 95.9 |
| cap1000 | 5,925,255 | 65,536 | 922 | 908 | 109.6 min | 38,273 | 192,888 | 5,694,094 | 0.6 | 96.1 |
| cap2000 | 5,694,094 | 131,072 | 500 | 493 | 195.8 min | 30,744 | 148,994 | 5,514,356 | 0.5 | 96.8 |

*(ladder stopped at cap2000 per Jason's decision 2026-06-30 — see Closing summary.
 cap2000 wall includes a clean kill+resume at 92% when the ladder was truncated.)*

> **cap50→cap100 survivor count grew 7,130,052 → 7,200,627**: cap50 was still
> flushing its last shards when first sampled; 7.20M is the true cap50 survivor set.

## Width-ramp curve at scale (cap50 — where throughput saturates)
The cascade auto-doubles batch width from 2,048 until throughput stops climbing,
then holds. cap50 saturated at **W=524,288** (going wider bought <5%):

| width | boards/s |
|------:|---------:|
| 2,048 | 3,306 |
| 4,096 | 7,142 |
| 8,192 | 13,061 |
| 16,384 | 21,370 |
| 32,768 | 27,566 |
| 65,536 | 35,411 |
| 131,072 | 38,953 |
| 262,144 | 41,403 |
| **524,288** | **43,089** ← knee |

**Matches the standalone sweep's cap50 knee (43,397 b/s).** The cascade reproduces
the measured throughput law in production: width is king until GPU saturation, then
flat. Note it settled at 524k, not the sweep's nominal 2M plateau — the extra width
gains nothing at cap50, so the ramp correctly stopped early.

## ⚠️ Deep-ladder time projection — the full run is ~DAYS, not overnight
The tail barely shrinks (each rung resolves only ~4% of its survivors) while
throughput ~halves per ladder step. So the deep rungs run on ~5M+ boards at
collapsing speed. Measured + extrapolated wall (cap2000 measured at 496 b/s):

| rung | survivors in | est. b/s | est. wall |
|-----:|-------------:|---------:|----------:|
| cap2000 | 5,694,094 | 496 (measured) | ~3.2 h |
| cap4000 | ~5.47M | ~250 | ~6 h |
| cap10000 | ~5.3M | ~110 | ~13 h |
| cap20000 | ~5.1M | ~55 | ~26 h |
| cap50000 | ~5.0M | ~22 | ~2.6 d |
| cap100000 | ~4.9M | ~11 | ~5 d |

**Full ladder to 100k ≈ ~10 days of continuous GPU.** The cheap rungs (≤1000)
finished in ~3.5 h and resolved 99.6% of the corpus; the remaining ~0.4% (~5.7M
positions, but really one stubborn hard-tail class) is what costs the days. This is
the deliberate "grind out the deep gold + record depth limits" run — fully
resumable, so it can run as long as desired — but the cost shape means **a decision
point:** let it run for days, cap the ladder (e.g. stop at 10k, make 100k a
separate targeted run), or sample the deep tail rather than solving all ~5M.
(Flagged to Jason 2026-06-30 11:10; no change made without him.)

## Notes / anomalies
- cap50 resolved **87.2%** of the whole corpus definitively (48.3% win + 38.9%
  no_win) at just 50 nodes — confirming most rapfi positions are tactically
  shallow. The interesting tail is the **12.8% (7.20M)** that cap and fall through.
- **THE KEY AT-SCALE FINDING — survivor-rung throughput collapses far below the
  single-shot knee.** The standalone sweep measured the knee on the *natural*
  rapfi mix; each cascade rung after cap50 runs only on the *hard survivors* (boards
  that already capped at the prior budget), which run the **full** node budget with
  no early-out. So measured b/s per rung << the sweep knee:

  | rung | survivor-rung b/s (this run) | single-shot knee (natural mix) | ratio |
  |-----:|-----------------------------:|-------------------------------:|------:|
  | cap50 | 43,089 | 43,397 | 0.99× (cap50 IS the natural mix) |
  | cap100 | 9,257 | 23,822 | **0.39×** |
  | cap250 | 3,867 | 10,107 | **0.38×** |
  | cap500 | 1,926 | 3,134 | **0.61×** |
  | cap1000 | 922 | 1,290 | **0.71×** |
  | cap2000 | 500 | — (no knee measured) | — |

  Plan deep-rung wall-clock off the *survivor* rate, but the ratio is **not
  constant — it climbs with budget** (0.38× → 0.61× → ~0.71×). Likely because the
  single-shot knee at a high budget is *itself* increasingly dominated by hard
  boards (easy ones resolve fast at any budget), so the survivor set looks more and
  more like the natural mix the knee was measured on — the two rates converge as the
  budget rises. Low-budget rungs are where survivor-vs-natural diverges most.
- **The deeper you go, the less budget buys.** cap100 (2× the nodes) converted only
  **7.8%** of cap50's survivors to a definitive verdict (1.5% win + 6.3% no_win);
  the other 92.2% still cap. The tail is hard, not slow — the deep-win gold is rare
  and lives only at the high-budget rungs, exactly as the cascade was built to find.

**Cross-links:** [vct-cascade-labeler.md](vct-cascade-labeler.md) (architecture +
knee sweep) · [mega-vct-solver.md](mega-vct-solver.md) ·
[gpu-vct-feasibility.md](gpu-vct-feasibility.md).
