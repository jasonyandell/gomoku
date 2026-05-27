# Research Board — the Δelo Derby

A race between 8 fresh-start self-play training recipes ("ideas") to a fixed
**140-epoch budget**, run in **10-epoch chunks**, scored by the model's
**anchored elo** (vs `random`, `heuristic`, `lookahead:depth=2`, and the slow
`lookahead:depth=4` anchor). This is a research board that produces real models:
each idea gets a production-style title card, but the question it answers is
*which training recipe climbs fastest*.

## CURRENT (2026-05-26) — Derby v8 LIVE, beads-runner operating model

**The lab runs as a single GPU executor (the derby) + code-only beads (other sessions).**
The orchestrating session IS the "derby runner": it owns the GPU, runs the derby in
300s (5-min) chunks doled by Δelo-rate (peak-progress + patience), and **swaps
contestants in/out by judgement** (plateaued/starved lane → fresh cell; a climber is
never swapped; everything gets run). **Beads never run the GPU** — a bead = code-only
work for another session that lands a cell in `run_sweep.CELLS` "available for the
derby." Config-only levers (existing flags) skip beads — the runner just adds the cell
and races it. (Full model: `wiki/topics/research-loop.md`; memory
`project_derby_operating_model.md` is the resume index.)

- **Live board:** `scripts/derby_v8_board.json` (base `sweep_runs/derby_v8`), 4
  contestants on the **vcf + global-pool** base: `control`, `mate-discount`
  (`--value-discount 0.98`), `disc-recency` (`--value-discount 0.98` +
  `--buffer-recency-frac 0.5`), `buffer-comp` (`--buffer-recency-frac 0.5`).
  Pipelined eval on; cap = 4h backstop (not a hard 1h kill).
  - **SWAP (2026-05-27, commit `fa61aac`):** dropped `stack` (+`--max-plies 45`) —
    it regressed below its 1531 peak while its no-truncation twin mate-discount
    climbed to 1699; the 45-ply cap clips the 50–80-ply defensive games strong
    models play. Replaced with `disc-recency` to test whether the two *climbing*
    levers (value-discount + recency) compound.
- **v8 INTERIM H2H VERDICT (52 chunks ≈ 5h, `round_robin_52chunks.json`, 24g/pair):**
  mean-centered ratings — **mate-discount +115 🥇** (value-discount confirmed the
  key lever) · **buffer-comp +90 🥈** (recency curator is a genuine additive win over
  control) · disc-recency +16 (beats control, but **stacking value-discount+recency
  did NOT exceed either alone** — caveat: fresh seed-0 start, fewer chunks, so the
  combo is handicapped, not cleanly refuted) · control −81 (baseline) · **stack −139**
  (max-plies 45 actively harmful — swap vindicated). Takeaway: value-discount and
  recency are each real wins on the vcf+gp base; truncation hurts; the combo needs an
  equal-footing rematch before calling it.
- **v8 RR2 VERDICT (73 chunks, `round_robin_73chunks.json`):** the top THREE
  tightened into a cluster — **mate-discount +93** (0.98 value-discount, still #1) ·
  **disc-recency +82** (the combo — matured from +16 @52ch to +82, confirming it's
  competitive) · **buffer-comp +82** (recency). control −97; **vdisc-097 −160** (0.97).
  - **Sharper value-discount (0.97) is WORSE** than 0.98 by this peak — the discount
    optimum is at/above 0.98, not below. (Caveat below: vdisc-097's RR peak was its
    under-trained 1555; it later climbed to 1620, so −160 is partly stale.)
  - **METHOD CAVEAT — fresh-start H2H lag (recurring, now seen twice):** a lane that
    starts fresh seed-0 (`disc-recency`, `vdisc-097`) is *systematically undervalued*
    by round-robin until it matures — its saved `peak.pt` lags its live trajectory.
    disc-recency: +16→+82 across 21 chunks; vdisc-097: anchored 1555→1620 *after* its
    −160 RR. **Rule: never retire a still-climbing fresh lane on an H2H verdict; judge
    fresh lanes on climb-RATE, warm-resumed lanes on peak H2H.**
- **Resume:** `python scripts/delo_derby.py --board scripts/derby_v8_board.json --resume`
  → confirm ONE `delo_derby` PID → `nohup bash scripts/derby_watchdog.sh
  scripts/derby_v8_board.json >/dev/null 2>&1 &`. A derby-runner **cron** (~30 min)
  drives swap/restock; the watchdog (startup-grace) keeps it alive.
- **Winner lineage** (rank by HEAD-TO-HEAD — anchored elo saturates ~1700, always
  `scripts/round_robin.py` over `_peaks/*/peak.pt`): vcf mate-teacher (v4) →
  **+global-pool** (v5 H2H win, compounds) → **+value-discount** (v6 H2H win, fixes the
  wins-anchored/loses-H2H overtraining gap). **sgd-steps sweep DEAD** (over-trains).
- **Backlog (beads, `derby-` prefix):** un-gated ideas = status `deferred` (hidden from
  `bd ready`); gate = `bd update <id> --status open`. Code-heavy epics: VCT solver
  (`derby-58f`, being built in a `gomoku-vct-solver` worktree), Rapfi opening-book
  (`derby-pyg`), reanalyze (`derby-3vs`).

## Open candidates — gomocup-AZ implementation survey (2026-05-27)

Researcher pass (session `87a46d75`) studying the AlphaZero-native gomocup engines
(**AlphaGomoku/MK** — Gomocup 2025 #2; **KataGo/KataGomo**) for levers we have NOT
raced. Full survey = [../sources/gomocup-az-techniques-2026-05-27.md](../sources/gomocup-az-techniques-2026-05-27.md);
intake mechanics = [../topics/derby-registration.md](../topics/derby-registration.md).
Six candidates were **red-teamed** (background reviewer) against the v1→v8 verdicts,
`TRAINING_WIKI.md`, the derby-idea backlog, and `model.py`/`mcts.py`. Verdicts:

| # | candidate | red-team verdict | why | status |
|---|---|---|---|---|
| 1 | **WDL (win/draw/loss) value head** | **PASS — #1 Δelo/wall bet** | only lever adding new *info capacity* (decisive-vs-drawn) vs our ~60-70%-draw data; purely TRAINING-side so it does NOT tax the Gumbel-SH generation hot path; aux-head precedent proves byte-identical-off is feasible | **BUILT + RACING** as `derby-x-wdl` (bead `derby-cgf` gated→built→raced by the factory in <20 min, commit 8100f87; scalar default byte-identical, 16 new tests). WDL ckpt can't warm-start a scalar champion → **FRESH-start lane** (judge on climb-RATE). Climb: 751 @ 5min → **1444 @ 15.7min, Δelo/hr 4032, beat-heuristic ✓** (07:09Z). Strong fresh-start trajectory. |
| — | Gumbel-`m` sweep (m=16→8) | red-team MISSED-idea: live flag frozen at v3 default, **never swept**, config-only/byte-identical | focus n=100 sims on fewer root candidates → sharper completed-Q targets | **RACING** as `derby-x-gumbel-m8` (swapped in, retired vdisc-097, commit 1068fb2). **1281 @ 16.2min, Δelo/hr 4977 = steepest on the board, beat-heuristic ✓** (07:09Z). |
| 2 | draw-contempt (`drawValue`) | **KILL standalone** | a knob *on* the WDL head (follow-on sweep, not its own cell); also conflicts with the White "force-the-draw" objective (`derby-7ic`) → needs color-split eval, don't run blind | **WDL-sequenced follow-on family** (only if `derby-x-wdl` clears the field): draw-contempt (`--draw-value`); **per-action (Q) WDL head** (AlphaGomoku `actionValues` — a WDL Q per move as a selection prior, the last un-triaged AlphaGomoku value-axis lever; Class-C new head). Both wait on the WDL verdict. |
| 3 | LCB root move selection | **KILL** | written against visit-count selection; production is **Gumbel SH argmax over completed-Q** (`self_play.py:548`); no per-node variance accumulator (scalar `W` only); redundant with Gumbel's `sigma(q_hat)`; C-hot-path cost | rejected — recorded so it's not re-proposed |
| 4 | variance/uncertainty-scaled PUCT | **KILL** | no variance state + wrong engine (SH governs the root, not per-node cPUCT); closest analog v3 `forced` landed mid-tier *below* Gumbel; high C build cost, low expected Δelo | rejected |
| 5 | moves-left head | **KILL** | throughput goal already WON by adjudicate (`--max-plies 45`, +44 H2H v6); value-target half duplicated by the champion `--value-discount`; novel residue (search tie-break) is small + C-hot-path | rejected (revisit only if a delta-vs-value-discount is articulated) |
| 6 | in-search VCF proven-score backup | **KILL** | not a dup of `derby-58f` (different axis) but contradicts the explicit `derby-7ic` design ("RELABEL via teacher, NOT runtime alpha-beta"); a VCF solve per node on the generation hot path is the worst possible thing for Δelo/**hour** | rejected (a root-only cheap variant could be reconsidered later) |

**Round-2 candidates — from a deep read of actual KataGo + AlphaGomoku source (2026-05-27,
training-side, ZERO generation cost so Δelo/hr is protected):**

- **REGISTERED `derby-79l` → cell `derby-x-soft-policy`** (the pick): KataGo's **soft-policy
  auxiliary target** — a second policy-loss term against a 4th-root temperature-flattened copy
  of the *already-recorded* completed-Q `pi`, scaled by `--soft-policy-weight 0.15` (default
  0.0 = byte-identical). Under 60-70% draws the sharp target concentrates mass on 1-2 defensive
  moves and the net loses the search's runner-up structure; the soft target re-injects it (KataGo
  added it for exactly this under-taught-drawish reason). ~6 lines in `train.py:compute_loss`,
  orthogonal to value-discount (value head) + VCF (target). **BUILT in ~6 min (commit 7450019);
  SWAPPED IN + RACING (commit 210b105, the runner retired result-locked `vdisc-099` for it).
  Just started — too young to read.**
- **Queued runners-up — HELD on sequencing discipline:** `derby-x-surprise-weight` (per-sample
  loss weight `1 + λ·KL(search_pi ‖ net_prior)`) and `derby-x-playout-weight` (SH visit-confidence
  weight) are both *policy-signal-enrichment* levers **correlated with soft-policy** — file them
  only after soft-policy's result is known (if soft-policy wins, they're well-motivated; if it
  loses, they're likely weak too). Don't fire correlated bets blind.
**Round-3 — ARCHITECTURE axis (deep source read of AlphaGomoku/KataGo net code, 2026-05-27):**

- **REGISTERED `derby-sib` → cell `derby-x-mish`** (the arch pick): swap ReLU→**Mish** activation
  (`--activation mish`, default relu = byte-identical). KataGo's `act()` factory offers it;
  zero added params, identical state_dict keys, **model.py-only**. *Load-bearing infra fact
  (verified): our native-C MCTS engine does NOT compute the forward — it does tree ops and calls
  back to a PyTorch `evaluate_planes` evaluator (`_mcts_native.c` has no conv/relu). So an
  activation swap needs NO native-C kernel* (correcting the initial worry). Fresh-start (ReLU-
  trained weights misbehave under Mish), judge on climb-rate. **BUILT (commit 793a86a, derby-sib
  CLOSED, +mish test); cell live, QUEUED for swap-in when a lane frees.**
- **SE (squeeze-excitation) blocks — DEPRIORITIZED, not filed:** the best *structural* uncorrelated
  lever (AlphaGomoku defaults to SE inside ConvNext; KataGo has none; it's a per-channel
  multiplicative *gate*, distinct from our global-pool's additive *bias*). But on our ~0.8M-param
  4×64 net the gain is uncertain and the unfused per-block GAP→FC→sigmoid the PyTorch eval must
  run is a possible small latency tax on the gen rate that defines Δelo/wall. File only if Mish
  clears the field and we want to spend a fresh-start arch lane. (`--se-blocks`, byte-identical-off.)
- **Nested-bottleneck / ConvNext depthwise — DOA (do NOT file):** KataGo's nested-bottleneck is a
  width/depth-amortization trick for deep wide trunks (b18c384); at 4×64 it has nothing to amortize
  and shrinks effective width, and it's multi-knob. ConvNext depthwise-7×7 is multi-knob + MPS-
  unfriendly. Wrong tools for our net size.

- **Multi-knob future study (NOT derby-shaped):** our optimizer is bare `AdamW(lr)` with no
  scheduler/warmup/grad-clip; both upstreams use SGD-momentum + LR warmup/decay + clip — real
  headroom but violates one-lever-per-cell, so it's a deliberate later study, not a single cell.
- **Confirmed dead-on-arrival (do NOT file):** soft *value* / TD-value / learned shortterm-
  value-error head (needs recorded bootstrap targets = pipeline change); extra symmetry aug
  (D4 = the full gomoku symmetry group, we're already complete); a score/margin head (degenerate
  for win/draw/loss).

**Loop status (2026-05-27, research cron `4e4dcc03`, 20-min — tick 5: STEADY-STATE monitoring):**
FOUR researcher contestants built across FOUR distinct axes; THREE now **racing** — `derby-x-wdl`
(value-rep, peak 1444), `derby-x-gumbel-m8` (search, peak 1309), `derby-x-soft-policy` (policy-
signal, just swapped in for the retired `vdisc-099`) — plus champion `mate-discount` (1718). The
two fresh value/search cells hold the **steepest Δelo/hr on the board** (the fresh-start climb-rate
signal). `derby-x-mish` (activation) BUILT + QUEUED for the next free lane. The loop is now in
**steady state: propose at the rate the board frees lanes**, not faster — it promotes nothing new
until a verdict lands. The sequenced backlog (all PENDING their trigger result): policy-signal
runners-up ← soft-policy; SE ← Mish; the WDL follow-on family (draw-contempt + per-action-Q head)
← WDL. Researcher monitors only — the derby-runner owns the GPU swaps. North
star = **Δelo/wall**.

## Rules

- **Race to 140 epochs.** 140 is the milestone because that's roughly where a
  fresh model historically first beats the heuristic baseline. **Beat-heuristic
  (model_elo ≥ 800)** is the early checkpoint; the real prize is the **strongest
  model by epoch 140**.
- **10-epoch chunks.** Each scheduling step advances one idea by a 10-epoch
  increment (`gomoku.train --resume <idea>/latest.pt --epochs 10`).
- **Δelo/hour hill-climb priority** (Jason 2026-05-24: "never-run, then delta
  elo/hour — hill climb elo"). Order: **(1) never-run / entry-fee first** — an
  idea needs **2 elo points** to have a Δelo/hr slope, so round-0 then round-1 run
  for every idea (fewest points first); **(2) then highest Δelo/HOUR** over the
  most recent chunk — compute follows the *steepest recent climb*, not the highest
  absolute elo. Everyone caps; the steepest climbers get there first.
  - *History:* v1 ranked by last-chunk *raw Δelo* and pathologically fed the
    *worst* idea (a floored idea at Δ0 outranked a strong idea whose chunk dipped).
    v2 patched that with current-elo *level* (but that over-feeds an already-peaked
    champion and starves a faster challenger). The Δelo/**rate** rule is neither
    pathology: a floored idea sits at 0/hr and any genuine climber outranks it, and
    ranking the *rate* (not the level) is the literal hill-climb. The round-0/1
    entry fee avoids the floor-noise artifact (at the floor all ideas are ~equal, so
    a 1-point "rate" is meaningless).
- **Fresh self-play, shared init.** All ideas start from an identical fresh init
  (`--size small --seed 0`). No warm-start, no shared parent.
- **One lever each.** Every idea changes exactly ONE flag vs **C0-baseline** —
  clean attribution. C0-baseline is the control.
- **Anchored-elo scoring.** Score = the last `eval/model_elo` in
  `<idea_dir>/checkpoints/eval_results.jsonl`. Δelo for the queue = the change in
  that score across the idea's most recent chunk.

**Shared knobs** (held constant across all ideas):
`--games-per-epoch 64`, `--training-steps 400`, `--batch-size 256`,
`--replay-buffer-size 100000`, `--lr 1e-3`, `--temperature-moves 8`,
`--c-puct 1.25`, `--size small`, `--seed 0`. C0's `--n-simulations 200` is the
generation-strength control point.

**Out of scope (future board).** Curator / curriculum ideas
(recency-weighted, lru, gomocup-seed) are deferred — they require the
`train_replay` flywheel engine (curated in-RAM sampling over an archived buffer),
not the fresh self-play engine this derby races on. They get their own board once
a headroom parent exists.

---

## v1 FINAL — verdict (called 2026-05-24 at 5/8 capped; ranked by ELO, NOT wall-clock)

> **Wall-clock is busted for this run** and must NOT be used to rank: the derby ran
> single-process (`gomoku.train`, one stream, GPU ~30%), so every wall-time / Δelo/hr
> here is single-stream and unrepresentative of production (wave-mode, 8 workers,
> saturated). The under-counting trap ([[project-perf-bench-lesson]]). Rank by elo.

| rank | idea | lever | peak | final | beat-heur @ep |
|---:|---|---|---:|---:|:--:|
| 1 | **open-div4** | random_opening_moves 4 (WL3) | **1385** | 1385 | 90 |
| 2 | **temp-16** | temperature_moves 16 | **1340** | 1240 | 90 |
| 3 | sgd-800 | training_steps 800 | 1284 | 1081 | 70 |
| 4 | sims-400 | n_simulations 400 | 1265 | 1094 | 50 |
| 5 | buf-30k | replay_buffer 30k | 908 | 751 | 110 |
| — | C0-baseline | control | 567 (climbing, called @ep60) | — | — |
| — | ema-099 | ema_tau 0.99 | 405 (floor, ep50) | — | — |
| — | sims-100 | n_simulations 100 | 389 (NEVER grokked, ep110) | — | — |

**Findings:**
1. **Exploration/diversity levers win the ceiling.** Random openings (1385) and high temperature (1340) are the top 2 — *above* the compute levers (more sims 1265, more SGD 1284). Diversifying self-play raises reachable strength.
2. **Compute levers grok FASTER but peak LOWER.** Beat-heuristic timing tracks per-epoch compute: sims-400 @ep50 < sgd-800 @ep70 < open-div4/temp-16 @ep90 < buf-30k @ep110. More sims/SGD = earliest crossing; exploration = highest ceiling.
3. **Overtraining is real and lever-dependent.** sims-400/sgd-800 peaked ~ep90 then regressed ~180 elo by ep140; `open-div4` ended *at* its peak (openings sustain the climb, no overtrain); temp-16 mild (1340→1240).
4. **`sims-100` (100 sims) never groks** — floor 389 at ep110. Weak MCTS targets cap the climb (and it's the only *trainer-bound* recipe: gen<train).
5. **Generation-bound, not trainer-bound.** Train is a fixed ~10.5s/epoch floor; MCTS generation is 2–5× that and scales with sims. The trainer is cheap; Δelo/hr leverage is all on generation speed.
   > ⚠ **CORRECTED 2026-05-24 (Jason): this v1 reading is stale + was measured single-process.** It was taken on the SINGLE-PROCESS v1 derby (GPU ~30%, busted wall-clock), so the gen/train ratio itself is suspect. More importantly, after the perf wins (fp16-eval, V=512, native MCTS) **the regime FLIPPED: generation now OUTPACES the trainer — it FLOODS it.** Per [perf-bench-vs-real-training-cost.md](../topics/perf-bench-vs-real-training-cost.md): "maximizing generation throughput floods the trainer"; the LF1 runaway (per-epoch 20s→7min) is the generator producing positions faster than the trainer can use them, with `sgd_per_position` blowing up trying to consume the flood. **"Generation-bound" is NOT a standing truth — it's recipe-dependent (high-sims wave leans slower-gen; optimized fp16/high-V floods).** The v3 cards below that invoke "generation-bound" should be read through this correction. The fix is a fixed per-epoch SGD cap decoupled from inflow → the `derby-gumbel-fast5s` lane.
6. **Method fixes (mid-run):** priority must rank by *current elo*, not last-chunk-Δelo (the latter fed the *worst* idea); peak checkpoints were lost to `keep-last-n=3`.

## v2 — what's next (queued)

Re-run the **top 3** (`open-div4`, `temp-16`, `sgd-800`) **HEAD-TO-HEAD**, using the **production multiprocess recipe** (`run_sweep` wave-mode, 8 `selfplay_worker`s — saturates the GPU, so wall-clock is REAL). Eval = round-robin *direct matches* among the 3 (they're all >1280, so they'd saturate the anchor ladder — head-to-head via `delta_e_harness --head-to-head` is the correct eval for strong models). **Wall-native budget** (hours, not epochs; chunk = wall-slice; allocate+stop by Δelo/hr) measured on the saturated machine so Δelo/hr is finally honest. Carry the fixes: current-elo priority, peak-checkpoint snapshotting. (Verified 2026-05-24: the production recipe IS multiprocess; the single-stream drift was only in the v1 derby harness.)

---

## v3 — UNIFIED prior-art race (LAUNCHED 2026-05-24, `scripts/derby_v3_board.json`)

A **unified board**: Jason called it — rather than run v2 (the top-3 head-to-head) to
cap and *then* a separate v3, we **ported the v2 carryover recipes into v3** and race
everything at once. v2 was stopped at round-0 (all at the 389 floor → zero data lost),
which freed the box for the native `.so` rebuild. The roster (9 ideas, one lever each
vs the `c0` control, fresh `--size small --seed 0`, scored by anchored elo then
head-to-head at the top): the **v1/v2 carryovers** (open-div4, temp-16, sgd-800) +
the **4 prior-art levers** (playoutcap, forced, swa, gumbel) + a **sims100 control**
for gumbel. Each picked to attack a **specific v1 finding** — v1 told us we're
**generation-bound** and **exploration/diversity beats raw compute for the ceiling**,
so the new levers are biased toward *better targets per unit of generation*. All ran
through the lab's two-queue fan-out (5 worktree code lanes, opt-in flags, production
byte-identical when off, merged `--no-ff` serially with one native rebuild).

> Wall-fairness resolved: **gumbel + forced-playouts both run in the native C engine**
> (`_mcts_native.c`, rebuilt). Gumbel's first cut came back python-only (~5× slow =
> DOA per Jason); the **native C port** (per-game Sequential Halving inside the wave)
> made it **0.86–1.26× native PUCT** — wall-fair, raced wall-matched like the rest.
> (Gumbel + sims100 run at sims=100: gumbel's value-prop is good targets at *cheap*
> sims; sims100 is the plain-MCTS control that isolates whether gumbel rescues them.)
> **aux-head** (opponent-reply, Class-C model-arch) is built + verified but parked on
> its **own axis/board** — not in this search/recipe race.

### v3 FINAL — verdict (called as-is 2026-05-24, in prep for new contenders)

> Jason called v3 once `gumbel-fast5s` proved itself, to clear the board for a new
> round. Standings are the live fine-grained peaks (the slice-close state lags the
> mid-slice high-water mark — e.g. `gumbel-fast5s` touched 1620 mid-slice but its
> slice-close points were ~1455). 6 ideas after the mid-run prune of temp-16 /
> sgd-800 / playoutcap / swa (all stuck at the 389 floor).

| rank | idea | peak | wall→peak | grokked? | what it is |
|---:|---|---:|---:|:--:|---|
| 1 | **gumbel-fast5s** | **1620** ✓ | **~17 min** | yes | Gumbel@100 gen + **fixed-step trainer** (non-wave, `--sgd-steps-per-epoch 64`) |
| 2 | **gumbel** | **1580** ✓ | ~22 min | yes | Gumbel@100 gen + wave + `sgd_per_position` |
| 3 | forced | 1262 ✓ | ~21 min | yes | KataGo forced playouts + target pruning |
| 4 | open-div4 | 776 | ~21 min | ~ | v1's #1 (random openings) |
| 5 | sims100 | 697 | ~23 min | ~ | plain MCTS@100 — gumbel's control |
| 6 | c0 | 603 | ~21 min | ~ | no-lever control |

**Findings:**
1. **The Gumbel cheap-sims generator dominates.** `gumbel`/`gumbel-fast5s` (Gumbel@100) peaked ~1580–1620 — *more than 2× the control* `sims100` (plain MCTS@100, 697). So Gumbel doesn't just ride cheap sims, it **rescues** them: good targets at n=100 ≫ plain visit-count targets at n=100. This was the highest-leverage import.
2. **Fixed-step trainer is CO-EQUAL with wave — its wins are STRUCTURAL, not a Δelo-rate separation.** (Corrected post-Reviewer: an earlier draft claimed "~2× Δelo/hr / beats wave" — that mixed fast5s's time-to-peak against gumbel's *total* wall and is wrong.) Same Gumbel@100 generator; only the training mode differs. The two peaked **within eval noise** (fast5s 1620, gumbel 1580 — a ceiling tie), and on the derby's **canonical Δelo/hr** ((peak−389)/wall-to-peak, `standings.md`) **wave actually edges it: gumbel 3031 vs fast5s 2825** (~10% apart; like-for-like time-to-peak is ~1.5× at most — it flips with the wall basis). So **no clean rate separation.** Fixed-step's real advantages are structural: it reached its peak in a **single contiguous slice** (no multi-chunk resume), it is **structurally incapable of the inflow-driven runaway** (the whole point vs `sgd_per_position`), and it validated the gen-flooding fix — which is why it's the cleaner v4 **control**, not because it out-paced wave. Diagnostics (healthy): `reuse` ~1.4 settling to ~1.05, `pl` **4.39→1.72 descending** vs climbing `cumsteps` (productive SGD, *not* the redundant-flattening failure), `plies` ~67 (real defensive play).
3. **Prior-art compute-efficiency > v1 exploration levers.** `open-div4` (v1's ceiling champion, random openings) reached only 776 here vs gumbel's ~1600. The v1 headline "*exploration beats compute for the ceiling*" is **superseded once you have good cheap targets + efficient training**: Gumbel (target quality per sim) + fixed-step (training efficiency under a gen flood) beat the exploration knobs outright.
4. **Regime correction (the load-bearing reframe).** Generation now **floods** the trainer (`gen=0.4s ≪ train=3.0s`); "generation-bound" was stale (see correction in v1 finding #5 + `perf-bench-vs-real-training-cost.md`). The cure — a **fixed per-epoch SGD cap decoupled from inflow** (structurally can't run away) — is exactly the fixed-step mode, now empirically the best trainer.
5. **forced (KataGo) is a solid mid-tier search lever** (1262) — cheaper exploration than more sims, but well below the Gumbel-generator + fixed-step combo.

**Built this round (the durable artifacts, all merged + tested):** native Gumbel C port (`_mcts_native.c`, wall-fair 0.86–1.26× PUCT); the **fixed-step trainer mode** (`--sgd-steps-per-epoch`, non-wave async + non-blocking ingest + `sample_reuse_ratio`/`cumulative_sgd_steps` metrics); the **wave-mode SIGTERM deadlock fix** (+ test); the **Δelo/hr hill-climb scheduler** (never-run → entry-fee → Δelo/hr, peak tiebreaker); `scripts/watch_derby.py` (live elo/Δelo·hr/wall viewer) + `scripts/derby_dashboard.py` (wandb workspace) + `scripts/derby_sync_elo.py`; the discovery that **eval was already in wandb history** (trainer forwards the eval jsonl); and the **generation-flooding** correction (memory + wiki).

**Caveats:** anchored elo saturates ~1700 (the strong climbers are near the ceiling) → a **head-to-head** (`delta_e_harness --head-to-head`) is the rigorous tiebreak if a clean gumbel-vs-fast5s separation is ever needed; the two are within eval noise on ceiling. The fixed-step A/B conflates wave→non-wave + scaled→fixed (the training-*mode* fork, by design, not a single knob).

**Prep for new contenders (next round):** board is clear. Open candidates: the parked **aux-head** (opponent-reply, Class-C — built, awaiting sign-off); an **N-sweep on `--sgd-steps-per-epoch`** (the reuse-ratio knob: how hard can we push fixed-step before redundant SGD?); **reanalyze / curator** ideas (need the train_replay flywheel — their own board); and `gumbel-fast5s` itself is the **new baseline** to beat. Promotion of fixed-step+Gumbel to a production lineage is a deliberate ESCALATE (Jason's call), deferred.

## v4 — best-shot COMBINATIONS (LAUNCHED 2026-05-25, `scripts/derby_v4_board.json`)

The first **combination** round (Jason: "no more one-lever — put forth our top 3
combinations that we think have the best shot at being great gomoku players").
Every lane shares the **v3-winning base** (fixed-step `--sgd-steps-per-epoch 64` +
Gumbel@100, non-wave) so the only delta per lane is the headline lever. Wall-slice
engine (`run_sweep_wall_slice`), 600s slices, **10800s (3 hr) per-idea cap** (the
deeper bets need absorption room — see `absorption-phase` memory), Δelo/hr
hill-climb priority. All four lanes start **fresh + fair** (the control is a
distinct cell `derby-v4-control`, byte-identical to `gumbel-fast5s`, so it does not
resume v3's 8.8G checkpoint).

| lane | cell | lever (vs control) | source |
|---|---|---|---|
| **control** | `derby-v4-control` | none — fresh v3 winner (fixed-step + Gumbel@100) | Derby v3 |
| **signal** | `derby-signal` | KataGo aux supervision: opp-reply policy head + per-cell ownership head, both `@0.15` | KataGo |
| **wholeboard** | `derby-wholeboard` | KataGo global-pooling residual blocks (latter half; +4.79% params) | KataGo |
| **vcf** | `derby-vcf` | exact VCF mate-teacher (overwrites policy/value targets on forced wins; value disc 0.98/floor 0.90) | Rapfi/classical |

**Integration (all merged to `main`, tests green, smoke-validated):** global-pool
(`c5a81d7`), VCF solver (`3a6c6d9`, 400-fuzz vs independent referee = zero false
positives), opp-reply+ownership aux heads (`5eb6eec`/`36a446a`, byte-identical-off
verified 96/96 examples + both heads ENABLED in smoke), Rapfi yardstick
(`0c30427`, START 9 → OK). Cells + board `251d1bf`/`3acdb00`; watchdog `b85c113`.
**Yardstick:** Rapfi (Gomocup Elo 2625) runs separately on the leader (above-ladder),
not as a per-chunk anchor.

**Live:** dashboard `https://wandb.ai/jasonyandell-forge42/gomoku?nw=gv4fh2vq2rr`;
`scripts/derby_v4_watchdog.sh` supervises (restart-if-dead + `narration.log`);
`python scripts/watch_derby.py --board scripts/derby_v4_board.json` for the live
terminal board. **Known minor:** max-plies *draws* yield `ownership=None` (masked)
rather than the all-zeros the code comment promises — benign (no winner to credit;
trained models rarely draw at max_plies). Under Reviewer at launch. Reviewer
verdict: **PASS** (no BLOCK; merge resolution correct, control fair, aux NaN-safe,
VCF no false positives).

### v4 FINAL — verdict (stopped 2026-05-25 after ~12.3 hr / 67 chunks, 0 watchdog restarts)

Ran fully autonomously overnight (cap raised 3h→24h mid-run so the leader kept
getting fed). Anchored elo **saturated ~1700**, so the overnight peaks were a tight
66-elo cluster that anchored eval can't separate — resolved with a **head-to-head
round-robin** (`scripts/round_robin.py`, reuses `delta_e_harness.head_to_head_eval`;
120 games/pair, paired 4-ply openings, sims=100).

| metric | vcf | control | signal | wholeboard |
|---|---:|---:|---:|---:|
| anchored peak elo | **1784** | 1760 | 1738 | 1718 |
| H2H round-robin rating | **+31** | −29 | +6 | −8 |
| H2H rank | **1** | **4** | 2 | 3 |

**The head-to-head reshuffled the order — and that's the lesson.** On anchored elo
`control` (plain v3 winner, no extra lever) looked like #2 (peak 1760), but played
directly against its peers it **loses all three matchups** (−9/−29/−50) and ranks
**last** — its anchored score was overtrained inflation against the fixed ladder.
**`vcf` is the genuine champion on BOTH metrics**: highest anchored peak (1784) AND
beats every lane head-to-head (+14/+29/+50). The **exact VCF mate-teacher is the
standout v4 lever.** The two KataGo combos (signal=aux heads, wholeboard=global-pool)
land in the middle and did *not* cleanly separate from the baseline.

**Compute-fairness caveat (load-bearing):** early "leads" were undertraining
artifacts. `vcf` sat at ~1497 for hours looking like a clear 4th, then — once the
Δelo/hr hill-climb kept feeding the only lane still gaining — climbed straight to
#1. The scheduler's apparent "over-feeding" of the laggard was the most informative
allocation of the night. Lesson: **rank above the anchored ceiling with head-to-head,
and don't trust an early anchored lead before lanes have equal compute.**

**Statistical honesty:** H2H CIs are wide (±62 elo; high draw rates ~50%, i.e. good
defense), so the top-3 ordering (vcf > signal > wholeboard) is *not* airtight — but
the directional signals are clean and consistent: **vcf beats everyone, control loses
to everyone.** Peak checkpoints saved at `sweep_runs/derby_v4/_peaks/<lane>/peak.pt`.

**Above-ladder Rapfi yardstick (`vcf` champion, 9x9 freestyle, 20 games/budget,
`sweep_runs/derby_v4/rapfi_vcf.jsonl`):** vcf vs `pbrain-rapfi` (arm64-NEON,
build 6e0a132) — **100ms: 0W-0L-20D (50%); 500ms: 2W-0L-18D (55%); 1000ms:
1W-3L-16D (45%)**. Read: **roughly draw-parity** — near-total draws, Rapfi only
edges ahead at the longest control (3L vs 1W @ 1s). Surprisingly strong for a
~1700-anchored from-scratch net. CAVEATS: tiny draw-dominated sample (noisy ±10%
per 2-game swing); **Rapfi's 2625 is a 15x15 Gomocup rating that does NOT transfer
to 9x9 freestyle** — this says "competitive with Rapfi *on 9x9 freestyle*", NOT
"~2625 elo"; and 9x9 freestyle is intrinsically drawish under solid two-sided
defense. Model plays sims=100 vs Rapfi's time budgets (not a matched control).

## v5 — STACK THE WINNERS (LAUNCHED 2026-05-25, `scripts/derby_v5_board.json`)

v4's champion was the **exact VCF mate-teacher** (`--vcf-teacher`) on the fixed-step
+ Gumbel@100 base. v5 asks the **compounding question**: do the *other* v4 levers
ADD anything on top of vcf, or is bare vcf already the bar? Every lane = the vcf
base + exactly ONE added lever; `control` is the bare vcf base (the bar to clear).
All four start **fresh + fair** on the same wall budget (global-pool changes the
trunk so it can't warm-start from a non-global checkpoint — uniform fresh start
keeps it apples-to-apples). Engine `run_sweep_wall_slice`, Δelo/hr hill-climb
priority, peak checkpoints at `sweep_runs/derby_v5/_peaks/<lane>/peak.pt`.

| lane | lever (vs control = bare vcf base) | source |
|---|---|---|
| **control** | none — bare vcf base (VCF mate-teacher + fixed-step + Gumbel@100) | Derby v4 champion |
| **vcf-signal** | + KataGo aux heads: opp-reply policy + per-cell ownership, both `@0.15` | KataGo |
| **vcf-wholeboard** | + KataGo global-pooling residual blocks (latter half; +4.79% params) | KataGo |
| **vcf-deep** | + deeper VCF solver (`--vcf-max-depth 32 --vcf-max-nodes 500000` vs 16/200k) — proves longer forced wins → more exact mate labels | Rapfi/classical |

### v5 FINAL — verdict (stopped 2026-05-26 at 38 chunks; H2H is the verdict)

> **Read this from the head-to-head, not the anchored peaks.** Anchored elo
> SATURATES ~1700 and at 38 chunks the peaks are a NOISY, less-settled cluster
> than v4's 67-chunk run. v5 was also **restarted several times mid-race to ship
> infrastructure** (board cap 3h→24h, slice 600s→300s, PIPELINED eval, and the new
> peak-progress+patience priority metric) — so within-v5 wall-clock / Δelo-rate
> comparisons are **confounded**. The honest framing: v5 is BOTH a lever-compounding
> test AND the round where we built two durable infra wins (pipelined eval + the
> peak-progress metric). The clean signal is the post-race **H2H round-robin**
> (`scripts/round_robin.py`, 120 games/pair, paired 4-ply openings, sims=100,
> `sweep_runs/derby_v5/round_robin.json`).

| metric | vcf-wholeboard | vcf-deep | vcf-signal | control |
|---|---:|---:|---:|---:|
| anchored peak elo | **1634** | 1455 | 1606 | 1476 |
| H2H round-robin rating | **+81** | +44 | +7 | −132 |
| H2H rank | **1** | 2 | 3 | **4** |

Pairwise H2H Δelo (row vs column; + = row beats column):

| | control | vcf-signal | vcf-wholeboard | vcf-deep |
|---|---:|---:|---:|---:|
| **control** | · | −95 | −165 | −137 |
| **vcf-signal** | +95 | · | −44 | −29 |
| **vcf-wholeboard** | +165 | +44 | · | +35 |
| **vcf-deep** | +137 | +29 | −35 | · |

**Compounding verdict (from H2H): the levers DO compound on vcf — bare vcf is NOT
the bar, it's the floor.** `control` (bare vcf base) **loses all three matchups**
(−95/−165/−137) and ranks dead last; every +1-lever lane beats it head-to-head.
The standout add-on is **vcf-wholeboard** (KataGo global-pooling): it beats every
peer (+44/+35 over signal/deep, +165 over control) AND holds the top anchored peak
(1634). So whole-board structure stacked on exact mate labels is the v5 win.

**The H2H reshuffled the order (the v4 lesson, again):** on anchored elo `vcf-signal`
(1606) looked like the clear #2 and `vcf-deep` (1455) the clear last — but played
directly, **vcf-deep ranks #2 (+44) and signal drops to #3 (+7)**, and deep beats
signal head-to-head (+29). vcf-signal's high anchored peak was ladder-inflation it
couldn't cash against live opponents; vcf-deep's low anchored peak understated a
model that actually plays well. **Don't trust an anchored lead near the ceiling.**

**Honest caveats:** H2H CIs are wide (±62–69 elo, draw rates ~30–35% decisive→~65%
draws, i.e. strong two-sided defense), so the **middle order (deep > signal, +44 vs
+7) is inside noise** and not airtight; only the bookends are clean (wholeboard
clearly #1, control clearly #4). The anchored peaks are only ~38 chunks (vs v4's 67)
and the mid-race restarts mean the climb signal is muddier than v4's. What IS robust:
**every lever beats bare vcf, and global-pool is the strongest of the three.** The
load-bearing read is directional, not the exact middle ordering.

**Two durable infra wins shipped mid-race (the other half of v5's value):**
**(1) pipelined eval** — eval runs concurrently with the next training slice instead
of blocking it (the `(pipelined)` PEAK milestones), so the GPU queue stays fed; and
**(2) the peak-progress + patience priority metric** — the scheduler now ranks lanes
by recent peak-progress with a patience window rather than raw last-chunk Δelo/hr,
which is what kept feeding lanes still gaining (the v4 lesson that the laggard `vcf`
was the most informative allocation, now baked into the scheduler).

**Next = v6.** vcf-wholeboard is the new base to beat (vcf + global-pool). Open
questions: does **vcf + global-pool + aux** stack a fourth lever cleanly, or do the
two KataGo levers (signal/wholeboard) overlap? A **longer, restart-free** v5-rerun
(no mid-race infra churn) would settle the muddy middle order; and the vcf-wholeboard
champion is the natural candidate for the **Rapfi above-ladder yardstick** + a
promotion-to-lineage ESCALATE (Jason's call).

## v6 — RESEARCHER ROUND 1 (LAUNCHED 2026-05-26, `scripts/derby_v6_board.json`)

First batch gated from the **beads backlog** (researcher proposes / Jason gates,
2026-05-25). All five lanes ride the **vcf base** (the v4/v5 champion: exact VCF
mate-teacher + fixed-step `--sgd-steps-per-epoch 64` + Gumbel@100); `control` =
bare vcf base, the bar. Three independent levers, each a delta vs control:

| lane | lever (vs control = bare vcf base) | bead | source |
|---|---|---|---|
| **control** | none — bare vcf base | — | v4/v5 champion |
| **adjudicate** | + `--max-plies 45` — truncate dead/drifting games → more fresh openings/hr (attacks Δelo/hr); blunt-cap first cut, confidence-resign is the follow-on | derby-24a | AGZ resignation |
| **mate-discount** | + `--value-discount 0.98` — z = outcome·γ^plies_to_end; generalizes the VCF mate-distance discount to ALL outcomes; targets the v4 "wins anchored but loses H2H" overtraining gap | derby-2yn | our VCF path |
| **sgd128** | + `--sgd-steps-per-epoch 128` (2× the 64 baseline) — reuse-ratio sweep | derby-g2j | — |
| **sgd256** | + `--sgd-steps-per-epoch 256` (4× baseline) — reuse-ratio sweep | derby-g2j | — |

### v6 FINAL — verdict (closed 2026-05-26 at 168 chunks; H2H is the verdict)

> **Read this from the head-to-head.** Anchored elo SATURATES ~1700 and at 168
> chunks the lanes are well-explored but the anchored peaks are a noisy cluster near
> the ceiling (adjudicate 1682, control & mate-discount tied 1665, sgd256 1606, sgd128
> 1555). The real verdict is the post-race **H2H round-robin** (`scripts/round_robin.py`,
> 120 games/pair, paired 4-ply openings, sims=100, `sweep_runs/derby_v6/round_robin.json`).
> mate-discount's WHOLE POINT is to fix the anchored-vs-H2H gap, so for it the H2H
> verdict is the only one that counts.

| metric | mate-discount | adjudicate | control | sgd256 | sgd128 |
|---|---:|---:|---:|---:|---:|
| anchored peak elo | 1665 | **1682** | 1665 | 1606 | 1555 |
| H2H round-robin rating | **+46** | +20 | +7 | −31 | −41 |
| H2H rank | **1** | 2 | 3 | 4 | **5** |

Pairwise H2H Δelo (row vs column; + = row beats column):

| | control | adjudicate | mate-discount | sgd128 | sgd256 |
|---|---:|---:|---:|---:|---:|
| **control** | · | −44 | −20 | +50 | +41 |
| **adjudicate** | +44 | · | −58 | +53 | +41 |
| **mate-discount** | +20 | +58 | · | +44 | +61 |
| **sgd128** | −50 | −53 | −44 | · | −17 |
| **sgd256** | −41 | −41 | −61 | +17 | · |

**Per-lever verdicts (from the H2H, the row that matters):**

**(a) adjudicate — YES, games/hr translated to real strength.** `--max-plies 45`
beats control head-to-head (+44, control winrate 0.438 over 120 games) and ranks #2.
Truncating dead/drifting games didn't poison the value head with false early
terminations (the AGZ false-resign worry); the extra fresh openings/hr cashed into a
stronger model. The blunt cap is a clean win — and the confidence-resign + disable-frac
follow-on (still in derby-24a's design) is the natural next refinement, now de-risked.

**(b) mate-discount — YES, it beats control HEAD-TO-HEAD (+20) and TOPS the table
(+46).** This is the hypothesis the lever was built for: distance-discounting the value
target for ALL outcomes (not just VCF mates) closes the v4 "wins anchored but loses
H2H" overtraining gap. mate-discount is tied with control on anchored peak (both 1665)
yet **beats control when they actually play** (+20) AND beats adjudicate directly
(+58) despite adjudicate's higher anchored peak (1682) — exactly the anchored-understates-
H2H signature the discount was meant to produce. It's the **clear #1** and the v6 win.

**(c) more SGD/epoch — NO, it HURT.** Both `sgd128` (−41) and `sgd256` (−31) rank
below control (+7) and **lose to it head-to-head** (control +50 over sgd128, +41 over
sgd256). Pushing SGD steps/epoch above the 64 baseline over-trains on the same buffer
(redundant-sample flattening) and costs real strength. sgd256 edges sgd128 (+17) but
that's inside noise; the robust read is **64 is at or below the reuse-ratio optimum —
do NOT raise it.** (If anything the sweep wants a point *below* 64, not above.)

**Honest caveats:** H2H CIs are wide (±62 elo, ~30–40% decisive → ~60–70% draws, strong
two-sided defense), so the bookends are clean but the middle is soft. **What's robust:**
the two losers (sgd128/sgd256) clearly sit below the three winners (every cross-pair is
−41…−61, all outside or at the CI edge), and **mate-discount clearly beats sgd128/sgd256
and adjudicate** (+44/+61/+58). **What's inside noise:** the control↔mate-discount (−20)
and control↔adjudicate (−44) margins and the mate-discount-vs-adjudicate ordering are
each near one CI half-width — directionally mate-discount > adjudicate > control, but not
airtight at the top. The load-bearing reads (mate-discount #1, both sgd lanes lose to
control) survive the CIs.

**Carry-forward = mate-discount (`--value-discount 0.98`).** It's the cleanest single
win of the round: tops the H2H, fixes the exact overtraining artifact it was designed
for, and is a one-flag change that composes with the vcf base. adjudicate is the
**second** carry candidate — it's a pure generation lever (orthogonal to the value-target
lever), so mate-discount + adjudicate is the natural v7 stack to test for compounding.
**sgd128/sgd256 are dead** — the 64 baseline stays (or drops). Open: does mate-discount
stack on the v5 global-pool champion (vcf + global-pool + value-discount)? And the
adjudicate confidence-resign follow-on (derby-24a design) vs the blunt `--max-plies` cap.

### v3-gumbel  (HIGHEST leverage)
**Lever:** `--gumbel-root` (+ `--gumbel-m 16`, `--gumbel-c-visit 50`, `--gumbel-c-scale 1`) — Gumbel-top-k root sampling + Sequential Halving, completed-Q policy target. **Source:** Gumbel AlphaZero/MuZero, Danihelka et al. (DeepMind, 2022).

**Hypothesis:** Directly attacks v1's #1 finding (generation-bound) and its sharpest failure (`sims-100` "never grokked" under vanilla MCTS). Gumbel *provably* improves the policy even at tiny sim budgets (n=2..16) — so it should let self-play run far fewer sims per move yet still emit strong, low-variance targets, buying generation speed without the target-quality collapse that floored sims-100. The risk: the completed-Q target is a different target shape than visit-count policy; it could interact badly with our short-game distribution or need m/c-tuning to beat plain PUCT at our sim counts.

**Expected Δelo signature:** *Confirm* = at LOW sims (e.g. 100), Gumbel's Δelo/hr clears C0's and clears vanilla-MCTS-at-100 by a wide margin — strong targets cheap = the generation-bound win. *Refute* = Δelo ≈ vanilla PUCT at matched sims (the completed-Q target bought nothing at 9×9 scale) or instability from the target-shape change.

**Config delta vs C0:** `--n-simulations 100 --gumbel-root` (the point is cheap-sims-that-still-train; an A/B at 200 is a secondary cell).

### v3-playoutcap
**Lever:** `--playout-cap-frac 0.25 --playout-cap-fast-sims 50` — most moves run a small budget and are NOT recorded; ~25% run the full budget and ARE the training targets. **Source:** KataGo, Wu (2019); inherited by KataGomo (the engine we surveyed @ Gomocup 2254).

**Hypothesis:** Concentrate expensive search where it actually trains the net. Generation-bound says wall-clock is dominated by sims/move; spending the full budget on only ¼ of moves (and a cheap budget elsewhere just to keep the game progressing) should multiply games/wall at near-constant target quality — a different route to the same "more, fresher self-play per wall" that cheap-sims chases, but without weakening the *recorded* targets. The risk: the cheap moves still shape the game trajectory, so low-quality intermediate play could bias the distribution the recorded positions come from.

**Expected Δelo signature:** *Confirm* = Δelo/hr above C0 — same-or-better climb at materially less wall, because the recorded targets stay full-strength while the game advances cheaply. *Refute* = a shallower climb (cheap intermediate moves degraded the trajectory the targets are drawn from) or no wall win (the fast moves weren't cheap enough to matter).

**Config delta vs C0:** `--playout-cap-frac 0.25 --playout-cap-fast-sims 50` (full budget stays C0's `--n-simulations 200`).

### v3-forcedplayout
**Lever:** `--forced-playout-k 2.0` — force ≥ `ceil(sqrt(k·P(a)·N))` visits to each root child, then prune the forced visits back out of the policy target. **Source:** KataGo, Wu (2019).

**Hypothesis:** v1 found exploration beats compute for the ceiling, but "more sims" is the expensive way to explore. Forced playouts buy *root exploration* of promising-by-prior moves that PUCT would starve — without raising the sim budget — and target-pruning keeps the forced visits from polluting the trained policy. So: the ceiling benefit of exploration at the cost profile of a normal sim budget. The risk: at our small sim counts the forced minimums could *crowd out* the search's own signal, and the pruning rule could over/under-correct the target.

**Expected Δelo signature:** *Confirm* = a higher ceiling than C0 (echoing open-div4/temp-16's exploration win) at C0's wall cost — exploration without the sim tax. *Refute* = Δelo ≈ C0 (forcing didn't add useful exploration at 9×9) or instability from target-pruning artifacts.

**Config delta vs C0:** `--n-simulations 200 --forced-playout-k 2.0`.

### v3-swa
**Lever:** `--swa-window K` — publish self-play generator weights as the flat average of the last K checkpoints, instead of EMA/live. **Source:** Stochastic Weight Averaging / Leela Chess Zero weight-averaging practice.

**Hypothesis:** v1's `ema-099` was a *floor* — the exponential moving average LAGGED the learner on a fast climb, generating from a staler/weaker policy. SWA is the targeted fix: a flat tail-average smooths target generation (the stability EMA was reaching for) *without* the unbounded lag of exponential decay, since old weights fall out of the window entirely. So it should recover EMA's stability benefit on the climb without the lag penalty that sank ema-099. The risk: on a *fast* climb even a short flat window still mixes in too-old weights and lags; or the smoothing simply isn't worth anything when fresh-start gradients are large and directionally consistent.

**Expected Δelo signature:** *Confirm* = lower chunk-to-chunk Δelo variance than C0 AND a climb that stays at-or-above C0 (beating ema-099's floor) — stability without lag. *Refute* = lags like ema-099 (window too wide / climb too fast) or no variance reduction (the live policy was already fine).

**Config delta vs C0:** `--n-simulations 200 --swa-window 5` (tune K; contrast directly with v1's `--ema-tau 0.99`).

### v3-auxhead (Class-C, design-only — see `auxiliary-targets-design.md`)
**Lever:** an opponent-reply auxiliary policy head (recommended), opt-in via an aux-loss weight; predicts the opponent's next-ply policy for extra gradient per position. **Source:** KataGo auxiliary targets, adapted to 9×9.

**Hypothesis:** Attacks the laptop-scale thin-signal problem ([[az-at-scale-vs-laptop]]): short gomoku games yield few near-opening positions, so each scarce position should teach the net more. An opponent-reply head squeezes a second supervised signal from data we already generate. *This is a model-architecture change (Class C)* — design first, user sign-off before any model.py edit. Card finalized from the design doc.

**Expected Δelo signature:** *Confirm* = steeper Δelo/hr than C0 at equal generation (more signal per position = faster learning from scarce data), aux head dropped at inference so self-play/eval cost is unchanged. *Refute* = aux loss distracts the shared tower (policy/value regress) or the extra signal is redundant with the value target on short games.

**Config delta vs C0:** TBD from the design doc (e.g. `--aux-opponent-reply-weight 0.15`, default 0.0 = off).

---

## Title cards

### C0-baseline
**Lever:** control — sims 200, 64 games, 400 steps, buf 100k, temp 8, lr 1e-3.

**Hypothesis:** The reference climb. Every other card is read as a delta against
C0's anchored-elo trajectory. No claim of its own; it defines "did the lever
help or hurt the *rate*."

**Expected Δelo signature:** A monotone climb that crosses elo ≥ 800
(beat-heuristic) somewhere near epoch ~140 — by construction the milestone is
calibrated to roughly this recipe. Sets the per-chunk Δelo baseline the queue
sorts against.

**Config delta vs C0:** none (`--n-simulations 200`).

---

### sims-400
**Lever:** `n_simulations 400` (vs 200) — stronger policy targets, slower gen.

**Hypothesis:** Deeper MCTS per move yields sharper, lower-noise policy/value
targets, which could steepen the early climb — *if* target quality, not game
volume, is the binding constraint on the fresh-start ascent. The cost is ~2×
slower generation per game, so within a fixed 10-epoch (not fixed-wall) chunk it
trains on the same number of games but better-labeled ones. The risk: at low
model strength the extra sims mostly refine an already-cheap-to-estimate policy,
buying little while the wall-clock per chunk balloons.

**Expected Δelo signature:** *Confirm* = a higher Δelo per chunk than C0,
especially in the mid-climb (epochs 40–100) where target sharpness should matter
most; reaches 140 with a higher final elo. *Refute* = Δelo tracking C0 within
noise (target quality wasn't the bottleneck) while each chunk costs ~2× the wall.

**Config delta vs C0:** `--n-simulations 400`.

---

### sims-100
**Lever:** `n_simulations 100` (vs 200) — weaker targets, ~2× faster gen / more
games per wall.

**Hypothesis:** The LF1 lesson, inverted. LF1 showed that *fast generation
floods the trainer* — cheap gen pushed new-positions/epoch so high the trainer
fell behind and re-ground stale buffer. Here gen is cheap by design: does
cheaper/faster generation *win the climb* (more, fresher self-play per wall
overcomes weaker per-move targets), or does it just produce *noisier targets*
that slow the ascent? This is the early-climb test of the
volume-vs-quality tradeoff that the converged-model flywheel work couldn't isolate.

**Expected Δelo signature:** *Confirm (volume wins)* = Δelo per chunk meets or
beats C0 at a fraction of the wall — cheap gen is the efficiency frontier.
*Refute (noise dominates)* = a visibly shallower climb than C0, late or never
crossing elo 800, the weak targets capping reachable strength.

**Config delta vs C0:** `--n-simulations 100`.

---

### sgd-800
**Lever:** `training_steps 800` (vs 400) — more fit per epoch.

**Hypothesis:** delta-e run-2 found that **extra SGD bought nothing on a
*converged* model** — `lru,sgd=300` netted the identical chess-score to
`lru,sgd=100`, just played sharper, no net strength. But that was a net at its
optimum re-grinding a fixed curated slice. **Does more fit-per-epoch help on the
*climb***, where the net is far from convergence and each fresh batch carries
real un-learned signal? If the binding constraint early is "we under-fit the
data we generate," doubling SGD steps should steepen Δelo. The risk is the same
over-grinding seen at convergence reappears once the buffer is dominated by
stale self-play.

**Expected Δelo signature:** *Confirm* = steeper early Δelo than C0 (epochs
0–60), tapering as the net approaches the data's information limit. *Refute* =
the run-2 result generalizes — Δelo ≈ C0 within noise despite 2× the SGD,
extra grinding wasted even on the climb.

**Config delta vs C0:** `--n-simulations 200 --training-steps 800`.

---

### buf-30k
**Lever:** `replay_buffer_size 30k` (vs 100k) — faster turnover, fits recent
self-play harder.

**Hypothesis:** A smaller buffer turns over faster, so each epoch's SGD sees a
higher fraction of *recent* (stronger-policy) self-play and less stale
early-model garbage. On a fast climb where the policy is improving every few
epochs, weighting toward recent games could steepen Δelo — a fixed-buffer echo
of the recency-weighted curator finding (recency >> lru). The risk: 30k is small
enough to over-fit a narrow recent distribution and lose the diversity that
keeps targets honest, inducing instability.

**Expected Δelo signature:** *Confirm* = Δelo at or above C0 with the gap opening
mid-climb as recency compounds. *Refute* = higher per-chunk variance and/or a
shallower climb, recent-overfit eating the freshness gain.

**Config delta vs C0:** `--n-simulations 200 --replay-buffer-size 30000`.

---

### open-div4
**Lever:** `random_opening_moves 4` (the WL3 lever) — opening diversity,
better-balanced climb.

**Hypothesis:** WL3's diversity lever. Forcing 4 random opening moves spreads
self-play across a wider opening distribution, preventing the fresh net from
collapsing onto one or two dominant lines and over-fitting them. A
better-balanced game distribution should produce a steadier, less-degenerate
climb — and is the same mechanism that fixed delta-e run-2's near-50% decisive
rate (paired random openings made games decisive instead of replaying one line).
The risk: opening randomness adds early-game noise that slows the first chunks
before the diversity pays off.

**Expected Δelo signature:** *Confirm* = a smoother, more monotone climb than C0
with fewer regressions per chunk, and equal-or-better final elo. *Refute* =
slower early Δelo (noise tax) that the diversity never recoups by 140.

**Config delta vs C0:** `--n-simulations 200 --random-opening-moves 4`.

---

### ema-099
**Lever:** `ema_tau 0.99` (the WL2 lever) — EMA self-play weights, smoother
targets.

**Hypothesis:** WL2's stability lever. Generating self-play from an
exponential-moving-average of the weights (τ=0.99) instead of the live net gives
a slower-moving, lower-variance target-generation policy — the actor lags the
learner, so targets stop chasing every SGD wobble. This should reduce
self-play-target variance and smooth the climb, potentially raising Δelo by
keeping the net from training against its own noise. The risk on a *fast* climb:
the EMA lag could hold generation behind the learner's actual strength, slowing
how fast better targets become available.

**Expected Δelo signature:** *Confirm* = lower chunk-to-chunk Δelo variance than
C0 and a steady, regression-light climb. *Refute* = a visibly lagged climb —
Δelo tracking below C0 because the EMA actor keeps generating from a staler,
weaker policy than the learner has already reached.

**Config delta vs C0:** `--n-simulations 200 --ema-tau 0.99`.

---

### temp-16
**Lever:** `temperature_moves 16` (vs 8) — more opening exploration.

**Hypothesis:** Doubling the temperature-1 (sampling) window from 8 to 16 plies
keeps self-play exploratory deeper into the game before switching to greedy
selection. More exploration → broader state coverage and richer policy targets
in the early/mid game, which could steepen the climb the same way opening
diversity does — but via in-game sampling rather than forced random openings.
The risk: 16 plies of sampling injects weaker, higher-entropy moves into the
training data, diluting target quality and slowing convergence.

**Expected Δelo signature:** *Confirm* = Δelo at or above C0 with better coverage
showing as a steadier mid-climb and equal-or-higher final elo. *Refute* = a
shallower climb than C0 — the extra sampled-move noise outweighs the coverage
gain.

**Config delta vs C0:** `--n-simulations 200 --temperature-moves 16`.

---

<!-- STANDINGS:AUTO — delo_derby.py rewrites everything below this line -->

## Standings

_Last updated: 2026-05-24T15:31:05Z — 92 chunks run._

**Champion so far:** `open-div4` at 1385 elo (140/140 epochs).

| Rank | Idea | Epochs | Elo | Peak | Wall (min) | Δelo/hr | Beat heuristic? | Status |
|-----:|------|:------:|----:|-----:|-----------:|--------:|:---------------:|--------|
| 1 | open-div4 | 140/140 | 1385 | 1385 | 73.5 | 813 | ✓ | capped |
| 2 | temp-16 | 140/140 | 1240 | 1340 | 76.2 | 823 | ✓ | capped |
| 3 | sims-400 | 140/140 | 1094 | 1265 | 140.1 | 614 | ✓ | capped |
| 4 | sgd-800 | 140/140 | 1081 | 1284 | 105.0 | 874 | ✓ | capped |
| 5 | buf-30k | 140/140 | 751 | 908 | 77.9 | 488 | ✓ | capped |
| 6 | C0-baseline | 60/140 | 567 | 567 | 27.8 | 384 |  | queued |
| 7 | sims-100 | 110/140 | 389 | 389 | 33.2 | -8 |  | queued |
| 8 | ema-099 | 50/140 | 389 | 405 | 21.6 | 57 |  | queued |

_Δelo/hr = (peak elo − 389 floor) ÷ wall-hours-to-peak: real-strength gain per wall-clock hour, the north-star. Beat-heuristic ✓ = peak ≥ 800._
