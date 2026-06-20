# Plan: Fixing white-side (defense / second-player) weakness — 15×15 freestyle gomoku

> 🛑 **2026-06-20 CONCLUSION — STOP patching defense; the white "weakness" is the
> first-player-win THEOREM, not a net flaw. The fix is swap2 (#22), not a teacher.**
> A full day's investigation closed this out. (1) The hole is recipe-deep, not
> warm-start/capacity/value-head — the from-scratch 0.44M `wdl@0` reproduces the
> champion's white sweep to the game (white 0-20 @100ms). (2) Three policy-stamp
> teachers ALL flattened — sparse-VCF (`--defense-detect-frac`), and the dense
> conv block-teacher (`--defense-teacher-conv`) — white never left ~0-2/20 vs Rapfi.
> (3) The diagnostic (`scripts/diag_white_failuremode.py`) showed WHY: white blocks
> forced fours essentially perfectly (Tier-1 error 5.6%, the 1 miss already-lost),
> has initiative on **1 ply in 30 games**, and is forced into an unstoppable
> double-four in **28/30** games — it competently retreats to a forced loss, it does
> not blunder. (4) **The clincher:** Rapfi(1000ms) vs Rapfi(1000ms) from the same
> 4-stone openings → **white 1-9 (~10%), black 9-1.** Even the #1 engine playing
> ITSELF gets crushed as the second player. 15×15 freestyle is a proven first-player
> win; from an empty/random opening white is a (near-)lost role, so NO policy/value
> teacher can make it win — there is no error to fix. **The real fix is to remove the
> doomed role: swap2** (Gomocup's balancing protocol — place 3, then stay/swap/place-2;
> the player is never *forced* onto the lost side). That also fixes the yardstick
> (Rapfi is a swap2 engine). Next build = **#22 swap2** (3-stone placement + 3-way
> color-choice node + net learning to negotiate + swap2 eval). The `--defense-detect-frac`
> + `--defense-teacher-conv` levers are sound + tested + default-off-byte-identical;
> they're kept as evidence, not a path forward. Full chronology: `TRAINING_WIKI.md`
> 2026-06-20. Everything below this banner predates the conclusion.

> The #33 / #18 defense plan (copied here from `/tmp/defense_plan.md`, dated 2026-06-15). §1A is implemented by `scripts/panel_white_elo.py` (the pure-analysis white-side reader over the panel JSONL); §5 is the eval-cadence fit.

Date: 2026-06-15. Author: research subagent. Status: design only — **no games run**
(GPU is busy with the panel tournament). All claims are grounded in code; wiki
claims are flagged where I could not re-verify them against the live cross-table.

Jason's framing (issue #18): black is the forced-win side → **convert it (100% as
black)**; white can never *win* (strategy-stealing) → its only job is to **never
lose = force the draw (0% loss as white)**. So "defense" == white-side loss-rate,
and the whole problem is asymmetric by construction.

---

## Step A RESULTS (2026-06-15) — I0 (FPU) and H3 (search budget) are FALSIFIED vs a real engine

Step A (the eval-only cheap branch) ran against the one reliable real attacker we
can head-to-head cleanly, **zetor17** (champion = `g15_128x10_bigbuf_eval502.pt`,
white = defending). The verdict is decisive: **both eval-side levers change NOTHING
vs a strong attacker → white-side defense is a genuine TRAINING gap.** The plan's
first *real* experiment is therefore **Step B (I1, the defense-teacher cell, #36)**;
I0 is demoted, I1 promoted.

**I0 — FPU-reduction (the UNCERTAIN §0 claim) is FALSIFIED for real-engine defense:**

| opponent | FPU | white result |
|---|---|---|
| `lookahead:depth=4` (weak searcher) | 0.0 | 88% (small residual loss-tail) |
| `lookahead:depth=4` (weak searcher) | 0.45 | **100%** (residual tail closed) |
| **zetor17** (real strong attacker) | 0.0 | **0–6 (100% loss)** |
| **zetor17** (real strong attacker) | 0.45 | **0–6 (100% loss)** |

The 9×9 wiki claim ("FPU c=0.45 *alone* drives white-loss → 0; the white weakness is
a search-prior miscalibration, not a training gap") **does NOT transfer to 15×15 real
engines.** FPU=0.45 closed only a small residual gap vs the *weak* depth-4 searcher
(88→100, a tail that was nearly closed already); vs the real attacker it is **0–6 at
both FPU=0.0 and FPU=0.45 — identical, 100% loss.** The §0 UNCERTAIN flag is resolved:
**the FPU eval-prior story is refuted.** It looked plausible because it worked on the
weak searcher; the real attacker falsified it.

**H3 — "search too shallow" (the search-budget hypothesis) is FALSIFIED:**

| opponent | sims | white result |
|---|---|---|
| zetor17 | 200 | **0–4 (0%)** |
| zetor17 | 800 | **0–4 (0%)** |

4× more search → **0% white at both.** H3 is **ruled out for real-engine defense.**

**The dissociation:** at *every* FPU and sims setting the champion is **perfect as
black** vs zetor17 (4–0 / 6–0) and **helpless as white** (0–4 / 0–6) — same net, same
search, same opponent, only the color flips. A weakness invariant to every search/eval
knob lives in the **weights**, not the search. **H1 (#18, teaching gap) and H2
(value-target asymmetry) STAND; H3 is RULED OUT.** Fix = relabeling, not a flag →
**Step B / I1 / #36** (escalate to I2, stamp the saving move, if value-only under-moves
the draw/loss boundary). Full synthesis: §15 of
[alphazero-lessons-15x15-gomoku.md](alphazero-lessons-15x15-gomoku.md). Tracked under
[#33](https://github.com/jasonyandell/gomoku/issues/33);
fix [#36](https://github.com/jasonyandell/gomoku/issues/36) /
[#18](https://github.com/jasonyandell/gomoku/issues/18).

---

## §1B.2 INSTRUMENT BUILT + first measurement (2026-06-16) — the probe CONFIRMS attacker-strength-gating; it needs a STRONGER attacker for I1/I2 headroom (#45 done → #49 next)

The §1B.2 forced-defense probe (Appendix item 3) is now **built and merged as #45**:
`gomoku/white_defense.py` + `scripts/white_defense_suite.py` + a versioned fixture
(`fixtures/white_defense_15x15_v1.json`, 80 white-to-move positions facing a live
black threat) + a `white_loss_rate` metric with Wilson CI + a `white_defense_tally`
gate primitive (drop-in for `sliding_gate.run_gate`'s `white_loss_fn`).

**Positive control PASSES** (the instrument discriminates defensive skill): champion
`eval502` `white_loss=0.0375` (3/80, CI [0.013, 0.105]) vs a random-init net `0.95`
(76/80, CI [0.878, 0.980]) — CIs strictly non-overlapping, attacker `lookahead:depth=2`,
sims=200, n=80/net. (random-init loses 54/54 `_three` positions → the spread is real,
not a fixture artifact.)

**The measurement does NOT contradict the training-gap thesis — it sharpens it.** This
probe uses a *weak* attacker (`lookahead:depth=2`), so the champion sits at its **floor**
(3.75% loss), fully consistent with Step A's depth-4 result (88–100% white success vs
the weak searcher). The brittleness is **attacker-strength-gated**: solid vs weak
searchers, **0–6 vs zetor17** (strong). The v1 fixture was mined from *weak-baseline*
games, so its threat distribution is the easy end — **it has no headroom to measure an
I1/I2 (#43) defense gain** (a better defender can only move ~3 losses). The champion's 3
losses concentrate in the `_four` provenance — the residual difficulty lives in the
strong-threat slice.

**Implication → #49:** the diagnostic instrument the defense race needs is a
**harder-attacker variant** — `lookahead:depth=3/4` or **champion-as-attacker** (our
strongest no-wine attacker, the #37 "strong attacker" side) over **strong-attacker-derived**
threat positions, so the champion posts a measurably-above-floor white_loss and a #43
improvement is resolvable within CI. #45 v1 is the validated *primitive* (CI + gate seam +
play-from-position); #49 makes it *diagnostic* for I1/I2.

---

## §1B.2 → Step B UPDATE (2026-06-16) — the value-only teacher (#36/#42) was RUN and FAILED; I1→I2 escalation has FIRED → fix is now #43 (stamp the saving move on the policy)

The conditional in Step B / §15 ("escalate to I2 *if* value-only under-moves the draw/loss boundary") has resolved: **it fired.**

- **#36/#42 (value-only `--defense-teacher`, relabel proven-lost white positions `z=−1`) FAILED.** The hard value relabel **with no fire-rate bound saturates the value head** — `vl 0.16→0.06` with plies HELD (40–50), the canonical value-poisoning death-tell (see [sliding-derby-measured-outcomes-design-v2.md](sliding-derby-measured-outcomes-design-v2.md) §CHANGE 1). The over-pessimized value head then **contradicts the untouched attacking policy → the shared trunk corrupts → policy degrades.** The gentler #42 variant still bled. **#42 is closed.**
- **Structural verdict:** value-only defense teaching is **structurally wrong for "never lose as white."** It teaches *"you were already lost"* (the `z=−1` relabel) — it does **not** teach *how* to defend, i.e. it never sharpens the **draw/loss boundary** where the white job actually lives. Stamping value alone, against an untouched attacking policy, poisons the value head before it can move that boundary.
- **Committed fix = #43 (I2): stamp the unique SAVING MOVE on the POLICY head** — one-hot the unique refutation when the opponent has a proven forced win *only if* white plays the wrong move (the literal [#18](https://github.com/jasonyandell/gomoku/issues/18) "teach the move it should have played" for the second player). This is now the **active** white-defense arm, not a contingency. Cap the refutation solve hard; fall through to no-op when ambiguous (sound: only ever skips).
- **Instrument status for #43:** the #45 white-defense suite is the validated *primitive* but its v1 fixture (weak depth-2 attacker) puts the champion at its **floor** (3.75% white_loss) → no headroom to resolve a #43 gain. **#49** (strong-attacker / champion-as-attacker variant over strong-threat positions) is the diagnostic the #43 race needs.

---

## §1B.2 → STRONG-ATTACKER MEASUREMENT ARRIVED (2026-06-18) — real Rapfi-NNUE confirms the gap is the WHOLE shortfall: champion white 0/12 vs #1 engine

The "harder attacker" §1B.2/#49 has been reaching for is now **online for real**: native
Rapfi-NNUE (the engine that won Gomocup 2024+2025) is a registered no-wine panel anchor —
the #28 "weightless yardstick" bug is fixed (it now loads NNUE and searches to its time
budget). See [external-engine-baselines.md](external-engine-baselines.md) § *Rapfi-NNUE
NATIVE ANCHOR ONLINE* and [reliable-eval-set.md](reliable-eval-set.md). First champion
contact (full provenance in `TRAINING_WIKI.md` 2026-06-18):

**Champion `eval502` vs Rapfi, 5s/move single-thread, n=24, #30 panel harness → 5W-19L-0D (20.8%).**
The color split IS the finding:
- **black (attacking): 5-7-0 = 42%** — competitive with the #1 engine even at a ~10×
  compute disadvantage (Rapfi 5s/move vs our net ~0.4s @ sims=400).
- **white (defending): 0-12-0 = 0%** — swept. **The entire strength shortfall is the
  white-side defense gap.**

This is the strong-attacker reading #45 v1 could not produce (it sat at the floor, 77/80
vs depth-2). Rapfi punishes the defensive lapse **every** game → the cleanest evidence yet
for the **[#37](https://github.com/jasonyandell/gomoku/issues/37) hypothesis** (white-side
defense weakness is the engine of degeneration), and a far sharper probe than self-play.
**Implications:**
- **#49 has a live strong attacker now** — Rapfi-as-attacker over white-to-defend
  positions is the diagnostic the #43 race needs; the floor problem is solved (0/12 is the
  opposite of a floor).
- **#43 (stamp the saving move on the policy) is validated as the target** — value-only
  teaching (#36/#42, above) was structurally wrong; the 0/12 says the *policy* never learns
  the refutation against a real attacker.
- **Caveat (don't over-read the 21%):** not compute-matched; freestyle is first-player-
  favored; n=24 (white 0/12 is conclusive for the gap, the overall rate has a wide CI).
  Absolute Gomocup-Elo calibration still pending (#35/#30).

**TC-tier calibration (2026-06-18, same day) — a CLIFF then a white-defense PLATEAU.**
Ran the champion across Rapfi thinking-time tiers via the new parallel `--jobs` path (#52,
8 workers / 18-core M5, 200 games in 7.6 min; `eval_vs_rapfi.py --jobs 8`, n=40/tier,
4-stone openings). Evidence + table: `TRAINING_WIKI.md` 2026-06-18, out
`sweep_logs/calib_champ_vs_rapfi_tctiers_20260618.jsonl`.

| Rapfi TC | win% | white W-L-D | black W-L-D |
|---|---|---|---|
| 10ms | 100% | 20-0-0 | 20-0-0 |
| 100ms | 27.5% | **0-20-0** | 11-9-0 |
| 250ms | 37.5% | 2-18-0 | 13-7-0 |
| 500ms | 27.5% | 2-18-0 | 9-11-0 |
| 1000ms | 27.5% | 3-17-0 | 8-12-0 |

Two readings: (1) **cliff not ramp** — 10ms is below Rapfi's useful search threshold (no time
→ raw NNUE move, swept 40-0); from 100ms the result PLATEAUS ~27% (250ms 37.5% is n=40 noise),
so 10× more Rapfi time barely moves it — it already exploits the one hole at 100ms. (2) **the
plateau is a white plateau** — black stays competitive at every real tier (40-65%, declining as
Rapfi deepens), white is pinned at the floor (0-15%, nowhere to fall). **The champion's entire
deficit vs the #1 engine is white-side defense, now confirmed across 5 independent measurements**
(the n=24 first-contact + these 4 real tiers = 160 games). This is the strongest mandate yet for
**#43**; real Rapfi is the before/after gate — re-run this exact calibration after the policy-stamp
and watch the white column move.

---

## §1B.2 → #43 (I2) LEVER IS BUILT (2026-06-18) — saving-move-on-policy lands; awaiting the live race

The I2 arm is now **code-complete and merged** (`Closes #43`, commit on `main`
2026-06-18); the live training race + the Rapfi white-column re-measurement is the remaining
(GPU, `needs-live-validation`) step. What shipped:

- **New solver primitive `vcf.vcf_refutations(board)`** — the piece the value-only teacher
  lacked. The defense teacher fires when the OPPONENT has a proven forced VCF win *as if to
  move*, but the recorded side **moves first** (one tempo), so there is usually a SAVING move.
  `vcf_refutations` enumerates the defender moves that, once played, leave the opponent with
  **no** forced VCF (near-stone candidate prune + per-move re-solve). **Sound:** a move is only
  reported when an explicit re-solve proves the win is gone (no false saving moves); complete
  within solver depth (an isolated move can neither block a four nor make one). Returns `[]` for
  a genuinely lost position (e.g. an open four / disjoint double-four).
- **`self_play._apply_defense_teacher_policy`** stamps a soft (uniform) policy target over the
  saving move(s) and **leaves the value at the natural game outcome** — a *pure* policy lever.
  A truly-lost position (no refutation) is left **entirely untouched** (no value crush), so this
  is cleanly separable from the failed #36/#42 value path. Mode is a process-wide switch
  `_DEFENSE_POLICY_MODE` (mirrors the #42 knob pattern); same per-game FRACTION budget applies.
- **Worker flag `--defense-teacher-policy`** (implies `--defense-teacher`). Default OFF =
  byte-identical self-play.
- **Tests:** `tests/test_defense_teacher_policy.py` — refutation soundness (incl. a 300-position
  fuzz: every stamped move provably breaks the VCF), one-hot/soft stamp, value-untouched,
  truly-lost skip, quiet/already-fired gates, trajectory integration (policy vs value mode), and
  the budget cap. Full suite green.

**Why this should work where #36/#42 didn't:** the value-only teacher poisoned the value head
against an untouched attacking policy (shared-trunk contradiction, #41: `vl→0.06`, `pl→3.4`). The
I2 target is a *real, consistent* policy label (the move that refuses the forced four) with a
moderate value — the expected signature is `white_loss ↓` AND `pl` bounded-or-improving AND `vl`
healthy (~0.10–0.16). **The gate is the Rapfi TC-tier calibration above**: re-run `eval_vs_rapfi.py
--jobs 8` after a policy-stamp training slice and watch the white W-L-D column move off the floor.

## §1B.2 → #43 (I2) LIVE RACE RAN, then KILLED (2026-06-19) — lever SOUND, but the 1.5M warm-buffer DILUTION is the binding constraint; gen cost (#60) + buffer freshness are the unlock

The I2 policy-stamp lever was **run live** against the warm-started 128×10 champion
(cell `G15-defense-i2`, warm-start `g15_128x10_bigbuf_eval502.pt`, board 15, 1.5M
bit-packed buffer, wandb `zrjfwny2`). Resumed e585 → ran to **e1286**, then
**deliberately killed** — not because it broke, but because it cannot produce a
*readable* answer at this configuration's pace.

**The lever is HEALTHY (the positive finding).** Over ~700 epochs the I2 signature
held: `pl` plateaued **~1.19–1.22** (bounded — NOT the #36/#42 shared-trunk corruption
that ran `pl→3.4`), `vl` clean **~0.13–0.14** (NOT the value-only `vl→0.06` saturation),
`plies` stable ~30–47 (no fast-attack collapse). Stamping the saving move on the POLICY
head and leaving value at the natural outcome does exactly what #43 designed: it injects
hard defensive policy targets **without** poisoning the value head. **I2 is vindicated as
a training mechanism** — the §1B.2 "why this should work where #36/#42 didn't" prediction
held in production.

**But the signal is un-readable — the 1.5M WARM BUFFER drowns the stamps.** Fresh stamped
games accumulate at only ~**0.16–0.3 % of the buffer per hour** even at 16 generators
(~**1,100 games/hr** sustained), and only the *forced-loss* plies inside those games get a
saving-move stamp. The Rapfi white-column gate needs a non-trivial fraction of the buffer
to carry the new lesson before it can leave the 0/12 floor — so "wait longer" was never
going to resolve it. **Gen *rate* is not the lever; buffer *freshness* (stamp density vs
the warm-start attacker-biased mass) is.** Killed at e1286 (clean SIGTERM; checkpoint +
wandb preserved) rather than burn the GPU on an unreadable run.

**Two levers built this session attack exactly this:**
- **#69 (merged) — `run_sweep --n-workers N`**, a launch-time generator-count knob.
  Generation is CPU-bound on the python VCF refutation (trainer is MPS-bound ~6 % CPU), so
  generator count is the throughput knob. **16 generators ≈ the 18-core M5 ceiling**; a
  10-min A/B *looked* like 16 beat 12 by ~25 %, but **sustained** rates were within noise
  (~1,100 vs ~1,300 /hr) — a short/cold measurement window does not predict steady state
  (the same trap as the LEAN-fp16 perf benchmark; cf. `TRAINING_WIKI.md` § LF1). Generator
  count alone does not break the dilution.
- **#60 — refute only the budget-kept plies.** The policy teacher was paying the expensive
  `vcf_refutations` enumeration (~83 % of gen wall) on *every* firing ply, then
  `--defense-max-fraction 0.25` discarded ~75 %. The fix detects fire-candidates cheaply
  forward, then refutes **latest-ply-first only until the budget of stamps is filled** —
  deterministic-equivalence-proven to yield the identical kept-stamp set, with a measured
  **4× fewer refutation solves** (8 candidates → 2 at frac 0.25). Cuts the dominant gen
  cost and should kill the 90–137 s/game dense-board tail.

**Next frontier:** the I2 mechanism is sound — the open work is making its signal
*readable*. Pair #60 (cheaper, denser stamps per gen-second) with a **buffer-freshness
rethink** (smaller or recency-weighted buffer, or a higher stamp fraction, so the defensive
lesson is not diluted into the champion's 1.5M attacker-biased mass) before the next live
race. Gate unchanged: `eval_vs_rapfi.py --jobs 8`, watch the white W-L-D column leave 0/12.

---

## §1B.2 → ROOT-CAUSE REFRAME + SPARSE-BITE LEVER (2026-06-19) — the deficit is RECIPE-DEEP, the gen bottleneck is the PER-PLY SOLVE (not buffer size), and the first fix is a 10% exact-solver sample

A focused session (Jason + Claude, hobby day) produced three findings that reframe the whole arc and put a *simpler* fix in flight.

### Finding 1 — the white hole is RECIPE-DEEP (not warm-start, not capacity, not the value head)
The wiki's open probe was: does the warm-started champion's 0/12-white hole come from
warm-start baking in the 9×9 attacker bias, or is it intrinsic to the recipe? **Answer:
recipe-deep.** The autolab's **from-scratch** `15x15-wdl@0` (0.44M, 64×4, WDL value head,
NO warm-start, NO teacher; HF `jasonyandell/gomoku-9x9@15x15-wdl-15x15-wdl@0`) was measured
vs native Rapfi-NNUE the same way as the champion (sims=400, 4-stone balanced, seed 7,
`eval_vs_rapfi.py --jobs 8`):

| net | @100ms white | @100ms black | @1000ms white |
|---|---|---|---|
| `wdl@0` (0.44M, from-scratch) | **0-20-0** | 11-9 (55%) | 1-19-0 |
| champion (3.3M, warm-started) | **0-20-0** | 11-9 | 3-17 |

The from-scratch run reproduces the champion's white sweep **to the game** (0-20 white,
11-9 black @100ms is byte-identical). A 7.5× smaller, from-scratch, different-value-head
net shows the identical hole ⇒ **warm-start, capacity, and the WDL head are all exonerated.**
The color **asymmetry is the clean signal** — this tiny net attacks at 55% vs the world #1
yet cannot defend at all. Corroborated by the contrast that the *same recipe* defends nearly
perfectly on 9×9 (the 9×9 champion was 43-3-74 vs Rapfi ⇒ white-loss ≤5%): the hole **scales
with board size** — classic fast-attack collapse wearing a white-defense mask. **The fix must
change the training DATA**, which is exactly what the #43 teacher does. Artifact:
`sweep_logs/probe_wdl0_vs_rapfi_n40.jsonl`.

### Finding 2 — the #43 "drowning" ROOT CAUSE is the per-ply solve on the gen hot path, NOT buffer dilution
A clean 2×2 of 180 s smokes (board 15, 8–16 workers) + a steady-state profiler (`wdl@0`,
policy teacher, caps 800/7) isolated the cost:

| net start | teacher | games/180s | gen rate |
|---|---|---|---|
| from-scratch | OFF | 1288 | 7.2 g/s |
| from-scratch | ON | 40 | 0.22 g/s |
| warm `wdl@0` | OFF | **1864** | **10.4 g/s** |
| warm `wdl@0` | ON | 24 | 0.13 g/s |

The teacher slows gen **~32× from scratch, ~78× on the mature net.** Profiler breakdown
(16 games, steady-state): **~7.1 s/game = 94 % of wall**, of which **~21 detection solves/game
@ ~180 ms** (`solve_vcf` on every opponent-four-threat ply + a *second* `solve_vcf` for the
StM-own-win guard) = 3.8 s/game, and refutation re-solves = 3.3 s/game. **So the #43 race
didn't drown because the buffer was big — it drowned because the per-ply VCF solve made gen
crawl** (the 1.5M buffer was the *symptom*; even a tiny fresh buffer can't be kept dense at
0.13 g/s). (Note: the 8-worker smoke read ~59 s/game vs the profiler's clean 7 s/game —
multi-process cold-window contention inflation, the same short-window trap as LF1; trust the
profiler.) Jason's instinct — "it drowned because gen was super slow" — was exactly right.

### Finding 3 (the fix in flight) — SPARSE-BITE: invoke the exact solver on only 10% of danger plies
AlphaZero distills a defensive lesson over many epochs from a **present** signal; it does not
need every forced loss stamped. So the simplest unlock is to **sample** the expensive solve:
new flag **`--defense-detect-frac F`** (self_play `_DEFENSE_DETECT_FRAC`, default 1.0 =
byte-identical) gates `solve_vcf` in `_defense_detect_candidate` to a fraction `F` of
four-threat plies — the cheap `has_four_threat` prescan still runs every ply, only the
expensive detection (and the guard + refute that follow) is sampled. Stamps stay **EXACT**
(no new correctness surface). Profiler @ `F=0.1`: detection solves **21 → 2.2/game**, teacher
cost **7.08 → 1.68 s/game** (~10× cut) ⇒ the *inline* teacher is performant again, no
off-path machinery required. **Density math:** 10% stamping in a fresh 150k buffer is
**~1000× denser** than the #43 race that drowned in a 1.5M warm buffer — i.e. plausibly
already past the density threshold, at 1/10 the cost. It's a **dial**, not a rewrite: too
sparse → turn it to 0.3.

**LIVE cell `G15-wdl-defense`** (`scripts/run_sweep.py`) = `G15-wdl` (the from-scratch WDL
champion — the 0/20 control above) **+ ONE lever** (`--defense-teacher-policy`, sparse
`--defense-detect-frac 0.1`, `--defense-max-fraction 0.25`, caps 800/7) into a **small fresh
150k buffer**, warm-started `--resume wdl@0`, 16 workers. Warm start = clean ~37-ply games
(fast gen + real forced-loss positions to teach), fresh small buffer = no #43 dilution.
**Gate:** `eval_vs_rapfi.py --jobs 8` on a matured checkpoint — watch the white W-L-D column
leave 0/20. Restartable (`--resume latest.pt`).

### Deferred: the OFF-PATH relabeler (designed, parked) and the reanalyze seam
The original plan was a separate relabel-worker process (gen teacher-off-fast → a pool of
solver processes stamp records → trainer; modeled on `eval_worker.py`, reusing
`_defense_detect_candidate`/`_defense_refute_stamp` — records store `planes`, so the solver
runs directly, no move-replay needed). Profiling killed it as the *first* move: the solve is
~7 s/game **wherever it runs**, so off-path doesn't make it faster — it only *parallelizes*
it, and inline-with-16-workers already uses every core, so both cap at the same ~2 g/s of
stamped games. (The shipped in-trainer `reanalyze.py`/derby-fm9 seam can't be reused either:
to avoid starving SGD you'd bound it so hard the stamp rate re-dilutes — the #43 failure.)
The off-path design's *real* value is keeping the trainer fed at full gen rate while a skim
of records gets stamped — worth building **if** sparse-bite's density proves insufficient.

### Rung two (idea for later): a GPU-NATIVE cheap defensive "bite"
The exact solver is pure-Python CPU (the CPU↔MPS bounce + 180 ms/solve). A **conv-based
threat detector** — find the opponent's open-fours / double-threats as kernels over the
board planes, in torch **on MPS**, derive the blocking square(s), stamp them on the policy —
would be **dense, every-ply, no CPU/GPU jump, no tree search.** It's *shallow* (catches the
immediate threat, not a deep VCF line), but shallow defense is exactly what white lacks
(0-20). The endgame is **layered**: cheap-dense-shallow (GPU, every ply) + rare-deep-exact
(solver, sparse) → the net learns "always block the obvious thing" *and* occasionally "here's
the deep save." Real build + a new correctness surface (a wrong block teaches the wrong move),
so deferred behind the sparse-bite result. Filed as a candidate lever.

---

## 0. What the code actually does today (ground truth, file:line)

These are the load-bearing facts the plan is built on. All verified by reading source.

- **MCTS is fully color-symmetric.** `gomoku/mcts.py` — same `n_simulations`,
  same Dirichlet noise (`_add_dirichlet_noise`, mcts.py:184), same FPU
  (`fpu_reduction_c`, mcts.py:103-107) for both sides. There is **no
  "more-sims-when-behind", no color-conditioned search.** FPU is **eval-only**
  (`mcts_picker(..., fpu_reduction_c=)`, eval.py:159/199-203; self-play MCTSGame
  is built with the 0.0 default).
- **The training loss has NO per-color or per-outcome weighting.**
  `gomoku/train.py`: combined loss is `pl + value_weight*vl (+aux +ownership +l2)`
  (train.py:336). `side` (0=black,1=white) is **diagnostic only** — it splits
  `train/value_mse/side_{0,1}` under `no_grad` (train.py:399-414) but never scales
  a gradient. White losing-positions get exactly the same weight as everything else.
- **Replay buffer sampling is uniform** (replay_buffer.py:173-178), with an
  optional **recency** curator (`--buffer-recency-frac/-window`, train.py:676-683)
  — **no curation by color or by outcome.**
- **Value target is a scalar MSE by default**; `--value-head {scalar|wdl|hlgauss}`
  exists (train.py:265-334). WDL target = `[relu(z),1-|z|,relu(-z)]` (train.py:177-196).
- **A defense teacher ALREADY EXISTS but is value-only and VCF-only.**
  `_apply_defense_teacher` (self_play.py:292-363): swaps planes so the opponent is
  attacker, runs a cheap `has_four_threat` prescan (self_play.py:348), and only if
  the opponent has a **proven VCF (continuous-four) forced win** does it relabel
  the recorded position's value to `z=-1.0` ("you were already lost; defend
  earlier"). **It never touches the policy** (defense is non-unique). Wired:
  `--defense-teacher` (selfplay_worker.py:185, 731, 752). Off by default →
  byte-identical baseline.
- **Value-shaping levers all live on the GENERATION side** (self_play.py), applied
  to `z` *before* it reaches the trainer: `--value-discount` (γ^plies, self_play.py:166-186),
  `--draw-value` (draw-contempt, self_play.py:176-183), `--vcf-teacher` /
  `--vct-teacher` (one-hot policy + mate-discounted +1 value, self_play.py:189-289),
  `--contempt-p` (search-contempt position-distribution, self_play.py:867-871).
- **Openings:** `--random-opening-moves N` exists (self_play.py:655-680); eval can
  share a swap-2-style balanced opening per color-pair
  (`play_match_pickers(random_opening_moves=)`, eval.py:419-440).
- **The panel arena records per-color data but reports only ONE aggregate Elo.**
  `scripts/panel_tournament.py`: every `PairRecord` carries
  `black=[w,l,d]`/`white=[w,l,d]` in the JSONL (panel_tournament.py:184-207), but
  the Bradley-Terry fit + calibration + printed ranking are **aggregate only**
  (per the panel-script reader; the white split is on disk, unused). The live file
  `sweep_runs/panel_tournament_results.jsonl` already has per-pair white arrays.
- `scripts/report_100pct.py` already computes the right *shape* of metric:
  `dist = Σ_baselines (1 − black_win_rate) + white_loss_rate`, color-split, over
  `{heuristic, lookahead2, lookahead4}`.

**Wiki claim I could NOT re-verify (flag as UNCERTAIN) — ❌ NOW REFUTED 2026-06-15:**
the research-board synthesis said FPU-reduction c=0.45 *alone* drives white-loss from
~20-30% → 0% across lookahead:depth=4/6/8, i.e. "the white weakness is a search-prior
miscalibration, not a training gap." That is an eval-only knob and was the strongest
single claim in the corpus. **Step A re-measured it at 15×15 vs the real attacker
zetor17 and FALSIFIED it:** FPU=0.0 and FPU=0.45 both give **0–6 white (100% loss)**
— identical. FPU closed only a small residual tail vs the *weak* depth-4 lookahead
(88→100). The 9×9 claim **does NOT transfer to 15×15 real-engine defense.** The white
weakness is a **training gap, not a search-prior miscalibration.** (See "Step A
RESULTS" at the top.)

---

## 1. CHARACTERIZE — quantify the white-side collapse

Goal: one number per (net, opponent, color) so a defense gain is *visible and
attributable*. Two layers: (A) reuse the live panel cross-table; (B) cheap targeted probes.

### 1A. Consume the panel cross-table — add a WHITE-SIDE Elo column

The data is already in the JSONL; we are not blocked on the GPU. Build a
**pure-analysis** reader (`scripts/panel_white_elo.py`, code-only, no games):

- Parse `sweep_runs/panel_tournament_results.jsonl`. For each net N and opponent O,
  compute three score-rates from the per-color arrays:
  - `wr_black(N,O) = (black_w + 0.5*black_d) / (black games)`
  - `wr_white(N,O) = (white_w + 0.5*white_d) / (white games)`  ← **the defense metric**
  - `white_loss_rate(N,O) = white_l / (white games)`  ← Jason's "0% loss as white"
- Run **two** Bradley-Terry fits over the pairwise scores — one on the black-side
  scores, one on the white-side scores — and calibrate both to the same engine
  anchors (reuse panel's existing anchor affine fit). Output per net:
  `black_elo`, `white_elo`, and **`elo_gap = black_elo − white_elo`** (the
  defense deficit in Elo). This is the headline number.
- Also print the raw `white_loss_rate` cross-row vs the calibrated engines
  (embryo26 ~2402, yixin18 ~2310, pela23, zetor17, eulring16) + heuristic, because
  a rate is more legible than Elo for "never lose as white."

**Done-definition for characterization:** a table with one row per net showing
`black_elo / white_elo / elo_gap / white_loss_rate-vs-each-engine`. The current
champion's `elo_gap` and worst-opponent `white_loss_rate` become the baseline to beat.

**Caveat (small-n):** panel uses `--n-games 8` (4 per color) per pairing → white
rates have ~±25% noise per cell. Treat single cells as hints; trust the BT-pooled
`white_elo` and the aggregate-over-engines `white_loss_rate`. For a verdict on a
candidate, bump that pairing to 40-80 games (still cheap vs a training slice).

### 1B. Cheap targeted probes (code-only, CPU, run anytime — GPU-free)

1. **Depth-N loss-tail, color-split** (the #18 metric). `play_match_parallel`
   already returns `white_l` (eval.py:564-620). Run champion vs
   `lookahead:depth=4` and `:depth=6`, n=200, and read `white_l / white games`.
   This is the canonical "tactical loss-tail" and is **CPU-only** (lookahead is
   torch-free, baselines.py:510). Run it now; it does not touch the GPU panel.
2. **Forced-defense probe set.** Build ~200 positions where white is *one tempo
   from losing*: take self-play/buffer positions, run the existing VCF solver with
   planes swapped (exactly `_apply_defense_teacher`'s swap, self_play.py:341-343)
   to find positions where the opponent has a proven forced win **iff white plays
   the wrong move**, i.e. there exists a unique refutation. Score: does the net's
   greedy policy (argmax, 0 noise) find the saving move? Reports a clean
   **"defensive-accuracy %"** decoupled from full-game noise. This is the probe
   that most directly measures the thing #18's defense-arm is trying to teach.
3. **FPU re-confirm sweep (eval-only).** Champion vs lookahead:4/6 and vs one real
   engine, white-side, at `fpu_reduction_c ∈ {0.0, 0.2, 0.45}` (eval.py:199-203).
   If white_loss collapses with FPU as the 9×9 wiki claims, **that reorders
   everything** (see §3, I0). Cheap, CPU, no retrain.

---

## 2. ROOT-CAUSE HYPOTHESES (tied to evidence; speculation flagged)

- **H1 — Teaching gap, not capacity (EVIDENCE-BACKED, #18).** A lost self-play
  game enters the buffer labeled only `z=-1` for the whole trajectory; it never
  says *which move* should have been played. The net is never taught the saving
  move. VCF/defense teachers exist precisely to stamp ground truth. This is the
  project's stated root cause and the design center of #18. **Strongest hypothesis.**
- **H2 — Value-target asymmetry at convergence (EVIDENCE-BACKED, mechanism).**
  At self-play convergence `selfplay/white_wins → 0` (wiki/TRAINING_WIKI). White
  positions are overwhelmingly labeled losses/draws; the value head sees few
  white *wins* and little gradient pressure to distinguish "drawable" from "lost"
  white positions. Uniform sampling + no per-color weight (train.py, verified)
  means the rare *recoverable* white position is drowned. Defense teacher partly
  addresses this (stamps clear losses) but **does nothing for the
  draw-vs-loss boundary**, which is exactly where "never lose as white" lives.
- **H3 — Search too shallow to see the opponent's threat. ❌ RULED OUT 2026-06-15.**
  Was: eval at 100-200 sims can miss a forcing line a depth-4 alpha-beta calculates
  exactly (#18's "a sophomore-grade searcher out-CALCULATES our net"). **Step A
  falsified it for real-engine defense:** vs zetor17, sims=200 → **0–4 white (0%)**
  and sims=800 → **0–4 white (0%)** — 4× more search changes nothing. The UNCERTAIN
  §0 FPU claim (the eval-prior compensator) is *also* refuted (FPU=0.0 and 0.45 both
  0–6). H3 does **not** dominate and the fix is **not** a flag → the gap is in the
  weights (H1/H2). See "Step A RESULTS" at the top.
- **H4 — Opening-book / first-mover bias (SPECULATION).** Self-play converges to a
  narrow attack opening; white only ever defends *that* line, so it is brittle on
  the wider openings real engines and lookahead play. `--random-opening-moves` and
  eval's swap-2 openings test this. Wider at 15×15, so likely *worse* than 9×9.
- **H5 — Covariate shift from the closed self-play loop (EVIDENCE-BACKED, frame).**
  The net only trains on positions it itself reaches; engine/heuristic opponents
  push it OOD, and white (reacting to a *foreign* attack) is the more-OOD side.
  Explains why white-loss concentrates vs *external* opponents, not in self-play.
- **RULED OUT — do not retry (from #18 + wiki):** (a) aux opponent-reply head —
  raced in v4/v5, did not beat bare VCF; (b) train-vs-baseline /
  `--opponent-mix-random` — value-loss collapsed to 0.04 (defense-blindness),
  making the opponent stronger breaks training; (c) eval-VCF-overlay *alone* —
  boosts black but tanks white-loss to ~35% (trades safe draws for losing
  attacks). Defense must come from **relabeling**, not from a harder opponent.

---

## 3. INTERVENTIONS (ranked by leverage ÷ cost)

Ranked best-first. "Cost" = engineering + GPU. Every training lever already has a
flag → most are config-only cells.

**I0 — FPU-reduction as the white fix (eval-only). ❌ FALSIFIED 2026-06-15 — DEMOTED.**
- ~~DO FIRST.~~ **Done (Step A); the risk fired.** `--fpu-reduction-c 0.45` at eval
  (eval.py:199-203), no retrain. The 9×9 wiki claim **did not transfer**: vs the real
  attacker zetor17, FPU=0.0 → **0–6 white (100% loss)** and FPU=0.45 → **0–6 white
  (100% loss)** — *identical*, no change (see "Step A RESULTS" at the top). It closed
  only a small residual tail vs the *weak* depth-4 lookahead (88→100). **Defense is
  NOT an eval-config problem; it is a training gap.** Do not re-chase FPU for
  real-engine defense. (Eval compensators may still help *arena* play marginally — I5
  — but that is orthogonal to the net's intrinsic defense.)

**I1 — Defense teacher + VCT attack (the #18 recipe). ⭐ PROMOTED — NOW THE FIRST
REAL EXPERIMENT (Step B, #36). HIGHEST training leverage.**
- Changes: turn on `--defense-teacher` (already wired, self_play.py:292-363) so
  proven-lost white positions get `z=-1`, paired with `--vct-teacher` (forced-three
  attack, self_play.py:239-289) on the current champion base
  (vcf + global-pool + value-discount 0.98).
- Expected: sharper value boundary on forcing lines → lower white tactical
  loss-tail vs lookahead:4/6 (the #18 acceptance metric).
- Cost: one training slice (`run_sweep --max-wall-secs N --final-eval`). VCT teacher
  caps are deliberately tiny (depth 4 / 800 nodes, self_play.py:78-79) so gen does
  not stall — verified that bug history (derby-b6r) already forced this.
- Risk: defense teacher is **value-only**; it teaches "you were lost" but not the
  saving move, so it sharpens the value head without fixing policy. May only move
  the *clearly-lost* tail, not the draw/loss boundary. Compose, don't expect a silver bullet.

**I2 — Defense teacher upgrade: stamp the SAVING move (policy refutation). NEW CODE.**
- Changes: extend `_apply_defense_teacher` so that when the opponent has a proven
  forced win *only if* white plays move X (i.e. there exists a defensive move that
  refutes it), stamp a **one-hot policy** on the refutation, not just `z=-1`. This
  is the missing half of H1 for the second player. Requires the solver to return a
  refutation move (search each defensive reply; if exactly one avoids a proven loss,
  that's the label).
- Expected: directly teaches the never-lose move → the highest-ceiling training fix
  for white. This is the literal "teach the move it SHOULD have played" from #18.
- Cost: solver/teacher code (~moderate) + fuzz gate (mirror derby-y8r) + a slice.
  More expensive than I1 but strictly higher ceiling. File as the #18 defense-arm child.
- Risk: refutation can be non-unique or expensive to prove → cap hard, fall through
  to value-only when ambiguous (sound: only ever skips).

**I3 — Loss-side / white upweighting in the loss (cheap NEW CODE, config after).**
- Changes: add a per-sample weight in train_step keyed on `side==1 & z<0` (the
  scaffolding exists — `side` already flows to train.py:399-414; today it is
  no_grad-only). Upweight recoverable white positions in the value loss.
- Expected: counters H2 (white-loss drowned by uniform sampling) without changing
  the opponent (avoids the ruled-out trap).
- Cost: ~30 LOC + one config cell to sweep the weight. Byte-identical when weight=1.
- Risk: crude (color is a weak proxy for "recoverable"); could overfit the value
  head to pessimism. Pair with the probe in §1B.2 to watch defensive-accuracy.

**I4 — Balanced / curriculum openings (config-only).**
- Changes: `--random-opening-moves k` in self-play (self_play.py:655) so white
  defends a *distribution* of openings, not just the converged attack line.
- Expected: addresses H4/H5 (OOD brittleness vs external openings) — likely more
  impactful at 15×15's wider board than at 9×9.
- Cost: config cell. Risk: dilutes on-policy signal; small k (2-4) recommended.

**I5 — Eval-side compensators stack (eval-only): tree-reuse + proven-prop.**
- Changes: `reuse_tree=True` + `proven_prop=True` (eval.py:164/204-219) on top of
  I0's FPU. Wiki: this stack pushed white-loss to ~0 vs lookahead:6.
- Expected/Risk: cheap, but it improves *our arena strength*, not the *net's*
  intrinsic defense — fine for "never lose as white *in matches*," orthogonal to training.

**Draw-contempt (`--draw-value`) is intentionally NOT ranked for defense**: it
pushes the net to *avoid* draws, i.e. toward decisive play — the opposite of
"white's job is to secure the draw." It is a black-side conversion lever; keep it
off the white arm.

---

## 4. FIRST EXPERIMENT (single highest-leverage, cheapest)

Two-step, because the cheapest thing is eval-only and might solve it outright.

**Step A — I0 — the FPU re-confirm. ✅ DONE 2026-06-15, ❌ FALSIFIED.**
- Spec: champion checkpoint, white-side, vs the real attacker zetor17 *and* the
  depth-4 lookahead, at `fpu_reduction_c ∈ {0.0, 0.45}` and (for H3) sims ∈
  {200, 800}.
- Result (the falsifying branch fired): vs zetor17, white stayed **0–6 (100% loss)
  at FPU=0.0 AND FPU=0.45**, and **0–4 (0%) at sims=200 AND sims=800**. FPU only
  closed a small residual tail vs the *weak* depth-4 lookahead (88→100). **The
  eval-prior story does NOT transfer; defense is a genuine training gap → proceed to
  Step B.** (Full tables: "Step A RESULTS" at the top; synthesis: §15 of the lessons
  page.)

**Step B (the FIRST REAL EXPERIMENT, #36): I1 — defense-teacher + VCT cell.**
- Knob: clone the reigning champion cell (`vcf + global-pool + value-discount 0.98`),
  flip on `--defense-teacher` and swap `--vcf-teacher`→`--vct-teacher`. **One lever
  family per cell** per derby rules (file as a `derby-idea` / #18 defense child).
- Slice: `run_sweep --max-wall-secs <one chunk> --final-eval` (time-capped,
  resumable, eval inside the bundle — the standard GPU-lane slice).
- Measure success via the arena: re-run the §1A white-Elo reader on the new
  checkpoint's panel rows; **success = `elo_gap` shrinks AND `white_loss_rate` vs
  lookahead:4/6 drops vs the champion base, with no black-Elo regression** (#18
  acceptance: better on loss-rate *and* H2H Δelo). Confirm on the §1B.2
  defensive-accuracy probe (should rise).
- Falsified if: white_loss_rate / defensive-accuracy do not move beyond small-n
  noise after a full chunk, or black-Elo regresses (the defense teacher is
  over-pessimizing the value head). Then escalate to I2 (stamp the saving move).

GPU discipline: Step A runs **today** (CPU, does not touch the live tournament).
Step B waits for a free GPU lane / the panel to finish — do not barge in.

---

## 5. EVAL-CADENCE FIT — non-blocking "checkpoint → arena every ~100 epochs",
   with WHITE-side Elo reported separately

The whole plan is worthless if a white gain is invisible in the dashboard, so the
**single most important deliverable is wiring white-side Elo into the cadence loop.**

- **The instrument (§1A) is the fix**: `scripts/panel_white_elo.py` reads the panel
  JSONL (which already stores `black`/`white` arrays, panel_tournament.py:184-207)
  and emits `black_elo / white_elo / elo_gap / white_loss_rate`. This is the one
  missing piece — the data exists, only the dual BT-fit + report is absent.
- **Cadence hook (non-blocking):** on each checkpoint (every ~100 epochs), append
  that checkpoint as a panel participant (`--only <new-net>` restricts to its new
  pairings; panel is **resumable** — completed pairs are skipped,
  panel_tournament.py JSONL-resume), play a small `--n-games` slice vs a *fixed
  engine subset* (e.g. heuristic + lookahead:4 + one real engine for a stable
  anchor), then run the white-Elo reader. Training never blocks on it — the arena
  is a separate process consuming checkpoints off disk, exactly the two-queue model.
- **The report must show, per checkpoint over epochs:** `white_elo` *and*
  `black_elo` as separate series (so defense progress is its own curve), plus
  `white_loss_rate` vs lookahead:4/6 trending → 0. A rising aggregate Elo that hides
  a flat white_elo is the failure mode this exists to catch. Track **Δwhite_elo/Δt**
  (the project north-star, specialized to the second player) as the defense KPI.
- **Anchor stability:** keep ≥2 fixed engines in every cadence slice so the affine
  calibration (panel's anchor fit) stays comparable across checkpoints; otherwise
  white_elo drifts with the field, not with the net.

---

## Appendix — concrete next actions (no GPU)
1. Write `scripts/panel_white_elo.py` (dual BT-fit + `elo_gap` + per-engine
   `white_loss_rate`) over the live JSONL. **Code-only, do now.**
2. Run the §1B.1 color-split loss-tail (champion vs lookahead:4/6, n=200, CPU) and
   the §1B.3 FPU sweep — Step A of the first experiment. **CPU, do now.**
3. ~~Build the §1B.2 forced-defense probe set.~~ **✅ DONE (#45, 2026-06-16)** as the
   white-defense suite (fixed white-to-move-threat fixture + `white_loss_rate` + Wilson
   CI + gate primitive). v1 uses a *weak* attacker → at the champion's floor; **#49** adds
   the strong-attacker variant that makes it diagnostic for I1/I2. (See the §1B.2 results block above.)
4. File the #18 **defense-arm child issue** for I2 (stamp the saving move) and a
   `derby-idea` cell for I1 (defense-teacher + VCT), per the one-lever-per-cell rule.
5. When a GPU lane frees: run the I1 slice; re-read white-Elo to judge.
