# Plan: Fixing white-side (defense / second-player) weakness — 15×15 freestyle gomoku

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
