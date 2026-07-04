# Wall-clock-to-elo as a first-class metric family (LF1-followup #4)
> **Status: DESIGN-NEVER-BUILT** *(2026-07-04)* — 2026-05-23 spec; MTTE/EPWH family unimplemented.

*Design document. Drafted 2026-05-23 in worktree `feat/perf-LF1-metric-design`,
as LF1-followup lane #4 from
[perf-bench-vs-real-training-cost.md](perf-bench-vs-real-training-cost.md)
("Research lanes", item 4). This is a **design** page, not an implementation —
it specifies the metric, the protocol, the quality gate, the proposed charter
edit, and how it reconciles with the EXISTING `scripts/delta_e_harness.py`. No
code is changed here, and the charter is NOT edited (charter changes are
Class-B — the user's call; see [the proposed diff below](#5-proposed-charter-edit-quoted-diff-do-not-apply)).*

## Why this lane exists (the one-paragraph problem)

The perf lab's headline metrics — `aug/s`, `epochs/s` (the R-S\* and R-TRAIN-\*
families) — are *throughput proxies*. LF1 proved they diverge from the real
objective: the R-TRAIN-LEAN-fp16 recipe **won the throughput bench by +152.9%**
and then, in a real run, **diverged** (per-epoch wall 20 s → 7.3 min and still
climbing; SGD steps/epoch 25 → 3236) once the replay buffer filled. A recipe
"won" the benchmark precisely by *flooding the trainer faster than it could
consume* — the exact imbalance that wrecks the run. The lab needs a metric that
**cannot be gamed by flooding the trainer**: one denominated in the only unit
the project actually cares about, **wall-clock time to a given strength**. That
is this metric family. It does not replace the throughput proxies; it is the
higher-order objective they must be *checked against* (see
[§6, lanes 1/2/6](#6-relationship-to-lanes-126)).

## 1. Metric definitions

We define two metrics. They are duals — a time and its inverse-rate — and the
project's existing north-star framing ("Δelo/Δt is delta-v for training";
[curated-buffer-and-curriculum-design.md](curated-buffer-and-curriculum-design.md),
the [delta_e_harness.py](../../scripts/delta_e_harness.py) docstring, and the
[`feedback-elo-per-wall` memory]) already names the **rate** as the north star.
This page adds a target-time companion and pins both to wall-clock.

### (a) MTTE — Minutes-To-Target-Elo *(primary)*

> **MTTE(target)** = the wall-clock minutes, measured from a common start
> checkpoint **C**, until a recipe's anchored elo *first reaches and holds* a
> fixed **target anchor elo**.

- **target** is a fixed point on the anchor ladder
  (`gomoku.rating.ANCHOR_ELOS`): `random=0`, `heuristic=800`,
  **`lookahead2=1200`**, `lookahead4=1500`, `lookahead6=1700`. The default
  target is **lookahead2 ≈ 1200** — high enough to be past the easy-collapse
  regime, low enough that a small-model cell can reach it inside a feasible
  window. A cell MAY also report MTTE(800) as an early, cheaper checkpoint.
- "first reaches **and holds**" = the *anchored* elo (≥100-game eval, §2) is at
  or above target at the first measurement point where it does so, AND the
  point estimate minus its CI half-width is still ≥ target at the *next*
  measurement point (a one-checkpoint persistence guard, so we don't credit a
  noisy single spike — cf. the ±100-elo trainer-eval noise that
  `delta_e_harness` was built to beat).
- **Lower is better.** Units: minutes. A recipe that never reaches target
  within the protocol's max window reports `MTTE = ∞ (did not reach within
  Wmax)` — and that is a legitimate, *informative* result, not a failed cell.
  (LEAN-fp16 might post a *good* MTTE despite its runaway, because it does more
  learning per epoch — which is exactly why we measure it directly instead of
  guessing from epochs/s. See the "subtlety that keeps the verdict honest" in
  [perf-bench-vs-real-training-cost.md](perf-bench-vs-real-training-cost.md).)

### (b) EPWH — Elo-Per-Wall-Hour, i.e. Δelo·Δt⁻¹ *(secondary / diagnostic)*

> **EPWH** = (elo(C+W) − elo(C)) / (wall-hours elapsed over the window W),
> measured from common start checkpoint **C** over a **fixed wall-clock
> window** W vs a **stable anchor set**.

- This is the **slope** of anchored-elo-vs-wall-clock from C — the literal
  "delta-v" the north-star memory sharpened to ("the elo-gain RATE — slope, not
  integral"). Identical in spirit to `delta_e_harness`'s `delta_elo_per_hr`,
  with one critical change: the window is **fixed in wall-clock, not in
  epochs** (see [§3 gap analysis](#3-reconciliation-with-delta_e_harnesspy-gap-analysis)).
- **Higher is better.** Units: elo/hour.
- EPWH is reported with the same **inside-noise** guard as `delta_e_harness`
  (`|Δelo| ≤ CI half-width` ⇒ `INSIDE-NOISE`). An EPWH whose Δelo is inside the
  noise is **not a result** — it's "run longer or more games".

### Which is primary, and why

**MTTE(1200) is the primary metric. EPWH is the secondary/diagnostic.** Three
reasons:

1. **MTTE is runaway-proof by construction.** A recipe cannot improve its MTTE
   by flooding the trainer with more positions, because MTTE counts *wall-clock
   to a strength*, and flooding *adds* wall-clock per epoch without adding
   target-elo. This is precisely the gameability LF1 exposed in `aug/s`. EPWH
   is *also* hard to game (the denominator is wall-clock), but a short-window
   EPWH can still flatter a recipe whose early slope is steep and whose late
   slope collapses (the classic "fast then plateau/diverge"). MTTE integrates
   over the whole approach to a fixed bar, so it can't be fooled by a good
   early slope.
2. **MTTE answers the operator's actual question.** "How long until this recipe
   gives me a lookahead2-strength model?" is the decision the WL-release lineage
   turns on. EPWH answers "what's the instantaneous learning rate right now?",
   which is a *diagnostic* for *why* one MTTE beat another.
3. **MTTE degrades gracefully to a partial result.** If a long cell is killed
   early, you still have "reached 1100 by minute 40, slope flattening" — an MTTE
   *lower bound* plus an EPWH. EPWH alone over a too-short window can be
   actively misleading (the LF1 trap, transposed from epochs to wall-clock).

EPWH stays first-class because it's what the existing harness computes, it's
cheaper (one fixed window, no "wait until it crosses a bar" open-endedness),
and it's the right tool for the **fork-and-rank** use that `delta_e_harness`
was built for (curated-buffer curator A/B at a *fixed* window). Use EPWH to
rank curator/knob variants off a common C at a fixed window; use MTTE to
adjudicate a finalist recipe against the production baseline all the way to a
strength bar.

## 2. Measurement protocol

The protocol is the `delta_e_harness` shape (common parent C → fixed window →
anchored eval → Δelo) **extended along the wall-clock axis and sampled at
multiple checkpoints** so a *crossing time* (MTTE) and a *true wall-clock slope*
(EPWH) can be read.

### Fixed ingredients (the honesty contract)

| Ingredient | Specification |
|---|---|
| **Common start C** | A single named checkpoint all arms fork from. Its anchored elo is measured **once** and shared (the `delta_e_harness` parent-C pattern). For a head-to-head vs production, C should be a real WL-lineage checkpoint, not fresh-random. |
| **Stable anchor set** | `heuristic` (800), `lookahead2` (1200), `lookahead4` (1500) by default — the `ANCHOR_ELOS` ladder, identical across all arms and across the whole run. Anchors NEVER change mid-experiment (that would move the ruler). |
| **Target (MTTE)** | `lookahead2` = 1200 by default. State it explicitly in the receipt; `MTTE(1200)` and `MTTE(1500)` are different metrics. |
| **Anchored eval depth** | **≥ 100 games per baseline** (default 120, per `delta_e_harness --eval-games 120`). This is the noise fix: 20-game trainer eval is ±100 elo; we need the per-baseline Wilson CI tight enough that a crossing isn't a coin-flip. |
| **Alternating colors** | Each anchored match alternates which side the model plays (gomoku has a first-move advantage; an un-alternated match biases elo). The eval engine (`gomoku.eval`) already does this — the protocol just requires it be on, n_games even. |
| **Checkpoint sampling cadence** | Eval the rolling checkpoint at a **fixed wall-clock cadence** (e.g. every 10 min of training wall, or every K epochs whichever is *more frequent*), so the elo-vs-wall curve has enough points to (a) locate the MTTE crossing and (b) fit an EPWH slope. Minimum **5 sampled points** across the window, plus C. |
| **Window / stop rule** | EITHER a fixed wall-clock window **W** (for EPWH; e.g. W = 2 h) OR a target-elo stop (for MTTE; stop once target is reached-and-held, or at **Wmax**, whichever first). State which mode the cell ran in. |
| **n-games for elo CI** | The propagated elo CI half-width must be **< ½ the elo gap between adjacent targets** (i.e. < 150 elo for the 1200↔1500 rungs) for a crossing to count. If the CI is wider than that, the cell is under-gamed — increase `--eval-games`. |

### This BREAKS the smoke-first 60–90 s doctrine — by design

The charter's [smoke-first doctrine](research-lab-charter.md#smoke-first-doctrine)
(60–90 s default cell, 5 min hard cap) is **explicitly inapplicable to this
metric family**, and the source writeup says so directly: *"even though it
forces cells to run much longer than the current smoke-first 60–90 s doctrine.
Some questions can only be answered by a long run; this is one of them."*

The reason is the LF1 mechanism: a wave-mode trainer's per-epoch cost and the
buffer's fill state both change *regime* well after 90 s. A cell that stops
before buffer-fill measures a transient real training never sits in. So:

- **Minimum honest window for an MTTE/EPWH cell:** the window MUST span the
  buffer-fill regime change **plus several post-fill epochs** — concretely, the
  cell runs until **(buffer reaches capacity) AND (≥10 epochs elapse after
  buffer-full) AND (≥5 anchored-eval points collected)**, whichever is *last*.
  For the WL5-lineage 1.5M buffer this is on the order of **30–60 minutes
  minimum**; for the 3M buffer ([project-buffer-undersized]) it is longer. A
  cell that ends before buffer-fill **MUST NOT report an MTTE/EPWH number** —
  it reports `cell_status: pre-fill, non-predictive` (same disposition the
  charter's metric-validity flag already gives the cold-window R-TRAIN-\*).
- **These cells are a separate cell class.** They live in the GPU serial queue
  like any live cell, but they are *not* smoke cells and are *not* bounded by
  the 5-min cap. Budget them as **Tier-1 holistic** lanes (charter §"Tiers":
  end-to-end production cell), estimated in Opus-minutes (per
  [feedback-lab-scheduler]) at tens of minutes to a couple hours each. Run them
  when a finalist recipe needs adjudication, not on every knob pivot.
- **Warm-start to dodge part of the cost.** The cheapest honest variant forks
  from a C whose buffer is *already warmed to capacity* (the Lhot pre-warm from
  lane 1). That removes the cold-buffer transient and lets the cell measure
  steady-state from minute 0 — shrinking the minimum honest window toward the
  "≥10 post-C epochs + ≥5 eval points" floor. **Lane-1's warm-buffer fix is a
  prerequisite for an efficient lane-4 cell** (see [§6](#6-relationship-to-lanes-126)).

### What a cell emits

A `results.json` (extending `delta_e_harness`'s existing record) with, per arm:
the elo-vs-wall-clock series `[(wall_min, elo, elo_lo, elo_hi), …]` from C
through the window; the derived `MTTE(target)` (or `∞`); the `EPWH` slope with
its inside-noise verdict; the buffer-fill wall-clock-min; and the paired quality
gate (§4). Ranked table: by **MTTE ascending** (primary), with EPWH and the
val-CE gate as adjacent columns.

## 3. Reconciliation with `delta_e_harness.py` (gap analysis)

**`scripts/delta_e_harness.py` already implements most of this metric family —
do not reinvent it; extend it.** What it has, and the precise gaps:

### What it ALREADY does (reuse verbatim)

- **Δelo off a common parent C**, with the parent measured once and shared
  across arms (`DeltaEResult.parent`, `delta_elo`). This is exactly the EPWH
  numerator.
- **`delta_elo_per_hr`** (`DeltaEResult.delta_elo_per_hr` = `delta_elo /
  (wall_secs/3600)`). This is **EPWH already** — same formula, same units.
- **The noise fix**: anchored eval at ≥100 games/baseline, Wilson-CI on each
  win-rate (`binomial_ci`), propagated to an elo CI by re-solving `implied_elo`
  at the joint-low/joint-high win-rate endpoints (`estimate_elo`), and an
  explicit **`inside_noise`** guard (`|Δelo| ≤ CI half-width`). The §2 "elo CI
  must be < ½ the rung gap" rule is a tightening of this same machinery.
- **Anchor table reuse** (`gomoku.rating.ANCHOR_ELOS`) — the exact `heuristic
  800 / lookahead2 1200 / …` ladder this page's targets reference.
- **External-anchor scaffold** (`ExternalAnchor`) — the hook for a pinned Rapfi
  / Gomocup engine at a known elo, wired into the same anchored path. MTTE
  against an *external* target (e.g. "minutes to first beat Rapfi-at-X") drops
  straight into this.
- **Fork orchestration + dry-run + self-test** — argv construction, sequential
  GPU forking, synthetic-data math test. All reusable.

### The GAPS (what lane-4 must add to it)

1. **Window is fixed in EPOCHS, not wall-clock.** `--window-epochs T` trains
   each fork a fixed number of epochs, and `wall_secs` is just the *total fork
   time* — an output, not a controlled input. EPWH-as-defined needs the window
   fixed in **wall-clock**, and MTTE needs a **target-elo stop**. *Gap: add a
   wall-clock window mode (`--window-minutes`) and a target-elo stop mode
   (`--target-elo`, `--max-minutes`).* This is the single most important gap:
   fixing the window in epochs reintroduces exactly the LF1 failure — a
   runaway recipe with huge epochs is given unbounded wall-clock under a fixed
   epoch budget, so its `delta_elo_per_hr` is computed over a wall time that
   itself depends on the runaway. **An epoch-fixed window cannot honestly score
   a recipe whose per-epoch wall diverges.**
2. **Endpoint-only eval — no trajectory.** It evals **C and the final
   checkpoint only** (`anchored_eval(parent)` then `anchored_eval(final_ckpt)`).
   MTTE needs the elo-vs-wall *curve* to locate a crossing, and a trustworthy
   EPWH wants ≥5 interior points (not a 2-point secant that can't see a
   plateau/divergence). *Gap: sample the rolling checkpoint at a fixed cadence
   during the fork and anchored-eval each sample, not just the endpoint.*
3. **No MTTE at all.** There is no crossing-time concept, no target, no
   reaches-and-holds persistence guard. *Gap: add MTTE computation over the
   sampled trajectory.*
4. **No buffer-fill / regime awareness.** The harness has no notion of "did
   this window span the buffer-fill regime change?" — it would happily report a
   `delta_elo_per_hr` from a cold pre-fill window, the exact non-predictive
   number the charter flag warns about. *Gap: record buffer-fill wall-min and
   refuse to emit a number (`cell_status: pre-fill`) if the window ended before
   buffer-full + the post-fill floor.*
5. **Trainer-CLI contract mismatch.** It shells out to a `gomoku.train_replay`
   *replay-mode* trainer (the curated-buffer fork tool), NOT the **wave-mode
   `gomoku.train`** that produces the LF1 runaway. To measure the runaway
   recipe's true wall-clock-to-elo you must drive the wave trainer (the
   R-TRAIN-\* harness, `scripts/lab_train_cell.py`). *Gap: a driver mode that
   wraps the live wave trainer + concurrent generators and samples its rolling
   checkpoint — i.e. an MTTE/EPWH variant of `lab_train_cell.py`, OR a flag on
   `delta_e_harness` to target the wave trainer.* This is the only gap that
   touches a different code path; the elo/CI/ranking math (`estimate_elo`,
   `rank_results`, `format_table`) is engine-agnostic and carries over
   unchanged.

**Summary:** the *scoring engine* (elo + CI + Δelo + Δelo/hr + inside-noise +
ranking) is done and battle-tested by `--self-test`. The lane-4 work is (i) a
wall-clock/target-elo window controller, (ii) trajectory sampling, (iii) the
MTTE crossing computation, (iv) a buffer-fill regime guard, and (v) a
wave-trainer driver path. Implementation is **out of scope for this design
lane**; it is the implementation follow-up this page justifies.

## 4. Quality-gate pairing (val/policy_ce vs the named TQ gate)

**A wall-clock-to-elo win with degraded validation CE is not a win.** This is
not optional — it is what separated "real" from "gamed" in LF1, and it is the
named [Training-Quality Promotion Gate](../ops/experiment-ledger.md#training-quality-promotion-gate)
the lab already runs behavior-borderline knobs through.

- **Gate metric:** `val/policy_ce` evaluated against the named fixed archive
  **`archives/wl5_validation_v1.pt`** (the `--validation-archive-path` default
  baked into `delta_e_harness` and the replay-trainer contract). Optionally
  also `val/value_*` and the plies/game-shape signal from
  [loss-floor-bouncing.md](loss-floor-bouncing.md) /
  [feedback-self-play-eta], but **policy-CE vs `wl5_validation_v1.pt` is the
  required gate.**
- **Pairing rule:** every MTTE/EPWH number is reported *alongside* the arm's
  `val/policy_ce` at the same checkpoint that crossed the target (for MTTE) or
  at the window end (for EPWH). The receipt presents them as a pair.
- **Gate verdict:** an arm whose MTTE beats the baseline but whose
  `val/policy_ce` at the crossing checkpoint is **worse than the baseline's at
  the same elo** is filed `needs_repeat` / `quality-regressed`, NOT `promote`.
  The elo number can be inflated by overfitting to the self-play distribution
  (sharpening on attacks while losing generalization — the project's documented
  fast-attack-collapse failure mode); the held-out CE catches that.
- **This is the LF1-specific honesty check.** LF1's elo was *noisy*
  (~339–751, not cleanly climbing) while steps/epoch exploded — lane 5 of the
  source writeup asks whether those extra steps are "productive learning or
  redundant SGD on stale data". The val-CE gate is the instrument that answers
  it: if elo and held-out CE improve together, the wall-clock was well spent;
  if elo wobbles while CE flatlines or degrades, the runaway burned wall-clock
  on redundant updates and the MTTE "win" is hollow.

## 5. Proposed charter edit (quoted diff — DO NOT APPLY)

This is a **Class-B** change (modifying the charter; see charter §"Autonomy
boundaries"). It is **the user's call.** Presented as a proposed diff for the
next charter pass, NOT applied in this worktree.

The edit adds a third metric family to
[research-lab-charter.md § Success metric](research-lab-charter.md#success-metric),
after the R-TRAIN-\* subsection, and updates the existing metric-validity flag
to point at its supersession.

````diff
--- a/wiki/topics/research-lab-charter.md
+++ b/wiki/topics/research-lab-charter.md
@@ Success metric
 Two metric families, each with reference points. A win at *either*
 moves the lab toward the mission.
+
+**Update (LF1-followup #4):** there are now **three** families. The two
+throughput families below (R-S\*, R-TRAIN-\*) are *proxies*; the third
+(R-ELO-\*, wall-clock-to-elo) is the **objective** they are proxies for, and
+supersedes the cold-window R-TRAIN-\* number as the adjudicator of a training
+recipe. A throughput win must be **confirmed by an R-ELO-\* cell** before it
+changes a training recipe. See
+[wall-clock-to-elo-metric.md](wall-clock-to-elo-metric.md).
@@ (immediately after the R-TRAIN-* subsection, before "### Promotion rules")
+### R-ELO-* — wall-clock-to-elo (the objective the proxies serve)
+
+The only metric denominated in the unit the project actually optimizes:
+**wall-clock time to a fixed strength.** Unforgeable by flooding the trainer
+(unlike aug/s, which LF1 proved a runaway recipe can win at +152% while
+diverging in a real run).
+
+- **Primary: MTTE(target)** — minutes from a common start checkpoint C until
+  anchored elo first reaches-and-holds a fixed anchor target (default
+  `lookahead2 ≈ 1200`). Lower is better.
+- **Secondary: EPWH (Δelo·Δt⁻¹)** — anchored-elo slope vs wall-clock from C
+  over a fixed wall-clock window vs a stable anchor set. Higher is better.
+  This is the project north-star "delta-e" rate; `scripts/delta_e_harness.py`
+  already computes its endpoint form (`delta_elo_per_hr`).
+
+Measured via an extension of `scripts/delta_e_harness.py` (anchored eval ≥100
+games/baseline, Wilson-CI, `inside_noise` guard — all already implemented).
+
+> **These cells INTENTIONALLY break the smoke-first 60–90 s doctrine.** An
+> R-ELO-\* cell MUST span the buffer-fill regime change + ≥10 post-fill epochs
+> + ≥5 anchored-eval points (≈30–60 min min for the 1.5M buffer; longer for
+> 3M). A cell that ends before buffer-fill does NOT emit an R-ELO-\* number
+> (`cell_status: pre-fill, non-predictive`). These are Tier-1 holistic lanes,
+> run to adjudicate a finalist recipe — NOT on every knob pivot. Warm-buffer
+> pre-warm (LF1-followup #1) shrinks the honest window toward the post-C floor.
+
+**Quality gate (required):** every R-ELO-\* number is paired with
+`val/policy_ce` vs `archives/wl5_validation_v1.pt` at the crossing/window-end
+checkpoint. A wall-clock-to-elo win with regressed held-out CE is filed
+`needs_repeat`, never `promote` (catches self-play-distribution overfitting,
+the documented fast-attack-collapse failure mode).
+
+| ref point | what it measures | status |
+|---|---|---|
+| **R-ELO-WL5** | MTTE(1200) + EPWH of the WL5 production recipe from a common C | *to be measured (first R-ELO baseline)* |
+| **R-ELO-LEAN-fp16** | same for the LF1 runaway recipe — the open question lanes 4–6 exist to answer | *pending; adjudicates whether the +152% bench recipe is ever faster in wall-clock-to-elo* |
@@ METRIC-VALIDITY FLAG (the existing R-TRAIN-* warning block)
 > ⚠️ **METRIC-VALIDITY FLAG (2026-05-23, LF1): this cold-window R-TRAIN-*
 > measurement is NON-PREDICTIVE of real training and can HIDE a runaway.**
 ...
-> Fix queued as LF1-followups lane #1 (warm-buffer metric). See
+> Fix queued as LF1-followups lane #1 (warm-buffer metric) — which repairs the
+> R-TRAIN-* *instrument*. The higher-order fix is **lane #4: the R-ELO-*
+> family (wall-clock-to-elo)**, which SUPERSEDES this cold-window number as the
+> adjudicator of a training recipe — a throughput win must be confirmed by an
+> R-ELO-* cell before it changes a training recipe. See
+> [wall-clock-to-elo-metric.md](wall-clock-to-elo-metric.md) and
 > [perf-bench-vs-real-training-cost.md](perf-bench-vs-real-training-cost.md).
````

Companion one-liner for the charter's **Tiers** table (Tier-1 row, "Examples"
cell) — append: *"R-ELO-\* wall-clock-to-elo adjudication cells (long; break
smoke-first by design)."* And the **§"Cell time ceiling"** rule-of-thumb table
gains a row: *"R-ELO-\* wall-clock-to-elo adjudication | 30–120 min (Tier-1; NOT
a smoke cell; must span buffer-fill)."*

### How R-ELO-\* relates to the cold-window R-TRAIN-\* flag

The charter already carries a **metric-validity flag** on R-TRAIN-\* saying the
cold-window number is non-predictive and can hide a runaway. Two distinct fixes
address it, and they compose:

- **Lane #1 (warm-buffer) repairs the R-TRAIN-\* *instrument*** — it makes the
  *throughput* number (aug/s, steps/epoch *slope*) honest by measuring at
  full-buffer steady state and reporting the tile-growth slope. After lane 1,
  R-TRAIN-\* still measures *throughput*, just non-misleadingly.
- **Lane #4 (this page) supersedes R-TRAIN-\* as the *adjudicator*** — even a
  perfectly-instrumented throughput number is still a *proxy*. R-ELO-\*
  measures the *objective*. So R-TRAIN-\* (warm-buffer-fixed) becomes a *fast
  screen* — cheap, smoke-able-ish, good for ranking knob pivots — and R-ELO-\*
  becomes the *gate* a finalist recipe must clear before it touches a training
  recipe. **Complement, not replacement:** you still want the cheap throughput
  screen to triage; you just stop letting it *decide*.

## 6. Relationship to lanes 1/2/6

The LF1 writeup's research lanes form a dependency chain; this metric (lane 4)
sits at the top of it:

| Lane | What it fixes | Relationship to lane 4 (this page) |
|---|---|---|
| **#1 warm-buffer R-TRAIN metric** | Repairs the throughput *instrument* — measures at full-buffer steady state, reports the tile-growth *slope* (bounded vs diverging), refuses to emit a number from a pre-fill window. | **Prerequisite & complement.** Lane 1 makes the cheap proxy honest; lane 4 is the objective that proxy is checked against. Lane 1's warm-buffer pre-warm also makes a lane-4 cell *cheaper* (forks from a buffer-full C, skipping the cold transient). The two share the same "don't measure before buffer-fill" discipline. |
| **#2 map the runaway stability boundary** | Sweeps wave_size V to find the knee where the tile stays bounded (V=64 stable, V=512 divergent). | **Feeds lane 4 with candidates.** Lane 2 finds *which* recipes are even stable enough to be worth a (long, expensive) R-ELO-\* adjudication. No point spending an hour measuring wall-clock-to-elo for a recipe lane 2 shows diverges unboundedly — unless lane 4's MTTE is being used precisely to test whether a *bounded-but-fast* point on lane 2's curve actually wins. |
| **#6 architectural fix (bound the tile / backpressure workers)** | Converts the open feedback loop into a closed one (cap games/version, or pause generation during long train phases). | **Validated BY lane 4.** Lane 6 asks "does bounding the tile cost anything in elo-per-wall-hour?" — that question is *literally an R-ELO-\* / EPWH measurement*. Lane 4 is the instrument lane 6 needs to prove its fix is free (or cheap). |

**The throughput proxies (lanes maximizing aug/s) are means; wall-clock-to-elo
is the end.** Lane 1 fixes the instrument so the means stop lying; lane 4 makes
the end measurable so a means can be *checked against it*. The governing rule
this whole chain implies: **a throughput win does not change a training recipe
until an R-ELO-\* cell confirms it lowers MTTE (or raises EPWH) without
regressing val/policy_ce.**

## Caveats and scope

- **Design only.** No code in this worktree; `delta_e_harness.py` is read, not
  edited. The charter is *not* edited (the diff in §5 is a proposal). The
  implementation (window controller, trajectory sampling, MTTE crossing,
  buffer-fill guard, wave-trainer driver) is the follow-up this page justifies.
- **MTTE depends on C.** Different start checkpoints give different MTTEs; a
  recipe's MTTE is only comparable across arms that fork the *same* C. State C
  in every receipt.
- **Long cells contend for the GPU.** An R-ELO-\* cell occupies the GPU serial
  queue for 30–120 min. Per [feedback-lab-scheduler], that's fine for a Tier-1
  adjudication, but it means R-ELO-\* cells are *scheduled deliberately* (when a
  finalist needs deciding), not fired reflexively. The CPU queue (code, wiki,
  reviewer) keeps turning in parallel.
- **Anchor calibration is assumed stable.** MTTE/EPWH inherit whatever error is
  in `ANCHOR_ELOS`. If the anchor ladder is recalibrated
  (`gomoku.calibrate_elo`), past MTTE numbers are not comparable to new ones —
  note the anchor-table version in the receipt.
- **External anchors are scaffolded, not wired.** MTTE-to-beat-Rapfi is a
  natural extension via `delta_e_harness.ExternalAnchor`, but that engine
  integration is gated behind the same data-integration chunk the harness
  docstring defers (Sequencing step 3).

## Cross-refs and primary sources

- [perf-bench-vs-real-training-cost.md](perf-bench-vs-real-training-cost.md) —
  the LF1 finding this lane closes (Research lanes, item 4). The "subtlety that
  keeps the verdict honest" paragraph is the direct motivation for MTTE over a
  naive epoch count.
- [research-lab-charter.md § Success metric](research-lab-charter.md#success-metric) —
  the R-S\*/R-TRAIN-\* families and the existing metric-validity flag the §5
  diff edits.
- [`scripts/delta_e_harness.py`](../../scripts/delta_e_harness.py) — the
  existing Δelo / Δelo/hr scoring engine this metric extends (gap analysis §3).
- [curated-buffer-and-curriculum-design.md](curated-buffer-and-curriculum-design.md)
  — the "Sequencing" + "Metric" framing the harness was built for; EPWH is the
  fork-and-rank metric there.
- `gomoku/rating.py` `ANCHOR_ELOS` — the target ladder
  (`heuristic 800 / lookahead2 1200 / lookahead4 1500 / lookahead6 1700`).
- [loss-floor-bouncing.md](loss-floor-bouncing.md) — the held-out val/CE
  interpretation backing the §4 quality gate.
- Memory: [[feedback-elo-per-wall]] (Δelo/Δt north-star, "delta-v for
  training"), [[feedback-lab-scheduler]] (long cells, Opus-minutes budgeting),
  [[project-buffer-undersized]] (3M buffer lengthens the honest window),
  [[feedback-self-play-eta]] (don't extrapolate wall-clock from a short sample —
  the general form of this trap).
