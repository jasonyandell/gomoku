# The WL era — wave-of-lockstep training series (index)

**Status: HISTORICAL (2026-05-20 → 05-21 design arc; runs through WL5).** The WL
("wave-of-lockstep") series was the 9×9 laptop-scale stability program: enforce
per-version tile uniformity, then bolt on scale-emulation levers to fight the
oscillation that uniformity exposed. WL4 was the best WL outcome (elo 1841 ATH);
WL5 closed the design arc. The series is **superseded** by the later sound-world
recipe ([sound-world-recipe.md](sound-world-recipe.md)) — but several WL levers
**survived as standard infrastructure** (see § What survived). This page is the
compact map; the individual designs live below.

## The series at a glance

| run | design page | one-line result |
|---|---|---|
| **WL1** | [wave-of-lockstep-design.md](wave-of-lockstep-design.md) — **LIVE** (canonical definition of wave mode) | validated per-version tile uniformity + the wave barrier; then broke into high-frequency chaotic elo oscillation (620↔1281). |
| **WL2** | [_archive: wl2-scale-emulation-design.md](../_archive/topics/wl2-scale-emulation-design.md) — **ARCHIVED** | four scale-emulation levers (EMA self-play weights, past-checkpoint mix, worker poll jitter, grad-accum 4×) to emulate AZ-at-scale in-flight version diversity. Implemented (`9wng4yu9`); smoothed the early trajectory + raised the la4 peak, **did NOT solve retention**. |
| WL3 / WL3.1 | (see [../../TRAINING_WIKI.md](../../TRAINING_WIKI.md)) | capacity/opening-diversity iterations; WL3.1→WL4 dropped `random_opening_moves` at e1537. |
| **WL4** | (see [../../TRAINING_WIKI.md](../../TRAINING_WIKI.md); dynamics in [loss-floor-bouncing.md](loss-floor-bouncing.md)) | best WL outcome — **elo 1841 ATH, la4=100%**, then a healthy lower-floor-bouncing plateau. |
| **WL5** | [_archive: wl5-diagnostics-archive-start-design.md](../_archive/topics/wl5-diagnostics-archive-start-design.md) — **ARCHIVED** | 3 always-on diagnostic streams + the Go-Exploit-style archive-start lever, resumed from WL4 e4024 (`o6cbjfnr`). Validated the pipeline without collapse but **did NOT beat WL4's ATH**. |

## Why WL2 and WL5 are archived, not deleted

Both were **launched and measured** — real evidence, not scrapped designs. WL2's
diversity stack didn't cure oscillation (diversity wasn't the binding constraint);
WL5's archive-start didn't move the floor below WL4. The full design docs (levers,
knobs, implementation plans, hypotheses + refutation criteria) are preserved
verbatim in `_archive/topics/`. Read them for the *reasoning*; read
[../../TRAINING_WIKI.md](../../TRAINING_WIKI.md) for the run-end numbers.

## What survived (these levers outlived the WL runs)

- **EMA self-play weights (`--ema-tau`, WL2 lever 1).** The single biggest WL2
  intervention — workers play the slow-moving EMA brain, the trainer learns on raw
  weights. This became **standard** and rides in every later recipe (sound-world
  clones it). The canonical "stability for self-supervised feedback loops" trick.
- **The white/per-color metric column (WL5 stream 3).** "Report white separately,
  never fold it into the aggregate" is now the load-bearing **defense gate** across
  the eval doctrine ([eval-suite.md](eval-suite.md), [eval-teacher-sensei.md](eval-teacher-sensei.md))
  — the exact instrument that later caught the attack-only-specialist collapse.
- **Policy-loss H+KL decomposition (WL5 stream 2).** Splitting policy CE into
  target entropy `H(pi_mcts)` vs the learning gap `KL(pi_mcts‖p_net)` — the
  interpretive tool behind [loss-floor-bouncing.md](loss-floor-bouncing.md)'s
  "target moved vs net fell behind" reading.
- **Archive-start / curated-seed precedent (WL5 lever).** The "seed self-play from
  curated trouble states" mechanism is the direct precedent the curated-buffer
  design generalizes to arbitrary curricula
  ([curated-buffer-and-curriculum-design.md](curated-buffer-and-curriculum-design.md)).

## See also
- [wave-of-lockstep-design.md](wave-of-lockstep-design.md) — WL1, the live wave-mode definition.
- [az-at-scale-vs-laptop.md](az-at-scale-vs-laptop.md) — the framing WL2 operationalizes (why laptop-scale AZ oscillates).
- [loss-floor-bouncing.md](loss-floor-bouncing.md) — the WL4 dynamics + the H/KL lens.
- [sound-world-recipe.md](sound-world-recipe.md) — the recipe that superseded the WL line.
