# Training Run Lineage

Compact route map for the main training story. This page is maintained
synthesis, not a replacement for the append-only evidence in
[../../TRAINING_WIKI.md](../../TRAINING_WIKI.md). Update it when a run ends,
when a run changes the working theory, or when a new run becomes the natural
starting point for future sessions.

## Current Read

> **2026-06-24 correction — the spine has moved past the WL series.** The
> "WL series is the current spine" / "WL4 is the best confirmed checkpoint"
> framing below is **9×9 era-1 history**, no longer where the live work is. The
> current spine is the **15×15 swap2/Bruce era**: Bruce-1 ("Bruce Lee") run
> `gogpmbhw`, the single-fair-opener fixed-openings net (idx-2 opener, recency-0.5
> curator), latest snapshot `~/data/swap2/babysit/snapshots/g15_e2659_0623_2025.pt`
> (e2659). Bruce is the self-play-only **baseline-to-beat**; its ~50-elo plateau is
> confirmed **buffer-knob-proof** (reuse/window/freshness levers all exhausted, flat
> strength, 0/16 vs Rapfi). The next lever is the external Rapfi-distillation
> teacher (#77/#46). See [swap2-opening-protocol.md](swap2-opening-protocol.md),
> [white-side-defense-plan.md](white-side-defense-plan.md), and the
> [TRAINING_WIKI.md](../../TRAINING_WIKI.md) tail (2026-06-16 onward). The WL-era
> table below is preserved as the era-1 trail, not as the current frontier.

- The WL series is the current spine of the project.
- WL4 remains the best confirmed strength checkpoint in the notes:
  `WL4-no-random-openings.plateau-e4024`, wandb `44cxzc9d`, ATH elo 1841,
  la4 100%, no NaN/crash, full buffer preserved.
- WL5 is a diagnostic and archive-start continuation from WL4. Phase 1
  validated that the archive-start lever and diagnostic streams run at scale,
  but did not yet beat WL4's ATH. Phase 2 is the same WL5 run after hot-restart
  onto Conv+BN-fused self-play workers.
- For a live/current claim, verify the tail of
  [../../TRAINING_WIKI.md](../../TRAINING_WIKI.md) and the W&B run before
  treating this page as current.

## Lineage Table

| Run | W&B | Question | Outcome | Next link |
|---|---|---|---|---|
| Z / `az-recipe-160k` | `sppjo3z5` | Can a trimmed AZ-faithful 9x9 recipe break fast-attack collapse? | Yes. Five explore/consolidate arcs, final e5000, lifetime peak elo 1718, defense regime proven. | Led to per-version buffer hypothesis and WL1. |
| WL1 / wave-lockstep | `l8mbntcm` | Does per-version uniformity fix the buffer-composition feedback loop? | Operationally yes, behaviorally incomplete. Hit strength milestones fast, then fell into high-frequency oscillation and la4 regression. | Led to in-flight diversity hypothesis and WL2. |
| WL2 / scale emulation | `9wng4yu9` | Do EMA self-play, past mix, jitter, and grad accumulation stabilize WL1? | Partially. Smoother early run and higher la4 peak, but retention failure magnitude matched WL1. | Led to opening-monoculture hypothesis and WL3. |
| WL3 / K=2 openings | `0o75gws5` | Does random opening diversity improve breadth and retention? | Training signal looked best so far: balanced baselines, la4 68%, plies regrowth. Crashed at e825 from native MCTS policy NaN. | Led to NaN fixes and WL3.1. |
| WL3.1 / fixed K=2 | `44cxzc9d` | Can WL3 reproduce after Python and C NaN guards? | Yes. Reached established strength by e1536: heuristic 100% sustained, la4 60-95%, plies 20-27. | Paused to test whether K=2 was curriculum or permanent crutch. |
| WL4 / K=0 from WL3.1 | `44cxzc9d` | If K=2 built representations, does removing random openings unlock canonical depth or collapse? | Best confirmed WL-series run. ATH elo 1841 at e2401, la4 100%, plies past Z endpoint, no collapse. Random opening diversity was necessary but not permanent infrastructure. | Led to diagnostics plus archive-start WL5. |
| WL5 phase 1 / diagnostics + archive-start | `o6cbjfnr` | Do fixed validation, H/KL split, side/ply metrics, and 15% archive-start run cleanly and lower the floor? | Ran e4001-e5051. No NaN, no worker death, no fast-attack collapse; diagnostics populated. Absorption shock validated, no new ATH yet. | Continues as phase 2 after fused-worker hot-restart. |
| WL5 phase 2 / fused workers | `o6cbjfnr` | Same WL5 design, but with Conv+BN-fused self-play inference. | Gen-side games/sec improved 1.53x. Per-epoch comparisons across e5052 boundary are not apples-to-apples; use per-wallclock or per-game rates. | Current monitoring target in the notebook. |

## Design Record Map

- WL1 design: [wave-of-lockstep-design.md](wave-of-lockstep-design.md)
- WL2 design: [wl2-scale-emulation-design.md](wl2-scale-emulation-design.md)
- WL5 design: [wl5-diagnostics-archive-start-design.md](wl5-diagnostics-archive-start-design.md)
- Launch and monitoring procedure: [launch-sequence-runbook.md](launch-sequence-runbook.md)
- Archive mining procedure: [mining-validation-archives.md](mining-validation-archives.md)
- Current perf arc: [m5-max-as-mainframe.md](m5-max-as-mainframe.md)

## Interpretation Rules

- Use fixed external baselines and enough games for strength claims. The
  notebook has several examples where a single 16-game eval bounced hard.
- Use plies and eval wall time as defense-regime indicators. Falling plies can
  mean fast-attack collapse; rising plies or rising fixed-eval wall time often
  precede clean win-rate improvements.
- Treat WL design pages as preserved design records once launched. The result
  lives here and in [../../TRAINING_WIKI.md](../../TRAINING_WIKI.md), not by
  rewriting old design hypotheses to sound inevitable.
- Do not flatten old run sections out of the notebook. This page is the map;
  the notebook remains the trail of evidence.
