# Training Run Lineage

**Status: LIVE (current to 2026-07-03).** Newest era on top of Current Read.

Compact route map for the main training story. This page is maintained
synthesis, not a replacement for the append-only evidence in
[../../TRAINING_WIKI.md](../../TRAINING_WIKI.md). Update it when a run ends,
when a run changes the working theory, or when a new run becomes the natural
starting point for future sessions.

## Current Read

> **2026-07-03 leading edge — the sound-world era (and the 9×9 chapter closed).**
> The spine moved again, past the Bruce/idx-2 era below. The **sound-world recipe**
> (on-policy GPU-oracle veto + VCT terminus + line-planes; net + cap50 oracle
> finisher) **closed the 9×9 chapter** — it killed the 9-ply fast-attack attractor
> that every off-policy VCT injection failed to, validated on 9×9 2026-07-01/02
> (#107; from-scratch run wandb `zeed2xw5`, champion `107b`). Product shape = **bare
> net + finisher = 95% vs heuristic** where the bare net draws ("drawishness is
> division of labor"). **Then the recipe did NOT graduate:** carrying it to 13×13
> failed as a **STRUCTURAL negative** — an attack-only specialist, **white 0/20**
> everywhere, from BOTH warm-start (`8rp0gjpm`) and from-scratch (`uublz536`) (#113);
> black forces a proven VCT by ply ~9–13, so white's sharp-defense examples never
> enter the buffer. The recipe-rebuild attempt **rails-v0** (drop terminus +
> attacker-preserve, 15×15 idx-2, `vraf0b6e`, #116) cured the terminus-ejection but
> **re-collapsed white via a different pathway** (side-favored opening → value
> poisoning), while banking one genuinely new signal — a mid-game **momentum swing**
> (blunder-punishing tactics). Full arc: [sound-world-recipe.md](sound-world-recipe.md).
> The Bruce/idx-2 note below is the immediately-prior era.

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
| WL5 phase 2 / fused workers | `o6cbjfnr` | Same WL5 design, but with Conv+BN-fused self-play inference. | Gen-side games/sec improved 1.53x. Per-epoch comparisons across e5052 boundary are not apples-to-apples; use per-wallclock or per-game rates. | Led (via era-2 swap2 + the 15×15 port) to the Bruce/idx-2 era. |
| Bruce / idx-2 fixed-openings (15×15) | `gogpmbhw` | Can a single-fair-opener self-play-only net (idx-2, recency-0.5) climb past the buffer-knob plateau? | Plateaued ~50 elo, **buffer-knob-proof** (reuse/window/freshness exhausted), 0/16 vs Rapfi, 0%-white hole. The self-play-only baseline-to-beat. | Motivated the external gradient (teacher #77/#46 → DEAD-END) and then the sound-world on-policy recipe. |
| Sound-world 9×9 (from-scratch) | `zeed2xw5` | Does an on-policy oracle veto + VCT terminus kill the 9-ply fast-attack attractor that off-policy injection never could? | **YES — 9×9 chapter CLOSED** (#107, 2026-07-01/02). Attractor killed in a day; bare net + cap50 finisher = 95% vs heuristic (bare draws). Champion `107b`. | Attempted graduation to 13×13. |
| Sound-world 13×13 (warm + scratch) | `8rp0gjpm` (warm) / `uublz536` (scratch) | Does the validated 9×9 recipe graduate to 13×13? | **NO — STRUCTURAL negative** (#113). Attack-only specialist, **white 0/20** from both seeds; black forces VCT by ply ~9–13 → white's defense examples never enter the buffer. OLD full-game 128×10 net beats both 40-0. | Reframed the line → recipe rebuilds (rails). |
| rails-v0 (15×15 idx-2) | `vraf0b6e` | Does dropping the terminus (+ attacker-preserve) cure white-starvation? | **PARTIAL** (#116). Terminus-ejection cured, but idx-2 is black-tilted → white re-collapsed via value-poisoning (a DIFFERENT pathway to the same #113 outcome); closed e5524. New signal: a mid-game **momentum swing** (blunder-punishing). | Levers next: tail-subsampling (#118), fairer opening (swap2). |

## Design Record Map

- WL series index: [wl-era.md](wl-era.md) (WL1–WL5 map + what survived)
- WL1 design: [wave-of-lockstep-design.md](wave-of-lockstep-design.md)
- WL2 design: [_archive: wl2-scale-emulation-design.md](../_archive/topics/wl2-scale-emulation-design.md)
- WL5 design: [_archive: wl5-diagnostics-archive-start-design.md](../_archive/topics/wl5-diagnostics-archive-start-design.md)
- Sound-world era: [sound-world-recipe.md](sound-world-recipe.md)
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
