# Experiments — a hub of hubs

Every research thread that isn't the main training arc. Each is banked honestly,
including the negatives, **reconstructed from the logs** where we didn't record
it cleanly. The largest by far is the **seek-VCT program** — it gets its own hub.

> **← Hubs:** [index](index.md) · sibling hubs: [AlphaZero](alphazero.md) ·
> [Derby](derby.md) · [Autolab](autolab.md) · [M5-as-Mainframe](m5-mainframe.md) ·
> [Reference](reference.md)

## The big one

| Sub-hub | The thesis |
|---|---|
| **⭐ [Seek-VCT program](seek-vct.md)** | The net *steers*, the GPU oracle *finishes* (anti-correlated tractability). A dozen+ threads: the mega-VCT solver, backward/reachability mining, the shape-library engine, the learnability trilogy, the corpus-scale cascade. **This is the current live research frontier.** Newest results: [VCT-terminus self-play](topics/vct-terminus-selfplay-result.md) (#100/#101) · [VCT-defense aux head](topics/vct-defense-aux-head-result.md) (#103). |

## Other threads

| Thread | What we found | Provenance |
|---|---|---|
| **The "claw"** — knight's-move defensive crystal (`2x+y≡0 mod 5`) | Proven unique optimal 5-blocker AND proven invisible to line-organized eval | recorded — [the-claw.md](topics/the-claw.md) |
| **Rapfi `mix9svq` architecture** teardown | The winning engine is a quantized line-shape CNN; VCF/VCT is pure search, never the net | recorded — [rapfi-mix9svq-architecture.md](topics/rapfi-mix9svq-architecture.md) |
| **Molecule discovery toolkit** (methods from comp-genetics) | v0 blocking-probe was NEGATIVE (blocking is itself line-shaped); pivot to position-dependent objectives | recorded — [molecule-discovery-toolkit.md](topics/molecule-discovery-toolkit.md) |
| **swap2 opening protocol** (#72 — the "real" white fix) | Rebalances the GAME; white wins 27% in swap2 self-play vs ~0% empty-board; strength still ~parity | recorded — [swap2-opening-protocol.md](topics/swap2-opening-protocol.md) |
| **Rapfi idx-2 distillation mine** ("Bruce Lee one-position") | Mine Rapfi at ~700 mv/s → pretrain → warm-start; one-hot harms, soft-target is the fix | recorded — [rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md) |
| **LeGomoku** — latent-space world model; can it make a better *search*? | Brainstorm only; "probably goes splat" is the honest prior (the Texas-42 wall), pre-stated bets on record; expected to grow child pages | proposed 2026-07-03 — [legomoku.md](topics/legomoku.md) |

## Reference theory (external, for these experiments)

- **Allis's threat formalism** — the gomoku threat theory. [allis-threat-theory.md](topics/allis-threat-theory.md)
- **Standard human strategy** — the "rule of priorities." [gomoku-standard-strategy.md](topics/gomoku-standard-strategy.md)

## Wild directions / the seed pile

- **The idea pile** — pick a research direction / seed the autolab. [idea-pile.md](topics/idea-pile.md)
- The research board (open candidates + verdicts). [ops/research-board.md](ops/research-board.md)

## Full page index — every page in this hub

*Complete map (9 pages); the sections above surface the headline ones.*

| Page | Note |
|---|---|
| [the-claw.md](topics/the-claw.md) |  |
| [rapfi-mix9svq-architecture.md](topics/rapfi-mix9svq-architecture.md) |  |
| [molecule-discovery-toolkit.md](topics/molecule-discovery-toolkit.md) |  |
| [swap2-opening-protocol.md](topics/swap2-opening-protocol.md) |  |
| [rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md) |  |
| [legomoku.md](topics/legomoku.md) | proposed; brainstorm hub-in-waiting |
| [allis-threat-theory.md](topics/allis-threat-theory.md) | reference theory |
| [gomoku-standard-strategy.md](topics/gomoku-standard-strategy.md) | reference theory |
| [idea-pile.md](topics/idea-pile.md) | the seed pile |
