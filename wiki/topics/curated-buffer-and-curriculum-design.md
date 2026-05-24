# Curated buffer + lazy generation + curriculum seeding — design

*Captured 2026-05-24 from the post-LF1 conversation. The plan for the next
training architecture. Status: DESIGN (not built). Companion to
[[project-buffer-curation]] (the research framing) and
[perf-bench-vs-real-training-cost.md](perf-bench-vs-real-training-cost.md)
(the runaway that motivates the decouple).*

## Why now — the thing it fixes

We honestly ~doubled **generation** (fp16-eval: R-S400 4,765 → 9,398 pos/s,
+97%, no behavior change). We did **not** double **training** — fed naively
into a rolling-FIFO buffer + `sgd_per_position`, the doubled generation made a
real run *run away* (LF1: epochs 20s → 7+ min, steps/epoch 25 → 3236, still
climbing). Generation outpaced the trainer, the buffer flooded, and
`sgd_per_position` turned the flood into unbounded SGD.

The lesson: **fast generation is only an asset if consumption is decoupled
from it.** The current architecture couples them. This design decouples them —
and the same decoupling gives us a curriculum-control surface for free.

> **Two payoffs, one mechanism (Jason: "control and performance"):**
> **performance** = the 2× generation becomes usable instead of a runaway;
> **control** = we choose what the model trains on (curricula, trouble
> positions, champion games), not just "whatever FIFO held."

## Two layers

### Layer 1 — Mechanism (the decouple → performance)

- **Retain everything on disk.** Append-only per-cycle archive with metadata
  (`version, plies, z, color, source`). Cheap: ~10–15 GB for 1M games packed.
  This is the "arbitrary pasts" store from [[project-buffer-curation]].
- **Curated in-RAM slice.** The trainer samples from a slice chosen by
  `curate(archive, cycle) -> indices`, refilled at cycle boundaries (not per
  minibatch — keeps the SGD hot path unchanged). Slice *selection* matters
  more than slice *width* (cf. [[project-buffer-undersized]]: width was the old
  axis; what's-in-the-slots is the real one).
- **Lazy / as-requested generation.** Pull-based: workers generate when the
  curator wants fresh candidates, not push-flood a FIFO. Generation rate and
  the trainer's appetite become independent.
- **Fixed SGD budget, independent of inflow.** Replace `sgd_per_position`
  (which scales SGD with the flood) with a fixed steps/epoch (or a budget
  decoupled from generation). **This is what kills the runaway** — SGD work no
  longer chases generation rate.

Net: generation rate and SGD become **orthogonal knobs**. Crank generation
(high W, V=512, fp16) to feed the curator a richer pool; the trainer consumes a
controlled budget from a controlled slice.

### Layer 2 — Policy (the curator → control)

The curator picks the slice from a set of **pluggable sources**, each with a
weight/schedule. This is the WL5 archive-start lever generalized from "our own
trouble positions" to "any curriculum":

| source | status | notes |
|---|---|---|
| self-play (current model) | have | the default stream |
| trouble-position archive | **have** (`archives/wl5_validation_v1.pt`) | WL5 Go-Exploit positions |
| opponent-zoo / past-checkpoint games | have (WL2 lever) | controlled-ratio mixing |
| **Gomocup champion games** | **needs acquisition + parser** | external strong play; Gomocup publishes tournament records (e.g. `.psq`); a real but separate data-ingest chunk |
| opening books / progressive opening curricula | needs build | easy→hard schedule |

Curator recipes (the research axis): recency vs diversity, version-stratified
(drift-rate), hard-example pinning, source-weighting, z/color/ply balance.
Pair every curator with **per-batch composition logging** (`(version, plies, z,
color, source)` histogram to W&B) — without it the curator's effect is
invisible after the fact.

## Metric — report the tuple, measure the real objective

- **Throughput is a tuple: (generation positions/s, train time/epoch).** Report
  them SEPARATELY. Conflated metrics (aug/s, epochs/min) both proved
  misleading — aug/s hid the runaway; epochs/min inverted against worker count.
- **The real objective is elo-per-wall-clock** (+ a quality gate:
  `val/policy_ce` vs the validation archive, plies-shape). Throughput proxies
  do not substitute for it. A 60s screen can't measure it (no elo signal in
  60s); only a real run can.

## Scale

Target **1M games** (the disk archive scale). The curated in-RAM slice is
separate and smaller; with curation, its *width* is secondary to its
*selection*.

## Sequencing (don't boil the ocean)

1. **Mechanism first.** Build retain-all + curated-slice + lazy-gen + fixed-SGD
   budget. Prove the runaway is gone and the (gen, train) tuple decouples.
2. **Prove the loop with what we have.** Self-play + the existing trouble
   archive as the only two sources. Measure **elo-per-wall vs WL5** — this is
   the first honest test of "did decoupling + curation make training faster
   *and/or* better," the claim we have NOT yet earned.
3. **Then plug external curricula.** Gomocup champion games (acquire + parse),
   opening books, progressive curricula — each as a weighted source. This is
   also-fun but a separate chunk; gate it behind step 2 working.

## Open questions

- Curation policy: recency vs diversity vs trouble-weighting vs champ-weighting
  — and how to schedule them over a run.
- Staleness: how old can slice samples be before they hurt (the absorption /
  drift-rate question, [[feedback-absorption-phase]]).
- Lazy-gen interface: how the curator signals "need N more from source X" to
  the worker pool without reintroducing a flood.
- Does external strong-play seeding actually accelerate elo/wall, or just
  shift the opening distribution? (The step-2 vs step-3 A/B.)

## Cross-refs

- [[project-buffer-curation]] — the research framing (position flow as an instrument).
- [[project-buffer-undersized]] — buffer *width* (the old axis; orthogonal to this).
- [perf-bench-vs-real-training-cost.md](perf-bench-vs-real-training-cost.md) — the runaway this fixes.
- [[feedback-self-play-eta]] — why throughput proxies mislead training ETA.
- WL5 archive-start lever (`wiki/topics/wl5-diagnostics-archive-start-design.md`) — the seed-from-curated-positions precedent this generalizes.
