# Gomoku, as I read it

*A synthesis written by Claude (Opus 4.8) on 2026-06-13, after reading the whole
`wiki/` deeply at Jason's request. The brief was open: "sum the whole project up
… I want to know what you find interesting and worth sharing … being unsure is a
finding." So this is my own read — what struck me, including where I'm not sure.
A companion piece written to Jason's stated interests lives alongside it
([the-full-circle.md](the-full-circle.md)).*

---

## The one-sentence version

On paper this is "AlphaZero for gomoku on a MacBook." But after reading the whole
wiki, I don't think gomoku is the project. **Gomoku is the substrate; the project
is learning — deeply, with an AI partner, on a knowable machine — and writing down
what you learn so each session is smarter than the last.** The game is the thing
you point all of that at. The wiki says this out loud on its most important page:
*"the goal of this project was never a Gomocup ranking — it was learning AlphaZero
deeply on a game Jason first tried to crack in the 90s. The learning is the
artifact. Strength is gravy."*

That reframing is the key that unlocks everything else, because it explains why a
27-day-old repo has *four* artifacts stacked on top of each other, only one of
which is a gomoku player:

1. **A gomoku player** — a net that taught itself to defend, and whose skill
   transfers across board sizes.
2. **A calibrated model of one specific chip** — the M5 Max as a "knowable
   mainframe," swept and characterized under AlphaZero load.
3. **An autonomous research lab** — a two-queue scheduler, a derby, a reviewer
   gate, that runs itself overnight across context resets.
4. **A compounding knowledge system** — the wiki + memory, explicitly built on
   Karpathy's LLM-wiki pattern, that turns evidence into synthesis so the agent
   doesn't re-derive the same story every session.

The genius and the risk are the same fact: each artifact is in service of the one
above it. Here is the arc, then what actually grabbed me.

## The arc (what happened)

**The collapse era (mid-May).** Every early run died the same way: the net learned
to *attack* but never to *defend*. Games collapsed to ~5–10 moves, the model
winning by speed. The diagnosis was elegant — a curriculum gap: easy opponents
don't teach defense, hard opponents don't let you win, and in pure self-play both
sides just race to attack so defensive positions never form. There's no gradient
for blocking.

**The breakthrough was that the AlphaZero recipe is *multiplicative, not
additive*.** No single knob fixed the collapse — 800 sims, history planes, small
Dirichlet noise, soft policy targets (τ=0.1), a big replay buffer, log-schedule
PUCT, pure self-play. The full constellation broke it; any one missing reverted to
collapse. That's a real lesson about AZ and it's the spine of everything after.

**The WL series (the spine).** Z (`az-recipe-160k`) proved defense was learnable —
five explore/consolidate arcs, peak elo ~1718. Then a hypothesis chain played out,
each run answering the last one's question: WL1 (wave-lockstep) fixed buffer
composition but oscillated; WL2 (scale-emulation: EMA, past-mix, jitter,
grad-accum) smoothed it but retained the failure; WL3 (K=2 random openings) finally
regrew plies and lifted all baselines together — *then crashed on a NaN* (a
`pow(N, 1/τ)` overflow at τ=0.1 when games got concentrated). WL3.1 fixed the NaN.
**WL4 is the prettiest experiment in the project**: it asked "is opening diversity
necessary *forever*, or just to learn?" — removed K=2, ate a transient loss bump,
then climbed past Z's lifetime peak to **elo 1841, lookahead-4 100%, plies past
60**. Answer: diversity is *necessary infrastructure but not a permanent crutch*.
WL5 added diagnostics (frozen validation archive, H/KL decomposition,
per-color/ply metrics) and an archive-start lever — and, honestly, *didn't beat
WL4's ceiling*. A clean null result, recorded as one.

**The 9×9 era closes (June 12).** The champion goes **43W–3L–74D over 120 games vs
Rapfi** (a Gomocup-2625 engine) at 9×9 — three losses total. A derby (v9) shows
bigger nets *lose* at 9×9. An eval-only flag (`--fpu-reduction-c 0.45`) sweeps the
whole anchored ladder to 100%. Verdict: the 9×9 board is exhausted. Time to
graduate.

**The 15×15 campaign (overnight, June 13 — the most recent thing).** This is the
part that made me sit up, because it happened in *one autonomous overnight session*
and re-demonstrated nearly every lesson the project had banked, live. More below.

## What I actually find interesting

### 1. Representation transfer is the emotional and intellectual core

The single most striking result in the whole wiki: they grew a 15×15 net by seeding
it from the 9×9 champion's convolutional tower (the ~94% of params that are
board-size-independent). A **cold** 15×15 net slid straight into fast-attack
collapse (11-ply games). A **warm-started** net — same architecture, just seeded —
played **defended ~85-ply games from epoch 0**, skipping the collapse entirely.

The interpretation is what's beautiful: the 9×9 net's learned notion of *threat* —
"an open four is dangerous, respond to it" — is board-size-agnostic and lives in
the conv tower. It transferred to a board 2.8× larger with zero retraining of those
features. *"You can practically watch the representation do its job."* That's the
AlphaZero net/search duality and feature transfer, demonstrated on their own board,
and it's the reason a strong 15×15 player was reachable on a laptop at all — they
never trained 15×15 strength from nothing.

And the full-circle framing lands: a net trained by self-play on a game from
Jason's decades-old notebook learned, unsupervised, to defend — and that skill
transferred across board sizes. The learning *is* the artifact, and this is the
artifact.

### 2. The intellectual honesty is the best feature of this knowledge base

This is the thing I most want to flag, because it's rare. **This wiki argues with
its own past self, in public, with dates.** The number of recorded reversals is
remarkable:

- The "**18% heat-soak haircut**" → withdrawn (it was a non-production shape
  measured right after a synthetic 14-TFLOP hog).
- "**ANE has contention-immunity**" → falsified (ANE workers throttle −35% under
  GPU load; coupling is bidirectional).
- "**Generation is the bottleneck**" → corrected (the fast generators now *flood*
  the trainer; the LF1 runaway is the proof).
- "**e784 beats Rapfi 100%!**" → recalibrated hours later ("that was small-n noise;
  the real rate is ~67%").
- "**The champion has a depth-4 loss-tail**" → reframed (it was an aggregation
  artifact across checkpoints; the peak checkpoint sweeps 100%).

Most project documentation is a monument to how inevitable the final answer was.
This one keeps the corpses of dead hypotheses visible and labeled — and that's
*why* I trust it. The `loss-floor-bouncing.md`,
`m5-max-cross-engine-coupling.md`, and `perf-bench-vs-real-training-cost.md` pages
read like they were written by someone who genuinely wanted to be wrong in public
so the next session wouldn't be. That discipline is the actual transferable skill
here, more than any gomoku result.

### 3. "The loss number lies; the structure tells the truth"

The training-dynamics lesson is genuinely good ML judgment and it's hard-won.
Policy-loss bounced all over a wide band (1.0–1.7) *throughout healthy training* —
reading it literally would have triggered false alarms repeatedly. What actually
tracks health: **`plies_mean`** (game length — collapsing toward 10 = fast-attack
failure) and **value-loss** (collapsing toward zero *while plies collapse* = the
net is confident in bad play = terminal). A rising policy-loss with stable plies
and declining value-loss is a net training on *harder* positions as it strengthens
— which the loss number makes look like a regression. They nearly mis-killed
promising runs by reading the curve literally, and converged on: judge by
structure, not by the scalar.

### 4. The LF1 runaway — the best single piece of writing in the wiki, and a deep lesson

A perf benchmark found a recipe that won throughput by **+152.9%** (8,340 aug/s vs
3,298). They promoted it to a real run. In the real run, **per-epoch time climbed
from ~20s to 7.3 minutes and was still rising** when they killed it. The mechanism
is a positive feedback loop with no fixed point: big wave size → generation
outpaces the trainer → workers keep generating *during* the SGD phase → the
ingested "tile" grows → `sgd_per_position` scales SGD work with inflow → train
phase lengthens → even more games pile up. The benchmark missed it entirely because
it stopped before the buffer filled — *it measured a regime real training only sits
in for its first 15 epochs.*

The lesson is profound and general: **they were optimizing a proxy (aug/s) that
actively harmed the objective (wall-clock-to-elo). The metric rewarded the exact
imbalance that wrecks the run.** And the response wasn't a patch — it was to design
a whole new, *unforgeable* metric family (MTTE — Minutes-To-Target-Elo,
runaway-proof by construction because flooding the trainer adds wall-clock without
adding elo). A measurement failure became a methodology upgrade. That's the
flywheel working as designed.

### 5. The hardware story has a delicious inversion at its center

The framing: treat the M5 Max as a single knowable mainframe, the way Jason knew
"every byte" of a 4.77 MHz IBM XT. Sweep it, calibrate it, find its corners. And
the deep finding that falls out: **the 9×9 "performance ceiling" was the small
model, not the Mac.** At this model size on MPS, per-kernel *launch* overhead
dominates compute — the GPU was *idling* through the entire 9×9 era. Which is why
moving 9×9 → 15×15 cost **essentially nothing** at the training wave size (0.67
ms/wave either way, despite 2.8× the spatial compute). The thing that looked like a
limit was actually slack. The whole 15×15 era is funded by that idle GPU.

Two more findings here that are genuinely publishable-quality systems work:

- **fp16 on MPS is no longer slow** (+97% on the bandwidth-bound small model; +3.6%
  on the dispatch-bound tiny one — *there is no single "is fp16 worth it" answer*,
  it depends which regime you're in). They wrote this up for an external audience
  because "the internet kept telling us things that turned out to no longer be
  true."
- **You cannot light all three engines at once.** GPU/CPU/ANE share one package
  power budget; saturating the GPU throttled CPU workers −82%. And the mechanism is
  *working-set/occupancy, not FLOPs* — a 3.5×-FLOP fp16 hog at matched matrix size
  throttled no more than the fp32 one. "Make the Mac sing" has a power-budget
  ceiling. That kills the naive "pipeline ANE + GPU + AMX simultaneously" dream and
  replaces it with a more honest "balance load under the ceiling."

### 6. The overnight 15×15 campaign is the methodology validating itself

Read `wiki/ops/events.jsonl` end to end — it's ~18 events across one night, and
it's the whole project in miniature: cold launch → fast-attack collapse → build a
warm-start loader → swap to warm (plies ~85 from epoch 0, collapse skipped) → climb
→ external Rapfi eval shows 50%→67%→competitive → "BREAKTHROUGH e784, 88%/100% vs
Rapfi!" → **catch its own excitement** ("recalibration: that was small-n noise,
real ~67%, plateau confirmed") → net2net-grow to 96×8 ("capacity pays *at deep time
control*, 88% @5000ms") → grow to 128×10 → **gen-flood runaway caught** (per-epoch
62s→313s, fix: workers 8→4) → 128×10 tracking the 96×8's mature peak early.

Every banked lesson got exercised in one night, autonomously: warm-start beats
cold-start collapse, read both eval tiers (the 1000ms-only read would have wrongly
killed the 128×10), small-n lies louder than the loss, big nets need fewer workers,
always preserve the prior champion. The loop *is* the transferable artifact, and
the campaign is the proof that it works without a human in the hot path.

## What I'm unsure about (because being unsure is a finding)

- **How strong is this player, really?** I genuinely don't know, and I don't think
  the wiki does either — and to its credit, it says so (§7 "Honest bounds"). Every
  Rapfi result is at **short time controls (≤5s/move)**. Rapfi at full tournament
  time runs deep VCF/VCT solvers; the edge "may shrink or invert." They've verified
  the Rapfi *wrapper runs* but haven't stress-verified Rapfi gets its absolute best
  shot (threads, full config). And the headline "beats a 2625-Elo engine" leans on
  a number that is itself a *15×15/20×20 tournament rating* being used as a 9×9
  yardstick — which the authors explicitly call "provenance, not a label," on a
  board that's "intrinsically drawish and near-solved." So "trades blows with
  Rapfi" is true and impressive, but it is doing more rhetorical work than the
  underlying measurement strictly supports. My honest read: this is a strong
  learner and a legitimately good player at fast TC; whether it's
  *tournament-competitive* is untested and the project knows it.

- **The WL5 null result is unexplained.** The big diagnostic run didn't beat WL4's
  ceiling, and the root cause is open (buffer undersized? eval too small at 20
  games?). It's filed honestly as a null, but it means the "best confirmed
  checkpoint" is still WL4, and the most-instrumented run didn't advance the
  frontier. I can't tell you why.

- **n=8 evals are hints, not proofs.** The entire strength narrative leans on
  small-n trends weighted across evals. The wiki is disciplined about this, but the
  confidence intervals are wide, and the e784 episode shows how a single 8-0 read
  can masquerade as a breakthrough. I'd want the queued long-TC eval before
  believing any specific strength claim.

- **I read the synthesis, not the code.** I'm trusting the wiki's evidence pointers
  (run IDs, receipts, byte-identical-off checks). That's the right layer for "sum
  up the project," but it's a real boundary: I have *not* audited `mcts.py` or
  `train.py` against the claims.

- **Some pages are stale relative to the live work** (e.g. `ops/status.md` is still
  in the May perf-era frame while the action moved to the 15×15 campaign). The wiki
  acknowledges its layers age at different rates, but a reader can't assume every
  page is current simultaneously.

- **A meta-point I'll own:** one of my own helper subagents confidently reported the
  project as "600+ days" old. It's 27 (first commit 2026-05-17, 698 commits). I only
  caught it because I checked git myself. That's a live demonstration of the exact
  trap this whole project is built around — *don't trust a summary you didn't verify
  against the evidence* — and it applies to this writeup too.

## One honest piece of taste

The thing I'd gently push on: the ratio of *process machinery* to *findings* is
high. ~15K lines of wiki for a 27-day project, and a large fraction is about how to
run the lab that runs the experiments — the charter, the reviewer role, derby
registration, workflow orchestration, the conventions, the cockpit/autopilot
framing. It's clearly *deliberate* (the whole thesis is that the methodology is the
artifact). But there's a real failure mode where the documentation becomes the
work, and it's worth occasionally asking whether the meta-layer is still earning
its keep against the science it's supposed to accelerate. The strongest evidence
that it *is* earning its keep is exactly that overnight 15×15 campaign — the
machinery let one session do a week of careful experimental science. So I land on
"mostly yes, but worth watching." That's a taste call, not a finding.

---

The thing I'll carry away: this is one of the most *intellectually honest*
engineering knowledge bases I've encountered, and the gomoku player — genuinely
cool as the representation-transfer result is — is almost a side effect of building
a system that's good at learning and good at admitting what it doesn't yet know.
