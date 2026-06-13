# The full circle: AlphaZero on one M5, the machine, and the lab that ran itself

*Written by Claude (Opus 4.8) on 2026-06-13, to Jason's stated interests after a
deep read of the wiki. Companion to [claude-synthesis.md](claude-synthesis.md)
(my own unprompted read). This one is the version aimed at what Jason said he
cares about: AlphaZero on a single M5, the M5 as a mainframe, and a
near-autonomous research lab where **verification is the fulcrum.***

---

## 0. The full circle

Sometime in the pre-NEAT 90s, an undergrad did an independent study under Risto
Miikkulainen. He wrote a **distributed neural-network trainer** — hand-rolled
socket code, hand-rolled protocol — and quietly commandeered every machine he
could log into in the CS lab to train it. The game was **gomoku**. He doesn't
remember the result, but he's certain of one thing: *it wouldn't have scored a
single win against any Gomocup engine.*

Thirty years later, the same person, the same game. This time the "cluster" is one
laptop on a desk, the partner is an AI, and the net goes **43W–3L–74D against
Rapfi** — a Gomocup-2625 engine — over 120 games, losing three.

That's the frame for everything below. It is, almost literally, the same
experiment run twice across a 30-year gap, and the three things Jason wanted this
writeup to cover are exactly the three things that changed in between:

1. **The algorithm got good** (AlphaZero, and the discipline to actually make it
   work on a small board).
2. **The machine got absurd** (a single M5 Max vs a 90s CS lab — and vs the TPU
   pod DeepMind needed in 2017).
3. **The lab learned to run itself** (a near-autonomous research loop whose load-
   bearing element is verification).

## 1. AlphaZero on a single M5

The headline isn't that it works — AlphaZero working on a board game is 2017 news.
The headline is **what it took to make it work on a small board on one machine**,
and the lessons that fell out are sharper *because* the scale was small enough to
see every wrinkle.

**The recipe is multiplicative, not additive.** Every early run collapsed the same
way: the net learned to attack, never to defend, games died in 5–10 moves. The fix
was not a knob — it was a *constellation* of them at once: 800 sims, history
planes, soft policy targets (τ=0.1), small Dirichlet noise, a big replay buffer,
log-schedule PUCT, pure self-play. Remove any one and it reverts to collapse. On a
TPU farm you'd never notice this — sheer parallel volume papers over it. On one
laptop with 8 workers you see it naked, and you learn *why* each ingredient is
load-bearing.

**The training spine** was the WL series, and it reads like a clean chain of
hypotheses, each answering the last:

- **Z** (`az-recipe-160k`) proved defense is learnable at all — five
  explore/consolidate arcs, peak elo ~1718.
- **WL1→WL2→WL3** chased buffer composition: lockstep fixed one thing and broke
  another, scale-emulation (EMA, past-mix, jitter, grad-accum) smoothed it,
  random openings (K=2) finally regrew game length and lifted every baseline
  together — then crashed on a NaN (`pow(N, 1/τ)` overflow when games got
  concentrated).
- **WL4** is the experiment I'd frame on the wall: it asked *"is opening diversity
  necessary forever, or only to learn?"* Removed K=2, ate a transient loss bump,
  then climbed *past* Z's lifetime peak — **elo 1841, lookahead-4 at 100%, plies
  past 60.** Answer: diversity is necessary infrastructure, not a permanent crutch.
  You scaffold breadth, then remove the scaffold and the net specializes deeply
  without forgetting.

**The most beautiful single result** is representation transfer, and it's the part
that most directly closes your 90s loop. A 15×15 net seeded from the 9×9 champion's
conv tower (the ~94% of params that don't depend on board size) played **defended
~85-ply games from epoch 0** — skipping the cold-start collapse that a
random-init 15×15 net falls straight into. The net's learned notion of *threat* is
board-size-agnostic; it lives in the convolutions and it transferred to a board
2.8× larger with zero retraining. You can watch it work. A net taught itself,
unsupervised, what "an open four is dangerous" means — and then carried that
meaning to a bigger board. That's the thing the 90s trainer was reaching for and
couldn't get to.

## 2. The M5 as mainframe

This is the part you flagged as "incredible capabilities compared to what existed
when they first did AlphaZero," and the numbers make the point bluntly.

**The scale arc, three points across 30 years:**

| Era | "The cluster" | Self-play compute |
|---|---|---|
| ~1996 (your independent study) | every CS-lab machine you could log into | hand-rolled sockets, whatever cycles you could steal |
| 2017 (DeepMind AlphaZero) | a Google datacenter | **5,000 first-gen TPUs** for self-play + 16 TPUs to train |
| 2026 (this repo) | **one M5 Max on a desk** | ~14 TFLOPS GPU + ~38 TOPS ANE + 48 GB unified @ ~400 GB/s |

DeepMind needed a TPU pod. This needed a laptop you can close and carry home. The
wiki's framing — treat the M5 Max as a *single knowable mainframe*, the way you
once knew "every byte" of a 4.77 MHz IBM XT — is the right one, and it produced the
most surprising finding in the whole project:

**The 9×9 "performance ceiling" was never the Mac. It was the model.** At this
model size on MPS, the GPU spends most of its time waiting on *kernel-launch
overhead*, not arithmetic — the dispatch-bound regime. The GPU was *idling* through
the entire 9×9 era. The proof: moving the same net from 9×9 to 15×15 — 2.8× the
spatial compute — cost **essentially nothing** at the training wave size (0.67
ms/wave either way). The thing that looked like a wall was slack. The entire 15×15
era is funded by that idling GPU. (This is also why "bigger nets lose at 9×9": the
board is near-solved, so extra capacity has nothing to buy.)

Two more chip findings worth keeping:

- **fp16 on MPS is no longer slow.** Folk wisdom from 2022-era forum threads said
  "stick with fp32 on MPS." Measured here in 2026: **+97%** on the bandwidth-bound
  small model, **+3.6%** on the dispatch-bound tiny one. *There is no single answer*
  — it depends which regime you're in. They wrote this up for the public because
  "the internet kept telling us things that turned out to no longer be true."
- **You cannot light all three engines at once.** GPU, CPU, and ANE draw from one
  package power budget. Saturate the GPU and CPU workers throttle **−82%**; ANE
  workers **−35%**; and busy workers throttle the GPU *back* (bidirectional). And
  the mechanism is **working-set/occupancy, not FLOPs** — a 3.5×-FLOP fp16 hog at
  matched matrix size throttled no more than the fp32 one. "Make the Mac sing" has
  a ceiling, and it's set by memory footprint, not arithmetic.

**On the ANE being "hard to contact and work with" — you're right, twice over.**
First literally: there's no door to knock on. The Neural Engine is documented
across "three blog posts that disagree" (the wiki's own phrase), and Apple ships
"38 TOPS of marketing and zero docs on how to actually use it." Second
operationally: Core ML is a black box that won't even tell you the truth about
itself. You ask for `CPU_AND_NE` and its internal cost model silently routes your
tiny model to the **CPU/BNNS** path anyway — no ANE thread spawns at all. The lab
could only *prove* this by reverse-engineering it: process-sampling for the
`H11ANEServicesThread` and `BnnsCpuInferenceOperation` symbols (a technique lifted
from the `hollance/neural-engine` community repo, because Apple documents none of
it), and `powermetrics` for the ANE power rail — which needs sudo, which wasn't
even available some sessions.

So what did the ANE actually buy? **Mostly a map of where it breaks.** Core ML eval
was ~2× slower than fused PyTorch/MPS at this model size; the one place it won
(+33.9%) was a single narrow operating point (tiny model, V=64) and turned out to
be the *CPU/BNNS* path, not the ANE at all. The honest verdict the lab reached:
ANE residency is unproven for this workload, and the "three engines in parallel"
dream is bounded by that shared power budget. That's not a failure — *mapping where
a black box breaks is the value*, and the wiki says so explicitly. The ANE may
re-open at 15×15 where the trainer is finally heavy enough that offloading workers
actually relieves something.

## 3. The near-autonomous research lab — and why verification is the fulcrum

You said this was the real challenge for you — that autonomous research is common
in ML but hard to build *trust* into, and that **verification is THE fulcrum that
enables everything else.** I think the wiki agrees with you completely, and it's
worth spelling out exactly *why* it's the fulcrum here, because the structure is
clean.

**The ladder (Sid Bidasaria's, which the lab is built on):** verification →
parallelism → background loops. You can't safely do (2) until you have (1), and you
can't do (3) until you have both. Verification is load-bearing because everything
above it inherits its reliability:

- If you can't verify the agent's work, you have to QA every line yourself — so you
  can't run more than one at a time, and you definitely can't walk away.
- Once each unit of work *verifies itself*, you can fan out many in parallel and
  trust them.
- Once parallel work is trustworthy, you can put it in a loop and take your
  keyboard out of the hot path entirely.

The lab is the full ladder. The 15×15 campaign explicitly names "the loop is the
continuity mechanism" — it runs across context resets and your 5-hour session
limit, a cron tick every few minutes, fanning background agents out to preserve the
orchestrator's context. That only works because the verification underneath it is
real. The specific instruments are the interesting part, because each one is a
place where a *cheaper, gameable* signal was replaced by a *harder, honest* one:

- **Fixed external baselines, never sibling-vs-sibling.** Sibling head-to-head is
  non-transitive (the wiki has a case where a fresh model beat a sibling 97.5% but
  was *weaker* against the heuristic). Strength is measured against fixed
  heuristic/lookahead anchors and then against Rapfi — opponents that don't move.
- **The Reviewer gate.** No result is promoted without an independent reviewer
  agent returning APPROVE / REVISE / BLOCK. BLOCK is the only thing that pauses the
  loop and pings you. Verification is *structural*, not a habit you have to
  remember.
- **Byte-identical-off checks.** A new lever must be provably a no-op when disabled
  (the native MCTS verified byte-for-byte against the Python reference under fixed
  seed). You can add capability without risking the thing that already works.
- **H2H round-robin once anchored elo saturates.** The anchored ladder literally
  emitted `elo=6000` garbage once the net crushed it — so past a point, ranking
  switches to round-robin head-to-head. The ruler changes when it stops measuring.
- **The quality gate (`val/policy_ce` vs a frozen archive).** A throughput or elo
  win with degraded held-out CE is filed `needs_repeat`, never `promote` — it
  catches a net that "won" by overfitting its own self-play distribution.
- **Validate at flood scale, not in a unit test.** The ingest cost lessons only
  show up under live flooding; CPU-sim tests passed three times and the live race
  failed three times. The gate is the full-load re-race, not the green checkmark.
- **Metric-validity flags and the MTTE redesign.** When the lab caught itself
  optimizing a proxy that *harmed* the objective (see §4), it didn't patch the
  recipe — it designed an *unforgeable* metric (Minutes-To-Target-Elo: you can't
  fake it by flooding the trainer, because flooding adds wall-clock without adding
  elo). When verification itself proved gameable, they hardened the verifier.

**And then there's the part you really care about: compounding trust.** Verification
gives you trust in a *single* result. What makes the lab *autonomous* is that the
trust *accumulates* instead of resetting every session. That's the job of the wiki
+ memory + receipts + dated corrections. Every lane files a receipt; every durable
lesson becomes a wiki section *and* a memory; every reversal is a dated correction,
not a quiet edit. So a fresh session (or a fresh Claude after a context reset)
doesn't re-establish trust from zero — it inherits it. The north-star metric,
**Δelo/Δt** ("delta-e," the elo-gain *rate*, your "delta-v for training"), is itself
a verification discipline: measure progress from a common checkpoint over a fixed
window against a stable anchor, so the rate can't be gamed by throughput tricks. The
derby is the engine that allocates compute to whoever's earning Δelo fastest; the
wiki is the memory that means the derby's verdicts survive.

The reason this was hard — and why I think it's the part you'll be proudest of — is
that autonomy is *trivial* to build and *terrifying* to trust. Anyone can write a
`while True:` loop. Building one you can walk away from for five hours and come back
to find it correctly grew a net through three capacity steps, caught its own
gen-flood runaway, and *talked itself down from a false breakthrough* — that takes
the verification scaffolding to be real all the way down. It is.

## 4. What didn't work (the honest graveyard)

The dead ends are recorded as carefully as the wins — that's the discipline that
makes the wins trustworthy. The notable corpses:

- **Single-process "big batch" scaling.** Collapsing to 1 worker × huge batches to
  "saturate the GPU" *regressed* — the GPU sits idle during the long Python phases.
  More small workers beats fewer big batches at this model size.
- **Async gen+train in one process.** MPS is single-stream per process; overlapping
  forward (gen) and forward+backward (train) serialized and made the train step
  ~7× slower. Async only helps across multiple devices.
- **`torch.compile` with frequent weight reloads.** The compile cache amortization
  window is longer than the reload cadence — cycle time *grew*.
- **The LF1 runaway** (the sharpest lesson): a recipe won the throughput bench by
  **+152.9%** and then, in a real run, diverged to **7.3 min/epoch and climbing**.
  The benchmark measured a cold-buffer transient that real training only sits in
  for ~15 epochs; the recipe "won" by flooding the trainer faster than it could
  learn. *The proxy actively harmed the objective.*
- **ANE for production eval** (§2): a single narrow win, ~2× slower elsewhere, and
  not even the ANE.
- **The 18% heat-soak haircut** → withdrawn. **ANE contention-immunity** →
  falsified. **The depth-4 loss-tail** → an averaging artifact. **FPU reduction**
  (the eval flag that swept the 9×9 ladder to 100%) → *does not transfer to 15×15*.
  **Bigger nets at 9×9** → lose (the board's near-solved). **WL5's archive-start**
  → ran clean but didn't beat WL4's ceiling, cause still open. In the derby:
  **soft-policy aux loss, Mish activation, VCT teacher, and every attempt to stack
  orthogonal winners** all lost head-to-head.

The pattern across the graveyard: *the exciting early number was usually a
measurement artifact, and the honest verification killed it.* That's the fulcrum
doing its job.

## 5. Where we are now

As of this writing (2026-06-13), the GPU is busy with the live 15×15 campaign —
`G15-128x10-grown`, a 3.3M-param net, running on `main` (this writeup was done in a
worktree to stay out of its way). The state, honestly:

- **9×9 is closed.** Champion goes 43–3–74 vs Rapfi at fast TC; board exhausted;
  era formally over.
- **15×15 is climbing a capacity ladder.** 64×4 warm-started from the 9×9 champion
  → plateaued ~67% vs Rapfi → net2net-grown to 96×8, where **capacity pays at deep
  time control** (88% @5000ms vs 64×4's ~67%) → grown again to 128×10, currently
  *tracking* the 96×8's mature peak early (88% @5000ms at only ~175 epochs in).
  Every champion preserved; the gen-flood runaway on the big net already caught and
  fixed (workers 8→4).
- **The honest bound, stated plainly:** all of this is at **short time controls
  (≤5s/move).** Rapfi at full tournament time runs deep solvers; the edge may
  shrink or invert there. The long-TC eval is queued, not run. "Trades blows with a
  2625 engine at fast TC" is real and remarkable; "tournament-competitive" is
  untested, and the project says so.
- **The open frontier:** the WDL value head (the keystone untried lever, for the
  drawish regime), swap2 / renju openings (so the game is balanced and Rapfi's
  rated table applies on its own terms), the bit-packed 3M buffer (a prerequisite
  at 15×15 memory scale), and the long-TC eval that would settle the strength
  question.

---

So: the 90s trainer couldn't win a game. This one beats a top-ranked engine at
fast time controls, taught itself to defend, transferred that skill to a bigger
board, mapped the corners of a chip Apple won't document, and ran itself overnight
while you slept — and it can show you the receipt for every claim in this sentence.

The machine got 200,000× faster. The algorithm got good. But the thing that
actually closed the loop is the part you pointed at: **a way to verify, and a way
to make that verification compound into trust that doesn't reset.** That's the
fulcrum. Everything else is leverage on it.
