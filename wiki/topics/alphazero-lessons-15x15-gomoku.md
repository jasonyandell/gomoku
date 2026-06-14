# What We Learned About AlphaZero on 15×15 Gomoku

**Date:** 2026-06-13. **Why this page exists:** the goal of this project was
never a Gomocup ranking — it was *learning AlphaZero deeply* on a game Jason
first tried to crack in the 90s. **The learning is the artifact.** Strength is
gravy (welcome, pursued, but not the point). This page is the distilled
understanding from the 15×15 campaign — the "aha"s, made legible, separate from
the run logs. Companion to [15x15-training-campaign.md](15x15-training-campaign.md)
(the operational story) and the lab event log.

The full-circle is worth stating plainly: a net trained by self-play on a game
from a decades-old notebook learned, unsupervised, to *defend* — and that
behaviour transferred across board sizes. The rest of this page is the mechanism
behind that sentence.

## 1. Representation transfer is real, and dramatic

The single most striking result. We grew/seeded 15×15 nets from a strong 9×9
champion's convolutional tower (the board-size-independent ~94% of params).

- A **cold** 15×15 net (random init) slid straight into the classic fast-attack
  collapse: the policy sharpened on "build a line fast," neither side learned to
  block, and games ended in ~11 plies. Value-loss stayed *high* (the net wasn't
  even confident) — it was thrashing, not converging.
- A **warm-started** net — identical architecture, just seeded from the 9×9
  champion — played **defended ~85-ply games from epoch 0**, skipping the
  collapse entirely. Anchored elo ~1253 almost immediately.

The lesson: the 9×9 net's learned notion of *threat* ("an open four is
dangerous; respond to it") is **board-size-agnostic and lives in the conv
tower.** It transferred to a board 2.8× larger with zero retraining of those
features. You can practically watch the representation do its job. This is the
clearest demonstration of feature transfer the project has produced, and it's
the reason "best player we can make" was reachable at all on a laptop —
we never trained 15×15 strength from nothing.

**Durable corollary:** cold-start fast-attack collapse is a *real* failure mode
at 15×15, not a bug; warm-start is the remedy (see
[loss-floor-bouncing.md](loss-floor-bouncing.md) for the dynamics).

## 2. Net capacity and search are multiplicative, not additive

We climbed a capacity ladder (64×4 → 96×8 → 128×10) via **function-preserving
net2net** growth — each bigger net starts at the smaller one's *exact* function
(output-equivalence ~1e-4), so it inherits strength and trains into the extra
capacity. No cold restart, no collapse.

Measured against Rapfi (Gomocup freestyle engine) at two time controls:

| net | params | vs Rapfi 1000ms | vs Rapfi 5000ms |
|---|---|---|---|
| 64×4 | 0.44M | ~67% | ~67% |
| 96×8 | 1.55M | 62→62→**75%** | 75→**88%** |
| 128×10 | 3.3M | (early) 50% | (early) **88%** |

The shape of this is the lesson: **the 96×8's advantage over the 64×4 showed up
at *deep* time control first** (88% @5000ms) and only later at fast TC. A bigger
net evaluates positions better; that better evaluation *compounds with more
search*. Net quality × search depth is multiplicative — the AlphaZero
net/search duality, measured on our own board. (It also means a single
fast-time-control eval can *miss* a capacity gain entirely — see §4.)

### 2a. Capacity-bound vs data-bound: which lever moves the needle is *net-size-specific*

Having climbed the capacity ladder, we tested the **data axis** independently and
got the cleanest pair of results in the campaign:

- **96×8 + 3.75× more data didn't help.** We resumed the 75/88 champion (400k
  replay buffer) on a **1.5M bit-packed buffer** — same recipe, same net, ~100
  more epochs, reuse held at ~0.5 (high diversity). Re-eval at e597:
  **50% @1000ms / 88% @5000ms** vs the champion's 75/88. The deep-TC tier — the
  one that reflects strength — was **identical at 88%**. The training loss had
  flattened into a steady-state band well before the eval and *stayed there*.
  Verdict: **the 96×8's ~88% deep-TC is an architectural ceiling, not a
  data-starved number. More data was not the lever for this net.**
- **128×10 + only 400k data overfit** (§2 table: 37.5% aggregate @1000ms — the
  capacity *overshot* the data).

Put together, the lesson is sharper than either result alone: **whether data or
capacity is your binding constraint depends on the net size, and you can only
find out by varying one axis at a time and reading an external yardstick.** The
96×8 is *capacity-bound* (saturates its data, ceiling set by params); the 128×10
is *data-bound* (has params to spare, starves on 400k). The loss curve hinted at
both (96×8 plateaued; 128×10's eval cratered) but couldn't *name* the constraint
— only the paired A/B against Rapfi could. The natural next experiment writes
itself: **128×10 *with* the 1.5M buffer** — pair the spare capacity with enough
data to fill it, and see whether the joint move breaks past the 96×8's 88%
ceiling. (That this was even runnable depended on the MPS INT_MAX bit-packing fix
in §5 — a big net *and* a big buffer is exactly the regime that needs it.)

**Interim result (e146, the joint run G15-128x10-bigbuf).** Ran exactly that —
128×10 from the net2net seed on a fresh 1.5M buffer. First Rapfi read at e146:
**75% @1000ms / 50% @5000ms.** Two things to read carefully:
- *The big buffer cured the overfit.* The identical 128×10 net scored **37.5%
  @1000ms on the frozen 400k buffer; now 75% on 1.5M** — back to the 96×8
  champion's fast-tier level. The capacity-needs-data half of the thesis is
  confirmed mechanistically: give the 3.3M net enough fresh data and it stops
  overfitting.
- *But deep-TC isn't there yet (50% vs the 96×8's 88%), and that's expected at
  e146.* Recall §2: capacity strength at deep TC **emerges with training** — the
  96×8 itself climbed 75%→88% @5000ms over e173→e343. The 3.3M net has ~3× the
  96×8's parameters and learns into them *slower*; at e146 it is earlier on that
  curve than the 96×8 was at its first strong eval. So a low deep-TC here is a
  "not yet," not a "no." Honest status: **overfit cured (clean win), ceiling
  question still open** — re-eval pending at ~e250/e350 to see whether deep-TC
  climbs as it trains into capacity. (Discipline note: don't crown *or* bury a
  big-net run on its first eval — the net×search interaction takes training to
  show, and one n=8 deep-TC read is noisy besides.)

**Confirmation (e248): both tiers climb — capacity×data is working.** Trained
~100 more epochs and re-evaluated. The trajectory is the result:

| run / epoch | vs Rapfi 1000ms | vs Rapfi 5000ms |
|---|---|---|
| 96×8 champion (e499) | 75% | **88%** |
| 128×10+1.5M @ e146 | 75% | 50% |
| 128×10+1.5M @ e248 | **88%** | 75% |

Both tiers climbed as the 3.3M net trained into its capacity (fast 75→88, deep
50→75) — the §2 "trains-into-capacity" curve, made concrete. Two reads:
- *The fast tier now exceeds the 96×8 champion* (88% vs 75%). The bigger net,
  once it has the data, is the better fast-TC player.
- *The deep tier climbed 25 points (50→75) but isn't past the 96×8's 88% yet —
  and e248 is only **half** the 96×8's training (the 96×8's own deep-TC didn't
  reach 88% until ~e343).* So the two nets are roughly **tied in aggregate at
  e248 with opposite tier-strengths**, and the 128×10's curve is still rising.
  Ceiling-break verdict deferred to ~e350; the honest status is "on track, not
  yet proven."

The compound lesson across §2/§2a: **capacity and data are complements, not
substitutes.** The 96×8 saturated its data (capacity-bound); the 128×10 starved
on 400k (data-bound) then recovered on 1.5M; and with both levers pulled the
bigger net climbs on both tiers toward and past the smaller one — but *capacity
costs training time to convert into deep-TC strength* (more params ⇒ slower up
the net×search curve). On a fixed laptop budget that trade — bigger ceiling, but
you pay for it in epochs — is the whole planning problem in one sentence.

**Reality check (e348): the "climb" was partly noise — deep-TC has NOT broken the
ceiling.** A third read forced honesty. Full deep-TC trajectory across three n=8
evals:

| metric | e146 | e248 | e348 |
|---|---|---|---|
| vs Rapfi 1000ms | 75% | 88% | **88%** |
| vs Rapfi 5000ms | 50% | 75% | **62%** |

The fast tier is *robust*: 88% twice, durably above the 96×8 champion's 75%. But
the deep tier went **50 → 75 → 62** — not a climb, a **scatter**. Three n=8 reads
bouncing in a ±18% band around a true value of ~60–65%, **clearly below the 96×8's
88%**. The e248=75% that looked like "climbing toward the ceiling" was the high
sample of a noisy distribution; e348=62% walks it back. **No ceiling break.** As
of e348 the 128×10+bigbuf is the *better fast-TC player and the worse deep-TC
player* than the 96×8 — the exact opposite of the capacity-pays-at-depth story we
saw going 64×4→96×8.

Two lessons, both sharp:
- *The n=8 trap caught us — using our own stated rule.* We'd written "don't read a
  single n=8 deep-TC point" and then half-did exactly that at e248. The fix isn't
  a better adjective, it's a **bigger n**: a deep-TC comparison with ±18% error
  bars can't answer a "did it cross 88%" question. (Both sides are suspect — the
  96×8's "88%" is *also* one n=8 read. Pinning the real values needs n≥16 on both,
  which is the follow-up in flight.)
- *Capacity didn't transfer to depth here, and that's the interesting part.*
  64×4→96×8 bought deep-TC strength (capacity pays at depth). 96×8→128×10+data did
  **not** (so far): it bought fast-TC and cured the overfit, but deep-TC sits
  lower. Candidate reads, not yet separated: (a) still early — the 3.3M net needs
  e500+ to convert capacity into depth (it learns slower); (b) deep-TC ~88% is a
  **recipe/search ceiling** (sim count, MCTS shape), not a net-capacity ceiling, so
  a bigger net can't move it; (c) measurement noise is hiding a real ~75%. A
  higher-n re-eval plus training to ~e500 is what tells these apart.

**The n=16 head-to-head (e348): the ceiling itself was noise — and capacity REVERSED
at depth.** We then ran both nets at n=16, 5000ms, to pin the deep-TC values with
±12% bars instead of ±18%. The result corrected the whole record:

| deep-TC (5000ms) | n=8 (what we'd been quoting) | **n=16 (pinned)** |
|---|---|---|
| 96×8 e499 | 88% | **69% (11-5-0)** |
| 128×10+bigbuf e348 | 62% | **50% (8-8-0)** |

Two corrections fall out:
- *The "88% deep-TC ceiling" never existed.* It was a high n=8 sample; the 96×8's
  real deep-TC is ~69%. We'd threaded "88%" through a dozen status updates as if it
  were a fixed wall. The champion's honest record is **75% fast / 69% deep**, not
  75/88. **Lesson, paid twice now: a number you're going to reason against for hours
  deserves n≥16 the first time.**
- *But the ordering is robust, and it's the surprise:* **96×8 (69%) > 128×10 (50%)
  at deep TC**, while **128×10 (88%) > 96×8 (75%) at fast TC.** So capacity didn't
  just "fail to transfer" to depth — it **reversed**: the bigger net is the better
  *snap* player and the worse *deep* player. Capacity-pays-at-depth held 64×4→96×8
  and then flipped 96×8→128×10. That asymmetry is the real finding, and it's the
  kind of "should-have-worked-didn't" that's worth more than a clean win.

The leading mechanistic guess (testable, not yet confirmed): the extra capacity
made the **policy** sharper — great for fast, low-search play (it picks strong
moves immediately, hence 88% fast-TC) — but the **value/positional eval didn't
improve proportionally**, so when the *opponent* gets deep search (5000ms Rapfi),
it out-reads the 128×10's flatter value surface. More params bought confident
intuition, not deeper judgement. The "still-early" alternative (the 3.3M net just
needs more epochs to convert capacity into depth) gets its fair test at e500 — a
final n=16 deep-TC read. If 50% climbs toward/past 69%, it was early; if it sticks,
the reversal is real.

**Verdict (e502): the reversal is real, and the dissociation is the prize.** The
final n=16 deep-TC read at e502 — ~154 epochs past e348 — came back **50%
(8-8-0), identical to e348.** Flat. The "still-early" hypothesis is refuted: more
training did not convert capacity into depth.

What makes this worth more than a clean win is *what moved while strength didn't.*
Over those 154 epochs the 128×10's **value-loss set fresh lows** (0.172 → 0.152)
and its **self-play games lengthened** (~40 → ~55 plies, i.e. more defended, more
"mature" play) — every *internal* signal said the net was still improving. And
its **deep-search strength against Rapfi sat motionless at 50%.** The net got
measurably better at its own training objective and not one game better at deep
play. That is §3's "the loss lies" lesson in its most distilled form: *value-loss
is a fit to the self-play distribution; it is not deep-search playing strength,
and the two genuinely decoupled here.*

The full capacity arc, with every number now at n≥16 where it matters:

| net | vs Rapfi 1000ms (fast) | vs Rapfi 5000ms (deep) |
|---|---|---|
| 64×4 | ~67% | ~67% |
| **96×8 (champion)** | 75% | **69%** |
| 128×10 + 1.5M buffer | **88%** | 50% |

It's an **inverted-U in deep strength**: capacity paid at depth up to 96×8, then
96×8→128×10 *traded* deep strength for fast strength (deep 69→50, fast 75→88).
**96×8 is the deep-TC sweet spot for this game/recipe/budget, and we are past it.**
The honest mechanism is still the leading guess above (capacity sharpened the
policy, not the value's deep-search discrimination) — but whether it's that, a
policy/value tournament inside a fixed trunk, or the 15×15 recipe simply capping
deep strength near where 96×8 already sits, the *operational* conclusion is firm:
on a laptop budget, **scaling the net past the sweet spot buys you snap strength
and costs you depth.** For competitive (long-TC) play, the smaller 96×8 is the
better champion — the opposite of what "bigger net" intuition predicts.

`96×8 e499 (75 fast / 69 deep)` stands as the 15×15 champion.

**Methodological keeper:** a negative result ("more data didn't help") is a real
finding when it's a clean, single-axis A/B with a trustworthy external metric. It
*reallocates the search* — it told us to stop spending GPU on 96×8 data and move
to the capacity×data corner. The learning is the artifact; negative results count.

## 3. The loss number lies; the structure tells the truth

Policy-loss bounced all over a wide band (1.0–1.7) throughout healthy training.
Reading it literally would have triggered false alarms repeatedly. The signals
that actually track training health:

- **`plies_mean`** — game length. Collapsing toward ~10 = fast-attack failure;
  stable/high (~30–45) = defended, healthy play.
- **value-loss** — collapsing toward zero *while plies collapse* = the net is
  *confident* in bad fast-attack play = terminal. Value-loss steady/declining
  while plies hold = healthy maturation.

The rule we converged on: **judge training by the structural signals (plies, vl,
external strength), not the raw loss.** A rising policy-loss with stable plies
and declining value-loss is a net training on *harder* positions as it
strengthens — exactly what you want, and exactly what the loss number makes look
like a regression.

## 4. Evaluation discipline: small-n lies louder than the loss

Strength was measured vs a fixed external yardstick (Rapfi), because the
anchored heuristic/lookahead ladder saturates (it literally emitted `elo=6000`
garbage once the net crushed it). Two hard-won eval lessons:

- **Small-n is brutally noisy.** An 8-game eval is ±~20%. A single eval read
  100% (8-0) at one point — pure upward noise; the true rate was ~67%. **Weight
  aggregates across evals, never a single number.**
- **Wait for the capacity-sensitive measurement.** The 128×10's first eval read
  50% at 1000ms (looked like a *loss* vs the 96×8) — but 88% at 5000ms (matching
  the 96×8's mature peak, *early*). The 1000ms-only read would have wrongly
  killed a promising capacity step. The deep-TC tier is where capacity shows;
  read both tiers before judging.

**Negative results are findings too:** the FPU eval-lever that gave the 9×9
champion its 100%-ladder-sweep does **not** transfer to 15×15 (clean A/B, no
gain). Recorded so no future session re-chases it. Knowing what *doesn't* work
is half the map.

## 5. Systems shape what you can learn

Lessons that aren't about ML at all but gate it — the things 90s-compute
couldn't even reach to stub a toe on:

- **The dispatch-bound regime.** At this model size on Apple MPS, per-kernel
  launch overhead dominates compute — which is *why* moving 9×9→15×15 cost
  almost nothing at the training wave size (the GPU was idling through 9×9).
  The "ceiling" was the small model, not the Mac.
- **Gen/train balance, and the flood runaway.** The 3.3M-param trainer was slow
  enough that 8 self-play workers *flooded* it: games piled up faster than it
  could ingest, per-epoch ingest cost spiralled (62s → 313s), a positive-feedback
  runaway. Fix: **fewer workers** (decouple inflow from a slow trainer). Big nets
  want fewer workers. Generators outpacing the trainer is a recurring trap.
- **Smoke-first earns its keep.** A 90s smoke of the real recipe caught that the
  VCF teacher's 9×9 budget cost 3–9 s/game on the wide-open 15×15 board before
  any multi-day run was committed. Cheap validation before expensive commitment.
- **MPS has an INT_MAX tensor-dim cliff that flood-scale finds.** The bit-packed
  buffer (the #25 capability, merged + unit-tested + small-smoked) crashed a
  live run at ~550k positions: `MPSGraph does not support tensor dims larger
  than INT_MAX`. Root cause — a helper unpacked the *whole* buffer into one
  float32 planes tensor (board-15: 3825 elems/pos), crossing 2³¹−1 at exactly
  `floor(INT_MAX/3825) = 561,629` rows, hit every cycle via `shape_stats()`.
  CUDA/CPU don't have this dim cap; MPS does. Fix: chunk the unpack (≤2²⁸
  elems/chunk) so no single tensor crosses the cap — packed store stays uint8
  (the 32× win), only bounded chunks + the sampled batch ever unpack. **The
  lesson is the recurring one: unit tests + a small smoke passed three times;
  the bug only appeared at real (flood) scale** — sibling of the cross-game
  ingest trap and the gen-flood runaway. When a capability has a scale axis,
  the validation must exercise that axis, not just correctness in the small.

## 6. The methodology that emerged (the meta-lesson)

The campaign converged on a repeatable loop that *is* the transferable artifact:

> **warm-start** a new regime from the nearest strong net → **smoke** the real
> config before committing → train as a crash-resumable slice → **gate every
> step on a fixed external yardstick** (weighting aggregates, both tiers) →
> **net2net-grow the confirmed winner** into more capacity (function-preserving,
> no collapse) → swap GPU-serially, **always preserving the prior champion** →
> repeat. Read training health structurally (plies/vl), not by the loss.

That loop took a 9×9 net that beats a Gomocup engine 43-3-74 (short TC) and,
overnight on one laptop, produced a 15×15 net that trades blows with the same
engine at fast time controls and is still climbing a capacity ladder — every
step backed by evidence, every champion preserved, with two genuine bugs
(gen-flood runaway, FPU mis-transfer) caught and filed along the way.

## 7. Honest bounds (so the astonishment stays calibrated)

- Rapfi results are at **short** time controls (≤5 s/move). Rapfi at full
  tournament time runs deep VCF/VCT solvers; our edge may shrink or invert there.
  "Trades blows at fast TC" ≠ "tournament-competitive." (A long-TC eval is the
  test that would settle it — queued, not yet run.)
- 8-game tiers are hints, not proofs; we lean on trends across evals.
- We validated the Rapfi wrapper *runs*; we have not stress-verified it gives
  Rapfi its absolute best shot (threads, full config). A "beats engine X" claim
  deserves that scrutiny before it's load-bearing.

None of which dims the actual artifact: **we learned, concretely and on our own
board, how AlphaZero teaches itself to defend — and how to grow that skill.**
