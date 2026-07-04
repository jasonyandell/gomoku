# The 15×15 yardstick reckoning — FULL chronicle (archived)

**Status: HISTORICAL / ARCHIVE — full-fidelity verbatim, 2026-07-04.** These are the complete,
blow-by-blow §2/§2a (capacity ladder), §8 (the yardstick was the weak link, 8A-8H), §9 (the
head-to-head reckoning), §10 (the silent self-play regression), and §11 (capacity×search
confirmed clean) sections, moved verbatim out of
[alphazero-lessons-15x15-gomoku.md](../../topics/alphazero-lessons-15x15-gomoku.md) during the
2026-07-04 curation. They are preserved intact because they *are* the learning artifact — a
multi-day chain of measurements, retractions, and corrections. The parent page now carries the
**settled conclusions** up top and a compressed forward-reading summary in place of these
sections; this archive holds every table, retraction banner, and mechanistic guess. No facts
were changed — only relocated. Read the parent's "Settled verdicts" first, then come here for
the full evidence trail.

---

## 2. Net capacity and search are multiplicative, not additive

> ⚠️ **RETRACTED by §8–§9 (2026-06-15).** Every strength number below is vs a
> *broken* Rapfi yardstick (weightless classical, ignores its own search time, so
> the "tiers" are one shallow engine measured twice). Direct head-to-head reverses
> the conclusions: 128×10 is **not** worse ("the reversal"), it *ties the strongest
> and beats the 96×8 "champion" 40-0; the crowned 96×8-e499 is actually the
> *weakest* trained net. Read §2/§2a as **evidence of how the metric misled**, not
> as the capacity verdict. Real capacity curve = future same-epoch head-to-heads.

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

> **2026-06-15 CORRECTION:** `96×8 e499` is NOT the champion and is not even a
> trained improvement over its own seed. See §9 (head-to-head overturns rankings)
> and §10 (the silent self-play regression). The strongest preserved net as of
> 2026-06-15 is `g15_128x10_bigbuf_eval502.pt`, with the caveat that the absolute
> strength ranking awaits a fixed yardstick (#28) and a clean same-epoch capacity
> ladder (#29). Do NOT use this line to choose a production checkpoint.

**Methodological keeper:** a negative result ("more data didn't help") is a real
finding when it's a clean, single-axis A/B with a trustworthy external metric. It
*reallocates the search* — it told us to stop spending GPU on 96×8 data and move
to the capacity×data corner. The learning is the artifact; negative results count.


---

## 8. The yardstick was the weak link — §7's caveat, made concrete (2026-06-15)

§7 flagged two soft spots in the Rapfi yardstick ("not stress-verified it gives
Rapfi its best shot"; "short TC, our edge may shrink"). A fresh autonomous
session pulled that thread and found **both the opponent and the measurement were
weaker than the record implies.** This is the "should-have-worked-didn't" of the
*evaluation itself*, and it partially re-opens §2a.

### 8A. The Rapfi binary is the weightless *classical* build — not the rated ~2625 NNUE engine
`engines/rapfi/build_rapfi.sh` builds `pbrain-rapfi` with Rapfi's **internal
classical config, no NNUE weights** (the script says so outright; the strong
NNUE evaluator needs the `Networks` submodule + `mix9svq` weights + `--config`).
The "Rapfi (Gomocup freestyle ~2625)" provenance stamped on every eval row is the
**NNUE** engine's rating; our yardstick was the much weaker weightless build.
Symptom that exposed it: re-running the champion's *exact* recorded deep-TC config
uncontended gave **~90–100%**, not 69% — and watching `top`, the classical Rapfi
sat at low CPU and **moved fast** (it wasn't using its time budget), whereas the
NNUE build pegs a core at ~97% and uses its full per-move time. So "trades blows
with a Gomocup engine" overstated it: we were beating *weak* Rapfi.

**The fix (no rebuild needed — done this session):** the `mix9svq` weights were
already present locally and `pbrain-rapfi --config <toml>` works (COMMAND_MODULES
is compiled in). A config pointing at the freestyle NNUE weights + a one-line
wrapper stands up the **strong NNUE Rapfi** as the honest yardstick. Engine
reports `Evaluator set to mix9svq` and actually searches (97% CPU).

### 8B. A single n=16 deep-TC read is not a measurement — it's device- and load-dependent
The §2a reversal verdict (96×8 = 69% deep > 128×10 = 50% deep) rests on **one
n=16 read per net, never repeated.** §4 says "weight aggregates, never a single
number" — the load-bearing reversal number violated our own rule. Re-measuring the
champion (96×8 e499, sims=100, @5000ms vs the *classical* Rapfi) gave, across
fresh runs:

| condition | n | win-rate | wall |
|---|---|---|---|
| campaign record (device unknown) | 16 | **69%** (11-5) | 493s |
| this session, MPS | 16 | **100%** (16-0) | 214s |
| this session, MPS (sims=200) | 16 | 75% (12-4) | 327s |
| this session, MPS | 24 | **96%** (23-1) | 323s |
| this session, **CPU** | 24 | **79%** (19-5) | 429s |

Two effects, both real: **(1) a device gap** — the same net/config scores
**MPS 96% vs CPU 79%** (n=24). `ladder_eval` *defaults to `GOMOKU_DEVICE=cpu`*
(chosen to dodge GPU contention during training); that default **systematically
understated model strength.** **(2) run-to-run variance** larger than the binomial
bars we quoted — 69→75→96→100 on nominally one config. The honest reading: the
champion beats *classical* Rapfi @5000ms somewhere ~85–95% (not 69%), and the
19-point "reversal" gap (~1.4σ at n=16) was never powered enough to be load-bearing.

### 8C. The biggest one: Rapfi never used its search time — the TC tiers were illusory
With `message_mode="normal"` the engine prints its search. On a contested 15×15
position, `timeout_turn 5000`:
```
MESSAGE OptiTime 4473ms | MaxTime 4970ms        <- time budget parsed correctly
MESSAGE Speed 455K | Depth 10-11 | Eval 501 | Node 455 | Time 1ms   <- stops at 455 nodes / 1ms
```
Rapfi **budgets ~5 s and then self-terminates after ~500 nodes / ~1-2 ms / depth
~10**, on every position type (opening, contested tangle, quiet) and every
`timeout_turn` (200/1000/5000/15000ms — all return in ~0.08 s). The search
*converges and stops* far under budget (≈0.1 % of it). Net effect: **the
"1000 ms" and "5000 ms" tiers the whole campaign quoted were the SAME ~depth-10
Rapfi.** The fast-TC/deep-TC distinction — and therefore the "capacity pays at
*depth*" and "reversal at *depth*" stories built on it — was measuring **one
shallow engine twice.** The 75/69 vs 75/88 vs 50 scatter was noise between
identical conditions. (Cause not fully root-caused: not the time fields, not
`timeout_match`, not the candidate range — Rapfi's iterative deepening just calls
the position resolved at ~depth 10. A genuinely deep Rapfi would search millions
of nodes in 5 s; ours searches hundreds.)

### 8D. Honest re-baseline (vs the NNUE engine, MPS, n=20, sims=100)
| net | vs NNUE Rapfi 1000ms | vs NNUE Rapfi 5000ms |
|---|---|---|
| 96×8 champion (e499) | **88%** (17-2-1) | **100%** (20-0-0) |

NNUE is confirmed loaded (`mix9svq nnue: load weight ... weight loaded in 15ms`).
The champion beats even the NNUE engine ~90-100% — but read with 8C: this is vs a
**shallow** (~depth-10) NNUE Rapfi, and freestyle gomoku is a **first-player win**,
so ~half those wins are the black-side forced-win advantage. "Beats Rapfi" means
"beats a shallow Rapfi, often while holding the first-move win." There is little
clean headroom here to *measure training progress* — which reframes what a useful
yardstick must be (next).

### 8E. The transferable lesson — audit the yardstick first
Every §2/§2a verdict was gated on an external metric that was **un-audited on three
axes that all mattered**: opponent evaluator (classical vs NNUE, 8A), measurement
reliability (device + n, 8B), and **whether the opponent actually searched** (8C).
The *relative* capacity-arc shape may survive (all nets measured the same broken
way), but the *absolute* "trades blows with a 2625 engine" framing was wrong on
every axis, and "the reversal is real (at depth)" is unsupported once you know
both tiers were the same shallow engine. **Audit the yardstick before it audits
your conclusions:** verify the opponent is at full strength (NNUE *and* actually
deep-searching), the read is reproducible across device and seed, and — for a
first-player-win game — that openings are **balanced** (swap2, #22) so strength
isn't dominated by who moved first. A fixed external metric is only as trustworthy
as those checks; they are far cheaper than the conclusions they protect.

### 8F. The deepgen experiment — deeper self-play SPECIALIZES, it doesn't strengthen
The thing the broken yardstick was *meant* to measure: does deeper self-play
search (n_simulations 100→200, warm-started from the champion) make a stronger
net? Trained `G15-96x8-deepgen` ~200 epochs at 200 sims. Internal signals all said
"improving": value-loss fell to fresh lows (0.19→0.15), plies stayed high (~42,
defended). Vs the (shallow) NNUE Rapfi it scored **83% @5000ms — indistinguishable
from the champion.** Looked fine. It was not. **Direct head-to-head, the only
yardstick-free test** (match.py validated: champion-vs-self = 6-6 = 50%):

| matchup (sims both sides) | deepgen win-rate |
|---|---|
| deepgen vs champion @100 | **0%** (0-40) |
| deepgen vs champion @200 | **50%** (10-10) |

The shape is the finding: deepgen is **search-specialized**, not stronger.
- At its *training* search depth (200) it's merely *even* with the champion — the
  deeper self-play bought **no strength**.
- At the *standard* depth (100) it's **catastrophically worse** (0-40) — it
  *lost the ability to play with shallow search.* Training on 200-sim MCTS policy
  targets taught the net to **offload judgement to the search**; strip the search
  and the raw policy/value can't stand alone. The champion (trained at 100) is
  robust at both 100 and 200 sims; deepgen is brittle below its training depth.

Two compounding lessons:
1. **A net inherits the search budget it was trained under.** Deeper self-play
   doesn't add free strength; it shifts the net's operating point and makes it
   *depend* on that depth. (Practical: don't train at a sim count you won't deploy
   at, and don't expect more self-play sims to be a strength lever — it's a
   specialization lever.)
2. **This is "the loss lies" (§3) in its sharpest form, and it proves §8C/8E.**
   vl↓ + plies↑ + 83% vs Rapfi *all* pointed up while the net was getting *worse
   for real deployment*. Three "improving" signals, one true one (head-to-head),
   and only the true one was trustworthy. **The shallow Rapfi yardstick didn't
   just mis-scale strength — it actively hid a regression.** Had we trusted it
   (as the campaign trusted its tiers), we'd have shipped a strictly-worse net.

### 8G. Eval mechanics that bit us chasing 8F (record so they don't bite again)
- **Eval the EMA/published weights, not raw `epoch*.pt`.** Same deepgen epoch:
  raw `epoch*.pt` = 35% vs NNUE Rapfi, EMA `worker_weights.pt` = 83% — a 48-pt
  gap. `epoch*.pt`/`latest.pt` carry the *raw* training weights (transiently weak
  mid-training); `worker_weights.pt` is the EMA that actually plays. A first
  "deepgen cratered to 30%" alarm was purely this artifact.
- **Eval-during-training contends** (suggestive): the champion scored 96-100%
  uncontended vs 75% (n=12) eval'd concurrently with an 8-worker run. Partly noise,
  but prefer uncontended evals — or, for net-vs-net, note that contention is
  *unbiased* (both sides sims-limited and equally slowed), which is another reason
  the **direct head-to-head is the robust measure**.

### 8H. The net-vs-net head-to-head is the yardstick we should have had
Every problem in 8A-8G traces to leaning on a weak, mis-scaled, sometimes-broken
external engine. The campaign avoided sibling head-to-head as "non-transitive" —
true for *ranking a pool*, but for the specific question **"is net B better than
its parent A?"** a color-alternated A-vs-B match is the cleanest, cheapest,
yardstick-free signal there is (no engine, no NNUE config, no time-control bug,
contention-unbiased). It is what exposed deepgen. Keep the external ladder for an
*absolute* rating once it's fixed (NNUE + deep + swap2), but gate "did this change
help?" on the direct head-to-head against the preserved champion.

## 9. The reckoning — head-to-head RE-RANKS the campaign (the yardstick inverted it)

§8 said the yardstick was broken. §9 is what that *cost*: we ran the preserved
nets against each other directly (color-alternated, sims-matched, match.py
validated: champion-vs-self = 50%). The result overturns the campaign's headline
rankings.

**The ladder (head-to-head, @100 sims, n=40):**

| matchup | winner |
|---|---|
| 128×10 (e500) vs 96×8 **champion** (e400) | **128×10, 40-0** |
| 64×4 (e900) vs 96×8 **champion** (e400) | **64×4, 40-0** |
| 128×10 (e500) vs 64×4 (e900) | tied, 20-20 |
| 96×8 champion (e400) vs deepgen (96×8, e620) | champion, 40-0 |

**Order: {128×10, 64×4} ≫ 96×8-"champion" ≫ deepgen.** The two nets at the top
both **beat the crowned champion 40-0**; 128×10 does it at 200 sims too (20-0).

Two things the broken yardstick (§8) did to us:
1. **It crowned the weakest trained net.** `96×8 e499` was declared the 15×15
   champion (75/69 vs Rapfi). Head-to-head it is the *weakest* of the well-trained
   nets — beaten 40-0 by both a bigger net *and a smaller one*. The Rapfi tiers
   that "ranked" it were the same shallow engine twice (§8C); the ranking was noise
   dressed as signal.
2. **It made us abandon a tied-strongest net.** `128×10+bigbuf` was retired on the
   "capacity reversed at depth, it's worse" verdict (§2a). Head-to-head it ties the
   strongest net and crushes the champion. **The "reversal" never happened** — it
   was the yardstick mis-ranking a *better* net as worse.

**Honest caveat (don't over-read capacity):** epochs are confounded — 64×4 has
e900, 128×10 e500, the champion only e400. A 64×4 (0.44M) tying a 128×10 (3.3M) is
almost certainly the smaller net's *3× extra training* compensating for capacity,
not "small = big." A clean capacity claim needs **same-epoch** head-to-heads
(future work). What is *not* confounded and *is* the lesson: **the Rapfi-based
ordering is wrong end-to-end, and a same-architecture continuation (cont100: 96×8
e400→e673) stays 50% vs its own e400 — so the champion was a real plateau that the
two stronger nets simply sit above.**

### The meta-lesson (the whole night in one line)
**A broken yardstick doesn't add noise — it can invert your ranking, so you crown
the worst option and discard the best, with every internal signal applauding.**
Across the night the same shape recurred five times: deep-TC tiers (§8C), the
"88% ceiling" (§4), the capacity "reversal" (§2a/§9), the deepgen "improvement"
(§8F), and the champion selection (§9) — *every one* was the measurement lying,
and *every one* dissolved under a direct head-to-head. The cheap, robust insurance
we should have had from day one: **a color-alternated match against the preserved
champion gates every "did this help?" decision.** No engine, no config, no
time-control, contention-unbiased. The external ladder is for an *absolute* rating
once it's actually fixed (NNUE + genuinely deep search + balanced openings, #22/#28);
it is **not** safe to rank with until then.

### What this changes operationally
- **`96×8 e499` is NOT the champion.** `128×10+bigbuf e502` (or `64×4 e909`) is the
  strongest preserved net; both beat 96×8 40-0. Re-crown via head-to-head, not Rapfi.
- **Re-run the capacity ladder at matched epochs**, head-to-head, to get the *real*
  capacity curve (the inverted-U "reversal" is retracted; shape unknown until then).
- **Fix the yardstick** (#28): NNUE evaluator (done, but it under-searches too),
  force genuine search depth, swap2 balanced openings (#22), n≥40, fixed device.
- Frozen artifacts: `g15_128x10_bigbuf_eval502.pt` (real strongest), `g15_champion_
  e909.pt` (64×4, tied), `g15_96x8_deepgen_searchspec_e621.pt` (the cautionary tale).

## 10. The silent self-play regression — 400 epochs of training made it 40-0 worse (2026-06-15)

§9 revealed the 96×8 "champion" was the weakest trained net. This section documents
a deeper finding: it is weaker than its own **untrained** starting point. The crowned
champion regressed *below the seed it was grown from*.

### The result

The 96×8 net2net seed (`g15_96x8_seed.pt`) is the function-preserving grow of the
64×4 champion — output-equivalent to within <1e-4 before any training. After 400+
epochs of v8 self-play, the trained "champion" (`g15_champion_96x8_e499.pt`) does
not improve on that seed; it is catastrophically **worse**:

| matchup (@100 sims, n=40) | result |
|---|---|
| seed vs champion-e499 | **seed wins, 40-0** |
| seed vs 64×4-e909 | **50-50** (seed equals the 64×4) |
| champion-e499 vs 64×4-e909 | **0-40** (champion loses) |
| champion-e499 vs 128×10-e502 | **0-40** (champion loses) |

400 epochs of self-play training on the v8 recipe left the 96×8 net 40-0 *worse*
than its starting point. The seed ties the second-strongest net in the campaign;
the trained "champion" loses to every other net, including its own untrained
predecessor. Training ran — and by every observable signal, ran *well* — and
produced a net less capable than what went in.

### The deception: internal metrics stayed healthy throughout

This is §3's "the loss lies" at its most extreme. Over the full 400-epoch run,
every standard diagnostic signal pointed *up* or held steady:

- **`plies_mean` 30–48.** Games were defended, non-trivial — no fast-attack collapse.
- **Value-loss 0.17–0.25.** Steady, declining — a net fitting its self-play
  distribution normally.
- **Internal-ladder win-rates 85–100%.** The net beat internal heuristic/lookahead
  opponents convincingly throughout.
- **Rapfi yardstick (broken, §8):** the run was compared against a classical
  (weightless) Rapfi that ignores its own search time; the yardstick showed no
  regression. The external signal was also blind.

Not one observable metric flagged the regression. With a broken external yardstick
(§8) *and* deceptive internal metrics, the campaign crowned a net that had trained
itself **backwards** — and had no way to know.

This is the same pattern as the deepgen experiment (§8F): vl↓ + plies↑ + reasonable
Rapfi score, while the net was actually getting worse for real deployment. It recurred
across two independent runs, which makes it a **systematic hazard**, not a one-off.

### The net2net grow was not the cause

The grow step itself is validated: the seed's output deviation from the 64×4 source
is <1e-4, and the seed *ties* the 64×4-e909 in head-to-head. The regression is not
in the grow; it is in the **subsequent training**. Why the v8 recipe silently craters
a 96×8 net via self-play is an open question. Candidates:

- **Self-play distribution drift.** The 96×8 net's wider capacity may shift the
  self-play distribution in a direction the training signal cannot self-correct.
  The 64×4 and 128×10 (at different ends of the capacity spectrum) did not regress
  below their seeds; the middle-capacity 96×8 did, twice.
- **Recipe mismatch for this net size.** The v8 hyperparameters (buffer 400k,
  SGD schedule, opponent mix) were tuned for a 64×4 net and extended unchanged to
  96×8. The combination may be unstable for this parameter count.

### The fix: head-to-head against the seed is a mandatory gate

A head-to-head vs a **preserved reference** (the seed or current best) is not
optional for AlphaZero training. It is the only signal that reliably detects this
failure mode. The complete gate:

1. **Champion-promotion gate.** Before declaring any net the new champion, run
   a color-alternated match against the prior champion. A net that loses to
   its predecessor is not a champion.
2. **Regression-against-seed gate.** Before continuing a run past ~100 epochs,
   run a spot-check vs the frozen seed. If the trained net is losing to an
   untrained net, something is wrong — stop, diagnose, don't continue training.
3. **Do not trust internal metrics alone.** plies/vl/internal-ladder can all be
   healthy while head-to-head regresses. The external check is not a confirmation
   of internal signals; it is a *different* measurement of a *different* quantity.

### The G15-96x8-redo experiment

The `G15-96x8-redo` cell re-trains from `g15_96x8_seed.pt` with the
**byte-identical v8 recipe** to test reproducibility.

### Result (2026-06-15)

**The redo reproduced the regression — this is a recipe-level hazard, not a
one-off.**

Head-to-head vs the frozen seed (`g15_96x8_seed.pt`), @100 sims, n=40:

| checkpoint | vs seed | note |
|---|---|---|
| redo e127 | **40-0 (redo wins)** | trained up; better than seed |
| redo e234 | **40-0 (redo wins)** | peak |
| redo e377 | **50-50** | regressed back to seed level; also 50% vs 64×4-e909 |
| (continuing) | dropping further | tracking toward original run's e499 = lost to seed 40-0 |

The clean re-train from the good seed, with the byte-identical v8 recipe, peaked
around e234 then trained backwards — the same arc as the original run.

**Two reinforced lessons:**

1. **Gate-and-freeze the peak live.** The e234 checkpoint (a genuinely strong 96×8,
   beating its seed 40-0) was lost to `keep_last_n` pruning because it was not
   frozen when the head-to-head gate flagged it. The gate must freeze the peak in
   real time, not retrospectively.
2. **Stop at the head-to-head peak, not a fixed epoch.** The campaign froze e499
   (post-peak, regressed) as "champion." For a peak-then-regress recipe, training
   to a fixed epoch guarantees you archive the wrong checkpoint.

**Open question:** Why does the v8 recipe produce peak-then-regress specifically at
the 96×8 capacity point when 64×4 and 128×10 did not visibly do this? Candidate:
the recipe (buffer 400k / opponent-mix / SGD schedule) was tuned for 64×4 and is
unstable at 96×8 capacity. Recipe surgery is the required follow-up before 96×8 can
contribute a clean capacity result.

### Correction (2026-06-15, later)

The "redo reproduced the regression" conclusion above was overstated. The full
trajectory vs the frozen seed (@100 sims, n=40) is:

| checkpoint | result | win% |
|---|---|---|
| redo e127 | 40W-0L | 100% |
| redo e234 | 40W-0L | 100% |
| redo e377 | 20W-20L | 50% |
| redo e440 | 20W-0L-20D | **75%, zero losses** |

This is **noisy and non-monotonic** (100 → 100 → 50 → 75). The redo declined from
its early peak but **never fell below seed** — zero losses at e440. The earlier
"regression reproduced" reading over-weighted the single e377=50% dip. Moreover,
the run was **stopped at e439**, before the epoch (~e499) where the original
96x8-grown run collapsed to losing-the-seed 40-0. We did not observe that
catastrophic below-seed drop in the redo at all.

**Honest status: INCONCLUSIVE.** Whether the v8 recipe systematically regresses
96×8 (vs the original run being partly anomalous) is unresolved. A clean re-test
would train past e499 with a live head-to-head gate that freezes the peak and
watches for a sustained drop below seed.

**Meta-lesson:** This correction is itself an instance of §4/§8's discipline. We
concluded "regression reproduced" from one noisy n=40 read at e377 — exactly the
single-read trap the whole campaign is about. n=40 single reads are noisy; trends
need multiple reads AND enough epochs to be load-bearing before a conclusion is
published.

## 11. Capacity × search IS multiplicative — confirmed clean, the reversal was the artifact (2026-06-15)

§2 originally claimed that net capacity and search depth are multiplicative — a
bigger net's better position evaluation *compounds* with search, so the capacity
advantage is invisible at shallow depth and decisive at deep. §2a then appeared to
refute this: the 128×10 was *worse* than the 96×8 at deep time control (the
"reversal"). §8 showed the deep-TC tiers were a broken yardstick (both "tiers" were
the same shallow-Rapfi engine). §9 confirmed 128×10 is *stronger* than 96×8, not
weaker. What remained open: is the net×search interaction actually multiplicative,
or just "bigger = better at all depths"?

A clean yardstick-free test now closes the question.

### The experiment

Color-alternated head-to-head matches via `gomoku.match`, validated (champion-vs-self
= 50% at both sim counts, n=40 throughout). Three nets with confirmed relative
ordering from §9: `64×4-e909`, `96×8-seed`, `128×10-e502`.

### Results

**@100 sims — all three nets are EQUAL:**

| matchup (@100 sims, n=40) | result |
|---|---|
| 64×4-e909 vs 96×8-seed | **50-50** |
| 96×8-seed vs 128×10-e502 | **50-50** |
| 64×4-e909 vs 128×10-e502 | **50-50** |

At shallow search, net size barely matters. The three nets — spanning a 7.5× range
in parameter count (0.44M to 3.3M) — play to a complete draw.

**@200 sims — 128×10 dominates:**

| matchup (@200 sims, n=40) | result |
|---|---|
| 64×4-e909 vs 128×10-e502 | **0-40 (128×10 wins every game)** |

With deeper search, the bigger net pulls decisively ahead. The same 128×10 that tied
the 64×4 at 100 sims beats it 40-0 at 200.

### Interpretation

The shape — tie at 100, domination at 200 — is the multiplicative signature. A bigger
net evaluates positions more accurately; that better evaluation *compounds* with every
additional simulation. At 100 sims the compound effect is too small to show; at 200
sims it is decisive. **Capacity × search depth is multiplicative, not additive.** This
is exactly the net/search duality §2 originally described ("the 96×8's advantage showed
up at deep TC first") — now confirmed directly via net-vs-net, no yardstick required.

The §2a "reversal" (128×10 was *worse* at depth) is hereby retracted as an artifact.
§8C showed that the "deep" and "fast" Rapfi tiers were one shallow engine measured
twice — there was never a depth axis to reverse on. The broken yardstick obscured the
multiplicative structure that was present all along. §2 was right; §2a's reversal
finding was the measurement lying.

**Re-crowning: 128×10 is champion specifically *because of* depth.** §9 established
128×10 as co-leader (tied with 64×4 @100 sims). This result explains *why* 128×10
deserves the crown: the tie is shallow-search parity; the separation that matters
for real play (deeper search) belongs to the bigger net. The 128×10 is not just
"tied at the top" — it is the net that converts more search into more wins.

### Honest caveats

1. **Epochs and lineage differ.** `64×4-e909` has ~400 more training epochs than
   `128×10-e502`. This is not a pristine same-everything capacity isolation. The
   @100-tie / @200-win *shape* is the multiplicative signature regardless of absolute
   epoch counts — but a clean same-epoch capacity ladder (issue #29) would let us
   read the magnitude more precisely.
2. **96×8-seed vs 128×10 @200 not yet tested.** The 200-sim result is 64×4 vs
   128×10 only. Testing 96×8-seed vs 128×10 @200 is the obvious next check — it
   would complete the triangle and confirm the effect generalizes across the full
   capacity range.
3. **100/200 sims are modest.** The multiplicative effect should strengthen further
   at higher sim counts (400, 800). These results establish the shape; the magnitude
   at production search depths is still to be measured.

