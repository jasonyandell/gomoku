# The Story — how this project actually went

*The narrative read of the whole project, start to today. The
[timeline](training-timeline.md) is the milestone index; the [hubs](index.md)
hold the durable lessons; this page is the **story** — what we thought, what
the machine did to our theories, and how each era's defeat became the next
era's premise. Maintained synthesis (updated at era boundaries); last updated
**2026-07-04**.*

---

## Prologue — a thirty-year rematch

In the 1990s, Jason hand-rolled a distributed neural-net trainer and pointed
it at gomoku. It didn't crack the game. Three decades later the rematch is an
AlphaZero engine on a single M5 Max — PyTorch on MPS, W&B from day one, a
Claude in the loop — with three declared goals: **learn AlphaZero for real**,
**learn agentic engineering**, and **squeeze one Mac like a knowable
mainframe**. Strength is gravy. *The learning is the artifact*
([alphazero-lessons](topics/alphazero-lessons-15x15-gomoku.md) §0).

## Chapter 1 — The collapse (May 17–19)

The origin 9×9 run (`o9npssu1`) died the way every naive AlphaZero-at-home
dies: it collapsed into defensive draws by epoch 136. The diagnosis that
emerged — **fast-attack collapse** — became the villain of the entire project:
self-play sharpens the policy on attack, so neither side ever *faces* mature
threats, so defense is never punished into existence. Jason spotted the
leading indicator early: a falling `selfplay/plies_mean` with a concave
buffer-fill curve predicts the collapse before any eval confirms it.

We threw sweeps at every plausible upstream cause — K, buffer size,
games-per-cycle, continuous generation. **Every one failed.** The first
durable meta-lesson landed alongside: a "crossing" at n=4 games is noise, and
sibling head-to-heads are non-transitive — **only fixed external baselines
count** ([timeline Era 0](training-timeline.md)).

## Chapter 2 — The recipe (May 19–23)

The breakthrough was humbling: our runs weren't broken, they were **20–30×
undertrained**, and our targets were too hard. Porting the michaelnny AZ
recipe (soft targets at τ=0.1, buffer 50k→1.5M, AGZ log-PUCT) produced
`sppjo3z5` — the first run to *sustain* wins over the heuristic and then
lookahead-2, with plies regrowing from 11 to ~30: real, learned defense
(model_elo 1718 at its peak). The collapse could be beaten
([timeline Era 1](training-timeline.md)).

Meanwhile the machine itself became a research subject. Generation dominates
training **25–30×** in wall-clock, so a perf era opened under the
[M5-as-Mainframe](m5-mainframe.md) philosophy and disproved its own folklore
along the way — fp16-on-MPS was no longer slow (**+97%** eval), independent
levers composed multiplicatively to four decimals, and a bench that stops
before buffer-fill turned out to be
[non-predictive of real training cost](topics/perf-bench-vs-real-training-cost.md).

## Chapter 3 — Waves, lockstep, and the 9×9 throne (May 20–22)

The WL series (wave-of-lockstep) attacked training-quality with buffer
uniformity. WL1 oscillated; WL2's scale-emulation levers smoothed it without
solving retention; WL5's always-on diagnostics validated the pipeline without
beating the throne. WL4 set the 9×9 all-time-high (elo **1841**). The durable
residue wasn't any single lever — it was doctrine:
[loss-floor bouncing is healthy turbulence](topics/loss-floor-bouncing.md),
and most "wrinkles" are
[laptop-scale artifacts, not bugs](topics/az-at-scale-vs-laptop.md)
(see [wl-era](topics/wl-era.md)).

## Chapter 4 — The Derby (May 23–28)

To stop hand-shepherding runs, the lab became the [Derby](derby.md): recipes
race in time-capped slices, scored by **Δelo/Δt**, promoted on receipts, with
a Reviewer auditing every lane. Nine verdict rounds later the durable levers
were unglamorous — a matured champion and an eval-side
[FPU-reduction knob](topics/fpu-reduction-eval-lever.md) — plus a reframe
that stuck: when anchored elo saturates, **the ruler is the problem**, so
"distance to 100%" replaced elo as the local north star
([research-board](ops/research-board.md)). The autonomous loop itself worked:
the autolab later ran unattended overnight, crowned champions, and had zero
failures before it was stopped (the stop itself went unrecorded — work simply
moved on to 15×15 and VCT-science).

## Chapter 5 — 15×15 and the wound (Jun 12–19)

The 9×9 frontier closed with a certificate — the v8 champion took 43W-3L-74D
off a 2625-elo Rapfi config — and the codebase went board-size-parametric
(15×15 was nearly free on MPS; the old ceiling was dispatch-bound, not the
Mac). Then came the project's most instructive failure: **the yardstick
reckoning**. A broken Rapfi build (weightless, mis-clocked, wrong device)
silently inverted our rankings — we crowned the weakest net and abandoned a
tied-strongest one. Days of conclusions had to be retracted and re-derived; "**audit
the yardstick before believing any result**" became bedrock
([alphazero-lessons](topics/alphazero-lessons-15x15-gomoku.md)).

Out of the wreckage came the corrected champion — the 128×10 bigbuf net — and
with it the first honest contact with real opposition (2026-06-18): **black won
42% of games vs Rapfi; white won zero of twelve.** The entire deficit was
one-sided. We had found the wound. Four days later that lineage was
re-specialized as ["Bruce Lee"](topics/bruce-lee-model.md) — the
one-opening-ten-thousand-times bet (idx-2 only) that defined the next era.

## Chapter 6 — The theorem (Jun 20–25)

The wound turned out not to be a flaw in our net at all. Even Rapfi playing
white against itself loses ~90% of the time: freestyle gomoku's first player
simply wins. **White-side weakness is a theorem, not a bug** — so "fix white"
means fixing the *game*, not the net. [Swap2](topics/swap2-opening-protocol.md)
rebalanced the opening and confirmed its core bet at the data level (white
went from ~0% to ~27% of self-play games (~30–40% of decisive ones) — white
became *trainable*), while absolute strength stayed flat against a saturated ruler.

The same weeks exhausted the teacher fantasy:
[one-hot Rapfi distillation was catastrophic](topics/eval-teacher-sensei.md)
(trunk corruption, 0/96), the gentle retry also regressed, and DAgger showed
the think-time wall never moves — **our net was eval-capped, not
search-capped**. Every eval-side lever for defense had been
[falsified one by one](topics/white-side-defense-plan.md). Something
categorically different was needed.

## Chapter 7 — The oracle (Jun 25–28)

The different thing was to stop asking the net to do what search does
perfectly. In four days the GPU went from "solvers don't batch" to a
[batched VCF crack (~2,500× CPU)](topics/gpu-vct-feasibility.md) to the
[bitboard VCT megakernel](topics/mega-vct-solver.md) — on-device, ~1600× CPU,
zero false positives or negatives. The CPU solver was retired; `mega_vct_bb`
became **the oracle**, and the [seek-VCT program](seek-vct.md) crystallized
around one thesis: **the net steers, the oracle finishes** — anti-correlated
tractability.

Mining the corpus with the oracle rewrote our intuitions: the first forced
win arrives at **median ply 19**; the pre-onset region isn't forgiving but a
**knife-edge** (up to 98% of alternative moves lose by force); a
[learnability trilogy](topics/vct-recognition-learnability.md) showed nets
can *see*, *steer toward*, and *regress the distance field of* VCTs — with a
plain CNN beating attention all three times. And Jason independently
rediscovered [the claw](topics/the-claw.md), a mod-5 defensive lattice that
is provably optimal and provably invisible to line-organized evaluation — a
hint about representation that still shapes the architecture debate.

## Chapter 8 — The sound world (Jun 30 – Jul 3)

First attempt to cash the thesis in training:
[VCT-terminus self-play](topics/vct-terminus-selfplay-result.md) (end every
game at the first provable win) was a throughput *win* and a robustness
*loss* — 0-of-120 against the play-to-five control. Training longer didn't
help: the defensive ceiling is a **structural attractor**, and a supervised
defense head learned the representation while the policy ignored it — a
["sensor with no actuator"](topics/vct-defense-aux-head-result.md).

The fix was to edit the *games*, not the targets: the
[**sound-world recipe**](topics/sound-world-recipe.md) — oracle-veto (no
training game may contain an unpunished blunder) plus terminus plus line
planes — structurally killed the 9–10-ply attractor. On 2026-07-02 the **9×9
chapter closed**: the sound-world net drew the old champion 0-0-40, the
net+finisher hybrid took 95% off the heuristic, and the oracle settled the
game itself — 9×9 freestyle is drawish between sound players, but a fast
**black win within the cap50 horizon**
([sound-world-recipe](topics/sound-world-recipe.md) §idea 3).

Graduation to 13×13 then failed **structurally** (#113): warm-start and
from-scratch both produced attack-only specialists — white 0/20. The recipe
that closed 9×9 does not yet scale. And with the oracle veto eating ~91% of
13×13 generation wall-clock, the perf lever moved into the solver itself —
the 2026-07 blitz (cap25, `lanes=K`, one-worker refill) closed the gen-loop
thread ([mcts-perf-ceiling](topics/mcts-perf-ceiling.md)).

## Chapter 9 — Now (Jul 2026): the open question

The first rebuild was tried immediately: **rails-v0** (2026-07-03, #116) dropped
the terminus and kept both sides oracle-sound — and the cure *worked* (white
positions finally get recorded) while exposing the next layer of the onion: on
the black-tilted idx-2 opener, black's forced-win tails **poison the value
head** (vl→0.03, the death-tell tripped) and white re-collapses. Removing the
terminus is necessary but not sufficient on an unfair opening. Next levers, in
order: **tail subsampling** (#118, staged and gated) and a **fairer opening**.
And the pivotal unknown the whole arc has been narrowing toward: **is 13×13 a
forced black win?** 15×15 is proven (Allis), 9×9 is drawish between sound
players (a fast black win within the oracle's cap50 horizon), 13×13 is open —
and we own a ~1600× solver to probe it with. See the
[AlphaZero hub](alphazero.md) and [sound-world-recipe](topics/sound-world-recipe.md)
§next.

## Epilogue — what we actually learned

The era-independent spine, each earned the hard way:

1. **Fast-attack collapse is the default fate** of self-play at this scale;
   watch `selfplay/plies_mean` and buffer-fill concavity.
2. **Audit the yardstick first.** Broken rulers cost us more days than broken
   nets ever did; gate on H2H vs a *frozen champion*, never siblings.
3. **White is losing because gomoku is unfair**, not because the net is dumb —
   know when a wound is a theorem.
4. **The net steers, the oracle finishes** — put learning where approximation
   is tolerated and search where it isn't.
5. **Curriculum beats targets**: editing which games exist (veto, swap2,
   terminus) moved what no loss-function patch could.
6. **The machine is knowable**: folklore falls to receipts (fp16 reversal,
   dispatch-bound ceilings, multiplicative levers).
7. **We try, we learn, we write it down** — the wiki compounds; failures are
   banked as carefully as wins.

*Corrections welcome; keep this page honest at era boundaries and link every
claim to its evidence page.*
