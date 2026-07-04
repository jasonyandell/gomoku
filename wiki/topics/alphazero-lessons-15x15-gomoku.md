# What We Learned About AlphaZero on 15×15 Gomoku
> **Status: LIVE synthesis** *(2026-07-04)* — settled verdicts first; the 15×15 campaign it distills is historical.

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

## Settled verdicts (read this first)

The 15×15 campaign's headline story is a **self-correction**: a broken measurement inverted
our rankings, we caught it with direct head-to-head, and the corrected conclusions below are
what stand. The blow-by-blow that produced them (the full §2/§2a/§8/§9/§10/§11 chronicle —
every table, retraction, and mechanistic guess) was preserved verbatim but has since been
removed *(2026-07-04; recover: `git show ca76350:wiki/_archive/topics/alphazero-lessons-yardstick-saga.md`)*.
The compressed forward-reading version is §2 and "the yardstick reckoning" below.

1. **`128×10+bigbuf` is the strongest preserved 15×15 net** — it beats the once-crowned
   `96×8 e499` **40-0** head-to-head. The `96×8` "champion" crown was a **yardstick artifact**;
   head-to-head it is the *weakest* well-trained net (beaten 40-0 by both a bigger *and* a
   smaller net). Do not use the old Rapfi tiers to pick a champion.
2. **Capacity × search is MULTIPLICATIVE** (§11). Bigger net + more search compound: the
   three nets (0.44M→3.3M) play to a dead tie at 100 sims but the 128×10 wins **40-0 at 200
   sims**. The §2a "capacity reversed at depth" finding is **retracted** — there was never a
   depth axis to reverse on (§8C: both "TC tiers" were the same shallow engine measured twice).
3. **The loss lies; gate on structure and head-to-head** (§3). value-loss can set fresh lows
   while deep-search strength sits motionless (deepgen §8F; the silent self-play regression
   §10). Judge training by `plies`/`vl`/external strength — and gate every "did this help?"
   on a **color-alternated match vs the preserved champion**, never the external ladder alone.
4. **Audit the yardstick before it audits your conclusions** (§8). Our Rapfi was the
   *weightless classical* build (not the rated NNUE engine), defaulted to CPU (understating
   our strength), and — the big one — **never used its search time** (self-terminates ~depth
   10 / ~500 nodes regardless of the time budget), so the "1000ms" and "5000ms" tiers were
   *one shallow engine measured twice*. A broken yardstick doesn't add noise — it can **invert
   your ranking** so you crown the worst and discard the best, with every internal signal
   applauding.
5. **White-side (second-player) defense weakness = the first-player-win THEOREM, not a net
   flaw.** It is search-invariant (immune to FPU and 4× search) → a genuine training gap that
   only relabeling/rebalancing can touch. **§15 is the canonical statement of this theorem**
   (the [white-side-defense-plan](white-side-defense-plan.md) investigation and
   [swap2](swap2-opening-protocol.md) §1/§10 point here). The fix is swap2 (delete the forced
   role), not a defense teacher.
6. **A history-conditioned net through an order-free protocol silently sandbags** (§13).
   The Gomocup `BOARD` command is order-free; a net that reads recency planes gets all-zero
   history (OOD) and quietly plays ~75 points weaker. Drive incrementally (`TURN`) so move
   recency is reconstructed. Same theme as §10/§8F: internal-healthy ≠ actually-strong.

**Where we stand vs engines (§12):** upper-mid-pack of a real Gomocup bracket — beat Yixin2018,
draw Pela23/Zetor17, below the 2026 frontier (Embryo26); ~62% vs a strong NNUE Rapfi, robust
across its depth settings (but ~half the margin is the black first-mover edge — white defense
is the concrete weakness). **Founding design decisions** (board, PUCT-not-determinized-MCTS,
ResNet-not-transformer, stack) are in §17.

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

## 2. Net capacity and search — the capacity ladder (settled; full saga archived)

> **This section is compressed.** The original §2/§2a carried a live-updating capacity ladder
> with a chain of retractions (a "reversal at depth" that turned out to be the broken yardstick,
> §8C) *(removed; see note above)*.
> The settled conclusion, confirmed yardstick-free in §11, is below.

We climbed a capacity ladder (64×4 → 96×8 → 128×10) via **function-preserving net2net** growth
— each bigger net starts at the smaller one's *exact* function (output-equivalence ~1e-4), so
it inherits strength and trains into the extra capacity with no cold restart. The **settled**
capacity finding (§9 head-to-head, §11 confirmation):

- **Capacity × search is multiplicative, not additive.** A bigger net evaluates positions more
  accurately, and that better evaluation *compounds* with search depth. Measured yardstick-free
  (color-alternated head-to-head, n=40): the 64×4 / 96×8 / 128×10 nets **tie at 100 sims** but
  the 128×10 **wins 40-0 at 200 sims**. Net quality × search depth is the AlphaZero net/search
  duality, on our own board.
- **`128×10+bigbuf` is the strongest** (§9): 40-0 over the old 96×8 "champion." The 96×8 crown
  came from the broken Rapfi tiers (§8); head-to-head it is the weakest well-trained net.
- **Capacity needs data.** The 128×10 overfit on a 400k buffer (37.5% aggregate) and recovered
  on a 1.5M buffer — capacity and data are *complements*, not substitutes.
- **Honest caveat (don't over-read absolute capacity):** epochs are confounded (64×4 e909 has
  ~2-3× the 128×10's / 96×8's training), so a clean capacity magnitude needs **same-epoch**
  head-to-heads (open work). The *shape* (tie@100 / dominate@200) is the multiplicative
  signature regardless.

**Methodological keeper:** a negative result ("more data didn't help the 96×8") is a real
finding when it's a clean single-axis A/B — it *reallocates the search*. The learning is the
artifact; negative results count.

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

## The yardstick reckoning (chronicle — §8–§11, compressed)

> **This is the compressed, forward-reading version of the original §8/§9/§10/§11.** The full
> blow-by-blow (8A-8H sub-findings, every table, the mechanistic guesses, the INCONCLUSIVE
> re-test) *(removed; see note above)*.
> Read it forward as one story:

**§7 flagged the yardstick as a soft spot; pulling that thread unravelled the campaign's
rankings.** Three things were wrong with the Rapfi yardstick, each worse than the last:

- **(8A) Wrong engine.** Our `pbrain-rapfi` was the **weightless classical** build, not the
  rated ~2625 NNUE engine stamped on every eval row. We were beating *weak* Rapfi. (Fixed this
  session — the NNUE `mix9svq` weights load and search.)
- **(8B) Wrong device + under-powered n.** `ladder_eval` defaulted to CPU, systematically
  understating model strength (MPS 96% vs CPU 79%, n=24); and the load-bearing "reversal"
  number was a single n=16 read, violating our own "never a single number" rule.
- **(8C) The big one — Rapfi never used its search time.** It budgets ~5s then self-terminates
  after ~500 nodes / ~1-2ms / depth ~10 on *every* position and *every* `timeout_turn`. So the
  "1000ms" and "5000ms" tiers the whole campaign quoted were **the same shallow engine measured
  twice.** The "capacity pays at depth" and "reversal at depth" stories were built on a depth
  axis that did not exist.

**§9 — the reckoning.** With the yardstick exposed, we ran the preserved nets **directly
against each other** (color-alternated, sims-matched, match.py-validated: champion-vs-self =
50%). The result overturned the headline rankings: `{128×10, 64×4} ≫ 96×8-"champion" ≫
deepgen`. The broken yardstick had **crowned the weakest trained net** (96×8, beaten 40-0 by
both a bigger *and* a smaller net) and **made us abandon a tied-strongest net** (128×10, retired
on a "reversal" that never happened).

**§8F — the yardstick actively hid a regression.** The `deepgen` experiment (deeper self-play
search, 200 sims) looked great on every internal signal (vl↓, plies↑, 83% vs the shallow Rapfi)
— but head-to-head it was **0-40 vs the champion at 100 sims**. Deeper self-play doesn't add
strength; it **specializes** the net to its training search depth (it offloads judgement to the
search, then can't stand without it). A net inherits the search budget it was trained under.

**§10 — the silent self-play regression.** The 96×8 "champion" was measured to be **40-0 worse
than its own untrained net2net seed** — 400 epochs of training ran it *backwards*, while
`plies`/`vl`/internal-ladder all looked healthy the whole way. A `G15-96x8-redo` re-test was
**INCONCLUSIVE** (noisy, non-monotonic 100→100→50→75% vs seed, never fell below seed, stopped
before the original's collapse epoch), so whether the v8 recipe *systematically* regresses 96×8
is unresolved — but the hazard (train-backwards while metrics applaud) is real and recurred.
Two operational gates fell out: **gate champion-promotion on head-to-head vs the prior champion**,
and **spot-check vs the frozen seed** before continuing past ~100 epochs — and **freeze the
peak live** (the strong e234 redo checkpoint was lost to `keep_last_n` pruning).

**§11 — capacity×search confirmed clean, the reversal retracted.** A yardstick-free test closed
the question: the three nets tie at 100 sims and the 128×10 dominates 40-0 at 200 sims — the
**multiplicative signature**. §2's original "capacity pays at depth" was right; §2a's "reversal"
was the measurement lying. **128×10 is champion specifically *because of* depth** (it converts
more search into more wins).

**The meta-lesson (the whole thing in one line):** *a broken yardstick doesn't add noise — it
can invert your ranking, so you crown the worst option and discard the best, with every internal
signal applauding.* The same shape recurred five times in two days (deep-TC tiers, the "88%
ceiling," the capacity "reversal," the deepgen "improvement," the champion selection) — every one
dissolved under a direct head-to-head. The cheap insurance we should have had from day one: a
**color-alternated match against the preserved champion gates every "did this help?" decision**;
the external ladder is for an *absolute* rating only once it's actually fixed (NNUE + genuinely
deep search + balanced swap2 openings). Frozen artifacts: `g15_128x10_bigbuf_eval502.pt` (real
strongest), `g15_champion_e909.pt` (64×4, tied), `g15_96x8_deepgen_searchspec_e621.pt` (the
cautionary tale).

## 12. Where we actually stand vs engines — and how hard we tried to disprove it (2026-06-15)

### 12A — Strength vs Rapfi (the only readily-runnable strong engine on ARM)

Champion net: `g15_128x10_bigbuf_eval502.pt` (128x10, re-crowned), sims=200, balanced openings
(--random-opening-moves 4), 5000ms time limit, n=40 games, head-to-head validated (match.py
champion-vs-self = 50%).

| Rapfi config | result | per-color |
|---|---|---|
| Shallow NNUE (mix9svq, early-stop) | 62.5% (25-15) | black 85% (17-3), white 40% (8-12) |
| Deep NNUE (force-full-search, db off) | 66.2% (26-13-1) | — |
| Full-strength NNUE (depth 32-53 verified, 18 cores, db on, full 5s) | 62.5% (25-15) | black 85% (17-3), white 40% (8-12) |

The full-strength result is BYTE-IDENTICAL to the shallow one. Same seed and openings,
deterministic champion, and Rapfi played the same moves at depth 15 and depth 40: its deep search
and database changed zero games. So we beat Rapfi ~62-66% robustly across its strength settings —
but "full strength" ≈ "shallow" because Rapfi's NNUE move-selection is depth-insensitive in these
positions.

### 12B — Why this is NOT "we beat a champion" (the cork stays in)

Honest demolition:

1. **Self-built, self-patched Rapfi, zero independent calibration.** We compiled and patched this
   Rapfi ourselves. There is no independent confirmation it actually plays at its rated ~2625 Elo.
   The force-full-search patch could have degraded its move quality.
2. **Depth-insensitivity is a yellow flag.** A genuinely strong engine usually plays better with
   1000× more search. Rapfi didn't change a move. That is suspicious, not reassuring.
3. **~Half the margin is first-mover edge.** Black 85% / white 40%: as the side-to-move defender
   we lose most games. White-side defense is our real, concrete, demonstrated weakness.
4. **Small-n, one engine, one harness, one session.** n=40 with a novel setup means every number
   here has wide error bars.
5. **Base rate on surprising results in this project.** A surprising strength result has turned out
   to be a measurement artifact roughly five times in two days. The prior on "we beat the champion"
   is "we mis-measured something" until independently proven.

Honest sentence: our net beats a strong NNUE Rapfi ~62%, robustly across its depth settings. That
is not a tournament-parity or world-ranking claim.

### 12C — The crush-ourselves discipline + the harness is the weak link (a real finding)

We red-teamed our own result — the right instinct: the fastest learning is trying to destroy your
own claim. The first crack we found was our own harness.

`gomoku/external_engine.py` is fragile. It desyncs with engines that emit non-standard output:
`DATABASE`-prefixed lines are not in its chatter-skip list (`MESSAGE/DEBUG/INFO/ERROR/UNKNOWN/SUGGEST`).
Real engines that announce moves as `my move [x,y]`-style lines or print engine banners cause it to
read a non-move line, desync, and forward a board coordinate as a top-level command. Observed
failures: engine-vs-engine crash `ERROR Unknown command: 9,2,1`; tournament crashes `ERROR my move
[14,3]` on three engines; `Board isn't initialized` on another.

Reusable lesson: an eval harness must accept ONLY a bare `X,Y` coordinate as the move (skip
everything else), confirm the START handshake, and be validated against multiple engines. Validating
against only Rapfi (whose chatter happens to be MESSAGE-prefixed) hid this fragility entirely.

### 12D — Real-engine tournament infrastructure now exists

Five actual Gomocup competition engines now run on this Apple-Silicon Mac via wine
(installed to `~/.cache/gomocup/`):

| Engine | Notes |
|---|---|
| Embryo26 | 2026 top engine, uses Vulkan |
| Yixin2018 | 64-bit |
| Pela23 | 64-bit |
| Zetor17 | 32-bit via wow64 |
| Eulring16 | 32-bit via wow64 |

Wrappers at `~/.cache/gomocup/bin/run-*`. All boot and pass a single-move probe. The fragile
harness (§12C) cannot play full games against them yet — it is being hardened. Once fixed, a real
official-bracket test is possible: the honest "is Rapfi our ceiling, or are we below the real
floor?" question. (Embryo26 also contends for the GPU via Vulkan, so it requires a clean-GPU run.)

### 12E — Status as of this writing

The crush-ourselves red-team — white-defense failure mode, tactical blind-spots, seed/re-crown
robustness — and the fixed real-engine tournament are in progress. The real-engine bracket results
are now in: see §12F.

### 12F — The real-engine bracket, measured (harness fixed)

**Harness fix (merged to main).** The fragility named in §12C is resolved.
`gomoku/external_engine.py` now: sends RESTART before each BOARD command (re-entrant
board replay); takes the move *only* from a bare `X,Y` line (skips banners,
`DATABASE`, `my move [..]`, and non-fatal `ERROR` chatter); uses `BEGIN` on an empty
board; and scales the read deadline. +5 regression tests, 124 tests pass, the Rapfi
path preserved. This unblocked full games against real Gomocup engines running under
wine on this Apple-Silicon Mac.

**The bracket.** Champion `g15_128x10_bigbuf_eval502.pt` (128×10), sims=200,
balanced openings (`--random-opening-moves 4`), 5000–10000ms, head-to-head validated
(match.py champion-vs-self = 50%):

| opponent | overall | as black | as white |
|---|---|---|---|
| Yixin2018 (n=8) | 8-0 (100%) | 4-0 | 4-0 |
| Pela23, 2023 (n=8) | 4-4 (50%) | 4-0 | 0-4 |
| Zetor17 (n=8) | 4-4 (50%) | 4-0 | 0-4 |
| Embryo26, 2026 top seed (n=6) | 2-4 (33%) | 2-1 | 0-3 |

**Interpretation.** The bracket shows a coherent strength gradient: we beat
Yixin2018, draw Pela23 and Zetor17, and sit below the 2026 frontier (Embryo26).
The champion is upper-mid-pack — not under the floor, not at the top. We cleared
the humble bar (win-as-black or draw-as-white) against every engine, including 2
wins over the 2026 top seed.

**The actionable finding (consistent across all opponents, including Rapfi's 40%
white in §12A):** the champion is dominant as black (4-0 vs every engine here) and
weak as white — strong engines beat us 0-4 when they move first. Defense (the white
side) is both the ceiling and the obvious next research target: we trained an
attacker, not a defender.

**Caveats (cork stays in):** n=6-8 is small. The engines run under wine and may be
slowed vs native, so against native Embryo26 we likely sit lower than 33% implies.
This is a placement, not a calibrated rating; much of the black dominance is
first-mover advantage. Operator note: a hung `sudo`/gstreamer install was timing
Embryo26 out — the human at the keyboard caught and approved it, unblocking the
top-seed measurement. A remote agent would have seen only timeouts.

**Honest process note.** The adversarial red-team workflow from §12C wedged
partway through (GPU contention while the tournament ran) and never produced formal
verdicts. Its primary target — the harness — was found and fixed manually. The
white-defense weakness is already established from this bracket; a focused
white-side failure-mode analysis is the clean follow-up.

## 13. The brain wrapper & the empty-history trap — history-conditioned nets need faithful recency (2026-06-15)

§12F hardened the **client** side of the Gomocup protocol (`external_engine.py`:
RESTART before BOARD, bare `X,Y` move). This section is the **brain** side — we
exposed our own net *as* a Gomocup engine (`gomoku/gomocup_brain.py`, #31 commit
`1834df0`; shell wrapper `scripts/run-gomoku-az`, registerable via
`external:cmd=run-gomoku-az --checkpoint X --sims N`) — and immediately hit a
silent strength loss that is the §10/§8F "internal-looking-healthy ≠ actually-strong"
theme in a new place: **the protocol itself can sandbag the net.**

### The trap: a history-conditioned net through an order-free protocol

Our input is **history-conditioned**: `gomoku/game.py` uses `HISTORY_PLY = 8`
recency planes per side (`N_INPUT_PLANES = 17`). `to_planes()` reads the *current*
board from `board[0]`/`board[1]` and the *older* recency frames from
`state.history`. The net was trained on inputs where "full current board" and
"populated recent frames" always co-occur.

The classic Gomocup `BOARD` command **re-dumps the whole position every move** and
is, by the protocol spec, **order-free** (it lists the stones, not the move
sequence — move order is *unrecoverable* from a single dump). A naive brain rebuilds
a fresh `GameState` from that dump, so its `history` is **empty**: `to_planes()`
emits a full current board but **all-zero recency planes**. That is a
self-contradictory, out-of-distribution input the net never saw in training — *"a
full board now, but zero stones in every recent frame."* The net still answers, the
harness looks healthy, and the net silently plays weaker.

### The measurement (same checkpoint, vs heuristic, sims=100, seed=0)

`g15_128x10_bigbuf_e588_best.pt`, 4 games unless noted:

| driving path | history seen by `to_planes()` | result |
|---|---|---|
| Native in-process picker | full (real `GameState`) | **4-0 = 100%** |
| Wrapped, BOARD-replay every move | **empty** (rebuilt each move) | 1-3 = 25% |
| Wrapped, `incremental=1` TURN-mode | real (accumulates) | 5-1 (n=6) = **83%** |

Re-dumping the board every move cost the net ~75 points (100% → 25%) against the
same opponent — entirely from the empty-history OOD input, not from any change to
the net or the search. Small-n (§4), but the gap is far larger than the noise band
and it reproduced as a path difference, not a sample difference.

### The fix: drive incrementally so move RECENCY is reconstructed

`external_engine.py` gained an `incremental=1` mode (`incremental: bool = False`,
default off). After a first `BOARD` sync, it feeds the opponent's single new move as
`TURN x,y` — *not* `RESTART` (which would wipe history) — so a stateful brain
accumulates real move history via `GameState.apply()`. `_can_turn()` gates it to
**clean continuations** (our stones unchanged, opponent +1 stone) and falls back to a
`BOARD` resync otherwise (game boundaries / desyncs / the opening). Only the
game-opening first move keeps empty history (few stones, near-harmless). **Default
off** so classical external engines (no history planes — Rapfi et al.) keep the
robust BOARD path from §12F; only our own stateful brain opts into TURN-mode. Nets
**must** therefore be registered with `incremental=1`.

### The general lesson

**When you expose a history-conditioned net through a stateless/order-free protocol,
you must reconstruct move *recency*, not just the static position.** A single board
dump is sufficient to rebuild *where the stones are* but not *when they arrived*, and
a net that consumes recency planes treats the missing-order input as out-of-distribution
and quietly underperforms. The Gomocup `BOARD` command is *explicitly* order-free for
gomoku, so the only faithful drive is incremental (`TURN`), letting history accumulate
move-by-move. This generalizes the project's recurring **silent self-play regression**
theme (§10) and the deepgen specialization (§8F): *internal-looking-healthy ≠
actually-strong; gate on real head-to-head.* Here the lie was not in the metric or
the recipe but in the **I/O adapter** — the net was fine; the protocol bridge fed it
garbage and nothing flagged it until a direct same-checkpoint comparison did. Cheap
insurance, same as everywhere else in this campaign: A/B the new path against the
known-good path on one fixed checkpoint before trusting it.

## 14. First panel tournament: the calibration broke (and that's the finding) (2026-06-15)

§13 built the brain wrapper; this section runs the first **panel tournament** with
it (the #30 "calibrated yardstick" epic) and reports an honest **partial failure**.
The headline is not a strength number — it is that *we tried to produce one and the
calibration came out backwards*. That refusal-to-print is the finding, and it is the
project's "be suspicious / a failure that teaches is value" ethos (§3, §8E, §10, §13)
doing exactly its job: the tooling **flagged its own broken output instead of
laundering it into a confident Elo**.

Data: `sweep_runs/panel_tournament_results.jsonl` (36 pair records). Reader:
`scripts/panel_white_elo.py` — its §1 per-color **rates** are computed directly from
the per-color arrays (ground truth, no model); its §2 Bradley-Terry Elo is a
**flagged estimate** that, on this run, *refuses* to print a Gomocup-calibrated
number. Tracked as [#35](https://github.com/jasonyandell/gomoku/issues/35).

### What broke #1 — the engines, not our nets

The full 9-player round-robin (3 nets + 6 opponents, n=6 each) is 36 pairs.
**Only 19 played; 17 ERRORED.** The errors are *not random* and *not ours*:

| failure mode | count | engine |
|---|---|---|
| `engine timed out after 30.0s` | 13 | mostly `embryo26 vs *` (Vulkan/GPU-contended) |
| `engine process has exited` / `closed stdout (EOF)` | 4 | `* vs zetor17` (crashes on back-to-back reuse) |

**Our brain-wrapped nets produced ZERO errors.** Every one of the 17 failures was an
*opponent engine* dying — Embryo timing out under GPU contention, Zetor crashing when
reused across consecutive pairs. The crashes cause **missing data, not fabricated
losses** — a cross-table with holes, not a corrupted one. (Almost all the failures
are wine-vs-wine pairs; our-net-vs-engine pairs mostly completed.)

### What broke #2 — the calibration anchor (the real lesson)

The affine fit of internal strength → published Gomocup Elo came out with a
**NEGATIVE slope (~−0.071)**: internal strength *anti-correlated* with published
rating. The smoking gun, straight from the completed games:

- **yixin18** (published ~2310, a *top-tier* engine) went **0–30** — it lost *every
  completed game*, including **0–6 to the heuristic**.
- **pela23** (published ~1499, mid-pack) went **24–6**.

The published Elos are *multi-thread tournament* ratings. Under **wine + a single
thread + a 10s/move budget**, the engines do **not** play at those ratings — Yixin in
particular is crippled. So the published numbers are **invalid anchors for our
harness**, and any affine fit to them is garbage-in. `panel_white_elo.py` **correctly
refuses** to emit a calibrated Gomocup Elo here: it detects the degenerate
(non-positive) slope, falls back to a mean-0 *relative* internal scale, and **loudly
flags** that the absolute scale is not trustworthy. The right design response is to
**measure effective strength under our exact harness**, not to assume the ladder
(#35).

### What IS trustworthy (completed games only — holes are missing, not fake)

Because the flakiness drops *whole pairs* rather than skewing scores within a pair,
the games that *did* finish are clean reads (small-n per §4, so hints not verdicts):

**(a) Net-vs-net** (n=6 each, mildly non-transitive = noise):

| matchup | result (win rate, n=6) |
|---|---|
| az-champ-128x10 vs az-96x8 | champ **5–1 (83%)** |
| az-96x8 vs az-128x10-e588 | 96×8 **4–2 (67%)** |
| az-128x10-e588 vs az-champ-128x10 | e588 **4–2 (67%)** |

A small rock-paper-scissors loop (champ > 96×8, 96×8 > e588, e588 > champ) — all
close, consistent with sampling noise on n=6, *not* a stable strict ordering.

**(b) Net-vs-heuristic** (the static-eval floor is non-trivial):

| net | vs heuristic (n=6) |
|---|---|
| az-128x10-e588 | **6–0 (100%)** |
| az-champ-128x10 | 5–1 (83%) |
| az-96x8 | 5–1 (83%) |

**(c) The white-side defense gap (the concrete next target,
[#33](https://github.com/jasonyandell/gomoku/issues/33)).** Aggregated over external
opponents, `az-champ-128x10` scores **94% as black (attacking)** but only **50% as
white (defending)** — a **+44pp** gap, a **50% white-loss rate over 18 white games**.
The collapse is opponent-specific and sharp: champion goes **0–3 white (100% loss)**
vs **embryo26** *and* vs **zetor17** (and 0–3 white vs net e588), yet holds **3–0
white** vs the *weaker* engines yixin18 and eulring16. Defense fails precisely
against the opponents strong enough to punish it.

**Caveat we do NOT hide:** the reader's own sanity check flags that **az-96x8 does
NOT show the gap** — it scores 67% white and 67% black (white_loss 33%), i.e. white ≥
black, *contradicting* the collapse on only n=12 white games. The gap is real and
large for the champion; it is **not yet a universal law across nets** on this thin
data. (e588 matches the champion's shape: 100% black, 67% white, +33pp, n=9.) Report
the contradiction; do not massage it.

### Conclusion — the yardstick is NOT yet achieved, but the tooling is sound

The #30 "calibrated yardstick" is **not delivered**: a calibrated Elo needs
**reliable engines** *and* **empirically measured effective strengths under our exact
harness** (wine, single-thread, 10s/move) — not assumed published ladder ratings
(#35). But the *machinery* worked flawlessly on our side: the brain wrapper (#31,
§13), the cross-table runner (#32), and the reader (#33, `panel_white_elo.py`) each
produced **zero errors and refused to over-claim**. And the white-side defense gap
(#33) is now **confirmed and quantified** — the concrete, actionable next target. The
honest one-liner: *the ruler is bent, the tool correctly told us so instead of lying,
and the one number we can trust says our defense is the hole.* Same discipline as
§10 and §13 — gate on real, harness-native head-to-head; never trust an anchor (here a
*published Elo*) you have not validated under your own conditions.

## 15. White-side defense is a TRAINING gap (search-invariant, not an eval flag) (2026-06-15)

> **⭐ CANONICAL HOME of the white-defense / first-player-win theorem.** This section is the
> single authoritative statement (it's the learning artifact). The operational investigation
> [white-side-defense-plan.md](white-side-defense-plan.md) and
> [swap2-opening-protocol.md](swap2-opening-protocol.md) §1/§10 carry short summaries and link
> here rather than re-deriving it.

**Verdict up front:** the champion's white-side (defending / second-player) collapse
vs strong attackers is **immune to every eval-side lever we can pull** — neither the
search prior (FPU-reduction) nor 4× more search budget moves it one game. A weakness
that survives both the cheapest knobs *and* deeper search is not a search/eval
miscalibration; it is a **genuine training gap**. The policy/value cannot *represent*
the saving defense — it must be **taught**. This closes the cheap branch of the
[#33](https://github.com/jasonyandell/gomoku/issues/33) defense investigation and
routes the fix to training ([#36](https://github.com/jasonyandell/gomoku/issues/36) /
the [#18](https://github.com/jasonyandell/gomoku/issues/18) recipe).

Champion = `g15_128x10_bigbuf_eval502.pt` (128×10, 15×15 freestyle). White =
champion **defending** (second player). The §14 panel surfaced the gap (94% as black,
~50% as white, collapsing **0–3 / 100% loss** as white specifically vs the *strong*
attackers embryo26 and zetor17, while holding white vs the weaker engines). §15 is
the falsification of the cheap fixes against the one reliable real attacker we can
run cleanly head-to-head, **zetor17**.

### The eval-lever falsification (all white-side, vs zetor17 unless noted)

**FPU-reduction — the 9×9 wiki's claimed white-loss fix — changes NOTHING vs a real
attacker.** The 9×9 corpus (and §0/§I0 of the defense plan) held that
`fpu_reduction_c = 0.45` *alone* drives white-loss to ~0 (the white weakness framed
as a search-prior miscalibration, not a training gap):

| opponent | FPU | white result |
|---|---|---|
| `lookahead:depth=4` (weak searcher) | 0.0 | 88% (small residual loss-tail) |
| `lookahead:depth=4` (weak searcher) | 0.45 | **100%** (residual tail closed) |
| **zetor17** (real strong attacker) | 0.0 | **0–6 (100% loss)** |
| **zetor17** (real strong attacker) | 0.45 | **0–6 (100% loss)** |

FPU=0.45 *did* close a small residual gap vs the depth-4 lookahead (88%→100%) — but
that loss-tail was nearly closed already, and a weak searcher is not the threat. Vs a
**real strong attacker**, FPU at 0.0 and at 0.45 both give **0–6, 100% loss** —
identical. **The 9×9 FPU-as-defense-fix does NOT transfer to 15×15 real-engine
defense.** This *corrects* the old 9×9 claim (§4 already flagged that the FPU
ladder-sweep lever did not transfer; §15 nails it specifically for white-side
defense vs a real engine): FPU is not the white fix here. It looked plausible because
it worked on the weak searcher — and was falsified on the real attacker.

**Search budget (H3, "search too shallow") — 4× more search changes nothing.**

| opponent | sims | white result |
|---|---|---|
| zetor17 | 200 | **0–4 (0%)** |
| zetor17 | 800 | **0–4 (0%)** |

Quadrupling the simulation count leaves white at **0%**. H3 (the net just needs to
see deeper) is **ruled out for real-engine defense.**

**The dissociation that names the cause:** throughout *every* FPU and sims setting,
the champion is **perfect as black** vs zetor17 (4–0 / 6–0) and **helpless as white**
(0–4 / 0–6). The *same net, same search, same opponent* — only the color flips, and
the result flips with it, at every eval setting. A weakness that tracks the
**training-target asymmetry** (black = forced-win side the net learned to convert;
white = the side whose saving moves were never labeled) and is **invariant to every
search/eval knob** is, by elimination, in the **weights**, not the search.

### Why this is the verdict (the ethos, again)

This is the project's recurring *"internal-looking-healthy ≠ actually-strong / be
suspicious"* discipline (§3, §8E, §10, §13, §14) applied to the **fix** rather than
the metric: the cheap eval-side fix *looked* plausible (it worked on the weak
searcher, exactly as the 9×9 wiki promised) and was **falsified on the real
attacker**. Had we trusted the depth-4 read, we'd have shipped an FPU flag and
declared defense solved while still losing 0–6 to every strong engine.

The standing root-cause map (defense plan §2):
- **H1 — teaching gap ([#18](https://github.com/jasonyandell/gomoku/issues/18)):**
  a lost white game enters the buffer labeled only `z=−1` for the whole trajectory;
  it never says *which move* would have saved it. The net is never taught the saving
  move. **STANDS** (strongest hypothesis).
- **H2 — value-target asymmetry:** at self-play convergence white wins → 0, so the
  value head sees few white wins and little gradient to distinguish "drawable" from
  "lost" white positions; uniform sampling drowns the rare recoverable one.
  **STANDS.**
- **H3 — search too shallow:** **RULED OUT** for real-engine defense (4× sims = 0
  change).

### The fix routes to training

The defense must come from **relabeling**, not from any eval knob:
[#36](https://github.com/jasonyandell/gomoku/issues/36) — the **defense-teacher +
VCT training cell** (the [#18](https://github.com/jasonyandell/gomoku/issues/18)
recipe): clone the champion cell and turn on the value-only `--defense-teacher`
(stamps proven-lost white positions `z=−1`, "defend earlier") paired with
`--vct-teacher`. If value-only under-moves the **draw/loss boundary** (exactly where
"never lose as white" lives, and exactly the boundary the value-only teacher does not
sharpen), escalate to **I2 — stamp the saving move** (a one-hot policy label on the
unique defensive refutation: teach the move it *should* have played, the literal
[#18](https://github.com/jasonyandell/gomoku/issues/18) ask for the second player).

**Cross-links.** §14 — the panel tournament where this gap first surfaced and was
quantified (champion 94% black / 50% white, 0–3 white vs the strong attackers). §13 —
the brain wrapper that let our net play these real engines head-to-head at all. The
operational plan and the falsified-vs-standing intervention ledger:
[white-side-defense-plan.md](white-side-defense-plan.md) (Step A FALSIFIED → Step B
is now the first real experiment).

## 16. Threat semantics: the static eval is credit-correct — search structure is where the bug lives

In free-style 9×9/15×15 gomoku the threat hierarchy is genuine, so the static
evaluator's heavy weights on N-in-window patterns are **correct, not
"over-crediting."** Do not pitch eval-credit-removal as a fix.

- **Open-3** (3 stones in a 5-window, both ends open): forces a defensive
  response — left unblocked, either-side extension makes an open-4 → mate-in-one.
  *Must* be blocked. Heavy weight is right.
- **Half-open-4** (4 stones, one end blocked, other open): immediate winning
  threat; opponent must block the open end on the very next move.
- **Open-4** (4 stones, both ends open): already mate — defender blocks one end,
  the other completes 5 next move.

**Diagnostic principle.** When a lookahead/MCTS search produces bad moves, the
bug is in *search structure* (quiescence, move ordering, branching, horizon
parity), **not** in the static evaluator's pattern credit. The static eval
correctly says "this position is great *if you got to keep playing*"; the
search's job is to make the opponent's forced response real before reading the
leaf. (Same theme as [mcts-perf-ceiling.md](mcts-perf-ceiling.md): don't lazily
blame the easy-to-rewrite layer when the bug is structural.)

**Evidence — the lookahead-depth-3 investigation.** The hypothesis that the eval
"over-credited" open-3 was *wrong*. Odd-depth weakness came from `depth=0` having
no quiescence to apply the forced block of an immediate-win-completable-4 — a
missing search step, not bad credit. **Shipped partial fix:**
`gomoku/baselines.py:_negamax` depth=0 quiescence catches the case where the
opponent just built a completable-4 and the leaf would otherwise credit an
unblockable-looking threat. It does **not** close all odd-depth weakness; the
residual is likely move-ordering + `_MAX_BRANCH=12` cutting necessary blocks from
the tree at internal nodes — a separate investigation, not a static-eval fix.

## 17. Founding design decisions (2026-05-17) and public artifacts

The project started 2026-05-17 with these decisions, recorded here so the
origin rationale isn't lost:

- **Board:** 9×9 free-style (first-to-5, no opening restrictions) — chosen as the
  smallest meaningful board for the fastest feedback loop before any 15×15 work.
- **Architecture rationale:** gomoku is a perfect-information game, so use
  **standard PUCT MCTS** — deliberately do **not** port the *determinized* MCTS of
  zeb (`~/code/mk5-main/forge/zeb/`, Jason's prior AZ project); that machinery is
  for imperfect-info games and buys nothing here. The 81-action space (9×9) is
  handled by a **ConvNet/ResNet** policy+value net, **not a transformer**.
- **Stack:** Python 3.12 + PyTorch 2.11 on MPS, `uv` for deps; W&B from day 1
  (mirrors zeb, `~/code/mk5-main/forge/zeb/`). CLI play first, web UI later.
- **Iteration stance:** "watch it work" over premature optimization — get a smoke
  run green, then tune.

**Public artifacts:** source [github.com/jasonyandell/gomoku](https://github.com/jasonyandell/gomoku) ·
model snapshots [huggingface.co/jasonyandell/gomoku-9x9](https://huggingface.co/jasonyandell/gomoku-9x9) ·
W&B [wandb.ai/jasonyandell-forge42/gomoku](https://wandb.ai/jasonyandell-forge42/gomoku).
