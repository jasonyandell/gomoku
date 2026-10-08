# walt on gomoku — a deliberate misapplication, at best effort

**DEAD-END (lesson kept)** *(2026-10-07)* — as a *player*, walt ties MCTS with the same net
against a real engine. Its big wins came only against opponents shaped like its own model.
As a *sense* (part 2), walt's count is informative and complementary to the net
(+0.05 AUC, replicated). As an actuator it barely moves play.

## Part 2 verdict: walt as a sense, not a brain (2026-10-07, same day)

Jason's reframe: *"walt is more of a sense than a brain — it detects more than it
investigates."* A Fable advisor distilled walt's essence to **a coverage count**: the share of
sampled worlds that one commitment survives. The stack has averages (value, PUCT) and exact
answers (oracle), but no coverage statistic. Tested on defense (`scripts/walt/exp_b_coverage.py`,
bets pre-committed in `scripts/walt/BETS_sense.md`).

**Detector setup.**
- The **defender** is the eventual loser of a 9×9 Rapfi@50 game, to move N plies before the
  winner's first cap50 VCT.
- For each top-16 candidate c, **pressure** = the share of 32 field worlds (the attacker's next
  stone o₁ ~ net@T1, tickertape) in which the attacker has a VCT:
  - **null-move version:** the attacker moves twice in a row;
  - **escape version:** the defender, *seeing* o₁, has no safe reply among its top-12. This is
    walt's semantics: future choices condition on the revealed card.
- **Judge:** Rapfi@50 plays both sides from after c.

**Results** (replication: 2,369 fresh games, 600 positions × top-8 judged, 247 positions where
the choice matters):

| Question | Answer |
|---|---|
| Is the sense informative? (Jason's bet) | **Yes.** Grouped-CV AUC for flagging a losing move: net signals 0.793 → net + pressure **0.845 ± 0.010**. The first sample gave 0.763 → 0.831 (n = 59). |
| Is it the *sampling* (sense) or the oracle (brain)? | **Standalone, sampling wins.** Within-position ranking for the null version: 32 worlds 0.705 vs 1 world 0.617 vs the net's single best line 0.610. The count over the ensemble beats following the most likely line by ~0.09 AUC, and it sharpens monotonically with W (1/4/8/32 = 0.63/0.72/0.76/0.79 on the first sample). |
| What does it add on top of the net? | **Mostly the investigation.** The escape version adds +0.048 but is flat in W (argmax 0.839, W32 0.841). The null version adds +0.018, and only +0.006 of that comes from sampling. |
| Does it change play? (actuator) | **Barely.** MCTS × (1 − p_null)² loses 0.735 vs MCTS + veto 0.745 (paired 44 saved / 30 lost, p ≈ 0.1). Escape pressure changes only 0.3% of choices. Pure walt (min p) is worse (0.762). |
| White defense | Still loses ~78% under Rapfi. The sense doesn't fix the chronic gap. |

**Reading.** The sense is real: counting outcomes over sampled futures detects losing moves far
better than investigating the single most likely line. But in an AlphaZero stack **the net
already is a sense**, a learned detector distilled from millions of games. walt's ensemble
mostly rediscovers what the net knows. The new part is the exact escape search, which the net
can't do.
- **Plausible why walt works in card games:** there's no comparable learned detector for the
  hidden deal, so the count *is* the sense. Gomoku's hidden information is logical (where the
  VCT is), and a trained value head already covers most of it.
- **This is a hypothesis about walt's home games, not a measurement there.**

**Side findings.**
- **The 50-node oracle is a *lagging* detector.** Under Rapfi, positions 1–5 plies before the
  first cap50 VCT are lost ~99% regardless of move, and 7–11 plies back ~93% are all-lost.
  9×9 Rapfi games are decided ≥15 plies before our oracle sees the forced win.
- **The knife-edge, measured.** Pre-onset, 37–81% of the net's top-16 candidates hand the
  attacker an immediate VCT.
- **Saturated readings can still rank.** The null-move sense looked useless on an absolute scale
  (mean 0.98 near onset) but ranks moves within a position well. The detector reads the test,
  and a ranking survives that.

Bets (`BETS_sense.md`):
- **Jason, "the detector is useful": CONFIRMED.**
- Claude:
  - pressure informative, AUC ≥ 0.65: confirmed;
  - standalone beats the best net signal by +0.03: refuted (the value head is better on absolute
    AUC);
  - replication ≥ +0.03 combined: confirmed (+0.052);
  - actuator ≥ 5 points: refuted (−1.0, n.s.);
  - pure walt worse: confirmed.
- Advisor, agreement with Rapfi +0.04: near miss (+0.034 null, +0.001 escape).
- E-A (odds-map information test) wasn't run separately: the detector test supersedes it.

## Verdict, part 1: walt as a player (2026-10-07)

texas-42 walt (level-k best response over sampled hidden deals, played from trick 1) was moved
onto perfect-information 9×9 freestyle, with the GPU mega-VCT oracle as its exact terminal.
- **Against Rapfi-NNUE** (50 ms and 1 s, about 20 games per arm), walt ties MCTS using the same
  champion net.
- **Against weaker opponents it looks great:**
  - 10-0 and 9-1 vs its own field model;
  - 4-0-1 head-to-head vs MCTS1600 at half the wall time;
  - converts more wins than MCTS vs lookahead4.

**Why.** The fast VCT oracle **detects opponent mistakes**, and walt's best response over the
field tapes **amplifies that signal**. walt steers into positions where the modeled opponent lets
a forced win appear.
- The share of leaves the oracle resolves falls from **53% vs the field** to **12% vs Rapfi@50**
  and **4% vs Rapfi@1000**.
- The finisher fires on 38% of moves vs the field, 1.4% vs Rapfi@50, and on 0 of 53 moves vs
  Rapfi@1000.
- Rapfi never lets a VCT appear. walt is then just net@T1 playouts plus a value-head cutoff,
  which carries no more information than MCTS with the same net.
- This is walt's own documented failure mode (plunge cross-game reports): **the gain vanishes
  once the opponent isn't the modeled field.**

**The deeper why: there is almost no hidden information to reason about.** Two candidates for
the hidden information:
- **Opponent tapes** (the random seed of a stochastic opponent model). These are a manufactured
  stand-in for a deal.
- **Jason's reframe (the right one): logical uncertainty**, i.e. *where the VCT is*. The truth is
  fixed, and you see it as you approach, the way cards are revealed by play.

Measured on 16K field positions:
- A clean no-win at 50 nodes **never** flips to a win at 2,000 nodes (it's a proof).
- Overall, only 0.7% of positions flip.
- Of the 14% that hit the 50-node cap, 4.9% flip, and most stay unknown even at 40×.

The hidden card is rare, rarely a win, and mostly unrevealable at any affordable budget.

## What we built

All code is in `scripts/walt/`.
- **Tickertape.** The field is policy (net@T or uniform-near) plus the VCT finisher, sampled via
  `splitmix(zobrist(board) ^ tape)`.
  - The same tape at the same position draws the same move in every branch, so tapes partition
    into buckets instead of multiplying (texas-42 Def 3.5).
  - The finisher makes a VCT leaf an exact statement about the modeled field.
- **Level 1.**
  - walt's nodes take the max over the top-M prior candidates, choosing one action that covers
    every tape.
  - Field nodes sum over the buckets of tapes.
  - Payoff is W=2, D=1, L=0 per tape.
- **Native-batch lockstep rollouts** (`walt_vec.py`).
  - All (leaf, tape) playouts advance one ply per net call plus one VCT call.
  - Parity-checked byte-identical against `GameState` (planes, five-check, rollouts).
- **Rollout cap.** After `cap` plies the value head scores what's left. Needed because cost is
  plies × per-call latency.

## Measurements

| Finding | Number |
|---|---|
| Oracle batch curve, cap50, real 9×9 boards | 16 → 50 ms · 1K → 227 ms · 4K → 202 ms · **16K → 264 ms (62K boards/s, the knee)** · 32K → 470 ms |
| Before batching (K=32) | 70 calls/move at ~53 boards; VCT = 90% of decision wall |
| More tapes, K 32→4096 | 128× the samples for 2.7× the time; **decisions identical on all 6 test positions** (bias-limited, not sample-limited) |
| H=4 cost | 11.5 s/move with a 69-ply lockstep tail → **5.4 s** at cap 12 (16 calls × full 16K batches) |
| Samples / depth / width at equal batch (vs lookahead4) | 6-0-4 / 7-0-3 / 6-1-3: no lever moved strength |
| Rapfi, white column | **0 wins as white for every arm** (the chronic white-defense gap; walt doesn't touch it) |

Bets were stated before results (session 2026-10-07):
- **B1** (leaves go to rollout): confirmed.
- **B2** (oracle-latency mismatch): confirmed, then fixed by batching.
- **B4** (little hidden mass): confirmed.
- **B3′** (walt doesn't beat MCTS): **refuted against net-shaped opponents, confirmed against
  Rapfi.**

## What would still be worth trying (not run)

- **Train on the escape-pressure label.** It's information the net lacks (+0.05 AUC). Distilling
  it into an aux head risks repeating #103 (sensor, no actuator). The actuator result above says
  the gain at move time is small.
- **Test the hypothesis in walt's home games.** Does walt's count beat a *learned* detector of
  the hidden deal (e.g. an E[Q]-style net) in texas-42? If a strong learned sense makes walt
  redundant there too, the gomoku lesson generalizes.

- **walt as an exploit finder.** Point it at a frozen net as the field to map *where* that net
  leaves VCTs open. It's a diagnostic, not a player.
- **Adaptive sampling** (texas-42's racing, paired sign-test elimination). It's moot here because
  the estimator is bias-limited, not variance-limited.

## Evidence

- Part 2: `~/code/gomoku/sweep_logs/walt_sense/` (Rapfi game shards `rapfi9*/`, `expB_back135/`
  (degenerate near-onset run), `expB/` (first detector read), `expB_rep/` (replication); `*.log`).
- `~/code/gomoku/sweep_logs/walt/` (gitignored evidence: phase1, phase2_bench, phase2b, phase3 JSONL; per-move walt diagnostics).
- [TRAINING_WIKI.md](../../TRAINING_WIKI.md) 2026-10-07.
- Branch `feat/walt-gomoku`.
- walt sources:
  - `~/code/texas-42/wiki/walt.md`
  - `~/code/plunge` branch `claude/walt-game-implementations-wrsu6q` (`lab/WALT-CARD.md`)
  - **Not** mk5-main's walt, a different algorithm with the same name.
