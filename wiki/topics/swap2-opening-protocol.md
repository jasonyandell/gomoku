# Swap2 opening protocol (#72) — the real fix for white

**Status (2026-06-20) — ERA 1 DONE, PIVOTING TO ERA 2.** Swap2 era-1 (warm-started from the
pre-swap2 champion) achieved its actual job — **the bootstrap fix is CONFIRMED**: swap2
self-play makes white *trainable* (white ~30–40% of decisive self-play games vs ~0% on an
empty board), and the net beats the frozen old champ a steady **~70%** (gates e235→e455:
57.0 → 66.8 → 76.2 → 67.6 → 70.7), white LOSS-rate down off the 88% floor to ~parity.

**But the Rapfi gut-check (§5.7) reframes all of that.** At e455, vs the real Rapfi-NNUE
engine (same config as our early anchors): **10.2% — statistically identical to the
pre-swap2 warm champ's 10.4%, and white is 0/11 = 0% vs Rapfi.** So in *absolute* terms the
whole era-1 run was **flat**: the net got better at beating its own weak ancestor (the
70%-vs-old-champ was a **saturated ruler**), not better at gomoku. Swap2 was never a
strength fix — it was the **prerequisite** (unblock white's gradient). Era-1 reached the
bottom of the hill; it did not climb.

**Diagnosis — the "lose-slowly basin" (§9).** White is 0% vs Rapfi yet ~parity vs the old
champ → it plays to *not lose*, not to *win*; plies climbed 27→50 (it learned to *stall*,
not finish). Smoking gun: we **warm-started the pre-swap2 champion**, whose optimal policy
in the old imbalanced regime literally *was* "delay the loss / steer to draw" — that
defensive attractor is baked into the weights. Swap2 fixed the data; the weights kept the
basin.

**Era 2 = Path A, now the 9×9→15×15 CURRICULUM (§9, revised 2026-06-20):** a **fresh-init**
net (no inherited basin) + **aggression shaping** (value-discount 0.98→0.95, "faster wins
are better"), bootstrapped **on the native 9×9 board** (3.2× cheaper, white trainable) then
**warm-started up to 15×15** ([[board-size-transfer-and-warm-start]] — the proven 98.9%-trunk
transfer that birthed the 15×15 nets in the first place). **v2a (choice head) is OFF** — we
killed it for the curriculum: it trains an opening-only head at ~20% throughput cost for no
measured gain; revisit *after* white-not-doomed is locked in.

**Live (2026-06-20, era-2 phase 1, 9×9, run `lywhy1ba`):** white-not-doomed is **showing** —
white takes **~45% of decisive self-play games** (47.7% at e47) from e17 on, vs era-1's
white-0%-vs-Rapfi basin. `plies_mean` is mid-**plies-collapse** (36→16.7) — the expected
offense-before-defense dip, healthy *because* white still wins ~45% through it. Watching for
the **warning**: white% trending *down toward 0* (basin relapse) — not seen. Model card:
[[gomoku-9x9-swap2-era2]].

This page is the durable synthesis. Evidence chronology lives in `TRAINING_WIKI.md`
(2026-06-20 entries); predecessor analysis is
[white-side-defense-plan.md](white-side-defense-plan.md).

Branch: `feat/swap2-opening-protocol` (worktree, **unmerged** by request). Era-1 run: cell
`G15-swap2`, wandb `8nq1a7cm`, board 15, home `/Users/jason/data/swap2/`. Era-2 run: cell
`G15-swap2-e2` (fresh, no `--resume`). Intermediate model capture → **HuggingFace push**
(not `cp` anchors), per Jason 2026-06-20.

---

## 1. Why swap2 (the road here)

The 2026-06-20 white-defense investigation closed a long arc: 15×15 freestyle
white-side "defense weakness" is **not a net flaw — it is the first-player-win
theorem.** Evidence (full chain in white-side-defense-plan.md):

- Three policy/value teachers ALL flattened (value-only #42 poisoned the value head;
  sparse-VCF #43 and dense-conv left white at 0-2/20 vs Rapfi).
- The diagnostic showed white **retreats competently to a forced loss** — it blocks
  fours ~perfectly, has initiative on 1 ply / 30 games, is forced into an unstoppable
  double-four in 28/30. There is no tactical error to correct.
- **Clincher:** Rapfi(1s) vs Rapfi(1s) from 4-stone openings → white 1-9. Even the #1
  engine playing itself is crushed as the second player.

So no teacher can make a (near-)solved-lost role win. **The fix is to delete the
forced role: swap2** — Gomocup's balancing protocol, where a player is never *forced*
onto the lost side. That also makes the Rapfi yardstick honest (Rapfi is a swap2
engine).

## 2. The ML thesis (why this is a bootstrap fix, not just a yardstick fix)

An imbalanced game **cannot bootstrap**, because the self-play *data* collapses: when
one side is hopeless, every game is a first-player win, so the value head only ever
sees "second-player = lost" and the policy gets **zero gradient on winnable
second-player positions** — there are none in its own data. No teacher bolted onto the
side can fix a distribution that contains no balanced games.

Swap2 rebalances the **game** (the responder is never forced onto the lost side), so
self-play generates ~50/50 data → second-player positions become winnable in the
training set → the heads finally get real defensive gradient. The honest yardstick is a
bonus; the **data rebalancing is the mechanism.**

## 3. What swap2 is (the negotiation)

Cross-confirmed (Wikipedia "Gomoku", RenjuNet rule 11, Gomocup `SWAP2BOARD` protocol):

1. **Opener** places 3 stones = 2 black + 1 white, anywhere.
2. **Responder** chooses one of:
   - **STAY**: take white and place a 2nd white stone → 2B+2W, opener moves next as black.
   - **SWAP**: take black (no new stone) → 2B+1W, opener moves next as white.
   - **PLACE2**: add 1 black + 1 white → 3B+2W, then the **opener picks a color**; white
     moves next either way (the pick only decides *which player* is white — the board is
     identical).

There is **no canonical freestyle opening book** (the renju "26 openings" exist only
because renju restricts black; freestyle is unrestricted). Swap2 doesn't need one — the
**negotiation is the balancer**: an unfair opening is punished by the responder's choice.

## 4. The architecture (full Path A)

We chose the complete path — the net *can* learn to negotiate — but kept the scary part
small. Key insight: **almost all of swap2 is still placing stones on squares**; only two
moments are abstract decisions.

| Piece | File | Idea |
|---|---|---|
| Negotiation state machine | `gomoku/swap2.py` | `OpeningState` in **absolute** (black,white) planes (the opening isn't alternating-color). Color fixed by placement ORDER (B,W,B,…) so every placement stays a plain spatial move over the existing board policy head — **zero action-space growth for placements**. Only the two negotiation moments are abstract (width-3 choice space). Value attribution via explicit OPENER/RESPONDER actor tags + `backup_sign()` (the opener acts 3× in a row, so "flip perspective every ply" does NOT hold). |
| Choice head | `gomoku/model.py` | `forward_with_choice() → (policy, value, choice_logits)`; a tiny `Linear(value_hidden, 3)` off the value-head penultimate. The existing `forward()` and every caller stay byte-identical. Warm-start-tolerant load: core strict, fresh choice head spliced for the champion (which predates it). |
| Negotiator (v1) | `gomoku/swap2_search.py` | `negotiate(oracle, rng)` drives the opening. PLACE nodes **sampled** for diversity (not trained). CHOICE nodes selected by a **one-ply value comparison** (no trained head needed yet) with honest minimax over the nested opener pick; re-signed via `backup_sign`. ~30 net forwards/game (~0.2% overhead, no MCTS). Emits `choice_records` as future choice-head targets. |
| Self-play wiring | `gomoku/self_play.py` + worker/train/run_sweep | `--swap2` flag at the `_random_opening_state` seam in all four gen paths; mutually exclusive with `--random-opening-moves`; **byte-identical when off**. Each game starts from `negotiate().normal_state`. |
| Eval vs engine | `gomoku/eval_swap2.py` + `gomoku/external_engine.py` | Both sides negotiate: our net via `agent_act`, the engine via the `SWAP2BOARD` protocol path. The honest gate (no forced-white floor). |
| H2H gate | `gomoku/eval_swap2.py` `eval_swap2_h2h` | Net-vs-net swap2 (both negotiate), jobs-parallel, exact-deterministic. The resolvable progress gate (see §5). |

~12 commits, 100+ tests. The negotiation, protocol, choice head, and value attribution
are all tested in isolation before any net/MCTS touched them.

## 5. What we learned (the evidence)

### 5.1 The core bet is CONFIRMED at the data level ⭐
Measured color balance of 64 recent swap2 self-play games (`_records` GameRecords):
**white wins 27% (black 69%, draw 5%).** Empty-board self-play gives white ~0%. So
white is now genuinely **winnable in the training data** — the heads get gradient on
winnable white positions for the first time. This is the bootstrap an imbalanced game
can't do, working. It is the single most important result of the run.

### 5.2 The negotiation mechanism works
In net-vs-net H2H the **responder wins ~80%** — it exploits its stay/swap/place2 choice
to take the better side (`opener_color_dist` shows the responder almost always grabs
black). Swap2's balancing flows through the responder's choice, as designed.

### 5.3 Strength-vs-champion is HOLDING ~70% — and the white side is around parity (the payoff)
The progress gate is net-vs-net swap2 H2H, **trained-latest vs the FROZEN warm
champion** (both negotiate). Same settings each gate (n=64, sims=200, seed 7) so the
points are directly comparable. **High-resolution trend** (win-rate is from the
trained net's view; splits are `W-L`):

| gate | epoch | n | overall | **white LOSS-rate** | as black | as opener | as responder |
|---|---|---|---|---|---|---|---|
| slice 2 end | e129 | 64 | 51.6% (33-31) | **88%** (3-22 W-L) | 77% (30-9) | 22% (7-25) | 81% (26-6) |
| slice 3 end | e181 | 64 | 64.1% (41-23) | 59% (12-17) | 83% (29-6) | 50% (16-16) | 78% (25-7) |
| slice 4 end | e235 | **128** | 57.0% (73-55) | **67%** (20-40) | 78% (53-15) | 41% (26-38) | 73% (47-17) |
| slice 5 end | e289 | **128** | **66.8%** (85-42-1) | **51%** (29-30) | 81% (56-12) | 52% (33-31) | 82% (52-11) |
| slice 6 end | e345 | **128** | **76.2%** (97-30-1) | **42%** (21-16) | 84% (76-14) | 67% (43-20) | 84% (54-10) |
| slice 7 end | e403 | **128** | **67.6%** (86-41-1) | **57%** (25-33) | 87% (61-8) | 50% (32-32) | 84% (54-9) |

(White column is the agreed metric — LOSS-rate, not win-rate; white's ceiling is the draw,
§6.6. Lower is better. The white sample is small per-gate (e345 n=38, e403 n=58) with
CI ~±13-16% on the loss-rate, so read each gate's white number as directional — the
negotiation/seed interaction shifts the color mix gate-to-gate, and all the variance is
concentrated on the hard white/opener side.)

**The LEVEL is holding ~70% overall — and the white side is fluctuating around parity, not
descending smoothly.** Discarding the e181 n=64 overshoot, the n=128 anchors read
**57.0% (e235) → 66.8% (e289) → 76.2% (e345) → 67.6% (e403)**, mean **~70%**, comfortably
above the ~58% relative-crown lower bound (§6.6). **Correction to the prior "slope
steepened" read at e345:** the e289→e345 jump to 76.2% was *partly* an upward
sample-fluctuation (same flavor as the earlier e181 64.1% n=64 spike), not a genuine
acceleration — the e403 pullback to 67.6% confirms the true level is ~70%, not a steepening
climb. The e403 dip is a **PULLBACK within noise, NOT a regression**: the CIs overlap
heavily (overall e403 [59.5%, 75.7%] vs e345 [68.8%, 83.6%]; white-LOSS e403 [44%, 70%] vs
e345 [26%, 58%]). On the white side the honest statement is now: **white LOSS-rate fell
from 88% (e129) to ~50% (now) and is fluctuating around parity (42% e345, 57% e403), NOT
monotonically marching to 0.** The early monotonic read (88→67→51→42) was real *as a fall
from the catastrophic floor*, but at the ~parity level the gate-to-gate motion is noise, not
a clean descent. Black (87%) and responder (84%) sides stay strong; the variance lives
entirely on the hard white/opener side, which also carries the smaller sample. White-side
defense is genuinely improved — via balanced data, not a teacher — and is holding around
parity vs the frozen champ's defense. Verdict at ~403 epochs: **NOT a plateau, NOT a
regression, still well above the crown bar — keep training.** But recalibrate expectations:
progress on the white side is **noisy around parity, not a smooth descent.** All gates use
n=128 (n=64 is too noisy — it produced the 64.1% overshoot). Lesson: read the *trend across
independent checkpoints* and its CIs, not any single gate — a single high gate can be an
upward fluctuation (as e345 partly was).

**Epoch context (why this isn't suspiciously fast):** the white move appeared between
e129 and e181. General AZ wisdom is "thousands of epochs to move," but THIS project's
lived experience is that real movement typically shows in **~100 epochs** (laptop-scale,
small buffer, high SGD-per-position). So a white-side shift at e129→e181 (~52 epochs of
balanced data on top of a warm start) is **on-schedule for this setup, not anomalous** —
which makes it more credible, not less. The "thousands" figure is the conservative
outer bound; ~100 is this lab's empirical inner bound. Keep gating every slice — a single
n=64 gate is a data point, the *trend across independent checkpoints* is the result.

**The pre-H2H gates were noise (do not use for the trend).** Gate-1/2 vs Rapfi at n=16-48
gave baseline 4-25% and trained 4-31% on pure variance (§5.6) — discarded in favor of the
H2H-vs-champion gate above.

### 5.4 The data isn't perfectly 50/50 — and we know why
Black still wins 69% because v1 **samples** opening placements for diversity rather than
**training** them, so the opener never learns to place a *fair* opening and the responder
keeps a swap-to-black edge. This is the seam the next lever attacks (§6).

### 5.5 Healthy training dynamics throughout
`vl` bounces ~0.14–0.21 (NO value-poisoning collapse to 0.06 — the failure mode that
killed value-only #42). `plies` rose ~30→42 (more contested games — the *opposite* of
fast-attack collapse, where plies fall). No fast-attack collapse, no policy blowup.

### 5.6 Methodology lesson — the Rapfi gate is noise-dominated near the floor
Vs Rapfi-NNUE our net sits at single-digit-to-~30% win-rate; at n=16–48 the SAME fixed
baseline reads **4%–25% on pure noise** — variance swamps the ~5-8pt signal. Both an
encouraging "31%" and a scary "4% (worse than baseline!)" were noise; a serial
diagnostic showed trained 12.5% > baseline 4.2% at matched seed (no regression). **Gate
"did this help?" on H2H vs the preserved champion (≈p0.5, resolvable), NOT on Rapfi**
(this is the wiki's own 2026-06-15 rule). Rapfi stays a coarse absolute anchor only.

### 5.7 The Rapfi gut-check (e455) — absolute strength is FLAT (the saturated-ruler finding) ⭐
The H2H-vs-old-champ trend held ~70% but stopped *climbing* (66.8 → 76.2 → 67.6 → 70.7),
which read as "relative plateau." To tell a real plateau from a **maxed ruler** (we'd
outgrown the weak frozen reference), we took the absolute read:

| Net | vs Rapfi-NNUE (200ms, sims=200, seed7, n=128) |
|---|---|
| warm champ (pre-swap2 baseline) | 10.4% |
| e129 (early swap2) | 4.2% |
| **e455 (end of era-1)** | **10.2%** — black 11%, **white 0/11 = 0%** |

**e455 is statistically identical to the pre-swap2 champion vs a real engine** (10.2 vs
10.4, n=128; CIs fully overlap), and **white still wins 0% vs Rapfi.** So the 70%-vs-old-champ
was almost entirely a **saturated ruler** — the net learned to beat its own weak ancestor,
not to play stronger gomoku. The "white parity" is *relative to the old champ's defense
only*. **Conclusion: era-1 moved relative strength and zero absolute strength.** That is not
a failure of swap2 — swap2's job was the *bootstrap* (white trainable), and it did that.
It's the signal that the *strength* climb needs a different lever (§9). **The deeper read
settled the fork:** e455 vs Rapfi at **sims=800** (4× search) = **11.7%** (15W-113L), white
**still 0/17 = 0%**. 4× search bought ~1.5pt (inside noise) and white couldn't find a single
win given 4× the thinking time → **NOT search-starved; basin/capacity-bound.** More search
of a defensive policy is still defensive. This rules out the "just add self-play sims"
shortcut and validates the era-2 basin-escape levers (§9); if a *fresh* aggressive net still
can't make white win, the wall is genuinely capacity → Path B.

**Methodology upgrade:** because the frozen old champ is now a saturated ruler, future
relative-progress gates must **re-anchor to a recent self-checkpoint** (e.g. latest vs
frozen-e455), not the ancient champ. Era-1's checkpoints were lost to `keep_last_n=3`
pruning — going forward, capture intermediates via **HuggingFace push** (not `cp`), which
also seeds the always-running-arena model registry.

## 6. What to try next (ranked) — SUPERSEDED by §9 (kept for the era-1 reasoning trail)

1. **Let v1 run and read the H2H trend first.** Don't add machinery ahead of the
   measurement (the exact lesson the teacher era taught). Gate trained-latest vs the
   frozen champ each cap. **As of e181 this has fired positive** — H2H past 60% with white
   12%→41%, so v1 (balanced data alone) is *already* showing it works; the job now is to
   CONFIRM the trend across more independent checkpoints (n=128 gates) before declaring a
   result. If it stalls, escalate to #2; if it holds, v1 is the win and #2/#3 become the
   "push 27%→50% and go further" lever rather than a rescue.
2. **Learned negotiation = the choice head into the loss (the deferred v2).** The choice
   head exists and the negotiator already emits `choice_records`, but those targets are
   **not yet in the trainer loss** — v1 negotiates by a one-ply value lookup. Wiring them
   in lets the net *learn* to negotiate. Risk surface: the record format + a choice-head
   loss term; build/test in isolation, deploy as a NEW cell (do not mutate a live run).
3. **Train opening PLACEMENTS for fairness.** v1 samples placements (untrained), so the
   opener can't learn a fair opening → the responder keeps the black edge (data stuck at
   69/27, not 50/50). Recording + training the opener's placement policy (punished when
   the responder profitably swaps) should push the data toward 50/50 and let white-side
   strength actually climb. Pairs naturally with #2.
4. **More warm-started training time.** 2h is short. The balanced data is in the buffer
   now; the value head is already fitting it (vl ↓ to ~0.14). Patience may convert
   balanced data into strength without new code.
5. **Bigger/sharper gates once a candidate looks real.** H2H n=128+ and a multi-seed
   Rapfi anchor at the strong tiers to put an absolute number on a confirmed H2H gain.

## 6.5 Why not vet this faster on 9×9? + the scale (Modal) era

**9×9 cannot carry the white-defense signal — it is qualitatively absent, not just
smaller (Jason, 2026-06-20).** On a small board the *edge is free defensive structure*:
white runs the attacker into a wall and the geometry does the defending, so white doesn't
have to be clever (the 9×9 champion already defends near-perfectly, white-loss ≤5%). The
signal exists only where there's enough open space for a competent attacker to force
unstoppable double-threats — i.e. it **scales with board size**. So 9×9 vets the
*plumbing* (does the negotiation run, does a learned choice head train) but tells you
NOTHING about whether white got better at defending — measuring a thermometer in a room
with no temperature gradient. **The white-strength question must be answered on a big
board.** (Middle ground: 13×13 has enough space for the signal at ~2× the speed of 15×15,
but the native MCTS ext is only compiled for 9/15 — 13 falls back to slow pure-Python gen,
eating the speedup unless you build a 13 ext. Buildable, not free.)

**Scale / "thousands of epochs" era (Modal, deferred).** The M5 Δelo/hour ceiling is
self-play GEN (CPU-bound; the trainer barely touches MPS at ~6% CPU) — an A100 trains
faster than this box can feed it. So the cloud unlock is **parallel gen** (fan self-play
across many cheap CPU workers + a shared GPU evaluator), not a bigger GPU per se. Port
friction is bounded to two things: (a) the native MCTS ext is compiled per-arch — needs a
CUDA/Linux build or it falls to slow pure-Python; (b) `run_sweep`'s local-subprocess
orchestration → cloud functions. The torch training loop itself is portable (mps→cuda is
trivial). Staging: **validate the recipe on the M5 first** (does swap2 + the learned
choice head actually move white — the current run), **then** pay the port tax for the
serious thousands-of-epochs run where parallel gen is the point. See the
[containerize-training-runs](containerize-training-runs.md) seam (already flagged: "no
MPS in Docker on macOS → targets off-Mac/at-scale").

## 6.6 Crowning a new champion — the bar (approved 2026-06-20)

Two separate questions; keep them apart.

**RELATIVE crown ("is this our new best net?") — the gate.** Beats the FROZEN preserved
champion under swap2 H2H (both negotiate):
- **Overall ≥ ~58-60% at n ≥ 200** (CI lower bound clearly above 50% — at n=128 a 60%
  reading still grazes significance, so n must grow to *call* it).
- **Diagnostic, must not regress:** black-side win-rate ≥ the champion's; white-side
  **LOSS-rate ≤ the champion's**.
- On crowning, freeze the new net as the next H2H anchor.

**Why white is judged on LOSS-rate, not win-rate.** In a first-player-win game perfect
white play is a **draw**, not a win — demanding ">60% wins as white" asks for something
that, if it happened, would disprove the first-player-win premise. White's job is to not
lose (#18: "100% as black, 0% loss as white"). So report **white loss-rate → low**, never
white win-rate → high. Also note: under swap2 a strong net rarely *ends up* white vs a
peer (it negotiates away from it), so "as white" is a small, selected sample — the cleaner
white-robustness probe is a separate FORCED-white stress test (4-stone openings) where you
require white-loss to fall vs the champion's 0/12-swept baseline.

**ABSOLUTE strength ("how good in the world?") — tracked, NOT a crown gate.** vs
Rapfi-NNUE under swap2, overall win% at a fixed TC (e.g. 1000ms), n ≥ 100, both
negotiating the real protocol. You can be the new champ without beating the world #1; the
headline absolute story is the champion's forced-white-swept ~21-27% climbing toward
40-50% overall under swap2 (no forced lost role).

**Forward (the vision, NOT today): an always-running SWAP2 arena.** Generalize the swap2
match primitive (`eval_swap2` net-vs-engine + net-vs-net, jobs-parallel) into a persistent
ladder where external champs (Rapfi + the [gomocup engines catalog](gomocup-engines-catalog.md)),
our historical model checkpoints, AND new candidates all play continuously under the real
swap2 protocol → a live Bradley-Terry Elo. The pieces mostly exist: the swap2 match
primitive (built), `panel_tournament.py`'s BT-fit + per-pair JSONL, and the autolab arena
daemon's pull-checkpoint/schedule/append loop. The upgrade swap2 brings is **honest Elos**
— a swap2 arena has no forced-white floor, so it fixes the yardstick at the arena level
(the old forced-color panel was broken for white; see the 2026-06-15 reckoning). This is
the post-validation era — validate the recipe on the M5 first, then build the arena.

## 6.7 Contingency plan — "if it stops working" (decision tree)

Standing directive (2026-06-20): **don't stop what's working** — keep the live run on its
current recipe + cadence. This menu is for *if/when* a gate says it stalled. Every option
deploys as a NEW cell (never mutate the live run), validated in isolation, gated vs the
frozen champion. First, triage: is it RECIPE, OPTIMIZATION, or OPERATIONAL?

**A. PLATEAU** (most likely "stops working") — H2H flat / white-loss stops falling across
≥2 consecutive **n=128** gates. First ask: plateaued ABOVE or BELOW the crowning bar
(§6.6)?
- **Above the bar → not a failure: CROWN it.** v1 is the win; freeze it, move to the
  arena/Modal era.
- **Below the bar → escalate the lever (in order):**
  - **(P1) Learned choice head into the loss** — wire `choice_records` into the trainer so
    the net *learns* to negotiate instead of the v1 value-lookup. Pushes self-play balance
    past 69/27 → raises the white ceiling. The pre-identified #1 escalation.
  - **(P2) Train opening placements for fairness** (pairs with P1; the opener learns fair
    3-stone openings so the responder can't always swap to a black edge).
  - **(P3) Stronger self-play** — more sims / a VCT/VCF attack teacher on top → better data.
  - **(P4) Capacity** — if 128×10 saturates on balanced data, grow the net (net2net).

**B. REGRESSION** — H2H / white-loss *worsens* beyond noise. FIRST confirm it's real at
n=128 (we were burned by n=64 noise — the e181 64% overshoot). If real: roll back to the
best epoch checkpoint; suspect **LR too high** (the value-target shift from balanced data
wrecking the policy → #44: lower LR / freeze-value-head warmup); check buffer composition.

**C. DYNAMICS DEATH-TELL** (well-documented: `loss-floor-bouncing.md`,
`az-at-scale-vs-laptop.md`) — `vl` → ~<0.08 = value poisoning; `plies` falling + concave
buffer-fill = fast-attack collapse; `pl` runaway = policy corruption. Roll back to the last
healthy checkpoint, lower LR, inspect labels/buffer. (So far ALL absent — vl bounces
0.14-0.21, plies rose 30→42.)

**D. DATA STUCK at ~27% white** — self-play white-win% flat, not trending toward 50%. This
is the v1 ceiling (sampled, untrained openings cap balance), so it **converges on P1+P2**
— the learned-negotiation lever is the fix for both plateau-below and data-stuck.

**E. OPERATIONAL** — process death / OOM / disk / my-session-lapse. The run is restartable
(`--resume latest.pt`, buffer embedded). For unattended resilience WITHOUT changing the
recipe: a self-relaunching wrapper that does exactly cap→gate→relaunch automatically
(removes the human as the single point of failure; honors "loops are cadence, not
load-bearing"). Identical behavior to the current loop — just not dependent on a live
session.

Meta: A/D are RECIPE (escalate the lever), B/C are OPTIMIZATION (roll back + LR), E is
OPERATIONAL (resume/automate). A plateau *above* the bar is the happy ending, not a
failure.

## 7. Operational notes (durable gotchas)

- **Warm-start = weights only.** Strip the champion to `{model_state_dict, model_config}`
  (drop the embedded buffer/optimizer/wandb) → fresh buffer + fresh optimizer (which
  covers the new choice-head params) + new wandb. Do NOT `--resume` the champion's 1.5M
  buffer — the white-defense era proved it drowns new signal. Use a **small fresh 150k
  buffer** so it turns over to the swap2 distribution fast (~3 min).
- **Worktree venv isolation.** `run_sweep` spawns subprocess workers/trainer that
  `import gomoku`; the shared editable install points at *main*, and native `.so`
  extensions live only in main's `gomoku/`. Give the worktree **its own venv**
  (`uv venv && uv pip install -e ".[dev]"`, which compiles the board-15 native ext into
  the worktree) so spawned procs import the worktree's swap2 code. Launch with
  `GOMOKU_BOARD_SIZE=15` and the worktree venv python (`PYTHON = sys.executable`
  propagates it to children).
- **Rapfi needs `x,y,color`.** SWAP2BOARD stone lines must be `x,y,color` (1=black,
  2=white); bare `x,y` → Rapfi errors and opens on an empty board. Drive the binary
  directly (`pbrain-rapfi --config <cfg> gomocup`) — the `run-rapfi` wrapper supplies the
  NNUE config but can lose its +x bit.
- **Eval must run during the PAUSE.** A parallel eval concurrent with the 8 gen workers
  oversubscribes the M5 and starves the net's CPU inference → contaminated numbers. Pause
  the trainer, eval on the free box, resume.
- **Babysit spine:** `/Users/jason/data/swap2/babysit/` holds the append-only `ledger.md`
  (slice/gate history), `run_eval.py` (vs-Rapfi), `run_h2h.py` (vs-champion). Slices are
  1h self-capping `run_sweep --max-wall-secs 3600 --resume latest.pt`.

## 8. Done / not done

**Done:** the full Path A pipeline (negotiation, choice head, negotiator, self-play
wiring, both eval harnesses), warm-started live run, the data-balance confirmation, the
H2H gate. **Not done:** a proven *strength* gain over the champion; the learned-choice-head
loss; trained opening placements; merge to main (held by request).

## 9. The new era (era-2 / Path A) — escape the lose-slowly basin ⭐

**This section supersedes §6.** Era-1 proved the bootstrap and revealed (via the §5.7 Rapfi
gut-check) that absolute strength never moved. The unifying diagnosis is a *behavioral
attractor*, not a capacity wall:

**The "lose-slowly basin."** Every era-1 number fits one story — the net plays to *survive*,
not to *win*:
- white 0% vs Rapfi but ~parity vs the old champ → a not-lose policy, not a win policy;
- `plies` climbed 27→50 over the run → it learned to *prolong*, not *finish*;
- **smoking gun:** era-1 **warm-started the pre-swap2 champion**, whose optimal policy in the
  old imbalanced regime literally *was* "delay the loss / steer to draw." That defensive
  basin is encoded in the weights. Swap2 fixed the *data*; the *weights* kept the basin, so
  incremental swap2 training on top of it never escaped.

### 9.1 Path A — the era-2 recipe (cell `G15-swap2-e2`, fresh, no `--resume`)
One run that bundles the three basin-escape levers:

1. **Fresh init (don't inherit the basin).** Launch with NO warm-start / NO `--resume` →
   random weights, empty buffer, fresh AdamW. Swap2's whole point is that even a *fresh* net
   gets balanced data from move one, so fresh+swap2 bootstraps white *without* importing the
   defensive champion. (The old warm-start was a speed hack that cost us the basin.)
2. **Aggression shaping — "faster wins are better."** `value-discount` 0.98 → **0.95**
   (worker arg). Effect: a 50-ply win's value target falls 0.36 → 0.077 (3.6× weaker than a
   25-ply win, vs only 1.66× under 0.98) → the value head — and through the shared trunk the
   policy and choice head — strongly prefer crisp, short wins. Risk: too steep blurs the
   value head on long games; back off to 0.96 if it bites, never below ~0.95.
3. **v2a — learn the swap (train the choice head into the loss).** Era-1's negotiation was a
   one-ply value heuristic and the choice head was scaffolded-but-untrained (records were
   *discarded at the seam*, `_swap2_opening_state`). v2a threads `Swap2Result.choice_records`
   out of the 4 gen seams → `GameRecord.choice_examples` → a small separate `ChoiceBuffer` →
   a masked choice-CE loss (`--choice-head-weight 0.3`, default 0.0 = byte-identical off).
   Target is **outcome-driven** (the head learns which negotiation slot actually *won*), with
   the chooser-sign from `backup_sign`'s **actor** form: `outcome_for_black` is really
   outcome-for-the-handoff-mover (`res.mover_actor`), so a choice node's chooser gets
   `+outcome` iff `cr.to_act == mover_actor`, else `−outcome` (handles the opener-acts-3×
   perspective without ply parity). **Staging:** v2a *trains* the head; the negotiator still
   *selects* via the one-ply heuristic (a fresh net's random head would negotiate garbage at
   cold-start). **Consuming the head for selection is v2b** — the immediate fast-follow once
   the head's loss is falling and `choice_buffer` has accumulated.

Keep buffer modest (150k) at cold-start so old slow/defensive games don't dilute the new
fast-win objective; grow to ~1.5M (bigbuf, `--pack-buffer`) as a *later* slice once the
win-length distribution is shortening and choice-loss is falling.

### 9.2 The fork: Path A first, Path B if A can't make white WIN
- **Path A (this run):** cheapest decisive test of the basin hypothesis *on the target
  board*. Success signal = white starts actually *winning* (vs Rapfi, not just not-losing)
  and the win-length distribution shortens.
- **Path B — 9×9→15×15 transfer curriculum:** the bigger "serious-model / Modal-era" bet if
  A still can't make white win (→ board-difficulty/capacity is the real wall). Feasible
  because the net is `global-pool` → the conv **trunk is board-size-agnostic and transfers
  directly**; only the policy head (81→225 outputs) re-inits and fine-tunes. Native ext
  exists for 9 and 15; zeb has transfer patterns to crib.

### 9.3 Measurement for era-2 (don't repeat era-1's blind spot)
- **Re-anchor** relative gates to a recent self-checkpoint, never the ancient champ
  (saturated ruler). Capture anchors via **HuggingFace push**, not `cp` (Jason 2026-06-20).
- **Absolute** read vs Rapfi at the same fixed config each gate (the honest yardstick) +
  watch the **win-length distribution** (is it learning to finish?) and self-play
  white/black balance (smoothed, not single-epoch — those swing wildly).
- North-star for "is the basin breaking?": **white win% vs Rapfi climbing off 0%**, and
  mean plies *falling* on wins.

## 10. The fairness diagnosis — the OPENER is the broken half (Rapfi-confirmed, 2026-06-21) ⭐

Jason's gut (2026-06-21): "our swap2 isn't a fair game — black has an edge, white is
statistically expected to lose; in Vegas the house takes black every time. Highest
priority." A deep Rapfi source audit + our own code audit **confirms it and localizes the
fault** (full evidence: `babysit/rapfi_swap2_research.md`; tracked as #73):

- **Our responder is FINE; our opener is the broken half.** Responder stay/swap/place2 is a
  one-ply value-head comparison (`swap2_search.py:206-225 _choice_values`) — a competent
  responder correctly **swaps to whichever side is better (black)**. But the **opener places
  are sampled from a policy head that has NO fairness gradient** (`swap2_search.py:133-170`):
  it can't learn to compose a *balanced* opening, so it keeps offering openings black wants
  → responder always takes black → stuck ~69/27. **No protocol bug exists** (counts,
  side-to-move, PLACE2 color, `backup_sign` all correct). The choice head (v2a) is a real
  lever but **not the cause** (it's off *and* never wired into selection anyway).
- **How Rapfi stays fair:** it decides the swap **purely by eval-sign** — full search, swap
  iff the assigned side's root value is negative (`opening.cpp:194-213 decideAction`; option
  (c) place-2 is dead code there — stay-vs-swap suffices). The strong evaluator grabbing the
  better color IS the fairness mechanism. As opener it offers **9 hand-curated balanced 3-stone
  openings** (`opening.cpp:81-91`) gated behind a hardcoded `board.size()==15` check — but the
  openings themselves are **tiny local patterns** (each 3-stone cluster has a footprint ≤ 3×6;
  most are 4×4), just pinned at varied absolute board locations (7 of 9 fit within coords ≤8;
  two sit near the far corner). So "15×15-only" is an artifact of hardcoded coordinates, NOT a
  board-spanning opening — a swap2 opening is just a small balanced cluster. A general
  `OpeningGenerator` (balanceWindow=50) mints such balanced openings at any size by balance-search. **No play-time opening book** ships
  (the `[database]` is an off-by-default result cache); our local `engines/rapfi/` has none.
  Rapfi is GPLv3 → don't import their book; **re-implement the balance-search idea over our
  own evaluator** (board-size-agnostic, no license/eval-mismatch entanglement).

**The fix, ranked (proposal — not yet implemented; Jason's call on go):**
- **TRY FIRST — Option A: train the OPENER against the responder's swap** (the project's own
  fix #3). Punish the opener when the responder profitably swaps → drives opening swap-value
  → 0, i.e. the true minimax **50/50**. Only this reaches fairness while preserving learning
  (the responder's already correct). Medium effort; risk = opener-diversity collapse (mitigate
  with an entropy bonus). **Validate:** `opener_color_dist` / black-share → 0.50.
- **Option B (parallel/seed): generate our own balanced opening set** via balance-search over
  *our* value head (board-size agnostic). Reusing Rapfi's 9 triples is a weak fit (15×15-only,
  "balanced for Rapfi ≠ balanced for us").
- **Option C (LAST, only after A): re-enable + wire the choice head (v2a→v2b)** — sharpening
  the responder *before* fixing the opener pushes *away* from 50/50.
- **Option D: protocol fixes — none needed.**

**Why this matters:** reframes the entire white-0% story. We were treating white weakness as a
*training* gap, but white has been playing a **-EV seat** — a rigged game no training can win.
Fairness (≈50/50 w/b) is **upstream of white engagement** and is the highest-priority fix.

## 11. The plan — the fixed-fair-opening LADDER (era-3, #73 fix) ⭐ LIVE 2026-06-21

**The thesis under test (one sentence):** *if white loses because the game is rigged (the opener
can't compose fair openings, §10), then on a KNOWN-FAIR board white should engage and the colors
should trend toward ~50/50.* If white still collapses from fair starts, the weakness is real skill,
not the seat — and that's a different (also valuable) finding.

**The method — sidestep the broken negotiation, and LADDER it for cheap epochs.** Don't make the net
negotiate a fair opening (it can't yet). Hand it fair boards directly and let it just *play* — and
climb the cheap small boards first to bank epochs before the 15×15 "big leagues":
- Every self-play game starts from one of **9 canned fair openings**, placed directly — **no
  negotiation, no opener policy, no choice head** (the net engages only *post-opening*). Construction
  is byte-identical to swap2's `to_normal()` for the 2B+1W → white-to-move outcome.
- **The openers are Rapfi's 9 shapes, RE-CENTERED per board** (Jason's call, 2026-06-21): translate
  each cluster's bbox to board center; footprints ≤3×6 so **all 9 fit on 9/11/13 — zero dropped**
  (his cookie-bet: they're sub-9×9 patterns). `gomoku/self_play.py _FAIR_OPENINGS[N]` selects the
  book by size. Same canned fair openers at every rung; D4 aug fans each into ~72 variants.
- **Fresh / from-scratch at 9×9 → 11 → 13 → 15**, warm-starting up each rung (the global-pool trunk
  carries; the 3 board-bound FCs re-init). aggression `value-discount 0.95`, the e2 recipe otherwise.
- **THE GATE IS BOARD-FILL** (#74, `babysit/ladder_grad.py`; corrected 2026-06-21). When white is
  strong enough it drags black all the way out and **the board fills** — `selfplay/plies_p90` climbs
  toward board capacity. That fill IS the promote signal: it proves white's defense saturated this
  board. Promote the moment it fills (`p90 ≥ FILL_FRAC·cells`, default 0.75, for 5 epochs) — because
  training *past* the fill just rewards dragging-out/retreating (a **dragger, not a killer**), the
  exact bad habit we avoid. **NOT draw rate** (draws only appear *after* p90 fills, so a draw gate is
  always too late) and **NOT a self-relative plateau** (that fires on a LOW plateau — p90 stuck at 18
  on an 81-cell board = black mating fast = white WEAK — the opposite of strong). The grad-check runs
  at the *top* of the rung loop, so a resumed already-filled rung promotes without another drag-out
  slice. **Minimal gating on 9/11/13** (bank cheap epochs, find a fresh killer); CAP=250 backstop.
- **9×9 BLACK-ADVANTAGE PRUNE GATE** (#73, Jason 2026-06-21): *"if 9×9 shows a black advantage, stop,
  drop one of the openers, and run again"* (9×9 trains to ~100 epochs in minutes — a fresh re-run is
  cheap). When 9×9 shows a persistent black edge (mean white-share < 43%) **while the board won't
  fill** — i.e. white gets mated fast, the pathological case, *not* a slight lean at full board — the
  opener set itself is unfair. `babysit/opener_balance.py` plays the net against itself from each
  opener and ranks by black-win%; the orchestrator drops the most black-favoring shape via
  `GOMOKU_DROP_OPENERS` (env-configured, reversible, indices stable across drops; commit `bd099d8`),
  wipes the rung, and re-runs FRESH. Keeps ≥4 openers (Jason: "even 2–4 real openers is fine").
  Board-fill *wins over* prune: when the board fills we promote even with a black lean (it's the
  residual first-player edge, not a rigged opener).
- Why 9 openings are enough to *learn* (not memorize): each is a deep game tree — "if any were an
  insta-win, gomoku would be solved" (Jason). The net has to actually learn to play.

**The run.** Cells `G{9,11,13,15}-fixed-openings` (swap2 OFF, `fixed_openings=True`); `--fixed-openings`
threaded through self-play / worker / trainer (commits `3c6e9d7`, `744849a`, tested at all sizes).
Orchestrator `babysit/fairladder.sh` (15-min slices, board-fill auto-promote, 9×9 prune gate,
warm-start between rungs; `touch babysit/STOP_fairladder` to stop). Rung-9 run `eilfnz1e`.
Card: [[gomoku-15x15-fixed-fair-openings]].

**Status 2026-06-21:** **rung 9 GRADUATED at e83** — it filled the board (`p90=[81,81,81,81,81]`, the
full 81 cells) with draws then spiking 25→59 (Jason's "draws come *after* fill" confirmed); promoted
before the drag-out deepened. White was ~38% of *decisive* games at fill — a black lean, but the
board filled so we promote (residual first-player edge, not a rigged opener → no prune). Warm-started
9→11 (98.9% params transferred). **Now climbing rung 11.** No opener has been pruned.

**What success looks like / what to watch:**
- 🟢 **white-share of decisive self-play games → ~50%** (the headline; on a fair board white should
  stop being the doormat). Contrast every prior era: white ~25–35% (rigged). **First read at 9×9:
  white ~46–51% in the very first epochs.**
- 🟢 plies healthy; auto-promote climbing the ladder fast (Jason expects it won't linger on 9/11/13).
- 🔴 if white-share stays ~25–35% from FAIR starts → the seat wasn't the (only) problem; white has a
  genuine skill gap → white-side teacher (#44).

**The 15×15 gate — the "different era" (the exact recipe, Jason 2026-06-21; TODO to build during the
climb):**
- **Hold all quality judgment until ≥100 REAL 15×15 self-play epochs.** "Real" = epochs with games,
  NOT the empty-epoch startup race (which inflates the counter while workers warm up). Until then the
  run is *just executing the recipe* — no crowning, no quality calls.
- Then a **1-hr cadence:** eval-vs-Rapfi → train 1 h → eval-vs-Rapfi → train 1 h → …
- The eval is **Rapfi-from-the-SAME-canned-openers**: both sides resume from the fixed fair
  post-opening positions — *the exact positions we trained on* — our net as **black AND as white**.
  Fixed openers constrain the search space and give a hard, fair yardstick (vs the OOD-confounded
  standard-opening Rapfi read). Build: extend `babysit/run_eval.py` to seed each game from a canned
  opener (both colors) instead of negotiating; drive Rapfi-NNUE (same `--config` as the swap2 read).

**Phases.** Phase 1 (now): climb the ladder, read white/black balance per rung. Phase 2 (later, gated
on Jason): **expand** — bring a *trained* fair-opening generator / negotiation back (choice head
v2a→v2b, or a balance-search generator, §10 Option A/B) and broaden beyond the 9.

**Relation to era-2 (the swap2 ladder).** That ladder (best net `epoch0235`, 25% vs Rapfi, preserved)
trained the net to *play swap2* on a **rigged** board → white capped ~0–29%. This fair-opening ladder
reuses the cheap-board-curriculum machinery but **removes the rig** to isolate whether fair play
unlocks white. Different question; both kept.
