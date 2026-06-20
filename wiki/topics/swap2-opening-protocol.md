# Swap2 opening protocol (#72) — the real fix for white

**Status (2026-06-20):** built end-to-end and LIVE. The core bet is **confirmed at the
data level** (swap2 self-play makes white winnable: white wins 27% vs ~0% on an empty
board), and the **strength** signal is strong and HOLDING: across independent n=128
checkpoints the H2H-vs-frozen-champion overall win-rate sits at a steady **~70% level**
(57.0% e235 → 66.8% e289 → 76.2% e345 → 67.6% e403), comfortably above the ~58%
relative-crown lower bound (§6.6). On the white side the **white LOSS-rate fell from 88%
(e129) to ~parity and is now fluctuating around it (42% e345, 57% e403)** — genuinely
improved off the catastrophic floor, but bouncing around parity, **not monotonically
marching to 0**. **Correction to the prior gate-6 framing:** the e345 76.2%/42% reading was
*partly an upward sample-fluctuation* (same flavor as the earlier e181 n=64 spike), not a
"slope steepening" — the e403 pullback to 67.6% confirms the true level is ~70%, and the
CIs overlap (it's noise, not a regression). The exact metric the project chased for months
is moving — via balanced data, not a teacher. Verdict at ~403 epochs: **NOT a plateau, NOT
a regression, level holding well above the crown bar — keep training; progress on the hard
white side is noisy around parity, not a smooth descent** (formal crown wants an n≥200 gate
to call it). The result is the *trend across independent checkpoints and their CIs* (n=128
gates; the lone n=64 e181 read was noise), not any single gate — a single high gate can be
an upward fluctuation, as e345 partly was. This page is the durable synthesis: why we did
it, what we built, what we learned, and what to try next. Evidence chronology lives in
`TRAINING_WIKI.md` (2026-06-20 entries); the predecessor analysis is
[white-side-defense-plan.md](white-side-defense-plan.md).

Branch: `feat/swap2-opening-protocol` (worktree, **unmerged** by request). Run: cell
`G15-swap2`, wandb `8nq1a7cm`, board 15, home `/Users/jason/data/swap2/`.

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

## 6. What to try next (ranked)

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
