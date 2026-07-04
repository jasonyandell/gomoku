# Swap2 opening-protocol — superseded sections (archived)

**Status: SUPERSEDED / ARCHIVE — full-fidelity verbatim, 2026-07-04.** These are the era-1
"what to try next" menu (§6), the 9×9-vetting / Modal-scale-era discussion (§6.5), and the
"if it stops working" contingency decision-tree (§6.7), moved verbatim out of
[swap2-opening-protocol.md](../../topics/swap2-opening-protocol.md) during the 2026-07-04
curation. §6 was already marked "SUPERSEDED by §9"; §6.5/§6.7 are planning scaffolding for a
run that has since concluded. The parent page keeps compressed summaries + the durable
one-liners (9×9 can't carry the white-defense signal; the crowning bar in §6.6 stays live on
the parent). No facts were changed — only relocated.

---

## §6 — What to try next (ranked) [superseded by §9]


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


## §6.7 — Contingency plan: "if it stops working" (decision tree) [superseded]


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

