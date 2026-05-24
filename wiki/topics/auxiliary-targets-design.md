# Auxiliary Training Targets (KataGo-style) — Design

**Status:** DESIGN-ONLY. No code written. Class C (model-architecture change)
— requires Jason's sign-off before any `.py` edit. Branch
`feat/v3-auxhead-design`, not merged.

**One-line:** Add a *droppable* auxiliary prediction head to squeeze more
gradient signal out of each scarce self-play position. Recommended first cut:
an **opponent-reply policy head**, because its target is already sitting in the
self-play trajectory and the arch add is tiny.

---

## 1. Motivation — tied to our own findings

KataGo's largest sample-efficiency wins came not from a bigger net or more
search, but from **auxiliary prediction heads** (ownership, score distribution,
opponent-reply, short-term value). Each head turns one position into several
supervised learning problems, so the shared tower gets many gradient signals per
forward pass instead of just policy + scalar value.

That is the *exact* countermeasure to the pain documented in
[az-at-scale-vs-laptop.md](az-at-scale-vs-laptop.md) § "Game length and per-game
signal density":

> Go games run 200-250 moves. 9×9 freestyle gomoku games at strong play often
> resolve in 12-50 moves; aggressive-attack regimes can drop below 5 moves. […]
> A decisive 5-move game has 5 positions of training signal, each an opening or
> near-opening, no mid-game strategic content, no endgame technique.

Our positions are **few and thin**. We already do the cheap multipliers
(D4 symmetry augmentation, 8× per position; per-side/per-ply diagnostics). The
remaining lever is *more supervised targets per position* — which is precisely
what an auxiliary head is. Where Go gets signal density from game length, we can
get it from prediction breadth.

The derby ([ops/research-board.md](../ops/research-board.md)) already showed the
top levers are **exploration/diversity**, not raw compute (open-div4 1385 and
temp-16 1340 beat sims-400 and sgd-800). Auxiliary targets are a *different axis*
from those: they don't change what self-play we generate, they change how much
the net learns from what we already generated. That makes a clean derby-v3 lever
— one flag, orthogonal to the v1/v2 exploration levers.

---

## 2. The three candidates

### Candidate 1 (RECOMMENDED) — Opponent-reply policy head

Predict the **opponent's MCTS policy at the very next ply** (`pi` of the
position that immediately follows in the same game). A second 81-way policy head
off the shared tower; the target is the next trajectory entry's `pi`.

- **Data cost:** ~zero. The next ply's `pi` is *already computed and already
  recorded* in `self_play.py`'s per-game `trajectories[g_idx]` list (an ordered
  list of `(planes, pi, side)`). We only need to also store "the next entry's
  `pi`" alongside each example at record-build time. No extra MCTS, no extra
  evaluator calls, no label-definition ambiguity.
- **Arch cost:** tiny — clone the existing policy head (1×1 conv → BN → flatten →
  Linear(., 81)). Identical shape, identical fuse path.
- **Signal:** teaches the tower a one-ply forward model of strong play
  ("what will a competent opponent do to me?"). In gomoku that is *threat
  awareness* — the opponent's reply to my move is exactly the block/counter I
  must anticipate. Highly relevant to our documented weakness (the model learns
  to attack but is slow to learn to defend; see
  [feedback_gomoku_threat_semantics] memory and the WL-series "learn to defend"
  thesis). This is the lowest-risk, highest-information-per-line-of-code first
  cut.

### Candidate 2 — Short-horizon / multi-value head

Predict the game value at **+2 / +4 plies ahead** (or a win-distance bucket)
in addition to the final outcome `z`. Richer credit assignment: the net learns
"am I winning *soon*" vs "am I winning *eventually*", which is sharper in short
games where the final outcome is only a few plies away.

- **Data cost:** low but non-trivial — needs the outcome-at-horizon, derivable
  from the trajectory but with edge cases (games shorter than the horizon, sign
  flips per side-to-move). Win-distance needs the terminal ply recorded per
  example.
- **Arch cost:** small — extra scalar(s) off the value head, or a small
  distributional head.
- **Why not first:** more target-definition surface than C1 (horizon clamping,
  per-side sign bookkeeping), and the scalar-value signal it adds is *correlated*
  with the existing `z` target — less novel gradient than C1's policy signal.
  Strong second experiment.

### Candidate 3 — Threat-map / "ownership" analog

Per-cell prediction (81 outputs) of which side ends up controlling each cell, or
where the winning 5-in-a-row line lands. The closest direct analog to KataGo's
ownership head; biggest *spatial* signal.

- **Data cost:** highest — requires **defining a per-cell label**. Gomoku has no
  natural "ownership" (stones don't get captured/surrounded like Go), so we'd
  invent a label: e.g. the cells of the winning line = winner's side, occupied
  cells = their owner, empty non-line cells = neutral/0. That's a real design
  decision with no obvious-correct answer, and it needs new bookkeeping in
  self-play to find and record the winning line.
- **Arch cost:** moderate — a per-cell conv head (1×1 conv → 81-way or 3-way
  per-cell logits), spatial output.
- **Why not first:** the label is the whole research question. Worth doing, but
  only after C1 proves the aux-head plumbing works end-to-end. Filing it as the
  ambitious third experiment.

**Recommendation: lead with Candidate 1 (opponent-reply policy).** It is the
least-invasive (a head clone + one recorded field), the target is *free* (already
in the trajectory), it has no label-definition ambiguity, and its signal —
anticipating the opponent's reply — directly attacks our documented
defense-learning weakness. C2 and C3 are queued follow-ups that reuse the same
plumbing this design establishes.

---

## 3. Recommended head — exact design (opponent-reply policy)

### 3.1 Architecture change (`model.py`)

The current tower output `h = self.tower(self.stem(x))` already feeds both
existing heads. Add a third head that taps the **same `h`**:

```
# in GomokuNet.__init__, gated on cfg.aux_opponent_reply:
self.aux_policy_conv = nn.Conv2d(c, cfg.policy_filters, 1, bias=False)
self.aux_policy_bn   = nn.BatchNorm2d(cfg.policy_filters)
self.aux_policy_fc   = nn.Linear(cfg.policy_filters * spatial * spatial, N_ACTIONS)
```

- A new `ModelConfig` field `aux_opponent_reply: bool = False`. When `False`,
  the head modules are **not constructed**, so `state_dict()` is byte-identical
  to today and the param count is unchanged. This is the backward-compat anchor
  (see § 4).
- `forward` returns `(policy, value)` as today when the head is off. When on, it
  returns `(policy, value, aux_policy)` — OR, to avoid touching every caller, the
  aux logits are exposed via a separate method `forward_aux(x)` /
  `model.aux_policy_logits` attribute used only by the trainer. Recommended: a
  `return_aux: bool = False` kwarg on `forward`, defaulting off, so self-play and
  eval (which call `forward(x)`) are untouched and only the trainer passes
  `return_aux=True`. **Decide this with Jason** (open question Q1).
- The head is a structural clone of the existing policy head — same fuse
  treatment.

### 3.2 `fuse_model_for_inference` — droppable at inference

This is the load-bearing property: **the aux head must cost nothing in self-play
and eval.** Two-part guarantee:

1. The head is gated off (`aux_opponent_reply=False`) for any model that will run
   inference-only — so it is never even constructed in the eval/self-play path.
2. For a trained model that *has* the head, `fuse_model_for_inference` simply
   does not touch the aux modules (it fuses stem, tower, policy, value as today).
   Since `forward(x)` with `return_aux=False` never executes the aux head, the
   aux conv/bn/fc are dead weight at inference — zero FLOPs. Optionally add a
   one-liner that `del`s the aux modules after load for a truly identical
   inference graph. The aux head contributes **zero** to MCTS evaluator latency
   regardless.

Net: the aux head is a *training-only* appendage. Self-play throughput (the
generation-bound bottleneck per the derby's finding #5) is unaffected.

### 3.3 Target data source + `self_play.py` changes

The target is **the next ply's recorded MCTS policy** in the same game. It is
already computed — `trajectories[g_idx]` is an ordered list of `(planes, pi,
side)` (native path: `self_play.py` ~L266; pure path ~L429). Concrete changes:

1. **`SelfPlayExample`** (dataclass ~L46-58): add
   `aux_pi: np.ndarray | None = None` — the *opponent's* `pi` at ply+1. Optional
   field defaulting `None` keeps every existing construction site valid.
2. **Record-build loop** (native ~L290-318; pure ~L457-475): when iterating
   `enumerate(trajectories[g_idx])`, the next entry `trajectories[g_idx][ply_idx
   + 1]` supplies the opponent's `pi`. For the **last** ply of a game (terminal —
   no next move) set `aux_pi = None` and mask it out of the loss (see § 3.4).
   Under D4 augmentation, the aux `pi` must be transformed by the **same**
   symmetry as the example's own `pi` (reuse `augment`'s pi-permutation on the
   aux target — they share the board geometry).
3. **`ReplayBuffer`** (`replay_buffer.py`): add an `aux_pi` tensor
   `(capacity, N_ACTIONS)` plus an `aux_mask` tensor `(capacity,)` bool (False =
   no next ply / aux undefined). Wire through `add`, `sample` (return a 7-tuple),
   `state_dict`/`load_state_dict` (tolerate-missing on old checkpoints, zero-fill
   + mask-false, mirroring the existing `side`/`ply` missing-tag handling at
   L142-153). When the lever is OFF, the trainer never reads these tensors;
   allocate them only when the flag is set to keep buffer RAM unchanged in the
   off case (open question Q2 — lazy alloc vs always-alloc).

### 3.4 Loss term + default weight (`train.py`)

`train_step` currently computes `loss = pl + value_weight * vl` (L65). Add:

```
# only when aux head is active and aux_pi/aux_mask provided:
aux_logits = model.forward_aux(...)            # or the return_aux path
aux_logp   = F.log_softmax(aux_logits, dim=-1)
per_aux_ce = -(aux_pi * aux_logp).sum(dim=-1)  # soft-target CE, same form as policy_loss
aux_l      = per_aux_ce[aux_mask].mean()       # masked: skip terminal-ply rows
loss       = pl + value_weight * vl + aux_weight * aux_l
```

- **Default `aux_weight = 0.0` → byte-identical to today** (term vanishes,
  no aux forward executed). Suggested *on* value: **0.15** (KataGo uses small
  aux weights ~0.15 for opponent-reply-class heads; policy CE and aux CE are the
  same scale, so it's a fraction of the policy term — start conservative, sweep
  {0.05, 0.15, 0.3} as a future board). Log `loss/aux_policy` alongside
  `loss/policy` and `loss/value`.
- Mask handling: rows with no next ply (`aux_mask == False`) are excluded from
  the mean so terminal positions don't inject a garbage target.

---

## 4. Backward-compat / migration

- **Derby v3 needs no migration.** The derby starts every idea from a *fresh*
  `--size small --seed 0` init ([ops/research-board.md](../ops/research-board.md)
  § Rules), so there is no checkpoint to migrate — the aux-on idea simply builds
  a net with the head from epoch 0. Note this explicitly: the head is a
  fresh-start lever, not a retrofit.
- **Existing checkpoints stay loadable.** `aux_opponent_reply` defaults `False`
  and `load_checkpoint` already uses `setdefault` for new config keys (the
  `stem_padding` precedent at `model.py` L169-170). An old checkpoint loads with
  the head absent — no shape mismatch, identical to today.
- **Buffer checkpoints** follow the same tolerate-missing pattern already in
  `ReplayBuffer.load_state_dict` (`side`/`ply` zero-fill, L142-153): an old
  buffer loads with `aux_mask` all-False, so its positions simply contribute no
  aux gradient until they evict.

---

## 5. Derby-v3 lever spec

**Flag:** `--aux-opponent-reply-weight FLOAT`, **default `0.0` (off)**.

- A single float that does double duty: it both **gates the head** (constructed
  iff weight > 0 at build time, via `ModelConfig.aux_opponent_reply = weight >
  0`) and **scales the loss term**. One flag, one lever — matches the derby's
  "one lever each, clean attribution" rule.
- Suggested derby-v3 idea: `aux-reply-015` = `--aux-opponent-reply-weight 0.15`,
  config delta vs C0 = exactly that one flag.
- **Byte-identical-when-off guarantee.** With the default `0.0`:
  - `ModelConfig.aux_opponent_reply = False` → aux modules not constructed →
    `state_dict()` and param count identical to current `main`.
  - `forward(x)` runs the `return_aux=False` path → zero aux FLOPs in self-play
    and eval.
  - `train_step` adds `aux_weight * aux_l` with `aux_weight == 0.0` and skips the
    aux forward entirely → the SGD graph is unchanged.
  - `ReplayBuffer` skips aux tensor allocation (lazy alloc) → buffer RAM
    unchanged.
  - Verification: a `tests/` check that `build_model("small")` with the flag off
    produces a `state_dict` with identical keys/shapes to `main`, and that a
    training step's `loss/total` matches a no-aux reference within float noise.
    (Test is Class A — can be written freely once code is greenlit.)

---

## 6. Expected Δelo signature (derby title card style)

### aux-reply-015
**Lever:** `--aux-opponent-reply-weight 0.15` (vs 0.0) — second 81-way head
predicting the opponent's next-ply MCTS policy; KataGo-style auxiliary target,
dropped at inference.

**Hypothesis:** Our self-play positions are *few and thin*
([[az-at-scale-vs-laptop]] item 2): short games yield few, near-opening
positions, so the tower is starved of gradient signal per position. An
opponent-reply head adds a free, dense supervised target to every non-terminal
position — the net learns a one-ply forward model of strong play, which in
gomoku *is* threat anticipation (predicting the opponent's block/counter). If
signal-per-position is a binding constraint on the fresh-start climb, the aux
gradient should steepen Δelo without changing what self-play we generate — an
axis orthogonal to the v1 exploration levers. The risk: the aux term steals
capacity/gradient from the primary policy+value objective at small `small`-size,
slowing rather than helping (the classic aux-task-interference failure).

**Expected Δelo signature:** *Confirm* = Δelo per chunk at or above C0 with the
gap opening in the mid-climb (epochs 40–100), where richer per-position signal
should compound; equal-or-higher final elo at 140, and a lower `loss/value` /
`val/value_mse` (the shared tower learned better features). The aux head costs
**nothing** at eval (dropped), so any Δelo/hr gain is pure. *Refute* = Δelo
tracking C0 within noise (signal density wasn't the binding constraint) or *below*
C0 (aux interference at small size) — in which case retry at lower weight
(0.05) or defer to a larger size where capacity isn't the bottleneck.

**Config delta vs C0:** `--n-simulations 200 --aux-opponent-reply-weight 0.15`.

---

## 7. Implementation surface + risk (why Class C)

**Files that change** (when greenlit — none touched in this design):

| File | Change | Est. lines |
|---|---|---|
| `gomoku/model.py` | `ModelConfig.aux_opponent_reply` field; aux head modules (conv/bn/fc); `forward(return_aux=...)`; leave `fuse_model_for_inference` untouched (head not in inference path) | ~25–35 |
| `gomoku/self_play.py` | `SelfPlayExample.aux_pi` field; record next-ply `pi` in both record-build loops (native + pure); D4-transform the aux target | ~20–30 |
| `gomoku/replay_buffer.py` | `aux_pi` + `aux_mask` tensors; wire `add`/`sample`/`state_dict`/`load_state_dict`; tolerate-missing on old buffers; lazy alloc when off | ~25–35 |
| `gomoku/train.py` | `--aux-opponent-reply-weight` flag; aux CE term in `train_step` (masked); `loss/aux_policy` logging; pass-through plumbing | ~20–30 |
| `tests/` (Class A) | byte-identical-when-off test; aux-target alignment test (next-ply pi under D4); masked-loss test | ~60–90 |

**Total core surface:** ~90–130 lines across 4 modules + tests. Modest, but it
**touches `model.py` (the network architecture)** and the data contract between
self-play → buffer → trainer.

**Why this is Class C (not just a big Class-A diff):** per
[conventions.md](conventions.md) § Risk classes, *"model architecture changes"*
are explicitly Class C — *"Discuss before starting."* Code size is not the gate
(reversibility is); a new prediction head changes the net's structure, the
checkpoint contract, and the self-play→buffer data schema simultaneously. That
trio is the kind of architectural change Jason asked to sign off on before any
`.py` edit. Hence: **design-only deliverable, no merge, await approval.**

---

## 8. Open questions for Jason

- **Q1 — `forward` signature.** Add a `return_aux: bool = False` kwarg to
  `forward` (keeps self-play/eval callers byte-identical, trainer opts in), OR a
  separate `forward_aux(x)` method, OR always-return-3-tuple (touches every
  caller). Recommend `return_aux` kwarg. Your call.
- **Q2 — buffer alloc when off.** Lazy-allocate the `aux_pi`/`aux_mask` tensors
  only when the lever is on (keeps off-case RAM identical), vs always-allocate
  (simpler code, +~100k×81 floats ≈ 32 MB at 100k buffer). Recommend lazy.
- **Q3 — derby slot.** Is aux-reply a **v3 idea** (its own board, since it's a
  new axis from the v1/v2 exploration levers), or a **v2 add-on** to the
  head-to-head re-run of the top-3? Recommend its own v3 board — it's orthogonal
  and deserves a clean control.
- **Q4 — weight default-on value.** 0.15 to start, sweep {0.05, 0.15, 0.3}
  later? Or a different starting point given `small` is capacity-constrained?
