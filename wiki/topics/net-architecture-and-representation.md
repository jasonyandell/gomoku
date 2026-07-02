# The net's architecture AND its representation design (incl. the recovered "Fable" rationale)

**One-line synthesis.** The gomoku net is a small AlphaZero-style residual conv
trunk with two heads (policy + value); the *interesting* engineering is not the
trunk but the **input-representation levers** (line-potential planes, global
pooling, stem padding) and the **design thesis** behind them — a thesis that was
partly written down by the assistant "Fable" on 2026-07-01 and partly survives
only as flags + code + Jason's memory. This page is both **synthesis** (the
architecture) and a **source record** (the recovered Fable quotes, cited to their
session). Read the honesty box (§5) before treating any of the size-choice
reasoning as authoritative.

Source files: `gomoku/model.py` (the net), `gomoku/features.py` (the line
planes). See also [sound-world-recipe.md](sound-world-recipe.md),
[vct-recognition-learnability.md](vct-recognition-learnability.md),
[the-claw.md](the-claw.md),
[vct-defense-aux-head-result.md](vct-defense-aux-head-result.md),
[idx2-vct-frontier-map.md](idx2-vct-frontier-map.md) (the "Bruce-Lee board").

---

## 1. The architecture — trunk + two heads

`GomokuNet` (`gomoku/model.py`) is a textbook AlphaZero residual tower:

- **Stem** — a 3×3 conv (`stem_padding` wide, see §2) → BatchNorm → activation,
  taking `n_input_planes` (17) channels (+8 when line planes are on, see §2).
- **Residual tower** — `n_blocks` `ResBlock`s, each `conv→bn→act→conv→bn`, then
  `act(x + h)`. **The skip connection `x + h` is why depth is trainable at all**:
  without it, gradients through a deep 3×3-conv stack vanish; the identity path
  lets each block learn a residual correction and keeps the signal alive. Every
  spatial dim is preserved through the tower.
- **Policy head** — a **per-cell 1×1 conv** (`policy_filters=2`) → BatchNorm →
  flatten → `Linear` to `board_size²` logits. The 1×1 conv is the load-bearing
  choice: it reads each cell's channel stack *independently*, so a per-cell
  feature (see §2) becomes a per-cell policy logit with no mixing.
- **Value head** — a 1×1 conv (`value_filters=1`) → BN → flatten → `Linear` to
  `value_hidden` → `Linear(hidden,1)` → `tanh`, a scalar in [-1,1]. (Derby levers
  can swap the final FC for WDL / HL-Gauss categorical heads; the scalar the rest
  of the codebase consumes is unchanged — see the `value_head` field.)

Auxiliary heads (opponent-reply, ownership, **VCT-defense "blunder map"**) and a
swap2 choice head tap the *same shared trunk* and are constructed only when their
config flag is set — off ⇒ byte-identical state_dict / param count / inference
graph. The VCT-defense head is the one that mattered in the #103 experiment (§4).

### Size, width², and the epoch-time gap

Two presets are in play (`SIZE_PRESETS`):

| preset | width×depth | params (measured) | role |
|---|---|---|---|
| **small** | 64 filters × 4 blocks | **345,885** (25-ch stem = 17+8 line planes; matches the 9×9 sound-world run) | the from-scratch 9×9 / 13×13 sound-world net |
| **large ("Bruce")** | 128 × 10 | **~3.05M** (~3.2M with line-planes+global-pool) | the 15×15 champion |

Params (and compute) scale as **≈ width² × depth**: the 64×4 → 128×10 jump is a
theoretical ~10× on the tower alone, and the measured **~14× epoch-time gap**
(large train ~44 s/epoch → small ~2.7 s) confirms **epoch time is net-size /
TRAIN-bound, not gen-loop-bound** (TRAINING_WIKI 2026-07-02, line 5797: the
"14× faster epoch" overnight win was *entirely* net size, not a gen-path
overhaul). The width-squared term is what makes the small net so much cheaper —
and, per Fable's stated rationale, so much *faster to iterate on* (§3).

---

## 2. The representation levers — what each one does and why

These are the design surface that distinguishes this net from a vanilla AZ net.
All three are opt-in flags; all three are byte-identical when off.

**`line_planes` (issue #107) — an INPUT-representation lever, not a head.**
When on, `forward()` derives **8 extra channels** in-model
(`gomoku.features.line_potential_planes`, `N_LINE_PLANES = 2×4 = 8`): per-cell ×
4-direction {H,V,diag,anti} × {me, opp}, each = the max over live 5-windows
through the cell of (stone count / 4). Semantics: an empty cell with
`feat[me,d]==1.0` completes five there; `feat[opp,d]==1.0` is a win-in-one to
block; **a double threat reads as TWO direction channels hot at the SAME cell** —
a purely *local* conjunction a 1×1 conv can see, instead of a long-range relation
between two 5-cell spans. Computed inside the model (a pure function of the two
current stone planes) so the external 17-plane contract — records, replay buffer,
D4 augmentation, native paths — is untouched and augmentation-consistency is
automatic. Because it changes the stem conv's `in_channels`, it *cannot* be
toggled on an existing checkpoint without a splice. This is the "sound-world"
input; the claw work ([the-claw.md](the-claw.md)) is *why* it exists — line
planes make cross-line structure LOCAL (see Fable, §3), though the claw itself
(knight's-move, spacing 5) remains invisible even to *these* line-organized
channels (§4, §5).

**`global_pool` (Derby v4 "Whole-board" lever).** When on, the latter half of
the residual blocks become `GlobalPoolResBlock`s that inject a board-global
signal — `[mean_HW(h); max_HW(h)]` → `FC` → per-channel bias broadcast to every
cell. Its rationale comment is the KataGo receptive-field argument verbatim:
*"a tiny 3×3-conv tower's receptive field barely spans 9×9, so it cannot cheaply
represent board-global facts ('is there a live-four ANYWHERE?'); global pooling
injects that context directly."* An int sets how many trailing blocks get it.

**`stem_padding`.** Default 3 (from michaelnny/alpha_zero's gomoku fix for
"fails to block on edge cases"): a 3×3 stem conv with padding 3 expands the
feature map from `board_size` to `board_size+4`, a "virtual padding zone" around
the real board so edge tactics aren't cramped. The sound-world launch used
`--stem-padding 1` (post-stem spatial = board_size). Post-stem spatial size, not
`board_size`, sizes the head FCs.

**Board-size guard.** Every head/output dim derives from `cfg.board_size`, and
`load_checkpoint` refuses a checkpoint whose board size differs from the active
process (pre-field checkpoints load as 9×9). This is what let the 9×9 tower
warm-start into a 13×13 net (tower transfers; the flattened-board FC heads reinit
— TRAINING_WIKI 2026-07-02, #113 slice 1).

---

## 3. The recovered Fable rationale (SOURCE record)

> **Citation.** The following are direct quotes from the assistant **claude-fable-5**,
> session `c67c873f-3e99-4c59-a929-ace6d9285fc6.jsonl`, **2026-07-01**. They are the
> only prose record of the sound-world net's design intent. **Fable's THINKING
> blocks are redacted/empty in the logs** — see the honesty box (§5) for what this
> means for the *size* reasoning specifically.

**Line planes — why (make cross-line structure LOCAL):**
> "It's not just 'lines, not positions' — it's making cross-line structure LOCAL.
> Per-cell, per-direction 'how close to five along this line, or is it dead'
> channels. A double open three — the thing the claw work proved is invisible to
> stone-planes — becomes two channels hot at the same cell, readable by a 1×1
> conv. You're not asking the net to learn long-range relational geometry; you're
> handing it the geometry and letting it learn the chemistry."

**Rapfi lineage (proudly parroting the ancestors):**
> "Classic gomoku evaluators (and Rapfi's own NNUE features) have always been
> built on line-shape tables, so yes, we're parroting the ancestors, proudly."

**Representation exoneration (it was never the representation's fault):**
> "the aux head learned the VCT-blunder map from the existing 17 raw-stone
> planes… representation is exonerated — same as capacity was."

**The falsifiable bet (a self-play death-tell):**
> "Watch plies_mean in the first hour: it structurally cannot sit at 9-10 in a
> sound world."

**The launch command** — the *only* record of the size decision — was
`--line-planes --stem-padding 1 --global-pool …` with **NO size preset**, so it
took the repo **default `small` (64×4)**. In the prose, the from-scratch 9×9
small net is justified **only by iteration speed** ("3-second laps, fastest
giggles") — i.e. the width²×depth epoch-time win of §1, not by any explicit
representation-test argument.

---

## 4. The NORTH STAR — the hybrid / Rapfi thesis

Fable's stated dream target (the design's north star):
> ">50% vs full Rapfi using WAY more compute" — "the hybrid player isn't a
> compromise, it's the design."

This is the load-bearing idea and it ties the whole representation program
together:

- **Seek-VCT division of labor.** The net steers only in the *tactically-quiet*
  region (approximation-tolerant); the exact GPU VCT/VCF **oracle** handles the
  forcing finish (approximation-*in*tolerant but tractable to solve). Recognition
  is COUNT-DOMINATED and cheap — logreg-on-threat-counts (0.946 AUROC) beats
  attention, and a CNN beats attention at half the params
  ([vct-recognition-learnability.md](vct-recognition-learnability.md)) — so
  recognition is left to the oracle, and the net's job is *quiet-phase steering*.
- **Line planes = handing the net Rapfi's own substrate.** "Proudly parroting the
  ancestors": line-shape features are exactly what classical evaluators and
  Rapfi's mix9svq NNUE are built on. The net gets the geometry for free and
  learns the "chemistry."
- **The claw / molecules are the known blind spot.** The line-organized
  representation is provably *incomplete*: the knight's-move crystal
  (`2x+y≡0 mod 5`, [the-claw.md](the-claw.md)) is invisible to any line-keyed
  eval — and, crucially, to vanilla local convolution too (it needs a periodic
  positional encoding). So the hybrid thesis is honest about where it's blind: the
  offense lives on lines (which line planes serve); the defense axis (§the-claw)
  may need periodic/modular channels the current levers don't provide.
- **The product shape.** The sound-world 9×9 chapter closed (TRAINING_WIKI
  2026-07-02) with exactly this verdict: **net + oracle finisher** (95% vs
  heuristic where the bare net draws) — "bare-net drawishness is division of
  labor, not weakness." The hybrid is the deliverable, not a fallback.

Where it stands: the VCT-defense aux head is a **working sensor with no
actuator** ([vct-defense-aux-head-result.md](vct-defense-aux-head-result.md),
#103) — the net *learns* the blunder map from the existing planes (Fable's
"representation is exonerated") but nothing yet makes the *policy* act on it;
self-play has no opponent to punish weak defense. Fable's `plies_mean` bet is the
death-tell for that same attractor.

---

## 5. RECOVERED vs LOST — the honesty box

> **What survives as PROSE (quotable, cited §3):** the line-planes "make
> cross-line structure local" rationale; the Rapfi-lineage "parroting the
> ancestors" framing; the representation-exoneration claim; the hybrid/">50% vs
> Rapfi with more compute" north star; the `plies_mean` falsifiable bet; and the
> literal launch command (`--line-planes --stem-padding 1 --global-pool`, no size
> preset → default small).
>
> **What is LOST:** **Fable's THINKING blocks are redacted/empty in the logs.**
> So the **small-net + global-pool reasoning** — the KataGo receptive-field
> argument that lives as a code comment in `model.py` — is **NOT** in Fable's
> prose. The prose justifies the small net *only* by iteration speed ("3-second
> laps"). The size choice was, on the written evidence, **the repo default
> preset**, not a stated deliberate decision.
>
> **Therefore, do not over-claim:** the tidy framing "the small net was a
> deliberate TEST of the representation" is a **plausible later reconstruction**,
> not something Fable said. What Fable *actually* said is that the small net was
> the fast-iteration default. Jason's own predictions, checked against the log:
> his guess that the small net "was an artifact of rethinking representation, not
> a deliberate size choice" is **BORNE OUT**; his guess that "Fable's big idea is
> hybrid + search in Rapfi's direction" is **CONFIRMED** by the §3–§4 quotes.
>
> **Numbers, verified today:** the small net is **345,885 params** (measured;
> logged in TRAINING_WIKI 2026-07-02 line 5534 as the from-scratch 9×9 run with a
> "25-ch stem"), *not* the ~396K a rougher recollection suggested. The large
> "Bruce" net is **~3.05M** (128×10). The ~14× epoch gap is net-size/train-bound
> (line 5797).
