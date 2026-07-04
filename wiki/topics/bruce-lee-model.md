# The "Bruce Lee" model — the 15×15 single-opener champion

> **Status: LIVE** *(2026-07-04)* — the current 15×15 champion subject page.

> **What it is.** *Bruce* (a.k.a. Bruce-1) is the 15×15 free-style AlphaZero
> champion of the **swap2 / single-opener era**: a **128×10** residual net,
> self-play-only, trained from **one** fair opening (idx-2) and *nothing else*.
> It is the current **baseline-to-beat** — and its ~50 Δelo plateau, with a
> single measured wound (white defense), is the reason for most of the
> experiments below.

## 1. The name — "one kick 10,000 times"

Jason's framing, the whole thesis in one line:

> "I fear not the man who has practiced 10,000 kicks once, but the man who has
> practiced **one kick 10,000 times**."

The 15×15 ladder had been training from many openings and could not hold
color-balance: re-centering Rapfi's shapes tested **0–95% black** at 13×13 (and
the *most*-central openers were the *most* black-favored — killing the "centered
is fairest" guess). The one opening that came out ~50/50 fair was **idx-2**. So
the call (2026-06-22) was **"drop anything that isn't fair first"**: restrict
self-play to idx-2 via `GOMOKU_DROP_OPENERS`, and **deliberately over-specialize**
— a strong specialist from one fair opening beats an imbalanced generalist from
many. Over-specialization is the *point*, not a risk: at eval the only position
that ever occurs is idx-2, so every parameter spent elsewhere is waste. Depth on
one position, not coverage. See
[swap2-opening-protocol.md](swap2-opening-protocol.md) §12 and
[training-run-lineage.md](training-run-lineage.md) (Current Read).

## 2. What Bruce is — arch, recipe, run, plateau

- **Architecture:** `large` net, **128 channels × 10 blocks** (`128×10`), the
  swap2-era checkpoint schema (carries a `choice_head` field for the swap2
  opener/responder role).
- **Recipe (`G15-fixed-openings` cell):** 15×15 board, self-play **only** from
  idx-2 (D4 symmetries recovered by the trainer's sample-time augment), **1M
  packed FIFO buffer**, **recency-frac 0.5** (the #17 curator — half of each
  batch from the most-recent 200k so the training distribution tracks the live
  policy; a big flat buffer without recency weighting goes stationary and the
  loss dies), 9 fair openings collapsed to the single fair opener, **3 self-play
  workers** (buffer-balance knob — target reuse ~1–4; reuse <1 is gen-flood
  self-sabotage), gumbel-root m16, value-discount 0.95, 64 SGD-steps/epoch.
- **Run(s) — two distinct wandb runs sit behind the "Bruce" checkpoints**
  (provenance verified 2026-07-02 by loading each `.pt`'s `epoch` /
  `wandb_run_id` / `model_config`):
  - **`gogpmbhw`** (`G15-fixed-openings-board15`) — the canonical **live-Bruce**
    training run: 128×10 15×15 **with `choice_head`** (the swap2 schema), the
    8-worker overnight e224→~616 and onward. Its latest babysit snapshot
    `~/data/swap2/babysit/snapshots/g15_e2659_0623_2025.pt` loads as **epoch
    2659, `wandb_run_id: gogpmbhw`, `choice_head: True`** (verified).
  - **`zrjfwny2`** (cell `G15-128x10-bigbuf`, the "**128×10 bigbuf**" eval
    ladder) — a **separate** wandb run, 128×10 15×15 **without `choice_head`**.
    It produced the frozen internal-strength-ladder nets `…_eval146` (epoch 100),
    `…_eval248` (e200), `…_eval348` (e300), **`…_eval502` (epoch 500)**, and
    **`…_e588_best` (epoch 605, the "best" checkpoint)** — every one loads with
    **`wandb_run_id: zrjfwny2`** and no `choice_head` (verified).
- **The three files this page cites, disambiguated (all verified by torch-load):**
  - `g15_e2659_0623_2025.pt` = **epoch 2659, run `gogpmbhw`** — the live-Bruce
    babysit snapshot (swap2 schema, `choice_head:True`).
  - `g15_128x10_bigbuf_e588_best.pt` = **epoch 605 (best), run `zrjfwny2`** — the
    #103 warm-start base.
  - `g15_128x10_bigbuf_eval502.pt` = **epoch 500, run `zrjfwny2`** — the
    white-wound measurement net (§4a).
  - So **`eval502` and `e588_best` are the SAME run (`zrjfwny2`) at different
    epochs** (e500 vs the e605 "best") — one eval ladder, not two runs. Neither is
    the same run as the `e2659` snapshot (`gogpmbhw`); the two lineages also
    differ in architecture (`zrjfwny2` carries **no** `choice_head`). The file's
    eval-index (146/248/348/502) does **not** equal its stored `epoch`
    (100/200/300/500) — the index is offset from the internal epoch counter
    (offset *reason* unverified; the `epoch` values themselves are verified).
  - Reconciling "#103 pivot warm-started *from* `e588_best`" (§4d) with
    `e588_best` already carrying `wandb_run_id: zrjfwny2`: the pivot **resumed the
    same `zrjfwny2` run** from its own ~e605/e613 checkpoint (consistent with
    [white-side-defense-plan.md](white-side-defense-plan.md) "resumed e585 →
    e1286" and `TRAINING_WIKI.md` `G15-128x10-bigbuf` 501→513), rather than
    starting a fresh run — so `zrjfwny2` names the *whole* bigbuf lineage, of
    which the aux-head pivot (§4d, Experiment B) is a later resume segment. *(The
    resume interpretation is inferred from the shared run-id plus the wiki's
    resume notes; the run-id sharing is verified, the resume mechanism is not
    independently re-verified here.)*
  - **Unverified / open:** whether the `zrjfwny2` bigbuf net is itself a
    warm-start descendant of the `gogpmbhw` Bruce (or an independent 128×10 run)
    is **not** established by the checkpoint metadata — the differing arch
    (`choice_head` present vs absent) and non-aligned epoch counters are
    consistent with either; treat the two as *related-but-distinct* runs unless a
    lineage link is confirmed elsewhere.
- **Strength / plateau:** plateaued at **~50 Δelo**, confirmed
  **buffer-knob-proof** — reuse/window/freshness levers all exhausted, strength
  flat, **0/16 vs Rapfi**. Trading real blows (stronger than past-self, even-ish
  with the era-2 milestone champ `epoch0235`) but can't touch max-strength Rapfi.
  A healthy *developing* net that stopped developing. Bruce is the self-play-only
  ceiling — the thing every external-teacher / defense experiment is trying to
  beat.

## 3. The idx-2 board — "the Bruce-Lee board"

The single position Bruce practices. From `gomoku/eval_panel.py`
(`IDX2_OPENING`, comment: *"white to move. The board 'Bruce' trains on. (row,
col) coordinates; 15×15."*):

```
IDX2_OPENING = ((3, 2), (5, 4), (4, 5))   # B, W, B  →  WHITE to move
```

Three stones already placed (Black, White, Black), so it is **white-to-move**.
This is a **15×15-only** board — the coordinates and the whole Bruce lineage live
at `GOMOKU_BOARD_SIZE=15`. It was chosen because among re-centered Rapfi openers
it was the **fairest (~50%)**. Whenever the wiki says "the idx-2 wound" or "solve
the Bruce-Lee board for black," this is the position.

## 4. The experiments on / around Bruce

### 4a. The white-defense wound (the binding measurement)

The one measured hole in Bruce, and the driver of everything after. Champion
**`eval502` vs native Rapfi-NNUE** (the Gomocup 2024+2025 winner), **5 s/move
single-thread, n=24, #30 panel harness → 5W-19L-0D = 20.8% overall**. The color
split *is* the finding
([white-side-defense-plan.md](white-side-defense-plan.md) §1B.2):

- **black (attacking): 5-7-0 = 42%** — competitive with the #1 engine even at a
  ~10× compute disadvantage.
- **white (defending): 0-12-0 = 0%** — **swept**. The *entire* strength shortfall
  is the white-side defense gap. Rapfi punishes the defensive lapse every game.

Freestyle gomoku has a provable first-player edge, so white's only job is to
*never lose*; Bruce never learned the refutation, and no external attacker in
self-play ever forced it to. This 0/12 wound is what the distillation mine and
the aux-head pivot were both aiming at.

### 4b. The idx-2 distillation mine — "Bruce Lee one-position" (#86)

The over-specialization bet, taken literally: mine **Rapfi's own** soft
policy+value over the idx-2 neighbourhood at scale, pretrain a `large` (Bruce-size
128×10) seed on it, then run standard AlphaZero self-play from idx-2 **only** —
can a Rapfi-distilled specialist stand against Rapfi *in that one position*, and
crack white? Full page:
[rapfi-idx2-distillation-mine.md](rapfi-idx2-distillation-mine.md).

- **Infra: SUCCESS.** `gomoku/rapfimine/` — multiprocess flat-file BFS mine,
  D4-canonical dedup, crash-robust sharded npz. Fixed a Rapfi multiPV mate-crash
  and a thread-per-line perf bug → **~700 moves/s** on the M5 Max; **1,126,597**
  canonical idx-2 positions banked at `/Users/jason/data/rapfimine/idx2_15x15/`.
  Then `rapfimine.pretrain` → `checkpoints/idx2_pretrain.pt` →
  `run_sweep --cell G15-idx2-warmstart --resume …` (byte-identical to Bruce's
  cell), wandb `idx2-warmstart-86`, trained to epoch 250 →
  `checkpoints/idx2_warmstart_final.pt`.
- **Science: inconclusive by choice.** At epoch 250 the warm-started net **still
  reads 0/48 vs strong Rapfi @idx-2** — the same wall the seed hit. It *did* climb
  the low end (100% vs random/heuristic/lookahead-d2, beats rapfi@25ms, loses at
  50ms+). Beating Rapfi is a **multi-day climb** (for scale, Bruce's black-42% bar
  took ~3,700 epochs); **not pursued** — Jason banked the run for its
  infrastructure value. Reusable byproducts: the `fast_eval` gradient (~20 s,
  think-time as the strength dial) and the `eval_idx2` gate.

An earlier, gentler variant of the same idea — layering a *policy* Rapfi teacher
onto warm-started Bruce (`--teacher-weight 0.3`, #77) — was **catastrophic**:
0W-48L-0D vs frozen Bruce-1 after 362 epochs. The naive external-signal injection
knocked Bruce below his own plateau and he stably stayed there (no self-heal),
mild evidence the equilibrium is a *delicate basin*, not a hardened floor.
Lesson: the injection must be gentle or it corrupts the trunk before any benefit
accrues. (See [eval-teacher-sensei.md](eval-teacher-sensei.md).)

### 4c. The forward VCT frontier + danger map — "solve the Bruce-Lee board for black" (2026-06-28)

Forward-expand idx-2 (white to move) as an **AND/OR frontier** where Rapfi-top-8
generates moves for both sides and the **GPU mega VCT solver is the only oracle**
(black VCT = win-terminus, white VCT = black-fumble loss-terminus; no
minimax/backup). A deliberately massive *approximation* — run to learn the
reachable shape, not a sound solve. run-a: **9.6M nodes / depth-11 / 90-min
wall**, throughput dead-flat ~1,750 nodes/s. The **danger map** (depths 0–7,
149,627 nodes) reads idx-2 as **black-favourable but ~half-unknown**
(`white_threat 0.28` vs `black_threat 0.08`), and shows Rapfi's mid-ranking is
**not danger-calibrated**. Harvest ≠ backed-up strategy. Full page:
[idx2-vct-frontier-map.md](idx2-vct-frontier-map.md).

### 4d. The VCT-defense aux-head pivot (#102/#103, 2026-07-01)

The #102 design executed: a per-cell supervised **"VCT-blunder map"** defense aux
head (GPU-oracle labels via escape-search over every legal move — a move is
"lost" iff all children lose), the opponent-independent gradient argued to be the
way past self-play's structural defense ceiling. Run **two ways, both fail to make
the net *defend* — instructively.** Full page:
[vct-defense-aux-head-result.md](vct-defense-aux-head-result.md).

- **A — from-scratch 9×9** (wandb **`8mtowemb`**, retired e1152): the head
  **learns the representation** (`train/vct_loss` 0.60→0.03, mask_frac ~0.9) but
  `selfplay/plies_mean` stays flat **~9-10 for 1152 epochs** — the #101 attractor,
  unchanged *even with the representation present*. Rules out "can't *see* the
  blunder": it can; self-play just has no opponent to *punish* weak defense.
- **B — Bruce/idx-2 pivot** (wandb **`zrjfwny2`**, warm-started from Bruce
  `g15_128x10_bigbuf_e588_best.pt` at ~e613, retired e862 ≈ 257 pivot epochs):
  layer the head onto the 128×10 champion via a new
  `load_checkpoint(force_aux_vct=True)` splice (`gomoku/model.py:635/673`,
  `train.py:1417-1422`, off = byte-identical) + restrict self-play to the idx-2
  wound + VCT-terminus. The head learns again (0.52→0.026) but the self-play
  **policy DRIFTS** (`loss/policy` 1.93→2.62, `plies` 11.6→9.6) — the
  terminus/narrow-single-opening regime **specializes / erodes** the champion.

**EVAL-SATURATION CATCH (read carefully).** The idx-2 gate (`eval_idx2`, **n=48,
sims=160**, `GOMOKU_BOARD_SIZE=15`) reads **0/48 on the pivot AND 0/48 on frozen
Bruce** — SATURATED, non-discriminating. So it does **NOT** show the pivot
degraded Bruce (Bruce also scores 0/48 here). **Do not write "the pivot degraded
the champion black 42%→0"** — that is false. The "Bruce black ~42% / white 0/12"
number comes from a *different, stronger-attacker eval config* (§4a: eval502,
5 s/move, n=24), not this sims=160 gate. The real "it fell apart" evidence is the
**policy drift**, not the Rapfi eval; a clean strength-delta would need a
higher-sim or direct-H2H eval — **not run** (abandoned at the pivot stage).

**Verdict:** the aux head reliably learns the defensive **representation**, but
nothing so far makes the **policy** act on it — a working **sensor** with **no
actuator**. *"Frankenstein + aux head is not the recipe"* (Jason). Next = target
the policy directly (escape-search as a policy target / defense head at MCTS
inference / an opponent that actually forces defense).

## 5. Status & lineage

- **Bruce is the current spine's baseline-to-beat.** The project moved past the
  9×9 WL series into the **15×15 swap2 / Bruce era** (2026-06-24 correction,
  [training-run-lineage.md](training-run-lineage.md)). Bruce's self-play-only ~50
  Δelo plateau is confirmed buffer-knob-proof.
- **Its one wound is white defense** (black ~42% / **white 0/12** vs Rapfi
  @idx-2). Every subsequent experiment — the distillation mine (#86), the forward
  VCT frontier, the VCT-defense aux head (#102/#103) — is an attempt to close that
  wound.
- **None have beaten Bruce yet.** The gentle/naive Rapfi teacher regressed it
  (#77); the distillation warm-start was banked before the multi-day climb; the
  aux head is a percept without an actuator. The open problem is unchanged:
  **wire a defensive signal to the move-selection**, not just the representation.
- **Key artifacts / pointers:** wandb `gogpmbhw` (Bruce), `zrjfwny2` (aux-head
  pivot B), `8mtowemb` (aux-head from-scratch A); checkpoints
  `~/data/swap2/babysit/snapshots/g15_e2659_0623_2025.pt` (e2659),
  `g15_128x10_bigbuf_e588_best.pt`, `g15_128x10_bigbuf_eval502.pt`;
  `checkpoints/idx2_pretrain.pt` + `checkpoints/idx2_warmstart_final.pt`; mine at
  `/Users/jason/data/rapfimine/idx2_15x15/`; issues #86 (mine), #102/#103 (aux
  head), #77 (teacher regression), #45/#49 (white-defense probe), #37 (the
  white-defense-degeneration hypothesis).

## Cross-refs

- [rapfi-idx2-distillation-mine.md](rapfi-idx2-distillation-mine.md) — the
  "one-position" mine + warm-start (#86).
- [idx2-vct-frontier-map.md](idx2-vct-frontier-map.md) — solving the Bruce-Lee
  board for black (forward VCT frontier + danger map).
- [vct-defense-aux-head-result.md](vct-defense-aux-head-result.md) — the
  sensor-with-no-actuator aux-head result (#102/#103).
- [white-side-defense-plan.md](white-side-defense-plan.md) — the white-defense
  wound (§1B.2) and the "target the policy" next step.
- [board-size-transfer-and-warm-start.md](board-size-transfer-and-warm-start.md)
  — the warm-start / head-layering mechanics (the `force_aux_vct` splice, same-size
  resume).
- [swap2-opening-protocol.md](swap2-opening-protocol.md) §12 and
  [training-run-lineage.md](training-run-lineage.md) — the swap2/Bruce era origin
  and lineage. Chronological evidence: `TRAINING_WIKI.md` 2026-06-22 (Bruce
  overnight), 2026-06-25 (#86 mine), 2026-07-01 (#103 aux head).
