# VCT-terminus self-play — the 9×9 A/B result (#100)

> **RESULT (2026-06-30): VCT-terminus self-play is a THROUGHPUT WIN but a ROBUSTNESS LOSS.**
> A net trained to end each self-play game at the first cap50 VCT (idea-pile #11, built #98/#99)
> reaches the **same fixed-baseline strength** as a play-to-five control at **~45% of the wall-clock** —
> but **loses head-to-head to that control 75–25 (0 wins in 120 games, every config)** and **0–40** to
> the frozen champion. Mechanism: ending every game at the first VCT (≈ply 9 in self-play) means the net
> only ever learns to beat a *non-defending copy of itself* by rushing to a forced win; it never learns to
> defend or play from behind. Against any opponent that defends well enough to deny a VCT (the control, the
> champion, even a shallow Rapfi@50ms) it **never reaches a VCT at all (finisher fires = 0)** and collapses.
> This is idea-pile #11's own caveat — "terminating at VCT removes the defender's play-it-out learning" —
> confirmed as the **dominant** effect: **attack-only specialization, a cousin of fast-attack collapse.**

## The experiment (matched 9×9 A/B)

Two `scripts/run_sweep.py` cells, byte-identical **except the terminus**:

- `vctsci-control` — standard self-play, plays out to a real five-in-a-row.
- `vctsci-terminus` — `--vct-terminus --vct-terminus-budget 50`: each ply, one bulk-synchronous cap50
  `solve_vct_mega_bb` across the whole wave ends any game whose side-to-move has a forced VCT, recording
  the oracle's one-hot winning move + exact terminal value (#98).

Both cloned from `derby-v9-small` (fresh seed-0 champion-recipe twin: 64×4, 100 sims, `global_pool`,
`ema_tau=0.99`, `sgd-steps-per-epoch 64`) with three **matched** deviations: **no `--gumbel-root`** (the
terminus code raises `NotImplementedError` on gumbel+terminus), **no `--vcf-teacher`** (VCF ⊆ VCT so the
terminus preempts it + it doubles the per-ply Metal solve), `--value-discount 0.98` kept on both.
`n_workers=4` (this recipe's trainer is non-blocking async at a fixed 64-SGD-step cadence, so 8 workers
just add Metal co-tenancy on the terminus solver without feeding the fixed-cadence trainer).

Grown by `--resume` (warm buffer, same wandb timeline): fresh 0→100 (smoke), then to **e500**.
**Epoch = SGD-step-matched** (64 steps/epoch, non-blocking ingest → the trainer cycles decoupled from
generation; "100 epochs" is 6 400 SGD steps). wandb: terminus `cc0fy0ao`, control `7cu4ho9w` (project
`gomoku`).

## Training dynamics

| | terminus | control |
|---|---|---|
| self-play plies (cold → e500) | 36 → **9.1** | 34 → 11.8 |
| wall/epoch (steady) | **~2.1 s** | ~4.7 s |
| buffer fills to 1.5 M by | ~e50 | ~e15 |
| pl / vl @ e500 | 2.35 / 0.055 | 1.89 / 0.030 |
| internal EMA elo @ e500 | **1366** | 1347 |

- Both **fast-attack-narrow** (plies collapse); the terminus harder/earlier — it is *directly rewarded for
  reaching a VCT as fast as possible*, so it converges to ~ply 9. By the launch-runbook's balanced-baseline
  test the collapse looks *healthy* (all three rungs h/la2/la4 climb together, not one-trick) — the defect
  is **relational** and only shows up head-to-head.
- **Strength climbs only in the EXTENDED epochs.** Raw internal elo sat at ~389 through e100, then broke
  out (terminus 389→1366; control 603→1347). **100 epochs was pure smoke** — the whole result needed ≥500.
- **Throughput win (real, holds):** the terminus reaches *equal* fixed-baseline strength at **~45% of the
  control's wall-clock** — shorter games ⇒ less generation + ingest/augment work per epoch.

## Eval methodology — the EMA-weights gotcha (reusable)

`worker_weights.pt` holds the **EMA (averaged) weights** — what self-play AND the internal eval use ("the
model"). `load_checkpoint(epochNNNN.pt)` returns the RAW `model_state_dict`, which under `ema_tau=0.99` on
sharp short-game training is **dramatically weaker** (terminus **6%** vs heuristic on RAW vs **~68%** on
EMA). The first eval pass used the epoch checkpoint and made the terminus look like a 6% net.
**Lesson: eval the EMA (`worker_weights.pt`), never the epoch checkpoint's raw state_dict** — the gap is
largest exactly for aggressive/short-game recipes where EMA smoothing does the most work.

## Fixed baselines (EMA, n=40, `scripts/vctsci_finisher_eval.py`) — the transitive ruler

| net (config) | heuristic | la2 | la4 |
|---|---|---|---|
| terminus raw | 66% | 81% | 49% |
| terminus **+finisher** | 66% | 81% | **61%** |
| control raw / +finisher | 80% | 75% | 38% |
| champion raw | 62%\* | 75%\* | 61%\* |

- **Finisher lift is concentrated at the top:** terminus **+12.5% vs la4** (rescues fumbled conversions),
  ~0 vs heuristic/la2 (already converts), control ~0 everywhere. The finisher helps *only* vs the strongest
  opponent the net can still out-attack.
- **Fixed baselines saturate** for strong nets — heuristic/la2 draw heavily (control 20-0-20 vs la2), so
  they cap out and stop discriminating. \*The champion's "low" 62% is **draw-saturation, not weakness** (it
  goes 40-0 on the control H2H). **Corollary: at this strength on 9×9 the fixed baselines are a coarse
  ruler — gate on H2H.**
- terminus ≈ control on fixed baselines (terminus better vs la2/la4-with-finisher, control better vs
  heuristic) — a **near-tie**, at ~half the terminus's wall-clock.

## Head-to-head (n=40, non-transitive — the discriminating ruler)

| matchup | winrate (first net) | finisher fires |
|---|---|---|
| terminus+finisher vs **control**-raw | **25%** (0W-20L-20D) | **0**/1060 |
| terminus+finisher vs control+finisher | 25% (0-20-20) | 40/2120 |
| terminus-raw vs control-raw | 25% (0-20-20) | 0 |
| terminus+finisher vs **champion**-raw | **0%** (0-40-0) | **0**/760 |
| terminus+finisher vs champion+finisher | 0% (0-40-0) | 80/1540 |
| **champion**-raw vs control-raw (calib) | **100%** (40-0-0) | — |

**The terminus wins 0 of 120 games vs the control**, in every config, and **never reaches a single VCT**
(fires = 0/1060). It draws as black (first-move cushion), loses every white game. The finisher is *inert*
because there is no VCT to convert — the control's full-game defense denies it. Same story, harder, vs the
champion (0-40; it denies every VCT too). **Pecking order: champion ≈ Rapfi@50ms ≫ control > terminus.**

## The mechanism (why both predictions — Jason's and Claude's — lost)

- **Going in:** Jason predicted terminus+finisher would **crush** the control ("just needs to stumble on
  one VCT → instawin, while the other must be flawless the whole way") and even beat the best-ever
  champion. Claude hedged to a control-crush too, but flagged a counter-mechanism as a *risk*.
- **What happened:** the counter-mechanism was the **whole story**. The terminus's self-play opponent was
  *a non-defending copy of itself*; because every game ended at the first VCT (~ply 9, usually its own), it
  accumulated **zero** experience defending or playing a long game. Its policy = "rush to a forced win
  against something that won't stop you." Against an opponent that **does** stop you — defends soundly and
  takes initiative — it never gets to attack (fires = 0), and the resulting long game is out-of-distribution.
- The control, forced to play to five, learned **both** sides — which is why, at *equal fixed-baseline
  strength*, it wins the sibling H2H 75-25. **Non-transitivity in the flesh** (the wiki's standing rule:
  gate on fixed rulers / H2H-vs-frozen, never sibling-strength alone).

## Rapfi coda (for giggles)

`champion+finisher vs Rapfi@50ms` (native mix9svq NNUE, 9×9, `run-rapfi`): **0W-0L-20D — 20 straight
draws.** Even a shallow 50 ms Rapfi defends well enough to deny every VCT; the champion can't break through
and never loses. On a 9×9 board sound defense walls the game off — it's drawish at this level, and the
whole "reach-a-VCT" edge evaporates the instant the opponent can defend.

## Verdict for idea-pile #11

- **CONFIRMED:** the throughput/efficiency claim — equal fixed-baseline strength at ~½ the wall-clock,
  exact terminal values, ~half the plies.
- **REFUTED as a way to train a strong/robust player:** ending at the VCT skips the defensive half of the
  game, producing attack-only specialization that loses H2H to a play-to-five net of *equal* fixed-baseline
  strength. Idea #11's own caveat is the dominant effect, not a footnote.
- **NOT killed:** the seek-VCT *objective* ("reach a VCT") is intact; the defect is specifically the missing
  **defense**. Untested fixes: keep recording moves past the terminus for the *losing* side (defense data);
  mix terminus + full games; a curriculum that lengthens the terminus over training. Natural next probe is
  **#101** — train the terminus player long: does p90 plies ever hit 81 (learns to *avoid* VCTs) or does it
  get "wicked strong at short games"?

## Long-run coda (#101): a stable attractor at p90 ≈ 14, never the 81 gate

> **RESULT (2026-07-01): HYPOTHESIS B HELD.** Trained LONG with **no evals**, the VCT-terminus player
> does **not** learn to avoid VCTs (p90 → 81, a full board = the old 9×9→11×11 graduation gate). Instead
> `selfplay/plies_p90` settles into a **rising-then-flattening fixed point at ≈14.5** (mean ≈9.6) — it gets
> *sharper at the same short game*, never longer ones. Every dial converges: a **stable attractor in the
> low teens**, ~6× short of 81.

**The run.** #101 asked the natural next question of #100: if you train the terminus player continuously
with `--internal-eval` **off** (just train, no train-time strength ladder), does `selfplay/plies_p90` ever
climb to **81** (the retired 9×9→11×11 gate = a full board)? Two hypotheses: **(A)** p90 climbs → the net
learns to *avoid* VCTs (emergent VCT-avoidance at equilibrium, both sides steering away from giving/taking
forced wins); **(B, Jason's bet)** it never gets there and just gets "wicked strong exploring short games."

A **fresh from-scratch** run (`vctsci-terminus` recipe verbatim — 64×4, `n_sim=100`, 4 workers,
`--vct-terminus --vct-terminus-budget 50`, `ema_tau=0.99`, 64 SGD-steps/epoch; the #100 terminus buffer
had died with its worktree, and fresh gives a clean single p90 timeline through the collapse phase). wandb
**`kgajrge4`** (`jasonyandell-forge42/gomoku`; run dir `~/data/vctsci-101-long/`, outside the repo, leaving
#100's preserved checkpoints untouched). At this writeup it had reached **~2,700 epochs (≈14× the #100 e500
slice) and was still riding its 12h / 1M-epoch wall** — but the trajectory had been flat at the fixed point
for ~1,000 epochs, so the scientific call (locked by ~e1,200) only hardened. Hand-off was to the #103
moonshot.

**The p90 trajectory (verified from `kgajrge4`, 200-epoch block means):**

| epoch block | p90 (block mean) | mean plies | loss/policy | loss/value |
|---|---|---|---|---|
| cold (e6–e21) | ~28 | ~19–21 | 4.38 | 0.39 |
| **200–399 (trough)** | **11.9** | 8.5 | 2.57 | 0.111 |
| 400–599 | 12.7 | 8.6 | 2.39 | 0.061 |
| 600–799 | 13.2 | 8.9 | 2.32 | 0.042 |
| 800–999 | 13.4 | 9.1 | 2.28 | 0.035 |
| 1000–1199 | 13.6 | 9.3 | 2.25 | 0.029 |
| 1200–1399 | 14.0 | 9.4 | 2.23 | 0.022 |
| 1600–1799 | 14.4 | 9.5 | 2.20 | 0.023 |
| 2000–2199 | 14.7 | 9.6 | 2.18 | 0.022 |
| 2400–2599 | 14.6 | 9.6 | 2.17 | 0.026 |
| 2600–2799\* | 14.6 | 9.6 | 2.17 | 0.030 |

\*partial (n=112). p90 **collapsed** from cold ~28 to the 11.9 trough by ~e85, then a **decelerating creep**
back up — increments off the trough of **+0.8, +0.5, +0.2, +0.2, +0.4, +0.2 …** flattening to **~14.5–14.6**
and holding there for the final ~1,000 epochs. **Never a hint of a climb toward 81.** mean plies pinned
~9 the whole way. Policy loss fell monotonically 4.38 → ~2.17 then flattened; value loss 0.39 → ~0.022,
flat. **Every dial converged to a fixed point** — the definition of a stable attractor, just a *rising* one
in the low teens.

**Mechanism — co-evolution, capped by the self-play ceiling.** The 11.9→14.5 creep is real but it is the
defender (its own EMA twin) learning to **postpone** the VCT by a few plies, *never to prevent* it — because
self-play offers no opponent strong enough to *punish* weak defense. The net got sharper at the **same ~9-ply
game** (pl/vl down), not at longer games. This is the #100 finding restated as a dynamical fact: the missing
half is DEFENSE, and self-play alone cannot supply it (the same self-play ceiling that #100 exposed
head-to-head). Hypothesis A (emergent VCT-avoidance) is **refuted**: avoidance would show as p90 marching
toward a full board; instead it asymptotes 6× short.

**Two confounds/caveats, stated honestly:**
- **`plies` is an unreliable defense proxy (the cap50-recall confound).** The terminus ends at the first
  *cap50*-detected VCT. As play sharpens, some genuine VCTs need **>50 nodes** and cap50 misses them, so
  **part** of the p90 creep is the detector **losing recall on the shifting distribution**, not the net
  genuinely defending longer. So the 11.9→14.5 rise over-states real defensive improvement — only
  **`fires>0` vs a real opponent** (the #100 finisher yardstick) can settle "did it learn defense," and #100
  already answered that **no** (fires = 0 vs the control/champion).
- **"Gets stronger at short games" is INFERRED, not measured.** #101 ran with `--internal-eval` **off**, so
  there is **no** strength number for this run — the "sharper" claim rests entirely on falling `loss/policy`
  and `loss/value`, not on any fixed-baseline or H2H measurement. Read it as a plausibility argument, not a
  proof.

**Verdict.** Jason's bet (B) held: no VCT-avoidance, no march to 81 — a **stable attractor at p90 ≈ 14.5 /
mean ≈ 9.6**, the net sharpening inside the fast forced-win regime. This confirms the #100 diagnosis from a
second angle: the self-play defensive ceiling is *structural*, not a matter of undertraining — 2,700 epochs
of pure self-play buys a few plies of postponement and nothing more. **The way past the ceiling is
opponent-independent defensive signal** — a supervised VCT aux-head (#102) / the from-scratch VCT-gate +
aux-head gauntlet (#103), which regress the VCT structure directly rather than hoping a non-defending twin
will teach defense.

## Reproduce

- Cells: `scripts/run_sweep.py` → `vctsci-control` / `vctsci-terminus` (fresh 9×9; grow with
  `--resume <latest.pt> --epochs N --internal-eval`).
- Eval: `scripts/vctsci_finisher_eval.py --run <id> --out <jsonl>` (one matchup/process, EMA weights =
  `worker_weights.pt`), then `--collate <jsonl>`; `--list` shows the 21 matchups. Fan out with
  `seq 0 20 | xargs -P N ...` (wall-clock; MCTS is single-core-bound so a lone process leaves the GPU idle).
- Rapfi: `model:checkpoint=<champ>,vct_finish=50 vs external:cmd=run-rapfi,timeout_ms=50,size=9`
  (`GOMOKU_REPO=~/code/gomoku` so the wrapper finds `engines/rapfi/pbrain-rapfi`).
- Checkpoints (worktree artifacts): `sweep_runs/vctsci-{terminus,control}/checkpoints/worker_weights.pt`
  (EMA @ e500).

## Cross-refs

- [idea-pile.md](idea-pile.md) #11 — the idea + this result's marker.
- [vct-cascade-run-2026-06-30.md](vct-cascade-run-2026-06-30.md) — the median-first-VCT-ply-19 finding that motivated it.
- [mega-vct-solver.md](mega-vct-solver.md) — the cap50 oracle (`solve_vct_mega_bb`) used as terminus + finisher.
- [launch-sequence-runbook.md](launch-sequence-runbook.md) — balanced-baseline / fast-attack-collapse indicators.
- `TRAINING_WIKI.md` 2026-06-30 (#100) + 2026-07-01 (#101) — the chronological run records.
- **#102 / #103** — the supervised VCT aux-head (and the from-scratch VCT-gate + aux-head gauntlet): the
  opponent-independent defensive gradient that the long-run coda argues is the way past this self-play ceiling.
