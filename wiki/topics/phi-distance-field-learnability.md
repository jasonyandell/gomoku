# Φ distance-to-VCT field learnability — can a net see "which moves move the proof frontier toward my VCT vs theirs?"

**One-line finding.** The dual **proof-frontier potential** Φ — `phi_off = γ^(my-moves-to-my-next-VCT)`
and `phi_def = γ^(opp-moves-to-their-next-VCT)` — is **strongly learnable and generalizes** to
held-out, shard-disjoint games: a **CNN** scores **offense ρ=0.719 / R²=0.761 / reach-AUROC=0.912**
and **defense ρ=0.761 / R²=0.690 / reach-AUROC=0.917** on 101,745 unseen positions. So the field
whose **gradient** answers *"which move advances my forced-win frontier and retreats theirs"* (Jason's
framing, 2026-06-27) is real and perceptible. Two sharp secondary results: **(1)** Φ is **NOT
count-dominated** — the CNN nearly *doubles* a ridge-on-raw-board baseline (offense ρ 0.36→0.72), unlike
is-VCT recognition which logreg-on-counts almost matched; closeness-to-a-fork has real spatial structure
beyond threat-counting. **(2)** the **global-receptive-field bet for attention does NOT pan out** — even on
this whole-board target (attention's claimed home turf), **a param-matched CNN beats attention a third
time** (offense ρ 0.719 vs 0.648), and this time attention can't hide behind undertraining: the CNN
early-stopped at epoch 6 (best val by epoch 1) while attention ran 25 epochs, plateaued ~0.72, and still
lost.

**Code:** `scripts/threat_shapes/gen_phi_dataset.py` (no-GPU label builder — reads the miner's per-ply
verdicts, no re-solve) · `scripts/threat_shapes/train_phi.py` (CNN + attention regression, ridge/mean
baselines, held-out R²/Spearman/reach-AUROC/calibration). **Builds on:**
[vct-reachability-mining.md](vct-reachability-mining.md) §1 (the free Φ target — *designed* there, **trained
here**) · [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 (the oracle that produced the verdicts).
**Feeds:** [shape-library-engine.md](shape-library-engine.md) **L2** (this is the first real L2 model — a
verifiable, non-bootstrapped potential) · [white-side-defense-plan.md](white-side-defense-plan.md) (the
defense channel). **Completes the trilogy:** [vct-recognition-learnability.md](vct-recognition-learnability.md)
(see a *present* VCT) → [seeker-steering-learnability.md](seeker-steering-learnability.md) (imitate the
*move*) → **this** (regress the *field*).

Date: 2026-06-27. Hardware: M5 Max, 48 GB; MPS (torch) for training; labels from the MLX-Metal oracle (no
re-solve this run — reused the miner's trits). A deliberately **small/untuned feasibility** test, run with
**zero GPU contention** with the live CPU-only `collect_rapfi` fleet. Wall: gen 21 s (CPU), train+eval 2,820 s.

---

## 1. The question and why Φ answers it

The seek-VCT plan's player ([shape-library-engine.md](shape-library-engine.md) §5) scores a move by
`Δ(my distance) + Δ(opp distance)`. Jason sharpened it to: **"which moves move the proof frontier toward
my VCT, and which toward theirs?"** That is exactly the **gradient of a two-channel potential**: `phi_off`
is how close my VCT-proof is to closing, `phi_def` is how close theirs is, and a move is good iff it raises
mine and lowers theirs. This experiment asks the prerequisite: can a net even **perceive** that field from
the raw board, on games it never saw? If yes, the steering half of the plan (L2) has a learnable substrate;
the exact forcing finish stays the oracle's job.

## 2. The target (free, no re-solve) and the honesty around it

The forward puzzle miner already wrote a per-ply verdict for every ply of every game, so the field is read
straight off disk ([vct-reachability-mining.md](vct-reachability-mining.md) §1):

- **`phi_off(p) = γ^d_off`**, `d_off` = the **mover's own moves** to its nearest *future same-parity* proven
  VCT (`win&~cap`), `(q−p)//2`; `phi_off=0` if none this game (the floor). `γ=0.8`.
- **`phi_def(p) = γ^d_def`**, `d_def` = **opponent-moves** until the opponent first holds a VCT, `(q−p+1)//2`.
- **`cap` plies excluded** (unknown verdict). Board is side-to-move-relative (`bs[p][0]` = the mover).

**Honesty (it's a proxy frontier).** Distance is measured **along Rapfi's realized line of play**, not under
optimal play — so `phi_off` is an **upper bound** on the true distance (the game found *a* path; a shorter one
it didn't take may exist) and the floor is a **lower bound** (it misses wins Rapfi didn't take). And per the
[knife-edge](vct-reachability-mining.md) §2, near the onset the frontier is a **cliff, not a slope** — the
smooth-gradient reading is cleanest in the quiet region, and a real player verifies every leaf with L0
anyway. The net is learning the *bracketed, noisy* frontier — enough to test perception and to seed L2.

**Data.** 40,000 games → 1,268,747 plies, `cap` 204,647 excluded → **1,167,002 train / 101,745 test**.
Shard-disjoint split (`md5%10`, identical rule to the recognition/seeker probes → all three comparable),
**overlap 0**, 49-shard val carve for early-stop. **0 frame mismatches** over all 400 shards. Floor 54.5%
(so ~45% of positions reach a same-parity VCT in their own realized game).

## 3. Result — the field is real, spatial, and (again) a CNN's game

Held-out (33 disjoint test shards, n=101,745). **ρ = Spearman rank** (the steering-gradient quality — does it
*order* positions by closeness?); **reach-AUROC** = can it separate "a VCT is reachable at all" (Φ>0) from the
floor; **R²** = variance explained.

| model | params | OFF R² | OFF ρ | OFF reach-AUROC | DEF R² | DEF ρ | DEF reach-AUROC |
|---|---|---|---|---|---|---|---|
| mean (floor) | — | −0.01 | 0.09 | 0.51 | −0.01 | 0.11 | 0.51 |
| ridge on raw board | tiny | 0.20 | 0.36 | 0.67 | 0.29 | 0.54 | 0.82 |
| **CNN** | **376k** | **0.761** | **0.719** | **0.912** | **0.690** | **0.761** | **0.917** |
| attention | 348k | 0.582 | 0.648 | 0.873 | 0.522 | 0.673 | 0.881 |

Val ρ: **CNN 0.775** (best at epoch 1, early-stopped epoch 6) · **attention 0.719** (epoch 20, plateaued).
Both nets are **well-calibrated** (predicted-decile ≈ true mean throughout; CNN top decile pred 0.91 → true
0.93).

**Reading it:**
- **The frontier field is learnable and generalizes — emphatic yes.** A CNN ranks unseen positions by
  distance-to-forced-win at ρ≈0.72–0.76 and tells reachable-from-floor at AUROC≈0.91. The gradient that
  answers Jason's question is there.
- **Φ is NOT count-dominated — the key contrast with recognition.** In [recognition](vct-recognition-learnability.md),
  logreg-on-threat-counts (0.946) nearly matched the nets — "is there a VCT *now*" is countable. Here the
  ridge baseline gets only ρ 0.36 (offense) and the CNN nearly *doubles* it. **Closeness-to-a-fork is genuinely
  spatial/relational**, not a threat tally — which is the whole reason it's worth learning (and a hint the
  molecule/stencil structure lives in *reachability*, not in present threats).
- **Defense reads slightly *better* than offense** (CNN ρ 0.761 vs 0.719; AUROC 0.917 vs 0.912). The opponent's
  looming VCT is a bit more rank-predictable than your own — plausibly because incoming danger is carried by
  concrete present threats (fours/open-threes) while your own future win is more latent buildup. **Good news for
  the white-defense wound: the net sees the incoming frontier at least as well as the outgoing one.**
- **CNN beats attention a THIRD time — and the undertraining alibi is gone.** Recognition and BC-steering were
  *local* targets, so a conv winning was unsurprising and attention was visibly still climbing. Φ is **global by
  construction** (closeness-to-a-fork is a whole-board fact) — the fair audition. Param-matched (376k vs 348k),
  attention got **3×+ the gradient steps** (25 epochs vs the CNN's early-stop at 6) and still lost by ρ 0.07.
  At this scale a **conv tower + global-average-pool captures the global signal better than token
  self-attention** — the theoretical global receptive field did not translate into better frontier perception.

## 4. What it means for the plan (and what it does NOT settle)

- **L2 has its substrate.** The first real L2 model regresses a **verifiable, non-bootstrapped** potential
  (oracle distances, not self-play value) and generalizes — so the "regress the gradient toward
  shape-reachability on verifiable targets" plan ([shape-library-engine.md](shape-library-engine.md) §4) is
  green at the perception level. **Default architecture for L2: the CNN.**
- **It does NOT prove strong play.** Ranking realized-play distance ≠ winning. The decisive test remains the
  **hybrid player** (consult L0 every ply for the forcing finish; let Φ steer the quiet region) vs a fixed
  baseline — Phase C, GPU-spending, gate with Jason. ρ on a proxy frontier is the green light, not the trophy.
- **The attention question is now strongly — not yet *finally* — settled.** Three losses, the last param-matched
  on attention's home turf with the epoch alibi removed, is real evidence the locality+GAP prior wins at
  laptop/feasibility scale. The remaining escape hatches: much larger data/capacity, or a *sequential* setting
  (this is still single-position regression, not whole-game seeking). If attention ever earns its keep, it is
  there — not here.

## 5. What we thought vs. what we found (banked)

- **Thought:** a global target would be attention's chance to finally beat the CNN. **Found:** it lost again,
  param-matched, with 3× the epochs — the cleanest CNN-wins evidence in the trilogy. *The global-receptive-field
  bet does not cash out at this scale.*
- **Thought (open):** maybe reachability is as count-dominated as recognition. **Found:** no — the CNN nearly
  doubles the ridge baseline; the field is genuinely spatial. *The interesting structure is in distance, not
  presence.*
- **Surprise:** defense slightly out-ranks offense — the net sees incoming danger best, the opposite of what the
  "fast-attack" training failure mode would predict, and aimed straight at the white wound.

## 6. Artifacts

| Path | What |
|---|---|
| `scripts/threat_shapes/gen_phi_dataset.py` | no-GPU Φ-field builder (per-ply offense/defense distance off the miner verdicts; cap excluded; shard split) |
| `scripts/threat_shapes/train_phi.py` | CNN + attention regression, ridge/mean baselines, held-out R²/ρ/reach-AUROC/calibration |
| `~/data/puzzle_miner/phi_exp/phi_metrics.json` | full metrics (this table + calibration deciles + disjointness proof) |
| `~/data/puzzle_miner/phi_exp/{phi_train,phi_test}.npz`, `phi_shards.json` | dataset + split/coverage manifest |
| `~/data/puzzle_miner/phi_exp/{phi_cnn,phi_attn}.pt` | trained checkpoints (CNN is the L2 default) |

**Cross-links:** [vct-reachability-mining.md](vct-reachability-mining.md) §1 (the Φ target, designed) ·
[vct-recognition-learnability.md](vct-recognition-learnability.md) + [seeker-steering-learnability.md](seeker-steering-learnability.md)
(the trilogy) · [shape-library-engine.md](shape-library-engine.md) §4 (L2) ·
[gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 (the oracle) ·
[white-side-defense-plan.md](white-side-defense-plan.md) (the defense channel's target).
