# Seeker steering learnability — can a net *imitate* how a winner steers toward a VCT?

**One-line finding.** A net **can** behaviorally-clone the **quiet-phase (pre-onset) moves of the
side that reaches the first forced VCT**, and it **generalizes to unseen games**: on a held-out,
shard-disjoint test set it matches the *exact* move the seeker played at **top-1 0.386 / top-5
0.696**, vs **0.025 / 0.121** for the strong "play next to a stone" prior and **0.005 / 0.023**
for a random legal move. So the **steering signal is learnable and real** — the cheap green light
the **seek-VCT thesis** needed before paying for the hybrid-play eval. As with recognition, a
**CNN (224k) beats attention (339k)** at *next-move* imitation (top-1 0.386 vs 0.263) — but
attention was still climbing at the epoch cap (undertrained, not saturated), and this BC proxy is
*local* (the next move usually sits near the action), so it **does NOT settle** attention's real
bet that a *global* receptive field helps **sequential seeking** — that is the hybrid-play eval
(Phase C), not this.

**Code:** `scripts/threat_shapes/gen_seeker_dataset.py` (no-GPU label builder — reuses the miner's
per-ply verdicts for the onset, no re-solve) · `scripts/threat_shapes/train_seeker.py` (CNN +
attention per-cell policies with legal-move masking, dumb spatial baselines, held-out top-k).
**Builds on:** [vct-recognition-learnability.md](vct-recognition-learnability.md) (the recognizer
half; §4 named the seeker as attention's real audition) · [gpu-vct-feasibility.md](gpu-vct-feasibility.md)
§8 (the exact oracle that produced the verdicts → the onset) · the forward every-ply puzzle corpus
`~/data/puzzle_miner/` ([vct-backward-mining.md](vct-backward-mining.md), [shape-library-engine.md](shape-library-engine.md)).
**Feeds:** [shape-library-engine.md](shape-library-engine.md) **L2** (the AlphaZero/steering layer)
and the seek-VCT plan's next two phases (§4).

Date: 2026-06-26. Hardware: M5 Max, 48 GB; MPS (torch) for training; onset/labels from the
MLX-Metal oracle (no re-solve this run — reused the miner's trits). A deliberately **small/untuned
feasibility** test — a go/no-go, run `nice`d alongside the live `collect_rapfi` fleet (CPU-bound;
MPS/Metal was free) so it never competed.

---

## 1. The question and why it matters

The seek-VCT thesis ([vct-recognition-learnability.md](vct-recognition-learnability.md) §4): don't
search toward 5-in-a-row; **steer** play toward a position where *you* have a forced VCT, then hand
the tactical finish to the exact oracle. The split works because tactics and strategy have
**anti-correlated tractability** — quiet play is intractable to *solve* but tolerant of
*approximation* (give to a net); the forcing finish is intolerant of approximation but tractable to
*solve* (give to the oracle). The recognizer probe showed a net can *see* a present VCT. The next
prerequisite worth checking cheaply: can a net learn the **steering** itself — the moves that *lead
you to* a forced win — and does that perception survive on **unseen games**? If not, the net half of
the plan has nothing to stand on; if yes, the hybrid player is worth building.

## 2. Data + the definitions that make the number honest

**Onset, seeker, steering example (all read off the miner's already-computed verdicts; NO re-solve):**
- `onset(game)` = the **first ply** where the side-to-move has a *proven* VCT (a `puzzles.jsonl.gz`
  row with `win=True & cap=False`). The mover at the onset ply is the **seeker S** — the side that
  first owns a forced win (whether it later converts or misses it; both are valid "you reached a
  winnable position" examples, so we keep both).
- **STEERING EXAMPLE** = every pre-onset ply `p < onset` with `p % 2 == onset % 2` (S to move).
  Input = the side-to-move-relative board `bs[p]` (so `bs[p][0]` = S's own stones). Target =
  `moves[p]`, the move S actually played — a flat row-major cell index in the **same frame** as the
  board. These are the moves on S's path to a position with a forced win: the steering signal,
  imitated from a strong engine (Rapfi self-play).
- Games with **no onset** (no proven VCT at the miner's budget) are **excluded** — there is no
  seeker to imitate. A `cap` ply before the first clean win cannot corrupt the onset (onset counts
  only `win&~cap`); a deeper hidden VCT would make our onset a slight **over**-estimate (a few extra
  steering plies), never an under-estimate.

**Reused machinery, same frame guard.** Boards are CPU-replayed with `mine_first_vct.all_boards`
(the exact constructor the miner used), and every *present* puzzle key's board is cross-checked
against the replay (guard against the live size-16 collection reusing a filename); a disagreeing
game is dropped wholesale. **0 frame mismatches over all 400 shards.**

**Shard-disjoint split — the load-bearing correctness fact.** Consecutive plies of one game differ
by a single stone, so a position split leaks. Split is **by shard** (`md5(basename)%10`), the
**identical rule** the recognizer used, so the two experiments are directly comparable. 400 manifest
shards → **367 train / 33 test, overlap 0**, plus a **49-shard val** carved from train for
early-stopping (so the stopping signal doesn't leak either).

| split | shards | steering examples |
|---|---|---|
| train | 367 | 459,415 (200k used, capped for minutes) |
| test (held-out) | 33 | 41,332 |

500,747 steering examples from **38,927 onset games** (1,073 decisive-ish games had no onset). Mean
**208 legal cells** per test board — so the random-legal floor is ≈ k/208.

## 3. Result — steering is learnable; CNN again leads next-move imitation

Held-out (33 disjoint test shards, n=41,332). **Top-k legal-move-match** = is the move the seeker
*actually played* among the policy's top-k legal cells. CE = masked cross-entropy (nats).

| model | params | top-1 | top-3 | top-5 | CE |
|---|---|---|---|---|---|
| uniform (random legal) | — | 0.005 | 0.014 | 0.023 | 5.37 |
| adjacency-to-stones | — | 0.025 | 0.072 | 0.121 | 4.78 |
| **CNN** | **224k** | **0.386** | **0.597** | **0.696** | **2.26** |
| attention | 339k | 0.263 | 0.457 | 0.569 | 2.76 |

Wall: dataset gen **15 s** (CPU, niced), train+eval **1,541 s** on MPS (CNN early-stopped at ep8,
~13 s/ep; attention ran the full 20 ep at ~71 s/ep).

**Reading it:**
- **Feasibility = emphatic yes.** The CNN matches the *exact* strong-engine move ~39 % of the time
  and is within top-5 ~70 %, on games it never saw — **~15× the adjacency prior** at top-1 and ~6×
  at top-5. There is real, generalizing structure in how winners steer toward a forced win.
- **CNN > attention again, with fewer params** (top-1 0.386 vs 0.263). Same inductive-bias story as
  recognition: the next steering move is largely *local* (near the existing shape), which is exactly
  a conv's built-in prior; attention must learn locality from scratch.
- **But the attention comparison is not yet fair.** Attention was still **climbing at the epoch cap**
  (val top-1 0.066 → 0.253 monotonically through ep0→ep19, no plateau) — it is *undertrained*, not
  capacity-capped. More epochs/data would narrow the gap (the recognizer saw the same pattern).
- **CE corroborates the ranking** (CNN 2.26 < attention 2.76 < adjacency 4.78 < uniform 5.37): the
  nets are genuinely *calibrated over the move distribution*, not just lucky at the argmax.

## 4. What it means for the plan (and what this does NOT prove)

This is **Phase A** — the cheap offline behavioral-cloning go/no-go — and it passes. But be precise
about its reach:

- **It is a WEAK-BUT-HONEST proxy.** Top-1 move-match is *not* "good play": many quiet moves are
  reasonable, and exact-matching a strong engine is **not required** to steer well. Beating the dumb
  spatial priors by a wide margin on unseen shards is evidence of *learnable, generalizing steering
  structure* — it is **not** evidence that the resulting player is strong. BC also conflates
  "seeking a VCT" with "playing like a strong engine in general" (the target is the winner's actual
  move, not a move *labeled* as increasing VCT-reachability).
- **It does NOT settle the attention question.** The thesis bet ([vct-recognition-learnability.md](vct-recognition-learnability.md)
  §4) was that attention's *global* receptive field might matter for **sequential seeking** — a
  whole-game, whole-board problem. Next-move BC is local, so a conv winning here says little about
  that bet. The decisive test is **Phase C**.

**The next two phases (the real tests — GPU-spending, design-laden; gate with Jason):**
- **Phase B — oracle-labeled reachability.** Replace the imitation target with a *constructed* one:
  for each pre-onset position, use the oracle to score candidate moves by **VCT-reachability gain**
  (does a forced win appear within k plies after this move + best reply?). This is the *principled*
  seeker target rather than "imitate Rapfi," but it costs k-step batched oracle lookahead per
  candidate — a real GPU run.
- **Phase C — hybrid-play eval.** The decisive test: a player that **consults the oracle every ply**
  (attack *and* defense — for each candidate, batch-solve "does the opponent have a VCT after this?"),
  lets the **exact solver finish** whenever a VCT exists, and uses the **net only to steer in the
  tactically-quiet region**. Play it vs a **fixed baseline** (heuristic/lookahead — *not* sibling
  H2H, which is non-transitive) and measure strength. This is where attention's global-receptive-field
  bet is actually adjudicated.

## 5. Caveats / honest edges

- **Small + untuned, attention undertrained.** 200k of 459k train examples used, ≤20 epochs;
  attention had not plateaued. The go/no-go answer is clear; the *architecture* verdict for seeking
  is **not** decided here (deferred to Phase C).
- **Onset semantics.** "VCT" = provable within the miner's **500-node** budget; a deeper hidden VCT
  lands as `cap` and would only *lengthen* the quiet phase we sample (a few extra steering plies),
  never mislabel one. As conservative as the solver.
- **Seeker includes missed wins.** S = mover at onset, *not* necessarily the game winner. This is
  deliberate — "you reached a position with a forced win" is the steering target regardless of
  whether S then converted. (A `winner`-restricted variant is a cheap ablation if wanted.)
- **Both colors, both phases of the proxy.** Examples are pre-onset only; the tactical phase
  (`p ≥ onset`) is the oracle's domain by construction and is excluded from the net's training.

## 6. Artifacts

| Path | What |
|---|---|
| `scripts/threat_shapes/gen_seeker_dataset.py` | no-GPU steering-label builder (onset from miner verdicts; pre-onset same-parity moves) |
| `scripts/threat_shapes/train_seeker.py` | CNN + attention per-cell policies, masked CE, dumb baselines, held-out top-k; writes metrics |
| `~/data/puzzle_miner/seeker_exp/seeker_metrics.json` | full metrics (this table + config + disjointness proof) |
| `~/data/puzzle_miner/seeker_exp/{seeker_train,seeker_test}.npz`, `seeker_shards.json` | dataset + split/onset manifest |
| `~/data/puzzle_miner/seeker_exp/{seeker_cnn,seeker_attn}.pt` | trained checkpoints |

**Cross-links:** [vct-recognition-learnability.md](vct-recognition-learnability.md) (the recognizer
half + the seek-VCT thesis) · [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 (the oracle / L0) ·
[shape-library-engine.md](shape-library-engine.md) (L2 = the steering/AlphaZero layer this informs) ·
[vct-backward-mining.md](vct-backward-mining.md) §5 (move-extraction, resolved) ·
[allis-threat-theory.md](allis-threat-theory.md) (VCF/VCT formalism).
