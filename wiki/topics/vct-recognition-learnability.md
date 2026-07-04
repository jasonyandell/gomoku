# Is-VCT recognition learnability — can a net *see* "you have a forced VCT"?

> **Status: LIVE** *(2026-07-04)* — feasibility result (trilogy 1/3).

**One-line finding.** A neural net **can** classify "does the side-to-move have a
forced VCT win?" straight from the raw 15×15 board and **generalize to games it has
never seen** (held-out, shard-disjoint, AUROC ≥ 0.92 — honest, not leakage). But for
these **local, translation-equivariant** threat shapes a same-scale **CNN beats a
transformer with half the params**, and even **logistic-regression on hand threat-counts
beats the attention model**. Recognition is *easy and count-dominated* — so it is **not
the bottleneck** and is better left to the exact oracle (L0). Attention's real audition
is the **seeker**, not the recognizer.

**Code:** `scripts/threat_shapes/gen_isvct_dataset.py` (no-GPU label builder — reuses the
miner's verdicts, no re-solve) · `scripts/threat_shapes/train_isvct_attn.py` (trains
attention + CNN + logreg + majority, evals on held-out shards, writes metrics).
**Builds on:** [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 (the exact oracle that
produced the labels) and the **forward every-ply puzzle corpus** `~/data/puzzle_miner/`
(`mine_puzzles.py`; see [shape-library-engine.md](shape-library-engine.md) §3 and
[vct-backward-mining.md](vct-backward-mining.md)). **Feeds:**
[shape-library-engine.md](shape-library-engine.md) **L2** (the AlphaZero/regression layer —
this is the first evidence about what a net can learn on verifiable VCT targets) and the
**seek-VCT thesis** (§4 below).

Date: 2026-06-26. Hardware: M5 Max, 48 GB, MPS (torch) for training; labels from MLX-Metal
oracle (no re-solve this run). This is a deliberately **small/untuned feasibility** test —
a go/no-go, not an architecture bake-off at scale.

---

## 1. The question and why it matters

The broader plan (§4) wants to **steer** play toward positions where a forced win exists,
then hand the tactical finish to the exact solver. A prerequisite worth checking cheaply:
can a net even *perceive* "a forced VCT is present here" from the board alone — and does
that perception survive on **unseen games** (not just unseen positions from seen games)?
If a net can't see it, nothing downstream works; if it can, we learn *which* architecture
fits the structure.

## 2. Data + the methodology that makes the number trustworthy

**Labels — reuse, no re-solve.** The forward puzzle miner already ran the oracle on
**every ply of every game**, so the labels already exist (Jason: *"null = no VCT found"*):
- **POSITIVE** = a `puzzles.jsonl.gz` row with `win=True & cap=False` (proven VCT; board
  stored as `atk`/`dfd`, side-to-move-relative).
- **NEGATIVE** = a ply that was **processed but is absent** from `puzzles.jsonl.gz`. "Processed"
  = the shard is in `manifest.txt` (fully-solved shards only — absence is meaningful only
  there). Negative boards are **CPU-replayed** in the exact same side-to-move frame as the
  stored positives.
- **EXCLUDE** = `cap=True` rows (unknown at the miner's 500-node budget — not a clean label).

This made the experiment **light** (CPU replay + MPS training, no GPU solve), so it ran
**without competing** with the live `collect_rapfi` game producer.

**Shard-disjoint split — the load-bearing correctness fact.** Consecutive plies of one game
differ by a single stone, so a position-level split would leak a near-duplicate of every
test board into train and report a *fake-high* score. Split is **by shard** (`md5(shard)%10`):
400 manifest shards → **367 train / 33 test, overlap = 0**. A further **49-shard val** set
(carved from train) drove early-stopping, so the stopping signal didn't leak either.

| split | shards | positions | pos rate |
|---|---|---|---|
| train | 367 | 1,167,002 | 17.3% |
| test (held-out) | 33 | 101,745 | 14.2% |

Training was class-balanced (30k/class = 60k); the held-out set is reported at **both**
balanced and the **natural 14.2%** base rate. Negative-board reconstruction was cross-checked
against stored positive boards: **0 frame mismatches across all 400 shards.**

## 3. Result — yes-learnable; attention is the laggard

Held-out (33 disjoint test shards, n=101,745, 14.2% positive). **AUROC is the fair,
threshold-independent comparison.**

| model | params | AUROC | acc (natural) | precision | recall | F1 |
|---|---|---|---|---|---|---|
| majority (predict 0) | — | 0.500 | 0.858 | 0.000 | 0.000 | 0.000 |
| logreg on threat-counts | tiny | 0.946 | 0.883 | 0.557 | 0.870 | 0.679 |
| **CNN** | **168k** | **0.971** | 0.936 | 0.761 | 0.797 | 0.779 |
| **attention** | **339k** | **0.924** | 0.857 | 0.498 | 0.807 | 0.616 |

(Balanced-test tells the same story: CNN 0.971, logreg 0.948, attention 0.924.) Wall:
dataset gen **20 s** (CPU), train+eval all four models **356 s** (MPS).

**Reading it:**
- **Feasibility = emphatic yes.** Every learner clears AUROC 0.92 vs the 0.5 floor, and
  attention's val→test gap is tiny (0.933 → 0.924) — real generalization across unseen
  shards, not leakage.
- **The CNN wins with *half* the params (168k vs 339k).** So it's not "bigger model won" —
  it's the **right inductive bias**: VCT structure is local and translation-equivariant (a
  four is a four anywhere), which is exactly a conv's built-in prior. Attention with learned
  position embeddings must *learn* locality from scratch.
- **Even logreg-on-counts (0.946) beats attention (0.924).** The signal is heavily carried
  by simple countable threats. At the default 0.5 threshold the attention model's accuracy
  (0.857) is **indistinguishable from always-predicting-no** (0.858) — it's FP-heavy and
  miscalibrated; only its *ranking* (AUROC) is informative.

## 4. What it means for the plan (the seek-VCT thesis)

The framing this experiment serves (Jason, 2026-06-26): **don't search toward 5-in-a-row;
seek a VCT and insta-solve it when you arrive.** The split works because strategy and
tactics have **anti-correlated tractability**:
- *Positional/strategic play* — intractable to **solve**, but **tolerant of approximation**
  (a slightly-wrong quiet move rarely loses on the spot) ⇒ give to the **net**.
- *Tactical/forcing finish* — **intolerant** of approximation (one wrong move and the forced
  win evaporates) but **tractable to solve exactly** (the forcing constraint collapses the
  tree — that's why the oracle does thousands/s) ⇒ give to the **solver**.

In play: **consult the oracle every ply** (attack *and* defense — for each candidate move,
batch-solve "does the opponent have a VCT after this?"); the net only ever steers in the
**tactically-quiet region** — precisely where being slightly wrong is survivable.

**So this result doesn't dent the plan — it clarifies it.** Recognition was always going to
be the oracle's job (exact, cheap). This experiment confirms that *if* you ever want a
learned recognizer (e.g. to prune before solving), a **CNN or even logreg** is the better,
cheaper choice — **not** attention. Attention's interesting bet is the **seeker** (a global,
sequential problem where whole-board receptive field may matter, and where this recognition
result does **not** predict the outcome). Recommended next experiment: the seeker, not a
fairer recognizer rematch. **(DONE 2026-06-26 — Phase A:
[seeker-steering-learnability.md](seeker-steering-learnability.md): the quiet-phase steering IS
learnable on unseen games — held-out top-1 0.386 / top-5 0.696 vs adjacency 0.025/0.121. CNN again
beats attention on *next-move* BC, but that proxy is local so it does **not** yet adjudicate the
global-receptive-field bet — that waits for the hybrid-play eval, Phase C.)**

## 5. Caveats / honest edges

- **Small + untuned, and attention is the most data-hungry of the three.** Trained on only
  **60k balanced** examples (of 1.17M available), 339k params, ≤12 epochs. Attention was
  still inching up at the last epochs — more data/capacity would likely narrow the CNN gap,
  but the go/no-go answer is already clear, and the CNN already owns the recognition role.
- **Label semantics.** "No VCT" = "no VCT provable within the miner's **500-node** budget"
  — the same oracle/budget applied consistently to both classes; deeper hidden VCTs land as
  `cap` (excluded), not mislabeled negatives. As conservative as the solver itself.
- **"Every ply" includes trivial early-game negatives** (near-empty boards) — this inflates
  natural-base-rate *accuracy*; the **AUROC / balanced** columns are the cleaner read.

## 6. Artifacts

| Path | What |
|---|---|
| `scripts/threat_shapes/gen_isvct_dataset.py` | no-GPU label builder (reuse miner verdicts; absence = negative) |
| `scripts/threat_shapes/train_isvct_attn.py` | trains attention + CNN + logreg + majority; held-out eval; writes metrics |
| `~/data/puzzle_miner/isvct_exp/isvct_metrics.json` | full metrics (this table + config + disjointness proof) |
| `~/data/puzzle_miner/isvct_exp/{isvct_train,isvct_test}.npz`, `shards.json` | dataset + split manifest |
| `~/data/puzzle_miner/isvct_exp/{isvct_attn,isvct_cnn}.pt` | trained checkpoints |

**Cross-links:** [gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 (the oracle / L0) ·
[shape-library-engine.md](shape-library-engine.md) (L2 = the AlphaZero layer this informs) ·
[vct-backward-mining.md](vct-backward-mining.md) §5 (the move-extraction gap, now resolved) ·
[allis-threat-theory.md](allis-threat-theory.md) (VCF/VCT formalism).
