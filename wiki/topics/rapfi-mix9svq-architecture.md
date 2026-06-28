# Rapfi `mix9svq` — exact NNUE architecture (the engine that made our corpus)

**What this is.** A source-grounded teardown of the evaluator inside the
`pbrain-rapfi` binary we run as the native NNUE yardstick and as the self-play
generator behind `~/data/games_raphi` (the ~1.19M-game Rapfi-vs-Rapfi corpus).
Every constant and code path below is read from the local checkout at
`.rapfi-build/rapfi/Rapfi/` — primarily `eval/mix9svqnnue.{h,cpp}`,
`game/pattern.*`, and `search/ab/`. The config that selects it is
`engines/rapfi/config.toml` → `[model.evaluator] type = "mix9svq"`, weights
`mix9svq*freestyle/standard/renju*.bin.lz4`.

**One-line takeaway.** "Rapfi does great search over an NNUE" undersells it: the
evaluator is a **quantized line-shape CNN** (efficiently-updatable input layer +
a real 2-D conv trunk), and — the part that matters for the seek-VCT thesis —
**the network never confirms a forced win. Threat confirmation (VCF/VCT) is pure
search and does not call the network at all.** Cheap approximate recognition,
expensive exact confirmation, cleanly separated. See
[vct-recognition-learnability.md](vct-recognition-learnability.md),
[seeker-steering-learnability.md](seeker-steering-learnability.md),
[the-claw.md](the-claw.md).

## Pipeline at a glance

```
board → per-cell line shapes (4 dirs, 11-cell window) → VQ codebook lookup
      → sum over 4 dirs = mapSum[64]                         (the EU accumulator)
      → 3×3 depthwise conv → mapConv[32]                     (spatial mixing trunk)
      → value head:  3×3 group-pool → "star blocks" → MLP → (win, loss, draw)
      → policy head: hypernet-generated 1×1 conv → per-cell logit
```

Key dims (`eval/mix9svqnnue.h`): `FeatureDim=64`, `FeatDWConvDim=32`,
`ValueDim=64`, `PolicyDim=32`, `PolicyPWConvDim=16`, `ShapeNum=442503`,
`NumHeadBucket=1`, `ArchHashBase=0x84a071fe`.

## 1. Input — line shapes → a vector-quantized codebook

Per cell, per direction, an **11-cell window** (radius 5; `length=11, half=5`).
Four directions `DX={1,0,1,1}, DY={0,1,-1,1}` (H, V, anti-diag, diag).

**Ternary base-3 shape index.** Each of the 11 positions is empty/black/white;
the shape is a base-3 integer built incrementally — placing a stone adds
`(color+1)·3^(dist+5)` (`dPower3 = pieceColor+1`: black→1, white→2). Board edges
are folded in via a "border encoding" (off-board cells get their own codes),
which is why the raw shape space is **442503**, not a clean 3¹¹ = 177147: it's the
on-board patterns plus all edge-truncated variants (max border encoding 441774 +
a 3⁶ = 729 buffer = 442503).

**The VQ trick (the "vq" in mix9svq).** A dense embedding table would be
442503 × 64 × int16 per direction-group. Instead:

```
codebook[2][65536][64]      // 2 shared dictionaries of 65536 learned 64-dim vectors
mapping_index[2][442503]    // each raw shape → which codebook entry (uint16, learned/frozen)
```

So **442503 distinct line shapes are clustered onto ≤65536 learned prototype
vectors.** `mapping_index` is the assignment, `codebook` is the dictionary; the
`.lz4` weights pack the codebook at **10-bit precision** with sign-extension on
load (the scalar-quantization half of "svq"). That's how a 442k-shape vocabulary
fits in a ~10 MB file.

**The "2":** `mappingIdx = dir/2` — H+V share codebook 0, the two diagonals share
codebook 1. Axis-aligned vs diagonal lines get separate dictionaries (sensible:
diagonal threats differ geometrically from orthogonal).

**The accumulator** `mapSum[cell][64]` = sum of the 4 directions' codebook
vectors. This is the NNUE accumulator.

## 2. Efficient update — why it is cheap to call inside search

`Accumulator::move()` is the EUNN core. A stone only touches the ≤11 cells along
each of 4 directions within radius 5 (`x0=max(x-5,1) … x1=min(x+5,boardSize)`),
doing the classic add/subtract:

```
newFeat = oldFeat − codebook[old_shape_vec] + codebook[new_shape_vec]
```

**Versioning, not mutation** (reducer-shaped, append-only — the design Jason
likes): instead of in-place edits it keeps version-indexed snapshot tables
(`versionInnerIndexTable`, `versionOuterIndexTable`, `versionChangeNumTable`), so
`undo()` is just `currentVersion--`. The accumulator is a log of versions.

## 3. Trunk — 3×3 depthwise conv

`feature_dwconv_weight[9][32]`: the "9" is a flattened 3×3 kernel (indexed
`8 - dy*3 - dx`), depthwise over 32 channels, stride 1. This is why the board is
padded to `(boardSize+2)²` ("outer" board) — the conv needs a one-cell halo.
Output `mapConv[32]` per cell. This step makes it a genuine **spatial** network
(adjacent cells' shape-features mix), not a per-cell bag-of-shapes.

## 4. Value head — group-pool → star blocks → (win, loss, draw)

- Features pooled into a **3×3 grid of groups** plus a global sum.
- Each group runs a **StarBlock**: two parallel linear branches → clipped-ReLU →
  **element-wise multiply** (`dot2`) → down-projection. The multiply is a cheap
  quadratic interaction (StarNet-style "star operation") — lets the value head
  model "threat A *and* threat B" conjunctions a purely additive MLP cannot.
- 3×3 groups → 2×2 quadrant average-pool → another StarBlock (`value_quad`) →
  concat with global → 3-layer MLP (`value_l1/l2/l3`) → **4 outputs scaled by
  1/16384**, read as (win, loss, draw).
- Quantization: int8 activations, per-stage clipped-ReLU scales (256 global,
  32 group, 128 in the MLPs).

**Search nuance:** the NNUE value is not always used. Search runs classical
pattern eval first and only swaps in the network value when the classical
estimate is within a margin of the αβ window (`eval/eval.cpp`). The net is the
accurate-but-expensive refinement *near the decision boundary*; classical eval
handles blowouts cheaply.

## 5. Policy head — a *dynamically generated* 1×1 conv

The policy filter is not fixed weights; it is generated per-position from the
global feature (a hypernetwork):

1. global feature → `policy_pwconv_l1` → `policy_pwconv_l2` emits **16×32 weights
   + 16 biases** on the fly.
2. Those dynamic weights run as a 1×1 conv at every cell over its 32-dim
   `mapConv` → 16 ch → ReLU → dot with `policy_output_weight[16]` (float) → **one
   logit per cell.**

So the policy "filter" is conditioned on a whole-board summary. Output feeds move
ordering (below).

## 6. Division of labor — the seek-VCT payoff

| Job | Done by |
|---|---|
| Move ordering | **Network policy** (logits + main-history + counter-move, softmax-normalized; `search/movepick.cpp`) |
| Leaf value near the αβ boundary | **Network value** (else classical eval) |
| Candidate restriction | **Pattern geometry** (`default_candidate_range = square3_line4`), *not* policy |
| **Threat confirmation (VCF/VCT)** | **Pure search** — `vcfsearch`/`vcfdefend` (`search/ab/search.cpp`, entered at `depth ≤ 0`) over forcing sequences, using *classical* eval; **does not call the network** |

Policy also drives policy-based pruning (~10 elo) and LMR reduction (~59 elo) at
shallow depths (`search/ab/search.cpp`). Main search is alpha-beta / PVS with
iterative deepening, aspiration windows, a transposition table, and Lazy SMP.

**Why this matters here.** Rapfi commits *architecturally* to the same
anti-correlated-tractability split the seek-VCT thesis circles: recognition (the
quantized shape network) is ~free and approximate; confirmation (VCF/VCT search)
is exact and expensive; the engine never *trusts* the net for a forced win, it
confirms by search. The seek-VCT bet is essentially "move more of the
confirmation burden into recognition" — Rapfi is the reference point that says
today's strongest engines deliberately *don't*. Also note the line-organized
input layer is exactly what [the-claw.md](the-claw.md) argues is provably blind
to the knight's-move defensive crystal: the blindness is in this codebook's
line-shape vocabulary, not in the search.

## Provenance

Read 2026-06-28 from `.rapfi-build/rapfi/Rapfi/` (Gomocup 2024+2025-winning
Rapfi-NNUE, native arm64 NEON-DOTPROD build). Constants from
`eval/mix9svqnnue.h`; forward pass + incremental update from
`eval/mix9svqnnue.cpp`; shape encoding cross-checked against `game/pattern.*`;
search integration from `search/ab/search.cpp`, `search/movepick.cpp`,
`eval/eval.cpp`. Weight-blob internals (exact 10-bit codebook layout) inferred
from the loader, not unpacked. Yardstick context:
[external-engine-baselines.md](external-engine-baselines.md).
