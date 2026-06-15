# Engine-Panel-Anchored Derby — Calibrated Strength Yardstick

**Issue:** [#30](https://github.com/jasonyandell/gomoku/issues/30).
**Status:** Design. Depends on harness fix (merged) and
[gomocup-engines-catalog.md](gomocup-engines-catalog.md).

---

## The Problem: the Broken-Yardstick Wound

The 2026-06-15 session proved the lab had no trustworthy calibrated strength
number (§8–§12 of [alphazero-lessons-15x15-gomoku.md](alphazero-lessons-15x15-gomoku.md)):

- The Rapfi binary was the **weightless classical build**, not the rated ~3073 NNUE
  engine (§8A); "deep-TC" and "fast-TC" were the **same shallow engine twice** —
  Rapfi self-terminates at ~depth 10 / ~500 nodes regardless of time budget (§8C).
- A single self-built engine gives no absolute calibration and no tier diversity.
- **Head-to-head** vs a preserved reference is the cleanest relative gate (§8H),
  but gives only ordering — not an absolute number.
- **Self-play Elo** saturates at ~1700 and is non-transitive across recipes.

A fixed **panel of real external engines of known relative strength** is the
calibrated, diverse, non-saturating yardstick the campaign was missing.

---

## Panel Composition

From [gomocup-engines-catalog.md](gomocup-engines-catalog.md), in priority order:

| Tier | Engine | Notes |
|---|---|---|
| Native A | **Rapfi** (NNUE, 1 core) | Built; **pin to 1 core** (Gomocup single-thread rule) |
| Native B | **KataGomo** (#2, ~2879 Elo) | AlphaZero / KataGo fork; ARM build unverified |
| Native B | **AlphaGomoku(MK)** (#3, ~2781 Elo) | AlphaZero C++; ARM build unverified |
| Wine | Embryo26, Yixin2018, Pela23, Zetor17 | Run via wine; Embryo contends GPU |
| Internal | 128×10-e502, 64×4-e909, 96×8-seed | Preserved checkpoints for lineage anchoring |

Our prior multi-core Rapfi result (~62%) UNDERstated us — rules-legal single-thread
Rapfi is weaker.

---

## Calibrated-Elo Architecture

1. **Cross-table eval.** Each checkpoint plays color-alternated, balanced-opening
   (`--random-opening-moves 4`, #22) matches vs each panel member via `gomoku.match` /
   `external_engine.py` (hardened harness: RESTART before BOARD, bare `X,Y` move, merged).
2. **BayesElo anchor.** Anchor the cross-table to known Gomocup ratings to produce a
   **calibrated absolute Elo** per checkpoint (not just relative rank).
3. **Eval EMA weights.** Use `worker_weights.pt` — raw `epoch*.pt` ran 48 points weaker
   in the deepgen experiment (§8G).
4. **n ≥ 20 per pair, uncontended.** Small-n is brutally noisy (§4, §8B).

---

## CPU-Parallel-While-GPU-Trains: the Occupancy Lever

Gomocup engines are CPU + single-thread. A whole panel runs in parallel on the M5
Max's ~18 cores while the net trains on MPS — the "light all engines" occupancy lever
finally given a job:

- Panel eval = **everything-else lane** of the two-queue scheduler (parallel subagents).
- GPU-serial training slices continue; eval results land at slice-end without stalling.
- Embryo26 uses Vulkan (GPU): schedule in a GPU-idle window or label the contention.

---

## Integration with Existing Infrastructure

| Component | How it plugs in |
|---|---|
| **Δelo Derby** (`delo_derby.py`) | Panel Elo replaces broken Rapfi/saturated-anchor metric; recipes race by calibrated Δelo-rate |
| **Head-to-head gate** (§10) | Champion-promotion still requires color-alternated win vs preserved ref *before* panel eval |
| **Training slices** (`--max-wall-secs N --final-eval`) | Panel eval runs as the `--final-eval` bundle |
| **North-star Δelo/Δt** | Panel calibrated Elo finally gives Δelo a real absolute number |

---

## Lessons Baked In

- **Head-to-head gate** (§10): gate champion-promotion on match vs preserved reference; never trust plies/vl/internal-ladder alone.
- **Balanced openings** (#22): `--random-opening-moves 4`; first-mover advantage distorted black 85% / white 40%.
- **Single-thread**: pin Rapfi to 1 core; label multi-core runs as non-competition-fair.
- **Robust harness** (§12C–F): RESTART before BOARD, bare `X,Y` only, skips DATABASE/banner chatter; validated against ≥3 engines.
- **EMA weights** (§8G): `worker_weights.pt`, not `epoch*.pt`.
- **n ≥ 20 per pair**: weight trends across evals, not single reads.
- **Audit the yardstick first** (§8E): verify full strength, actual search, and reproducibility before a number becomes load-bearing.

---

## Build Steps

1. **Pin Rapfi to 1 core**; re-baseline champion vs NNUE Rapfi single-thread.
2. **Attempt KataGomo ARM build** (CMake, Eigen backend, no CUDA); record verdict.
3. **Attempt AlphaGomoku(MK) ARM build** (CMake, own ml layer); record verdict.
4. **Prototype cross-table eval**: champion + 2–3 nets vs panel → BayesElo calibration. Validate pipeline before wiring into the derby.
5. **Wire panel eval into derby loop** as the ranking metric.
6. **Drive the defense chapter**: track white-side improvement vs the panel (§12F: white 0–40% is the concrete weakness).

---

## Open Questions

- KataGomo / AlphaGomoku ARM buildability (may need Eigen backend or source patches).
- BayesElo anchoring: which Gomocup ratings to use; how to handle internal checkpoints not in the official ladder.
- Panel size vs wall-time: find the batch that completes inside a GPU training slice.
- Embryo26 GPU contention policy; wine-engine hang reliability (Eulring16 observed to hang).

---

## Related

- [alphazero-lessons-15x15-gomoku.md](alphazero-lessons-15x15-gomoku.md) §8–§12 — the broken-yardstick diagnosis
- [gomocup-engines-catalog.md](gomocup-engines-catalog.md) — engine landscape + buildability
- [external-engine-baselines.md](external-engine-baselines.md) — Rapfi wrapper + match harness
- GitHub #30 (this design), #22 (balanced openings), #28 (fix yardstick), #29 (capacity ladder)
