# Cross-game value sidecar — the lever, and why it kept failing under live flooding

## What it is
The cross-game value aggregation **"position-stats sidecar"**: blend a position's
single-game value target `z` with a low-variance aggregate of outcomes across ALL
self-play games that transposed through that position.

Motivation: single-game ±1 punishes a good move for a blunder 10 plies later → the
net sees contradictory labels for the same position → mushy ~0 values → no killer
instinct, slow opening convergence. Aggregating across games de-noises the credit
assignment → confident values. The opening is the most-transited region, so it gets
the richest transposition counts and should sharpen first.

Opt-in (`--cross-game-value`), **trainer-owned single-writer** store, lane-isolated,
**byte-identical when OFF**. Cell `derby-x-crossgame` = clone of
`derby-v7-mate-discount` + the cross-game flags. Code: `gomoku/position_stats.py`
(canonical key + store) and `gomoku/train.py` (ingest / relabel-on-sample).

## The friction: three principled fixes, three live-load failures
Beads `derby-eft` → `derby-eda` → `derby-4bq` (with reopens). **Each fix passed its
CPU tests and still failed the derby runner's LIVE re-race**, because per-epoch
ingest cost scales with self-play **inflow**, and the CPU tests didn't replicate the
live flooding regime.

| Round | Bead | Fix | Why it still failed live |
|---|---|---|---|
| impl | `derby-eft` | the sidecar | recency-decay traversed the whole store every cycle — O(store) |
| 2 | `derby-eda` | lazy global decay `scale` → O(1) decay | `save()→_renormalize()` still folded the scale over every entry every epoch (O(store)); store grew unbounded (116k→234k, 14MB); epoch wall 14s→128s |
| 3 | `derby-4bq` | cap store to opening plies (`ply<10`) + decouple renormalize from save | bounded the STORE + save cost, but the per-position canonical keygen still runs on EVERY ingested position *before* the ply filter → keygen dominates under flooding (flat ~10×) |

The runner pulled `derby-x-crossgame` **twice** and hardened its own skill to *"verify
a re-raced fix at FULL load (epoch 50+), not early"* (commit `7ec7637`).

## Root cause (round 3, confirmed in the code — not a sim)
`train.py` per-epoch ingest calls `canonical_key_from_planes(e.planes)` for **every**
newly-ingested example, and the `ply < max_ply` filter is applied **after**, inside
`add()`. The keygen (`canonical_key_from_board`) is pure-Python: min over 8 D4
symmetries × an 81-cell big-int trit-pack ≈ **~650 Python ops/position**. Under
flooding the inflow is large → the keygen dominates per-epoch ingest. The store cap
(`derby-4bq`) bounds storage + save cost but not the keygen-over-inflow cost.

## Workaround / fix (round 4 — in flight 2026-05-27, bead `derby-eda` reopened)
1. **Ply-gate the keygen**: compute the key only for `ply < max_ply` positions (cheap
   int check *before* the expensive keygen).
2. **Vectorize `canonical_key`** with numpy (find the lex-min orientation by comparing
   trit arrays, pack the winner once) — guarded by a **property test proving
   byte-identical keys** vs the scalar reference over random boards + all 8 symmetries
   (a wrong key silently corrupts value targets).
3. **Flood-scale regression test**: per-epoch ingest cost stays flat as inflow grows.

If round 4 misses the live re-race too, the lever gets **shelved** pending design
attention (the code stays, OFF by default and byte-identical).

## The meta-lesson (the durable bit)
**CPU-sim / unit-test validation of a training-LOOP ingest path does NOT capture the
live FLOODING regime.** Per-epoch ingest cost is **inflow-bound**; a toy benchmark (a
few thousand positions, a small opening alphabet) under-counts it. Three fixes
"passed CPU tests" and still regressed live. Corollaries:

- **Validate ingest/perf changes under realistic flood inflow**, and treat the derby
  runner's **full-load live re-race (epoch 50+)** as the only true gate.
- **Cheap filter before expensive per-item op** (ply-gate before keygen).
- **A hand-rolled per-position hash in a hot ingest loop is a perf liability** —
  vectorize it or it bites under inflow.

Sibling of [[perf-bench-vs-real-training-cost]] and the perf-bench-lesson memory
(single-process benches under-count production parallelism). Same shape: a number
measured outside the production regime lied.
