# Engine-Panel-Anchored Derby — Calibrated Strength Yardstick

**Issue:** [#30](https://github.com/jasonyandell/gomoku/issues/30).
**Status:** Design → tooling BUILT → **first run DONE (2026-06-15); calibration
BLOCKED on engine reliability + anchor validity ([#35](https://github.com/jasonyandell/gomoku/issues/35)).**
Depends on harness fix (merged) and
[gomocup-engines-catalog.md](gomocup-engines-catalog.md).

> **FIRST RUN — partial failure, and the failure is the finding (2026-06-15).**
> The first 9-player round-robin ran via `scripts/panel_tournament.py`. Of 36 pairs,
> **only 19 played; 17 ERRORED** — every failure an *opponent engine dying* (Embryo
> timing out under GPU/Vulkan contention; Zetor crashing on back-to-back reuse). Our
> brain-wrapped nets produced **zero errors**. Worse, the **calibration came out with
> a NEGATIVE slope (~−0.07)**: under wine + single-thread + 10s/move the engines do
> NOT play at their published multi-thread ratings — **yixin18 (published ~2310) went
> 0–30, even 0–6 to the heuristic; pela23 (~1499) went 24–6** — so published Elos are
> **invalid anchors** and the reader (`panel_white_elo.py`) correctly *refuses* to
> print a calibrated Elo (degenerate-slope → relative fallback, loudly flagged).
> **No calibrated yardstick yet; #35 tracks the fixes.** What IS trustworthy: the
> completed games (net-vs-net, net-vs-heuristic) and the **white-side defense gap**
> (champion 94% black vs 50% white, +44pp — the #33 next target). Full write-up:
> [lessons §14](alphazero-lessons-15x15-gomoku.md).
>
> **Concrete reliability fixes before calibration (#35):**
> - **Per-engine timeout** — a flat 30s budget killed Embryo (GPU-contended, slow to
>   reply); set the move/reply timeout per engine, and schedule Embryo only in a
>   GPU-idle window (or drop it from the calibration panel).
> - **Process-per-pair (no engine reuse)** — Zetor crashes (`engine process has
>   exited` / `EOF`) on back-to-back reuse across pairs; spawn a **fresh engine
>   process for every pair** and tear it down after, so one pair's death can't
>   poison the next.
> - **Measure-don't-assume anchors** — do NOT fit to published Gomocup Elos; they are
>   multi-thread tournament ratings invalid under our wine/single-thread/10s harness.
>   **Empirically measure each engine's effective strength under our exact harness**
>   (e.g. a dense internal round-robin) and anchor to *that*, or report **relative**
>   internal Elo only and stop claiming a Gomocup-absolute scale.

> **BUILT (2026-06-15).** The two pieces this design needs now exist on `main`:
> - **The cross-table runner** — `scripts/panel_tournament.py` (#32, commit
>   `0fb7fc1`): calibrated round-robin cross-table (Bradley-Terry + affine
>   calibration to real engines' published Gomocup Elos). This *is* the
>   "Calibrated-Elo Architecture" below, implemented.
> - **The brain wrapper** — `gomoku/gomocup_brain.py` (#31, commit `1834df0`):
>   our net answers the Gomocup/Piskvork protocol on stdin/stdout, loadable by
>   checkpoint path, registerable as a first-class panel engine via
>   `external:cmd=run-gomoku-az --checkpoint X --sims N` (no zip packaging).
>
> **REQUIRED: register every net with `incremental=1`.** Our net is
> history-conditioned (`HISTORY_PLY=8` recency planes). Driven by plain
> BOARD-replay every move it rebuilds an **empty-history** input and silently
> sandbags itself (measured 100% → 25% on a fixed checkpoint —
> [lessons §13](alphazero-lessons-15x15-gomoku.md)). `incremental=1` drives it via
> `TURN` so move history accumulates faithfully. So a panel registration is
> `external:cmd=run-gomoku-az --checkpoint X --sims N,...,incremental=1`. Classical
> engines (no history planes) keep the default BOARD path.

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
   **calibrated absolute Elo** per checkpoint (not just relative rank). **⚠ The first
   run (2026-06-15) proved published Gomocup ratings are INVALID anchors under our
   wine/single-thread/10s harness** (negative-slope fit; yixin18 ~2310-published went
   0–30). Anchor instead to **empirically measured effective strengths under our exact
   harness**, or report relative-only Elo. See the FIRST RUN note above and #35.
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
- BayesElo anchoring: ~~which Gomocup ratings to use~~ — **RESOLVED (negatively):
  published Gomocup Elos are invalid anchors under our harness (#35); measure
  effective strength under our exact conditions instead.** How to handle internal
  checkpoints not in the official ladder remains open.
- Panel size vs wall-time: find the batch that completes inside a GPU training slice.
- **Engine reliability (now the gating blocker, #35):** Embryo26 GPU/Vulkan
  contention (timed out 30s in the first run); Zetor17 crashes on back-to-back reuse;
  Eulring16 / wine-engine hang reliability. Per-engine timeouts + process-per-pair are
  the planned fixes.

---

## Related

- [alphazero-lessons-15x15-gomoku.md](alphazero-lessons-15x15-gomoku.md) §8–§12 — the broken-yardstick diagnosis
- [gomocup-engines-catalog.md](gomocup-engines-catalog.md) — engine landscape + buildability
- [external-engine-baselines.md](external-engine-baselines.md) — Rapfi wrapper + match harness
- GitHub #30 (this design), #22 (balanced openings), #28 (fix yardstick), #29 (capacity ladder)
