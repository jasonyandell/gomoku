# Sliding Derby — design v1 (issue #38, 2026-06-16) — reuse-ledger only

> **Status (2026-07-04): SUPERSEDED-BY([sliding-derby-measured-outcomes-design-v2.md](sliding-derby-measured-outcomes-design-v2.md)); reduced to the reuse-ledger.**
> The built methodology of record is v2. This page has been **trimmed to the two
> parts worth keeping** — the grep-verified infra reuse map (§0) and the
> reuse-vs-net-new ledger (Appendix) — per its own original "keep as the
> reuse-ledger" instruction. The full v1 design (§§1–7: architecture, the
> frozen-reference gate, GPU-contention model, priority engine, state/resumability,
> the adversarial wedge-check, and the MVP checklist) is preserved **verbatim** at
> [../_archive/topics/sliding-derby-design-v1-full.md](../_archive/topics/sliding-derby-design-v1-full.md).
> The autonomous Derby is stopped (see [../derby.md](../derby.md)).

> **One-line thesis (retained):** the Sliding Derby is `delo_derby.py`'s DNA
> (Δelo-rate priority, leaders-first, atomic board-json, `snapshot_peak`) re-wired
> from a *synchronous parallel scheduler* into an *async single-track pipeline*
> whose only net-new load-bearing parts are (a) a **frozen-reference gate wrapper**
> and (b) a **non-blocking cadence watcher (#34)**. Everything else is reuse — which
> is what the two tables below document.

---

## 0. Ground-truth verification (what actually exists, file:line)

Before designing on top of the grounding reports, the load-bearing claims were
checked against source. All confirmed:

| Claim | Verified at | Verdict |
|---|---|---|
| Lap engine: `run_sweep --max-wall-secs N --final-eval`, trainer self-caps on epoch boundary, clean resumable save, no cold restart | `scripts/run_sweep.py:2086` `launch_cell(... max_wall_secs, final_eval ...)`; supervise loop force-saves on cap + `TRAINER_CAP_GRACE_SEC` | **REAL** — lock-1 "training never stops" already satisfied by the cap-then-resume seam |
| Training-time internal eval is OFF by default (must stay off in lap configs) | `run_sweep.py` `launch_cell`: `internal_eval` opt-in, prints "internal-baseline eval DISABLED" | **REAL** — #34's "drop training-time evals" is already the default |
| Gate executor: model-vs-model paired-opening H2H, tight CI, no anchor ceiling | `scripts/delta_e_harness.py:687` `head_to_head_eval(...) -> HeadToHeadResult` with `wins/draws/losses`, `win_rate`, `delta_elo`, `delta_ci_half`; runs on CPU across `n_workers` | **REAL** — this is the gate's measuring instrument |
| Frozen peak: atomic `latest.pt → peak.pt` copy, pruning-proof | `scripts/delo_derby.py:616` `snapshot_peak(board, idea, elo)` → `os.replace(tmp, peak.pt)` | **REAL** — the freeze *primitive* exists |
| **freeze-peak-live / §10 / "AGZ anti-collapse brake" named in #38** | `grep -rniE 'freeze.?peak\|section.?10\|anti.?collapse\|frozen.?reference'` over `wiki/ scripts/ TRAINING_WIKI.md` → **ZERO hits**; `white-side-defense-plan.md` has §0–§5 + appendix, **no §10** | **DOES NOT EXIST** — the gate ORCHESTRATION is net-new |
| Async instruments: resumable panel JSONL (`--only` skips completed pairs), pure-analysis white-Elo (no GPU/torch) | `scripts/panel_tournament.py:122` `incremental=1` net specs; `scripts/panel_white_elo.py:380` `compute_dual_elo` → `black_elo/white_elo/elo_gap`, `white_loss_rate = white_l/white_games` (line 43) | **REAL** |
| N-way wrapper | `scripts/round_robin.py:26` imports `head_to_head_eval`, mean-centered ranking | **REAL** |
| Concurrent two-full-runs = ~half speed per run, aggregate flat | `wiki/topics/m5-max-cross-engine-coupling.md:155,168`: gen −45%/SGD −55% per run; aggregate gen +8%/SGD −9% | **REAL but for 20-proc dual** — the eval co-run is much lighter; lock-1 still needs its own measurement |
| Panel anchors invalid (#35): 17/36 pairs crash, negative affine slope | issue #35 body; `yixin18 0-30`, `Elo = -0.07*internal + 1876.9` | **REAL** — gate must NOT depend on calibrated absolute Elo |

**Design consequence of the last row:** the gate is built on the two
*calibration-free* signals — `head_to_head_eval` win-rate vs the frozen peak
(relative, tight CI) and `white_loss_rate` (a raw count ratio, immune to the
broken affine fit). Calibrated absolute Elo is reported but is **advisory only**
until #35 is fixed.

---

> **§§1–7 archived.** The full v1 design body — architecture / state machine, the
> frozen-reference gate (the promotion rule + why it makes silent regression
> un-shippable), the GPU-contention model, the ELPL priority engine, the
> state/resumability contract, the adversarial 5-wedge check, and the MVP checklist
> — is preserved verbatim at
> [../_archive/topics/sliding-derby-design-v1-full.md](../_archive/topics/sliding-derby-design-v1-full.md).
> The gate mechanics that survived into the build are re-specified in
> [sliding-derby-measured-outcomes-design-v2.md](sliding-derby-measured-outcomes-design-v2.md).

---

## Appendix — the reuse-vs-net-new ledger (one glance)

| Piece | Status | Where |
|---|---|---|
| Time-capped lap (no cold restart) | REUSE | `run_sweep.py:2086` `launch_cell(max_wall_secs, final_eval)` |
| Training-time eval OFF by default | REUSE | `run_sweep.py` `launch_cell` internal_eval opt-in |
| Gate match executor (H2H, tight CI) | REUSE | `delta_e_harness.py:687` `head_to_head_eval` |
| N-way field | REUSE | `round_robin.py:26` |
| Atomic frozen-peak primitive | REUSE | `delo_derby.py:616` `snapshot_peak` |
| Resumable panel slice (skip done pairs) | REUSE | `panel_tournament.py:122` `--only`, `incremental=1` |
| White-side reader (CPU, no torch) | REUSE | `panel_white_elo.py:380` `compute_dual_elo`, `white_loss_rate` (l.43) |
| Δelo-rate priority math, board rewrite | REUSE | `delo_derby.py` `pick_priority`/`delo_per_hr`, `write_research_board:642` |
| Board-json + `_peak/` state pattern | REUSE | `derby_v9_board.json` shape |
| Cron/watchdog autonomy loop | REUSE | `gomoku-derby-runner` 10-min check model |
| **Frozen-reference promote/revert wrapper** | **NET-NEW** | `scripts/sliding_gate.py` |
| **Non-blocking cadence watcher (#34)** | **NET-NEW** | `scripts/eval_cadence_watcher.py` |
| **Orchestrator state-machine + ELPL re-ranker** | **NET-NEW** | `scripts/sliding_derby.py` |
