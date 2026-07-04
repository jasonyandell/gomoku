# Training timeline — the curated read-path over TRAINING_WIKI

The append-only [TRAINING_WIKI.md](../TRAINING_WIKI.md) is ~5,800 lines of
chronological evidence. This page is the **milestone index** into it: the ~50
dated turning points, grouped by era, each with its W&B run id. Read a row, then
jump to that date in the notebook for the full evidence. Hub column = where the
durable lesson is synthesized.

*(Curated 2026-07-02 from a full-notebook extraction pass. Provenance:
recorded — every row traces to a dated TRAINING_WIKI entry.)*

## Era 0 — 9×9 origins & the fast-attack-collapse diagnosis (May 17–19)

| Date | Hub | Milestone |
|---|---|---|
| 2026-05-17 | alphazero | **Origin** 9×9 run (`o9npssu1`) collapsed to defensive draws by e136; baseline **gen dominates train 25–30×**. |
| 2026-05-18 | alphazero | Distributed self-play workers (file-handoff, zeb-modeled) give **5–6× speedup** — multi-process is the real unlock (single-process GIL-bound); collapse persists regardless of throughput. |
| 2026-05-18 | alphazero | 13×13 medium (`qx69005o`) diagnosed **pure-offense/zero-defense** — the self-play curriculum gap (neither side builds/faces mature threats). |
| 2026-05-18 | alphazero | Jason's leading indicator: **buffer fill-curve concavity** (slope = games/cycle × avg-plies) predicts fast-attack collapse before eval confirms it. |
| 2026-05-19 | experiments | Sweeps **refute** every proposed upstream cause: K, buffer-size, games-per-cycle, continuous-gen all FAILED; `pacifist_blocker` partner useless. |
| 2026-05-19 | alphazero | Correction: `kze1lcti` e85 "crossing" was **n=4 eval noise** — sibling H2H non-transitive; always benchmark vs a fixed external opponent. |

## Era 1 — the AZ-recipe breakthrough (May 19–23)

| Date | Hub | Milestone |
|---|---|---|
| 2026-05-19 | alphazero | Ported the **michaelnny AZ recipe** (tau_final=0.1 soft targets, buffer 50k→1.5M, AGZ log-PUCT); horizon calibration shows prior "collapses" were 20–30× undertrained. |
| 2026-05-19 | alphazero | **BREAKTHROUGH:** `sppjo3z5` (az-recipe-160k) is the FIRST run to sustain heuristic (70%) then lookahead2 (55%) — real defense learned (plies regrow 11→27–32). |
| 2026-05-20 | reference | Calibrated true baseline Elos via n=50 round-robin (heuristic≈591, la2≈604, la4≈629, la5≈711). |
| 2026-05-20 | alphazero | `sppjo3z5` stopped e5000: peak **model_elo=1718** @e3881; recovered 5-for-5 (refuted "arcs shrink asymptotically" and "eventually won't recover"). |
| 2026-05-23 | alphazero | **LF1 correction** (`h9al2e0k`): perf-lab's +152% was *generation* throughput (cold-buffer transient); once the buffer fills, training cost explodes to ~3 min/epoch. |

## Era 2 — WL wave-of-lockstep series (May 20–22)

| Date | Hub | Milestone |
|---|---|---|
| 2026-05-21 | alphazero | WL1/WL2 raise the ceiling (la4 52→62%) but per-version uniformity does NOT fix retention — all versions share one opening lineage. |
| 2026-05-21 | reference | WL3 crashed e825 on a **native-MCTS NaN cascade** (policy(tau) pow overflow at tau=0.1); fixes: double-precision normalization + pi-sanitization + sample guard. |
| 2026-05-21 | alphazero | WL3.1 (`44cxzc9d`) strongest WL state (la4 95%, heuristic 100%); WL4 K-decay → **WL-series ATH elo=1841** @e2401. Random-opening diversity is necessary-but-not-permanent. |
| 2026-05-22 | alphazero | WL5 (`o6cbjfnr`, archive-start) closes e10200: ATH not broken; buffer undersized, 20-game evals ±100 elo. |

## Era 3 — 9×9 frontier closed → 15×15 port & the white wound (Jun 12–19)

| Date | Hub | Milestone |
|---|---|---|
| 2026-06-12 | reference | **9×9 frontier CLOSED:** v8 champion vs Rapfi (2625 elo) 43W-3L-74D/120; codebase ported to parameterized board size (15×15 ~free), SMOKE15 GO. |
| 2026-06-15 | alphazero | **CRITICAL CORRECTION:** the 96×8 15×15 "champion" is catastrophically regressed below its own seed (40-0) despite healthy internal metrics — best net = `128×10 bigbuf eval502`. |
| 2026-06-15 | reference | **Empty-history trap:** a history-conditioned net through the stateless Gomocup BOARD protocol sandbags ~75 elo — register `incremental=1` (TURN-mode). |
| 2026-06-16 | alphazero | 15×15 gen-stall fixed: uncapped defense-teacher VCF solve was the killer (0 games/6 min) → cap nodes/depth; **9×9 solver budgets do NOT transfer to 15×15**. |
| 2026-06-18 | seek-vct | **First Rapfi-NNUE contact:** champion 20.8% overall (black 42%, **white 0/12 swept**) — deficit is 100% white-side. |
| 2026-06-19 | alphazero | **Autolab went LIVE** (launchd-supervised); first self-driving 15×15 run survived cold-start collapse, **first 15×15 champion crowned (elo 1918)**. |

## Era 4 — white-defense is a theorem → swap2 → teacher exhaustion (Jun 19–25)

| Date | Hub | Milestone |
|---|---|---|
| 2026-06-20 | seek-vct | **CONCLUSION:** white-defense is the **first-player-win THEOREM**, not a net flaw (even Rapfi-vs-Rapfi white loses ~90%) → pivot to swap2. |
| 2026-06-20 | seek-vct | **Swap2 core bet CONFIRMED** at the data level (`8nq1a7cm`): white wins 27% of swap2 self-play (vs ~0% empty-board); strength signal ~70% H2H, non-monotonic. |
| 2026-06-22 | seek-vct | "Bruce Lee" idx-2 15×15 (`gogpmbhw`) still 0/16 vs Rapfi; found the **gen-flood double-tax** (8 workers → cut to 3 → 65s→26s/epoch). |
| 2026-06-23 | seek-vct | **Data-pipeline levers exhausted:** buffer knobs plateau-proof; only remaining lever is an external teacher. |
| 2026-06-24 | seek-vct | **Rapfi distillation CATASTROPHIC** (#77 0/96 trunk corruption; #86 gentle retry also collapsed) — the one-hot signal itself is the culprit. |
| 2026-06-25 | seek-vct | **DAgger/soft-target NEGATIVE:** the Rapfi think-time wall never moves — the net is **eval-capped, not search-capped**; the gap is positional evaluation. |

## Era 5 — the GPU VCT oracle & the seek-VCT program (Jun 25–28)

| Date | Hub | Milestone |
|---|---|---|
| 2026-06-25 | seek-vct | **GPU batch-VCF CRACKED** (~2,500× CPU, 100% correct) — the OR/reachability insight makes it batchable. |
| 2026-06-26 | seek-vct | **VCT bitboard megakernel** (~900–1600× aggregate, on-device, 0 FP/FN); recognition + seeker-steering learnable (**CNN beats attention**, becomes default L2 arch). |
| 2026-06-26 | seek-vct | VCT-reachability fan-mining refutes the "forgiving pre-onset region" — it's a **knife-edge** (up to 98% of alt moves lose by force); only 3.5% of wins need real combinations. |
| 2026-06-27 | seek-vct | **CPU `vcf.py` retired**; GPU `mega_vct_bb` is the sound ~1600× oracle; molecule corpus banked (146,655 non-VCF forced wins). |
| 2026-06-28 | seek-vct | **md-extraction cracked** (#91): mate-distance GPU-self; load-bearing white = the long-VCT phenomenon; the vocabulary/saturation question left OPEN. |

## Era 6 — the sound world (Jun 30 – Jul 2)

| Date | Hub | Milestone |
|---|---|---|
| 2026-06-30 | seek-vct | **VCT-terminus self-play** (#98/#99): generator ends at first cap50 VCT (halves plies); the finisher-hybrid never hurts. |
| 2026-06-30 | seek-vct | **#100 science A/B:** throughput win (45% of control's wall for equal elo) but **robustness LOSS** — terminus loses 0-of-120 H2H, never fires vs a real opponent → attack-only specialization. |
| 2026-07-01 | seek-vct | **Ceiling is STRUCTURAL, not undertraining** (#101 ~2700 epochs, attractor ≈14.5); #103 aux head learns the representation but the policy never acts — **"sensor with no actuator."** |
| 2026-07-01 | seek-vct | **Sound-world recipe** (#107, `zeed2xw5`): oracle-veto + line-planes structurally **kills the 9–10 ply attractor** (plies → high-50s); veto confirmed causally by cap-ablation. |
| 2026-07-02 | seek-vct | **9×9 chapter CLOSES:** fix validated; 9×9 freestyle within cap50 is a fast **black win** (not a draw); product = net + oracle finisher (95% vs heuristic). |
| 2026-07-02 | seek-vct | **13×13 graduation NEGATIVE** (#113): warm-start AND from-scratch both **white 0/20**, lose 0-40 to Rapfi and the old full-game net; **not a product at 13×13 as-is**; VCT oracle veto = 91% of gen wall. |

---

**Where the durable lessons live** (not the chronology): the AlphaZero hub
([what-worked / what-didn't](alphazero.md)), the [Seek-VCT program](seek-vct.md),
[M5-as-Mainframe](m5-mainframe.md) for the perf findings, and the
[Derby](derby.md) for the lab-operation lessons.
