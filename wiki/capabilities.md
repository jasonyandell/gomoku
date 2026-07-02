# Capabilities — what this repo can DO

A **synthesis layer across key concepts**, complementary to the other two doorways:
`index.md`'s *Start Here* maps **tasks → docs**; `TRAINING_WIKI.md` is the **dated
evidence spine**; **this page maps capabilities → how-to + deep doc**, so a new
session can see the whole toolbox at a glance without reading 56 topic pages.

Purely additive: each row points to the deep page that owns the detail; nothing
here supersedes those. **NEW** = added in the 2026-06-25 idx-2 distillation run
(`feat/gentle-rapfi-teacher`). Always `uv run …` (per-worktree venv);
`GOMOKU_BOARD_SIZE=15` for the 15×15 era.

## Data & teacher signal

| Capability | How | Deep doc |
|---|---|---|
| **Mine Rapfi teacher data at scale** (soft policy + value over a fixed opening's neighbourhood; D4-canonical dedup; crash-robust sharded npz; ~700 moves/s) **NEW** | `python -m gomoku.rapfimine run --out <dir> --workers 24 --max-node 5000` | [topics/rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md) |
| **Pretrain a net by distillation, then warm-start AZ** (soft-target CE + value MSE → standard checkpoint → `run_sweep --resume`) **NEW** | `python -m gomoku.rapfimine.pretrain --shards <dir> --out checkpoints/seed.pt` | [topics/rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md) |
| **Always-on Rapfi "sensei"** (warm Rapfi pool behind HTTP; CPU-only, non-competing) + **policy-side teacher distillation** during training (`--teacher-weight`) | sensei daemon; `gomoku.teacher` | [topics/eval-teacher-sensei.md](topics/eval-teacher-sensei.md) |
| **Mine validation archives** (bucketed fixed eval sets for H/KL decomposition) | `scripts/mine_validation_archive.py` | [topics/mining-validation-archives.md](topics/mining-validation-archives.md) |

> ⚠️ Distillation lesson (#77/#86): **one-hot** Rapfi targets flatten the policy head and **regress** the net even at half-LR/low-weight; the **soft-target** winrate distillation is the designed fix (validated at scale 2026-06-25 — see spine). Gate any teacher run on **H2H-vs-frozen-parent**, not the Rapfi cadence.

## Train

| Capability | How | Deep doc |
|---|---|---|
| **AlphaZero self-play loop** (trainer + N self-play workers, replay buffer, W&B, checkpointing) via named **cells** | `scripts/run_sweep.py --cell <name>` | [topics/launch-sequence-runbook.md](topics/launch-sequence-runbook.md) |
| **Fixed-fair-openings training** (every game starts from a known-fair Rapfi opening; sidesteps the unfair-opener black edge) — the live 15×15 recipe | cell `G15-fixed-openings` (+ the `G{9,11,13}` ladder) | [cards/gomoku-15x15-fixed-fair-openings.md](cards/gomoku-15x15-fixed-fair-openings.md) |
| **swap2 opening protocol** (rebalances the GAME so white becomes winnable; the principled white fix) | swap2 cells | [topics/swap2-opening-protocol.md](topics/swap2-opening-protocol.md) |
| **Single-opening over-specialization** ("Bruce Lee one position": restrict self-play to one opening) **NEW** | `GOMOKU_DROP_OPENERS=…` (drop all but the target index) | [topics/rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md) |
| **Cross-board warm-start** (seed a 15×15 net from a 9×9 champion's shared conv tower) | `scripts/warmstart_15x15.py` | [topics/15x15-era-feasibility-and-plan.md](topics/15x15-era-feasibility-and-plan.md) |
| **Bit-packed replay buffer** (1M positions ≈ 1.3 GB; `--pack-buffer`, recency sampling) | cell flag `--pack-buffer --buffer-recency-frac` | [topics/buffer-bit-packing.md](topics/buffer-bit-packing.md) |

## Evaluate strength

| Capability | How | Deep doc |
|---|---|---|
| **Reliable eval set** (net-vs-net pure-torch + pure-python `random`/`heuristic`/`lookahead:depth=N`) | `gomoku.eval_panel`, `gomoku.match` | [topics/reliable-eval-set.md](topics/reliable-eval-set.md) |
| **Native Rapfi-NNUE anchor** (Gomocup winner, full search budget; the strength yardstick) | registered panel ruler `rapfi` | [topics/external-engine-baselines.md](topics/external-engine-baselines.md) |
| **FAST eval-gradient vs graded Rapfi** (~20 s: batched net MCTS + parallel `RapfiPool.label_states`; **think-time is the strength dial, NOT max_node**) **NEW** | `python -m gomoku.rapfimine.fast_eval --checkpoint <pt>` | [topics/rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md) |
| **idx-2 gate** (H2H of a checkpoint vs native Rapfi from one opening, white split) **NEW** | `python -m gomoku.rapfimine.eval_idx2 --checkpoint <pt>` | [topics/rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md) |
| **Δelo engine-panel derby** (calibrated relative-strength ladder across engines/nets) | derby runner | [topics/engine-panel-derby-design.md](topics/engine-panel-derby-design.md) |
| **White-side (defense) quantification** (per-color split; the "never lose as white" wound) | `scripts/panel_white_elo.py` | [topics/white-side-defense-plan.md](topics/white-side-defense-plan.md) |

> ⚠️ Eval lesson: **gate "did this help?" on H2H vs the preserved champion, not Rapfi**; short evals are noisy (small-n = hint). The Rapfi yardstick was broken once already (#28/#40) — see the index banner.

## Search & solve

| Capability | How | Deep doc |
|---|---|---|
| **Wave-batched MCTS** (PUCT; many games' leaf evals batched in one pass — the eval-throughput trick) | `gomoku.mcts.run_batched_mcts` | [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) |
| **Native MCTS / state-ops extensions** (A/B with `GOMOKU_DISABLE_NATIVE_MCTS=1`, `GOMOKU_DISABLE_NATIVE_STATE_OPS=1`) | env flags | [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) |
| **On-device GPU VCT solver** (the whole AND/OR proof search on-device as a `ulong[4]` bitboard; ~1600× CPU, 0 FP/FN; bulk-synchronous per the call-cost law; outputs `move`/`support`/`carriers`/`w`/`winmask`/`max_depth`; `work_steal` persistent-cursor dispatch + `solve_vct_streaming` iterative-deepening for pools larger than one dispatch, #93) | `scripts.vct_metal.mega_vct_bb.solve_vct_mega_bb` | [topics/mega-vct-solver.md](topics/mega-vct-solver.md) |
| **Mate-distance (md_min) + md-invariant stencil minimizer** (#91; order-independent depth-cap binary search; reduces a proven VCT to a typed stencil `(B, W, support, md0)` by md-invariant ablation. Tool works; load-bearing-W count rises with mate depth; stencils still over-inclusive ⇒ the vocabulary/minimal-W questions are open — see the 2026-06-28 second-pass calibration) | `solve_md_min`, `scripts/threat_shapes/md_minimize.py` | [topics/shape-library-engine.md](topics/shape-library-engine.md) §3/§8 |
| **Streaming stencil minimizer + vocab analysis** (append-only, trivially-resumable reducer-over-a-log; content-addressed boards, status ok/capped/dead; `--order shuffled` = 93% yield vs deepest-first 20%; proves the vocabulary null at n=1225; scales to unbounded n) | `scripts/threat_shapes/md_minimize_stream.py`, `analyze_vocab_stream.py` | [topics/shape-library-engine.md](topics/shape-library-engine.md) §8 |
| **Forward VCT frontier expansion + opening danger map** (expand a fixed opening as an AND/OR frontier — Rapfi-top-8 moves both sides, GPU VCT solver as the only oracle, both-colour termini; append-only / resumable / D4-content-addressed; then a depths-1→7 danger map with both-sides danger densities + honest cap/gap accounting + Rapfi-prior-vs-oracle overlay) **NEW** | `scripts/idx2_vct/{frontier,probe_capped,analyze_opening}.py` | [topics/idx2-vct-frontier-map.md](topics/idx2-vct-frontier-map.md) |
| **CPU VCF/VCT solver** (retired as a runtime dep — gated bootstrap oracle only; `GOMOKU_ALLOW_CPU_SOLVER=1`) | `gomoku.vcf` | [topics/mega-vct-solver.md](topics/mega-vct-solver.md) § CPU solver retired |

## Operate

| Capability | How | Deep doc |
|---|---|---|
| **uv-native dev loop** (one per-worktree `.venv`, uv.lock-pinned; never `source activate`) | `uv sync --extra dev`, `uv run …` | [topics/conventions.md](topics/conventions.md) |
| **One-worktree-per-task workflow** (off `main` → `feat/<slug>` → `merge --no-ff` → push; cleanup is MANUAL — janitor retired 2026-07-01) | `scripts/gh_worktree.py <N>` | [topics/worktree-hygiene.md](topics/worktree-hygiene.md) |
| **Self-driving autolab** (out-of-git ledger spine + trainer/arena/research/worker loops; crowns champions unattended) | `gomoku/lab/`, `autolab up`/`down` | [topics/autolab-architecture.md](topics/autolab-architecture.md) |
| **Play a checkpoint** (local web UI / live SPA) | `uv run gomoku-web`, `uv run gomoku-play --checkpoint <pt>` | [topics/playing-the-model.md](topics/playing-the-model.md) |
| **W&B run overlays** (workspaces for comparison sets) | `scripts/wandb_workspace.py` | [topics/external-engine-baselines.md](topics/external-engine-baselines.md) |

---

*Maintenance: when a run adds or sharpens a capability, add/update a row here AND
append a dated `TRAINING_WIKI.md` entry (the spine). A capability with no deep doc
is a smell — write the topic page too.*
