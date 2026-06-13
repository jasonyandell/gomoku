# Gomoku Wiki Index

This wiki is the maintained synthesis layer for the Gomoku training project.
It should compound what we learn from experiments instead of forcing each new
session to rediscover the same story from W&B runs, checkpoints, logs, and chat
history.

## Start Here

Pick the doorway that matches the task. The big training notebook is still the
source of chronological evidence; these routes keep future sessions from
reading it front-to-back unless the work actually needs that.

| Need | Start with | Then read |
|---|---|---|
| Current training story or "how did we get here?" | [topics/training-run-lineage.md](topics/training-run-lineage.md) | [TRAINING_WIKI.md](../TRAINING_WIKI.md) tail, then [log.md](log.md). |
| Launch, resume, monitor, or stop a run | [topics/launch-sequence-runbook.md](topics/launch-sequence-runbook.md) | The relevant design page, then the latest run section in [TRAINING_WIKI.md](../TRAINING_WIKI.md). |
| Interpret training dynamics | [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md) and [topics/az-at-scale-vs-laptop.md](topics/az-at-scale-vs-laptop.md) | The run's evidence section in [TRAINING_WIKI.md](../TRAINING_WIKI.md). |
| Plan a WL-series follow-up | [topics/training-run-lineage.md](topics/training-run-lineage.md) | [topics/wave-of-lockstep-design.md](topics/wave-of-lockstep-design.md), [topics/wl2-scale-emulation-design.md](topics/wl2-scale-emulation-design.md), [topics/wl5-diagnostics-archive-start-design.md](topics/wl5-diagnostics-archive-start-design.md). |
| Plan or work the 15×15 era (port, feasibility, Gomocup path) | [topics/15x15-era-feasibility-and-plan.md](topics/15x15-era-feasibility-and-plan.md) | [topics/external-engine-baselines.md](topics/external-engine-baselines.md), [sources/gomocup-az-techniques-2026-05-27.md](sources/gomocup-az-techniques-2026-05-27.md), [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md). |
| Work on performance or hardware strategy | [topics/research-lab-charter.md](topics/research-lab-charter.md) | [topics/m5-max-as-mainframe.md](topics/m5-max-as-mainframe.md), [topics/research-lab-session-runbook.md](topics/research-lab-session-runbook.md), [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md), [topics/activity-monitor-perf-runbook.md](topics/activity-monitor-perf-runbook.md), [topics/ane-int8-inference.md](topics/ane-int8-inference.md), [topics/buffer-bit-packing.md](topics/buffer-bit-packing.md). |
| Operate the autonomous research lab | [topics/research-lab-charter.md](topics/research-lab-charter.md) | [topics/conventions.md](topics/conventions.md), [topics/research-lab-reviewer-role.md](topics/research-lab-reviewer-role.md), [ops/gpu-queue.md](ops/gpu-queue.md), [ops/best-cells.md](ops/best-cells.md), [ops/perf-log.md](ops/perf-log.md). |
| Understand cross-cutting conventions (autonomy, merge-commits, memories-to-wiki) | [topics/conventions.md](topics/conventions.md) | [topics/research-lab-charter.md](topics/research-lab-charter.md) for lab-specific rules. |
| Run a perf cell, training slice, or sweep (procedure) | [topics/research-lab-session-runbook.md](topics/research-lab-session-runbook.md) | [ops/perf-log.md](ops/perf-log.md), [ops/experiment-ledger.md](ops/experiment-ledger.md), [ops/baselines.md](ops/baselines.md). |
| Run frontier-lab perf fanout | [ops/status.md](ops/status.md) | [ops/frontier.md](ops/frontier.md), [ops/baselines.md](ops/baselines.md), [ops/experiment-ledger.md](ops/experiment-ledger.md), [ops/test-ledger.md](ops/test-ledger.md), [ops/perf-log.md](ops/perf-log.md). |
| Add or interpret external engine baselines | [topics/external-engine-baselines.md](topics/external-engine-baselines.md) | [sources/gomocup-external-engines-2026-05-22.md](sources/gomocup-external-engines-2026-05-22.md), then `gomoku.match` / `gomoku.eval_worker`. |
| Mine or use validation archives | [topics/mining-validation-archives.md](topics/mining-validation-archives.md) | [topics/wl5-diagnostics-archive-start-design.md](topics/wl5-diagnostics-archive-start-design.md) and [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md). |
| Play a checkpoint | [topics/playing-the-model.md](topics/playing-the-model.md) | Latest plateau/run-end notes in [topics/training-run-lineage.md](topics/training-run-lineage.md). |
| Maintain the wiki | [topics/wiki-operating-model.md](topics/wiki-operating-model.md) | [sources/karpathy-llm-wiki.md](sources/karpathy-llm-wiki.md) and [log.md](log.md). |

## Current Synthesis

The central project question is not just "can the code run faster?" It is "what
training loop reliably teaches 9x9 AlphaZero-style Gomoku to defend, not merely
to imitate fast attacks from its own search?"

Current evidence lives in [TRAINING_WIKI.md](../TRAINING_WIKI.md). The high-level
read is:

- The strongest current training lineage is the WL series. See
  [topics/training-run-lineage.md](topics/training-run-lineage.md) for the
  compact Z -> WL1 -> WL5 map and exact run IDs.
- Local throughput work succeeded; distributed self-play made iteration much
  faster.
- Native MCTS plus eval-only Conv+BatchNorm fusion moved the self-play
  bottleneck from Python tree churn toward evaluator and engine-boundary
  questions. See [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md).
- First engine scout says Core ML is slower than fused PyTorch/MPS for raw
  small-model eval, but it hurts concurrent MPS training far less than a
  competing PyTorch/MPS eval process. Treat the next step as a production
  overlap experiment, not a naked eval microbench. See
  [topics/ane-int8-inference.md](topics/ane-int8-inference.md).
- The main training failure mode is fast-attack collapse: policy targets sharpen
  around attacks, self-play opponents fail to punish missing defense, and fixed
  heuristic/lookahead opponents expose the gap.
- Short evals are noisy. Strength claims need fixed baselines, enough games, and
  clear checkpoint/run IDs.
- The next useful additions to the wiki should preserve evidence: command,
  config, W&B run ID, checkpoint path, metrics, and the working-theory change.

## Page Catalog

### Core

| Page | Role |
|---|---|
| [AGENTS.md](../AGENTS.md) | Schema for agents: wiki rules, repo map, and working conventions. |
| [TRAINING_WIKI.md](../TRAINING_WIKI.md) | Primary append-oriented training notebook: run history, hypotheses, results, and corrections. |
| [log.md](log.md) | Chronological wiki maintenance log. |
| [topics/wiki-operating-model.md](topics/wiki-operating-model.md) | Gomoku-specific adaptation of the LLM wiki pattern. |
| [topics/training-run-lineage.md](topics/training-run-lineage.md) | Compact route map for the Z and WL-series run sequence. |
| [sources/karpathy-llm-wiki.md](sources/karpathy-llm-wiki.md) | Source record for the LLM wiki charter that inspired this structure. |
| [sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md](sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md) | Transcript of Sid Bidasaria's "Stop babysitting your agents" talk (verification → multi-Claude → background loops). Locally Whisper-transcribed because the video has no captions. |
| [sources/gomocup-az-techniques-2026-05-27.md](sources/gomocup-az-techniques-2026-05-27.md) | Frozen survey (2026-05-27) of AlphaGomoku/KataGo training/search techniques considered for the lab: WDL value head, LCB/variance-PUCT, moves-left, in-search-VCF, SE blocks, and others. The synthesis-and-verdict layer for each lever lives in [ops/research-board.md](ops/research-board.md) ("Open candidates" + the v8/v9 verdicts). |

### Training Dynamics

| Page | Role |
|---|---|
| [topics/az-at-scale-vs-laptop.md](topics/az-at-scale-vs-laptop.md) | Why exploration arcs, plies swings, and age oscillations are laptop-scale artifacts before they are bug evidence. |
| [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md) | Why low-floor loss bounces can be healthy in small-scale AZ, and when to suspect a real bug. |

### Run Designs

| Page | Role |
|---|---|
| [topics/wave-of-lockstep-design.md](topics/wave-of-lockstep-design.md) | WL1 design: per-version uniformity via wave-of-lockstep + greedy fill. Now a preserved design record plus WL1 status pointer. |
| [topics/wl2-scale-emulation-design.md](topics/wl2-scale-emulation-design.md) | WL2 design: EMA self-play + past-checkpoint mix + worker poll jitter + grad accumulation. Now a preserved design record plus WL2 status pointer. |
| [topics/wl5-diagnostics-archive-start-design.md](topics/wl5-diagnostics-archive-start-design.md) | WL5 design: fixed validation archive, H/KL decomposition, per-color/ply metrics, and archive-start. Now a preserved design record plus WL5 status pointer. |

### Performance And Hardware

| Page | Role |
|---|---|
| [topics/15x15-era-feasibility-and-plan.md](topics/15x15-era-feasibility-and-plan.md) | 2026-06-12 report: measured board/net scaling on MPS (96×8 @ 15×15 costs only 2.32× at wave=64), week-scale feasibility envelope, phased plan (Rapfi certify → rules decision → port → smoke → first run → derby). |
| [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) | Where MCTS gen-time wins are and are not. Do not re-port "v2 storage"; we are already there. |
| [topics/activity-monitor-perf-runbook.md](topics/activity-monitor-perf-runbook.md) | Practical knobs and interpretation rules for Mac Activity Monitor perf experiments. |
| [topics/m5-max-as-mainframe.md](topics/m5-max-as-mainframe.md) | Guiding philosophy and sequence for the post-WL5 perf era on Jason's M5 Max. |
| [topics/ane-int8-inference.md](topics/ane-int8-inference.md) | Engine-isolation plan and first scout for Core ML / ANE / CPU lanes around MPS training. |
| [topics/coreml-ane-residency-lab.md](topics/coreml-ane-residency-lab.md) | Rail-proof lab for Core ML / ANE residency claims; caps `CPU_AND_NE` label checks below ANE-backed unless powermetrics shows nonzero ANE rail. |
| [topics/buffer-bit-packing.md](topics/buffer-bit-packing.md) | Post-WL5 replay-buffer compression plan: bit-packed planes plus FP16 policy, with cheap-test gate. |
| [topics/conventions.md](topics/conventions.md) | Cross-cutting project conventions: deny-list autonomy (Class A/B/C), merge-commits-never-rebase, memories-also-go-to-wiki, Opus-minutes-not-human-days. Source of truth mirrored in memory. |
| [topics/workflow-orchestration.md](topics/workflow-orchestration.md) | How the Claude Code *Workflow* feature maps onto the lab: deterministic agent-chaining (not the `/loop` looper) = the everything-else lane of the two-queue scheduler, never the GPU lane. Fit/misfit table, the cockpit/verify-gate why, and the first real workflow (`.claude/workflows/reviewer-gated-fanout.js`). |
| [topics/research-lab-charter.md](topics/research-lab-charter.md) | Charter for the autonomous research lab: mission, two research areas (perf + training-recipe), GPU-required vs everything-else queues, training-slice protocol, smoke-first doctrine, operating loop, priority function, tier system, reviewer gate, autonomy boundaries, worktree discipline, stop conditions. |
| [topics/wall-clock-to-elo-metric.md](topics/wall-clock-to-elo-metric.md) | LF1-followup #4 design: wall-clock-to-elo as a first-class metric family (MTTE primary, EPWH/Δelo·Δt⁻¹ secondary) the throughput proxies must be checked against; protocol, val/policy_ce gate, gap analysis vs `delta_e_harness.py`, proposed charter diff. |
| [topics/research-lab-reviewer-role.md](topics/research-lab-reviewer-role.md) | Codified Reviewer role: when it fires (post-lane + mid-loop), the audit prompts, the three verdicts (APPROVE/REVISE/BLOCK), and what it does NOT do. |
| [topics/research-lab-session-runbook.md](topics/research-lab-session-runbook.md) | End-to-end procedure for running a GPU-required lab item (perf cell, training slice, or sweep): pre-flight, naming, command surfaces, receipt, surfaces to update. |
| [topics/probe-100pct.md](topics/probe-100pct.md) | `scripts/probe_100pct.py` — one-command driver for the RESUME PLAYBOOK step 1 sweep (eval-sims × eval-VCF vs lookahead4) on a matured checkpoint; per-cell distance-to-100% via the existing `report_100pct.py` formula. |

### Operations And Use

| Page | Role |
|---|---|
| [topics/launch-sequence-runbook.md](topics/launch-sequence-runbook.md) | Reusable playbook for launching, smoking, monitoring, and ending training runs. |
| [topics/mining-validation-archives.md](topics/mining-validation-archives.md) | Operational recipe for `scripts/mine_validation_archive.py`: buckets, knobs, throughput, anti-patterns. |
| [topics/playing-the-model.md](topics/playing-the-model.md) | How to play a trained checkpoint through the local web UI or live SPA. |
| [topics/external-engine-baselines.md](topics/external-engine-baselines.md) | Rated OSS/source-available Gomocup engine candidates and the Piskvork wrapper plan. |
| [topics/containerize-training-runs.md](topics/containerize-training-runs.md) | **Backlog (for soon):** containerize the training run, one container at a time, refine the skill for lower startup friction/time. Open question: no Metal/MPS in Docker on macOS — targets off-Mac/at-scale or a non-Docker run unit. |
| [../scripts/wandb_workspace.py](../scripts/wandb_workspace.py) | Creates W&B workspaces for run overlays. Regenerate when a new run joins the comparison set. |

### Frontier Lab Ops

| Page | Role |
|---|---|
| [ops/status.md](ops/status.md) | Current ML performance frontier control-room summary. |
| [ops/frontier.md](ops/frontier.md) | Human-readable board projected from `.frontier/lanes.json`. |
| [ops/baselines.md](ops/baselines.md) | Benchmark command surfaces and reference results. |
| [ops/experiment-ledger.md](ops/experiment-ledger.md) | Receipt ledger for promote/reject/block decisions. |
| [ops/test-ledger.md](ops/test-ledger.md) | Validation command ledger for frontier decisions. |
| [ops/perf-log.md](ops/perf-log.md) | Day-by-day narrative timeline for the M5 Max perf era. |
| [ops/gpu-queue.md](ops/gpu-queue.md) | Live, ordered queue for GPU-required lab items (perf cells and training slices). Source of truth for the autonomous lab loop. |
| [ops/best-cells.md](ops/best-cells.md) | Current best cell per quality reference point; promotion log. |

## Layers

- **Evidence sources**: W&B histories, local logs, checkpoint files, match
  outputs, scripts, raw command output, and external source records under
  [sources/](sources/).
- **Maintained synthesis**: this index, topic pages under [topics/](topics/),
  and the training notebook.
- **Schema**: [AGENTS.md](../AGENTS.md), which tells future sessions how to
  maintain and use the wiki.

## Maintenance Rules

- Read this index first, then drill into the pages it names.
- Keep source records and artifacts immutable unless the user explicitly asks
  for cleanup.
- Keep the training notebook append-oriented. When a conclusion changes, add a
  dated correction with evidence instead of polishing the old entry.
- File useful answers back into the wiki when they would save a future session
  from recomputing the same synthesis.
- Update [log.md](log.md) whenever the wiki structure, index, or synthesis pages
  change in a meaningful way.
