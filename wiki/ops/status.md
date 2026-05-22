# ML Perf Frontier Status

This page is the current control-room summary for bounded Gomoku performance work. Raw evidence stays in logs, W&B, sweep artifacts, worktrees, and open notes; this page is the curated mixdown.

## Current Focus

Build an evidence-backed M5 Max performance map for Gomoku self-play/training while preserving training-quality guardrails. The next work should produce receipts, not intuition: command, commit, hardware/env, metrics, artifacts, confidence, and decision.

Use the frontier lab commands from pi:

```text
/frontier-start --dry-run
/frontier-start --max=5
/frontier-status
/frontier-curate
```

## Operating Rule

BFS by default: current-main baseline receipts and contour mapping before deep implementation. DFS only when a lane is hot because it has an active implementation worktree, a benchmark regression, a verifier failure, or a blocker that prevents progress.

## Current Read

- Native MCTS and eval Conv+BN fusion are already landed; new perf work should not rediscover those wins.
- Active frontier run `20260522T054739Z` has five claimed lanes: baseline receipts, production contour, ANE residency rail proof, quality gates, and this curation lane. Before writing this lane's receipt, no sibling worker receipts were present under `.frontier/runs/20260522T054739Z/workers/{01,02,03,04}-*/receipt.md`; keep later board edits receipt-backed as workers finish.
- Perf10 evidence in `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv` says production shape matters: native 8 workers × 8 games beat native 4×16 and fallback 8×8, but the production win is smaller than the single-process microbench.
- Engine-isolation evidence has split into three tiers: the first aggressive scout showed Core ML pressure hurts MPS training less than PyTorch/MPS eval pressure; the first rail check showed `CPU_AND_NE` alone is not ANE proof; the detached 934b residency lab now has powermetrics-positive Gomoku Core ML FP16 cells at batch sizes 8/32/128/1024, with the cleanest candidates at b32/b128. Treat these as unintegrated `ane-metered` candidates until the ANE lane reproduces or integrates them with explicit commands and a nearby positive control.
- Activity Monitor GPU percent remains a supporting observation only. Score by wall-clock, games/sec, positions/sec, trainer slowdown, plies/game shape, and fixed baseline/archive quality gates.
- Replay-buffer bit-packing is a post-WL5 candidate, but the cheap buffer-width ablation should decide whether the refactor is worth doing.

## Active Hot Lanes And Evidence

| Lane | Status | Evidence / blocker |
| --- | --- | --- |
| Baseline receipts | active worker | Waiting for current-main smoke/microbench receipts under `.frontier/runs/20260522T054739Z/workers/01-baseline-receipts/`; ops pages still only have seed/reference rows. |
| Production contour | active worker | Seed evidence is perf10 TSV in `/Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv`; exact launcher commands still need capture on current main. |
| ANE residency rail proof | queued/active frontier lane, external worktree hot | `/Users/jason/.codex/worktrees/934b/gomoku` has uncommitted harness/tests/wiki plus rail JSONs under `sweep_logs/coreml_ane_residency/`; integrate or reproduce before unblocking production engine-overlap. |
| Quality promotion gates | active worker | Gate addendum is already seeded in `wiki/ops/experiment-ledger.md`; wait for worker receipt before tightening further. |
| Control-room curation | active worker | This pass syncs the ops pages and writes open-note/manager receipts; no new benchmark launched. |

## Next Action

Run or finish the first five lanes from `.frontier/lanes.json`: baseline receipts, production contour sweep, ANE residency rail proof, quality promotion gates, and control-room curation. Keep production engine-overlap blocked until the residency lane integrates or repeats an ANE-metered Gomoku candidate, or deliberately selects an explicit CPU-only isolation candidate.
