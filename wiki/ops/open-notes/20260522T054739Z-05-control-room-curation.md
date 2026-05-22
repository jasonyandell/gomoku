# 20260522T054739Z-05-control-room-curation

## Summary

Curated the active perf frontier control room from disk evidence only. No new benchmark/training run was launched.

## Receipt

```yaml
lane: control-room-curation
hypothesis: Ops pages can be synchronized with actual perf evidence on disk without broad rewrites or launching new benchmarks.
code_ref: frontier/20260522T054739Z/05-control-room-curation HEAD; base a418f677; external perf worktree 4f21cdd; external 934b detached b9b9eab dirty
dataset_ref: wiki/ops pages; /Users/jason/code/gomoku-perf-extension/sweep_logs/perf10-summary.tsv; /Users/jason/.codex/worktrees/934b/gomoku/sweep_logs/coreml_ane_residency/*.json; .frontier/runs/20260522T054739Z state
baseline_command: find .frontier/runs/20260522T054739Z/workers -name receipt.md; read existing wiki/ops pages and .frontier/lanes.json
candidate_command: curate wiki/ops/status.md, frontier.md, baselines.md, experiment-ledger.md, test-ledger.md, and wiki/log.md from disk evidence
hardware: M5 Max evidence curated; no local benchmark run by this lane
seed: n/a
baseline_metric: prior ops pages had perf10 seed rows and generic 934b worktree mention; no sibling worker receipts existed yet under .frontier/runs/20260522T054739Z/workers/{01,02,03,04}-*/receipt.md
candidate_metric: status/frontier now name active frontier worktrees and 934b ANE-metered candidates; baselines/test-ledger include 934b b1 negative, b32/b128 positive ANE rail rows; experiment-ledger has a needs_repeat receipt for 934b
delta: control-room coverage improved; no throughput delta claimed by this lane
confidence: medium; curation is evidence-backed from files on disk, but 934b remains unintegrated/detached and needs reproduction by the ANE lane
artifacts: wiki/ops/status.md; wiki/ops/frontier.md; wiki/ops/baselines.md; wiki/ops/experiment-ledger.md; wiki/ops/test-ledger.md; wiki/log.md; wiki/ops/open-notes/20260522T054739Z-05-control-room-curation.md; /Users/jason/code/gomoku/.frontier/runs/20260522T054739Z/workers/05-control-room-curation/receipt.md
commands_run:
  - git status --short --branch
  - find /Users/jason/code/gomoku/.frontier/runs/20260522T054739Z -maxdepth 4 -type f | sort
  - git worktree list --porcelain
  - find /Users/jason/code/gomoku-perf-extension ...
  - find /Users/jason/.codex/worktrees/934b/gomoku ...
  - python JSON summary inspection for 934b coreml_ane_residency artifacts
  - git diff --check
  - read required wiki/topic/ops files
decision: promote
next_action: Let 01/02/03/04 workers finish receipts; ANE lane should integrate or reproduce 934b with exact commands, positive control, CPU_ONLY negative control, and production-overlap candidate before unblocking engine-overlap-production.
```

## Files Touched

- `wiki/ops/status.md`
- `wiki/ops/frontier.md`
- `wiki/ops/baselines.md`
- `wiki/ops/experiment-ledger.md`
- `wiki/ops/test-ledger.md`
- `wiki/log.md`
- `wiki/ops/open-notes/20260522T054739Z-05-control-room-curation.md`
- external manager receipt path after write

## Result

- `wiki/ops/status.md` now names active hot lanes, current frontier receipt state, and the 934b ANE-metered candidate caveat.
- `wiki/ops/frontier.md` still matches `.frontier/lanes.json` and now has precise worktree evidence/inventory.
- `wiki/ops/baselines.md` includes engine-isolation/residency reference rows for aggressive scout and 934b b1/b32/b128 cells.
- `wiki/ops/experiment-ledger.md` has a `needs_repeat` receipt for 934b Core ML Gomoku ANE residency candidates.
- `wiki/ops/test-ledger.md` has command/result rows for perf10 and 934b rail artifacts plus this curation inventory check.

## Blockers / Caveats

- No sibling worker receipts existed yet for this frontier run when checked, so this pass could not integrate 01/02/03/04 final receipts.
- 934b evidence is promising but unintegrated: detached dirty worktree, synthetic random inputs, short cells, and no production self-play overlap yet.
- No new validation benchmark or pytest was run by this wiki-only lane.

## Board-Update Recommendation

Keep `engine-overlap-production` blocked until `ane-residency-rail-proof` either reproduces/integrates 934b with exact commands and positive/negative controls, or explicitly chooses a CPU-only isolation candidate. Use perf10 as seed evidence only until baseline/production workers capture exact commands on current main.
