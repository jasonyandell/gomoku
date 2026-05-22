# 20260522T054739Z — quality-promotion-gates

## Receipt

```yaml
lane: quality-promotion-gates
hypothesis: A documented training-quality gate will prevent behavior-touching performance changes from being promoted on throughput alone.
code_ref: branch frontier/20260522T054739Z/04-quality-promotion-gates; commit recorded in manager receipt/final response
dataset_ref: docs-only; evidence basis is existing wiki training history, WL5 validation archive design, and ops ledgers
baseline_command: N/A — documentation-only gate codification; no performance baseline run
candidate_command: N/A — documentation-only gate codification; edited wiki/ops/experiment-ledger.md Training-Quality Promotion Gate
hardware: MacBook Pro, Apple M5 Max, 18 CPU cores, 48 GB unified memory; Darwin 25.4.0 arm64
seed: N/A
baseline_metric: Prior ops gate named generic baseline/archive, plies, noise caveat, run IDs, and decision but did not specify accepted gate names, minimum n/repeat policy, or full reproducibility fields.
candidate_metric: Gate now names heuristic/lookahead2/lookahead4 and archives/wl5_validation_v1.pt metrics; requires plies_mean plus game-length collapse check; defines n<20 smoke, n=20 canary, n>=50 or repeated n>=20 preferred; requires checkpoint/run IDs, commit, seed policy, and env/backend flags; preserves explicit decision rule.
delta: Promotion criteria changed from brief addendum to explicit five-part behavior-changing perf gate.
confidence: high for documentation completeness; no strength/performance claim made.
artifacts: wiki/ops/experiment-ledger.md; wiki/ops/open-notes/20260522T054739Z-04-quality-promotion-gates.md
commands_run:
  - wc -l TRAINING_WIKI.md wiki/ops/experiment-ledger.md && git status --short --branch
  - git log --oneline -5 && git status --short
  - pytest -q
  - git diff -- wiki/ops/experiment-ledger.md && git status --short
  - git diff --check
  - uname -a && system_profiler SPHardwareDataType | grep -E 'Model Name|Chip|Total Number of Cores|Memory'
decision: promote
next_action: Curator should mark the lane done in ops status/frontier and keep this gate as the required receipt rule for behavior-changing perf work.
```

## Files touched

- `wiki/ops/experiment-ledger.md` — expanded promotion addendum into explicit Training-Quality Promotion Gate.
- `wiki/ops/open-notes/20260522T054739Z-04-quality-promotion-gates.md` — worker note/receipt.

## Result

Lane done shape satisfied: named fixed external/archive gates, plies/game-length-collapse guardrail, short-eval n/repeat policy, required run/checkpoint IDs, and explicit decision rule are documented.

## Blockers

None.

## Board-update recommendation

Curator can mark `quality-promotion-gates` complete/promoted. No baseline/test-ledger numeric update is needed because this was a docs-only gate codification, not a benchmark.
