# ML Perf Workflow Frontier

Source of truth for machine-readable lanes: `.frontier/lanes.json`.

This page is the human-readable board. The frontier lab extension claims lanes from `.frontier/lanes.json`, fans out isolated workers, and asks the wiki curator to roll receipts back into this page.

| Lane | Stage | Heat | Resource | Next action |
| --- | --- | --- | --- | --- |
| Baseline perf harness | seeded | hot | code | Inventory benchmark surfaces and write a baseline receipt before changing hot paths. |
| Self-play throughput hot path | open | hot | gpu | Compare baseline vs one candidate under bounded production-shaped microbench. |
| Apple Silicon engine isolation | open | warm | gpu | Design the next production-overlap experiment for MPS/Core ML/CPU split. |
| Outer self-play loop profiling | open | warm | cpu | Profile post-native Python overhead outside the C search boundary. |
| Training quality guardrails | open | active | cpu | Define minimum eval/validation receipt before promoting perf changes. |
| Perf wiki control room | seeded | active | wiki | Curate receipts into status/frontier/baselines/experiment-ledger/test-ledger. |

## Promotion Rule

A lane advances only with a receipt: hypothesis, baseline command, candidate command or blocker, hardware/env, metrics, artifacts, confidence/noise caveat, decision, and next action.
