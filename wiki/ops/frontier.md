# ML Perf Workflow Frontier

> ⚠️ **RETIRED 2026-07-02 — dead mechanism.** The pi frontier-lab extension that
> projected this board from `.frontier/lanes.json` is retired; the state file is
> frozen at 2026-05-22. Superseded by the two-queue scheduler in
> [gpu-queue.md](gpu-queue.md). To operate today, start at the
> [Ops hub](../ops.md). Kept as historical evidence.

Source of truth for machine-readable lanes: `.frontier/lanes.json`.

This page is the human-readable board. The frontier lab extension claims lanes from `.frontier/lanes.json`, fans out isolated workers, and asks the wiki curator to roll receipts back into this page.

## Current Board

| Lane | Stage | Heat | Resource | Next action |
| --- | --- | --- | --- | --- |
| Baseline receipts and artifact convention | completed | done | code | Done in run `20260522T054739Z`; repeat only when WL5 is idle if absolute MPS baseline numbers matter. |
| M5 Max production self-play contour | completed | done | gpu | Done in run `20260522T054739Z`; promote native small 8w8g sims400 wave64 as throughput default. |
| Core ML / ANE residency rail proof | blocked | blocked | gpu | Re-run only when `sudo -n true` passes; capture fresh Vision positive control plus CPU_ONLY negative and CPU_AND_NE cells. |
| Training quality promotion gates | completed | done | cpu | Done; use `wiki/ops/experiment-ledger.md` gate for behavior-changing perf receipts. |
| Perf wiki control room curation | completed | done | wiki | Done; stale active-run claims trimmed after manual recovery. |
| Outer self-play loop profiling | completed | done | cpu | Done in run `20260522T061713Z`; reject/no-op for post-search Python native pass. |
| Production engine-overlap experiment | blocked-on-ane-residency | blocked | gpu | Do not launch until ANE proof or explicit CPU-only isolation candidate exists. |
| Replay-buffer width cheap test | seeded | warm | cpu | After WL5 reports out, run 1.5M vs 750k buffer ablation before bit-packing. Note: BAB1-buf-ablation-1p5M is running in another session as of 2026-05-22; see [perf-log.md](perf-log.md). |
| Canonical 5-axis M5 Max contour sweep | completed | done | gpu | Completed 2026-05-23 in `sweep_logs/canonical-sweep-20260523T015614Z`; 23/23 cells ok; wave=128 promoted as new throughput default (+27% over V=64); receipt in [experiment-ledger.md](experiment-ledger.md). |

## Completed Run 20260522T061713Z

- Claimed lane: `outer-loop-python-profile`.
- Integration status: worker branch merged (`5e20aaa` worker, `411ed75` integration).
- Result: post-search worker-loop Python is too small for a 10-20% optimization lane in the bounded production-shaped profile.
- Key wave-mode numbers: wall 1.064s; `native_search_batch` 1.013s / 95.2%; evaluator 0.896s / 84.3%; native search excluding evaluator 0.117s / 11.0%; all measured post-search Python 0.050s / 4.7%; file handoff 3.2%; D4 0.82%; action sampling 0.30%.
- Decision: `reject` for action sampling / trajectory staging / D4 / record creation / file handoff optimization. Next perf attention belongs to evaluator/engine overlap or the native-search/evaluator boundary, with engine-overlap still blocked on ANE rail proof.
- Open note: `wiki/ops/open-notes/20260522T061713Z-01-outer-loop-python-profile.md`.

## Completed Run 20260522T054739Z

- Workers completed and receipts were merged manually because the manager failed at integration with a stale UI context after all worker exits. The extension was patched in `7e26e7c` to tolerate stale background UI handles.
- Baseline receipts: `sweep_logs/frontier-baselines/20260522T054845Z/`.
- Production contour: `sweep_logs/production-contour-20260522/`.
- ANE lane: harness integrated, but fresh `powermetrics required` run blocked by unavailable cached/passwordless sudo.
- Open notes: `wiki/ops/open-notes/20260522T054739Z-*.md`.

## Promotion Rule

A lane advances only with a receipt: hypothesis, baseline command, candidate command or blocker, hardware/env, metrics, artifacts, confidence/noise caveat, decision (`promote`, `reject`, `blocked`, or `needs_repeat`), and next action. Behavior-changing perf changes must also satisfy the Training-Quality Promotion Gate in `wiki/ops/experiment-ledger.md`.
