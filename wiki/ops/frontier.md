# ML Perf Workflow Frontier

Source of truth for machine-readable lanes: `.frontier/lanes.json`.

This page is the human-readable board. The frontier lab extension claims lanes from `.frontier/lanes.json`, fans out isolated workers, and asks the wiki curator to roll receipts back into this page.

## Current Board

| Lane | Stage | Heat | Resource | Next action |
| --- | --- | --- | --- | --- |
| Baseline receipts and artifact convention | active | hot | code | Run current-main CPU smoke + MPS microbench, pair native/fallback where practical, and write baseline/test ledger receipts. |
| M5 Max production self-play contour | open | hot | gpu | Use perf10 as seed evidence, then run a bounded production-shaped fractional sweep across workers/games/sims/wave/model. |
| Core ML / ANE residency rail proof | active | hot | gpu | Inspect/integrate the 934b residency worktree and run powermetrics-required conv/resnet/gomoku scouts with a Vision positive control. |
| Training quality promotion gates | active | active | cpu | Codify the minimum gate for behavior-touching perf changes: fixed baseline/archive, plies, noise caveat, and decision. |
| Perf wiki control room curation | active | active | wiki | Curate perf10, Core ML/ANE residency, and frontier receipts into ops pages. |
| Outer self-play loop profiling | open | warm | cpu | Profile post-native worker-loop Python: sampling, trajectory staging, D4, record creation, and file handoff. |
| Production engine-overlap experiment | blocked-on-ane-residency | warm | gpu | Wait for ANE-metered or explicit CPU-only isolation candidate, then measure self-play + MPS-trainer overlap. |
| Replay-buffer width cheap test | seeded | warm | cpu | After WL5 reports out, run 1.5M vs 750k buffer ablation before bit-packing. |

## Worktree Evidence To Curate

- Main at frontier launch was `a418f677`; the active frontier worktrees are under `/Users/jason/code/gomoku/.frontier/worktrees/20260522T054739Z-*` on branches `frontier/20260522T054739Z/*`.
- `/Users/jason/code/gomoku-perf-extension` (`codex/gomoku-perf-extension`, HEAD `4f21cdd`) holds perf10 production-shaped artifacts: `sweep_logs/perf10-summary.tsv` plus trainer/worker logs for native 8w8g, native 4w16g, and fallback 8w8g.
- `/Users/jason/.codex/worktrees/934b/gomoku` (detached HEAD `b9b9eab`, dirty) contains the Core ML / ANE residency lab: uncommitted `scripts/coreml_ane_residency_scout.py`, `tests/test_coreml_ane_residency_scout.py`, draft `wiki/topics/coreml-ane-residency-lab.md`, and updated ANE rail notes.
- 934b artifacts under `sweep_logs/coreml_ane_residency/` include negative early cells (`*_fixed_fp16_ne` with 0 mW ANE) and later positive Gomoku cells, notably `v3_gomoku_fixed_fused_fp16_b32_ne` (122k pos/s, ANE mean 4061 mW, max 6605 mW) and `v3_gomoku_fixed_fused_fp16_b128_ne` (99.5k pos/s, ANE mean 3683 mW, max 5728 mW). These are unintegrated ANE-metered candidates, not yet production-ready.
- No sibling worker receipt files existed yet under `.frontier/runs/20260522T054739Z/workers/{01,02,03,04}-*/receipt.md` when this curation pass checked; integrate them as they appear.
- Older Claude worktrees currently have no curated perf frontier role; treat them as historical branches unless a lane explicitly revives one.

## Promotion Rule

A lane advances only with a receipt: hypothesis, baseline command, candidate command or blocker, hardware/env, metrics, artifacts, confidence/noise caveat, decision (`promote`, `reject`, `blocked`, or `needs_repeat`), and next action.
