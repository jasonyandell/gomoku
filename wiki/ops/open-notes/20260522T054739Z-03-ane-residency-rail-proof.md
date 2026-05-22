# Open Note — 20260522T054739Z / 03 ANE residency rail proof

## Receipt

```yaml
lane: ane-residency-rail-proof
hypothesis: Core ML `CPU_AND_NE` labels are insufficient; only same-window powermetrics ANE rail activity can promote a Gomoku-like shape above `coreml-scheduled`.
code_ref: branch frontier/20260522T054739Z/03-ane-residency-rail-proof, base a418f677b831488a71333a3e60d3a0ca7108dbfc plus the lane commit
dataset_ref: synthetic random Core ML inputs; existing Vision positive-control raw rail file /tmp/vision-ane-powermetrics-1779421070.txt
baseline_command: python parser over /tmp/vision-ane-powermetrics-1779421070.txt using scripts/coreml_ane_residency_scout.py::parse_powermetrics_text
candidate_command: python scripts/coreml_ane_residency_scout.py --model-kinds conv,resnet,gomoku --compute-units CPU_ONLY,CPU_AND_NE --compute-precision FLOAT16 --filters 8 --hidden 32 --blocks 1 --depth 1 --batch-size 16 --workers 1 --duration-s 2 --warmup 1 --max-iters 1 --powermetrics required --output sweep_logs/coreml-ane-residency-20260522-lane03/blocked-powermetrics-required.json --raw-dir sweep_logs/coreml-ane-residency-20260522-lane03/blocked-powermetrics-required-raw
hardware: Apple M5 Max, macOS 26.4.1 (Darwin 25.4.0), arm64, Python 3.12.13, torch 2.11.0, coremltools 9.0, MPS available
seed: 0
baseline_metric: existing Vision positive control parsed 25 ANE samples, mean 4474.36 mW, max 4488 mW, active_samples_10mw=25
candidate_metric: required scout blocked before pressure; all conv/resnet/gomoku CPU_ONLY and CPU_AND_NE phases report `sudo -n true failed; cached/passwordless sudo is unavailable`; no same-window powermetrics raw logs produced
delta: no ANE residency delta; only a `coreml-scheduled` smoke ran without powermetrics (CPU_AND_NE smoke positions/sec: conv 42.7k, resnet 41.9k, gomoku 45.3k; ready_workers=1; errors=0; cap remains `coreml-scheduled`)
confidence: high for blocker and harness integration; none for ANE residency because same-window powermetrics could not run
artifacts:
  - scripts/coreml_ane_residency_scout.py
  - tests/test_coreml_ane_residency_scout.py
  - wiki/topics/coreml-ane-residency-lab.md
  - sweep_logs/coreml-ane-residency-20260522-lane03/dry-run.json
  - sweep_logs/coreml-ane-residency-20260522-lane03/coreml-scheduled-smoke.json
  - sweep_logs/coreml-ane-residency-20260522-lane03/blocked-powermetrics-required.json
  - sweep_logs/coreml-ane-residency-20260522-lane03/summary.json
  - /tmp/vision-ane-powermetrics-1779421070.txt
commands_run:
  - git -C /Users/jason/.codex/worktrees/934b/gomoku status --short --branch
  - sudo -n true && echo sudo_cached || echo no_sudo
  - python -m py_compile scripts/coreml_ane_residency_scout.py
  - pytest -q tests/test_coreml_ane_residency_scout.py
  - python scripts/coreml_ane_residency_scout.py --dry-run --model-kinds conv,resnet,gomoku --powermetrics never --duration-s 0.1 --output sweep_logs/coreml-ane-residency-20260522-lane03/dry-run.json
  - python scripts/coreml_ane_residency_scout.py --model-kinds conv,resnet,gomoku --compute-units CPU_ONLY,CPU_AND_NE --compute-precision FLOAT16 --filters 8 --hidden 32 --blocks 1 --depth 1 --batch-size 16 --workers 1 --duration-s 2 --warmup 1 --max-iters 1 --powermetrics required --output sweep_logs/coreml-ane-residency-20260522-lane03/blocked-powermetrics-required.json --raw-dir sweep_logs/coreml-ane-residency-20260522-lane03/blocked-powermetrics-required-raw
  - python scripts/coreml_ane_residency_scout.py --model-kinds conv,resnet,gomoku --compute-units CPU_ONLY,CPU_AND_NE --compute-precision FLOAT16 --filters 8 --hidden 32 --blocks 1 --depth 1 --batch-size 16 --workers 1 --duration-s 2 --warmup 1 --max-iters 5 --powermetrics never --output sweep_logs/coreml-ane-residency-20260522-lane03/coreml-scheduled-smoke.json --raw-dir sweep_logs/coreml-ane-residency-20260522-lane03/coreml-scheduled-smoke-raw
  - pytest -q
decision: blocked
next_action: Re-run when `sudo -n true` passes; capture a fresh Vision positive control and a fresh powermetrics-required conv/resnet/gomoku scout in adjacent windows, then cap each cell by rail evidence before any production overlap work.
```

## Files touched

- Added `scripts/coreml_ane_residency_scout.py` from detached 934b worktree.
- Added `tests/test_coreml_ane_residency_scout.py`.
- Added and updated `wiki/topics/coreml-ane-residency-lab.md`.
- Updated `wiki/topics/ane-int8-inference.md`, `wiki/topics/m5-max-as-mainframe.md`, `wiki/index.md`, and `wiki/log.md` with the rail-evidence rule and blocker.

## Result

The harness is integrated and validated, and it can create `coreml-scheduled` receipts. The actual `ane-metered` lane is blocked because this session cannot run privileged powermetrics (`sudo -n true` fails). No Gomoku/Core ML shape is called ANE-backed.

## Board-update recommendation

Keep the lane active/hot but mark current worker result `blocked-on-sudo-powermetrics`. Next worker should run the exact candidate command plus a fresh Vision positive control after authenticating sudo.
