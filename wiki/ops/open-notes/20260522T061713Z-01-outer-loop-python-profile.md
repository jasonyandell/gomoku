# 20260522T061713Z-01 — Outer self-play loop profiling

## Receipt

```yaml
lane: outer-loop-python-profile
hypothesis: After native MCTS and eval fusion, remaining Python outside native search is large enough to justify another outer-loop native/format pass.
code_ref: frontier/20260522T061713Z/01-outer-loop-python-profile at a14191f plus this profiling instrumentation commit
dataset_ref: fresh self-play from a freshly initialized small stem_padding=1 checkpoint; no training dataset or strength claim
baseline_command: python -m gomoku.selfplay_worker --weights-path sweep_logs/outer-loop-profile-20260522T061713Z/checkpoints/worker_weights.pt --output-dir sweep_logs/outer-loop-profile-20260522T061713Z/records-wave --worker-id profile --device mps --games-per-batch 8 --n-simulations 400 --wave-size 64 --max-plies 16 --wave-mode --seed 0 --max-batches 1 --profile-output sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave-mode-8g-s400-p16.json
candidate_command: no implementation candidate promoted; same-shape verifier for any future candidate is the same worker command and JSON profile diff, preferably repeated 3x
hardware: macOS-26.4.1 arm64; Apple M5 Max class machine; device=mps; Python 3.12.13
seed: 0
baseline_metric: wave-mode bounded worker wall=1.064s for 8 games / 128 plies / 1024 augmented examples; native_search_batch=1.013s (95.2% wall); evaluator=0.896s (84.3% wall); native_search_excluding_evaluator=0.117s (11.0% wall); post_search_python=0.050s (4.7% wall); file_handoff=0.034s (3.2% wall); record_build=0.011s (1.0% wall); D4=0.0087s (0.82% wall); sample_action=0.0032s (0.30% wall)
candidate_metric: non-wave cross-check wall=1.235s; evaluator=86.9%; native_search_excluding_evaluator=8.7%; post_search_python=4.4%; no post-search Python owner exceeds file handoff at ~3%
delta: no 10-20% outer-loop Python opportunity found; deleting all measured post-search Python would cap at ~4-5% on this shape, while evaluator plus native search boundary owns ~95%
confidence: medium-low; one bounded MPS run per shape on fresh random weights, but production-shaped 8g/400sims/wave64/max_plies16 and enough to reject a large post-search-Python pass
artifacts:
  - sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave-mode-8g-s400-p16.json
  - sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave-mode-8g-s400-p16.log
  - sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave8-s400-p16.json
  - sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave8-s400-p16.log
  - sweep_logs/outer-loop-profile-20260522T061713Z/cpu-smoke.txt
  - sweep_logs/outer-loop-profile-20260522T061713Z/pytest-q.txt
commands_run:
  - python -m py_compile gomoku/self_play.py gomoku/selfplay_worker.py
  - python - <<'PY' ... build_model('small', stem_padding=1); save_checkpoint('sweep_logs/outer-loop-profile-20260522T061713Z/checkpoints/worker_weights.pt', m, epoch=0) ... PY
  - python -m gomoku.selfplay_worker --weights-path sweep_logs/outer-loop-profile-20260522T061713Z/checkpoints/worker_weights.pt --output-dir sweep_logs/outer-loop-profile-20260522T061713Z/records --worker-id profile --device mps --games-per-batch 8 --n-simulations 400 --wave-size 64 --max-plies 16 --seed 0 --max-batches 1 --profile-output sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave8-s400-p16.json
  - python -m gomoku.selfplay_worker --weights-path sweep_logs/outer-loop-profile-20260522T061713Z/checkpoints/worker_weights.pt --output-dir sweep_logs/outer-loop-profile-20260522T061713Z/records-wave --worker-id profile --device mps --games-per-batch 8 --n-simulations 400 --wave-size 64 --max-plies 16 --wave-mode --seed 0 --max-batches 1 --profile-output sweep_logs/outer-loop-profile-20260522T061713Z/profile-mps-wave-mode-8g-s400-p16.json
  - python scripts/perf_microbench.py --device cpu --size tiny --games 2 --n-simulations 2 --wave-size 1 --max-plies 2 --repeats 1 --warmup 0
  - pytest -q
decision: reject
next_action: Do not start another native pass for action sampling, trajectory staging, D4, record creation, or worker file handoff yet. Focus next perf work on evaluator/engine overlap or native_search_batch internals; if file handoff is revisited, use the same JSON profile command as a verifier and require a repeated >=10% wall win.
```

## Files touched

- `gomoku/self_play.py`: optional `profile` timing hooks for native self-play generation.
- `gomoku/selfplay_worker.py`: added `--max-plies` bounded-worker cap and `--profile-output` JSON profile writer.
- `wiki/ops/open-notes/20260522T061713Z-01-outer-loop-python-profile.md`: this note.

## Result

Top owners in the production-shaped wave-mode profile:

| owner | seconds | wall share |
| --- | ---: | ---: |
| evaluator callback | 0.896 | 84.3% |
| native_search_batch total | 1.013 | 95.2% |
| native_search_batch excluding evaluator | 0.117 | 11.0% |
| all measured post-search Python | 0.050 | 4.7% |
| file handoff | 0.034 | 3.2% |
| record build | 0.011 | 1.0% |
| D4 augmentation | 0.0087 | 0.82% |
| action sampling | 0.0032 | 0.30% |

No blocker. No optimization candidate in the requested post-search Python surface clears the 10-20% bar.

## Board-update recommendation

Curator can close `Outer self-play loop profiling` as `completed/reject-no-op`: post-search Python is too small in this bounded production shape. Recommend next hot lane be evaluator/engine overlap or deeper native_search_batch internals, not D4/action-sampling/file-handoff.
