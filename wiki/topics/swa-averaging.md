# SWA peak averaging (post-training tool)
> **Status: HISTORICAL** *(2026-07-04)* — derby-era tool (built 2026-05-28, no recorded win).

`scripts/swa_average.py` — average the last K saved `peak*.pt` checkpoints in a
lane's `_peaks/<idea>/` directory into a single `peak_swa.pt`. The averaged
checkpoint is a free new contestant the GPU runner can H2H against the
un-averaged peak. Built per bead **derby-79y** (2026-05-28).

Source: Izmailov et al. 2018,
[*Averaging Weights Leads to Wider Optima and Better Generalization*](https://arxiv.org/abs/1803.05407).
True SWA averages **saved checkpoints, POST-training** — distinct from our
`--ema-tau 0.99` (training-time EMA in the gen workers). The paper reports
consistent 10-30 elo / 1-2pp accuracy lifts in supervised settings; the RL
signal is weaker but the cost is ~free.

## When to invoke

- A lane has 10+ `peak*.pt` files (champion continuation board, long matured
  lanes, etc.) and you want a free-elo H2H contestant.
- Bottom of the cost × probability-of-success curve: seconds of CPU + one H2H
  round.

## CLI

```bash
python scripts/swa_average.py \
  --lane-dir sweep_runs/<lane>/_peaks/<idea>/ \
  --k 10 \
  [--output <lane-dir>/peak_swa.pt] \
  [--dry-run]
```

- Discovers `peak*.pt` files in `--lane-dir`, sorted oldest→newest by embedded
  `epoch` metadata (falling back to mtime). Prior `peak_swa*.pt` outputs are
  excluded so re-running doesn't feed the tool's own output back in.
- If fewer than K peaks exist, averages all available (with a warning).
- Loads each peak with `torch.load(map_location='cpu')` — no GPU/MPS touched.
- Validates that every peak shares the same key set and tensor shapes; refuses
  loudly on any mismatch (no partial output written).

## What gets averaged

`peak.pt` is the standard `gomoku.model.save_checkpoint` payload:
`model_state_dict`, `model_config`, `epoch`, `total_games`, optional
`optimizer_state_dict`, `wandb_run_id`. The tool:

- Averages **only `model_state_dict`** — element-wise uniform mean across the K
  selected peaks.
  - Floating tensors: accumulated in float32, divided by K, cast back to the
    source dtype.
  - Integer tensors (e.g. BN's `num_batches_tracked`): taken from the most
    recent peak — averaging a counter to a non-integer is wrong.
- **Drops `optimizer_state_dict`** — averaging running moments across runs is
  not meaningful.
- **Preserves non-weight metadata** from the most recent peak (`model_config`,
  `epoch`, `total_games`, `wandb_run_id`) so the output is loadable by the
  existing trainer/eval pipeline (`gomoku.model.load_checkpoint`).
- Adds a `swa` metadata block: `{k, method, source_peaks, source_epochs,
  created, lane_dir, note}`.

## BatchNorm caveat

Our model **has** `nn.BatchNorm2d` (running_mean, running_var,
num_batches_tracked) in `gomoku/model.py`. The standard SWA recipe recommends
re-estimating BN stats with a fresh forward pass over training data after
averaging — we don't do that here (CPU-only offline tool, no data, no GPU).
Instead, the BN running stats are themselves averaged element-wise across
peaks, which is a reasonable approximation given that each peak's stats are
already a training-time EMA of the same data distribution.

If the averaged checkpoint underperforms its individual sources at H2H, the BN
recalibration variant (run a few hundred batches through the averaged model in
training mode, then snapshot) is a follow-up lever — out of scope for this
bead.

## Out of scope (deferred)

- "Running SWA" — integrating averaging into the training loop (Izmailov's
  second variant).
- BN recalibration with a real data pass.
- A K sweep — the GPU runner can vary K at invocation time.
