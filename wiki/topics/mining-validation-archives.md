# Mining Validation Archives

> **Status: LIVE** *(2026-07-04)* operational how-to (training-diagnostics / archive-mining
> cluster). Despite the name, this is **NOT part of the seek-VCT program** — "mining"
> here means mining a checkpoint for a *validation-archive* of positions, unrelated to
> VCT solving.

Operational recipe for `scripts/mine_validation_archive.py`, the script that
turns a finished training checkpoint into a frozen archive of positions used
for two things:

1. **Diagnostics** — trainer scores the archive every eval cycle and logs
   `val/policy_ce`, `val/policy_kl`, `val/value_mse`, `val/policy_acc` plus
   per-bucket variants. Separates *target-distribution noise* from *learning
   gap* in the policy loss. See
   [loss-floor-bouncing.md](loss-floor-bouncing.md) and
   wl5-diagnostics-archive-start-design.md *(removed 2026-07-04; recover:
   `git show ca76350:wiki/_archive/topics/wl5-diagnostics-archive-start-design.md`)*.
2. **Behavioral lever** — workers seed a fraction of self-play games from
   the archive (Go-Exploit pattern) instead of empty board.

## TL;DR command

For a 5-bucket archive (~1000-2000 positions) from a WL-series checkpoint,
running on MPS:

```bash
cd ~/code/gomoku
PYTHONUNBUFFERED=1 GOMOKU_DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1 \
  uv run python -u scripts/mine_validation_archive.py \
    --wl4-checkpoint sweep_runs/<RUN-NAME>/checkpoints/latest.pt \
    --output archives/<NAME>_v1.pt \
    --target-per-bucket 200 \
    --mcts-sims 200 \
    --batch-games 64 \
    --wave-size 8 \
    --max-rounds 40
```

`-u` and `PYTHONUNBUFFERED=1` are both needed because the script flushes
its `print` calls — without them, log lines stay in stdout's block buffer
until the process exits, making it impossible to monitor progress.

`PYTORCH_ENABLE_MPS_FALLBACK=1` is required because a few ops fall back to
CPU on MPS (e.g. argsort, certain reductions). Without it the script crashes
mid-bucket.

## Buckets

| Tag | How it's mined | Cost driver |
|---|---|---|
| `hard_kl_selfplay` | Batched self-play games; rank ALL model-move examples by `KL(p_net || pi_mcts)`; keep global top-K. | Cheap — one batched MCTS round per opponent move + a single batched forward pass to score KL. |
| `hard_kl_heuristic` | Same against the heuristic opponent. | Similar to selfplay. |
| `hard_kl_lookahead2` | Same against depth-2 alpha-beta. | Depth-2 opponent is cheap per move. |
| `hard_kl_lookahead4` | Same against depth-4 alpha-beta. | **Bottleneck**: depth-4 opponent is CPU-serial per game (see "perf gotchas"). |
| `long_defense` | Batched self-play; harvest positions with `ply > 30` from games of `plies > 50`. | Cheap once games exist; bound by the fraction of long games. |
| `canonical_opening` | Batched self-play; harvest positions with `ply < 10`. | Cheap; every game contributes. |
| `high_kl` | Scan replay buffer in WL4 checkpoint; pick top-K by `KL(p_net || pi_mcts)` under the WL4 model. | Single forward pass over up to `target × 50` sampled positions — seconds. Requires checkpoint to still have `replay_buffer` (the trainer drops the buffer past `keep_last_n`, so save the last buffered checkpoint explicitly when ending a run). |

### Why "hard_kl" instead of "lost games"

The original design (now retired) captured positions from games where the
model LOST against each baseline. That breaks for any model strong enough
to never lose: WL4 vs heuristic = 88-100% winrate, so `heuristic_loss`
returned **0 positions** across 2560 games before the bucket strategy was
changed.

`hard_kl_*` instead asks "where does the model's prior most disagree with
what MCTS concluded after 200 sims of search?" That always returns
something — even a saturated model has positions where its raw policy is
imperfect compared to its searched policy. It's also a *better* diagnostic
signal: it captures "what the network needs to learn" rather than "what
the network outright fails at."

The `KL(p_net || pi_mcts)` direction is intentional: it penalizes
positions where the *net is confidently wrong* about MCTS, not just where
the two distributions differ. Symmetric distance would conflate "MCTS
explored more options" with "net got it wrong."

## Throughput notes (2026-05-21 measurements)

Single in-process run on M5 Max, MPS evaluator, WL4 checkpoint, with the
`hard_kl` strategy and `--max-rounds 3`:

- `hard_kl_selfplay` (target=200): ~2 min — 3 rounds × 64 games.
- `hard_kl_heuristic` (target=200): ~2 min.
- `hard_kl_lookahead2` (target=200): ~2 min.
- `hard_kl_lookahead4` (target=200): ~5 min — depth-4 alpha-beta on every
  opponent move is CPU-serial; the GPU sits idle during opponent search.
- `long_defense` + `canonical_opening` (target=200 each): ~2 min combined.
- `high_kl`: seconds (single forward pass over buffer chunks).

Total wall: roughly **13-15 min** for a 7-bucket × 200-position archive.

### Perf gotcha: lookahead4 GPU idle

In `gomoku/self_play.py:_generate_games_vs_baseline` the opponent picker
is called serially per game during opponent turns (one Python iteration
per game). For `heuristic` and `lookahead:depth=2` that's microseconds
and dominated by MCTS rounds (GPU saturated). For `depth=4` it's
50-200ms of CPU alpha-beta per call, and the GPU sits idle while it
thinks.

If lookahead4 mining becomes the wall-time bottleneck, parallelize the
opponent picker via `multiprocessing.Pool` or a thread pool. Numpy
releases the GIL during heavy ops, so a thread pool may be sufficient
without pickle headaches. Not implemented yet — mine wall is acceptable
as-is.

## Knobs that actually move the needle

- **`--target-per-bucket`**: linear effect. Drop to 60-100 if a smaller
  archive is acceptable. Diagnostic signal stabilizes well below 200; the
  archive's main purpose is to give per-bucket trends, not high-resolution
  CIs.
- **`--mcts-sims`**: linear on per-move cost. 200 matches production-ish
  pi quality; 100 is acceptable for diagnostics.
- **`--batch-games`**: bigger = better MPS saturation. 64 is a good
  default; 128-256 may help on machines with idle MPS headroom but with
  the hard_kl strategy each batch already yields plenty of positions.
- **`--max-rounds`**: max number of batches per `hard_kl_*` bucket. The
  script also bails out after round 3 if the top-K is stable; setting
  this above 3 has no effect with current logic.

## Anti-patterns (don't do these)

1. **Don't fan out N parallel mine processes against the full checkpoint.**
   The WL4 checkpoint is 8.2 GB (it includes the 1.5M replay buffer). N
   processes each `torch.load` independently → N × 8 GB of memory. If you
   need parallel mining (rare; the in-process batched path is already
   fast), pre-strip the buffer with a one-liner and use the slim
   checkpoint for the game-playing buckets, then run `high_kl` once from
   the full checkpoint.

2. **Don't run without `-u` or `PYTHONUNBUFFERED=1`.** The script's
   per-bucket prints stay in stdout's userspace block buffer for the
   entire run, so you can't tell whether it's making progress vs hung.

3. **Don't run on CPU** unless training is actively using MPS. MPS is
   ~3-5x faster for this workload (MCTS leaf evaluation is batched
   forward passes; that's exactly MPS's wheelhouse).

4. **Don't co-run with training.** Mining and training both want MPS;
   they'll fight and slow each other down. Mine BEFORE the run you're
   launching, or AFTER stopping the previous run.

## Output format

```python
{
    "planes":     torch.Tensor (N, 17, 9, 9) float32,
    "pi_mcts":    torch.Tensor (N, 81)        float32, sums to 1
    "z":          torch.Tensor (N,)           float32 in [-1, 1],
    "provenance": list[str] of length N,      # bucket tag
    "side":       torch.Tensor (N,)           int8,  # 0=black-to-move at position
    "ply":        torch.Tensor (N,)           int16, # ply count when captured
}
```

Validated by `_validate`: no NaN, pi rows sum to 1 within 1e-3.

Cell wiring in [`scripts/run_sweep.py`](../../scripts/run_sweep.py) reads
the archive via `validation_archive_path` (trainer-side diagnostics) and
`archive_start_path` (worker-side behavioral lever); both can point at the
same file or differ.

## When to re-mine

- **Every new big run** if archive-start is enabled: positions mined from
  an old model are a stale snapshot of "what was hard." A model that has
  moved well past that snapshot may not gain much from those positions.
- **Not every checkpoint** during a run: the validation scoring uses a
  fixed reference so it's meaningful to compare across cycles. Re-mining
  mid-run defeats the purpose of "fixed validation set."
- **Once per training era** is the sweet spot: mine at the end of run N,
  use during run N+1.

## Cross-refs

- wl5-diagnostics-archive-start-design.md *(removed; see note above)* —
  why the archive exists and how WL5 uses it.
- [loss-floor-bouncing.md](loss-floor-bouncing.md) — the article that
  motivated the diagnostic side of the archive.
- [launch-sequence-runbook.md](launch-sequence-runbook.md) — fits mining
  into the pre-launch checklist.
