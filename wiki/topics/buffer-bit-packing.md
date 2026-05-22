# Bit-Packed Replay Buffer (Post-WL5 Task)

Scoping doc for shrinking per-position storage in the replay buffer so we
can hold many more games at the same RAM footprint. Captured during WL5
monitoring (2026-05-21).

## Why this is worth doing

Buffer width is a smoothing knob. AlphaZero at scale used very wide
replays to dilute single-version artifacts. Our 1.5M-position buffer
≈ 6,250 games — narrower than even modest published AZ-Go reproductions
(KataGo's published runs use millions of games of context).

Wider buffers smooth target-distribution churn, reduce the "absorption
phase" amplitude when a new lever (like WL5's archive-start) flips on,
and probably reduce dependency on EMA + past-mix kludges (which exist
because the buffer is too narrow).

## Current cost (per position, in current 1.5M buffer)

| field | dtype | bytes |
|---|---|---:|
| planes  | float32 (17, 9, 9) | 5,508 |
| pi      | float32 (81,)      |   324 |
| z       | float32            |     4 |
| side    | int8               |     1 |
| ply     | int16              |     2 |
| **total** | | **~5,839** |

1.5M positions × 5.7 KB = **~8.2 GB on MPS** (matches WL4 checkpoint
buffer footprint).

After 8-fold symmetry augmentation, one game ≈ 30 plies × 8 = 240
positions. So 1.5M positions ≈ **6,250 games**.

## Why current encoding is wasteful

`planes` are literally **binary stones** (0/1). We store them as
float32. That's a 32× overhead. The planes are also AlphaZero-style
history (17 = 8 me + 8 opp + 1 constant), so every plane is sparse-bool
data.

`pi` is a probability distribution over 81 actions. After MCTS visits,
typically only 5-15 actions have meaningful mass. The rest are near-zero
floats. Either FP16 over all 81 or sparse top-K would shrink it 2-7×.

## Target encodings

### Option A: Bit-packed planes + FP16 pi (the practical one)

| field | dtype | bytes |
|---|---|---:|
| packed_planes | uint8 (ceil(17·81/8) = 173 bytes) | 173 |
| pi            | float16 (81,) | 162 |
| z             | float16 | 2 |
| side          | int8 | 1 |
| ply           | int16 | 2 |
| **total** | | **~340** |

**17× smaller** than current. At the same 8.2 GB footprint, that's
~25M positions ≈ **100,000 games** (16× the games at no RAM cost).

At a 10 GB footprint (still trivial on M5 Max with 32-128 GB RAM):
**1M games**, which is the Jason target.

### Option B: Game-level storage + sparse top-K pi (the maximally compact)

Store the game as an action sequence; replay to get planes at sample
time. Store sparse top-K pi per position.

| field | encoding | bytes/game |
|---|---|---:|
| actions | uint8 sequence | 30 |
| sparse pi (K=8 actions+probs per ply, ~24B × 30 plies) | uint8 + float16 | 720 |
| z, side, etc | | 5 |
| **total** | | **~750-1000** |

1M games × 1 KB = **~1 GB** total. Massive headroom.

Cost: per-sample replay (~30 state.apply() calls = ~300 µs at native
speed). Per-batch of 256 = ~80 ms overhead — too slow without batched
replay support.

### Option C: Hybrid — Option A in RAM, Option B on disk

Persist game-level (Option B) to disk for archival. Expand to
bit-packed (Option A) when loading into the working buffer. Lets us
keep a huge offline corpus and a wide in-memory buffer simultaneously.

## Refactor surface

Touches `gomoku/replay_buffer.py` and the training loop's sample path
in `gomoku/train.py`. Estimated ~250 LoC.

1. **Move buffer off MPS to CPU.** MPS allocations are limited by
   MPSGraph INT_MAX (2.147 B elements) — bit-packed 25M positions is
   fine in CPU memory but would hit limits if we tried to pre-allocate
   on MPS. Sample minibatches CPU→MPS per train step.

2. **Add packed storage.**
   ```python
   self.packed_planes: torch.Tensor  # uint8 (cap, 173) on CPU
   self.pi: torch.Tensor             # float16 (cap, 81) on CPU
   self.z: torch.Tensor              # float16 (cap,) on CPU
   self.side: torch.Tensor           # int8 (cap,) on CPU
   self.ply: torch.Tensor            # int16 (cap,) on CPU
   ```

3. **Unpack on sample.** numpy or torch bit-unpack into float32 planes:
   ```python
   def sample(self, batch_size):
       idx = torch.randint(0, self.size, (batch_size,))
       planes = unpack_bits(self.packed_planes[idx])  # → (B, 17, 9, 9) float32
       pi = self.pi[idx].float()
       z = self.z[idx].float()
       # Apply 8-fold symmetry on-the-fly (was previously baked into the buffer)
       planes, pi = random_symmetry(planes, pi, rng)
       # Transfer to MPS
       return planes.to(device), pi.to(device), z.to(device), ...
   ```

4. **Bench**: validate that the unpack + symmetry + transfer doesn't
   meaningfully slow per-cycle wall time. Target overhead: < 10% of
   current train step.

5. **Backward compat loader**: existing 1.5M float32 buffers
   (WL4-era saves) need an unpack-style conversion path on
   `load_state_dict`. Detect by dtype, convert once at load time.

## Storage scaling table

| buffer size | encoding | RAM | games |
|---|---|---:|---:|
| 1.5M positions | current FP32 | 8.2 GB on MPS | 6.3k |
| 1.5M positions | packed | 480 MB on CPU | 6.3k |
| 25M positions | packed | 8 GB on CPU | 100k |
| 75M positions | packed | **25 GB on CPU** | **~300k ← practical target on 48 GB Mac** |
| 240M positions | packed | 80 GB on CPU | 1M (needs 128 GB SKU) |

On Jason's 48 GB M5 Max: baseline OS + apps ≈ 10-15 GB, training
pipeline working set (trainer + 8 workers + eval) ≈ 5 GB. Safe replay
budget is therefore **~25-30 GB**, which lands at **~250-300k games**
of replay — a ~50× widening from current 6,250 games. Going past that
requires a Mac with more RAM.

## Trade-offs vs current

| | current | packed (option A) |
|---|---|---|
| RAM per pos | 5.7 KB | 340 B (**17× smaller**) |
| Sample overhead | ~free (slice + .to(device)) | unpack ~50 µs/sample + symmetry ~50 µs/sample |
| MPS budget | dominates the 8.2 GB device alloc | CPU-only, no MPS impact |
| Augmentation | baked into stored data (×8 already there) | applied on-the-fly per sample |
| Buffer save/restore | direct tensor save | unchanged shape, smaller file |

The sample overhead at batch=256 is ~25 ms — current train step is
~10 ms. That's a 2.5× train-step slowdown if naively implemented.

**Mitigation:** prefetch the next minibatch on a worker thread while
the current train step runs. Hides the unpack cost behind GPU
compute. Standard PyTorch DataLoader pattern.

## When to do this

After WL5 reports out. If WL5's archive-start absorption phase resolves
into a new lower-floor regime, we don't need the wider buffer urgently.
If WL5 plateaus without breakthrough, the buffer-width hypothesis is
the next obvious lever (per WL5 design's "Held-back levers" list:
this isn't listed, but bigger model + bigger sims are, and replay
width is in the same category).

If we ALSO want to land the [ANE INT8 inference](ane-int8-inference.md)
work, do that first — it doesn't conflict with buffer changes and pays
off immediately in self-play throughput.

## Risk: this matters less than expected

A 16× wider buffer (100k vs 6k games) is a meaningful but not
revolutionary smoothing knob. WL4 reached elo 1841 with the 6k-game
buffer; the constraint may be *capacity* (small model) or *search depth*
(200-400 sims) more than buffer width. We should validate that
expanded buffer actually moves a metric before committing the
refactor.

**Cheap test before the refactor:**
1. Train two short ablations from the same checkpoint: 1.5M pos buffer
   vs 750k pos buffer (half).
2. Measure plies, loss bounces, baseline winrate over 500 epochs.
3. If halving the buffer noticeably worsens stability, expanding it is
   a clear win. If it doesn't, buffer width isn't load-bearing right
   now and the refactor can wait.

## Cost estimate

- Bit-pack encoding + tests: 1 day
- Buffer refactor (move to CPU + sample path + backward compat): 1 day
- Perf validation (prefetch thread, end-to-end cycle time check): 0.5 day
- Smoke with new buffer at a moderate width (50k games): 0.5 day
- **Total: ~3 days** if cheap-test confirms the lever is worth it.

## Cross-refs

- [loss-floor-bouncing.md](loss-floor-bouncing.md) — "Concrete lessons
  for the next run" #3 lists `careful replay size/age` as a
  scale-substitute. Wider buffer reduces the need for EMA + past-mix
  workarounds.
- [wl2-scale-emulation-design.md](wl2-scale-emulation-design.md) — EMA
  and past-checkpoint mix were introduced because the buffer was
  narrow. A 16× wider buffer might let us simplify back.
- [ane-int8-inference.md](ane-int8-inference.md) — independent
  post-WL5 task; do that first since it pays off in cycle time
  immediately.
- [project-gomoku-perf-ceiling.md](../../) — current MPS INT_MAX
  ceiling that forced 1.5M position cap originally.
