# WL5 — Diagnostics + Archive-Start Design

Status note (2026-05-21): implemented and launched as WL5 (`o6cbjfnr`).
Phase 1 (e4001-e5051) validated the diagnostic streams and archive-start
pipeline without NaN, worker death, or fast-attack collapse, but did not beat
WL4's ATH. Phase 2 is the same run after self-play workers were hot-restarted
with Conv+BN-fused inference. Treat this page as the preserved WL5 design
record; read [training-run-lineage.md](training-run-lineage.md) and the WL5
tail of [../../TRAINING_WIKI.md](../../TRAINING_WIKI.md) for current status.

Design recorded 2026-05-21 by Jason and assistant after WL4 reached the
"healthy lower-floor-bouncing" plateau described in
[loss-floor-bouncing.md](loss-floor-bouncing.md). WL4 produced the best
WL-series outcome to date (elo 1841 ATH, la4=100%, plies past Z's
endpoint) but plateaued without further breakthrough.

The article's "Candidate Next-Run Shape" specifies measurement first,
then one behavioral lever (archive-start diversity). This doc commits
to running both in a **single WL5 cell** that **resumes from WL4 e4024**
and uses a **static archive mined from WL4 artifacts**.

Cross-refs:
- [loss-floor-bouncing.md](loss-floor-bouncing.md) — the article driving
  this design; "Candidate Next-Run Shape" + "Next-Run Lessons From The
  Sources" sections are the proximate spec.
- [wl2-scale-emulation-design.md](wl2-scale-emulation-design.md) — the
  template this design follows in structure.
- [wave-of-lockstep-design.md](wave-of-lockstep-design.md) — WL1 design.
- [../../TRAINING_WIKI.md](../../TRAINING_WIKI.md) "WL4 plateau-end" entry —
  evidence trail for the plateau being healthy and the next-lever rationale.

## Hypothesis

WL4 confirmed that **opening diversity was necessary but not permanent
training infrastructure**. The remaining plateau is structural: the model
has saturated what it can learn from canonical-opening self-play with
its current capacity and search depth. The article's first-suspect
mechanism for the structural limit is **Go-Exploit-style state coverage
gaps** — initial-state self-play under-covers deeper or rarer states
that fixed external opponents do exercise.

WL5 tests this in two parts:

1. **Diagnostic question** (always-on instrumentation): is the loss
   floor a *target-distribution* effect (H(pi_mcts) moving) or a
   *learning* effect (KL(pi_mcts || p_net) not closing)? Are losses
   uniform across color and ply, or concentrated in specific regions?
2. **Behavioral question** (archive-start lever): does seeding a
   minority fraction of self-play games from curated trouble states
   (instead of canonical empty board) move the loss floor down AND
   improve fixed-baseline strength?

If diagnostics show the floor is target-entropy bouncing (not KL-bound),
that's confirmation the article's reading is right and points us toward
either capacity (bigger model) or smarter exploration as the next axis.

If archive-start moves the floor down with the same diagnostics
signatures, that validates Go-Exploit-style coverage-gap fixing as a
laptop-scale lever.

## Parent state and lineage

- **Resume from:** WL4 e4024 (`sweep_runs/WL4-no-random-openings.plateau-e4024/checkpoints/latest.pt`,
  8.2G — has model + EMA + full 1.5M buffer)
- **wandb:** new run id (force fresh — see "Handoff friction" #3 in the
  launch runbook; cell rename → new dirs makes a clean wandb timeline
  for the WL5-specific charts even though weights continue from WL4)
- **Buffer:** inherited from WL4 — 1.5M positions of K=0 self-play data
- **All WL3.1/WL4 levers stay:** EMA τ=0.99, past-mix 0.4/0.1, poll
  jitter 2-8s, grad-accum 4×, random_opening_moves=0

## Architecture: diagnostics layer

Three independent instrumentation streams, all opt-in via flags.

### Stream 1: Fixed validation archive

Frozen set of ~500-2000 positions with their MCTS targets and ground-truth
values. Scored every `eval_every` cycles. Provides a stationary measure
of policy quality independent of the moving self-play distribution.

**Format** — `archives/wl5_validation_v1.pt`:
```python
{
    "planes":   torch.Tensor (N, N_INPUT_PLANES, BOARD, BOARD),  # float32
    "pi_mcts":  torch.Tensor (N, N_ACTIONS),                     # float32, sums to 1
    "z":        torch.Tensor (N,),                               # float32 in [-1, 1]
    "provenance": list[str] of length N,  # tag for each position
    "side":     torch.Tensor (N,) int,    # 0=black, 1=white at this position
    "ply":      torch.Tensor (N,) int,    # ply count when captured
}
```

**Provenance buckets** (drives per-source breakdowns):
- `heuristic_loss` — positions where WL4 lost to heuristic in eval
- `lookahead2_loss` — vs depth-2 alpha-beta
- `lookahead4_loss` — vs depth-4 alpha-beta
- `high_kl` — positions where WL4 buffer entries had high KL(net || mcts)
- `long_defense` — positions deep in long-defensive games (ply > 30)
- `canonical_opening` — first 5-10 plies of canonical-line games

Target counts: ~200-400 per bucket. Total ~1000-2000.

**Wandb metrics** (per eval cycle):
- `val/policy_ce` — overall cross-entropy
- `val/policy_kl` — KL(pi_mcts || p_net)
- `val/value_mse` — value head MSE vs z
- `val/policy_acc` — argmax accuracy
- `val/policy_ce/<provenance>` — per-bucket cross-entropy
- `val/policy_kl/<provenance>` — per-bucket KL
- `val/value_mse/<provenance>` — per-bucket value MSE

CLI flag: `--validation-archive-path <PATH>`. Defaults None (disabled).

### Stream 2: Policy-loss decomposition

Inside the training step, split policy cross-entropy into its components:
```
policy_ce = H(pi_mcts) + KL(pi_mcts || p_net)
```

Log each separately per minibatch (averaged across the cycle):
- `train/policy_target_entropy` — H(pi_mcts), the irreducible floor
- `train/policy_net_entropy` — H(p_net), how sharp the network is
- `train/policy_kl` — KL(pi_mcts || p_net), the learning gap

These let us tell "target moved" (target entropy changed) from "net
fell behind target" (KL grew). The article's central interpretive
distinction.

Always-on; no flag needed.

### Stream 3: Per-color and per-ply-bucket metrics

The Tablut paper (cited in the article) warned about role asymmetry
masking as forgetting. The buffer already tags weight_version; need to
add side + ply at sample time.

**Implementation:**
- `ReplayBuffer.add()` already receives examples; tag each with `side`
  (0 or 1) and `ply` (int, capped at game length).
- `ReplayBuffer.sample()` returns `side` and `ply` alongside planes/pi/z.
- Trainer logs per-color and per-ply-bucket losses each cycle.

**Wandb metrics:**
- `train/policy_ce/side_0`, `train/policy_ce/side_1`
- `train/value_mse/side_0`, `train/value_mse/side_1`
- `train/policy_ce/ply_00_10`, `train/policy_ce/ply_10_25`, `train/policy_ce/ply_25_60`
- Same buckets for value_mse

Always-on. Cost: ~32 bytes per buffer slot for side+ply tags.

## Architecture: archive-start lever

Worker-side mechanism. At game start, with probability `archive_start_frac`,
load a random position from the archive instead of starting from empty
board.

**Mechanics:**
- Each game start in `_generate_games_native`: roll U(0,1).
- If `< archive_start_frac` AND archive loaded: pick uniform-random
  position from archive, initialize game state with the archive's planes,
  side-to-move, and play history.
- Otherwise: start from empty board (canonical K=0 behavior).
- Training examples ARE recorded for all moves after the archive start
  (model picks every move from this state forward). The model learns
  from real MCTS-decided moves on positions it would otherwise not
  reach from canonical play.

**Why this isn't just "another opening randomization":**
- Random opening plies (K=2) put games into positions reachable via 2
  uniform-random first moves — typically structureless early positions.
- Archive-start uses positions that came from real adversarial play
  (positions where the model actually lost or was uncertain). The model
  is forced to learn how to handle the specific positions that have
  been giving it trouble.
- Closer to KataGo's "playout cap randomization" + Go-Exploit's
  archived-state-starts than to random openings.

**CLI flags:**
- `--archive-start-path <PATH>` — same archive file as validation (or
  separate; default same)
- `--archive-start-frac <P>` (default 0.0; recommended live value 0.15)

If both `--random-opening-moves` and `--archive-start-frac` are set:
each game first rolls archive-vs-canonical; if canonical, then applies
random-opening-moves. (For WL5, random_opening_moves=0 so no
interaction.)

## Implementation plan

### Step 1: Archive mining script (~200 LoC, separate file)

`scripts/mine_validation_archive.py`:
- Inputs: `--wl4-checkpoint <PATH>`, `--wl4-records-dir <PATH>` (the
  preserved WL4 paused dir), `--output <PATH>`, target counts per
  bucket.
- For each bucket:
  - **heuristic_loss / lookahead{2,4}_loss**: load WL4 final checkpoint,
    play N games vs each baseline, collect the (planes, pi_mcts, z, side, ply)
    of the position right before the model's losing move. ~200-400 positions
    per baseline.
  - **high_kl**: scan WL4's late buffer (if accessible) or replay WL4
    self-play to recover KL distribution; pick top-K by KL.
  - **long_defense**: replay WL4 self-play games, sample positions where
    ply > 30 and game length > 50.
  - **canonical_opening**: hand-construct or sample from first-5-plies
    distribution of self-play games.
- Output: torch.save dict matching the format above.
- Validation: roundtrip check, NaN check, mass-sums-to-1 check.

Cost: ~30 minutes wall to mine (sequential vs baselines on CPU).

### Step 2: Trainer instrumentation (~150 LoC, `gomoku/train.py`)

- Add `--validation-archive-path` flag, load archive at startup.
- New `_score_validation_archive()` function called every eval cycle.
  Computes per-bucket metrics, logs to wandb.
- In the per-minibatch training step, decompose policy_loss:
  - Already computing `policy_loss = -sum(pi * log_softmax(logits))`.
  - Add: `target_entropy = -sum(pi * log(pi + eps))`,
    `kl = policy_loss - target_entropy`.
  - Log batch-averaged values.
- After each cycle, log per-color and per-ply-bucket losses (need buffer
  changes from Step 3).

### Step 3: Buffer side + ply tagging (~80 LoC, `gomoku/replay_buffer.py`)

- New tensors: `self.side` (int8, capacity), `self.ply` (int16, capacity).
- `add()` accepts examples with side + ply; tags them.
- `sample()` returns them.
- `save_buffer` / `restore_buffer` includes them.
- Backward-compat: when loading an old buffer without side/ply, default
  to 0; per-color/per-ply metrics will be uninformative until a few
  ingest cycles fill them in. Acceptable for WL5 since we're resuming
  from WL4's buffer.

### Step 4: Worker archive-start (~80 LoC, `gomoku/selfplay_worker.py` + `gomoku/self_play.py`)

- New flags: `--archive-start-path`, `--archive-start-frac`.
- At worker startup: load archive into memory if path given.
- In `_generate_games_native`'s per-game loop, before initializing the
  game state: roll U(0,1); if < frac, sample a random archive index and
  initialize the native MCTS game state from that snapshot.
- For the native game state init, need `NativeMCTSGame.from_planes(planes, side, history)`
  — if the C extension doesn't already support this, add a Python-side
  workaround (replay the implied move sequence from empty board).

### Step 5: Cell wiring (~30 LoC, `scripts/run_sweep.py`)

- Add 5 new Cell fields: `validation_archive_path`, `archive_start_path`,
  `archive_start_frac` (plus diagnostic-related fields if needed).
- Wire trainer_cmd + worker_cmd appropriately.
- Add WL5 cell: clone of WL4 + the new flags pointing at the mined
  archive + `archive_start_frac=0.15`.

### Step 6: Tests (~100 LoC)

- Archive load/save roundtrip.
- Per-color metrics correctness on a synthetic buffer.
- H/KL decomposition matches policy_loss (sum check).
- Archive-start mechanism produces games that start from archived
  positions (not empty board).

### Step 7: Smoke (30 epochs, like prior runs)

- Validate all metrics appear in wandb.
- Archive-start fraction matches configured value (count by `mix_source`
  or new equivalent in payload).
- No crashes from buffer side/ply changes.

**Total LoC:** ~640 lines + tests. Bigger than WL2 (~120). Worth
parallelizing across 2-3 agents.

## Held-back levers (for follow-up runs, NOT WL5)

If WL5 plateaus too, the next-after candidates:

1. **Bigger model** (small 324k → medium ~1M). Pure capacity bet.
2. **Higher sim count** (400 → 800). Search depth.
3. **Dynamic archive growth** (archive updates as WL5 finds new trouble).
4. **KataGo-style playout-cap randomization + policy target pruning**.
5. **Multi-population training** (N models trained against each other).

Held back because WL5 already has 3 streams of instrumentation + one
new lever. Don't add more variables until we see what these tell us.

## Why we expect this to work

The diagnostic streams alone should answer: "what's the actual nature
of the WL4 plateau?"
- If `train/policy_kl` floors near zero while `train/policy_target_entropy`
  bounces: target-distribution noise, not a learning bug. Confirms the
  article's reading.
- If KL doesn't close: there's a fittability gap, points at capacity.
- If per-color metrics diverge sharply: role asymmetry confirmed,
  points at Tablut-style mitigations.
- If long-defense val/policy_ce is much worse than canonical-opening:
  state-coverage gap confirmed, archive-start should help.

The archive-start lever then either moves the floor down (validates
the coverage-gap hypothesis) or doesn't (rules it out, points elsewhere).

Either way, WL5 produces actionable next-direction information that
WL4's plateau alone doesn't.

## Cost estimate

- Implementation: ~640 LoC + tests; 2-3 agents in parallel, ~30-45 min
  wall to land all of it.
- Archive mining: ~30 min wall (separate script run before launch).
- Smoke: ~10 min for 30 epochs.
- Full run: same as WL4 budget (5000 epochs from resume; ~5-7h at
  current cycle rates).
- Diagnostics overhead in training: ~2-5% per cycle (mostly the
  validation-archive scoring; per-batch decomposition is microseconds).

## Sanity test before full launch

Standard WL-series smoke pattern:
- Cell `WL5` with `archive_start_frac=0.15`, validation archive loaded
- 30 epochs, 4 workers (cheap)
- Verify: all 3 diagnostic streams appear in wandb; archive-start games
  detectable in worker logs; no NaN / crash; per-color metrics non-zero
- Wipe smoke dir, launch full WL5 (8 workers, --epochs 5000)

## Open questions for the implementation session

- Does `NativeMCTSGame` already support `from_planes`-style initialization?
  If not, use a Python-side replay-from-empty workaround or add to the
  C extension. (Probably the workaround is fine for first-pass.)
- Archive size: start at ~1000 positions or push to ~5000? Larger gives
  more variety per epoch, but if too large the archive-start sample is
  too narrowly distributed within any one cycle. Probably 1000-2000 is
  the sweet spot.
- High-KL bucket: do we have WL4 buffer state to mine, or do we need
  to re-generate? (The paused dir has latest.pt which includes the
  buffer — extractable via `buffer.state_dict()`.)
- Should the validation archive and the archive-start source be the
  same file? Default yes (simpler) but they could differ — e.g.
  validation = small held-out set, archive-start = larger pool.

## References

- [loss-floor-bouncing.md](loss-floor-bouncing.md) — the article driving this
- [TRAINING_WIKI.md WL4 plateau-end](../../TRAINING_WIKI.md) — evidence
- [Trudeau & Bowling 2023, Go-Exploit](https://arxiv.org/abs/2302.12359)
  — the explicit prior for archive-start
- [Wu 2020, KataGo](https://arxiv.org/abs/1902.10565) — playout cap
  randomization + policy target pruning (held back from WL5)
- [Lees & Matiisen 2026, Tablut reproduction](https://arxiv.org/abs/2604.05476)
  — role asymmetry mitigation via 25% past-checkpoint sampling
