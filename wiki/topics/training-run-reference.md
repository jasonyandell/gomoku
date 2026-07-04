# Training-Run Reference — the zillion knobs and switches
> **Status: LIVE** *(2026-07-04)* — the parameter dictionary (length is its job).

The definitive launch/tune surface for a training run. **Quick-start first**
(the handful of commands + knobs you actually reach for), then a
**comprehensive** knob reference grouped by subsystem. When you need to *launch,
resume, monitor, or stop* a run, use [[launch-sequence-runbook]] for the
procedure; this page is the parameter dictionary that sits behind it.

Three layers of configuration, in precedence order:

1. **`Cell`** dataclass fields in `scripts/run_sweep.py` (the sweep point;
   `scripts/run_sweep.py:53`). A cell fans out to 1 trainer +
   `n_workers` self-play workers + (optionally) 1 eval worker, deriving all
   dir/run names from `cell.name`.
2. **`extra_train_args` / `extra_worker_args`** — free-form flag lists on the
   Cell that pass straight through to `gomoku.train` / `gomoku.selfplay_worker`.
   Every lever that isn't a first-class Cell field lives here (teacher flags,
   `--sgd-steps-per-epoch`, `--pack-buffer`, gumbel, value-discount…).
3. **Process env** — `GOMOKU_BOARD_SIZE` (board size is NOT a Cell field),
   `GOMOKU_DISABLE_NATIVE_*`, `GOMOKU_DROP_OPENERS`.

---

## QUICK-START

### The commands you use

```bash
# List every defined cell:
uv run python scripts/run_sweep.py --list

# Time-capped research SLICE (the lab's default unit of work): run a cell for
# ~1 h, self-cap on an epoch boundary, save a resumable latest.pt, tear down
# workers+eval, then one final eval cycle. --run-base keeps DATA off the worktree.
uv run python scripts/run_sweep.py --cell <CELL> \
    --max-wall-secs 3600 --final-eval --run-base /Users/jason/data/<area>

# Resume the same timeline (continues the embedded wandb run id + buffer):
uv run python scripts/run_sweep.py --cell <CELL> \
    --resume sweep_runs/<cell-name>/checkpoints/latest.pt --max-wall-secs 3600

# Non-9x9 board (env var, NOT a flag — must be set BEFORE any gomoku import):
GOMOKU_BOARD_SIZE=15 uv run python scripts/run_sweep.py --cell <CELL> ...

# Foreground for debugging (Ctrl-C stops the whole bundle):
uv run python scripts/run_sweep.py --cell SMOKE --foreground

# Wipe a cell's on-disk state (checkpoints + logs; does NOT stop a running cell):
uv run python scripts/run_sweep.py --cell <CELL> --clean
```

`run_sweep.py` CLI flags (all at `scripts/run_sweep.py:3021`):

| Flag | Default | Purpose |
|---|---|---|
| `--cell NAME` | — | Cell key from `CELLS`. Omitted / `--list` prints the catalog. |
| `--foreground` | off (background) | Run inline; Ctrl-C kills all procs. |
| `--clean` | — | Delete the cell's `sweep_runs/` + `sweep_logs/` dirs. |
| `--epochs N` | cell.epochs | Override the epoch budget. |
| `--n-workers N` | cell.n_workers | Override self-play generator count. |
| `--resume PATH` | none | Resume trainer from a `.pt`; **continues the embedded wandb run id + buffer**. |
| `--max-wall-secs S` | 0 (no cap) | Turn a run into a supervised SLICE: trainer self-caps at an epoch boundary, saves a fully-resumable `latest.pt` (with buffer), launcher tears the bundle down cleanly. |
| `--final-eval` | off | After teardown, run one `eval_worker` cycle so `eval_results.jsonl` ends on a fresh `model_elo`. Pairs with `--max-wall-secs`. |
| `--internal-eval` | off | Spawn the continuous internal-baseline (random/heuristic/lookahead) eval worker. Off by default (saturated noise on mature nets; steals CPU from the Rapfi ladder). Turn on for cold-start runs. |
| `--run-base DIR` | repo root | Root for `sweep_runs/`+`sweep_logs/`; sets `GOMOKU_RUN_DIR`. The autolab points this at `~/data` so run DATA lives outside the ephemeral code worktree. |

### The knobs that matter most (start here, tune the rest only with a reason)

| Knob | Where | Typical | What it buys |
|---|---|---|---|
| `size` | Cell field | `small`(64×4) → `large`(128×10) | Capacity. `tiny`/`small`/`medium`/`96x8`/`large`. Bigger = slower epochs, needs more data + fewer workers (gen-flood). |
| `GOMOKU_BOARD_SIZE` | env | 9 / 15 | Board. Native ext exists for 9,11,13,15; others fall back to pure-Python. |
| `n_workers` | Cell field | 3–8 | Generation parallelism. Big nets need FEWER (gen-flood: 8 workers flooded the ~3.05M trainer → 313 s/epoch). Watch reuse ≈ 1–1.8. |
| `buffer_size` | Cell field | 150k–1.5M | Replay window. 1.5M is the **MPS INT_MAX safe ceiling** at 17-plane 9×9; use `--pack-buffer` for bigger / at 15×15. |
| `n_simulations` | Cell field | 100 | Self-play MCTS depth. 100 is the standard training-regime point. |
| `--sgd-steps-per-epoch` | train_arg | 64 | **The runaway-proof learner**: EXACTLY N steps/epoch, decoupled from inflow. The v8+ recipe standard (supersedes `sgd_per_position` scaling). |
| `--value-discount` | worker_arg | 0.95–0.98 | Mate-distance value shaping; 0.95 = more aggressive (values faster wins). The single biggest cheap winning lever. |
| `global_pool` | Cell field | `True` | KataGo whole-board pooling; a confirmed derby win. |
| `--gumbel-root --gumbel-m 16` | worker_arg | on | Gumbel AZ root + Sequential Halving; the champion recipe's search. |

**The reigning "v8 recipe"** (the base most current cells clone): `small`/`large`
+ `stem_padding=1` + `global_pool=True` + `--value-discount 0.98`(or 0.95) +
`--gumbel-root --gumbel-m 16` + the WL2 stack (EMA 0.99 / grad-accum 4 / league
mix 0.4/0.1 / poll-jitter 2–8 s) + `--sgd-steps-per-epoch 64` + `--pack-buffer`
at 15×15. Change **one lever per cell** and keep everything else byte-identical
(the derby discipline).

---

## COMPREHENSIVE KNOB REFERENCE

Convention: **"byte-identical-off"** means the default value leaves the buffer,
loss graph, and gen hot path bit-for-bit unchanged from before the lever
existed — the whole derby methodology rests on this, so it's called out per
knob. `train_arg` / `worker_arg` = passed via `extra_train_args` /
`extra_worker_args`. Line refs: `gomoku/train.py`, `gomoku/selfplay_worker.py`.

### Cell fields (first-class; `scripts/run_sweep.py:53`)

| Field | Default | Purpose / when to change |
|---|---|---|
| `name` | — | **Load-bearing**: sets wandb run name, `sweep_runs/`, `sweep_logs/` dirs; grepped throughout the wiki. Rename → new dirs (the "branch an experiment" pattern). |
| `sgd_per_game` | 1.0 | K in the AZ `steps = K·games_ingested` schedule (mostly superseded by `--sgd-steps-per-epoch`). |
| `buffer_size` | — | Replay ring capacity. See buffer/replay below. |
| `games_per_epoch` | 32 | Also the default `worker_min_games` target. |
| `n_simulations` | 800 (cells set 100) | Self-play MCTS sims. |
| `wave_size` | 32 (cells set 64) | Leaves collected per game per evaluator call (virtual-loss wave batching). 64 = the dispatch-bound free-regime point at 15×15. |
| `batch_size` | 512 | SGD minibatch. |
| `lr` | 1e-3 | Learning rate. |
| `size` | `medium` | Model capacity: `tiny`/`small`(64×4)/`medium`/`96x8`/`large`(128×10). |
| `c_puct` / `c_puct_base` | 1.25 / 19652 | AGZ log-schedule PUCT constants. |
| `dirichlet_alpha` / `dirichlet_eps` | 0.13 / 0.25 | Root exploration noise. |
| `temperature_moves` / `temperature_final` | 10 / 0.1 | Warm-up sampling plies, then near-greedy temp. Cells use 30 warm-up. |
| `n_workers` | 4 | Self-play generator count. |
| `epochs` | 100 | Budget; continuous cells set 1e6 and cap by `--max-wall-secs`. |
| `save_every` | 1 | Slim (weights) checkpoint cadence. |
| `save_buffer_every` | 20 | `latest.pt` (with ~1.4 GB buffer) cadence. Resume rolls back to this. |
| `keep_last_n` | 3 | Auto-prune old epoch checkpoints — **brutally short** (~15 s of history at prod cadence); snapshot forensic epochs aside immediately. |
| `stem_padding` | None→3 | 3 = michaelnny AGZ edge-fix; **1** = legacy 9×9 feature map, ~2× cheaper forward, loses edge-blocking. Cells set 1. |
| `global_pool` | None (OFF) | Derby v4 whole-board lever. `True`=latter half of blocks, int K=trailing K. OFF = byte-identical arch. |
| `worker_min_positions` | 0 | Constant-age ingest by positions instead of games. |
| `sgd_per_position` | None | Position-based step schedule (stable across attack/defense game-length swings). Cells use 0.0025. |
| `games_per_batch` | 8 | Per-worker games/batch. |
| `wave_mode` | False | Per-version tile barrier (wave-lockstep distributed self-play). |
| `compile_workers` | False | `torch.compile` worker eval models (~1.3–1.5× fwd at batch≥32 on MPS). |
| `ema_tau` | 0.0 | WL2 #1 (see value/EMA). |
| `grad_accum_steps` | 1 | WL2 #4. |
| `opponent_mix_recent` / `_history` / `_recent_window` | 0/0/100 | WL2 #2 past-checkpoint league mix. |
| `weights_poll_min_sec` / `_max_sec` | None | WL2 #3 poll jitter. |
| `random_opening_moves` | 0 | WL3 K random opening plies (not recorded). |
| `swap2` | False | Swap2-negotiated opening. |
| `fixed_openings` | False | Rapfi 9 fair openings, placed directly. |
| `validation_archive_path` | None | WL5 frozen validation set for stationary eval. |
| `archive_start_path` / `archive_start_frac` | None / 0 | WL5 seed games from an archive. |
| `wandb` | True | Log to W&B (SMOKE cells set False). |
| `extra_train_args` / `extra_worker_args` | [] | Pass-through flag lists (everything below that isn't a Cell field). |

### Self-play / MCTS (`selfplay_worker.py`)

| Flag | Default | Notes |
|---|---|---|
| `--n-simulations` | 800 | Sims per move; cells set 100. |
| `--wave-size` | 32 | Wave-batched leaves/call. |
| `--c-puct` / `--c-puct-base` | 1.25 / 19652 | PUCT constants (`:111`). |
| `--temperature-moves` / `--temperature-final` | 10 / 0.1 | (`:115`). |
| `--dirichlet-alpha` / `--dirichlet-eps` | 0.13 / 0.25 | Root noise. |
| `--forced-playout-k` | 0.0 (OFF) | KataGo forced-playouts + policy-target-pruning (Wu 2019). Recommended 2.0. Byte-identical-off. (`:122`) |
| `--playout-cap-frac` / `--playout-cap-fast-sims` | 1.0 / 0 (inert) | Playout-Cap Randomization: fraction of moves full-search+recorded; rest run fast-sims and aren't recorded. Byte-identical-off. (`:150`) |
| `--gumbel-root` | OFF | Gumbel AZ root selection + Sequential Halving; policy target becomes completed-policy, not visits. Python tree only (native C has no Gumbel). Byte-identical-off. (`:167`) |
| `--gumbel-m` | 16 | Root actions sampled via Gumbel-top-k. |
| `--gumbel-c-visit` / `--gumbel-c-scale` | 50.0 / 1.0 | σ(q) constants (paper defaults). |
| `--concurrent-games` | 0 (legacy lockstep) | Issue #112 continuous refill: keep the active set at this width, seed a replacement the instant a game finishes so the merged oracle solve + MPS wave run full-width. Native path only. Byte-identical-off. (`:94`) |
| `--stream` | off | Issue #112 unbounded continuous-refill production (requires `--concurrent-games>0`); flushes chunks + hot-reloads weights, no batch ramp/drain. (`:101`) |
| `--max-plies` | None | Profiling/smoke game cap. |
| `--opponent` | self | `self` / `random` / `heuristic` / `defensive` / `pacifist` / `lookahead:depth=N`. Non-self forces decisive games; only model plies train. (`:455`) |
| `--opponent-mix-random` | 0.0 | Wrap opponent to play uniform-random with prob p (weaken a strong baseline). |
| `--model-first-frac` | 0.5 | Fraction of non-self games where model moves first. |

Trainer-side MCTS mirrors (`train.py`): `--wave-size` (default 1),
`--eval-sims` (50, eval-only), `--c-puct`/`--c-puct-base` etc. The trainer runs
its own in-process gen only when `--worker-input-dir` is unset.

### Oracle levers (GPU mega-VCT solver in the loop; see [[mega-vct-solver]], [[sound-world-recipe]])

| Flag | Default | Notes |
|---|---|---|
| `--vct-terminus` | OFF | Issue #98: end each game at the FIRST position with a forced VCT (batched GPU oracle), taking the exact win+move instead of playing to five. Cuts trajectory ~2×. Native / Python-non-Gumbel only. Byte-identical-off. (`:218`) |
| `--vct-terminus-budget` | 50 | Per-board oracle node cap — governs BOTH the terminus test AND (with `--oracle-veto`) the per-ply veto escape-solves, so it is the sound-world gen-throughput dial. Flag default cap50 (sweet spot: 98.8% of VCTs, 40–850× cheaper than deep search); **the `sound-world` cell overrides to 25** (#114, Jason-approved 2026-07-03: ~1.98× solve, ≥98.64% veto recall — see [[sound-world-recipe]] § Oracle budget). Eval-time finisher stays cap50. |
| `--oracle-veto` | OFF | Sound-world #107: every ply, bulk escape-solve and MASK proven-losing moves out of BOTH the played move and the recorded policy target (on-policy). All-moves-lose ⇒ defender terminus (z=−1). Composes with `--vct-terminus` for fully oracle-sound games. Native-non-Gumbel only. Byte-identical-off. (`:232`) |
| `--oracle-veto-max-cands` | 0 (full breadth) | Staged-escalation breadth cap (big-board lever). K>0 solves only the K empty cells nearest stones first, escalates to full breadth before the defender-terminus check (stays sound). **Note: capping this resurrects the 9-ply attractor — the veto IS the mechanism.** (`:245`) |
| `--oracle-precheck` | OFF | Null-board precheck (byte-identical results); measured SLOWER at 9×9 live, a big-board experiment only. (`:255`) |
| `--oracle-overlap` | OFF | Perf: run the per-ply bulk oracle solve (MLX/Metal) on a background thread while the native MCTS wave searches on MPS. **NOT byte-identical** to serial order (shifts evaluator batch shapes, same numeric class as a wave-size change; deterministic per seed). (`:266`) |

### Teacher / distillation levers

Offensive (stamp proven wins) and defensive (teach "you were lost") teachers.
All byte-identical-off (solver never runs). Worker-side unless noted.

| Flag | Default | Notes |
|---|---|---|
| `--vcf-teacher` | OFF | Exact VCF (forced-four win) teacher: overwrite policy target with the proven winning move (one-hot) + mate-discounted value. (`:186`) |
| `--vct-teacher` | OFF | Exact VCT teacher — strict SUPERSET of VCF (proves VCF wins + forcing-threes wins). REPLACES `--vcf-teacher` on the offensive seam. **Aggressively bounded** (`_VCT_TEACHER_MAX_DEPTH/_NODES = 4/800`); the threes tree fans out, so the cap is the gen-starve guard. (`:191`) |
| `--vct-max-depth` / `--vct-max-nodes` | None→4 / 800 | VCT teacher per-move solve caps. NOT the 20k general default (which starved gen in derby v8). Raise at your own gen-cost risk. (`:205`) |
| `--vcf-max-depth` / `--vcf-max-nodes` | None→16 / 200k | VCF teacher depth/node caps (derby v5 "vcf-deep"). Higher proves longer forced wins → more labels, more solve time. (`:340`) |
| `--defense-teacher` | OFF | VALUE-ONLY mirror of `--vcf-teacher`: when the OPPONENT has a proven forced win vs side-to-move, relabel value to −1 ("defend earlier"). Policy untouched (defense is non-unique). Gen-cost-gated (four-threat pre-scan). (`:278`) |
| `--defense-soft-value` | None→−1.0 | Gentler defense (#42): stamp e.g. −0.5 instead of the hard −1 that collapsed the value head (the −458 crash). (`:298`) |
| `--defense-max-fraction` | None→1.0 | Gentler defense (#42): cap the fraction of a game's to-move positions the teacher may relabel (spends budget on latest/closest-to-mate). (`:305`) |
| `--defense-detect-frac` | None→1.0 | Sparse-bite (#43): run the exact solver on only this fraction of danger plies (solver is the ~7 s/game gen bottleneck; 0.1 cuts ~10×, stamps stay exact). (`:313`) |
| `--defense-teacher-policy` | OFF | Defense I2 (#43): stamp the SAVING move on the POLICY head instead of crushing value. Implies `--defense-teacher`. (`:289`) |
| `--defense-teacher-conv` | OFF | Cheap vectorized board-scan block-teacher (NO solve, ~µs/ply): fires every ply, stamps the block to the opponent's immediate threat on policy. Tier1 sound (forced-four block); Tier2 heuristic (open-three) on by default. Implies `--defense-teacher`. (`:322`) |
| `--defense-conv-no-tier2` | OFF | Disable the conv teacher's Tier2 heuristic, keep only the sound forced-four block. (`:335`) |

Trainer-side **data distillation** (from a pre-built expert npz, not the live
solver):

| Flag | Default | Notes |
|---|---|---|
| `--teacher-data-path` | None (OFF) | npz of expert-labelled positions (`python -m gomoku.teacher generate`). v1=one-hot move, v2=`--soft` dense winrate map. POLICY-ONLY CE/KL (value untouched, per #18/#44). (`:1024`) |
| `--teacher-weight` | 0.0 (OFF) | Distillation term weight; suggested ~0.3. Byte-identical-off. |
| `--teacher-batch-size` | 0 (=batch-size) | Teacher mix-in batch. |
| `--teacher-temp` | 0.10 | Softmax temp for a v2 soft target; temp→0 recovers v1 one-hot. |
| `--no-teacher-augment` | off (augment) | Disable D4 augmentation of teacher positions. |

### Representation (input planes + arch)

| Flag / field | Default | Notes |
|---|---|---|
| `--line-planes` (train_arg) | OFF | Sound-world #107: derive 8 extra in-forward channels (per-cell × 4-dir × {me,opp} live-5 counts) so double threats read as two hot channels. **FRESH models only** (stem in_channels widens; resume must agree). External 17-plane contract untouched. Byte-identical-off. (`:998`) |
| `global_pool` (Cell) / `--global-pool` (train_arg) | None (OFF) | KataGo global pooling in the residual tower. Bare = latter half of blocks; K = trailing K. Byte-identical-off. (`:707`) |
| `stem_padding` (Cell) / `--stem-padding` | None→3 | 3 = AGZ edge-fix; 1 = legacy cheaper map. (`:703`) |
| `--activation` (both) | relu | Derby x-mish: `relu` or `mish` (KataGo default) for every tower nonlinearity. Zero added params, identical state_dict keys. Byte-identical-off. Resume must agree. (`:748`) |

Auxiliary heads (all DROPPED at inference — self-play/eval pay nothing;
byte-identical-off). Worker records the target, trainer weights the head:

| Trainer flag | Worker flag | Default | Notes |
|---|---|---|---|
| `--aux-opponent-reply-weight` | `--record-aux` | 0.0 | V3: 2nd 81-way head predicts opponent's next-ply MCTS policy. Suggested 0.15. (`:958`) |
| `--aux-ownership-weight` | `--record-ownership` | 0.0 | V4 KataGo ownership: per-cell final control (+1/−1/0). Suggested 0.15. (`:970`) |
| `--aux-vct-weight` | `--record-vct` (+`--vct-terminus`) | 0.0 | Moonshot per-cell "VCT-blunder map" defense head (masked BCE). Suggested 0.1. `--vct-defense-max-cands` caps labeler breadth. (`:985`) |
| `--soft-policy-weight` | — (trainer transform) | 0.0 | KataGo soft-policy aux target: 4th-root-flattened copy of recorded `pi`. NO new head, ZERO gen cost. Suggested 0.15. (`:1009`) |

### Value head (`scalar` / WDL / HL-Gauss + shaping)

| Flag | Default | Notes |
|---|---|---|
| `--value-head` (both) | scalar | `scalar` (tanh, byte-identical) / `wdl` (categorical {W,D,L} CE; derby-cgf) / `hlgauss` (distributional, N bins over [−1,1]; derby-tn4). Scalar v is derived for MCTS either way. Worker flag is a consistency check only. (`:712`) |
| `--hlgauss-bins` | None→51 | HL-Gauss bins (Farebrother 2024). Only with `hlgauss`. (`:726`) |
| `--hlgauss-sigma` | None→0.05 | Gaussian-smoothing σ of the HL-Gauss target. (`:732`) |
| `--value-discount` (worker_arg) | None→1.0 | Derby v6 mate-discounted value: scale outcome targets by γ^(plies_to_end), e.g. 0.98 / 0.95(aggressive). Applied before the VCF teacher. Byte-identical-off. **The standard cheap winning lever.** (`:349`) |
| `--draw-value` (both) | 0.0 (OFF) | Draw-contempt (derby-9q4): on a draw, set value target to −Δ (mildly losing) so the net avoids draws. Composes with value-discount. Byte-identical-off. (`:738`) |
| `--contempt-p` (worker_arg) | 0.0 (OFF) | Search-contempt (derby-qoq): with prob p, replace the move pick with one favoring Q≈0 (contested) positions so self-play oversamples hard-to-convert boards. Recorded `pi` UNCHANGED. Paper p=0.5. Byte-identical-off. (`:364`) |
| `--value-weight` (train) | 1.0 | Value-loss weight. |

### Buffer / replay

| Flag | Default | Notes |
|---|---|---|
| `--replay-buffer-size` (Cell `buffer_size`) | 1.5M | Ring capacity. **MPS INT_MAX**: `capacity × planes × board²` > 2.147e9 crashes on first `shape_stats()` — ~1.56M at 17×81, so **1.5M is the safe ceiling**; bigger needs `--pack-buffer` or a CPU buffer. (`:813`) |
| `--shape-stats-every` (train_arg) | 10 | #115: run the O(buffer) `shape_stats()` diagnostic scan every N epochs instead of every epoch. The scan is ~25.6 ms @ 150k rows → linear ~250 ms @ a full 1.5M buffer, so at scale this saves **~230 ms/epoch amortized**. `1` restores the old every-epoch cadence; logged stats are unchanged on the epochs it runs. |
| `--pack-buffer` (train_arg) | OFF | Issue #25: bit-pack binary planes (uint8, ~32× smaller) on CPU, unpack per-batch. Byte-identical to float32. **Prerequisite at 15×15** (a 3M-pos float32 buffer is ~45 GB; packed ~3 GB). Existing float32 checkpoints still load. (`:826`) |
| `--buffer-recency-frac` (train_arg) | 0.0 (uniform) | Derby v7 buffer-composition curator: draw this fraction of each batch from the most-recent `--buffer-recency-window` positions (rest uniform). The +90-elo v8 buffer-comp winner. Byte-identical-off. (`:818`) |
| `--buffer-recency-window` | 200k | Size of the "recent" slice. |
| `--cross-game-value` (+`--cross-game-store`, `-recency-decay`, `-min-visits`, `-max-blend`, `-max-ply`) | OFF | Derby position-stats: aggregate value returns across all games through each canonical position; blend into the z target (visit-gated, capped). De-noises credit assignment, zero extra GPU. Opening-only cap (`--cross-game-max-ply 10`) keeps the store bounded. Byte-identical-off. (`:835`) |
| `--reanalyze` (+ `-fraction`, `-max-positions`, `-sims`, `-mcts-batch`, `-relabel-value`, `-every-epochs`, `-every-positions`, `-cooldown-cycles`, `-cooldown-positions`) | OFF | Re-MCTS a small sample of OLD buffer positions with the current net and overwrite their policy targets (optionally value). Bounded per-cycle envelope + cadence + per-row cooldown (feedback-loop guard). Byte-identical-off (engine never imported). (`:866`) |

### Ingest / SGD schedule

| Flag | Default | Notes |
|---|---|---|
| `--sgd-steps-per-epoch` (train_arg) | 0 (OFF) | **TRAINER MODE**: EXACTLY N steps/epoch, FIXED, independent of inflow — structurally runaway-proof (the LF1 20 s→7 min blow-up). In the non-wave worker-ingest path it also switches ingest to NON-BLOCKING (async continuous-learner). **The v8+ standard (64).** Supersedes the scaling schedules. (`:1275`) |
| `--sgd-per-game` (train_arg / Cell) | None | `steps = K·games_ingested`; K≈1 = AZ recipe. Varies with game length. (`:926`) |
| `--sgd-per-position` (train_arg / Cell) | None | Position-based version; stable across attack/defense length swings. Cells use 0.0025. Overrides `--sgd-per-game`. (`:943`) |
| `--min-training-steps` | 16 | Floor for the scaling schedules. |
| `--training-steps` | 400 | Static steps/epoch when no schedule set. |
| `--max-sgd-steps-per-epoch` (train_arg) | 0 | LF1 anti-runaway: `min(computed, cap)` clamp (still scales below cap). (`:1266`) |
| `--max-tile-games` (train_arg) | 0 | LF1: cap games ingested per model version (wave-mode); excess deterministically dropped. Closes the gen→ingest loop. (`:1255`) |
| `--grad-accum-steps` (train_arg / Cell) | 1 | WL2 #4: accumulate over N minibatches before `.step()`. Recommended live 4. (`:1093`) |
| `--worker-min-games` / `--worker-min-positions` | 0 | Ingest barrier per cycle (games vs positions). Positions keep turnover constant across game-length swings. (`:1219`) |
| `--batch-size` / `--lr` / `--l2` | 256 / 1e-3 / 1e-4 | Standard SGD. |

**Trainer step perf — the quick wins (#115, 2026-07-03), and the honest profile.**
Two internal (no-flag) micro-optimizations, both verified value-preserving: (1)
**fused L2** — the `--l2` penalty computed via `torch._foreach_pow(params, 2)`
instead of a ~41-tensor `sum((p**2).sum())` dispatch storm = **−9.7%/step**
(23.4→21.2 ms), **gradient bitwise-identical** (`2·p` per param; the loss *value*
shifts ~7e-8 rel, below the 8th decimal); NOT folded into AdamW `weight_decay`, so
the intended double-regularization is preserved. (2) **Packed host-sync** — up to
~15 `float()`/`bool()` per-microbatch host syncs collapsed into **one** packed
`.cpu()` transfer (on-device masked means + counts); logged values byte-matched.
**But the honest profile: at fixed `--sgd-steps-per-epoch 64` the epoch is
GEN-dominated** — 13×13 gen 10–34 s vs train 1.4–1.9 s/epoch — so the trainer was
never the wall; the per-step win is real but the epoch-scale effect (~0.15 s/epoch)
is swamped by gen variance. It scales linearly with the step count. (Same-seed
3-epoch trajectory identical before/after; 1129 tests green.)

### Weight-publish smoothing + opponent mix (WL2 scale-emulation stack)

| Flag | Default | Notes |
|---|---|---|
| `--ema-tau` (train_arg / Cell) | 0.0 (OFF) | WL2 #1: publish an EMA copy of the model to workers (θ_ema←τθ_ema+(1−τ)θ per step). Decouples the "brain that plays" from the "brain that learns". Recommended 0.99. (`:1065`) |
| `--swa-window` (train_arg) | 0 (OFF) | V3: publish the flat average of the last K saved state_dicts instead of EMA. REPLACES EMA-publish (alternatives, not composed). Generation-only. Recommended 5. (`:1076`) |
| `--opponent-mix-recent` / `--opponent-mix-history` / `--opponent-mix-recent-window` (worker / Cell) | 0 / 0 / 100 | WL2 #2: per-wave prob of full self-play vs a past checkpoint (recent window / anywhere in history). Games still written to the current tile. Recommended 0.4 / 0.1. (`:482`) |
| `--weights-poll-min-sec` / `--weights-poll-max-sec` (worker / Cell) | None | WL2 #3: per-worker poll interval drawn once at startup from U(min,max) (de-synchronizes reloads). Recommended 2–8 s. (`:471`) |
| `--weights-poll-sec` | 1.0 | Base poll interval when no jitter. |
| `--gen-once-per-publish` (worker) | off | One batch per weight publish → each cycle's games are exactly one model version (clean stratification, trades worker idle). (`:493`) |

### Openings

| Flag | Default | Notes |
|---|---|---|
| `--random-opening-moves` (both / Cell) | 0 | WL3: K uniform-random legal opening plies (NOT recorded); MCTS takes over after. Breaks opening monoculture. Mutually exclusive with `--swap2`. (`:778`) |
| `--swap2` (both / Cell) | OFF | Swap2-negotiated opening (net plays opener+responder; v1 one-ply-heuristic choice, no trained head). Emitted to BOTH trainer + workers. Byte-identical-off. (`:784`) |
| `--fixed-openings` (both / Cell) | OFF | Rapfi's 9 known-fair swap2 openings placed directly (no negotiation/net/choice); net plays only post-opening. Hands the net known-fair boards. Mutually exclusive with swap2. Byte-identical-off. (`:791`) |
| `--choice-head-weight` (train_arg) | 0.0 (OFF) | swap2 v2a: TRAIN the choice head (negotiation records → ChoiceBuffer, outcome-driven soft target). Selection still uses the one-ply heuristic (v2b wires selection later). Suggested 0.3. Byte-identical-off. (`:1052`) |
| `GOMOKU_DROP_OPENERS` (env) | — | Comma list of fair-book indices to drop (e.g. `0,1,3,4,5,6,7,8` keeps only idx-2 — the moonshot-bruce Bruce-Lee wound). Launch-env only. |
| `--archive-start-path` / `--archive-start-frac` (worker / Cell) | None / 0 | WL5: per-game prob of seeding from a mined archive position instead of the empty board. (`:444`) |

### Board size + native extensions (process env — set BEFORE any `gomoku` import)

| Env | Default | Notes |
|---|---|---|
| `GOMOKU_BOARD_SIZE` | 9 | Board side length. **NOT a Cell field** — set in the launch env; propagates to workers. Native ext exists for **9, 11, 13, 15** (`NATIVE_BOARD_SIZES`); others use pure-Python fallback. Resolution order: `--board-size` flag > env > 9, fixed once at import (`board_config.py:36`). |
| `GOMOKU_DISABLE_NATIVE_MCTS` | unset | Force the pure-Python MCTS tree (same path Gumbel already takes). A/B the native C engine. (`native_mcts.py:21`) |
| `GOMOKU_DISABLE_NATIVE_STATE_OPS` | unset | Force pure-Python state ops. (`state_ops.py:29`) |
| `GOMOKU_DISABLE_NATIVE_LOOKAHEAD` | unset | Force pure-Python lookahead baseline (vs the native one). (`baselines.py:28`) |

### Eval (in-trainer) — mostly eval-only, does NOT touch gen/train

`--eval-every` (5), `--eval-sims` (50), `--eval-baselines`
(`random,heuristic`), `--eval-baseline-games` (4), `--eval-baselines-slow`
(""), `--eval-slow-every` (4), `--eval-slow-games` (6), `--no-eval-arena`
(use legacy one-game path; #106 arena is default), `--no-eval` (skip
in-trainer eval entirely; pair with a separate `eval_worker`).
**Eval-only search levers** (auto-disable the batched arena): `--eval-vcf-nodes`/
`--eval-vcf-depth` (root VCF overlay), `--fpu-reduction-c` (KataGo FPU;
c≈0.45/0.20), `--reuse-tree` (tree reuse across plies), `--proven-prop` +
`--proven-vcf-leaf-nodes` (proven-win propagation). All byte-identical-off.
See [[launch-sequence-runbook]] §leading-indicators for what to watch. The eval
harness is documented in [[eval-suite]]; the code lives in `gomoku/eval.py`,
`eval_worker.py`, `eval_swap2.py`, `match.py`.

### Perf / backend (worker-side, eval-only)

`--compile` (torch.compile, ~1.3–1.5× fwd at batch≥32 MPS; NOT on the trainer),
`--fp16-eval` (cast worker eval to fp16; outputs back to fp32 so records
unchanged), `--evaluator torch|coreml` (Core ML ANE offload; `--coreml-compute-units`
`CPU_ONLY`/`CPU_AND_NE`/`CPU_AND_GPU`/`ALL`). `--device` (cpu/mps).

### Resume consistency guards (`train.py` main, ~`:1416–1495`)

`--resume` **hard-fails** (SystemExit) if any of these disagree with the loaded
checkpoint's config — the representation comes FROM the checkpoint, so a
mis-launched resume is caught, not silently corrupted:

- `--activation` vs checkpoint (`:1456`)
- `--line-planes` vs checkpoint (stem width; `:1467`)
- `--value-head` (+ `--hlgauss-bins`/`--hlgauss-sigma`) vs checkpoint (`:1481`)
- `--board-size` vs the import-time board (`board_config.require_board_size`)

Also: resume **continues the embedded wandb run id** (no override flag; strip
it from the checkpoint for a clean new run) and rolls the trainer back to
whatever epoch `latest.pt`'s buffer snapshot was last written (up to
`save_buffer_every` behind the newest slim checkpoint). `--swap2` +
`--random-opening-moves` are mutually exclusive (`:1315`).

---

## NOTABLE CELLS (`CELLS` dict, `scripts/run_sweep.py:138`)

Not exhaustive — the dict has ~70 cells. These are the ones worth knowing as
launch templates or lineage anchors. Board size is the env var; the run-dir
suffix (`-board9/11/13/15`) just isolates artifacts.

**Plumbing / smoke**
- `SMOKE` / `SMOKE15` — tiny/2-worker/30-sim bundle-plumbing smokes (no wandb). Exercise launch→supervise→teardown→`--final-eval`. Pair with `--max-wall-secs`.

**15×15 capacity ladder (epic #21)**
- `G15-seed` — the v8 recipe on 15×15 from scratch (64×4 small). The base clone.
- `G15-96x8` / `-redo` / `-deepgen` / `-cont100` / `-bigbuf` — the 96×8 capacity step + its data (1.5M bigbuf) and search-depth (200-sim deepgen) arms. 96×8 is capacity-bound.
- `G15-128x10` / `-bigbuf` — the 128×10 step (large, ~3.05M base / ~3.2M with line-planes + global-pool). Data-bound: overfits on 400k, needs the 1.5M packed buffer. `n_workers=4` (gen-flood remedy).

**Swap2 / fair-opening white-defense line (#72/#73)**
- `G15-swap2` / `-e2` / `-e3`, `G9-swap2-e2` — swap2 opening to delete the doomed white role; e2 adds aggression (value-discount 0.95) + trained choice head.
- `G-ladder-{11,13,15}` — the 9→11→13→15 swap2 curriculum (warm-start up a rung on draw-dominance).
- `G15-fixed-openings`, `G{9,11,13}-fixed-openings` — the fair-opening ladder (Rapfi 9 balanced openers, no negotiation). `G15` tuned: 1M packed buffer + `--buffer-recency-frac 0.5`, `n_workers=3` (reuse ~1.8).

**Defense teacher arms (#36/#42/#43)**
- `G15-defense` — champion + gentler `--defense-teacher` (soft-value −0.5, max-fraction 0.25, capped solve). The value-only crush FAILED (−458 elo, value saturation).
- `G15-defense-i2` — the policy-stamp defense (I2) that supersedes the value crush.
- `derby-x-defense` — champion + `--defense-teacher` on the 9×9 derby.

**VCT-terminus / sound-world (#98–#107)** — see [[sound-world-recipe]]
- `vctsci-control` / `vctsci-terminus` — the matched A/B for VCT-terminus self-play (#100: throughput win, robustness loss).
- `moonshot` — vct-terminus + `--record-vct` + `--aux-vct-weight 0.1` (VCT-defense aux head; #103: a sensor with no actuator).
- `sound-world` — **the validated recipe**: `--oracle-veto --oracle-overlap --vct-terminus --vct-terminus-budget 25` (cap25 as of 2026-07-03, #114) `--line-planes`, ONE streaming continuous-refill worker (`--stream --concurrent-games 256`), fixed `--sgd-steps-per-epoch 64`. The 9-ply attractor is GONE.
- `moonshot-bruce-idx2` — warm-start the 128×10 champion + layer the VCT-defense head + restrict self-play to the idx-2 wound (`GOMOKU_DROP_OPENERS`).

**Derby lineage** (the recipe-search cells that produced "v8") — `derby-v4/v5/v6/v7`
controls + single-lever arms (`-mate-discount`, `-wholeboard`, `-vcf`,
`-gumbel`, `-signal`, `-sgd128/256`), `derby-x-*` (vct / defense / soft-policy /
draw-contempt / mish). Each is a verbatim clone of its era's champion with ONE
lever flipped — the clone-the-champion / one-lever discipline.

**Historical WL series** — `WL1`–`WL5`, `Z`/`Zc`, `A`–`F` (the original K×buffer
matrix), `LF1` (lean-fp16 anti-runaway), `PERFA/B` (degrade tests). Anchors for
the WL design docs; see [[launch-sequence-runbook]] cross-refs.

---

## Cross-refs

- [[launch-sequence-runbook]] — the launch/resume/monitor/stop PROCEDURE (this page is the parameter dictionary behind it).
- [[sound-world-recipe]] — the oracle-veto + terminus + line-planes recipe (#107).
- [[mega-vct-solver]] — the GPU VCT oracle the terminus/veto/teacher levers call.
- [[training-run-lineage]] — how the recipe got here (the run history).
- `TRAINING_WIKI.md` — append-only live run logs.
- `scripts/run_sweep.py` — the canonical `CELLS` dict + launcher.
