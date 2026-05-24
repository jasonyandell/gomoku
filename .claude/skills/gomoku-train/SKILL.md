---
name: gomoku-train
description: Manage the AlphaZero training loop for Jason's gomoku project at ~/code/gomoku. Start, resume, stop, status-check, tune. Also covers the web UI server. Trigger on phrases like "start training", "resume gomoku", "stop training", "kick off a run", "how's training going", "is it still running", "play against the latest", "spin up the UI", "show me a self-play", or anything about gomoku checkpoints, MCTS sims, batch settings, wandb run.
---

# gomoku-train

Operate the gomoku AlphaZero pipeline at `~/code/gomoku/`. Reference codebase: [[reference-zeb]] (`~/code/mk5-main/forge/zeb/`).

The pipeline is a single-process self-play loop: model + MCTS generate games → replay buffer → SGD → push weights back to MCTS → repeat. Runs on Apple Silicon MPS. W&B for telemetry.

## Quick mental model

- **Two launch paths** in this repo:
  - **Production**: `python scripts/run_sweep.py --cell <CELL>` — spawns trainer + N self-play workers + 1 eval worker, all backgrounded. Uses wave-mode barrier, native MCTS, and (for WL2+) EMA self-play + past-checkpoint mix. This is what real runs use. See [Production launch — sweep cells](#production-launch--sweep-cells) below and the wiki runbook at `wiki/topics/launch-sequence-runbook.md`.
  - **Single-process**: `python -m gomoku.train` — one trainer that does everything including self-play. Good for ad-hoc smoke tests and the small-buffer A-F cells; not how WL1/WL2/Z were run.
- **One web server** (`python -m web.server`) loads any checkpoint on demand and exposes a play/replay UI.
- **Both want MPS.** They CAN share, but it's noticeably slower. Default to running the web UI on CPU while training holds MPS — set `GOMOKU_DEVICE=cpu` for the server.
- Checkpoints land per-cell at `~/code/gomoku/sweep_runs/<cell>/checkpoints/epochNNNN.pt` (single-process path uses `~/code/gomoku/checkpoints/`).
- W&B credentials come from the macOS Keychain (service: `wandb-api-key`). The training script pulls it automatically; nothing to set up.

## Production launch — sweep cells

The real production path. Follow the playbook at
[`wiki/topics/launch-sequence-runbook.md`](/Users/jason/code/gomoku/wiki/topics/launch-sequence-runbook.md) end-to-end whenever the user says "start a run", "launch WL3", "kick off the next training run", etc. The runbook covers:

1. Pre-launch state check + the two known gotchas (MPS INT_MAX at 1.5M+ buffer, worker race fix verification)
2. **Title card** — present it (never launch without one; no ACK, autonomous lab)
3. **Smoke** (30 epochs, validate every new lever's signature in logs, clean the dir)
4. **Real launch** + spin-up verify via Monitor
5. **Wiki updates** — append-only run log in `TRAINING_WIKI.md`, maintenance entry in `wiki/log.md`, refresh `scripts/wandb_workspace.py` with the new run id
6. **Monitoring** — two cadence patterns:
   - **Active** (`/loop`, user present): 3min → 15min → 30min escalating
   - **Overnight** (`CronCreate` at `7,22,37,52 * * * *`, fixed cadence,
     each tick self-contained): the WL5 phase-2 pattern. The runbook's
     Phase 5b spells out the cron prompt template, the "filter procs
     by cell name" rule, and the push-trigger list.
7. **Run end** — three flavors (cap-reached / user-stopped / crash); in
   all three, **trainer-only exit leaves 8 workers + 1 eval polling**,
   so `pkill -TERM -f 'sweep_runs/<CELL>/'` is mandatory before claiming
   the run is done. Then the **phase-N close-out template** (canonical
   examples: WL5 phase-1 close 2026-05-21, WL5 phase-2 close 2026-05-22)
   for `TRAINING_WIKI.md`, a `wiki/log.md` entry, then commit + push
   with the deploy-trigger pre-check (`app/**` paths trigger CF deploy).

Useful background skills inside that playbook:
- **Fan-out implementation**: when the next-run design has multiple independent levers, spawn background `Agent` calls in parallel — group by file to avoid merge conflicts. WL2's two-agent split (train.py vs selfplay_worker.py) landed cleanly in ~10min wall.
- **wandb workspace generator**: `scripts/wandb_workspace.py` creates a saved view with 6 sections pre-tuned for run-overlay comparison. Re-run with the new RUN id prepended to its RUNS dict to refresh.
- **Validation archive mining**: when a run uses `--validation-archive-path` or `--archive-start-path` (WL5 onward), mine a fresh archive with `scripts/mine_validation_archive.py` BEFORE launch — recipe + perf notes in [`wiki/topics/mining-validation-archives.md`](/Users/jason/code/gomoku/wiki/topics/mining-validation-archives.md). Always pass `-u` / `PYTHONUNBUFFERED=1` (the script's progress prints stay block-buffered otherwise) and run on MPS, not CPU. Don't fan out N parallel mine processes against the full checkpoint — each loads the 8 GB buffer independently and explodes memory; the in-process batched path is already MPS-saturated. Mine BEFORE launching the run that needs the archive (mining + training both want MPS).

## Cell map — what's in `scripts/run_sweep.py`

| cell | name | wandb id | notes |
|---|---|---|---|
| A-F | K × buffer-size matrix | — | small cells, single-process, mostly historical |
| Z | az-recipe-160k | `sppjo3z5` | continuous self-play, python MCTS, AGZ recipe, 5000 epochs |
| Zc | az-recipe-160k-constage | — | Z + positions-based ingest (not deployed) |
| WL1 | wave-lockstep, 1.5M buffer | `l8mbntcm` | per-version uniformity hypothesis; peaked elo 1281 at e360 then oscillation collapse |
| WL2 | scale-emulation | `9wng4yu9` | WL1 + EMA + past-checkpoint mix + poll jitter + grad accum; peaked la4=62% at e900, regressed to 18% by e1100 (same arc/regression as WL1 at higher peaks) |
| WL3 | random-openings | `0o75gws5` | WL2 + K=2 random opening plies (training examples not recorded for the random plies). Slower first crossing (e487 vs WL2's e370) but better-balanced wins across baselines (h/la2/la4 climbing together rather than single-baseline spikes); plies started bumping 13→15. Retention test underway. |
| WL4 | no-random-openings.plateau-e4024 | `44cxzc9d` | K=2 → K=0 curriculum mid-run. Reached the WL-series ATH **elo 1841**, plateaued and held there before becoming the resume parent for WL5. |
| WL5 | diagnostics-archive-start | `o6cbjfnr` | WL4 + validation archive (`val/policy_*` per-bucket) + H/KL decomposition + per-color/per-ply metrics + Go-Exploit archive-start lever (15% game-start from curated WL4 trouble positions). Two phases across one wandb run: phase 1 (e4001-e5051) un-fused-workers, phase 2 (e5052-e10200) fused. Peaked elo 1738 at e5477 (didn't break WL4 ATH 1841). 0 NaN over 6199 epochs. Buffer cycled ~28× by 1M games. |
| LF1 | lean-fp16-canary | `h9al2e0k` (1000-ep run; `geft5xmy` = the 100-ep test that exposed the steady-state cost) | The perf lab's R-TRAIN-LEAN-fp16 recipe as a REAL run: WL5 + **wave_size 512 + sgd_per_position 0.001 + workers `--fp16-eval`**. Fresh, 1000 epochs, started HOT (2026-05-23). TQ canary for the +152%-throughput recipe. Steady-state ~3 min/epoch (buffer full → ~1300 steps/epoch), NOT the perf-lab's 15s — see "Tuning knobs → LEAN-fp16" above. Learns fast/epoch (elo 437→776 by ~e28). |

When the user references "the previous run" or "Z" or "WL1", these are the canonical IDs.

The current wandb workspace (4-run WL3/WL2/WL1/Z overlay): regenerate via `python scripts/wandb_workspace.py`; the URL changes each time (workspaces API doesn't update in place). Bookmark whatever the latest run prints.

## Status check (always do this first)

```bash
cd ~/code/gomoku

# Is training running?
pgrep -fa "gomoku.train" || echo "(no training process)"

# Latest checkpoint + recent training output
ls -la checkpoints/ 2>/dev/null | tail -5
tail -10 wandb/latest-run/files/output.log 2>/dev/null || tail -10 scratch/train.log

# Is the web UI up?
lsof -iTCP:8766 -sTCP:LISTEN 2>/dev/null | tail -1 || echo "(web UI not running)"
```

W&B dashboard: project is `gomoku` under entity `jasonyandell-forge42`. The active run's URL is at the top of `scratch/train.log`.

## Starting / resuming training

> **Title card FIRST — never launch a run without one.** Before running any command below, present a title card (What / Lever / Parent / Config+cap / Why / Expect / Track) so it's clear what *this specific* run is doing, then proceed — **no ACK gate; this is an autonomous lab.** The card is for clarity, not permission. Template lives in the research-lab skill ([[gomoku-research-lab]] § Title card) and mirrors the Derby cards in `wiki/ops/research-board.md`. This is step 2 of the launch-sequence-runbook — don't skip it just because you jumped straight to the command.

The defaults below are what we know works on Jason's M5 Max — ~28s/epoch in steady state, ~125 epochs/hour:

```bash
cd ~/code/gomoku && source .venv/bin/activate

# RESUME (preferred — keeps W&B lineage via wandb_run_id in checkpoint)
PYTORCH_ENABLE_MPS_FALLBACK=1 nohup python -u -m gomoku.train \
  --resume checkpoints/latest.pt \
  --epochs 1000 \
  --games-per-epoch 64 \
  --n-simulations 200 \
  --training-steps 400 \
  --batch-size 256 \
  --replay-buffer-size 50000 \
  --eval-every 5 \
  --save-every 5 \
  --wandb --wandb-project gomoku \
  > scratch/train.log 2>&1 &

# FRESH START (new W&B run, fresh weights — use --run-name to label it)
PYTORCH_ENABLE_MPS_FALLBACK=1 nohup python -u -m gomoku.train \
  --size small \
  --epochs 1000 \
  --games-per-epoch 64 \
  --n-simulations 200 \
  --training-steps 400 \
  --batch-size 256 \
  --replay-buffer-size 50000 \
  --eval-every 5 \
  --save-every 5 \
  --wandb --wandb-project gomoku --run-name <descriptive-name> \
  > scratch/train.log 2>&1 &
```

## Eval baselines (logged every eval cycle)

By default the eval block runs the model against four baselines and logs `eval/vs_<name>_winrate` to wandb:

| baseline | when | games | wallclock | what it tells you |
|---|---|---|---|---|
| `random` | every eval | 16 | <2s | smoke-test floor; pinned ~100% past epoch 5 |
| `heuristic` | every eval | 16 | ~10s | rule-based pattern eval — meaningful signal across all of training |
| `lookahead:depth=2` | every eval | 16 | ~30s | 2-ply alpha-beta with same heuristic — mate-in-1 baseline |
| `lookahead:depth=4` | every 4th eval | 6 | ~3min | 4-ply alpha-beta — the real strength bar |

Customize with `--eval-baselines random,heuristic,lookahead:depth=2`, `--eval-baselines-slow lookahead:depth=4`, `--eval-slow-every 4`, `--eval-baseline-games 16`, `--eval-slow-games 6`. Spec syntax is the same as `gomoku/match.py` (KIND[:K=V,K=V...]). Model-vs-model is not allowed here — the eval block always pits the training model against non-learned baselines.

The old `--eval-games` arg is gone; use `--eval-baseline-games` (fast) and `--eval-slow-games` (slow) instead.

### Eval interpretation gotchas

- **Sample-size variance**: 16 games per eval is *noisy*. A single eval reading like `heuristic=5%` (0-1 wins on 16 games) and `heuristic=35%` (5-6 wins) can be the same true rate just bouncing. Don't read into single evals; trust the trend across 3-5 evals or use larger n via `--eval-baseline-games 40` for one-off forensic checks.
- **`time/eval_vs_heuristic_s` climb is a hidden plies-regrowth signal**. 16 games at constant per-move cost ≈ constant time *unless* games are getting longer. If `time/eval_vs_heuristic_s` climbs (e.g. 6s → 17s during WL2), the model is fighting longer vs heuristic *before* the same growth shows up in `selfplay/plies_mean` (because in selfplay the model still beats its own brain fast). Watch the time chart, not just the win rate.
- **Eval-distribution matters but less than you'd think**: WL3 trained with K=2 random openings; eval is from canonical start. We tested whether re-evaluating with K=2 openings (matched distribution) would show hidden strength. **It didn't** — same checkpoint scored ~35% at K=0 and ~34% at K=2 on a 40-game test (test run 2026-05-21). The eval-distribution mismatch is real but the signal hidden by it is small. Don't reach for "fix the eval" if win rates are slow — the model is genuinely slow, not just mis-measured.
- **Native MCTS vs python MCTS**: the trainer eval uses the native engine (via `make_torch_evaluator`); CPU-only ad-hoc tests via `mcts_picker` use the python path. Strength may differ by a few percent. If you need bit-for-bit reproducibility, set `GOMOKU_DISABLE_NATIVE_MCTS=1` and rerun the trainer eval.

**Always** use `python -u` (unbuffered) when redirecting to a file — otherwise per-epoch prints stay in the OS buffer and `tail -f` shows nothing. Wandb mirrors stdout to `wandb/run-*/files/output.log` which IS flushed, but that's not as obvious.

After starting, wait ~5s and verify:

```bash
sleep 5 && tail -10 scratch/train.log
```

You should see the `wandb: 🚀 View run at ...` line. Save that URL — it's the user's window into the run.

## Stopping training

```bash
pkill -f "gomoku.train"     # graceful — SIGTERM triggers a clean resumable save
# Or by PID if you have it:
# kill <PID>
```

SIGTERM/SIGINT is now a **clean stop**: the trainer finishes the current epoch, force-saves a fully-resumable `latest.pt` (replay buffer embedded, *ignoring* `--save-buffer-every`) + worker weights, then exits 0. A `--resume latest.pt` afterward continues WITHOUT a cold buffer refill. Don't `kill -9` — it skips the clean save and wandb won't flush.

### Time-capped slices (`--max-wall-secs`) — design a run as a research SLICE

A training run can be **time-capped** instead of epoch-capped. This is how the research lab ([[gomoku-research-lab]]) schedules training on its GPU-required queue — a run becomes a wall-bounded, resumable slice:

```bash
# single-process (smoke / ad-hoc):
python -m gomoku.train --resume checkpoints/latest.pt --max-wall-secs 600 ...

# production bundle (the real path — trainer + 8 workers + eval):
python scripts/run_sweep.py --cell <CELL> \
  --resume sweep_runs/<CELL>/checkpoints/latest.pt \
  --max-wall-secs 600 --final-eval
```

- **`--max-wall-secs N`**: at the next epoch boundary past N seconds, the trainer self-caps with the clean resumable save above and exits 0. Resume the next slice from `latest.pt` — no cold refill. Keep N ≫ one epoch's wall (at a full buffer an epoch is minutes, so the cap rounds up to the next boundary).
- **`run_sweep --max-wall-secs`** also SUPERVISES the bundle: when the trainer self-caps it tears down the workers + eval cleanly (hard-deadline SIGTERM fallback if the trainer is stuck mid-epoch). Backgroundable via `nohup`.
- **`--final-eval`** (run_sweep): after teardown, runs one `eval_worker --max-cycles 1` so `<CELL>/checkpoints/eval_results.jsonl` ends on a fresh `eval/model_elo`. The research lab reads Δelo from there; eval stays inside the bundle.

This is the seam between **clean training** (this skill — the machine) and **clean research** (the research lab — which schedules time-capped slices as GPU-required items). See the research-lab charter's "training run as a GPU-required item" section.

## Web UI server

```bash
# Start (default port 8766, CPU eval so it doesn't fight training)
PYTORCH_ENABLE_MPS_FALLBACK=1 GOMOKU_DEVICE=cpu nohup \
  python -m web.server --port 8766 > scratch/web.log 2>&1 &

# Open in browser: http://127.0.0.1:8766
```

If port 8766 is busy, pick another free port:

```bash
for p in 8766 8767 8768 8769; do nc -z 127.0.0.1 $p 2>/dev/null || { echo $p; break; }; done
```

**Stop:**

```bash
pkill -f "web.server"
```

**To play with full MPS speed** (no training contention): stop training first, then start the server WITHOUT `GOMOKU_DEVICE=cpu`. Restart training afterward.

## Tuning knobs (when the user asks for stronger / faster / different)

| Knob | What it does | Trade-off |
|---|---|---|
| `--n-simulations` | MCTS sims per move | Doubling sims ≈ doubles gen time but better policy targets. 100 = fast, 200 = default, 400 = strong, 800 = slow |
| `--games-per-epoch` | Games generated each epoch | More games = better stats per epoch but slower wallclock |
| `--training-steps` | SGD steps per epoch | More steps = faster fit to current buffer but risk of overfitting on stale data |
| `--size` | Network size | `tiny` (54k) / `small` (316k, default) / `medium` / `large`. Bigger = stronger ceiling, slower per-sim |
| `--replay-buffer-size` | Ring buffer capacity | Bigger = smoother targets, more samples-per-eviction. 50k ≈ 7 epochs at defaults |
| `--temperature-moves` | Plies with sampling temperature 1.0 | More = more diverse openings; default 8 is reasonable for 9x9 |
| `--dirichlet-alpha`, `--dirichlet-eps` | Root noise | AlphaZero defaults (0.3, 0.25). Lower eps = less exploration |
| `--lr` | AdamW learning rate | Default 1e-3. Drop to 1e-4 if losses oscillate |

Changing `--size` requires a fresh start — checkpoint config is locked once initialized.

### The LEAN-fp16 "faster" recipe — and why perf-lab epochs/s LIES about training wall-clock (2026-05-23, LF1)

The perf lab found an "R-TRAIN-LEAN-fp16" recipe — WL5 + `wave_size 512` + `sgd_per_position 0.001` + workers `--fp16-eval` — that measured **+152% throughput** (8,340 aug/s, 0.0667 epochs/s ≈ **15s/epoch**) in a 120s `lab_train_cell` window. It's wired as run_sweep cell **LF1** (`extra_worker_args=["--fp16-eval"]`; the Cell dataclass passes it straight to the workers).

**CRITICAL — do NOT extrapolate training wall-clock from that epochs/s.** It was a *cold-buffer transient*. In a real sustained run (LF1, 2026-05-23): `wave_size=512` fills the 1.5M buffer in **~27 epochs**, after which `sgd_per_position × (the fast V=512 position inflow)` produces **~1300+ and *growing* SGD steps/epoch** → each epoch takes **~3 min** (the `train=` phase dominates: ~100-130s), not 15s. So "1000 epochs" is a **multi-day** run, not ~4 hours. This is the L11 perf-lab finding (V=512 fills the buffer faster → more SGD steps/epoch) showing up at production scale.

Two takeaways:
1. **Perf-lab `epochs/s` / `aug/s` measure GENERATION throughput in a short cold window; they do NOT predict the trainer's steady-state per-epoch cost**, which is set by `full-buffer × sgd_per_position → steps/epoch`. Always sanity-check the *real* `train=Xs` phase from `trainer.log` before quoting a wall-clock ETA. (Sibling lessons: [[feedback-self-play-eta]], [[project-perf-bench-lesson]].)
2. **But it learns fast *per epoch*** — LF1 went elo 437→776 in one epoch around the buffer-full transition (~epoch 28) precisely because each epoch does ~1300 SGD steps. So **epochs-to-a-given-elo may be FEWER** even though wall-per-epoch is higher. The honest comparison for any "faster" recipe is **wall-clock-to-elo + val/policy_ce quality**, never epochs/s or epoch-count alone.

Op note: `save_buffer_every` defaults high (WL5=100); if you kill a run before the first buffer-save, a `--resume` reloads the model but the 1.5M buffer re-fills from empty (≈27 cold epochs again). For a clean fresh restart use `run_sweep.py --cell <C> --clean`.

## Publishing snapshots — two destinations

There are TWO independent publish targets, and the user may mean either or both:

1. **HuggingFace** (`jasonyandell/gomoku-9x9`) — model registry. Anyone can `hf_hub_download` the weights.
2. **Cloudflare live SPA** (https://gomoku.jasonyandell.workers.dev) — the playable demo. The model is baked into the static bundle as `app/public/model.onnx` at deploy time.

If the user says "publish epoch N" or "push the latest", ask whether they mean HF, the live demo, or both — unless context already makes it obvious. Default to **both** when they say things like "ship it" or "publish."

### HuggingFace push

`gomoku/hf.py` slims a training checkpoint (drops optimizer + replay buffer, ~69 MB → ~1.3 MB) and pushes:

```bash
cd ~/code/gomoku && source .venv/bin/activate
python -m gomoku.hf push --checkpoint checkpoints/epoch0NNN.pt
# Optional: --name <filename>  (default: model.pt — overwrites in place)
# Optional: --repo <repo_id>   (default: jasonyandell/gomoku-9x9)
```

Each push updates `model.pt`, `config.json`, `training_state.json` (epoch + total_games + wandb_run_id) on the HF repo. HF auth comes from `~/.cache/huggingface/token` (already cached). Atomic; takes seconds.

### Cloudflare live SPA deploy

Three steps: re-export ONNX → commit `app/public/` → push (GH Actions does the deploy automatically). Total ~1-2 minutes including the workflow run.

```bash
cd ~/code/gomoku && source .venv/bin/activate

# 1. Re-export ONNX from the target checkpoint
python scripts/export_onnx.py --checkpoint checkpoints/epoch0NNN.pt
# Writes app/public/model.onnx + app/public/model.meta.json.
# Prints an ONNX-vs-PyTorch fidelity check — both deltas should be < 1e-4.

# 2. Commit + push
git add app/public/model.onnx app/public/model.meta.json
git commit -m "bump deployed model to epoch NNN (XXXX games)"
git push

# 3. Watch the deploy
gh run watch         # or: gh run list --workflow=deploy-app.yml --limit 3
```

After GH Actions reports success (~60 s), verify:
```bash
curl -s https://gomoku.jasonyandell.workers.dev/model.meta.json
# Should show the new epoch + total_games
```

### Publish to both at once

```bash
EPOCH=100  # or whatever
CKPT="checkpoints/epoch0${EPOCH}.pt"
python -m gomoku.hf push --checkpoint "$CKPT" && \
python scripts/export_onnx.py --checkpoint "$CKPT" && \
git add app/public/model.onnx app/public/model.meta.json && \
git commit -m "bump model to epoch ${EPOCH}" && \
git push
```

### When to publish

- **HF**: any meaningful checkpoint — research artifact, low cost. After major training milestones (every few hundred epochs) or when the user asks.
- **CF live demo**: less often. Each deploy invalidates browser caches of the 1.3 MB ONNX for returning visitors. Wait until the model is materially stronger (e.g., a +10% jump on `eval/vs_lookahead4_winrate` vs the deployed snapshot).
- Don't auto-publish every epoch. Noisy improvements aren't worth the noise of a deploy.

## Code repo on GitHub

Source lives at https://github.com/jasonyandell/gomoku (public). Push to main triggers the CF deploy workflow (`.github/workflows/deploy-app.yml`), so don't push half-baked WIP to main — if you wouldn't want it live in 60 seconds, don't push it yet.

## Unattended-run policy — fix infrastructure bugs, restart freely

When Jason is asleep / away (or just busy), the default posture is
**act on infrastructure bugs, escalate on training decisions.** This
includes **discarding training progress to get a clean run**. From
Jason 2026-05-21:

> "I'm totally tolerant of throwing away 850 epochs (well, leaving them
> alone for later perusal) if it means a clean run. If you need to
> restart to fix things, that's all approved as part of this training
> pipeline."

Clean state > preserved epochs. Don't ratchet through hot-resumes when
a fresh restart is cheaper to reason about.

### Pre-authorized autonomous actions

If a check detects any of these — diagnose, apply the fix, restart,
update the wiki, notify. **Always push notify what you did + the
final state at each transition.**

| Detected state | Fix | Restart strategy |
|---|---|---|
| Worker dying from a single deterministic exception (`ValueError: Probabilities contain NaN`, etc.) | Add the missing guard at the failure site (e.g. `if not np.isfinite(s) or s <= 0` at `gomoku/self_play.py:_sample_action`). Commit. | Prefer **cold restart** (fresh wandb id, e.g. `WL3 → WL3.1`). Hot-resume only if checkpoint integrity is verified (see "Hot-resume trap" below). |
| Same exception poisoning the training buffer (NaN/Inf in stored pi, etc.) | Sanitize at the storage path too, not just the play path. (WL3 lesson: pi was written into trajectories BEFORE `_sample_action` ran — fixing only the play path left the buffer poisoned.) | Cold restart. Buffer is in-memory; once poisoned, only a restart cleans it. |
| Trainer barrier-stalled because workers died (logs show some `expected` workers absent from `_records/v{N}/`) | Wipe the runaway `v{N}/` greedy-fill garbage before restart: `rm -rf sweep_runs/<cell>/checkpoints/_records/v*` (or selectively the open tile). | Cold restart preferred. The barrier-stall window often coincides with corrupted in-memory state — don't risk hot-resume here. |
| Disk pressure (records-dir runaway from greedy fill after a stall) | `rm -rf` the stale `v{N}/`. | No restart needed if everything else is healthy. |
| MPS `INT_MAX` at startup (buffer > ~1.56M positions on 17 planes × 81 cells) | Reduce capacity to ≤ 1.5M. Document. | Restart fresh. |
| Native code bug that requires investigation (e.g. C-level numerical overflow) | Apply Python-side band-aid for immediate continuity, **spawn a background `Agent` in a worktree to find the root cause** in parallel. When the agent lands the C fix, cherry-pick + rebuild + restart. (Don't try to keep running the buggy `.so` once the fix is in — restart.) | Cold restart after the rebuild. |

### Cold restart vs hot-resume — pick cold by default

Hot-resume is only correct when **both** the on-disk checkpoint AND
the in-memory state at crash were healthy. WL3 burned us here:

1. WL3 trainer crashed at e825 with healthy weights on disk.
2. First recovery attempt hot-resumed from e825 with only the play-path
   NaN guard. The trainer trained for 9 epochs on NaN-poisoned buffer
   (the storage-path guard didn't exist yet) and saved corrupted
   checkpoints e826-834.
3. `keep_last_n=3` then pruned the healthy e825 — the only NaN-free
   checkpoint we had — leaving only the corrupted ones.

**Default: cold restart with a new cell name suffix** (e.g. `WL3 →
WL3.1`, `WL3.1 → WL3.1.1`). This gives:

- Fresh wandb run id — clean charts, no backward step graphs
- Fresh sweep_runs / sweep_logs dirs — no risk of mixed-state pollution
- Old artifacts preserved as `sweep_runs/<old-cell>.dead-eN/` or
  `sweep_runs/<old-cell>.preFix-eN/` (rename, don't delete — forensic
  value is real, the GB cost is negligible)
- A clear lineage in the wiki: "WL3 → crashed → WL3.1 with fix → ..."

Hot-resume is only the right move if you're certain the buffer was
healthy (e.g. the crash was at startup, not mid-run). When in doubt,
cold.

### Things that require Jason (always)

- **Change a training hyperparameter** (lr, sims, batch, K, τ, EMA tau, etc.). These are *experimental* decisions, not infrastructure.
- **Stop a run that's still making progress.** Don't kill on a hunch.
- **Re-architect**: add new levers, change cell config, change buffer sampling. New cells need a title card (present it; no ACK).
- **Push to main of any other repo, deploy, or anything that affects external surfaces** (gomoku.jasonyandell.workers.dev, HuggingFace, etc.).
- **Reach for a completely new run design** (e.g. "let me try WL4 with a different lever") — that's a planning conversation.

### Patterns to be vigilant about

These are bug classes we've hit. Run-check checklist:

- **Float32 cast before normalize**: in numerical code (especially C extensions), if you `pow()` or `exp()` in `double` then cast to `float32` *before* normalizing, the cast can overflow to `+Inf` long before the normalized result would saturate. Always normalize in `double`, cast the [0,1] result. (Fixed in `_mcts_native.c::policy` commit `7c3e405`.)
- **NaN comparisons return False**: `if s <= 0: ...` silently passes NaN through. Always `if not np.isfinite(s) or s <= 0` when you want to catch degenerate distributions.
- **In-memory state vs disk state**: when triaging a crash, ask "was the in-memory buffer healthy at crash time?" If anything could have poisoned it (NaN write before the play-time guard, value collapse, etc.), cold restart.
- **`keep_last_n=3` is short**: surviving checkpoints can be overwritten in ~3 save_every cycles. If you need a healthy checkpoint preserved for restart, *copy it aside immediately* (`cp ... $CLAUDE_JOB_DIR/`). The eval-snapshot pattern in the runbook does this.

### Friction workarounds learned during WL5 overnight (2026-05-22)

The "smooth recipe" Jason calls out: cell-launch → cron-monitor →
cap-reach → workers cleanup → close-out → commit + push. These are the
sharp edges that nearly drew blood:

- **macOS `pgrep -fc` returns empty on no match, not "0"**. Polling
  loops like `until [ "$(pgrep -fc gomoku)" = "0" ]; do sleep 2; done`
  hang forever when the process is already gone. Use
  `ps -A | grep <pattern> | grep -v grep | wc -l` — always returns a
  number.
- **Concurrent worktrees inflate proc counts**. `pgrep -fc 'gomoku\.train'`
  matches sibling sessions' training processes (frontier perf benches,
  contour sweeps). Always **scope by cell name** in the grep:
  `ps -A | grep -E 'gomoku\.(train|selfplay_worker|eval_worker)' | grep <CELL>`.
  A proc count going *up* is concurrent activity, not corruption.
- **macOS `awk` has no 3-arg `match()`**. Stats summaries that
  capture groups via `match($0, /regex/, m)` syntax-error on macOS.
  Write a `python3 <<'EOF' ... EOF` heredoc instead — the run-end
  summary script in the runbook is the template.
- **zsh chokes on `==` / `===`** inside bash compound commands.
  Symptom: command output truncated at the first `=`. Either move
  the comparison into a python `-c` block or split the shell call.
- **Trainer cap-reach leaves workers + eval orphaned**. The trainer
  exits cleanly at `--epochs N`, prints the wandb finalize banner —
  but the 8 self-play workers and the eval worker keep polling.
  Process count drops from 10 to 9 (no trainer) and never lower until
  you `pkill -TERM -f 'sweep_runs/<CELL>/'`. Build this into the
  run-end procedure; don't assume the cron's "epoch >= cap" alert
  means everything stopped.
- **Buffer snapshot lags slim checkpoints** by `save_buffer_every`
  (default 100). Resuming from `latest.pt` rolls back the model to
  the buffer-snapshot epoch — *up to* 100 epochs behind the most
  recent `epochNNNN.pt`. This is **usually the right trade**: ~100
  epochs of model drift is recoverable in <30 min; rebuilding a 1.5M
  buffer takes hours. WL5 phase-2 resume burned 49 epochs to keep
  the buffer warm and the run was fine.
- **CronCreate auto-expires after 7 days**, dies on session restart
  (in-memory only). For runs longer than a week or that need to
  survive a Claude restart, fall back to `/loop` or pass
  `durable: true` explicitly.
- **Other-session commits piled up**. After an overnight, expect
  `git status` to show wiki changes from a sibling session AND the
  branch to be many commits ahead of origin. Commit the foreign
  changes as their own commit first (with a short message describing
  what they are), THEN your run-end commit. Reviewers benefit;
  history is cleaner.
- **`app/**` paths trigger Cloudflare deploy on push to main**. Run
  `git diff --stat origin/main..main -- app/ .github/` before
  pushing. If `app/` shows up unexpectedly, stop and decide whether
  a deploy is intended.

### How to write the autonomous-action commit

```
<area>: <one-line problem statement>

Detected at <time>. <2-3 sentence diagnosis>. Fix: <what changed>.
Recovery: <how the run was brought back>.

Pre-authorized by gomoku-train skill "unattended-run policy".
```

The commit message + the PushNotification together are the audit trail.

### Native-style "root cause requires investigation" pattern

If the immediate fix is "catch the exception and fallback" but the
underlying bug requires real investigation, do BOTH in parallel:

1. Apply the Python-side immediate-stability fix. Restart.
2. **Spawn a background `Agent` in a worktree** with a focused brief on
   the root cause. The agent works while the run continues with the
   stability patch.
3. When the agent lands a real fix, cherry-pick + rebuild + cold-restart
   the run to pick up the C-side correctness.

WL3 → WL3.1 was this pattern: Python band-aid kept it running (until
e92 where we had a chance to merge the C fix), then cold restart as
`44cxzc9d` once the C extension was rebuilt with the real fix in place.

## Common asks and how to handle them

| User says | Do |
|---|---|
| "start training" / "kick off a run" / "launch WLn" | **Follow the production launch runbook end-to-end** at `wiki/topics/launch-sequence-runbook.md`. Status check first. Title card before launching (present it; no ACK). Smoke if any new lever. Wiki + workspace updates after spin-up. |
| "resume from latest" | Single-process path: see RESUME command below. **Cell-based runs DO support resume** via `python scripts/run_sweep.py --cell <CELL> --epochs N --resume sweep_runs/<CELL>/checkpoints/latest.pt` — keeps wandb lineage, keeps the 1.5M-position buffer (at the cost of up to `save_buffer_every` epochs of weight drift). This is the WL5 phase-2 resume pattern. |
| "how's it going" / "status" | Run the status check block. Tail recent epoch lines. Quote W&B URL. |
| "stop training" | `pkill -f gomoku.train`. Confirm with status check. |
| "play against the latest" | Make sure web UI is up. If training is running, server should already be on CPU. Print the URL. |
| "watch it play itself" | Same — UI's Replay tab. |
| "spin up the UI" / "start the server" | Web UI start command. Print URL. |
| "make it stronger" | More `--n-simulations` (try 400), or step up `--size` (requires fresh start). Always with W&B for visibility. |
| "publish" / "push to HF" / "share the model" | Ask: HF only, live demo only, or both? Default to both for "ship it" / "publish." Use the relevant block in the "Publishing snapshots" section. Always report the HF URL + epoch + total_games, and the live URL if deployed. |
| "deploy the latest" / "update the live demo" | Re-export ONNX + commit `app/public/` + push (see CF SPA block). Watch `gh run watch`. Curl `/model.meta.json` to verify after. |
| "let me see a self-play" without UI | `python -m web.server` is overkill for one game — instead a one-liner: `python -c "from gomoku.eval import play_vs_random; ..."` or just point them at the UI. |
| "is the current eval actually measuring what we trained" / "ad-hoc match vs baseline" | Snapshot the live checkpoint aside (trainer's `keep_last_n=3` prunes fast): `cp sweep_runs/<cell>/checkpoints/$(ls -t sweep_runs/<cell>/checkpoints/epoch0*.pt \| head -1) $CLAUDE_JOB_DIR/test_ckpt.pt`. Then run a 40-game match on CPU (won't fight MPS training): `GOMOKU_DEVICE=cpu python -m gomoku.match "model:checkpoint=$CLAUDE_JOB_DIR/test_ckpt.pt,sims=200" vs heuristic --n-games 40`. For experiments with random openings or other off-by-default eval shapes, write the script into `$CLAUDE_JOB_DIR/` (one-off, not committed). 40 games gives much tighter CI than the trainer's default 16. |

## Don'ts

- **Don't** start a second training process without checking first. It'll fight the existing one for MPS and corrupt checkpoints.
- **Don't** delete the wandb/ directory while a run is active.
- **Don't** delete `sweep_runs/<cell>/` while it's running or while a follow-up cell might use it for past-checkpoint mix.
- **Don't** change `--size` mid-run. The model architecture is baked into the checkpoint.
- **Don't** size a replay buffer past **1.5M positions on MPS** — at 17 planes × 81 cells per position that's 2.07B elements, just under MPSGraph's INT_MAX (2.147B). 1.5M is the practical ceiling. 5M will crash on first `buffer.shape_stats()`.
- **Don't** launch wave-mode without the worker race-fix in place (commit `0d2c106` added "drop wave game" in `_atomic_save_wave_game`). Without it, a single crashed worker stalls the barrier forever.
- **Don't** train without `--wandb` unless the user explicitly asks. Visibility is the whole point.
- **Don't** run the web UI on MPS while training is on MPS unless the user specifically asked to play with full speed — it slows training perceptibly.
- **Don't** kill -9. SIGTERM (default `kill` or `pkill`) lets Python flush wandb and finish writing the in-flight checkpoint.
- **Don't** launch a multi-lever next-run config without a 30-epoch smoke first. WL1 ate two failed launches before catching the worker race; the smoke would have caught both.
- **Don't** auto-start the /loop monitor unless the user asks. They may want quiet observation; ask if unsure.

## Files to know

```
~/code/gomoku/
├── gomoku/
│   ├── train.py           # trainer; --ema-tau, --grad-accum-steps, --wave-mode flags
│   ├── selfplay_worker.py # wave-mode worker; --opponent-mix-*, --weights-poll-{min,max}-sec
│   ├── self_play.py       # parallel game generation (called by both paths)
│   ├── mcts.py            # PUCT MCTS dispatcher (uses native C engine when available)
│   ├── _mcts_native.c     # native MCTS engine — 1.28x throughput win, opt-out via GOMOKU_DISABLE_NATIVE_MCTS=1
│   ├── model.py           # ResNet (tiny/small/medium/large presets)
│   ├── game.py            # 9x9 free-style rules, symmetries
│   ├── eval.py            # vs random / vs other checkpoint
│   ├── cli.py             # terminal play — `python -m gomoku.cli --checkpoint ...`
│   └── util.py            # device pick, keychain wandb-key pull
├── scripts/
│   ├── run_sweep.py       # production launcher: cells (A-F, Z, Zc, WL1, WL2) + trainer/worker cmd builders
│   └── wandb_workspace.py # one-shot generator for the multi-run overlay workspace
├── wiki/
│   ├── index.md           # entry point
│   ├── topics/launch-sequence-runbook.md     # the production launch playbook
│   ├── topics/wave-of-lockstep-design.md     # WL1 design
│   └── topics/wl2-scale-emulation-design.md  # WL2 design
├── web/
│   ├── server.py          # FastAPI server — `python -m web.server`
│   └── static/            # index.html + app.js + style.css
├── checkpoints/           # single-process path artifacts (epochNNNN.pt + latest.pt symlink)
├── sweep_runs/<cell>/     # production-path per-cell artifacts (checkpoints + _records outbox)
├── sweep_logs/<cell>/     # production-path per-cell logs (trainer.log + wN.log + eval.log)
├── TRAINING_WIKI.md       # lab notebook; "<CELL> live run log" sections per run
├── scratch/               # train.log, web.log, ad-hoc files
└── wandb/                 # per-run W&B local mirror (run-<ts>-<id>/)
```

## Useful one-liners

```bash
# Count games trained across all checkpoints
ls checkpoints/epoch*.pt | tail -1 | xargs python -c "import sys,torch; print('total_games:', torch.load(sys.argv[1], weights_only=False, map_location='cpu').get('total_games'))"

# Quick matchup vs any baseline (separate from training, doesn't touch its MPS)
GOMOKU_DEVICE=cpu python -m gomoku.match \
  "model:checkpoint=checkpoints/latest.pt,sims=200" vs heuristic --n-games 20

# Full baseline matrix (random < heuristic < lookahead:d=2 < lookahead:d=4)
GOMOKU_DEVICE=cpu python -m gomoku.match --matrix \
  random heuristic "lookahead:depth=2" "lookahead:depth=4" --n-games 30

# Tail per-epoch output from the active run (works around stdout-buffering)
tail -f wandb/run-*/files/output.log 2>/dev/null | tail -F

# Publish current snapshot to HF
python -m gomoku.hf push --checkpoint checkpoints/latest.pt
```
