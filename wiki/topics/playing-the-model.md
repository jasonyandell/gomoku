# Playing the model

> ✅ **LIVE — procedure current.** Invocation is the canonical `uv run gomoku-web`
> (per CLAUDE.md; the `gomoku-web` console script *is* `web.server:main`, same
> `--checkpoints-dir`/`--port` flags). The "strongest checkpoint" example below
> (WL4 e4024) is **era-1 (9×9, 2026-05-21)** — for the current strongest snapshot
> see [training-run-lineage.md](training-run-lineage.md).

How to actually sit down at a board and play a trained checkpoint. Two surfaces:

1. **Local web UI** (`uv run gomoku-web`) — picks any checkpoint on disk, runs full MCTS via the trained model. This is the strong-play surface.
2. **Live SPA** (https://gomoku.jasonyandell.workers.dev) — static site that bakes one snapshot in as `app/public/model.onnx`. Convenient anywhere; usually behind whatever's local.

The terminal CLI (`gomoku-play --checkpoint …`) also exists but the web UI is strictly nicer; skip it unless you want a no-deps shell session.

## TL;DR — play the latest local checkpoint

```bash
cd ~/code/gomoku

# Find the most recent cell with checkpoints
ls -dt sweep_runs/*/checkpoints | head -3

# Pick one (example: WL4 plateau end) and point the UI there.
# Use MPS if no training is running; CPU if it is, to avoid contention.
pkill -f "gomoku-web" 2>/dev/null
PYTORCH_ENABLE_MPS_FALLBACK=1 nohup uv run gomoku-web \
  --port 8766 \
  --checkpoints-dir sweep_runs/WL4-no-random-openings.plateau-e4024/checkpoints \
  > scratch/web.log 2>&1 &
```

Open http://127.0.0.1:8766 → **play** tab → checkpoint dropdown → pick the
highest epoch (e.g. `epoch4024.pt`).

If training IS running, prepend `GOMOKU_DEVICE=cpu` so the UI doesn't fight the
trainer for MPS:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 GOMOKU_DEVICE=cpu nohup uv run gomoku-web \
  --port 8766 \
  --checkpoints-dir sweep_runs/<active-cell>/checkpoints \
  > scratch/web.log 2>&1 &
```

## Picking the right checkpoint

| Source | Path pattern | Notes |
|---|---|---|
| Production runs (WL1/WL2/WL3/WL4/…) | `sweep_runs/<cell>/checkpoints/epochNNNN.pt` | What you almost always want. Small (~5 MB), slim. |
| Production resume artifact | `sweep_runs/<cell>/checkpoints/latest.pt` | **Big** (~8 GB) — full state incl. replay buffer for resume. The UI can load it but loading is slow. Prefer `epochNNNN.pt`. |
| Single-process runs (`python -m gomoku.train`) | `checkpoints/epochNNNN.pt` | Mostly the old A-F / smoke / ad-hoc lineage. |

Quick way to discover what's currently the strongest snapshot: see the latest
"plateau-end" / "best WL series outcome" entries in `TRAINING_WIKI.md` and the
`wiki/log.md`. As of 2026-05-21, **WL4 epoch 4024** is the best WL-series
checkpoint we have.

The web UI's checkpoint dropdown shows `epoch`, file size, and total_games for
each pick, sorted by epoch — pick the largest epoch unless you want to A/B test
two snapshots.

## Play tab — the knobs that matter

| Knob | Default | What to change to make it harder |
|---|---|---|
| **model sims** | 400 | Bump to 800-1600 for noticeably stronger play (slower per move). 200 ≈ training-strength. |
| **your color** | black | Black moves first on 9x9 free-style. Pick white if you want a small edge. |
| **temperature moves** | 0 | Replay-tab only. Play tab is fully greedy, so the model picks the argmax. |

The play tab also shows the model's **top-K candidate moves** with their
visit-count distributions after each search — useful to learn what the model
"sees" and where you might be able to surprise it.

## Replay tab — watching self-play

Same checkpoint dropdown. Set `sims per move` (this is the search budget the
model will use) and `temperature moves` (number of opening plies with sampling
on — 8 is reasonable to get variety, 0 = fully deterministic). Hit **generate**
to draw a fresh game; step through ply-by-ply with the slider.

Two endpoints behind the scenes:
- `/api/selfplay-detailed` — runs MCTS live for each move. Faithful but slow.
- `/api/selfplay` — fallback that argmaxes a stored policy if MCTS isn't viable.
  You almost always want detailed.

## Live SPA (gomoku.jasonyandell.workers.dev)

Static ONNX, no server. As of 2026-05-21 the SPA serves **multiple models**
side-by-side via an in-app dropdown — pick any from the header. The set is
indexed at `/models.json`:

```bash
curl -s https://gomoku.jasonyandell.workers.dev/models.json | jq
# [{ id: wl4-e4024 (default), 17-plane, 458kg },
#  { id: epoch0100, 3-plane legacy, 6.4kg }]
```

### Adding another model to the live SPA

1. Export ONNX with the right plane count (the exporter derives it from
   `model_config.n_input_planes` automatically):
   ```bash
   python scripts/export_onnx.py \
     --checkpoint sweep_runs/<cell>/checkpoints/epochNNNN.pt \
     --out app/public/<id>.onnx
   ```
2. Append an entry to `app/public/models.json`:
   ```json
   {
     "id": "<id>", "label": "<human label>",
     "url": "/<id>.onnx", "meta_url": "/<id>.meta.json",
     "n_input_planes": 17, "epoch": N, "total_games": G,
     "n_filters": 64, "n_blocks": 4
   }
   ```
   Set `"default": true` on the one that should load first. Only one entry
   should have `default: true`.
3. `git add app/public/<id>.onnx app/public/<id>.meta.json app/public/models.json`
   and push. GH Actions deploys in ~60 s.

### Feature-plane caveat (3 vs 17)

The SPA carries both featurizers: 3-plane (legacy, no history) and 17-plane
(AZ-style, 8 plies of history per side + const). Adding a *new* plane count
would require extending `app/src/game.ts:toPlanes` and the golden parity test
at `app/src/__tests__/planes17.test.ts`. The Python source of truth is
`gomoku/game.py:to_planes`.

## Common annoyances

- **"latest.pt is huge"**: that's the full training state (model + optimizer +
  replay buffer) saved for resume. UI loads it eventually but it's slow. Pick a
  plain `epochNNNN.pt` instead.
- **"the UI feels weak even at 800 sims"**: confirm you're not on the
  `epoch0136.pt` smoke checkpoint in `./checkpoints/`. The dropdown shows the
  epoch — if it's three digits you're playing an early training snapshot.
- **Port 8766 already in use**: another `web.server` is running. `pkill -f gomoku-web` then relaunch, or pick a different `--port`.
- **MPS contention with active training**: if training is up, use
  `GOMOKU_DEVICE=cpu` for the UI — see the launch runbook for why.
