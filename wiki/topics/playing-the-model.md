# Playing the model

How to actually sit down at a board and play a trained checkpoint. Two surfaces:

1. **Local web UI** (`python -m web.server`) — picks any checkpoint on disk, runs full MCTS via the trained model. This is the strong-play surface.
2. **Live SPA** (https://gomoku.jasonyandell.workers.dev) — static site that bakes one snapshot in as `app/public/model.onnx`. Convenient anywhere; usually behind whatever's local.

The terminal CLI (`gomoku-play --checkpoint …`) also exists but the web UI is strictly nicer; skip it unless you want a no-deps shell session.

## TL;DR — play the latest local checkpoint

```bash
cd ~/code/gomoku

# Find the most recent cell with checkpoints
ls -dt sweep_runs/*/checkpoints | head -3

# Pick one (example: WL4 plateau end) and point the UI there.
# Use MPS if no training is running; CPU if it is, to avoid contention.
pkill -f "web.server" 2>/dev/null
PYTORCH_ENABLE_MPS_FALLBACK=1 nohup .venv/bin/python -m web.server \
  --port 8766 \
  --checkpoints-dir sweep_runs/WL4-no-random-openings.plateau-e4024/checkpoints \
  > scratch/web.log 2>&1 &
```

Open http://127.0.0.1:8766 → **play** tab → checkpoint dropdown → pick the
highest epoch (e.g. `epoch4024.pt`).

If training IS running, prepend `GOMOKU_DEVICE=cpu` so the UI doesn't fight the
trainer for MPS:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 GOMOKU_DEVICE=cpu nohup .venv/bin/python -m web.server \
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

Static ONNX, no server. Snapshot is whatever was last `git push`-ed via
`scripts/export_onnx.py`. Check what's live:

```bash
curl -s https://gomoku.jasonyandell.workers.dev/model.meta.json
# {"epoch": 100, "total_games": 6400, ...}  ← whatever the deployed snapshot is
```

If the live model is meaningfully behind a strong local checkpoint and you want
to update it, see the **Cloudflare live SPA deploy** block in the
`gomoku-train` skill — it's a three-step `export_onnx.py` → commit
`app/public/` → push flow.

## Common annoyances

- **"latest.pt is huge"**: that's the full training state (model + optimizer +
  replay buffer) saved for resume. UI loads it eventually but it's slow. Pick a
  plain `epochNNNN.pt` instead.
- **"the UI feels weak even at 800 sims"**: confirm you're not on the
  `epoch0136.pt` smoke checkpoint in `./checkpoints/`. The dropdown shows the
  epoch — if it's three digits you're playing an early training snapshot.
- **Port 8766 already in use**: another `web.server` is running. `pkill -f
  web.server` then relaunch, or pick a different `--port`.
- **MPS contention with active training**: if training is up, use
  `GOMOKU_DEVICE=cpu` for the UI — see the launch runbook for why.
