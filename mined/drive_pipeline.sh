#!/usr/bin/env bash
# Bruce-Lee idx-2 pipeline driver (#86): wait for the mine to finish, then
# pretrain, then gate the pretrained net vs Rapfi @idx-2. Sequential — the mine
# owns ~15 CPU cores; pretrain wants the machine to itself. Logs everything.
set -uo pipefail
cd /Users/jason/code/gomoku-gentle-rapfi-teacher
export GOMOKU_BOARD_SIZE=15
SHARDS=mined/idx2_15x15
CKPT=checkpoints/idx2_pretrain.pt
LOG=mined/pipeline.log
TARGET=1000000

echo "[drive] $(date) waiting for mine to reach >= $TARGET ..." >> "$LOG"
# Wait until the mine process is gone OR on-disk count >= TARGET.
while pgrep -f "gomoku.rapfimine run" >/dev/null 2>&1; do
  sleep 30
done
echo "[drive] $(date) mine process exited; counting shards ..." >> "$LOG"
N=$(uv run python -m gomoku.rapfimine status --out "$SHARDS" 2>/dev/null | grep -oE '[0-9]+ examples' | grep -oE '[0-9]+')
echo "[drive] $(date) on disk: ${N:-?} examples" >> "$LOG"

echo "[drive] $(date) PRETRAIN start (large, gp=1, stem=1, 12 epochs warm-start seed)" >> "$LOG"
# 12 epochs (was 40): a warm-START seed only needs to capture Rapfi's idx-2 move
# preferences (policy_ce ~2.0); AZ self-play continues training. save-every 3 =
# crash-robust + lets AZ launch from an early checkpoint if needed.
uv run python -m gomoku.rapfimine.pretrain \
  --shards "$SHARDS" --out "$CKPT" --size large --global-pool 1 --stem-padding 1 \
  --epochs 12 --batch-size 1024 --lr 3e-4 --value-weight 0.5 --save-every 3 \
  >> mined/pretrain.log 2>&1
echo "[drive] $(date) PRETRAIN done -> $CKPT" >> "$LOG"

echo "[drive] $(date) GATE pretrained net vs Rapfi @idx-2 (n=48)" >> "$LOG"
uv run python -m gomoku.rapfimine.eval_idx2 \
  --checkpoint "$CKPT" --n-games 48 --sims 160 >> mined/gate.log 2>&1
echo "[drive] $(date) GATE done; tail:" >> "$LOG"
tail -3 mined/gate.log >> "$LOG"

# --- stage 3: warm-start AlphaZero from the pretrained net, idx-2 ONLY ----------
# run_sweep default-backgrounds (fire-and-forget) the trainer + 3 workers, then
# returns; GOMOKU_DROP_OPENERS restricts self-play to the single idx-2 opening
# (D4 symmetry recovered by the trainer's sample-time augment). Guarded on the
# pretrain seed existing (a crashed pretrain must not launch AZ from a stale net).
AZ_DIR=sweep_runs/G15-idx2-warmstart-board15
AZ_CKPTS=$AZ_DIR/checkpoints
# Probe the NEWEST epochNNNN.pt (weights, written every save_every=5). NOT
# latest.pt — that embeds the 1.4 GB buffer and is only rewritten every
# save_buffer_every=100 epochs, so it does not exist for the first ~100 epochs.
if [ -f "$CKPT" ]; then
  echo "[drive] $(date) WARMSTART AlphaZero @idx-2 (G15-idx2-warmstart --resume $CKPT)" >> "$LOG"
  GOMOKU_DROP_OPENERS=0,1,3,4,5,6,7,8 GOMOKU_BOARD_SIZE=15 \
    uv run python scripts/run_sweep.py --cell G15-idx2-warmstart --resume "$CKPT" \
    >> mined/az_launch.log 2>&1
  echo "[drive] $(date) AZ launched (backgrounded by run_sweep); see sweep_logs/" >> "$LOG"
else
  echo "[drive] $(date) ABORT stage 3: pretrain seed $CKPT missing (pretrain crashed?)" >> "$LOG"
  echo "[drive] $(date) PIPELINE COMPLETE (stages 1-2 only)" >> "$LOG"
  exit 1
fi

# --- stage 4: periodic H2H of the live AZ net vs Rapfi @idx-2 -------------------
# "Does it stand a chance?" — probe latest.pt every 30m while AZ runs; one curve
# of idx-2 strength over training. Modest CPU (48 games @160 sims ~ a few min,
# ~30m cadence) so it does not starve self-play. Survives a missing-yet ckpt.
PROBE_LOG=mined/az_vs_rapfi.log
echo "[drive] $(date) entering periodic Rapfi-probe loop (every 30m) -> $PROBE_LOG" >> "$LOG"
while pgrep -f "$AZ_DIR/" >/dev/null 2>&1; do
  sleep 1800
  CKPT=$(ls -t "$AZ_CKPTS"/epoch*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    echo "[probe] $(date) H2H $CKPT vs Rapfi @idx-2 (n=48 sims=160)" >> "$PROBE_LOG"
    GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.eval_idx2 \
      --checkpoint "$CKPT" --n-games 48 --sims 160 >> "$PROBE_LOG" 2>&1
  fi
done
echo "[drive] $(date) AZ process gone; probe loop exit. PIPELINE COMPLETE" >> "$LOG"
