#!/usr/bin/env bash
# Eval-GRADIENT loop for the idx-2 warm-start AZ run (#86). Every 45 min, play the
# newest epochNNNN.pt @idx-2 (white split) against native Rapfi graded by think-time
# (rapfi@{25,50,100,250,1000}ms — see eval_gradient.DEFAULT_RUNGS) and append one
# GRADIENT line to az_gradient.log — so progress is visible long before the net can
# touch max Rapfi (which alone reads 0 for hours). Detached cadence worker (survives
# the session); a fresh session tails az_gradient.log. Exits when the AZ run is gone.
set -uo pipefail
cd /Users/jason/code/gomoku-gentle-rapfi-teacher
export GOMOKU_BOARD_SIZE=15
AZ_DIR=sweep_runs/G15-idx2-warmstart-board15
AZ_CKPTS=$AZ_DIR/checkpoints
LOG=mined/az_gradient.log
echo "[gradient-loop] $(date) start (every 45m, newest epoch*.pt, n=12 sims=160)" >> "$LOG"
while pgrep -f "$AZ_DIR/" >/dev/null 2>&1; do
  sleep 2700
  CKPT=$(ls -t "$AZ_CKPTS"/epoch*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.eval_gradient \
      --checkpoint "$CKPT" --n-games 12 --sims 160 2>&1 \
      | grep -E 'GRADIENT|Error|Traceback' >> "$LOG"
  fi
done
echo "[gradient-loop] $(date) AZ gone; exit" >> "$LOG"
