#!/usr/bin/env bash
# Eval-GRADIENT loop for the idx-2 warm-start AZ run (#86). Every 20 min, play the
# newest epochNNNN.pt @idx-2 (white split) against native Rapfi graded by think-time
# (rapfi@{25,50,100,200}ms — see fast_eval.DEFAULT_RUNGS) and append one GRADIENT
# line to az_gradient.log. Uses the FAST batched harness (net MCTS batched across
# all games + Rapfi fanned out per-timeout) at sims=32 → ~20 s/pass even while AZ
# trains, vs minutes for the serial eval. Detached cadence worker (survives the
# session); a fresh session tails az_gradient.log. Exits when the AZ run is gone.
set -uo pipefail
cd /Users/jason/code/gomoku-gentle-rapfi-teacher
export GOMOKU_BOARD_SIZE=15
AZ_DIR=sweep_runs/G15-idx2-warmstart-board15
AZ_CKPTS=$AZ_DIR/checkpoints
LOG=mined/az_gradient.log
echo "[gradient-loop] $(date) start (every 20m, newest epoch*.pt, fast_eval n=12 sims=32)" >> "$LOG"
while pgrep -f "$AZ_DIR/" >/dev/null 2>&1; do
  sleep 1200
  CKPT=$(ls -t "$AZ_CKPTS"/epoch*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    # Snapshot first: the trainer keeps only --keep-last-n 3, so a checkpoint can
    # be rotated out mid-eval. Copy to a stable epoch-preserving name so fast_eval
    # still parses the epoch number.
    SNAP=mined/_snap_$(basename "$CKPT")
    if cp "$CKPT" "$SNAP" 2>/dev/null; then
      GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.fast_eval \
        --checkpoint "$SNAP" --n-games 12 --sims 32 2>&1 \
        | grep -E 'GRADIENT|Error|Traceback' >> "$LOG"
      rm -f "$SNAP"
    fi
  fi
done
echo "[gradient-loop] $(date) AZ gone; exit" >> "$LOG"
