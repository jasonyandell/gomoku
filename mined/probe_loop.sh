#!/usr/bin/env bash
# Standalone Rapfi-probe loop for the idx-2 warm-start AZ run (#86). Every 30 min,
# H2H the NEWEST epochNNNN.pt vs native Rapfi-NNUE @idx-2 (n=48), by color, into
# az_vs_rapfi.log — the verdict curve. Probes epoch*.pt (NOT latest.pt: that embeds
# the 1.4 GB buffer and is only written every save_buffer_every=100 epochs).
# Exits when the AZ run is gone. Detached; the experiment processes are independent.
set -uo pipefail
cd /Users/jason/code/gomoku-gentle-rapfi-teacher
export GOMOKU_BOARD_SIZE=15
AZ_DIR=sweep_runs/G15-idx2-warmstart-board15
AZ_CKPTS=$AZ_DIR/checkpoints
PROBE_LOG=mined/az_vs_rapfi.log
echo "[probe-loop] $(date) start (every 60m, newest epoch*.pt, n=48 sims=160)" >> "$PROBE_LOG"
while pgrep -f "$AZ_DIR/" >/dev/null 2>&1; do
  sleep 3600
  CKPT=$(ls -t "$AZ_CKPTS"/epoch*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    echo "[probe] $(date) H2H $CKPT vs Rapfi @idx-2 (n=48 sims=160)" >> "$PROBE_LOG"
    GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.eval_idx2 \
      --checkpoint "$CKPT" --n-games 48 --sims 160 >> "$PROBE_LOG" 2>&1
  fi
done
echo "[probe-loop] $(date) AZ gone; exit" >> "$PROBE_LOG"
