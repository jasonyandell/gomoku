#!/usr/bin/env bash
# GPU gate (issue #97): wait until MLX/Metal can build a kernel again, THEN launch
# the labeling cascade under the watchdog. Survives a Metal compiler-service wedge
# that self-heals (idle backoff) without needing a human at the exact recovery
# moment. Does NOT survive a reboot — if the machine is rebooted to clear the
# wedge, just re-run this script.
set -uo pipefail

OUT="${1:-$HOME/data/raphi_vct}"
WAIT_MIN="${WAIT_MIN:-180}"     # give up waiting after this long
POLL="${POLL:-60}"
export GOMOKU_BOARD_SIZE=15
cd "$(dirname "$0")/../.." || exit 1

LOG="$OUT/gpu_gate.log"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
gpu_ok() {
  uv run python -c "import mlx.core as mx; mx.eval(mx.ones((64,64))@mx.ones((64,64)))" \
    >/dev/null 2>&1
}

say "gpu_gate up — waiting for Metal compiler to recover (<= ${WAIT_MIN}m, poll ${POLL}s)"
deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
until gpu_ok; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "ALERT: GPU still wedged after ${WAIT_MIN}m — giving up. Likely needs a reboot."
    say "       After the GPU is back, relaunch: bash scripts/vct_cascade/watchdog.sh"
    exit 1
  fi
  sleep "$POLL"
done
say "GPU recovered — launching watchdog + cascade"
exec bash scripts/vct_cascade/watchdog.sh "$OUT"
