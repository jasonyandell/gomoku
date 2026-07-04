#!/usr/bin/env bash
# Supervised throughput characterization across node depths (issue #97).
# Each cap's sweep is timeout-wrapped so a deep budget can't hang the night;
# completed dispatches' perf rows persist (atomic) even if a cap is cut off.
# NOT -e: one cap timing out must not abort the rest.
set -uo pipefail

OUT="${1:-$HOME/data/raphi_vct}"
POS="${2:-$HOME/data/raphi_vct/positions/*.parquet}"
export GOMOKU_BOARD_SIZE=15
LOG="$OUT/sweep_all.log"

run() {  # cap  widths  secs-per-width  timeout
  echo "[$(date '+%H:%M:%S')] === sweep cap=$1 widths=$2 ===" | tee -a "$LOG"
  timeout "$4" uv run python -m scripts.vct_cascade.sweep \
      --positions "$POS" --cap "$1" --widths "$2" --secs-per-width "$3" --out "$OUT" \
      2>&1 | tee -a "$LOG" \
    || echo "  (cap=$1 cut off at ${4}s or errored — partial perf kept)" | tee -a "$LOG"
}

# standard ladder — wide widths, the knee climbs as cap falls
for c in 100 250 500 1000 2000 4000 10000; do
  run "$c" 8192,65536,524288,2097152 6 150
done
# deep caps — small widths + short observation; "does 1M/2M even run?" + depth limit
for c in 20000 50000 100000 1000000 2000000; do
  run "$c" 1024,8192,65536 4 150
done
echo "[$(date '+%H:%M:%S')] === sweep_all complete ===" | tee -a "$LOG"
