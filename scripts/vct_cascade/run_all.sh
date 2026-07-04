#!/usr/bin/env bash
# End-to-end VCT cascade pipeline (issue #97): extract -> dedup -> cascade.
# Resumable reducer — re-running skips finished shards/rows. All stages log to
# <out>/pipeline.log. Run from the worktree (uv resolves the right env).
#
#   GOMOKU_BOARD_SIZE=15 bash scripts/vct_cascade/run_all.sh \
#       ~/data/games_raphi ~/data/raphi_vct 12 50,100,250,500,1000,2000,4000,10000
set -euo pipefail

GAMES_DIR="${1:?games-dir}"
OUT="${2:?out}"
WORKERS="${3:-12}"
LADDER="${4:-50,100,250,500,1000,2000,4000,10000}"

export GOMOKU_BOARD_SIZE=15
mkdir -p "$OUT"
LOG="$OUT/pipeline.log"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== STAGE 1: extract ($GAMES_DIR -> $OUT, $WORKERS workers) ==="
uv run python -m scripts.vct_cascade.extract extract \
    --games-dir "$GAMES_DIR" --out "$OUT" --workers "$WORKERS" 2>&1 | tee -a "$LOG"

say "=== STAGE 2: dedup (-> positions/) ==="
uv run python -m scripts.vct_cascade.extract dedup \
    --out "$OUT" --workers "$WORKERS" 2>&1 | tee -a "$LOG"

say "=== STAGE 3: cascade (ladder $LADDER) ==="
uv run python -m scripts.vct_cascade.cascade \
    --out "$OUT" --ladder "$LADDER" 2>&1 | tee -a "$LOG"

say "=== PIPELINE COMPLETE ==="
uv run python -m scripts.vct_cascade.stats --out "$OUT" 2>&1 | tee -a "$LOG"
