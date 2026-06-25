#!/bin/bash
# Crash-resumable training WORKHORSE for autonomous overnight ops (token-free).
# Runs run_sweep in resumable time-capped slices forever, auto-relaunching on
# slice-cap OR crash, with a periodic heartbeat from trainer.log. Pairs with a
# Claude cron that wakes to eval vs Rapfi + update the wiki (see CLAUDE.md
# "Autonomous overnight ops"). Detach with: nohup ... >/dev/null 2>&1 &
#
# Usage: train_workhorse.sh <CELL_KEY> <CELL_NAME> <SEED_CKPT> [SLICE_SECS=3600] [HB_SECS=900]
#   CELL_KEY  = run_sweep cell key (e.g. G15-96x8-deepgen)
#   CELL_NAME = run dir name      (e.g. G15-96x8-deepgen-board15)
#   SEED_CKPT = warm-start checkpoint used ONLY on the first launch (no latest.pt yet)
# Stop cleanly:  touch /tmp/STOP_TRAIN
set +e
REPO=/Users/jason/code/gomoku
KEY="$1"; NAME="$2"; SEED="$3"; SLICE="${4:-3600}"; HB="${5:-900}"
STOP=/tmp/STOP_TRAIN
RUNDIR="$REPO/sweep_runs/$NAME"
TLOG="$REPO/sweep_logs/$NAME/trainer.log"
WLOG="$REPO/sweep_logs/$NAME/workhorse.log"
mkdir -p "$REPO/sweep_logs/$NAME"
cd "$REPO" || exit 1
echo "[workhorse] START $(date) key=$KEY name=$NAME seed=$(basename "$SEED") slice=${SLICE}s hb=${HB}s" >> "$WLOG"
while [ ! -f "$STOP" ]; do
  latest="$RUNDIR/checkpoints/latest.pt"
  if [ -f "$latest" ]; then resume="$latest"; else resume="$SEED"; fi
  echo "[workhorse] launch slice resume=$(basename "$resume") $(date)" >> "$WLOG"
  ( GOMOKU_BOARD_SIZE=15 uv run python scripts/run_sweep.py \
      --cell "$KEY" --resume "$resume" --max-wall-secs "$SLICE" --foreground ) \
      >> "$REPO/sweep_logs/$NAME/slice.log" 2>&1 &
  RSPID=$!
  while kill -0 "$RSPID" 2>/dev/null; do
    sleep "$HB"
    hb=$(grep '^epoch ' "$TLOG" 2>/dev/null | tail -1)
    echo "[hb $(date +%H:%M)] ${hb:-<no epoch line yet>}" >> "$WLOG"
    if [ -f "$STOP" ]; then echo "[workhorse] STOP file -> terminating slice" >> "$WLOG"; kill "$RSPID" 2>/dev/null; break; fi
  done
  wait "$RSPID" 2>/dev/null
  echo "[workhorse] slice ended rc=$? $(date)" >> "$WLOG"
  sleep 5
done
echo "[workhorse] STOPPED (STOP file present) $(date)" >> "$WLOG"
