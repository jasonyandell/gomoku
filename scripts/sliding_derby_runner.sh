#!/usr/bin/env bash
# SLIDING DERBY runner (issue #38/#39) — the GPU producer for the overnight lab.
#
# Trains the #36 G15-defense cell in resumable, time-capped slices, and BETWEEN
# slices runs the frozen-reference promotion gate (scripts/sliding_gate.py).
# The gate is AUTONOMOUS here: a PROMOTE actually slides the frozen peak (the
# candidate becomes the new reference). This is safe — the seed champion file
# (eval502) is never mutated; the peak is a derby-owned copy.
#
# RELIABLE EVALS ONLY (Jason, 2026-06-16): NO wine engines. The verdict is the
# anchor-free net-vs-frozen-peak H2H (pure torch, always reliable). The secondary
# white-loss signal (#18) is also net-vs-peak (reliable), NOT vs zetor17 (wine,
# nixed). Calibration-immune (#35): the gate never reads absolute Elo.
#
# Crash-resumable: each slice resumes the cell's own latest.pt (seed only on the
# very first slice). Stop cleanly:  touch /tmp/STOP_SLIDING_DERBY
# Detach:  nohup bash scripts/sliding_derby_runner.sh >/dev/null 2>&1 &
set -uo pipefail
REPO=/Users/jason/code/gomoku
CELL_KEY=G15-defense
CELL_NAME=G15-defense-board15
SEED="$REPO/sweep_runs/g15_128x10_bigbuf_eval502.pt"     # warm-start, first slice only
SLICE="${SLICE:-3600}"                                    # train wall-secs per slice
GATE_N="${GATE_N:-120}"                                   # H2H games per gate (power: ~120 resolves ~70 elo)
STOP=/tmp/STOP_SLIDING_DERBY
RUNDIR="$REPO/sweep_runs/$CELL_NAME"
LATEST="$RUNDIR/checkpoints/latest.pt"
LOGDIR="$REPO/sweep_logs/$CELL_NAME"
WLOG="$LOGDIR/sliding_runner.log"
BOARD="$REPO/sweep_runs/sliding_derby_board.json"
VLOG="$REPO/sweep_runs/sliding_derby_verdicts.jsonl"
PEAKOUT="$REPO/sweep_runs/sliding_derby_peak.pt"
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
export GOMOKU_BOARD_SIZE=15 GOMOKU_DEVICE=mps
lap=0
echo "[runner] START $(date) cell=$CELL_KEY slice=${SLICE}s gate_n=$GATE_N" >> "$WLOG"
while [ ! -f "$STOP" ]; do
  lap=$((lap+1))
  if [ -f "$LATEST" ]; then resume="$LATEST"; else resume="$SEED"; fi
  echo "[runner] lap=$lap TRAIN slice resume=$(basename "$resume") $(date)" >> "$WLOG"
  ( source .venv/bin/activate; GOMOKU_BOARD_SIZE=15 python scripts/run_sweep.py \
      --cell "$CELL_KEY" --resume "$resume" --max-wall-secs "$SLICE" --foreground ) \
      >> "$LOGDIR/slice.log" 2>&1
  if [ ! -f "$LATEST" ]; then
    echo "[runner] lap=$lap no latest.pt produced — skipping gate" >> "$WLOG"; sleep 5; continue
  fi
  cand="$RUNDIR/checkpoints/sliding_lap${lap}_candidate.pt"
  cp "$LATEST" "$cand"
  echo "[runner] lap=$lap GATE candidate=$(basename "$cand") vs frozen peak (n=$GATE_N) $(date)" >> "$WLOG"
  ( source .venv/bin/activate; GOMOKU_BOARD_SIZE=15 python scripts/sliding_gate.py \
      --candidate "$cand" --board "$BOARD" --verdict-log "$VLOG" --peak-out "$PEAKOUT" \
      --n-games "$GATE_N" --sims 200 --device mps --white-loss ) \
      >> "$LOGDIR/gate.log" 2>&1
  verdict=$(tail -40 "$LOGDIR/gate.log" | grep -oE '(PROMOTE|REVERT)' | tail -1)
  echo "[runner] lap=$lap GATE verdict=${verdict:-?} $(date)" >> "$WLOG"
  sleep 3
done
echo "[runner] STOPPED (STOP file) $(date)" >> "$WLOG"
