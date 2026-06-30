#!/usr/bin/env bash
# Overnight watchdog for the labeling cascade (issue #97).
# Runs the cascade as a child and polls every POLL seconds. The cascade is
# resumable (ledger row-count offset), so recovery = just restart it.
#   * child dies WITHOUT the completion marker  -> crash -> restart
#   * child dies WITH    the completion marker  -> success -> stop
#   * no new result shard for STALL_MIN minutes -> wedged -> kill + restart
#   * > MAX_RESTARTS restarts                   -> give up, write ALERT, stop
# Per-dispatch wall is watchdog-bounded inside the cascade (<=180s), so a 30-min
# progress gap is a genuine wedge, not a slow dispatch.
set -uo pipefail

OUT="${1:-$HOME/data/raphi_vct}"
LADDER="${2:-50,100,250,500,1000,2000,4000,10000,20000,50000,100000}"
POLL="${POLL:-300}"
STALL_MIN="${STALL_MIN:-30}"
MAX_RESTARTS="${MAX_RESTARTS:-6}"
export GOMOKU_BOARD_SIZE=15

WLOG="$OUT/watchdog.log"
CLOG="$OUT/cascade.log"
cd "$(dirname "$0")/../.." || exit 1   # worktree root (uv env)

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$WLOG"; }
n_results() { find "$OUT/results" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' '; }
finished() { grep -q '\[cascade\] complete' "$CLOG" 2>/dev/null; }

start() {
  uv run python -m scripts.vct_cascade.cascade --out "$OUT" --ladder "$LADDER" >> "$CLOG" 2>&1 &
  echo $!
}

restarts=0
PID=$(start)
say "watchdog up · cascade pid=$PID · ladder=$LADDER · poll=${POLL}s · stall=${STALL_MIN}m"
last_count=$(n_results); stall_polls=0
stall_limit=$(( STALL_MIN * 60 / POLL ))

while true; do
  sleep "$POLL"
  cnt=$(n_results)
  if kill -0 "$PID" 2>/dev/null; then
    if [ "$cnt" -gt "$last_count" ]; then
      say "alive pid=$PID · result shards=$cnt (+$((cnt - last_count)))"
      last_count=$cnt; stall_polls=0
    else
      stall_polls=$((stall_polls + 1))
      say "alive pid=$PID · NO new shards ($cnt) · stall ${stall_polls}/${stall_limit}"
      if [ "$stall_polls" -ge "$stall_limit" ]; then
        say "WEDGE: no progress ${STALL_MIN}m — killing pid=$PID and restarting"
        kill -9 "$PID" 2>/dev/null; sleep 3
        restarts=$((restarts + 1)); stall_polls=0
        [ "$restarts" -gt "$MAX_RESTARTS" ] && { say "ALERT: >$MAX_RESTARTS restarts — giving up"; break; }
        PID=$(start); say "restarted cascade pid=$PID (restart $restarts)"
      fi
    fi
  else
    if finished; then
      say "cascade COMPLETED cleanly · result shards=$cnt — watchdog stopping"
      break
    fi
    restarts=$((restarts + 1))
    say "cascade DIED (no completion marker) · result shards=$cnt"
    [ "$restarts" -gt "$MAX_RESTARTS" ] && { say "ALERT: >$MAX_RESTARTS restarts — giving up"; break; }
    PID=$(start); say "restarted cascade pid=$PID (restart $restarts)"; last_count=$cnt; stall_polls=0
  fi
done
say "watchdog exiting"
