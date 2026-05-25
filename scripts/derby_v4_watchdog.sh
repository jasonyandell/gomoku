#!/usr/bin/env bash
# Derby v4 watchdog/narrator: keeps the crash-resumable race alive and logs a
# per-lane snapshot every INTERVAL seconds. Pure shell (no Claude invocations).
#   - restarts `delo_derby.py --board derby_v4_board.json --resume` if it died
#     and the race is NOT yet complete (any lane still under cap),
#   - appends a one-line snapshot (elo/peak/wall/status per lane) to narration.log,
#   - exits cleanly once every lane has capped (race complete).
# Usage: nohup bash scripts/derby_v4_watchdog.sh >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
REPO="$(pwd)"
BOARD="scripts/derby_v4_board.json"
STATE="sweep_runs/derby_v4/derby_state.json"
NARR="sweep_runs/derby_v4/narration.log"
DLOG="${CLAUDE_JOB_DIR:-/tmp}/derby_v4.log"
PY="$(command -v python)"
INTERVAL="${WATCHDOG_INTERVAL:-300}"

is_alive() { pgrep -f "delo_derby.py --board $BOARD" >/dev/null 2>&1; }

race_complete() {
  # complete iff state exists AND no lane has status running/queued
  [ -f "$STATE" ] || return 1
  "$PY" - "$STATE" <<'PYEOF'
import json,sys
s=json.load(open(sys.argv[1]))
act=[i for i in s.get("ideas",{}).values() if i.get("status") in ("running","queued")]
sys.exit(0 if not act else 1)
PYEOF
}

snapshot() {
  [ -f "$STATE" ] || { echo "$(date -u +%H:%M:%SZ) (no state yet)" >> "$NARR"; return; }
  "$PY" - "$STATE" >> "$NARR" <<'PYEOF'
import json,sys,datetime
s=json.load(open("sweep_runs/derby_v4/derby_state.json"))
ts=datetime.datetime.utcnow().strftime("%H:%M:%SZ")
parts=[]
for n,i in s.get("ideas",{}).items():
    pk=i.get("peak_elo"); pk="%4d"%pk if pk is not None else "  - "
    parts.append(f"{n[:10]}:e{(i.get('elo_history') or [None])[-1]} pk{pk} {int(i.get('wall_secs_total',0))}s {i.get('status','?')[:3]}")
print(ts, "r0done="+str(s.get("round0_done")), "chunks="+str(s.get("total_chunks_run")), "| "+" | ".join(parts))
PYEOF
}

while true; do
  if race_complete; then
    echo "$(date -u +%H:%M:%SZ) RACE COMPLETE — watchdog exiting" >> "$NARR"
    exit 0
  fi
  if ! is_alive; then
    echo "$(date -u +%H:%M:%SZ) derby DOWN — restarting (--resume)" >> "$NARR"
    nohup "$PY" scripts/delo_derby.py --board "$BOARD" --resume >> "$DLOG" 2>&1 &
    sleep 8
  fi
  snapshot
  sleep "$INTERVAL"
done
