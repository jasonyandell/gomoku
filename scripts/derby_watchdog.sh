#!/usr/bin/env bash
# Generic Δelo Derby watchdog/narrator. Keeps a crash-resumable race alive and
# logs a per-lane snapshot every INTERVAL seconds. Pure shell (no Claude calls).
#   - restarts `delo_derby.py --board <BOARD> --resume` if it died and the race
#     is NOT complete (any lane still under cap),
#   - appends a one-line snapshot (elo/peak/wall/status per lane) to narration.log,
#   - on all-lanes-capped: bumps every-lane cap +12h and restarts (keep-going).
# Usage: nohup bash scripts/derby_watchdog.sh scripts/derby_v5_board.json >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
BOARD="${1:?usage: derby_watchdog.sh <board.json>}"
PY="$(command -v python)"
INTERVAL="${WATCHDOG_INTERVAL:-300}"
BASE="$("$PY" -c "import json,sys; print(json.load(open('$BOARD'))['global']['base_out_dir'])")"
STATE="$BASE/derby_state.json"
NARR="$BASE/narration.log"
DLOG="${CLAUDE_JOB_DIR:-/tmp}/$(basename "$BOARD" .json).log"

is_alive() { pgrep -f "delo_derby.py --board $BOARD" >/dev/null 2>&1; }

race_complete() {
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
  "$PY" - "$STATE" "$NARR" <<'PYEOF'
import json,sys,datetime
s=json.load(open(sys.argv[1])); narr=sys.argv[2]
ts=datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%SZ")
parts=[]
for n,i in s.get("ideas",{}).items():
    pk=i.get("peak_elo"); pk="%4d"%pk if pk is not None else "  - "
    parts.append(f"{n[:14]}:pk{pk} {int(i.get('wall_secs_total',0))}s {i.get('status','?')[:3]}")
line=f"{ts} r0done={s.get('round0_done')} chunks={s.get('total_chunks_run')} | " + " | ".join(parts)
open(narr,"a").write(line+"\n")
PYEOF
}

bump_cap() {
  "$PY" - "$BOARD" <<'PYEOF'
import json,sys
p=sys.argv[1]; b=json.load(open(p))
b["global"]["cap_wall_secs"]=float(b["global"]["cap_wall_secs"])+43200.0
json.dump(b,open(p,"w"),indent=2)
PYEOF
}

# Startup grace: don't race a just-launched derby (else the first is_alive check
# misses it and we spawn a DUPLICATE orchestrator). Wait before the first check.
sleep "${WATCHDOG_STARTUP_GRACE:-20}"

while true; do
  if race_complete; then
    bump_cap
    echo "$(date -u +%H:%M:%SZ) all lanes capped — bumped cap +12h, will restart (keep-going)" >> "$NARR"
  fi
  if ! is_alive; then
    echo "$(date -u +%H:%M:%SZ) derby DOWN — restarting (--resume)" >> "$NARR"
    nohup "$PY" scripts/delo_derby.py --board "$BOARD" --resume >> "$DLOG" 2>&1 &
    sleep 8
  fi
  snapshot
  sleep "$INTERVAL"
done
