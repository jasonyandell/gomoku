#!/usr/bin/env bash
# Keep the sliding derby runner alive (token-free, low-latency). Mirrors the
# repo's derby_watchdog.sh pattern. nohup-detached; 60s checks; relaunches the
# runner if it died and no STOP file is present. Stop both: touch /tmp/STOP_SLIDING_DERBY
# Detach:  nohup bash scripts/sliding_derby_watchdog.sh >/dev/null 2>&1 &
set -u
REPO=/Users/jason/code/gomoku
STOP=/tmp/STOP_SLIDING_DERBY
NARR="$REPO/sweep_logs/G15-defense-board15/watchdog.log"
cd "$REPO" || exit 1
mkdir -p "$(dirname "$NARR")"
echo "[watchdog] START $(date)" >> "$NARR"
sleep 30   # startup grace — don't race a just-launched runner
while [ ! -f "$STOP" ]; do
  if ! pgrep -f 'sliding_derby_runner.sh' >/dev/null 2>&1; then
    echo "[watchdog] runner DOWN — relaunching $(date)" >> "$NARR"
    nohup bash "$REPO/scripts/sliding_derby_runner.sh" >/dev/null 2>&1 &
    sleep 12
  fi
  sleep 60
done
echo "[watchdog] STOP file — exiting $(date)" >> "$NARR"
