#!/bin/bash
# walt-on-gomoku, phase 1. GPU-serial; every pairing hard-capped. 10 games = 5 openings x both colors.
set -u
cd "$(dirname "$0")/../.."
OUT=sweep_logs/walt/phase1.jsonl
run() { uv run python scripts/walt/run_walt.py --games 10 --out "$OUT" "$@" 2>&1 | grep -v Warning; }

# references: the champion's bare policy and the champion product (MCTS200 + finisher)
for opp in heuristic lookahead4 field; do run --player netargmax --opp $opp --max-wall-secs 300; done
for opp in heuristic lookahead4 field; do run --player mcts200   --opp $opp --max-wall-secs 600; done
# walt, level-0 = net@T1 + finisher (the best-effort field)
for opp in heuristic lookahead4 field mcts200; do run --player walt-net --opp $opp --max-wall-secs 600; done
# walt, level-0 = uniform-near random (texas-42's literal bottom rung)
for opp in heuristic lookahead4; do run --player walt-uniform --opp $opp --max-wall-secs 600; done
echo PHASE1_DONE
