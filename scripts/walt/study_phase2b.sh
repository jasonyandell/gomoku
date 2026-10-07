#!/bin/bash
# walt-on-gomoku phase 2b: spend the native batch (~16K items/ply) on samples vs depth vs width,
# against the discriminating opponents, with MCTS at equal wall time (~4 s/move) as the reference.
set -u
cd "$(dirname "$0")/../.."
D=sweep_logs/walt/phase2b; mkdir -p $D
export WALT_IMPL=2
run() { local tag=$1; shift; uv run python scripts/walt/run_walt.py --games 10 --max-wall-secs 600 --out $D/$tag.jsonl "$@" 2>&1 | grep -v Warning; }

for opp in lookahead4 field; do
  run samples_K2048_H2_M8 --player walt-net --opp $opp --tapes 2048 --horizon 2 --cand 8 --rollout-cap 12
  run depth_K256_H4_M8    --player walt-net --opp $opp --tapes 256  --horizon 4 --cand 8 --rollout-cap 12
  run width_K1024_H2_M16  --player walt-net --opp $opp --tapes 1024 --horizon 2 --cand 16 --rollout-cap 12
  run mcts1600            --player mcts1600 --opp $opp
done
run depth_vs_mcts1600 --player walt-net --opp mcts1600 --tapes 256 --horizon 4 --cand 8 --rollout-cap 12
echo PHASE2B_DONE
