#!/bin/bash
# walt-on-gomoku phase 3: the external anchor. Rapfi-NNUE (single-thread) at 50 ms (project standard)
# and 1000 ms, vs the two best walt arms and MCTS refs. Read the WHITE column (freestyle favours black).
set -u
cd "$(dirname "$0")/../.."
# wait for phase 2b to release the GPU (serial queue)
while pgrep -f study_phase2b.sh >/dev/null; do sleep 20; done
D=sweep_logs/walt/phase3; mkdir -p $D
export WALT_IMPL=2
run() { local tag=$1; shift; uv run python scripts/walt/run_walt.py --games 10 --max-wall-secs 900 --out $D/$tag.jsonl "$@" 2>&1 | grep -v Warning; }

for opp in rapfi50 rapfi1000; do
  run samples_K2048_H2_M8 --player walt-net --opp $opp --tapes 2048 --horizon 2 --cand 8 --rollout-cap 12
  run depth_K256_H4_M8    --player walt-net --opp $opp --tapes 256  --horizon 4 --cand 8 --rollout-cap 12
  run mcts200             --player mcts200  --opp $opp
  run mcts1600            --player mcts1600 --opp $opp
done
echo PHASE3_DONE
