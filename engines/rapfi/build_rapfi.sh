#!/usr/bin/env bash
# Build Rapfi (Gomocup external-engine yardstick) for Apple silicon (ARM64 NEON).
#
# Produces engines/rapfi/pbrain-rapfi — a self-contained Piskvork-protocol
# engine that accepts START 9 (9x9 freestyle). Uses Rapfi's internal classical
# config (no external NNUE weights required to run).
#
# Requires: cmake (>=3.23), ninja, clang/clang++ (Apple toolchain is fine).
#   brew install cmake ninja
#
# Usage:  bash engines/rapfi/build_rapfi.sh
#
# For the stronger NNUE evaluator, init the Networks submodule and pass
# --config pointing at Networks/config-example/gomocalc-mix9svq.toml plus the
# mix9svq weight files. The internal config is enough for a yardstick eval.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/.rapfi-build"
PRESET="arm64-clang-NEON-DOTPROD"   # falls back to arm64-clang-NEON if no FEAT_DotProd

export PATH="/opt/homebrew/bin:$PATH"

mkdir -p "$BUILD_DIR"
if [ ! -d "$BUILD_DIR/rapfi/.git" ]; then
  git clone --depth 1 https://github.com/dhbloo/rapfi.git "$BUILD_DIR/rapfi"
fi
COMMIT="$(git -C "$BUILD_DIR/rapfi" rev-parse HEAD)"

cd "$BUILD_DIR/rapfi/Rapfi"
cmake --preset "$PRESET"
cmake --build "build/$PRESET"

mkdir -p "$REPO_ROOT/engines/rapfi"
cp "build/$PRESET/pbrain-rapfi" "$REPO_ROOT/engines/rapfi/pbrain-rapfi"
{
  echo "$COMMIT"
  echo "engine: Rapfi (pbrain-rapfi)"
  echo "source: https://github.com/dhbloo/rapfi"
  echo "preset: $PRESET (internal classical config; no external weights)"
  echo "built: $(date -u +%Y-%m-%d)"
} > "$REPO_ROOT/engines/rapfi/BUILD_COMMIT.txt"

echo "Built engines/rapfi/pbrain-rapfi from rapfi@$COMMIT"
printf 'START 9\nINFO rule 0\nINFO timeout_turn 200\nBEGIN\nEND\n' | "$REPO_ROOT/engines/rapfi/pbrain-rapfi"
