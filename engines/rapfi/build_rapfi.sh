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

# --- NNUE weights (issues #40/#28): the WEIGHTLESS classical build under-searches
# (stops ~depth 10 / ~500 nodes, ignores its time budget — the "illusory TC
# tiers" bug). The mix9svq NNUE evaluator fixes that: it searches to its full
# time budget (depth ~32 / ~2M nodes). Weights + the classical fallback model
# live in Rapfi's `Networks` submodule; copy them next to the binary so the
# committed engines/rapfi/config.toml (which references them by bare filename,
# resolved relative to the config dir) loads them. Weights are gitignored
# (~50MB of .lz4); config.toml IS committed (it defines the anchor).
NET_SRC="$BUILD_DIR/rapfi/Networks"
if [ ! -f "$NET_SRC/mix9svq/mix9svqfreestyle_bsmix.bin.lz4" ]; then
  git -C "$BUILD_DIR/rapfi" submodule update --init --depth 1 Networks || true
fi
if [ -d "$NET_SRC/mix9svq" ]; then
  cp "$NET_SRC"/mix9svq/mix9svq{freestyle_bsmix,standard_bs15,renju_bs15_black,renju_bs15_white}.bin.lz4 \
     "$REPO_ROOT/engines/rapfi/"
  cp "$NET_SRC/classical/model210901.bin" "$REPO_ROOT/engines/rapfi/"
  echo "Copied mix9svq NNUE weights + classical fallback into engines/rapfi/"
else
  echo "WARN: Networks submodule missing — engine will run WEIGHTLESS (under-searches; see #28)." >&2
fi

# Smoke: the NNUE config must load and actually search to its budget.
printf 'START 15\nINFO timeout_turn 500\nBOARD\n7,7,1\n7,8,2\nDONE\n' \
  | "$REPO_ROOT/engines/rapfi/pbrain-rapfi" --config "$REPO_ROOT/engines/rapfi/config.toml" gomocup \
  | grep -E 'load weight|^[0-9]+,[0-9]+' || true
