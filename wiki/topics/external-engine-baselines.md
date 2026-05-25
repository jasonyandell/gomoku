# External Engine Baselines

This page tracks rated, runnable OSS/source-available Gomoku engines that could
be wrapped as fixed external baselines. It exists because the in-repo anchor
ladder is useful but self-contained: random, heuristic, and lookahead tell us
when a checkpoint is improving inside our 9x9 world, while a rated external
engine can tell us when the model is starting to survive contact with the
broader Gomoku engine ecosystem.

Source trail: [../sources/gomocup-external-engines-2026-05-22.md](../sources/gomocup-external-engines-2026-05-22.md).

## Status: Rapfi BUILT + RUNS + accepts START 9 (2026-05-24)

PRACTICALITY VERDICT: **YES**. Rapfi builds and runs as a 9x9 freestyle
yardstick on the M5 Max, and the eval wrapper is implemented and smoke-tested.

- Build: shallow-cloned `dhbloo/rapfi@6e0a132`, `brew install cmake ninja`,
  configured with the `arm64-clang-NEON-DOTPROD` CMake preset (M5 Max has
  `hw.optional.arm.FEAT_DotProd=1`), built with ninja in ~seconds. Binary is
  `pbrain-rapfi` (Mach-O arm64, ~1.9MB).
- 9x9 fit: source `gomocup.cpp` accepts any `5 <= size <= MAX_BOARD_SIZE`
  (MAX_BOARD_SIZE = 22). `START 9` → `OK`; empty-board `BEGIN` → `4,4` (center,
  legal). `START 4` → `ERROR Unsupported board size!` (confirms the smoke is
  meaningful). So 9x9 is supported at the source level, not just "recommended".
- Weights: Rapfi has a built-in **internal classical config** (`internalConfig.cpp`,
  `model_type` 1/2) — it runs with **zero external weight files**. The stronger
  NNUE path needs the `Networks` submodule (mix9svqfreestyle_bsmix.bin.lz4, ~10MB)
  + `--config Networks/config-example/gomocalc-mix9svq.toml`; that is a future
  strength lever, not required for a working yardstick.
- Wrapper: `external:cmd=...,timeout_ms=...,label=...` player spec in
  `gomoku.match`, implemented in `gomoku/external_engine.py`. Encodes
  `state.board[0]` (side-to-move = the engine) as field 1, `board[1]` as field 2;
  protocol coords are X=col, Y=row. Skips `MESSAGE`/`DEBUG` chatter; validates
  the reply is empty + in-range; raises on illegal/out-of-range/EOF.
- Eval path: `scripts/eval_vs_rapfi.py` runs a checkpoint vs Rapfi at several
  timeout tiers (difficulty tiers), color-alternated, writing JSONL with explicit
  provenance (engine, build ref, timeout, board size, rule, wrapper version).
- Reproducible build: `engines/rapfi/build_rapfi.sh` (binary + the 88MB source
  clone are gitignored; `BUILD_COMMIT.txt` + the build script are tracked).
- Smoke result: WL5 seed model (sims=100) vs rapfi(timeout=100ms), 4 games,
  color-alternated → 2W-0L-2D (75%). Same at timeout=1000ms in a 4-game sample.
  Note: at low time controls with the internal classical config Rapfi is weak;
  pushing it stronger likely needs the NNUE weights (follow-up).

EVAL-ONLY. Do not mix Rapfi into self-play training.

## Current Recommendation

Start with **Rapfi** as the first external baseline.

Why:
- It is the cleanest combination of high external rating, strong current
  maintenance, source availability, explicit GPL-3.0 licensing, Piskvork
  protocol support, and vectorized build paths.
- Gomocup's best-version freestyle table lists Rapfi 0.34.05 at Elo 2625.
  Gomocup 2025 downloads list Rapfi 2025 as the first-place engine.
- The Rapfi README says the engine benefits from x86 vector instructions and
  documents an ARM64 NEON build path, which matters on the M5 Max.
- It is strong enough that low time controls can probably create several useful
  local difficulty tiers without changing engines.

Treat the Gomocup Elo as provenance, not a direct 9x9 label. The tournament
ratings come from Gomocup rules, boards, openings, and time controls. Our local
question is narrower: "what does checkpoint X score against Rapfi at local time
control Y on 9x9 freestyle?"

## Candidate Tiers

| Candidate | External signal | OSS/source status | Fit for this repo |
|---|---:|---|---|
| Rapfi | Freestyle Elo 2625; Gomocup 2025 first place | GPL-3.0, C++ source, Piskvork, ARM64 NEON | First target. Smoke `START 9`, then build time-control tiers. |
| AlphaGomoku(MK) | Freestyle Elo 2256; Gomocup 2025 second place | GPL-3.0, C++ source, OpenCL-capable releases | Second target. Strong neural/MCTS style, heavier build/runtime surface. |
| KataGomo | Freestyle Elo 2254 for 2021 release | KataGo-derived public source | Research target. Useful conceptual contrast; may be less plug-and-play. |
| SlowRenju | Freestyle Elo 1857 | GPL-3.0, C++ source, protocol, board sizes 5-20 | Good medium-strength anchor if Rapfi is too strong or too slow. |
| Pela/Piskvork | Freestyle Elo 1499 | GPL C++ source via Piskvork surface | Good weaker historical anchor; probably easier than modern engines. |
| Chis | Freestyle Elo 1448 | MPL-2.0 C++ source | Possible mid-low source-licensed anchor. |
| Carbon | Freestyle Elo 1670 | Source available; Gomocup calls it open source | Check license/build before relying on it as strict OSS. |
| PentaZen | Freestyle Elo 2143 | Source available; license unclear in this pass | Interesting strength, but hold for strict OSS until license is verified. |
| JAX | Gomocup 2025 third place; CPU/CUDA/TensorRT note | No source/license trail found | Do not use for this OSS harness yet. |

## Wrapper Shape

The common integration surface should be a Piskvork/Gomocup protocol player.
That should become a new player spec in `gomoku.match`, something like:

```text
external:cmd=/path/to/pbrain-rapfi,timeout_ms=50,label=rapfi50
```

Implementation sketch:
- Spawn the engine as a subprocess with pipes.
- Send `START 9`; fail fast if the engine rejects the board size.
- Send `INFO rule 0` for freestyle and `INFO timeout_turn <ms>` to create
  local difficulty tiers.
- For each pick, send `BOARD`, every occupied coordinate, and `DONE`, then
  parse the engine's `X,Y` reply.
- Because `GameState` is canonical, `state.board[0]` is the player to move.
  When the external picker is called, that player is the engine, so encode
  `board[0]` as field `1` and `board[1]` as field `2`.
- Validate the returned coordinate is empty and in range before applying it.
- Keep the eval worker JSONL fields explicit: engine name, source URL,
  downloaded/build commit or release, timeout, board size, rule, and wrapper
  version.

The first implementation can be eval-only. Do not mix external engines into
self-play training until the eval wrapper has stable smoke tests, reproducible
results, and clear time-control labels.

## First Smoke Plan

1. Build Rapfi locally for ARM64 NEON or download a compatible release if one
   exists for this machine.
2. Run a protocol smoke:
   - `START 9`
   - `INFO rule 0`
   - `INFO timeout_turn 50`
   - empty-board `BEGIN` or `BOARD`/`DONE`
   - verify the reply is a legal 9x9 coordinate.
3. Add `external` player support and a tiny fake-protocol engine test.
4. Run `random/heuristic/lookahead2/model` against `rapfi:timeout=10,50,100`
   as local calibration, with color alternation and enough games to avoid
   single-digit noise.
5. Only then consider adding AlphaGomoku(MK), SlowRenju, and Pela/Chis as
   additional external anchors.
