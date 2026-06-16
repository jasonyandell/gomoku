# The reliable eval set (wine engines shelved)

**Directive (Jason, 2026-06-16):** *"bail on wine evals — another wine crash.
eval just with reliable things. have workflows attempting to bring new reliable
things online."* This page is the canonical definition of what counts as a
**reliable** evaluator after that directive. Siblings:
[gomocup-engines-catalog.md](gomocup-engines-catalog.md) (what's open/buildable),
[engine-panel-derby-design.md](engine-panel-derby-design.md) (the #30 panel).

## Why wine engines are shelved (issue #35)

The five external Gomocup anchors (embryo26, yixin18, pela23, zetor17, eulring16)
are Windows `pbrain-*.exe` binaries run **under wine**. In the #30/#35 panel
calibration attempt they were **flaky**: ~**17/36 panel pairs crashed** (wine
segfaults / protocol desyncs), which broke the affine anchor fit and made the
"calibrated Gomocup Elo" untrustworthy. They are now **OPT-IN ONLY** — never in
the default eval path.

They are **kept, not deleted** — the catalog is evidence, and an engine that
proves reliable can be re-listed. Re-enable explicitly with
`GOMOKU_ENABLE_WINE_ENGINES=1` or the `--wine` flag on
`scripts/panel_tournament.py`. Both default OFF.

## The reliable eval set (the default)

Two pillars, both **pure** (no wine, no flaky subprocess):

1. **Net-vs-net head-to-head — PURE TORCH.** Our nets play each other / a frozen
   reference directly through `gomoku.eval` / `gomoku.match`. This is what the
   **sliding derby already does reliably** (`scripts/delta_e_harness.py`
   `head_to_head_eval`). It is non-transitive across recipes (use a fixed
   anchor), but it never crashes. Our checkpoints `sweep_runs/g15_*.pt`
   (e.g. `g15_128x10_bigbuf_eval502`, `g15_96x8_bigbuf_eval597`,
   `g15_128x10_bigbuf_e588_best`) form a stable **internal strength ladder**.

2. **Pure-python fixed baselines** (`gomoku/baselines.py`): `heuristic` and
   `lookahead:N`. No subprocess, no timeout, no wine. The absolute floor and the
   anchored-Elo ladder (saturates ~1700 — that ceiling is the reason we still
   want one strong external anchor, see below). These are the right strength
   probes per the project's ML-judgment rule (fixed baselines, not sibling H2H).

The default `panel_tournament.py` field is therefore: **our nets (via the
`run-gomoku-az` brain wrapper, pure torch) + the `heuristic` floor + any NATIVE
engine** in `_NATIVE_ENGINES` (currently empty). Zero wine.

## What's enforced in code

- `scripts/panel_tournament.py`
  - `_WINE_ENGINES` — the 5-engine wine catalog, **kept but gated**.
  - `_NATIVE_ENGINES` — prepared (empty) slot for reliable non-wine engines.
  - `wine_engines_enabled(cli_wine)` — OFF unless `--wine` or
    `GOMOKU_ENABLE_WINE_ENGINES` truthy.
  - `real_engines(enable_wine=...)` — native always, wine only on opt-in.
  - `_REAL_ENGINES` — back-compat alias, now the reliable (native-only) default.
- `tests/test_panel_wine_optin.py` — pins the contract: the default field
  carries **zero** wine labels/wrappers; opting in restores all five.

## Bringing a NATIVE strength anchor online (issue #40, refs #28)

The anchor-ladder saturates ~1700; one strong external engine is still wanted —
but **natively, no wine**. Prime candidate **Rapfi** (Gomocup #1, GPL-3, C++17):

- **Already proven buildable here.** `engines/rapfi/build_rapfi.sh` records the
  recipe (last `rapfi@6e0a132`, 2026-05-24, preset `arm64-clang-NEON-DOTPROD`).
  Apple clang + CMake; no wine. Source-build only (no macOS prebuilt asset).
- **Speaks Piskvork/Gomocup by default** — `gomoku/external_engine.py` drives it
  unchanged (START/BEGIN/TURN/BOARD/INFO/END).
- **Bring-up gap:** there is no `run-rapfi` wrapper in `~/.cache/gomocup/bin/`
  yet (only `run-gomoku-az` + the wine `run-*` wrappers). Plan: build → add a
  `run-rapfi` wrapper → register `("rapfi", …)` in `_NATIVE_ENGINES`.
- **Real blocker is #28, not the build:** the stock build uses Rapfi's internal
  *classical* config (weightless, weak) and was found to **ignore search time**
  (TC tiers illusory). For a trustworthy anchor: place the `mix9svq` freestyle
  NNUE weight + `config.toml` next to the binary, pin **one thread** (Gomocup
  single-core rule), and use a **node/depth cap** if time-control is ignored.

Full plan: **GitHub issue #40**.
