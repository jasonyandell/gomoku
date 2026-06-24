# Third-party components & attribution

This project's own code is **MIT-licensed** (see `LICENSE`). It gratefully builds
on external work, credited below. Nothing here changes this project's MIT license;
each item carries its own license.

## Rapfi — the gomoku/renju engine (GPL-3.0)

- **Upstream:** https://github.com/dhbloo/rapfi — author Haobin Duan (dhbloo) and
  contributors. Pinned commit `6e0a1329e725d854d316cbe3e1fd436d7f86926e`.
- **How it's used:** as an **external engine** — this project spawns `pbrain-rapfi`
  as a *separate process* and communicates over the Gomocup stdin/stdout protocol
  (`gomoku/external_engine.py`, `gomoku/rapfi_pool.py`). It is **not** linked into,
  statically bound to, or derived from this project's source. This arm's-length
  aggregation means Rapfi's GPL-3.0 does **not** extend to this project's MIT code
  (per the FSF's stated position on communication-at-arm's-length via pipes/exec).
- **Redistribution:** a prebuilt arm64 macOS binary + NNUE weights are mirrored —
  under **GPL-3.0**, with corresponding source = the upstream commit above and the
  build recipe in `engines/rapfi/build_rapfi.sh` — at
  https://huggingface.co/jasonyandell/rapfi-arm64 so `rapfi_pool` can resolve the
  engine friction-free from any worktree/machine.

## Python dependencies

PyTorch, NumPy, FastAPI, uvicorn, huggingface_hub, pytest, and the rest are
installed via `pip`/`uv` and remain under their own licenses (predominantly
BSD / MIT / Apache-2.0). They are dependencies, not redistributed by this repo.

## Inspirations / techniques

AlphaZero (DeepMind), KataGo, and the `michaelnny/alpha_zero` reference informed
the architecture and training recipe; credited in the wiki where applied.
