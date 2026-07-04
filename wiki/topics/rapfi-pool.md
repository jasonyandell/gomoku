# The Rapfi engine pool — the strength yardstick + teacher, warm

> **Status: LIVE infrastructure** *(2026-07-04)* — provisioning + warm pool.

**Rapfi** is the Gomocup-winning NNUE gomoku engine (Gomocup 2024 **and** 2025
first place; freestyle Elo ~2625). In this project it plays two roles: the
**strength yardstick** (the first honest reference past the ~1700 pure-baseline
ladder ceiling) and a **teacher** (a master whose moves we distill). Both roles
run through one substrate — a warm pool of persistent Rapfi processes,
`gomoku/rapfi_pool.py`. This page is the single doorway for *what Rapfi is, how the
binary gets onto a machine, and how every consumer reaches it*.

Related: [external-engine-baselines.md](external-engine-baselines.md) (why Rapfi,
the native-anchor bring-up, coord/config faithfulness audit),
[reliable-eval-set.md](reliable-eval-set.md) (Rapfi as the sole native anchor in
the no-wine reliable set), [eval-teacher-sensei.md](eval-teacher-sensei.md) (the
always-on eval daemon + distillation teacher built on the pool),
[eval-suite.md](eval-suite.md) (the broader eval surface),
[rapfi-idx2-distillation-mine.md](rapfi-idx2-distillation-mine.md) (the idx-2
mining harness that outgrew this in-process pool for multi-engine throughput).

## Provisioning — a pinned, sha256-verified HF binary

The Rapfi binary + NNUE weights + `config.toml` are ~40 MB of **gitignored** local
build artifacts, so a fresh worktree / machine / CI does not have them.
`rapfi_pool.py` resolves them with `rapfi_artifacts()` in strict precedence
(`gomoku/rapfi_pool.py:133`):

1. **Local build (fast path):** `engines/rapfi/` iff it holds a runnable
   `pbrain-rapfi` + `config.toml` (`_local_rapfi_dir`, `rapfi_pool.py:98`). A box
   that already built the engine **never touches the network** — byte-identical to
   the pre-HF behaviour.
2. **Pinned HF snapshot:** repo **`jasonyandell/rapfi-arm64`** (public GPL mirror
   of `dhbloo/rapfi @ 6e0a132`, arm64 macOS Mach-O), pinned to an **immutable
   commit SHA** (`RAPFI_HF_REVISION = "697aec1a50ab8b8b5b280749516fb37ea95435c5"`,
   `rapfi_pool.py:74`) — NOT a mutable tag, for reproducibility. `snapshot_download`
   pulls it into the **machine-global** `~/.cache/huggingface` (the one store that
   is both worktree- and venv-invariant — one fetch per machine, shared across every
   worktree/venv). The blob is `chmod +x`'d (hub cache stores 0644) and its
   **sha256 is asserted** against the pin
   (`RAPFI_BINARY_SHA256 = "9f8efade…"`, `rapfi_pool.py:75`) — we refuse to run a
   binary that isn't exactly the build we pinned, since we're chmod-executing a file
   pulled from the network.

Both pins are filled by **`scripts/publish_rapfi.py`** — the only human-gated step,
run **once per Rapfi build**. It uploads the 7 artifacts (binary, `config.toml`,
`model210901.bin`, the four `mix9svq*.bin.lz4` weights) to the HF repo in one atomic
commit, writes a GPL model card citing the upstream source commit, and **prints the
two pin constants** to paste back into `rapfi_pool.py`. (`--private` is available but
public is the default so a cold machine can pull token-free.)

Two resolution predicates gate callers:
- `rapfi_available()` — cache-only, **never downloads** (local present or snapshot
  already cached). Used to gate tests so they never fetch as a side effect.
- `rapfi_obtainable()` — may fetch on first use; also requires arm64-macOS
  (`_is_arm64_mac`) since the binary is a single-arch Mach-O. Use this to decide
  whether to *offer* Rapfi.

The launch command is `pbrain-rapfi --config <config.toml> gomocup` — the
engine-agnostic **Gomocup/Piskvork protocol**, driven unchanged by
`gomoku/external_engine.py` (`START`/`INFO timeout_turn`/`RESTART`+`BOARD`). Nothing
about the pool is Rapfi-specific: any Gomocup-protocol engine works via
`external:cmd=...`.

## The warm-pool pattern — why, and why it's safe

`RapfiPool` (`rapfi_pool.py:224`) pre-spawns `size` `ExternalEnginePlayer`
processes **once** and lends them out through a thread-safe lease queue. The
motivation: a 15×15 NNUE engine pays a real start-up + weight-load tax on **every**
spawn. The old babysit scripts respawned Rapfi for every eval pass; on a *cadence*
(eval every checkpoint) or when *teaching* (labelling thousands of positions) that
respawn tax dominates. Warm processes = the **10×+ speedup** that makes always-on
eval and Rapfi-as-teacher practical.

Why reuse is safe: classical Rapfi runs in **BOARD mode** (`incremental=False`,
forced by the pool at `rapfi_pool.py:266`). Every move does a full `RESTART`+`BOARD`
re-dump of the position, so an engine carries **no state** between calls or games —
one instance can label arbitrary unrelated positions back-to-back. The single
invariant is *one borrower per instance at a time* (each engine owns one
stdin/stdout pipe; two threads must not interleave on it), which the lease queue
enforces (`lease()`, `rapfi_pool.py:319`).

- **True parallelism:** `size` engines == `size` OS processes computing at once
  (Rapfi is CPU NNUE; the GIL is released during the blocking pipe read).
- **Self-healing:** a crashed/timed-out engine raises `ExternalEngineError`; the
  lease retires the corpse, spawns a replacement to keep the pool at `size`, and
  retries once — a caller never sees a transient engine death.
- **Work surface:** `pick(state)` (one move), `label_states(states)` (many moves,
  fanned across the pool for the teacher), `analyze_state`/`analyze_states` (the
  whole-board `{action: winrate}` map — the *soft*-teacher signal). Context-manager
  lifecycle tears every engine down.

## CPU-only co-tenancy

Rapfi is **CPU NNUE**, so the pool **never competes with the MPS trainer** for the
GPU — the property that makes an always-on eval cadence and Rapfi-as-teacher safe to
run **during** a live training run. The eval daemon defaults `GOMOKU_DEVICE=cpu` to
keep this guarantee (see [eval-teacher-sensei.md](eval-teacher-sensei.md)).

## How each consumer reaches the pool

- **Arena** (`gomoku/arena.py`) resolves opponent specs through the pool:
  `rapfi@50ms` is sugar for `rapfi:timeout_ms=50` (`_RAPFI_SUGAR`,
  `arena.py:314`); the full form `rapfi[:timeout_ms=N,size=K,cmd=PATH]`
  (`arena.py:363`) builds a `RapfiPool` wrapped as an `EnginePoolAgent` that fans
  each move across the warm pool (default `engine-pool-size=8`). **`external:cmd=…`
  gets the same warm pool** — `RapfiPool` is engine-agnostic (`arena.py:428`).
- **Eval** — the `eval_panel` / `eval_daemon` (#34 series) register a `rapfi` ruler
  in the panel of fixed reference opponents, using the pool so the anchor stays warm
  across every checkpoint on the cadence. Rapfi is **required by default**
  (fail-fast `SystemExit(2)`, no silent baseline-only fallback).
- **Teacher** — `gomoku/teacher.py`'s `label_states_with_pool` labels self-play
  positions with Rapfi's move for **policy-side distillation** (`--teacher-weight`
  in `gomoku/train.py`). Rapfi exposes only a move (no policy/value), so the teacher
  is one-hot policy distillation; the soft variant distills `analyze_state`'s
  per-move winrate. Both distillation attempts have so far **regressed** — read the
  warnings at the top of [eval-teacher-sensei.md](eval-teacher-sensei.md) before
  running a teacher.

## The strength dial: think-time, NOT node budget

The single strength knob for Rapfi-as-yardstick is **think-time** (`timeout_ms`),
not a node/depth budget. Once the mix9svq NNUE `config.toml` is in place, Rapfi
searches to its **full time budget** — a 5 s move reached Depth 32 / ~2.4 M nodes
(`advanced_stop_ratio = 0.9`), single-threaded (Gomocup-legal). This resolved the
#28 "under-search / broken yardstick" wound, which had turned out to be the
*weightless classical* build ignoring time, not Rapfi (#40 delivered the native
anchor; details in [external-engine-baselines.md](external-engine-baselines.md)).
So longer `timeout_ms` = a stronger opponent, and time-control tiers are real and
comparable. Absolute Elo calibration (effective single-thread strength under our
harness + balanced/swap2 openings) is still pending (#35/#22).

## Known caveat

The **first** fetch on a cold machine needs the network once (then it's cached
forever). The binary is **single-arch arm64-macOS** — `rapfi_obtainable()` returns
False off arm64-Mac. And interactively-authed / headless runs that lack the HF cache
*and* the network (or lack `huggingface_hub`) will get a clean `RapfiUnavailable`
rather than a silent skip; consumers that require Rapfi refuse to start with an
actionable message (build `engines/rapfi/build_rapfi.sh`, or ensure network for the
one-time HF fetch).

Evidence: `gomoku/rapfi_pool.py`, `gomoku/arena.py`, `scripts/publish_rapfi.py`,
HF repo `jasonyandell/rapfi-arm64` (@ `697aec1`), issues #40 / #28 / #34 / #35 / #22.
