# Gomocup Engines Catalog — what's open source, what we can run

> **Status: LIVE reference** *(2026-07-04)* — snapshot 2026-06-15/16 (wine shelved; Rapfi the usable one).

Catalog of competitor engines from the **Gomocup** gomoku/renju AI tournament
([gomocup.org](https://gomocup.org/)), focused on the one question that matters
for us: **which engines have PUBLIC SOURCE CODE** so they could be studied,
built natively on Apple Silicon, or wrapped as honest fixed opponents.

Snapshot date: **2026-06-15** (wine-shelved update **2026-06-16**). Sibling pages:
[reliable-eval-set.md](reliable-eval-set.md) (**read this first** — the post-nix
eval definition) and [external-engine-baselines.md](external-engine-baselines.md)
(the Rapfi build + wrapper). Earlier source trail:
[../sources/gomocup-external-engines-2026-05-22.md](../sources/gomocup-external-engines-2026-05-22.md).

> **⚠️ WINE ENGINES ARE SHELVED (2026-06-16; issue #35).** The five wine-run
> Gomocup anchors (embryo26, yixin18, pela23, zetor17, eulring16) are **OPT-IN
> ONLY** — they crashed ~17/36 panel pairs (wine segfaults/desyncs) and broke
> the calibration. Jason: *"bail on wine evals — another wine crash. eval just
> with reliable things."* The **reliable default eval set** is now net-vs-net
> head-to-head (pure torch) + the pure-python `heuristic`/`lookahead` baselines —
> see [reliable-eval-set.md](reliable-eval-set.md). Re-enable wine explicitly
> with `GOMOKU_ENABLE_WINE_ENGINES=1` or `--wine` on `scripts/panel_tournament.py`
> (default OFF). The catalog below is **kept as evidence**; an engine that proves
> reliable (a **native** Rapfi, issue #40/#28) can be re-listed as an anchor.

> **We now speak the Gomocup/Piskvork protocol on BOTH sides** (grounded in the
> GomocupJudge protocol doc: `START`/`BEGIN`/`BOARD`/`TURN` over stdin/stdout).
> The **client** side is `gomoku/external_engine.py` (drives external `pbrain-*`
> engines into `gomoku.match`); the **brain** side is `gomoku/gomocup_brain.py`
> (#31) — our own net answers the protocol, so it is a first-class, path-registerable
> Gomocup engine (`scripts/run-gomoku-az`). One caveat for the brain side: our net
> is history-conditioned, so it must be driven with `incremental=1` (`TURN`-mode)
> or it sandbags itself on empty history — see
> [engine-panel-derby-design.md](engine-panel-derby-design.md) and
> [alphazero-lessons-15x15-gomoku.md](alphazero-lessons-15x15-gomoku.md) §13.

## The honest headline

**Most Gomocup engines are CLOSED source** — distributed only as precompiled
Windows `pbrain-*.exe` binaries on the
[download page](https://gomocup.org/download-gomoku-ai/). The competition is
Windows-native (since 2022 engines must be Win10/11 executables) and the strong
modern field is dominated by binary-only entries (Jax, Barbakan, Yixin, Embryo).

The **handful that ARE open** cluster at the top and bottom of the ladder:
- **Strong + relevant + buildable on ARM:** **Rapfi** (the reigning #1; the only
  competition engine with a first-class ARM64+NEON build path — already built &
  running here).
- **Strong + AlphaZero-lineage (our philosophy), but Apple-Silicon-unverified:**
  **KataGomo** (#2, KataGo fork, Apache-2.0, pretrained nets) and
  **AlphaGomoku(MK)** (#3, GPL-3 C++ AlphaZero).
- **Weaker historical / classical alpha-beta, mostly x86/Windows-VS:** SlowRenju,
  PentaZen, Chis, XL-engine, Carbon, Stahlfaust, the old MIT Rapfi-2018.

Everything genuinely usable today as an honest local opponent is in the
**"what's actually usable"** section below — it's a short list.

## Engine catalog

Source column: **OPEN** = real source in a public repo; **BINARY** = public
repo but precompiled executables only; **CLOSED** = no public source at all.
"ARM build?" assesses *from the repo* (build system, deps, SIMD) — **not** an
actual build attempt — for whether it could plausibly compile/run on M-series
macOS.

> **Rating-pool note:** the Elo figures in this table (Rapfi ~3073, KataGomo
> ~2879, AlphaGomoku ~2781, …) come from a **different rating pool/year** than our
> internal-anchor numbers in [external-engine-baselines.md](external-engine-baselines.md)
> and [rapfi-pool.md](rapfi-pool.md) (Rapfi ~2625, AlphaGomoku ~2256, KataGomo
> ~2254). They are not directly comparable — Rapfi's strength is **not**
> single-valued-canonical across pools; use the internal-anchor numbers for our
> calibration.

| Engine | Best Gomocup (rank/yr) | Source | Repo URL | License | Lang | ARM build? | Approach |
|---|---|---|---|---|---|---|---|
| **Rapfi** | **#1** 2022–2025 (Elo ~3073) | **OPEN** | [dhbloo/rapfi](https://github.com/dhbloo/rapfi) | GPL-3.0 | C++17 | **Yes** — CMake preset `arm64-clang-NEON-DOTPROD`, built & runs here | alpha-beta + **NNUE** |
| **KataGomo** | **#2** 2023–2026 (Elo ~2879) | **OPEN** | [hzyhhzy/KataGomo](https://github.com/hzyhhzy/KataGomo) | Apache-2.0 | C++ + Python | Unverified — KataGo CMake; GPU backends (CUDA/OpenCL/TensorRT/Eigen), macOS historically rough; **pretrained nets released** | **AlphaZero** (KataGo fork) |
| **AlphaGomoku(MK)** | **#2–3** 2023–2025 (Elo ~2781) | **OPEN** | [MaciejKozarzewski/AlphaGomoku](https://github.com/MaciejKozarzewski/AlphaGomoku) | GPL-3.0 | C++ (+Py) | Unverified — CMake; bundles own ml/SIMD layer, no ARM/macOS notes | **AlphaZero** (NN+MCTS) |
| **Jax** | **#3** 2024–2025 (Elo ~2662) | **CLOSED** | — (binary only) | — | — | n/a | NN; CPU/CUDA/TensorRT note on download page |
| **Embryo** (Embryo26) | top-5 / Caro #3 (Elo ~2402) | **BINARY** | [Hexik/Embryo_engine](https://github.com/Hexik/Embryo_engine) | none | — (Win/Linux x86 exe) | n/a (no source; x86 AVX2, Linux-x86 only) | Stockfish-derived alpha-beta search |
| **Barbakan** | top-3 2023–2024 (Elo ~2442) | **CLOSED** | — (binary only) | — | — | n/a | NN (KataGo-derived) |
| **Yixin** | top ~2014–2018 (Elo ~2310) | **CLOSED** | GUI only: [accreator/Yixin-Board](https://github.com/accreator/Yixin-Board) (BSD) | — | — | n/a | classical/NN, commercial |
| **PentaZen** | Elo ~2143–2171 | **OPEN** | [sun-yuliang/PentaZen](https://github.com/sun-yuliang/PentaZen) | not stated | C/C++ | Unlikely — Makefile via **mingw-w64 (Windows)**; NN added v0.5.0 | alpha-beta (+NN ≥0.5.0) |
| **SlowRenju** | Elo ~1857 | **OPEN** | [wind23/SlowRenju](https://github.com/wind23/SlowRenju) | GPL-3.0 | C++ | Plausible — plain C++ alpha-beta, no SIMD/ARM notes, boards 5–20 | iterative-deepening alpha-beta + TT (no NN) |
| **Carbon** | Elo ~1670 | **OPEN** | [gomoku/Carbon-Gomoku](https://github.com/gomoku/Carbon-Gomoku) | not stated on page | C++ | Unlikely — Visual Studio `.vcxproj` (Windows), 2002-era | classical minimax/alpha-beta + TT |
| **Pela** | Elo ~1499 | **OPEN** | [plastovicka/Piskvork](https://github.com/plastovicka/Piskvork) (bundled in manager) | GPL | C++ | Unlikely — Windows GUI manager; Pela source in `/source`, not cleanly separable | pattern / classical |
| **Chis** | Elo ~1448 | **OPEN** | [ChisBread/Chis](https://github.com/ChisBread/Chis) | MPL-2.0 | C++ | Unlikely — Visual Studio `.sln`; README: "年久失修" (unmaintained, kept as memorial) | minimax/alpha-beta, pattern |
| **XL-engine (Niren)** | Renju competitor | **OPEN** | [accreator/xl-engine](https://github.com/accreator/xl-engine) | GPL-2.0 | C/C++ | Unknown — no build docs | classical, **Renju-only** (no gomoku) |
| **Stahlfaust** | Elo ~715 | **OPEN** | [gomoku/Stahlfaust...](https://github.com/gomoku/Stahlfaust---Gomoku-AI-player) | not stated | C++ | Unknown | classical |
| **Rapfi-2018** | #4 Gomocup 2018 (Elo ~2096) | **OPEN** | [dhbloo/Rapfi-gomocup](https://github.com/dhbloo/Rapfi-gomocup) | MIT | C++ | Unlikely — Visual Studio `.sln`; **frozen** (superseded by new Rapfi) | alpha-beta + pattern eval (no NN) |
| **Wine** | Gomocup competitor | OPEN (off-GitHub) | source ZIP on gomocup developer page (no public GitHub found) | not stated | — | Unknown | classical |
| Zetor, Eulring, Noesis, Sparkle, Valkyrie, Mushroom, PureRocky | tournament entries | **CLOSED** | none found | — | — | n/a | binary-only `pbrain-*.exe` |

Notes on the "Tier-1 strong-open" trio: of the three, **only Rapfi is proven to
build and run on this M5 Max** (see external-engine-baselines.md). KataGomo and
AlphaGomoku(MK) are the same *family* as our own project (AlphaZero), making them
valuable to **read**, but neither documents Apple-Silicon support and both lean on
GPU/ML backends whose ARM-macOS story is unverified — treat "build on ARM" as a
research task, not a given.

### Companion / tooling repos (not engines)
- **[dhbloo/pytorch-nnue-trainer](https://github.com/dhbloo/pytorch-nnue-trainer)** —
  PyTorch training code for Rapfi's NNUE/CNN nets (CPU/multi-GPU). The companion
  if we want to study or retrain Rapfi-style evals.
- **[hzyhhzy/gomoku_nnue](https://github.com/hzyhhzy/gomoku_nnue)** — the original
  NNUE-Gomoku research that influenced Rapfi (Python+C++, VS-centric).
- **[nkg114mc/c-gomoku-cli](https://github.com/nkg114mc/c-gomoku-cli)** — a
  `cutechess-cli`-style match runner for Gomocup-protocol engines. Potentially
  useful for engine-vs-engine brackets.
- **[junghyun397/mintaka](https://github.com/junghyun397/mintaka)** — a modern
  from-scratch **Rust** Renju engine (PVS, **NEON** + AVX-512 SIMD, planned
  NNUE/AlphaZero). Pre-alpha, not a Gomocup entry, but the cleanest modern-arch
  reference with an Apple-Silicon-friendly NEON path.
- Educational AlphaZero-Gomoku repos (Python/LibTorch, **not** pbrain engines):
  `junxiaosong/AlphaZero_Gomoku` (canonical), `hijkzzz/alpha-zero-gomoku`,
  `zhixiangli/alphazero-board-games` (ships 9×9+15×15 presets + checkpoints).

## What's actually usable as an honest opponent

Short list, in priority order (post-nix, 2026-06-16):

0. **Net-vs-net (pure torch) + pure-python `heuristic`/`lookahead`** — the
   **reliable default eval set** ([reliable-eval-set.md](reliable-eval-set.md)).
   No subprocess, no wine, never crashes; this is the sliding derby's eval and
   the default `panel_tournament.py` field. Use this first.
1. **Rapfi, natively** — the one strong, modern, ARM-native, source-available
   engine, and the right path to a strength anchor past the ~1700 ceiling. The
   build recipe is recorded (`engines/rapfi/build_rapfi.sh`, CMake preset
   `arm64-clang-NEON-DOTPROD`, last `rapfi@6e0a132` 2026-05-24) and it speaks
   Piskvork by default. **ONLINE as of 2026-06-18 (#40):** `scripts/run-rapfi`
   (installed to `~/.cache/gomocup/bin/`) runs `pbrain-rapfi --config config.toml`,
   loading the mix9svq NNUE weights so it actually searches to its time budget —
   the #28 "weightless + TC-blind" bug was the no-config build, now resolved.
   Registered in `_NATIVE_ENGINES`; default reliable anchor, no wine. Absolute
   calibration (single-thread effective strength + balanced openings) still
   pending (#35/#22). Details: [external-engine-baselines.md](external-engine-baselines.md).
2. **The wine-run binaries** (embryo26, yixin18, pela23, zetor17, eulring16) —
   Windows `pbrain-*.exe` under wine. **SHELVED — opt-in only** (`--wine` /
   `GOMOKU_ENABLE_WINE_ENGINES=1`); unreliable (~17/36 panel pairs crash, #35).
   Kept as evidence, not in the default path.
3. **KataGomo / AlphaGomoku(MK)** — only if someone invests in an ARM-macOS build;
   strong and AlphaZero-flavored, but unproven on this hardware. Study targets
   more than drop-in opponents today.

Everything else is either too weak to be interesting, Windows-VS-only with no ARM
path, or closed binary-only.

## Gomocup rules facts (and what they mean for our Rapfi number)

Confirmed from [gomocup.org/detail-information](https://gomocup.org/detail-information/)
and the download/developer pages:

- **Single-core rule.** *"Although multi-thread programs are allowed in the
  Gomocup tournaments, one brain is restricted to use only one CPU core by setting
  CPU affinities to the program."* Competition play is effectively **single-thread**.
- **Windows-native.** Since 2022, engines must be Win10/11 executables
  (`pbrain-*.exe`, x64 names contain "64"). The whole ecosystem is Windows-first.
- **Limits.** RAM ≥70MB (announced per-tournament); time ≤30s/turn, ≤3min/match.
- **GPU / neural-net eval is allowed.** There is no rule against GPU inference;
  the top of the modern field (Rapfi NNUE, KataGomo, AlphaGomoku, Jax) is neural.
  (The 2026 Embryo release reportedly adds Vulkan NN eval — its *repo* remains
  binary-only, AVX2 / x86, so the GPU detail is from release notes, not source.)

### ⚠️ Our "full-strength Rapfi" test OVER-powered Rapfi — our champion is understated

Our prior "full-strength Rapfi" head-to-head ran Rapfi with **all 18 cores**,
which is **beyond the Gomocup single-thread rule**. That means:

- The Rapfi we measured ~62% against was **rules-illegally over-powered**
  (multi-core), so **~62% UNDERSTATES our champion** — a rules-legal,
  single-thread Rapfi would search far shallower and be **weaker**, and our win
  rate against *that* legal opponent would be **higher**.
- The **other engines we bracketed ran single-thread by default** (their binaries
  don't parallelize), so those bracket numbers are **competition-fair-ish** and
  don't carry the same over-powering caveat.
- Takeaway: when comparing to the field, pin Rapfi to **one core** for a
  rules-legal baseline; keep the multi-core number only as a "vs an over-clocked
  reference" stress point, clearly labeled.

## Bottom line

The Gomocup landscape is **mostly closed binary-only Windows engines**. The
open-source set is small: one strong ARM-native engine we already use (**Rapfi**),
two strong AlphaZero-lineage C++ engines worth reading but unproven on Apple
Silicon (**KataGomo**, **AlphaGomoku(MK)**), and a tail of weaker classical
alpha-beta engines that are largely Windows-Visual-Studio-only. For honest
opponents *today*: Rapfi natively (pin to one core for rules-legality) plus the
wine-run binaries we already execute.
