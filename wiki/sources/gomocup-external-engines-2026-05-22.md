# Gomocup External Engine Snapshot

Snapshot date: 2026-05-22.

Purpose: preserve the source trail for rated, runnable Gomoku engines that could
become external fixed baselines for the 9x9 training harness.

## Source URLs

- Gomocup homepage: https://gomocup.org/
- Gomocup Elo ratings: https://gomocup.org/elo-ratings/
- Gomocup AI downloads: https://gomocup.org/download-gomoku-ai/
- Gomocup developer/source page: https://gomocup.org/download-for-developers/
- Gomocup detail/protocol overview: https://gomocup.org/detail-information/
- Piskvork/Gomocup protocol: https://plastovicka.github.io/protocl2en.htm
- Rapfi: https://github.com/dhbloo/rapfi
- AlphaGomoku(MK): https://github.com/MaciejKozarzewski/AlphaGomoku
- KataGomo: https://github.com/hzyhhzy/KataGomo
- SlowRenju: https://github.com/wind23/SlowRenju
- PentaZen: https://github.com/sun-yuliang/PentaZen
- Carbon-Gomoku: https://github.com/gomoku/Carbon-Gomoku
- Chis: https://github.com/ChisBread/Chis/
- Piskvork/Pela source surface: https://github.com/plastovicka/Piskvork
- PyGomo protocol client: https://pypi.org/project/pygomo-lib/

## Evidence Notes

- Gomocup is the long-running AI Gomoku tournament. Its ratings page computes
  Elo from historical Gomocup results and exposes freestyle, fastgame,
  standard, and renju tables.
- Best-version freestyle ratings on the 2026-05-22 page include:
  Rapfi 0.34.05 at 2625, Embryo 0.6.4.2600 at 2437, Barbakan 1.0 at 2321,
  AlphaGomoku(MK) 5.3.0 at 2256, KataGomo 20210502 at 2254, PentaZen 0.4.18
  at 2143, SlowRenju 5.1.3 at 1857, Carbon 2.4 at 1670, Pela 7.8 at 1499,
  Chis 4.5 at 1448, and Stahlfaust 1.0 at 715.
- Gomocup 2025 placement order starts Rapfi, AlphaGomoku(MK), JAX, Barbakan,
  KataGomo, Embryo. JAX looks interesting because the download note mentions
  CPU/CUDA/TensorRT support, but no source/license trail was found in this pass.
- The Gomocup/Piskvork protocol is the common wrapping surface. It is stdin/stdout
  text, with required commands including `START`, `BEGIN`, `INFO`, `BOARD`,
  `TURN`, and `END`. Coordinates are zero-based `X,Y`. `BOARD` sends stones
  from the engine's point of view: `1` means own stone and `2` means opponent.
- The protocol requires engines to support size 20 for Gomocup, while other
  board sizes are recommended rather than guaranteed. That makes 9x9 support
  a required local smoke test for any candidate.
- Rapfi is GPL-3.0, has C++ source, uses Piskvork as its default protocol, and
  has build knobs for native x86 vector instructions plus ARM64 NEON.
- AlphaGomoku(MK) is GPL-3.0, C++, and has released Gomocup binaries. Release
  notes mention OpenCL backend work and Gomocup protocol fixes.
- SlowRenju is GPL-3.0 C++, supports the Gomocup protocol, and explicitly
  supports freestyle/standard/renju on board sizes 5 through 20.
- Chis is MPL-2.0 C++ and participates through Piskvork/Gomocup.
- Piskvork is open source and includes the Pela source surface; Gomocup's
  developer page identifies Pela as GPL C++ source.
- Carbon and PentaZen are source-available through GitHub and Gomocup's source
  page. Carbon is called open source by Gomocup; PentaZen is called source
  code on the developer page, but this pass did not find a clear OSI license on
  its GitHub page. Treat strict OSS status as unresolved until checked.

## Harness-Relevant Takeaways

- Use Gomocup Elo as external provenance, not as a direct 9x9 strength number.
  The tournament ladder is mostly 15x15/20x20 with tournament openings/time
  controls; our current project is 9x9 freestyle from an empty board.
- The right local metric is: "current checkpoint vs rated engine X at local
  time control Y on 9x9." That gives a stable external regression/strength
  anchor while preserving the source rating as context.
- A Piskvork wrapper should start by sending `START 9`, `INFO rule 0`, a bounded
  `INFO timeout_turn`, then `BOARD` snapshots for each move. If `START 9`
  returns `ERROR`, that engine is not a direct fit for the current 9x9 harness.
