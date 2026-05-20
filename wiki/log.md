# Gomoku Wiki Log

Chronological record of wiki maintenance. Keep entries append-only and use a
consistent heading so future sessions can scan recent changes with simple tools.

## [2026-05-19] setup | LLM wiki operating model

- Added [index.md](index.md) as the wiki entry point and content catalog.
- Added [topics/wiki-operating-model.md](topics/wiki-operating-model.md) to adapt
  the Karpathy LLM wiki pattern to this repo.
- Added [sources/karpathy-llm-wiki.md](sources/karpathy-llm-wiki.md) as the
  source record for the organizing charter.
- Updated [../AGENTS.md](../AGENTS.md) and [../TRAINING_WIKI.md](../TRAINING_WIKI.md)
  so future sessions treat the wiki as a compounding synthesis layer, not just a
  large experiment transcript.

## [2026-05-19] synthesis | MCTS perf ceiling topic page

- Added [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) capturing
  the finding that our `gomoku/mcts.py` is already at the AGZ "mcts_v2"
  storage layout that other AZ codebases advertise as a big upgrade. Ports
  of that design are a no-op for us.
- The cross-game BFS-vectorized descent (Exp 9 in
  [../TRAINING_WIKI.md](../TRAINING_WIKI.md)) is real but small: 1.26× at
  G=32 / wave=16, ~1.05–1.10× at our dist G=8 / wave=32 config. The next
  2× requires batched `state.apply` on tensor, C-extension `_init_node`,
  or multi-device gen-vs-train split — not numpy reshuffling.
- Updated [index.md](index.md) Start Here table to surface the new topic.

## [2026-05-19] notebook | az-recipe-160k launched, first sustained heuristic crossing

- Live run `9x9-sweep-az-recipe-160k` (wandb `sppjo3z5`) is at e1285 with
  heuristic 50-70% sustained across e1119/1179/1227 — never seen in dist
  mode before. Prior best dist crossing was fresh-dist plateauing at ~35%
  by e552; all other dist cells stayed at 0% heuristic for the entire 100
  epochs they ran.
- Launched config differs from the documented recipe: small model +
  stem_padding=1 + sims=400 (instead of medium / 3 / 800) to keep wall-
  clock inside one day on the M5 Max. The recipe defaults benched at
  ~30 s/cycle = ~46h for 5000 cycles; trimmed config is ~2 s/cycle when
  games are short.
- Captured Jason's calibration that cycle-time scales super-linearly
  with mean plies, so the current ~2 s/cycle is a *floor* — real defense
  learning would blow out the ETA. Filed as a feedback memory so future
  sessions don't quote naive ETAs.
- Added live SUMMARY to [../TRAINING_WIKI.md](../TRAINING_WIKI.md) with
  the eval table, plies puzzle, and open questions to resolve as the run
  continues (does heuristic hold, does lookahead2 climb, do plies
  regrow, head-to-head vs kze-e176).
