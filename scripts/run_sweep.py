"""Multi-run-safe launcher for the K x buffer-size sweep.

Each cell defines a sweep point; the launcher generates unique directory paths
from the cell name so multiple cells can run in parallel without colliding on
checkpoint dirs, worker_records dirs, or wandb run names. Each cell launches:

  - 1 trainer (gomoku.train --no-eval)
  - N self-play workers (gomoku.selfplay_worker)
  - 1 eval worker (gomoku.eval_worker)

Usage::

    # See all defined cells.
    python scripts/run_sweep.py --list

    # Run one cell in the foreground (logs to stdout).
    python scripts/run_sweep.py --cell C --foreground

    # Default: spawn a cell in the background, return immediately.
    python scripts/run_sweep.py --cell C

    # Wipe a cell's persistent state (checkpoint dir + records dir).
    python scripts/run_sweep.py --cell C --clean

Per-process logs land in `sweep_logs/<cell>/{trainer,w0..wN,eval}.log`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
# Reuse whichever interpreter launched us — works from main repo OR a worktree
# that doesn't have its own .venv.
PYTHON = sys.executable

# Time-capped-slice teardown tuning (used only when --max-wall-secs is set).
# The trainer self-caps at --max-wall-secs on an epoch boundary and exits clean;
# these govern the supervisor's fallbacks around that.
TRAINER_CAP_GRACE_SEC = 180.0   # let one long epoch finish past the cap before forcing
TEARDOWN_GRACE_SEC = 90.0       # time for the trainer's final buffer save (~1.4 GB)
FINAL_EVAL_TIMEOUT_SEC = 360.0  # one eval_worker cycle incl. lookahead:depth=4


@dataclass
class Cell:
    """One sweep point. dirs/run_name are derived from `name`."""
    name: str
    sgd_per_game: float
    buffer_size: int
    games_per_epoch: int = 32              # also the worker_min_games default
    n_simulations: int = 800
    wave_size: int = 32
    batch_size: int = 512
    lr: float = 1e-3
    size: str = "medium"
    c_puct: float = 1.25
    c_puct_base: float = 19652.0
    dirichlet_alpha: float = 0.13
    dirichlet_eps: float = 0.25
    temperature_moves: int = 10
    temperature_final: float = 0.1
    n_workers: int = 4
    epochs: int = 100
    save_every: int = 1
    save_buffer_every: int = 20
    keep_last_n: int = 3
    stem_padding: int | None = None  # None = use train.py default (3, AGZ edge-fix)
    # Derby v4 "Whole-board" lever. None = OFF (byte-identical current arch);
    # True = global pooling on the latter half of residual blocks; int K =
    # trailing K blocks. See gomoku.model.GlobalPoolResBlock.
    global_pool: bool | int | None = None
    # Constant-age ingest: when set, ingest by positions instead of games. Pairs
    # with sgd_per_position so SGD steps also scale with positions. Together
    # they keep buffer turnover + training intensity stable regardless of
    # self-play game length (which swings between attack and defense regimes).
    worker_min_positions: int = 0
    sgd_per_position: float | None = None
    # Per-worker games-per-batch. Bigger batches give MCTS more leaves per
    # evaluator call, but the 2026-05-20 production read showed that one big
    # worker leaves MPS idle during Python tree work. At this model size, more
    # small workers plus wave batching beats a single wide worker.
    games_per_batch: int = 8
    # Wave-lockstep mode is a per-version tile barrier: workers generate a
    # tile against one published model, the trainer ingests that tile, then
    # publishes the next model. Only cells that opt in should receive the flag.
    wave_mode: bool = False
    # Apply torch.compile to worker models (eval-only). ~1.3-1.5x forward
    # speedup at batch>=32 for the small model on MPS.
    compile_workers: bool = False
    # WL2 levers (wiki/topics/wl2-scale-emulation-design.md). All default-off
    # so existing cells are unchanged. WL2 turns all four on.
    ema_tau: float = 0.0                          # lever #1: EMA self-play weights
    grad_accum_steps: int = 1                     # lever #4: gradient accumulation
    opponent_mix_recent: float = 0.0              # lever #2: past-checkpoint mix (recent)
    opponent_mix_history: float = 0.0             # lever #2: past-checkpoint mix (history)
    opponent_mix_recent_window: int = 100         # lever #2: recent window size
    weights_poll_min_sec: float | None = None     # lever #3: poll-interval jitter min
    weights_poll_max_sec: float | None = None     # lever #3: poll-interval jitter max
    # WL3 lever: K plies of uniform-random legal moves at game start. Training
    # examples are NOT recorded for those K plies — MCTS picks up at K+1 and
    # the model trains on post-random positions. Breaks opening monoculture.
    random_opening_moves: int = 0
    # WL5 levers (wiki/topics/wl5-diagnostics-archive-start-design.md). Trainer
    # scores a frozen validation set every eval cycle for stationary policy/
    # value quality. Workers seed `archive_start_frac` of games from the same
    # archive instead of the canonical empty board.
    validation_archive_path: str | None = None
    archive_start_path: str | None = None
    archive_start_frac: float = 0.0
    # Log this cell to W&B. Real cells do; the SMOKE plumbing cell sets False
    # so repeated bundle smokes don't litter the project with junk runs.
    wandb: bool = True
    extra_train_args: list[str] = field(default_factory=list)
    extra_worker_args: list[str] = field(default_factory=list)


# Sweep matrix: K = sgd-per-game, buffer-size axes.
# Cell E ≈ what we just ran (control). C is the highest-contrast first try.
CELLS: dict[str, Cell] = {
    # Fast bundle-plumbing smoke: tiny model, 2 workers, small buffer, low sims.
    # Not a training experiment — exists to exercise the launch → supervise →
    # teardown → --final-eval path quickly (pair with --max-wall-secs). epochs
    # high + save_buffer_every high so the only resumable save is the clean-stop.
    "SMOKE": Cell("SMOKE-bundle-plumbing", sgd_per_game=1.0,
                  buffer_size=5_000, games_per_epoch=8,
                  size="tiny", stem_padding=1, n_simulations=30,
                  n_workers=2, games_per_batch=4,
                  temperature_moves=10, temperature_final=0.1,
                  save_buffer_every=100_000, epochs=100_000, wandb=False),
    "A": Cell("A-K1-buf50k",   sgd_per_game=1.0, buffer_size=50_000),
    "B": Cell("B-K2-buf50k",   sgd_per_game=2.0, buffer_size=50_000),
    "C": Cell("C-K4-buf50k",   sgd_per_game=4.0, buffer_size=50_000),
    "D": Cell("D-K1-buf500k",  sgd_per_game=1.0, buffer_size=500_000),
    "E": Cell("E-K2-buf500k",  sgd_per_game=2.0, buffer_size=500_000),
    "F": Cell("F-K4-buf500k",  sgd_per_game=4.0, buffer_size=500_000),
    # AZ-recipe long run, tuned for laptop wall-clock: small model + padding=1
    # + sims=400. Trades the stem-padding edge-fix and some sim quality for ~4x
    # the per-cycle throughput, so a 160k-step run fits in a multi-day budget
    # rather than 2+ days. K=1, 1.5M buffer, tau_final=0.1, AGZ log-PUCT all
    # stay from the recipe imports.
    #
    # Production-proven config: 4 workers × 8 games × wave=64. The 4-worker
    # setup keeps MPS busy via OS-scheduled kernel interleaving across
    # processes — single-worker × big-batch hurts GPU utilization because
    # the GPU sits idle during Python tree-traversal between forward calls
    # (subagent's "MPS serializes processes" claim turned out wrong in
    # practice). wave=64 is the genuine win from the bench (1.71× per-worker
    # vs wave=32 with no plies regression). torch.compile dropped — gains
    # don't survive the per-cycle weight-reload recompiles.
    "Z": Cell("az-recipe-160k", sgd_per_game=1.0, buffer_size=1_500_000,
              size="small", stem_padding=1, n_simulations=400,
              lr=5e-4, epochs=5000,
              n_workers=8, wave_size=64, games_per_batch=8,
              compile_workers=False),
    # AZ-recipe long run with constant-age ingest. Same recipe as Z but uses
    # positions-based ingest + SGD so buffer turnover stays stable when
    # self-play game length swings (which Z's run showed it does — plies
    # dropped from ~60 to ~38 mid-run and broke training-loss stability).
    # Target: 32 games * 50 mean plies * 8 D4 aug = 12800 positions/cycle,
    # 32 SGD steps/cycle = 0.0025 SGD/position. Matches Z's effective rate
    # at the 50-plies baseline.
    "Zc": Cell("az-recipe-160k-constage", sgd_per_game=1.0,
               buffer_size=1_500_000, size="small", stem_padding=1,
               n_simulations=400, lr=5e-4, epochs=5000,
               worker_min_positions=12800, sgd_per_position=0.0025),
    # WL1: wave-lockstep at the proven 1.5M replay-buffer scale (matches
    # az-recipe-160k and avoids changing the buffer-size axis). First test of
    # the per-version uniformity hypothesis from
    # wiki/topics/wave-of-lockstep-design.md: 8 workers each produce an 8-game
    # tile against one model version, then the trainer steps and publishes the
    # next version. Existing sgd_per_position convention is K=1 per game at
    # ~50 plies with D4 augmentation:
    #   64 games * 50 plies * 8 aug = 25,600 positions/tile
    #   64 SGD steps / 25,600 positions = 0.0025
    "WL1": Cell("WL1-wave-lockstep-1p5M-buffer", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100),
    # WL2: WL1 + the four scale-emulation levers from
    # wiki/topics/wl2-scale-emulation-design.md, designed in response to WL1's
    # high-frequency strength oscillation (elo bounced 620-1281 across single
    # evals after e500, la4 regressed 52% -> 5%). Each lever emulates one
    # AZ-at-scale property our 8-worker laptop setup lacks:
    #   #1 EMA self-play weights (tau=0.99): decouples the brain that plays
    #      from the brain that learns -> emulates AZ's publish-to-workers lag.
    #   #2 past-checkpoint opponent mix (recent=0.4, history=0.1): some waves
    #      run an older snapshot on both sides -> emulates the multi-snapshot
    #      generation that AZ has by default from async publishing.
    #   #3 worker poll jitter (2-8s): workers pick up new weights at slightly
    #      different times -> emulates per-worker async-publish skew.
    #   #4 gradient accumulation 4x: cuts per-step gradient noise -> emulates
    #      AZ's batch=4096-against-broad-buffer stability.
    # Everything else identical to WL1 for an apples-to-apples test.
    "WL2": Cell("WL2-scale-emulation", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100,
                ema_tau=0.99,
                grad_accum_steps=4,
                opponent_mix_recent=0.4,
                opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0,
                weights_poll_max_sec=8.0),
    # WL3: WL2 + K=2 random opening plies. WL2 reached higher peaks than WL1
    # (la4=62% vs 52%) but exhibited the same arc-then-regression failure
    # mode. Working theory: the four scale-emulation levers raised the
    # ceiling by stabilizing the training feedback loop, but didn't address
    # opening monoculture — even past-checkpoint mix delivers diverse brains
    # that all share the same opening lineage. WL3 adds K=2 uniform-random
    # legal moves at game start; training examples are NOT recorded for the
    # random plies (per train.py:165-170), so the model just sees more
    # diverse mid-game starting positions without learning broken-move signal.
    "WL3": Cell("WL3-random-openings", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100,
                ema_tau=0.99,
                grad_accum_steps=4,
                opponent_mix_recent=0.4,
                opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0,
                weights_poll_max_sec=8.0,
                random_opening_moves=2),
    # WL3.1: identical to WL3, restart after WL3 died at e825 from a NaN-
    # in-policy crash. The native MCTS engine occasionally emits NaN visit-
    # policies; commits c5049be + 0557671 add NaN guards at the play path
    # (gomoku/self_play.py::_sample_action) and the data-recording path
    # (trajectory pi sanitization). With those guards in place, an
    # occasional NaN no longer kills workers or poisons the buffer — it
    # falls back to argmax / uniform target. A parallel investigator is
    # looking at the underlying MCTS NaN root cause.
    "WL3.1": Cell("WL3.1-random-openings-nanfix", sgd_per_game=1.0,
                  buffer_size=1_500_000, games_per_epoch=64,
                  size="small", stem_padding=1, n_simulations=400,
                  n_workers=8, games_per_batch=8, wave_mode=True,
                  c_puct=1.25, c_puct_base=19652.0,
                  dirichlet_alpha=0.13, dirichlet_eps=0.25,
                  temperature_moves=30, temperature_final=0.1,
                  sgd_per_position=0.0025,
                  save_buffer_every=100,
                  ema_tau=0.99,
                  grad_accum_steps=4,
                  opponent_mix_recent=0.4,
                  opponent_mix_history=0.1,
                  opponent_mix_recent_window=100,
                  weights_poll_min_sec=2.0,
                  weights_poll_max_sec=8.0,
                  random_opening_moves=2),
    # WL4: WL3.1 with random openings TURNED OFF. Resumes from WL3.1 e1536
    # (latest.pt snapshotted as $CLAUDE_JOB_DIR/wl3.1_e1536_latest.pt) which
    # reached the "established model" trigger Jason proposed:
    # eval/vs_heuristic hit 100% sustained (e1123-1157), la4 sustained 60-95%
    # across 5+ evals, plies hit 27 (single-eval). Hypothesis: with diversity
    # baked in, canonical-opening depth becomes the bottleneck. Pulling
    # random openings should either (a) unlock further breakthrough as the
    # model can finally compound on canonical lines, or (b) cause rapid
    # collapse — showing diversity is permanent training infrastructure at
    # this model size. Both outcomes informative.
    "WL4": Cell("WL4-no-random-openings", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100,
                ema_tau=0.99,
                grad_accum_steps=4,
                opponent_mix_recent=0.4,
                opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0,
                weights_poll_max_sec=8.0,
                random_opening_moves=0),  # ← the experimental change
    # WL5: WL4 + WL5 diagnostics + archive-start lever (see
    # wiki/topics/wl5-diagnostics-archive-start-design.md). Resumes from
    # WL4 e4024 (sweep_runs/WL4-no-random-openings.plateau-e4024/checkpoints/
    # latest.pt) via CLI --resume; cell rename gives clean wandb timeline.
    # WL4 plateaued at elo 1841 ATH with la4=100% — best WL-series outcome
    # to date but no further breakthrough. Two streams here:
    #   diagnostics (validation_archive_path): trainer scores a frozen ~1-2k
    #     position archive every eval cycle to separate target-entropy noise
    #     from learning-gap effects in the policy loss.
    #   archive-start (archive_start_path/frac): per-game, with prob 0.15,
    #     workers seed from a curated trouble-state position instead of empty
    #     board. Tests Go-Exploit-style state-coverage gap hypothesis.
    # All WL4 levers preserved (EMA, past-mix, poll jitter, grad-accum 4×).
    "WL5": Cell("WL5-diagnostics-archive-start", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100,
                ema_tau=0.99,
                grad_accum_steps=4,
                opponent_mix_recent=0.4,
                opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0,
                weights_poll_max_sec=8.0,
                random_opening_moves=0,
                validation_archive_path="archives/wl5_validation_v1.pt",
                archive_start_path="archives/wl5_validation_v1.pt",
                archive_start_frac=0.15),
    # LF1: the perf lab's R-TRAIN-LEAN-fp16 recipe (+152% throughput vs WL5 in
    # lab_train_cell) as a REAL training run — the TQ canary. Exact WL5 recipe
    # with the 3 perf deltas: wave_size 64->512, sgd_per_position 0.0025->0.001,
    # workers +--fp16-eval. 100-epoch fresh test: does the faster recipe LEARN
    # cleanly (val/policy_ce down, plies healthy, eval-vs-baselines climbing,
    # 0 NaN)? Started HOT (chip heat-soaked from the 2026-05-23 perf session;
    # note for cold/hot comparison). Production adoption stays TQ-gated.
    "LF1": Cell("LF1-lean-fp16-canary", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                wave_size=512,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.001,
                save_buffer_every=100,
                ema_tau=0.99,
                grad_accum_steps=4,
                opponent_mix_recent=0.4,
                opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0,
                weights_poll_max_sec=8.0,
                random_opening_moves=0,
                epochs=1000,
                validation_archive_path="archives/wl5_validation_v1.pt",
                archive_start_path="archives/wl5_validation_v1.pt",
                archive_start_frac=0.15,
                extra_worker_args=["--fp16-eval"]),
    # PERFA / PERFB: two IDENTICAL clones of the WL4 production recipe, used
    # ONLY to measure concurrent-run perf degradation (Jason 2026-05-24). Each
    # is a full 8-worker + trainer + eval production launch with its own wandb
    # run. Distinct cell names give independent sweep_runs/sweep_logs dirs so
    # each run's per-epoch (gen=/train=) timing can be parsed separately. Not a
    # training experiment — fresh weights, torn down after the measurement.
    "PERFA": Cell("PERFA-degrade-test", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100,
                ema_tau=0.99,
                grad_accum_steps=4,
                opponent_mix_recent=0.4,
                opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0,
                weights_poll_max_sec=8.0,
                random_opening_moves=0,
                epochs=100),
    "PERFB": Cell("PERFB-degrade-test", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025,
                save_buffer_every=100,
                ema_tau=0.99,
                grad_accum_steps=4,
                opponent_mix_recent=0.4,
                opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0,
                weights_poll_max_sec=8.0,
                random_opening_moves=0,
                epochs=100),
    # Δelo Derby v2 roster (raced by scripts/delo_derby.py via the
    # run_sweep_wall_slice engine + scripts/derby_v2_board.json). The v1 top-3,
    # now on the PRODUCTION multiprocess wave-mode recipe (WL4-style base, no
    # WL5 archive/validation levers — clean fresh race). Each changes exactly
    # ONE lever vs the shared base. epochs huge so only --max-wall-secs stops a
    # slice. NOTE (sgd-800): in wave-mode, training intensity is governed by
    # sgd_per_position; train.py overrides --training-steps when it's set, so
    # the faithful "2× SGD" lever is sgd_per_position 0.0025 -> 0.005, not
    # training_steps.
    "derby-open-div4": Cell("derby-open-div4", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000,
                random_opening_moves=4),   # ← the one lever (was 0)
    "derby-temp-16": Cell("derby-temp-16", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=16, temperature_final=0.1,   # ← the one lever (was 30)
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000,
                random_opening_moves=0),
    "derby-sgd-800": Cell("derby-sgd-800", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.005, save_buffer_every=100,   # ← the one lever (2× baseline)
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000,
                random_opening_moves=0),
    # ── Δelo Derby v3 ── prior-art levers (learn-from-those-who-came-before),
    # raced wall-matched against the SAME base as the v2 carryovers above. C0 is
    # the no-lever control; each lever cell changes exactly one thing vs C0 via
    # the extra_worker_args / extra_train_args escape hatches (the v3 flags live
    # on selfplay_worker [gen levers] and gomoku.train [swa]). In wave-mode the
    # workers generate, so playout-cap / forced-playouts / gumbel are WORKER
    # args; SWA publishes the generator weights, so it is a TRAINER arg.
    "derby-c0": Cell("derby-c0", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000,
                random_opening_moves=0),   # control: no lever
    "derby-playoutcap": Cell("derby-playoutcap", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                # ← the one lever: KataGo playout-cap randomization
                extra_worker_args=["--playout-cap-frac", "0.25",
                                   "--playout-cap-fast-sims", "50"]),
    "derby-forced": Cell("derby-forced", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                # ← the one lever: KataGo forced playouts + target pruning
                extra_worker_args=["--forced-playout-k", "2.0"]),
    "derby-swa": Cell("derby-swa", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=400,
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                # ← the one lever: SWA tail-avg generator (replaces the EMA publish)
                extra_train_args=["--swa-window", "5"]),
    "derby-gumbel": Cell("derby-gumbel", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,  # cheap sims — the gumbel value-prop
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                # ← the lever: native Gumbel root + SH at cheap sims (good targets, fast gen)
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16"]),
    "derby-sims100": Cell("derby-sims100", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,  # gumbel's control: plain MCTS at 100 sims
                n_workers=8, games_per_batch=8, wave_mode=True,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000,
                random_opening_moves=0),   # control for gumbel: 100 sims, no gumbel (v1: floored)
    # A/B vs derby-gumbel: SAME generator (Gumbel@100, 8 workers) but the FIXED-STEP /
    # fast-cadence trainer instead of wave + sgd_per_position. wave_mode=False routes to
    # the non-wave async path; --sgd-steps-per-epoch N pins SGD to N steps/epoch
    # (decoupled from the gen flood → structurally can't run away) with non-blocking
    # ingest (~5s/epoch, many fast epochs). Tests whether "many faster epochs" climbs
    # faster than wave-scaled SGD when generation floods. N=64 is a ~5s starting point —
    # confirm/tune from the first chunk's trainer.log epoch wall + watch train/sample_reuse_ratio.
    "derby-gumbel-fast5s": Cell("derby-gumbel-fast5s", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,  # ← non-wave async path
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,  # (overridden by --sgd-steps-per-epoch)
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16"],      # same gen as derby-gumbel
                extra_train_args=["--sgd-steps-per-epoch", "64"]),            # ← the fixed-step lever

    # ── Derby v4 combination lanes ──────────────────────────────────────────
    # derby-v4-control is a FRESH instance of the v3-winning recipe (byte-identical
    # to derby-gumbel-fast5s) under a NEW name, so all four v4 lanes start from
    # scratch on the same wall budget (fair race) with a clean wandb timeline and
    # no collision with v3's 8.8G derby-gumbel-fast5s checkpoint dir. The next three
    # each take that exact fixed-step + Gumbel@100 base and add ONE combination
    # lever (no more single-lever cells; these are our best shots at a GREAT
    # player). Only deltas are global_pool / aux heads / vcf-teacher.
    "derby-v4-control": Cell("derby-v4-control", sgd_per_game=1.0,   # fresh copy of the v3 winner
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    "derby-signal": Cell("derby-signal", sgd_per_game=1.0,           # 'Signal-rich': both aux heads
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                # worker records BOTH aux targets; trainer enables BOTH heads via weights>0
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--record-aux", "--record-ownership"],
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--aux-opponent-reply-weight", "0.15",
                                  "--aux-ownership-weight", "0.15"]),
    "derby-wholeboard": Cell("derby-wholeboard", sgd_per_game=1.0,   # 'Whole-board': KataGo global-pooling
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,                                    # ← latter-half global-pool blocks
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    "derby-vcf": Cell("derby-vcf", sgd_per_game=1.0,                 # 'VCF-taught': exact tactical labels
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                # worker overwrites policy/value targets with VCF mate labels when found
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--vcf-teacher"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),

    # ── Derby v5 'stack the winners' ────────────────────────────────────────
    # v4 champion = vcf (VCF mate-teacher on fixed-step + Gumbel@100). v5 asks: do
    # the other levers COMPOUND on top of vcf? Every cell is the vcf base + ONE
    # added lever; derby-v5-control is the bare vcf base (the bar to clear). All
    # start FRESH + fair (global-pool can't warm-start a non-global trunk, so
    # uniform fresh start keeps it apples-to-apples).
    "derby-v5-control": Cell("derby-v5-control", sgd_per_game=1.0,   # bare vcf base (the bar)
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--vcf-teacher"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    "derby-v5-vcf-signal": Cell("derby-v5-vcf-signal", sgd_per_game=1.0,  # vcf + aux heads
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--vcf-teacher", "--record-aux", "--record-ownership"],
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--aux-opponent-reply-weight", "0.15",
                                  "--aux-ownership-weight", "0.15"]),
    "derby-v5-vcf-wholeboard": Cell("derby-v5-vcf-wholeboard", sgd_per_game=1.0,  # vcf + global-pool
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--vcf-teacher"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    "derby-v5-vcf-deep": Cell("derby-v5-vcf-deep", sgd_per_game=1.0,  # deeper/aggressive VCF solver
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                # deeper solver: proves longer forced wins -> more positions labeled
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--vcf-teacher", "--vcf-max-depth", "32",
                                   "--vcf-max-nodes", "500000"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),

    # ── Derby v6 'researcher round 1' ──────────────────────────────────────
    # Gated 2026-05-25 from the beads backlog. All on the vcf base (derby-v5-control);
    # control is the bar. Levers: adjudicate (--max-plies first cut, games/hr),
    # mate-discount (--value-discount 0.98, generalizes the VCF discount to all
    # outcomes), and an sgd-steps sweep (128/256 vs the 64 baseline = control).
    "derby-v6-control": Cell("derby-v6-control", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vcf-teacher"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    "derby-v6-adjudicate": Cell("derby-v6-adjudicate", sgd_per_game=1.0,  # + --max-plies (games/hr)
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vcf-teacher",
                                   "--max-plies", "45"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    "derby-v6-mate-discount": Cell("derby-v6-mate-discount", sgd_per_game=1.0,  # + --value-discount
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vcf-teacher",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    "derby-v6-sgd128": Cell("derby-v6-sgd128", sgd_per_game=1.0,  # sgd-steps sweep: 2x baseline
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vcf-teacher"],
                extra_train_args=["--sgd-steps-per-epoch", "128"]),
    "derby-v6-sgd256": Cell("derby-v6-sgd256", sgd_per_game=1.0,  # sgd-steps sweep: 4x baseline
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vcf-teacher"],
                extra_train_args=["--sgd-steps-per-epoch", "256"]),

    # ── Derby v7 'best base' ───────────────────────────────────────────────
    # Rebased on the v5 H2H winner: vcf + global-pooling (the new control/base).
    # Folds in the v6 H2H winners as carry lanes (mate-discount #1, adjudicate #2 —
    # re-tested on the stronger base to confirm they still compound) + the gated new
    # lever: buffer-composition (recency curator). sgd-sweep is dead (v6). Base recipe
    # = vcf (gumbel + --vcf-teacher + fixed-step 64) + global_pool=True.
    "derby-v7-control": Cell("derby-v7-control", sgd_per_game=1.0,  # vcf+wholeboard = new base
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vcf-teacher"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    "derby-v7-mate-discount": Cell("derby-v7-mate-discount", sgd_per_game=1.0,  # + v6 winner
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vcf-teacher",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    "derby-v7-adjudicate": Cell("derby-v7-adjudicate", sgd_per_game=1.0,  # + v6 #2
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vcf-teacher",
                                   "--max-plies", "45"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    "derby-v7-buffer-comp": Cell("derby-v7-buffer-comp", sgd_per_game=1.0,  # gated new lever
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                # curated sampling: 50% of each batch from the most-recent 200k positions
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vcf-teacher"],
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--buffer-recency-frac", "0.5"]),

    # ── Derby running-pool candidates (swapped in/out by the derby runner) ──
    # 'stack' = combine the v6 H2H winners on the v5 base: vcf+global-pool +
    # value-discount + max-plies. The 'best recipe so far' bet.
    "derby-x-stack": Cell("derby-x-stack", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vcf-teacher",
                                   "--value-discount", "0.98", "--max-plies", "45"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
}


def cell_dirs(cell: Cell) -> dict:
    base = REPO_ROOT / "sweep_runs" / cell.name
    return {
        "base": base,
        "checkpoint_dir": base / "checkpoints",
        "records_dir": base / "checkpoints" / "_records",
        "worker_weights": base / "checkpoints" / "worker_weights.pt",
        "log_dir": REPO_ROOT / "sweep_logs" / cell.name,
    }


def trainer_cmd(cell: Cell, dirs: dict) -> list[str]:
    cmd = [
        PYTHON, "-u", "-m", "gomoku.train",
        "--size", cell.size,
        "--epochs", str(cell.epochs),
        "--games-per-epoch", str(cell.games_per_epoch),
        "--n-simulations", str(cell.n_simulations),
        "--wave-size", str(cell.wave_size),
        "--sgd-per-game", str(cell.sgd_per_game),
        "--min-training-steps", "16",
        "--batch-size", str(cell.batch_size),
        "--lr", str(cell.lr),
        "--replay-buffer-size", str(cell.buffer_size),
        "--c-puct", str(cell.c_puct),
        "--c-puct-base", str(cell.c_puct_base),
        "--dirichlet-alpha", str(cell.dirichlet_alpha),
        "--dirichlet-eps", str(cell.dirichlet_eps),
        "--temperature-moves", str(cell.temperature_moves),
        "--temperature-final", str(cell.temperature_final),
        "--worker-input-dir", str(dirs["records_dir"]),
        "--worker-weights-path", str(dirs["worker_weights"]),
        "--worker-min-games", str(cell.games_per_epoch),
        "--checkpoint-dir", str(dirs["checkpoint_dir"]),
        "--save-every", str(cell.save_every),
        "--save-buffer-every", str(cell.save_buffer_every),
        "--keep-last-n", str(cell.keep_last_n),
        "--no-eval",
        "--wandb-project", "gomoku",
        "--run-name", f"9x9-sweep-{cell.name}",
        ("--wandb" if cell.wandb else "--no-wandb"),
        *cell.extra_train_args,
    ]
    if cell.stem_padding is not None:
        cmd += ["--stem-padding", str(cell.stem_padding)]
    if cell.global_pool is not None and cell.global_pool is not False:
        # bare flag (latter half) vs explicit trailing-K-blocks
        cmd += ["--global-pool"] if cell.global_pool is True else [
            "--global-pool", str(int(cell.global_pool))
        ]
    if cell.worker_min_positions > 0:
        cmd += ["--worker-min-positions", str(cell.worker_min_positions)]
    if cell.sgd_per_position is not None:
        cmd += ["--sgd-per-position", str(cell.sgd_per_position)]
    if cell.wave_mode:
        cmd += [
            "--wave-mode",
            "--wave-workers", str(cell.n_workers),
            "--wave-games-per-worker", str(cell.games_per_batch),
        ]
    if cell.ema_tau > 0:
        cmd += ["--ema-tau", str(cell.ema_tau)]
    if cell.grad_accum_steps > 1:
        cmd += ["--grad-accum-steps", str(cell.grad_accum_steps)]
    if cell.random_opening_moves > 0:
        cmd += ["--random-opening-moves", str(cell.random_opening_moves)]
    if cell.validation_archive_path is not None:
        cmd += ["--validation-archive-path", cell.validation_archive_path]
    return cmd


def worker_cmd(cell: Cell, dirs: dict, worker_id: str, seed: int) -> list[str]:
    cmd = [
        PYTHON, "-u", "-m", "gomoku.selfplay_worker",
        "--weights-path", str(dirs["worker_weights"]),
        "--output-dir", str(dirs["records_dir"]),
        "--worker-id", worker_id,
        "--games-per-batch", str(cell.games_per_batch),
        "--n-simulations", str(cell.n_simulations),
        "--wave-size", str(cell.wave_size),
        "--c-puct", str(cell.c_puct),
        "--c-puct-base", str(cell.c_puct_base),
        "--temperature-moves", str(cell.temperature_moves),
        "--temperature-final", str(cell.temperature_final),
        "--dirichlet-alpha", str(cell.dirichlet_alpha),
        "--dirichlet-eps", str(cell.dirichlet_eps),
        "--seed", str(seed),
        *cell.extra_worker_args,
    ]
    if cell.wave_mode:
        cmd += ["--wave-mode"]
    if cell.compile_workers:
        cmd += ["--compile"]
    if cell.opponent_mix_recent > 0:
        cmd += ["--opponent-mix-recent", str(cell.opponent_mix_recent)]
    if cell.opponent_mix_history > 0:
        cmd += ["--opponent-mix-history", str(cell.opponent_mix_history)]
    if cell.opponent_mix_recent > 0 or cell.opponent_mix_history > 0:
        cmd += ["--opponent-mix-recent-window", str(cell.opponent_mix_recent_window)]
    if cell.weights_poll_min_sec is not None:
        cmd += ["--weights-poll-min-sec", str(cell.weights_poll_min_sec)]
    if cell.weights_poll_max_sec is not None:
        cmd += ["--weights-poll-max-sec", str(cell.weights_poll_max_sec)]
    if cell.random_opening_moves > 0:
        cmd += ["--random-opening-moves", str(cell.random_opening_moves)]
    if cell.archive_start_path is not None and cell.archive_start_frac > 0:
        cmd += [
            "--archive-start-path", cell.archive_start_path,
            "--archive-start-frac", str(cell.archive_start_frac),
        ]
    return cmd


def eval_cmd(cell: Cell, dirs: dict) -> list[str]:
    # Lookahead:depth=4 is included as a 4th anchor: with depth=4 anchored at
    # 1500 Elo the implied_elo MLE doesn't saturate against lookahead2 (1200),
    # so the model_elo trajectory shows real strength changes instead of
    # ceiling-clamping when the model crushes the lower-rated baselines.
    # n_workers=4 keeps the multi-baseline eval cycle under ~3min via parallel
    # game-playing (see gomoku/eval.py:play_match_parallel).
    return [
        PYTHON, "-u", "-m", "gomoku.eval_worker",
        "--checkpoint-path", str(dirs["worker_weights"]),
        "--baselines", "random,heuristic,lookahead:depth=2,lookahead:depth=4",
        "--n-games", "20",
        "--sims", "100",
        "--device", "cpu",
        "--n-workers", "4",
        "--poll-sec", "2.0",
    ]


def list_cells() -> None:
    print(f"{'cell':<6} {'name':<20} {'K':<6} {'buffer':<10} {'sims':<6} {'workers':<8} {'epochs':<8}")
    for k, c in CELLS.items():
        print(f"{k:<6} {c.name:<20} {c.sgd_per_game:<6} {c.buffer_size:<10} "
              f"{c.n_simulations:<6} {c.n_workers:<8} {c.epochs:<8}")


def clean_cell(cell: Cell) -> None:
    dirs = cell_dirs(cell)
    for key in ("checkpoint_dir", "log_dir"):
        p = dirs[key]
        if p.exists():
            shutil.rmtree(p)
            print(f"removed {p}")
    base = dirs["base"]
    if base.exists() and not any(base.iterdir()):
        base.rmdir()


def _terminate_all(procs: list[tuple[str, subprocess.Popen]]) -> None:
    """SIGTERM everything still alive, then wait. The trainer's SIGTERM handler
    force-saves a resumable latest.pt (buffer embedded), so we give the group up
    to TEARDOWN_GRACE_SEC to flush before hard-killing any straggler. Workers and
    the eval_worker are stateless and exit promptly."""
    for _label, p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + TEARDOWN_GRACE_SEC
    for label, p in procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            print(f"  {label} did not exit within {TEARDOWN_GRACE_SEC:g}s; killing")
            p.kill()


def _run_final_eval(cell: Cell, dirs: dict) -> None:
    """One-shot eval of the final published weights → a fresh eval/model_elo line
    in eval_results.jsonl. This is the training machine evaluating itself at the
    end of a capped slice (eval stays inside the bundle, not the lab's job)."""
    cmd = eval_cmd(cell, dirs) + ["--max-cycles", "1"]
    log_path = dirs["log_dir"] / "final_eval.log"
    print(f"=== final eval (one-shot --max-cycles 1) for cell {cell.name} ===")
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env["GOMOKU_DEVICE"] = "cpu"
    with open(log_path, "a") as log_f:
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                             env=env, cwd=str(REPO_ROOT))
        try:
            p.wait(timeout=FINAL_EVAL_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            print(f"  final eval exceeded {FINAL_EVAL_TIMEOUT_SEC:g}s; killing")
            p.kill()
    # Surface the freshest model_elo so the launcher's stdout carries the result.
    jsonl = dirs["checkpoint_dir"] / "eval_results.jsonl"
    try:
        last = jsonl.read_text().strip().splitlines()[-1]
        rec = json.loads(last)
        elo = rec.get("eval/model_elo")
        ep = rec.get("eval_worker/epoch_evaluated")
        if elo is not None:
            print(f"  final eval: epoch={ep} model_elo={elo:.0f}")
    except (OSError, ValueError, IndexError):
        print(f"  final eval: no eval_results.jsonl line found at {jsonl}")


def launch_cell(cell: Cell, foreground: bool, resume_path: str | None = None,
                max_wall_secs: float = 0.0, final_eval: bool = False) -> None:
    dirs = cell_dirs(cell)
    dirs["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    dirs["records_dir"].mkdir(parents=True, exist_ok=True)
    dirs["log_dir"].mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    procs: list[tuple[str, subprocess.Popen]] = []

    def spawn(label: str, cmd: list[str], env_override: dict | None = None):
        log_path = dirs["log_dir"] / f"{label}.log"
        log_f = open(log_path, "a")
        e = dict(env)
        if env_override:
            e.update(env_override)
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=e, cwd=str(REPO_ROOT))
        procs.append((label, p))
        print(f"  spawned {label} pid={p.pid} -> {log_path}")

    print(f"=== launching cell {cell.name} ===")
    t_cmd = trainer_cmd(cell, dirs)
    if resume_path:
        t_cmd += ["--resume", str(resume_path)]
        print(f"  trainer resuming from {resume_path}")
    if max_wall_secs > 0:
        t_cmd += ["--max-wall-secs", str(max_wall_secs)]
        print(f"  trainer time-capped at {max_wall_secs:g}s (clean self-save on cap)")
    spawn("trainer", t_cmd)
    # Give the trainer a head start so workers find the initial weights file.
    time.sleep(2.0)
    for i in range(cell.n_workers):
        spawn(f"w{i}", worker_cmd(cell, dirs, f"w{i}", seed=1000 + i))
    spawn("eval", eval_cmd(cell, dirs), env_override={"GOMOKU_DEVICE": "cpu"})

    capped = max_wall_secs > 0
    if not foreground and not capped:
        # Fire-and-forget background launch (unchanged behaviour).
        print(f"backgrounded {len(procs)} processes for cell {cell.name}")
        print(f"  tail logs:  tail -F {dirs['log_dir']}/trainer.log")
        print(f"  stop cell:  pkill -f 'sweep-{cell.name}|sweep_runs/{cell.name}/'")
        return

    # Supervised mode: a foreground run OR a time-capped research slice. We stay
    # alive, watch the bundle, and tear it down cleanly. In capped mode the
    # trainer self-caps on an epoch boundary and exits clean (force-saving a
    # resumable latest.pt); a hard deadline (cap + grace) SIGTERMs a trainer
    # stuck in a long epoch — its handler still force-saves. This is
    # backgroundable via nohup: the lab launches it and is notified when the
    # launcher exits at the cap.
    trainer_proc = procs[0][1]  # trainer is always spawned first
    hard_deadline = (time.monotonic() + max_wall_secs + TRAINER_CAP_GRACE_SEC
                     if capped else None)
    if capped:
        print(f"supervising cell {cell.name}: {max_wall_secs:g}s cap "
              f"(+{TRAINER_CAP_GRACE_SEC:g}s grace), final_eval={final_eval}")
    else:
        print(f"foregrounded — Ctrl-C to stop all {len(procs)} processes")
    forced = False
    reason = "trainer exited"
    try:
        while True:
            time.sleep(2.0)
            dead = [lbl for lbl, p in procs if p.poll() is not None]
            if dead:
                reason = f"{', '.join(dead)} exited"
                break
            if hard_deadline and not forced and time.monotonic() >= hard_deadline:
                print(f"  [cap] trainer still in an epoch {TRAINER_CAP_GRACE_SEC:g}s "
                      f"past the {max_wall_secs:g}s cap; SIGTERM for a clean save")
                trainer_proc.terminate()
                forced = True
    except KeyboardInterrupt:
        reason = "KeyboardInterrupt"
    print(f"=== tearing down cell {cell.name} ({reason}) ===")
    _terminate_all(procs)
    if final_eval:
        _run_final_eval(cell, dirs)
    print(f"cell {cell.name} done")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="list defined cells and exit")
    p.add_argument("--cell", type=str, help="cell key from CELLS")
    p.add_argument("--foreground", action="store_true",
                   help="run in foreground (Ctrl-C stops all). Default backgrounds.")
    p.add_argument("--clean", action="store_true",
                   help="delete the cell's checkpoint+log dirs (does not stop a running cell)")
    p.add_argument("--epochs", type=int, default=None, help="override cell.epochs")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint (.pt) to resume the trainer from. "
                        "Passes through as --resume to gomoku.train. wandb run id "
                        "embedded in the checkpoint will continue the same wandb "
                        "timeline. Useful for swapping perf configs mid-run.")
    p.add_argument("--max-wall-secs", type=float, default=0.0,
                   help="Time-cap the slice: the trainer self-caps at this many "
                        "wall-seconds (epoch boundary), force-saves a resumable "
                        "latest.pt, and exits; this launcher supervises the bundle "
                        "and tears down workers + eval cleanly. 0 = run to the "
                        "cell's epoch budget (legacy fire-and-forget background). "
                        "This is how a training run becomes a research SLICE.")
    p.add_argument("--final-eval", action="store_true",
                   help="After teardown, run one eval_worker cycle (--max-cycles 1) "
                        "against the final published weights so eval_results.jsonl "
                        "ends on a fresh eval/model_elo. Pairs with --max-wall-secs.")
    args = p.parse_args()

    if args.list or not args.cell:
        list_cells()
        return

    if args.cell not in CELLS:
        raise SystemExit(f"unknown cell {args.cell!r}; one of {sorted(CELLS)}")
    cell = CELLS[args.cell]
    if args.epochs:
        cell.epochs = args.epochs

    if args.clean:
        clean_cell(cell)
        return

    launch_cell(cell, foreground=args.foreground, resume_path=args.resume,
                max_wall_secs=args.max_wall_secs, final_eval=args.final_eval)


if __name__ == "__main__":
    main()
