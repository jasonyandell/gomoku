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
    # 15x15 plumbing smoke (epic #21 Phase 3). Board size is process-level
    # config, not a Cell field yet: LAUNCH WITH GOMOKU_BOARD_SIZE=15, e.g.
    #   GOMOKU_BOARD_SIZE=15 python scripts/run_sweep.py --cell SMOKE15 \
    #       --foreground --max-wall-secs 90
    # Own name -> own dirs, so it never collides with 9x9 SMOKE artifacts.
    "SMOKE15": Cell("SMOKE15-board15-plumbing", sgd_per_game=1.0,
                    buffer_size=5_000, games_per_epoch=8,
                    size="tiny", stem_padding=1, n_simulations=30,
                    n_workers=2, games_per_batch=4,
                    temperature_moves=10, temperature_final=0.1,
                    save_buffer_every=100_000, epochs=100_000, wandb=False),
    # First real 15x15 training run (epic #21 Phase 4). Replicates the v8
    # champion recipe (= derby-v7-mate-discount: small 64x4 + global_pool +
    # value-discount 0.98 + gumbel-m16 + vcf-teacher + fixed 64 SGD steps/epoch)
    # on the 15x15 board. LAUNCH WITH GOMOKU_BOARD_SIZE=15. Starts FRESH (no
    # warm-start loader yet) so expect a cold ramp. buffer=400k (~6GB at 15x15,
    # no bit-packing yet -> raise to 3M once #25 lands); wave=64 = the
    # dispatch-bound free training-regime point at 15x15. Continuous + resumable
    # (epochs huge; latest.pt embeds the buffer). See
    # wiki/topics/15x15-training-campaign.md.
    #
    # VCF-teacher DROPPED from the cold-start seed (staged decision, 2026-06-13
    # smoke). The 9x9 defaults (depth16/200k nodes) cost 3-9 s/game on a
    # wide-open 15x15 board; even capped to depth10/12k it was ~5.4 s/game,
    # starving generation. WITHOUT it: ~1.8 s/game/worker (~4.4 games/s, reuse
    # ~1.3 = trainer paced, not starved). On a cold board forced wins are rare
    # so the teacher's early benefit is low while its cost is highest -> defer.
    # A 15x15-tuned vcf-teacher (early-game skip + small per-move budget) is a
    # planned derby contestant once the net matures (readiness-audit S3, epic
    # #21). Kept cheap winning levers: global_pool + value-discount + gumbel.
    "G15-seed": Cell("G15-seed-v8recipe-board15", sgd_per_game=1.0,
                buffer_size=400_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # G15-96x8: the CAPACITY plateau-breaker (epic #21 derby). IDENTICAL recipe
    # to G15-seed but size="96x8" (96ch x 8blk, ~1.55M params at 15x15 vs the
    # 64x4's ~0.44M). Launched via --resume sweep_runs/g15_96x8_seed.pt, which
    # is the G15-seed champion GROWN function-preservingly (net2net widen+deepen,
    # output-equiv ~1.5e-4) so it STARTS at the champion's ~67%-vs-Rapfi strength
    # and trains into the extra capacity — no cold-start fast-attack collapse.
    # The bet: more capacity breaks the 64x4 recipe's ~67%-vs-Rapfi plateau. The
    # bigger net is ~2.3x eval cost (wave=64, dispatch-bound bench), so epochs
    # are ~2.3x slower; buffer kept at 400k (bit-packing not yet flood-validated).
    "G15-96x8": Cell("G15-96x8-grown-board15", sgd_per_game=1.0,
                buffer_size=400_000, games_per_epoch=64,
                size="96x8", stem_padding=1, n_simulations=100,
                n_workers=8, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # G15-96x8-redo (2026-06-15): RE-TRAIN from the GOOD net2net seed to diagnose
    # the silent self-play regression. Context: the original G15-96x8 run trained
    # 400+ epochs and was crowned 15×15 champion — but head-to-head (2026-06-15)
    # reveals it is CATASTROPHICALLY REGRESSED: its own untrained net2net seed
    # (g15_96x8_seed.pt) BEATS the trained e499 champion 40-0, and that seed
    # TIES the 64×4-e909 (50%). So 400 epochs of self-play made the net 40-0 WORSE
    # than its starting point. This was INVISIBLE to every internal signal throughout:
    # plies 30-48, vl 0.17-0.25, internal-ladder win-rates 85-100% — all healthy.
    # The net2net grow itself was valid (output deviation <1e-4); the regression is
    # in the TRAINING, not the grow step. Two hypotheses:
    #   H1 CAPACITY: a re-run with the same recipe will produce a strong 96×8 net
    #      (the original run was an unlucky outlier / one-off data-flow artifact).
    #   H2 RECIPE: the v8 recipe systematically regresses 96×8 via self-play
    #      distribution drift — a second run will also crater vs its seed.
    # Evaluation: head-to-head vs the FROZEN seed and vs 64×4-e909, NOT Rapfi
    # (the Rapfi yardstick is broken — §8 of alphazero-lessons-15x15-gomoku.md).
    # A re-run that BEATS its seed (H1) rebuilds confidence in capacity monotonicity
    # and unlocks 96×8 as a valid step on the capacity ladder. A re-run that REGRESSES
    # again (H2) pins the recipe as the cause and points to recipe surgery.
    # Seed: sweep_runs/g15_96x8_seed.pt (the good net2net grown checkpoint).
    # BYTE-IDENTICAL to G15-96x8 — only the run-dir name changes.
    "G15-96x8-redo": Cell("G15-96x8-redo-board15", sgd_per_game=1.0,
                buffer_size=400_000, games_per_epoch=64,
                size="96x8", stem_padding=1, n_simulations=100,
                n_workers=8, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # G15-96x8-deepgen (2026-06-14, #27): the SEARCH-DEPTH axis. Every closed
    # 15x15 experiment (capacity 64x4->96x8->128x10, data 400k->1.5M, epochs
    # ->e617) held n_simulations=100 FIXED. The deep-TC plateau (96x8 champion
    # = 75 fast / 69 deep, n=16) is a SEARCH symptom: our fixed-100-sim net vs a
    # far-deeper Rapfi@5000ms. BYTE-IDENTICAL to G15-96x8 EXCEPT n_simulations
    # =200 (deeper self-play search -> richer policy/value targets). Warm-start:
    #   --resume sweep_runs/g15_champion_96x8_e499.pt  (strongest 96x8, 400k buf)
    # Q: does richer self-play search break the 69% deep-TC plateau? At 200 sims
    # workers make ~half the games/epoch but the 400k buffer still turns over
    # ~every ~22 epochs (trainer is train-bound ~25-30s/epoch, gen only ~2-3s, so
    # epoch pace is ~unchanged). One lever changed (sims); eval n>=16 BOTH tiers.
    "G15-96x8-deepgen": Cell("G15-96x8-deepgen-board15", sgd_per_game=1.0,
                buffer_size=400_000, games_per_epoch=64,
                size="96x8", stem_padding=1, n_simulations=200,
                n_workers=8, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # G15-96x8-cont100 (2026-06-15, #27 control): the CONTROL ARM for the deepgen
    # result. deepgen (200-sim self-play, warm from champion) REGRESSED: at sims=100
    # it lost 0-40 to the frozen champion, at sims=200 it was 50-50 — search-
    # SPECIALIZED, not stronger, and the shallow Rapfi yardstick masked it (83%).
    # Q: is that the 200-sim change, or does ANY continued training degrade the
    # champion? cont100 = BYTE-IDENTICAL to deepgen EXCEPT n_simulations=100 (the
    # champion's own regime). Same warm-start (--resume g15_champion_96x8_e499.pt),
    # same recipe. Measure by head-to-head vs the FROZEN champion (yardstick-free):
    #   >50% = continued 100-sim training IMPROVES the champion (reopens "saturated")
    #   ~50% = champion saturated at 100 sims; deepgen's regression was the 200 sims
    #   <50% = warm-continuation itself degrades (fragile peak)
    "G15-96x8-cont100": Cell("G15-96x8-cont100-board15", sgd_per_game=1.0,
                buffer_size=400_000, games_per_epoch=64,
                size="96x8", stem_padding=1, n_simulations=100,
                n_workers=8, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # G15-96x8-bigbuf: the DATA experiment (2026-06-13). The 128x10 capacity step
    # OVERSHOT — the 3.3M net got WORSE on the 400k buffer (overfit; aggregate
    # 37.5% @1000ms vs the 96x8's 75%). FINDING: capacity must be matched by
    # DATA. So test the data axis on the CONFIRMED-GOOD 96x8 (75/88, still
    # improving when swapped out): same recipe but a 3.75x-bigger BIT-PACKED
    # buffer (1.5M, #25; the 96x8 is fast enough that 8 workers fill it in
    # ~40min). Q: does more data push the sweet-spot net past 75/88 AND avoid
    # the overfit that bit the 128x10? Launch --resume from g15_champion_96x8_e499.pt.
    "G15-96x8-bigbuf": Cell("G15-96x8-bigbuf-board15", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="96x8", stem_padding=1, n_simulations=100,
                n_workers=8, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64", "--pack-buffer"]),
    # G15-128x10: SECOND capacity step (epic #21 derby). The 96x8 confirmed a
    # capacity win vs Rapfi (1000ms 62->75%, 5000ms 75->88%, both > 64x4's
    # ~67%, broadening with training), so ride the thesis. IDENTICAL recipe to
    # G15-96x8 but size="large" (128ch x 10blk, ~3.3M params). Launched
    # --resume sweep_runs/g15_128x10_seed.pt = the 96x8 e499 champion GROWN
    # function-preservingly (net2net, output-equiv 1.8e-4) so it STARTS at the
    # 96x8's strength and trains into the extra capacity. ~4.6x the 64x4 eval
    # cost → epochs SLOW (~50-70s); buffer kept 400k.
    "G15-128x10": Cell("G15-128x10-grown-board15", sgd_per_game=1.0,
                buffer_size=400_000, games_per_epoch=64,
                size="large", stem_padding=1, n_simulations=100,
                # n_workers=4 (NOT 8): the 3M-param trainer is slow (~60s+/epoch),
                # so 8 workers FLOODED it — per-epoch ingest cost ran away
                # (62→313s, new/epoch 168→512) on 2026-06-13. Fewer workers
                # decouple inflow from the slow big-net trainer. Big nets need
                # fewer workers (gen-flood lesson, memory feedback_gen_flooding).
                n_workers=4, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # G15-128x10-bigbuf: CAPACITY x DATA, jointly. Two findings set this up:
    # (1) 128x10 on the 400k buffer OVERSHOT/overfit (37.5% @1000ms vs 96x8's
    #     75%) — capacity outran data; (2) the 96x8 on a 1.5M buffer did NOT
    #     improve (e597: 50/88 vs the e499 champion's 75/88 — deep-TC IDENTICAL
    #     at 88%, the 96x8's capacity ceiling; data alone wasn't the lever).
    # The clean conclusion: 96x8 is capacity-bound, 128x10 is data-bound. So
    # pair them — 128x10 (size="large") WITH the 1.5M bit-packed buffer, resumed
    # from the net2net seed (starts at 96x8 strength, no overfit baggage) so it
    # trains INTO the extra capacity on enough data to not overfit. Q: does
    # capacity+data break past the 96x8's 88% deep-TC ceiling? n_workers=4 (the
    # gen-flood remedy for the slow 3.3M trainer); --pack-buffer (the MPS-fixed
    # bit-pack; 1.5M board-15 positions need it). Launch:
    #   --resume sweep_runs/g15_128x10_seed.pt
    "G15-128x10-bigbuf": Cell("G15-128x10-bigbuf-board15", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="large", stem_padding=1, n_simulations=100,
                n_workers=4, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64", "--pack-buffer"]),
    # G15-defense (#36, sliding-derby Lap 1): the DEFENSE-TEACHER cell — the proven-
    # needed white-side fix. Diagnosis (#33) is closed: the champion's white-side
    # collapse vs strong attackers (eval502: 0-6 white vs zetor17 while 6-0 as black)
    # is a TRAINING gap, not a search gap — FPU=0.45 and 4x search (sims 200->800)
    # both did nothing (0-6 / 0-4 either way). No eval lever helps -> the net must be
    # TAUGHT to defend. BYTE-IDENTICAL to the reigning champion G15-128x10-bigbuf
    # (the eval502 lineage: 128x10 + global_pool + value-discount 0.98 + gumbel +
    # 1.5M bit-packed buffer + fixed 64 SGD steps/epoch + n_workers=4) in EVERY Cell
    # field, with the ONLY deltas on the offensive/defensive teacher seam:
    #   (1) --vct-teacher REPLACES --vcf-teacher: VCT (Victory-by-Continuous-Threes)
    #       is the strict SUPERSET of VCF (per #36/#18) — it proves every VCF forced
    #       win PLUS wins that need forcing threes, stamping MORE positions with exact
    #       mate labels. (Note bigbuf carries no offensive teacher today, the cold-
    #       start lineage having dropped vcf, so this ADDS the offensive teacher in
    #       its VCT/superset form.) The continuous-threes tree fans out on the
    #       defender side, so the per-move solve MUST be aggressively capped on the
    #       gen hot path: --vct-max-depth 4 / --vct-max-nodes 800 (the proven 9x9
    #       derby-x-vct budget; the loose library defaults depth7/20k starved self-
    #       play to ZERO games in ~50s, bead derby-b6r). A wide-open 15x15 position
    #       bails to no-forced-win/hit_cap almost instantly while short tactical wins
    #       (open-four mate, double-three fork) are still proven within the cap.
    #   (2) --defense-teacher ADDED: the VALUE-ONLY mirror of the offensive teacher —
    #       when the OPPONENT has a proven forced win against the side to move, relabel
    #       the recorded value target to -1 ("you were already lost, defend earlier").
    #       POLICY target untouched (defense is non-unique). Gen-cost-gated: skips
    #       positions where the offensive teacher already fired + a cheap opponent-
    #       four-threat pre-scan, so quiet positions cost zero extra solver calls.
    # Teacher flags ride extra_worker_args (no Cell fields; same plumbing as the 9x9
    # derby-x-vct / derby-x-defense pair). WARM-START from the champion at launch:
    #   python scripts/run_sweep.py G15-defense \
    #       --resume sweep_runs/g15_128x10_bigbuf_eval502.pt \
    #       --max-wall-secs <one chunk> --final-eval
    # (--resume is a CLI-time arg, like every G15 cell — not a Cell field; the wandb
    # run id embedded in eval502 continues the same timeline.) JUDGE via the panel
    # arena (#34), NOT training-time evals: success = white-loss-rate vs zetor17/
    # lookahead:6 drops AND elo_gap shrinks with NO black regression (#18 acceptance).
    "G15-defense": Cell("G15-defense-board15", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="large", stem_padding=1, n_simulations=100,
                n_workers=4, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                # 2026-06-16: champion + --defense-teacher ONLY (clean single
                # variable — eval502/bigbuf had no offensive teacher), with the VCF
                # solve budget HARD-CAPPED. --vct-teacher dropped (clean experiment;
                # also the more expensive teacher). ROOT-CAUSE RESOLVED by controls:
                # --defense-teacher at the DEFAULT 200k-node VCF budget STARVES 15x15
                # generation — 0 games in 6 min, workers 100% CPU. Isolated decisively:
                # a no-teacher control genned eval502 at ~2.6 s/game, and the SAME
                # config + --vcf-max-nodes 2000 --vcf-max-depth 10 genned at ~3.1 s/game
                # (gen rescued). So the per-move defense VCF solve, uncapped, is the
                # gen-killer; the cap bounds it while still proving the SHORT forced
                # losses "defend earlier" needs (deep 200k-node proofs are unnecessary
                # for the draw/loss boundary). The 9x9 teachers' default budgets do NOT
                # transfer to 15x15's wider branching — always cap the solve on the gen
                # hot path here. (Evidence: /tmp/{noteacher,capdef}_ctrl.log, 2026-06-16.)
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98",
                                   "--defense-teacher",
                                   "--vcf-max-nodes", "2000", "--vcf-max-depth", "10"],
                extra_train_args=["--sgd-steps-per-epoch", "64", "--pack-buffer"]),
    # G15-vcf: the planned 15x15-tuned vcf-teacher derby contestant (epic #21
    # readiness-audit S3). BYTE-IDENTICAL to G15-seed in every Cell field EXCEPT
    # it re-enables --vcf-teacher with a 15x15-appropriate per-move budget. This
    # is the one-lever-changed sibling of G15-seed: race it against the base once
    # the net matures and starts producing real threats.
    #
    # Budget rationale — why depth 8 / nodes 2500 (NOT the 9x9 defaults
    # depth16/200k, NOR even depth10/12k):
    #   The G15-seed comment records the measured gen cost on a wide-open 15x15
    #   board: depth16/200k = 3-9 s/game; depth10/12k = ~5.4 s/game. The solver
    #   runs at EVERY recorded ply and, on a quiet board, burns its full NODE
    #   budget proving 'no win' before bailing — so per-game cost is dominated by
    #   max_nodes x plies-solved, not depth. To hit the ~0.5 s/game vcf-overhead
    #   target we cut the node budget ~5x below the 12k that already cost 5.4 s,
    #   to 2500 (=> a quiet-position solve bails ~5x faster), and trim depth
    #   16 -> 8 (a depth-8 attacker line = our four + their block, x4 — covers the
    #   short forced wins that actually appear in self-play; longer proofs are
    #   rare and the depth cap just makes the wide-open bail quicker).
    #   The solver (gomoku/vcf.py) is sound at ANY budget: a cap-hit returns
    #   'no forced win' (never a false positive), so a tight cap only trades
    #   recall of LONG forced wins for generation throughput — exactly the
    #   tradeoff we want on a maturing net where short tactics dominate.
    #
    # NO early-game ply-gate is applied: as of 2026-06-13 there is NO ply-gate /
    # 'skip the first N plies' FLAG in selfplay_worker / self_play. The only
    # gen-cost guards that exist are --vcf-max-depth/-nodes (tuned here) and
    # --defense-teacher's INTERNAL already-fired + four-threat pre-scan (not a
    # reusable ply-gate, and only on the defensive teacher). A cheap ply-gate
    # (skip the solver for plies < ~10, where a forced win is impossible on a
    # near-empty 15x15) would cut early-game cost further essentially for free,
    # but it is a SEPARATE code task (filed as a recommendation), NOT built here.
    # Budget tuning alone is expected to land near the target because the 2500-
    # node cap already makes the early-game quiet solves bail fast.
    "G15-vcf": Cell("G15-vcf-v8recipe-board15", sgd_per_game=1.0,
                buffer_size=400_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98",
                                   "--vcf-teacher",
                                   "--vcf-max-depth", "8",
                                   "--vcf-max-nodes", "2500"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # G15-wdl: the 15x15 WDL (win/draw/loss) value-representation derby contestant
    # (epic #21; sibling of the 9x9 derby-x-wdl). BYTE-IDENTICAL to G15-seed in
    # every Cell field EXCEPT it adds the ONE lever --value-head wdl on BOTH the
    # train + worker args. The value head emits 3 logits over {win,draw,loss}
    # trained with cross-entropy (target = the WDL generalization of the scalar z:
    # (relu(z), 1-|z|, relu(-z))); every scalar consumer (MCTS leaf eval, the
    # anchor ladder) sees the DERIVED v=P(win)-P(loss) computed in model.forward(),
    # so the native C MCTS hot path and self-play generation are byte-identical —
    # the C engine never sees the 3 logits, only the scalar that crosses the
    # evaluate_planes() boundary. The --value-head wdl on the worker is a
    # consistency assert; the model's head comes from the checkpoint config.
    #
    # WARM START: the conv tower warm-starts from the 9x9 scalar champion via
    # `scripts/warmstart_15x15.py --target-value-head wdl` (the value head is
    # FRESH regardless — a scalar source has value_fc2, a WDL target has
    # value_wdl_fc, so the value-head weights never transfer by name). Launch
    # with `--resume <wdl_warmstart_seed.pt>` exactly like G15-seed's swap to the
    # warm-started seed (wandb qvr95npw). So: WARM tower (94.6% param transfer
    # from the 9x9 champion) + FRESH 3-logit WDL value head — the head relearns
    # the WDL representation from scratch while inheriting all board-pattern
    # features. NOTE the value-discount target (0.98^plies) flows through the
    # SAME WDL target builder, so this is still exactly ONE lever vs G15-seed:
    # the value REPRESENTATION (scalar tanh -> categorical WDL).
    "G15-wdl": Cell("G15-wdl-v8recipe-board15", sgd_per_game=1.0,
                buffer_size=400_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, wave_size=64, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.25,
                temperature_moves=30, temperature_final=0.1,
                sgd_per_position=0.0025, save_buffer_every=100, save_every=5,
                ema_tau=0.99, grad_accum_steps=4,
                opponent_mix_recent=0.4, opponent_mix_history=0.1,
                opponent_mix_recent_window=100,
                weights_poll_min_sec=2.0, weights_poll_max_sec=8.0,
                epochs=1_000_000, random_opening_moves=0,
                global_pool=True,
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16",
                                   "--value-discount", "0.98",
                                   "--value-head", "wdl"],
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--value-head", "wdl"]),
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
    # === v9 SCALE-LADDER (2026-05-27, Jason-gated: "go larger models, keep the evals,
    # target 100% = win-all-black/never-lose-white across heuristic+lookahead2+lookahead4").
    # The v8 champion saturated ANCHORED elo (~1700) but NOT the 100% target — it's 4-0-6
    # (six DRAWS) as black vs lookahead4. Hypothesis: the small net (64x4) lacks capacity to
    # convert those draws to wins. These clone the champion recipe VERBATIM and change ONLY
    # the net size (config-only: size preset in model.py SIZES). small(64x4)=champion baseline,
    # medium(96x6), large(128x10, ~= AlphaGomoku's 128x8). Judge by lookahead4 black-win-rate
    # (the binding gap), NOT anchored elo (saturated). Fresh-start (bigger net != small ckpt shape).
    "derby-v9-medium": Cell("derby-v9-medium", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="medium", stem_padding=1, n_simulations=100,
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
    # 'x-medium-signal' = CAPACITY-UNLOCK probe (2026-05-28, Jason: "activate an old lever that
    # didn't do much on its own but might compound with medium where small didn't have the juice").
    # = derby-v9-medium (champion recipe @ 96x6) + the v4 KataGo AUX-SUPERVISION lever (opp-reply
    # policy head + per-cell ownership head, both @0.15). Aux heads were MIDDLING at small (v4
    # 'signal') — they add EXTRA representational load (2 heads), so a 64x4 net starves its main
    # heads to feed them; a 96x6 net has the spare capacity to exploit the extra signal-per-position
    # (the scarce-near-opening-positions problem, az-at-scale-vs-laptop). Config-only (flags exist,
    # byte-identical-off, model.py:40/47). Tests whether aux supervision COMPOUNDS at scale.
    "derby-x-medium-signal": Cell("derby-x-medium-signal", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="medium", stem_padding=1, n_simulations=100,
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
                                   "--value-discount", "0.98", "--record-aux", "--record-ownership"],
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--aux-opponent-reply-weight", "0.15", "--aux-ownership-weight", "0.15"]),
    "derby-v9-large": Cell("derby-v9-large", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="large", stem_padding=1, n_simulations=100,
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
    # v9-small = the FRESH seed-0 small-net CONTROL for the v9 net-capacity ladder.
    # Byte-identical recipe to derby-v7-mate-discount (the v8 champion) but its OWN
    # output dir (sweep_runs/derby-v9-small/) so it starts from scratch — the
    # apples-to-apples Δelo-RATE baseline against v9-medium/large from identical starts.
    # (Reusing derby-v7-mate-discount would RESUME the matured 1811 champion, not fresh.)
    "derby-v9-small": Cell("derby-v9-small", sgd_per_game=1.0,
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
    # 'x-vct' = extend the exact OFFENSIVE teacher from VCF to VCT (bead derby-6us /
    # derby-rxf). VERBATIM clone of derby-v7-mate-discount (the reigning champion:
    # gumbel-root + value-discount 0.98 + global-pool + gumbel-m 16 + the 0.4/0.1
    # league mix + sgd-steps-per-epoch 64) with ONE lever changed: --vcf-teacher ->
    # --vct-teacher. VCT (Victory-by-Continuous-Threes) is a strict SUPERSET of VCF:
    # it proves every VCF forced win PLUS wins that need forcing threes, so it stamps
    # MORE positions with exact mate labels (winning move + mate-discounted value).
    # The flag REPLACES (does not augment) --vcf-teacher on the offensive seam — the
    # deeper solver supersedes the shallower; running both would be redundant work.
    # Gen-cost: the continuous-threes tree fans out on the defender side, so the per-move
    # solve MUST be aggressively bounded on the generation hot path. Derby v8 raced this
    # cell with the loose library defaults (depth 7 / nodes 20k) and got ZERO games /
    # buf=0 in ~50s — the solve never returned, fully starving self-play (bead derby-b6r).
    # Fix: pin an AGGRESSIVE per-move cap (--vct-max-depth 4 / --vct-max-nodes 800) so a
    # wide-open position bails to no-forced-win/hit_cap almost instantly and generation
    # runs at ~normal rate, while short tactical wins (open-four mate, double-three fork)
    # are still proven within the cap. Lane-isolated outputs under sweep_runs/derby-x-vct/.
    "derby-x-vct": Cell("derby-x-vct", sgd_per_game=1.0,
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
                extra_worker_args=["--gumbel-root", "--gumbel-m", "16", "--vct-teacher",
                                   "--vct-max-depth", "4", "--vct-max-nodes", "800",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # 'x-defense' = the exact DEFENSIVE VCF teacher (bead derby-1xf). VERBATIM clone
    # of derby-v7-mate-discount (the reigning champion: gumbel-root + vcf-teacher +
    # value-discount 0.98 + global-pool + gumbel-m 16 + the 0.4/0.1 league mix +
    # sgd-steps-per-epoch 64) with ONE lever added: --defense-teacher on the worker.
    # It is the VALUE-ONLY mirror of --vcf-teacher: where the offensive teacher proves
    # a forced WIN for the side to move and stamps +1, the defensive teacher proves a
    # forced win for the OPPONENT against the side to move and relabels the value target
    # to -1 (the position is already lost -> "you should have defended earlier"). The
    # POLICY target is left untouched (defense is non-unique). Gen-cost-gated: it skips
    # positions where --vcf-teacher already fired and runs a cheap opponent-four-threat
    # pre-scan before the (swapped-plane) solve, so quiet positions cost zero solver
    # calls. Correlated with the VCF teacher (same solver), so it is tested STACKED on
    # the champion (which has VCF on) -- that's intended. Lane-isolated outputs under
    # sweep_runs/derby-x-defense/.
    "derby-x-defense": Cell("derby-x-defense", sgd_per_game=1.0,
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
                                   "--value-discount", "0.98", "--defense-teacher"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # 'x-soft-policy' = the KataGo soft-policy auxiliary target (bead derby-79l).
    # VERBATIM clone of derby-v7-mate-discount (the reigning champion: gumbel-root
    # + vcf-teacher + value-discount 0.98 + global-pool + gumbel-m 16 + the 0.4/0.1
    # league mix + sgd-steps-per-epoch 64) with ONE lever added: --soft-policy-weight
    # 0.15 on the TRAIN side only (worker args UNCHANGED). The same policy logits are
    # trained against a SECOND target -- a 4th-root temperature-flattened copy of the
    # already-recorded policy target `pi` (KataGo's exact transform) -- re-injecting
    # the runner-up structure the sharp completed-Q target drops under our 60-70% draw
    # regime. NO new head; the soft target is a pure TRAINER transform of an already-
    # recorded vector, so generation cost is ZERO and the gen hot path (_mcts_native.c,
    # selfplay_worker.py) is byte-identical. 0.15 = conservative start (KataGo's scale
    # is ~O(0.1)) so the sharp target still dominates. The matched control is the
    # scalar champion. Lane-isolated outputs under sweep_runs/derby-x-soft-policy/.
    "derby-x-soft-policy": Cell("derby-x-soft-policy", sgd_per_game=1.0,
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
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--soft-policy-weight", "0.15"]),
    # 'x-gumbel-m8' = Gumbel-m sweep (gomocup-AZ survey 2026-05-27). VERBATIM clone
    # of derby-v7-mate-discount (the reigning champion: gumbel-root + vcf-teacher +
    # value-discount 0.98 + global-pool) with ONE lever: --gumbel-m 16 -> 8. The
    # Gumbel top-k/Sequential-Halving breadth has been frozen at the v3 default (16)
    # and NEVER swept in any derby (red-team flag). m=8 focuses the n=100 sims on
    # fewer root candidates -> more sims/candidate -> sharper completed-Q targets;
    # tests whether narrower-but-deeper root search beats the default breadth in our
    # drawish regime. Config-only, byte-identical to the champion except this flag.
    "derby-x-gumbel-m8": Cell("derby-x-gumbel-m8", sgd_per_game=1.0,
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
                extra_worker_args=["--gumbel-root", "--gumbel-m", "8", "--vcf-teacher",
                                   "--value-discount", "0.98"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # 'x-mish' = the Mish activation-function lever (bead derby-sib). VERBATIM clone
    # of derby-v7-mate-discount (the reigning base recipe: gumbel-root + vcf-teacher +
    # value-discount 0.98 + global-pool + the 0.4/0.1 league mix + sgd-steps-per-epoch
    # 64) with ONE lever added: --activation mish. Every residual-tower nonlinearity
    # (stem + ResBlock/GlobalPoolResBlock) becomes nn.Mish (smooth self-gated
    # x*tanh(softplus(x)), KataGo's newer default) instead of nn.ReLU. ZERO added
    # params / IDENTICAL state_dict keys (Mish is parameter-free), so the lever is a
    # genuinely orthogonal ARCHITECTURE axis vs the value-rep (x-wdl), search-breadth
    # (x-gumbel-m8), and policy-signal (x-soft-policy) cells. It lives ENTIRELY in
    # model.py: the native-C MCTS engine does tree ops and calls BACK into the PyTorch
    # evaluator for the forward, so the generation hot path (_mcts_native.c) is
    # byte-identical -- no C kernel change. --activation mish rides on BOTH train +
    # worker args for cell symmetry (the worker flag is a consistency assert; the
    # model's activation comes from the checkpoint config). FRESH-START lane: Mish
    # keeps identical tensor SHAPES (a checkpoint loads) but ReLU-trained weights
    # behave differently in a Mish tower, so it must NOT warm-start the ReLU champion
    # -- the runner launches it fresh and judges on climb-RATE like the other new-lever
    # lanes. Lane-isolated outputs under sweep_runs/derby-x-mish/.
    "derby-x-mish": Cell("derby-x-mish", sgd_per_game=1.0,
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
                                   "--value-discount", "0.98", "--activation", "mish"],
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--activation", "mish"]),
    # 'x-wdl' = the WDL (win/draw/loss) value-representation lever (bead derby-cgf).
    # VERBATIM clone of derby-v7-mate-discount (the reigning base recipe: gumbel-root
    # + vcf-teacher + value-discount 0.98 + global-pool + the 0.4/0.1 league mix +
    # sgd-steps-per-epoch 64) with ONE lever added: --value-head wdl. The net emits
    # 3 logits over {win,draw,loss} trained with cross-entropy; the scalar consumers
    # (MCTS leaf eval, anchor ladder) see the DERIVED v=P(win)-P(loss), so the C MCTS
    # hot path and self-play generation are byte-identical. The value-discount and
    # VCF-stamp targets are re-expressed natively in WDL inside the trainer (the
    # scalar z is mapped to (relu(z), 1-|z|, relu(-z))), so this stays ONE lever:
    # the value REPRESENTATION. --value-head wdl rides on BOTH train + worker args
    # for cell symmetry (the worker flag is a consistency assert; the model's value
    # head comes from the checkpoint config). The matched control is the existing
    # scalar champion. Lane-isolated outputs under sweep_runs/derby-x-wdl/.
    "derby-x-wdl": Cell("derby-x-wdl", sgd_per_game=1.0,
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
                                   "--value-discount", "0.98", "--value-head", "wdl"],
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--value-head", "wdl"]),
    # 'x-hlgauss' = the HL-Gauss distributional value head (Derby 'x-hlgauss',
    # bead derby-tn4). VERBATIM clone of derby-v7-mate-discount (the reigning
    # champion: gumbel-root + vcf-teacher + value-discount 0.98 + global-pool +
    # the 0.4/0.1 league mix + sgd-steps-per-epoch 64) with ONE lever added:
    # --value-head hlgauss --hlgauss-bins 51 --hlgauss-sigma 0.05. The net
    # emits 51 logits over evenly-spaced bin centers in [-1, 1], trained with
    # cross-entropy against a Gaussian-smoothed target N(z, sigma^2); the scalar
    # consumers (MCTS leaf eval, anchor ladder) see the DERIVED v=sum(prob*bin),
    # so the C MCTS hot path and self-play generation are byte-identical to
    # scalar/WDL. The value-discount + VCF-stamp + draw-contempt targets all
    # reshape the scalar z and flow through the same target builder
    # (hlgauss_target_from_z), so this stays ONE lever: the value REPRESENTATION,
    # generalized from WDL's 3 bins to N=51 evenly-spaced bins for finer
    # resolution in drawish positions (Farebrother+2024, arxiv 2403.03950).
    # --value-head hlgauss + --hlgauss-bins/--hlgauss-sigma ride on BOTH
    # train + worker for cell symmetry (the worker flags are consistency
    # asserts; the model's value head + bins/sigma come from the checkpoint
    # config). FRESH-START: an HL-Gauss checkpoint is NOT loadable by scalar/WDL
    # builds (different FC shape) — the trainer/worker hard-error on mismatched
    # load. Lane-isolated outputs under sweep_runs/derby-x-hlgauss/.
    "derby-x-hlgauss": Cell("derby-x-hlgauss", sgd_per_game=1.0,
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
                                   "--value-discount", "0.98",
                                   "--value-head", "hlgauss",
                                   "--hlgauss-bins", "51",
                                   "--hlgauss-sigma", "0.05"],
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--value-head", "hlgauss",
                                  "--hlgauss-bins", "51",
                                  "--hlgauss-sigma", "0.05"]),
    # === AGGRESSIVE COMBINATION phase (2026-05-27, Jason: "combine the best we found
    # so far into new runs") — v4-style multi-lever stacks (deliberately NOT one-lever).
    # Base = the champion (vcf+global-pool+value-discount 0.98) + the v8 survey's SOLE
    # keeper (WDL value head, H2H +35 replicated). Stacked with the other VALIDATED
    # winners: buffer-recency (v8 buffer-comp +90) and vcf-deep (v5 deeper exact-mate
    # solver +44). A clean 2x2 over {recency, vcf-deep} on the WDL base (derby-x-wdl =
    # WDL alone, derby-x-wdl-recency = +recency, both already racing). adjudicate
    # (--max-plies 45) deliberately EXCLUDED — it won v6 standalone but REGRESSED when
    # stacked in v8. The round-robin over these tells which combination is the best player.
    #
    # 'x-wdl-deep' = champion + WDL + vcf-deep (value-representation x exact-mate-label depth).
    "derby-x-wdl-deep": Cell("derby-x-wdl-deep", sgd_per_game=1.0,
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
                                   "--value-discount", "0.98", "--value-head", "wdl",
                                   "--vcf-max-depth", "32", "--vcf-max-nodes", "500000"],
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--value-head", "wdl"]),
    # 'x-wdl-max' = the MAXIMAL stack: champion + WDL + recency + vcf-deep (all validated
    # training-side winners at once — the "best shot at a great gomoku player" bet).
    "derby-x-wdl-max": Cell("derby-x-wdl-max", sgd_per_game=1.0,
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
                                   "--value-discount", "0.98", "--value-head", "wdl",
                                   "--vcf-max-depth", "32", "--vcf-max-nodes", "500000"],
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--value-head", "wdl", "--buffer-recency-frac", "0.5"]),
    # 'x-crossgame' = the cross-game value sidecar (Derby 'position-stats',
    # bead derby-eft). VERBATIM clone of derby-v7-mate-discount (the reigning
    # base recipe: gumbel-root + vcf-teacher + value-discount 0.98 + global-pool
    # + the 0.4/0.1 league mix + sgd-steps-per-epoch 64) with ONE lever added:
    # the trainer aggregates the (value-discounted) returns of ALL games through
    # each canonical position into a lane-isolated single-writer store, then
    # relabels the sampled z with a confidence-weighted, visit-gated blend.
    # NOTE the flags live in extra_train_args, NOT extra_worker_args: the store
    # is TRAINER-OWNED (single writer; self-play workers are unchanged), and the
    # selfplay_worker arg parser would hard-error on these flags. The store path
    # is lane-isolated so it cannot touch any other contestant -> a clean A/B.
    "derby-x-crossgame": Cell("derby-x-crossgame", sgd_per_game=1.0,
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
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--cross-game-value",
                                  "--cross-game-store",
                                  "sweep_runs/derby-x-crossgame/position_stats.pkl",
                                  # bead derby-4bq: OPENING-ONLY cap. Only ply<10
                                  # positions are aggregated -> the store can't
                                  # blow past the finite set of distinct openings,
                                  # so save() stays cheap forever (kills the convex
                                  # ~1s->~48s/epoch runaway) and the de-noised value
                                  # signal lands exactly on opening convergence.
                                  "--cross-game-max-ply", "10"]),
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

    # 'disc-recency' = stack the two LEVERS THAT CLIMBED in v8 (value-discount 0.98,
    # the leader; + buffer-recency-frac 0.5, the curator that was climbing) on the
    # vcf+global-pool base. Replaces 'stack' (--max-plies 45), which regressed below
    # peak while its no-truncation twin mate-discount climbed — truncation is a drag.
    "derby-x-disc-recency": Cell("derby-x-disc-recency", sgd_per_game=1.0,
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
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--buffer-recency-frac", "0.5"]),

    # 'vdisc-097' = mate-discount's recipe with a SHARPER value-discount (0.97 vs
    # 0.98). The v8 H2H verdict named value-discount THE key lever; this probes
    # which way the optimum lies (sharper mate-urgency vs the 0.98 default).
    # Replaces 'control' (vcf+gp baseline) — its result is locked (anchored 1476,
    # H2H -81) and its peak.pt remains the saved round-robin anchor.
    "derby-x-vdisc-097": Cell("derby-x-vdisc-097", sgd_per_game=1.0,
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
                                   "--value-discount", "0.97"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),

    # 'vdisc-099' = MILDER value-discount (0.99). Brackets the discount optimum from
    # above: with vdisc-097 (0.97) on the board, 0.97/0.98/0.99 is a clean gradient.
    # Replaces 'buffer-comp' (pure recency) — plateaued (csnp 8), result locked (+82
    # H2H, recency proven additive; peak.pt kept as anchor) and disc-recency carries
    # recency forward better.
    "derby-x-vdisc-099": Cell("derby-x-vdisc-099", sgd_per_game=1.0,
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
                                   "--value-discount", "0.99"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),

    # 'wdl-recency' = STACK THE v8 KEEPER (WDL value head, RR3/RR4 #2 @ +35) with the
    # proven-additive recency curator (buffer-comp +82/+90 in earlier RRs). Both
    # config-only, on the champion base. The "stack the winners" play that made
    # disc-recency competitive — applied to the new keeper.
    "derby-x-wdl-recency": Cell("derby-x-wdl-recency", sgd_per_game=1.0,
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
                                   "--value-discount", "0.98", "--value-head", "wdl"],
                extra_train_args=["--sgd-steps-per-epoch", "64", "--value-head", "wdl",
                                  "--buffer-recency-frac", "0.5"]),

    # 'dir15' = champion base with LESS root Dirichlet noise (eps 0.25 -> 0.15). The
    # exploration-noise knob was frozen at the v3 default and NEVER swept (like
    # gumbel-m was). Tests whether a sharper/more-exploitative root helps the mature
    # recipe. Config-only. Replaces gumbel-m8 (RR4 dud -48, plateaued).
    "derby-x-dir15": Cell("derby-x-dir15", sgd_per_game=1.0,
                buffer_size=1_500_000, games_per_epoch=64,
                size="small", stem_padding=1, n_simulations=100,
                n_workers=8, games_per_batch=8, wave_mode=False,
                c_puct=1.25, c_puct_base=19652.0,
                dirichlet_alpha=0.13, dirichlet_eps=0.15,
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
    # 'x-draw-contempt' = DECISIVENESS lever (bead derby-9q4). VERBATIM clone of
    # derby-v7-mate-discount (the reigning champion: gumbel-root + vcf-teacher +
    # value-discount 0.98 + global-pool + gumbel-m 16 + the 0.4/0.1 league mix +
    # sgd-steps-per-epoch 64) with ONE lever added: --draw-value 0.05 on the
    # worker (mild contempt). Training-side value-TARGET reshape — when a game
    # ends in a draw, the target becomes -0.05 (composed with value-discount via
    # gamma^plies, same shape as decisive outcomes) instead of exactly 0, so the
    # net learns to avoid draws -> MCTS prefers non-drawing continuations. Zero
    # gen-hot-path cost (sibling of --value-discount). Targets the measured
    # binding gap: the v8 champion 100%s heuristic + lookahead2 but only converts
    # ~51% of lookahead4-as-BLACK to wins (the rest are draws, not losses); the
    # gap is structural (does not improve with training or capacity per v9).
    # Lane-isolated under sweep_runs/derby-x-draw-contempt/.
    "derby-x-draw-contempt": Cell("derby-x-draw-contempt", sgd_per_game=1.0,
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
                                   "--value-discount", "0.98", "--draw-value", "0.05"],
                extra_train_args=["--sgd-steps-per-epoch", "64"]),
    # 'x-reanalyze' = SAMPLE-EFFICIENCY lever (bead derby-wdw; reanalyze epic
    # 3/3, parent derby-3vs). VERBATIM clone of derby-v7-mate-discount (the
    # reigning champion: gumbel-root + vcf-teacher + value-discount 0.98 +
    # global-pool + gumbel-m 16 + the 0.4/0.1 league mix + sgd-steps-per-epoch
    # 64) with ONE lever added: --reanalyze on the TRAINER (workers unchanged
    # — selfplay_worker would hard-error on these flags). Periodically re-runs
    # MCTS on a small slice of OLD buffer positions with the CURRENT (stronger)
    # net and DEEPER search (sims=200 vs gen's 100), overwriting stale (pi,z)
    # targets — bakes the deeper-search insight into what the net learns so
    # its 100-sim prior absorbs the deeper plan (the H2 lookahead4-black ~50%
    # draw ceiling is a search-depth-at-eval gap; reanalyze attacks it from
    # the training side). KataGo paper §4 credits reanalyze with ~2x
    # elo/compute. Defaults are the engine/scheduler defaults (passed
    # explicitly so the recipe is unambiguous): conservative first A/B vs the
    # champion — fraction 0.05, max 1024 positions/cycle, sims 200, every
    # epoch, cooldown 3 cycles (the feedback-loop guard that prevents the net
    # reinforcing its own biases). Lane-isolated under sweep_runs/derby-x-reanalyze/.
    "derby-x-reanalyze": Cell("derby-x-reanalyze", sgd_per_game=1.0,
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
                extra_train_args=["--sgd-steps-per-epoch", "64",
                                  "--reanalyze",
                                  "--reanalyze-fraction", "0.05",
                                  "--reanalyze-max-positions", "1024",
                                  "--reanalyze-sims", "200",
                                  "--reanalyze-every-epochs", "1",
                                  "--reanalyze-cooldown-cycles", "3"]),
    # 'x-search-contempt' = POSITION-DISTRIBUTION lever (bead derby-qoq,
    # arxiv 2504.07757 Singh & Eindhoven 2025 — "Search-Contempt: a Hybrid
    # MCTS for AlphaZero-style training at ~100x less compute"). VERBATIM
    # clone of derby-v7-mate-discount (the reigning champion: gumbel-root
    # + vcf-teacher + value-discount 0.98 + global-pool + gumbel-m 16 +
    # 0.4/0.1 league mix + sgd-steps-per-epoch 64) with ONE lever added:
    # --contempt-p 0.5 on the WORKER (paper's default). At each self-play
    # move, with prob 0.5, REPLACE the SH-argmax (or temperature-sampled)
    # move with a contempt-perturbed pick that favors children with Q
    # closest to 0 (most contested), so self-play oversamples hard-to-
    # convert positions. The recorded `pi` is UNCHANGED — only the MOVE
    # PLAYED (and thus the buffer position distribution) shifts. Sibling
    # of draw-contempt: both target the lookahead4-as-BLACK ~50% draw
    # ceiling, but from different sides — draw-contempt reshapes the
    # value TARGET on drawn games (training-side); search-contempt
    # reshapes the POSITION DISTRIBUTION (generation-side). Demonstrated
    # in Odds Chess (the closest published analogue to our hard-to-convert
    # regime). Lane-isolated under sweep_runs/derby-x-search-contempt/.
    "derby-x-search-contempt": Cell("derby-x-search-contempt", sgd_per_game=1.0,
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
                                   "--value-discount", "0.98", "--contempt-p", "0.5"],
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
                max_wall_secs: float = 0.0, final_eval: bool = False,
                internal_eval: bool = False) -> None:
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
    # Continuous internal-baseline eval (random/heuristic/lookahead) is OPT-IN.
    # For mature nets it is saturated noise (all pin ~100%, Elo ceiling-clamps)
    # AND costs real CPU (a lookahead:depth=4 cycle is ~200-320s, recurring every
    # ~15-20 epochs) that competes with the Rapfi ladder — our actual yardstick.
    # The trainer's own plies/vl/pl/wall already cover liveness for free. So it is
    # off by default; opt back in with --internal-eval for a COLD-START run, where
    # the random→heuristic→lookahead ladder genuinely tracks the early climb.
    if internal_eval:
        spawn("eval", eval_cmd(cell, dirs), env_override={"GOMOKU_DEVICE": "cpu"})
    else:
        print("  internal-baseline eval DISABLED (saturated for mature nets; "
              "Rapfi ladder is the yardstick). Re-enable with --internal-eval.")

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
    p.add_argument("--internal-eval", action="store_true",
                   help="Spawn the continuous internal-baseline eval worker "
                        "(random/heuristic/lookahead). OFF by default — saturated "
                        "noise for mature nets, and the ~200-320s/cycle lookahead "
                        "games steal CPU from the Rapfi ladder. Turn ON for "
                        "cold-start runs where the baseline ladder tracks the climb.")
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
                max_wall_secs=args.max_wall_secs, final_eval=args.final_eval,
                internal_eval=args.internal_eval)


if __name__ == "__main__":
    main()
