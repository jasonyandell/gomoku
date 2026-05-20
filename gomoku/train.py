"""AlphaZero-style training loop: self-play -> replay buffer -> SGD -> repeat."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gomoku.eval import mcts_picker, play_match_pickers
from gomoku.match import build_player, parse_spec
from gomoku.mcts import make_torch_evaluator
from gomoku.model import build_model, load_checkpoint, n_params, save_checkpoint
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import generate_games, generate_games_vs_baseline
from gomoku.util import load_wandb_key_from_keychain, pick_device


def policy_loss(logits: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft targets. pi may have zeros on illegal moves; those contribute 0."""
    logp = F.log_softmax(logits, dim=-1)
    return -(pi * logp).sum(dim=-1).mean()


def value_loss(v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v, z)


def train_step(
    model,
    optimizer,
    planes: torch.Tensor,
    pi: torch.Tensor,
    z: torch.Tensor,
    *,
    value_weight: float = 1.0,
    l2_weight: float = 1e-4,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, v = model(planes)
    pl = policy_loss(logits, pi)
    vl = value_loss(v, z)
    loss = pl + value_weight * vl
    if l2_weight > 0:
        l2 = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
        loss = loss + l2_weight * l2
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        # Accuracy: fraction where argmax(logits over legal entries of pi) matches argmax(pi).
        pred = logits.argmax(dim=-1)
        target = pi.argmax(dim=-1)
        acc = (pred == target).float().mean()
    return {
        "loss/total": float(loss.detach()),
        "loss/policy": float(pl.detach()),
        "loss/value": float(vl.detach()),
        "train/policy_acc": float(acc),
    }


def _baseline_log_key(spec) -> str:
    """Turn a PlayerSpec into a stable wandb-friendly metric key suffix.

    Examples:
        random                  -> vs_random
        heuristic               -> vs_heuristic
        lookahead:depth=4       -> vs_lookahead4
        lookahead               -> vs_lookahead
    """
    if spec.kind == "lookahead":
        depth = spec.kwargs.get("depth", "")
        return f"vs_lookahead{depth}"
    return f"vs_{spec.kind}"


def _parse_specs(s: str) -> list:
    s = s.strip()
    if not s:
        return []
    out = []
    for raw in s.split(","):
        raw = raw.strip()
        if not raw:
            continue
        spec = parse_spec(raw)
        if spec.kind == "model":
            raise SystemExit(f"--eval-baselines must not include model specs: {raw!r}")
        out.append(spec)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--size", type=str, default="small", help="tiny / small / medium / large")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--games-per-epoch", type=int, default=64)
    p.add_argument("--n-simulations", type=int, default=100)
    p.add_argument("--wave-size", type=int, default=1,
                   help="zeb-style wave-batched MCTS: leaves collected per game per "
                        "evaluator call. 1 = original behavior, 8-32 = much bigger "
                        "batches via virtual loss (fewer MPS syncs).")
    p.add_argument("--opponent", type=str, default="self",
                   help="Self-play opponent. 'self' (default) = model vs model with "
                        "MCTS both sides. Any other valid player spec "
                        "(e.g. 'heuristic', 'random', 'lookahead:depth=2') = model "
                        "plays MCTS against a fixed picker — only model plies "
                        "contribute training examples. Stationary opponents force "
                        "decisive games and break self-play attractors.")
    p.add_argument("--model-first-frac", type=float, default=0.5,
                   help="Fraction of games where model plays the first move when "
                        "--opponent != self. 0.5 = model sees both sides equally.")
    p.add_argument("--opponent-mix-random", type=float, default=0.0,
                   help="If > 0, wraps the opponent picker so it plays a uniform "
                        "random legal move with this probability per turn, and "
                        "the configured opponent otherwise. Useful to weaken a "
                        "strong baseline so the model gets a sometimes-win signal.")
    p.add_argument("--random-opening-moves", type=int, default=0,
                   help="If > 0, each self-play / vs-baseline game starts with this "
                        "many uniformly-random legal moves played (alternating sides). "
                        "MCTS only takes over after that, and no training examples are "
                        "recorded for the random opening. Breaks the 'always-same-opening' "
                        "collapse by forcing the model to learn from diverse positions.")
    p.add_argument("--c-puct", type=float, default=1.25,
                   help="c_puct_init in the AGZ log-schedule PUCT formula. Effective "
                        "exploration constant at N_parent=0. Default 1.25 = AGZ value.")
    p.add_argument("--c-puct-base", type=float, default=19652.0,
                   help="c_puct_base in the AGZ log-schedule PUCT formula. Controls "
                        "how fast the effective exploration grows with parent visits. "
                        "AGZ default 19652 keeps it nearly constant at our sim budgets.")
    p.add_argument("--temperature-moves", type=int, default=8)
    p.add_argument("--temperature-final", type=float, default=0.1,
                   help="After the warm-up plies (--temperature-moves), sample MCTS "
                        "actions at this temperature instead of fully greedy (tau=0). "
                        "Default 0.1 matches michaelnny/alpha_zero — sharp but not "
                        "one-hot, preserves a tiny amount of whole-game exploration "
                        "and keeps the policy training target as a real distribution.")
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--replay-buffer-size", type=int, default=1_500_000,
                   help="Capacity of the replay ring buffer. Default 1.5M matches "
                        "michaelnny/alpha_zero's gomoku config — much larger than our "
                        "earlier 500k runs, which gives the network more diverse "
                        "opponent-version exposure before old positions evict.")
    p.add_argument("--training-steps", type=int, default=400,
                   help="SGD steps per epoch. Static unless --sgd-per-game is set.")
    p.add_argument("--sgd-per-game", type=float, default=None,
                   help="Dynamic balance knob. If set, each cycle computes "
                        "training_steps = max(--min-training-steps, "
                        "round(K * games_ingested_this_cycle)). Overrides "
                        "--training-steps. Keeps SGD intensity in lockstep with "
                        "actual gen throughput from the workers — if they speed up "
                        "or slow down, training scales with them. K ≈ 1.0 matches "
                        "the AZ recipe (1 SGD step per game played).")
    p.add_argument("--min-training-steps", type=int, default=16,
                   help="When --sgd-per-game is set, never do fewer than this "
                        "many SGD steps in a cycle (keeps the trainer moving "
                        "even if workers stall briefly).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-sims", type=int, default=50,
                   help="MCTS sims used by the model during eval matches")
    p.add_argument("--eval-baselines", type=str,
                   default="random,heuristic",
                   help="comma-separated player specs played every eval cycle. "
                        "Default dropped lookahead:depth=2 (45 s+ for noisy signal at "
                        "n=4 games — bring it back explicitly if needed)")
    p.add_argument("--eval-baseline-games", type=int, default=4,
                   help="games per matchup vs each fast baseline (kept small to stay "
                        "in 10 s total eval budget per cycle)")
    p.add_argument("--eval-baselines-slow", type=str, default="",
                   help="comma-separated player specs gated by --eval-slow-every (empty "
                        "to disable; default empty since lookahead:depth=4 dominates "
                        "an eval epoch)")
    p.add_argument("--eval-slow-every", type=int, default=4,
                   help="run --eval-baselines-slow every Nth eval cycle (=Nx eval-every epochs)")
    p.add_argument("--eval-slow-games", type=int, default=6,
                   help="games per matchup vs each slow baseline")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--save-buffer-every", type=int, default=20,
                   help="Rewrite `latest.pt` (which embeds the ~1.4 GB replay "
                        "buffer) every N epochs. Set higher than --save-every "
                        "to throttle disk IO; older latest.pt is still valid "
                        "for resume, just slightly stale.")
    p.add_argument("--keep-last-n", type=int, default=3,
                   help="Auto-prune older epoch checkpoints, keeping only this many "
                        "of the most recent ones (plus whatever `latest.pt` points to). "
                        "Set to 0 to disable pruning (old behaviour). The replay buffer "
                        "is still embedded in `latest.pt` for resume; intermediate "
                        "checkpoints are weights+optimizer only (cheap).")
    p.add_argument("--orphan-sweep-age-sec", type=float, default=300.0,
                   help="In each save cycle, delete .tmp / unrecognized .pt files in the "
                        "checkpoint dir AND any leftover *.pt.tmp / orphaned *.pt in the "
                        "worker-input-dir that are older than this many seconds. Keeps "
                        "interrupted writes from accumulating across runs.")
    p.add_argument("--no-eval", dest="eval_in_trainer", action="store_false",
                   help="Skip the in-trainer eval block entirely. Pair with "
                        "`python -m gomoku.eval_worker` running separately — it polls "
                        "the published worker-weights file and posts results to the same "
                        "wandb run. Lets generation+training run at full speed.")
    p.set_defaults(eval_in_trainer=True)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--worker-input-dir", type=str, default=None,
                   help="If set, the trainer ingests game-record batches from this "
                        "directory (written by `gomoku.selfplay_worker` processes) "
                        "instead of running generate_games in-process. Files are "
                        "deleted after ingest. Pair with --worker-weights-path so "
                        "workers can refresh their model.")
    p.add_argument("--worker-weights-path", type=str, default=None,
                   help="If set, the trainer atomically writes its latest model "
                        "(state_dict + cfg) to this path after each checkpoint, "
                        "so worker processes can hot-reload.")
    p.add_argument("--worker-min-games", type=int, default=0,
                   help="When using --worker-input-dir, ingest at least this many "
                        "new games before starting an SGD cycle. 0 = use "
                        "--games-per-epoch as the target.")
    p.add_argument("--worker-poll-sec", type=float, default=0.5,
                   help="Sleep this long between dir scans when waiting for workers.")
    p.add_argument("--device", type=str, default=None, help="torch device override (e.g. cpu, mps)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.set_defaults(wandb=False)
    p.add_argument("--wandb-project", type=str, default="gomoku")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device = {device}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Build / load model
    start_epoch = 0
    total_games = 0
    wandb_run_id = None
    if args.resume:
        model, payload = load_checkpoint(args.resume, device=device)
        start_epoch = int(payload.get("epoch", 0))
        total_games = int(payload.get("total_games", 0))
        wandb_run_id = payload.get("wandb_run_id")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        print(f"resumed from {args.resume} @ epoch {start_epoch}, total_games={total_games}")
    else:
        model = build_model(args.size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print(f"fresh {args.size} model: {n_params(model):,} params")

    buffer = ReplayBuffer(args.replay_buffer_size, device=device)
    if args.resume:
        # Optional: restore buffer if it was saved.
        payload_buf = payload.get("replay_buffer")
        if payload_buf is not None:
            buffer.load_state_dict(payload_buf)
            print(f"replay buffer restored: {buffer.size} examples")

    # wandb setup
    run = None
    if args.wandb:
        key = load_wandb_key_from_keychain()
        if not key:
            print("warning: --wandb set but no WANDB_API_KEY in keychain or env")
        else:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                id=wandb_run_id,
                resume="allow" if wandb_run_id else None,
                config=vars(args),
            )
            wandb_run_id = run.id

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Pre-parse eval baseline specs so misconfiguration fails before the first epoch.
    fast_specs = _parse_specs(args.eval_baselines)
    slow_specs = _parse_specs(args.eval_baselines_slow)
    # Build baseline pickers once (stateless across calls — RNG is passed in).
    fast_pickers = [(s, build_player(s)) for s in fast_specs]
    slow_pickers = [(s, build_player(s)) for s in slow_specs]
    eval_counter = 0

    # Pre-build the self-play opponent picker if requested. 'self' = vanilla
    # generate_games (model-vs-model). Any other valid spec routes through
    # generate_games_vs_baseline.
    opponent_picker = None
    opponent_label = "self"
    if args.opponent != "self":
        opp_spec = parse_spec(args.opponent)
        if opp_spec.kind == "model":
            raise SystemExit("--opponent model:... not supported for self-play; use 'self' "
                             "or one of random/heuristic/lookahead:depth=N")
        base_picker = build_player(opp_spec)
        opponent_label = opp_spec.label()
        if args.opponent_mix_random > 0.0:
            from gomoku.baselines import random_player as _rand
            p_random = float(args.opponent_mix_random)
            def _mixed_picker(state, rng, *, _base=base_picker, _p=p_random):
                if rng.random() < _p:
                    return _rand(state, rng)
                return _base(state, rng)
            opponent_picker = _mixed_picker
            opponent_label = f"{opponent_label}+{p_random:.0%}random"
        else:
            opponent_picker = base_picker
        print(f"self-play opponent: {opponent_label}")

    # If worker-input-dir is set, the trainer ingests batches from N worker
    # processes via the file-handoff protocol in `selfplay_worker.py` instead
    # of running generate_games in-process. The worker-weights-path is where
    # the trainer publishes its latest state_dict for workers to hot-reload.
    worker_input_dir = Path(args.worker_input_dir) if args.worker_input_dir else None
    if worker_input_dir is not None:
        worker_input_dir.mkdir(parents=True, exist_ok=True)
        target_games = args.worker_min_games or args.games_per_epoch
        print(f"distributed mode: ingest from {worker_input_dir} (target {target_games} games/epoch)")
    worker_weights_path = args.worker_weights_path
    if worker_weights_path:
        print(f"workers will read weights from {worker_weights_path}")

    def _publish_worker_weights(epoch: int = 0) -> None:
        """Atomic-rename the model + cfg to `worker_weights_path` so workers
        (selfplay or eval) can mtime-poll and reload. Embeds the trainer's
        wandb run id + current epoch so eval_worker can log to the same run.
        Also bumps the buffer's current_weight_version so subsequent add() calls
        get the new tag."""
        buffer.set_weight_version(epoch)
        if not worker_weights_path:
            return
        tmp = worker_weights_path + ".tmp"
        save_checkpoint(tmp, model, optimizer=None,
                        epoch=epoch, total_games=total_games,
                        wandb_run_id=wandb_run_id)
        os.replace(tmp, worker_weights_path)

    def _ingest_worker_batches(target_games: int) -> list:
        """Block until at least `target_games` worth of records have arrived in
        the input dir. Returns a flat list of GameRecord, files deleted."""
        out_records: list = []
        n_games_collected = 0
        while n_games_collected < target_games:
            files = sorted(
                (p for p in worker_input_dir.glob("*.pt")),
                key=lambda p: p.stat().st_mtime,
            )
            if not files:
                time.sleep(args.worker_poll_sec)
                continue
            for f in files:
                try:
                    data = torch.load(f, map_location="cpu", weights_only=False)
                    out_records.extend(data["records"])
                    n_games_collected += int(data.get("n_games", len(data["records"])))
                finally:
                    try:
                        f.unlink()
                    except OSError:
                        pass
                if n_games_collected >= target_games:
                    break
        return out_records

    def _sweep_orphans() -> None:
        """Delete stale .tmp files and unrecognized .pt files in checkpoint dir
        + worker_input_dir. Anything older than --orphan-sweep-age-sec that
        doesn't match {latest.pt, worker_weights.pt, epochNNNN.pt} gets removed.
        Catches interrupted writes and leftover artifacts from killed runs."""
        cutoff = time.time() - args.orphan_sweep_age_sec
        protected = {"latest.pt", "latest.pt.tmp", "eval_results.jsonl"}
        if worker_weights_path:
            protected.add(os.path.basename(worker_weights_path))
        ck_dir = Path(args.checkpoint_dir)
        for f in ck_dir.iterdir() if ck_dir.exists() else []:
            if not f.is_file():
                continue
            if f.name in protected:
                continue
            # epochNNNN.pt is governed by --keep-last-n, leave to that path.
            if f.name.startswith("epoch") and f.suffix == ".pt":
                continue
            # Sweep .tmp leftovers and unexpected files older than cutoff.
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
        if worker_input_dir is not None and worker_input_dir.exists():
            for f in worker_input_dir.iterdir():
                if not f.is_file():
                    continue
                # In the records dir, the only valid live files are *.pt batches
                # ingested then deleted by _ingest_worker_batches. *.pt.tmp left
                # over from interrupted worker writes is the main thing to sweep.
                if f.suffix != ".tmp":
                    continue
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except OSError:
                    pass

    # Eval JSONL: when running with --no-eval, an out-of-process gomoku.eval_worker
    # appends one JSON line per pass to <checkpoint_dir>/eval_results.jsonl.
    # The trainer is the SOLE wandb writer (two processes calling wandb.init for the
    # same run id silently drops one writer's logs), so we tail this file and merge
    # any new eval rows into the current cycle's log dict.
    eval_jsonl_path = Path(args.checkpoint_dir) / "eval_results.jsonl"
    eval_jsonl_offset = 0
    if eval_jsonl_path.exists():
        # Don't replay rows that predate this trainer run.
        eval_jsonl_offset = eval_jsonl_path.stat().st_size

    def _consume_eval_jsonl() -> dict:
        """Read new lines appended to eval_results.jsonl since last consumed.
        Returns the merged metric dict (later rows win on key collision)."""
        nonlocal eval_jsonl_offset
        if not eval_jsonl_path.exists():
            return {}
        try:
            cur_size = eval_jsonl_path.stat().st_size
        except OSError:
            return {}
        if cur_size <= eval_jsonl_offset:
            return {}
        merged: dict = {}
        try:
            with open(eval_jsonl_path, "r") as f:
                f.seek(eval_jsonl_offset)
                new = f.read()
                eval_jsonl_offset = f.tell()
        except OSError:
            return {}
        for line in new.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k, v in rec.items():
                if k == "ts":
                    continue
                merged[k] = v
        return merged

    # Initial weight publication + orphan sweep, if applicable.
    _publish_worker_weights()
    _sweep_orphans()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()

        # --- self-play (in-process) or ingest (workers) ---
        gen_start = time.time()
        if worker_input_dir is not None:
            records = _ingest_worker_batches(args.worker_min_games or args.games_per_epoch)
        elif opponent_picker is None:
            evaluator = make_torch_evaluator(model, device)
            records = generate_games(
                args.games_per_epoch,
                evaluator,
                n_simulations=args.n_simulations,
                c_puct=args.c_puct,
                c_puct_base=args.c_puct_base,
                temperature_moves=args.temperature_moves,
                temperature_final=args.temperature_final,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps,
                rng=rng,
                wave_size=args.wave_size,
                random_opening_moves=args.random_opening_moves,
            )
        else:
            evaluator = make_torch_evaluator(model, device)
            records = generate_games_vs_baseline(
                args.games_per_epoch,
                evaluator,
                opponent_picker,
                n_simulations=args.n_simulations,
                c_puct=args.c_puct,
                c_puct_base=args.c_puct_base,
                temperature_moves=args.temperature_moves,
                temperature_final=args.temperature_final,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps,
                rng=rng,
                wave_size=args.wave_size,
                model_first_frac=args.model_first_frac,
                random_opening_moves=args.random_opening_moves,
            )
        gen_time = time.time() - gen_start

        n_examples_new = sum(len(r.examples) for r in records)
        for r in records:
            buffer.add(r.examples)

        plies_mean = float(np.mean([r.plies for r in records])) if records else 0.0
        # Plies-distribution percentiles for the new records — quick check on
        # game-length shape going INTO the buffer this cycle.
        plies_array = np.array([r.plies for r in records]) if records else np.array([0])
        plies_p10 = float(np.percentile(plies_array, 10))
        plies_p50 = float(np.percentile(plies_array, 50))
        plies_p90 = float(np.percentile(plies_array, 90))
        outcomes = [r.outcome for r in records]
        # In self-play, outcome is +1 if first-mover (black) won.
        # In vs-baseline, outcome is +1 if model won.
        wins_pos = sum(1 for o in outcomes if o > 0)
        wins_neg = sum(1 for o in outcomes if o < 0)
        draws = sum(1 for o in outcomes if o == 0)

        # --- training ---
        # Decide how many SGD steps this cycle. If --sgd-per-game is set, scale
        # with games actually ingested so the trainer self-balances against the
        # worker pool's throughput. Otherwise use the static --training-steps.
        n_games_this_cycle = len(records)
        if args.sgd_per_game is not None and n_games_this_cycle > 0:
            steps_this_cycle = max(
                args.min_training_steps,
                round(args.sgd_per_game * n_games_this_cycle),
            )
        else:
            steps_this_cycle = args.training_steps

        train_metrics_acc: dict[str, list[float]] = {}
        train_start = time.time()
        if buffer.size >= args.batch_size:
            for _ in range(steps_this_cycle):
                planes, pi, z = buffer.sample(args.batch_size)
                m = train_step(model, optimizer, planes, pi, z,
                               value_weight=args.value_weight, l2_weight=args.l2)
                for k, v in m.items():
                    train_metrics_acc.setdefault(k, []).append(v)
        train_time = time.time() - train_start
        train_metrics = {k: float(np.mean(v)) for k, v in train_metrics_acc.items()}

        total_games += n_games_this_cycle if worker_input_dir is not None else args.games_per_epoch

        # Log key names depend on opponent: self-play tracks black/white, vs-baseline
        # tracks model wins/losses.
        if opponent_picker is None:
            wins_key, losses_key = "selfplay/black_wins", "selfplay/white_wins"
        else:
            wins_key, losses_key = "selfplay/model_wins", "selfplay/model_losses"
        log = {
            "epoch": epoch + 1,
            "total_games": total_games,
            "buffer_size": buffer.size,
            "selfplay/new_examples": n_examples_new,
            "selfplay/new_games": n_games_this_cycle,
            "selfplay/plies_mean": plies_mean,
            "selfplay/plies_p10": plies_p10,
            "selfplay/plies_p50": plies_p50,
            "selfplay/plies_p90": plies_p90,
            wins_key: wins_pos,
            losses_key: wins_neg,
            "selfplay/draws": draws,
            "time/gen_s": gen_time,
            "time/train_s": train_time,
            "train/steps_this_cycle": steps_this_cycle,
            "train/actual_sgd_per_game": (
                steps_this_cycle / n_games_this_cycle if n_games_this_cycle else 0.0
            ),
            **train_metrics,
            # Buffer-shape snapshot: distribution over n_stones buckets + z mix.
            # Computed every cycle so wandb shows shape evolution over time.
            **buffer.shape_stats(),
        }

        # --- forward any new eval_worker results into wandb ---
        ext_eval = _consume_eval_jsonl()
        if ext_eval:
            log.update(ext_eval)

        # --- in-trainer eval (only when --no-eval not passed) ---
        if args.eval_in_trainer and (epoch + 1) % args.eval_every == 0:
            eval_counter += 1
            eval_start = time.time()
            eval_evaluator = make_torch_evaluator(model, device)
            model_picker = mcts_picker(eval_evaluator,
                                       n_simulations=args.eval_sims,
                                       c_puct=args.c_puct)

            run_slow = bool(slow_pickers) and (eval_counter % args.eval_slow_every == 0)
            batches = [("fast", fast_pickers, args.eval_baseline_games)]
            if run_slow:
                batches.append(("slow", slow_pickers, args.eval_slow_games))

            for batch_label, pickers, n_games in batches:
                for spec_idx, (spec, baseline_picker) in enumerate(pickers):
                    m_start = time.time()
                    res = play_match_pickers(
                        model_picker, baseline_picker,
                        n_games=n_games,
                        seed=args.seed + epoch * 1000 + spec_idx,
                    )
                    key = _baseline_log_key(spec)
                    log[f"eval/{key}_winrate"] = res.win_rate
                    log[f"eval/{key}_wins"] = res.wins
                    log[f"eval/{key}_losses"] = res.losses
                    log[f"eval/{key}_draws"] = res.draws
                    log[f"time/eval_{key}_s"] = time.time() - m_start

            log["time/eval_s"] = time.time() - eval_start

        epoch_time = time.time() - epoch_start
        log["time/epoch_s"] = epoch_time

        # Print
        msg = (
            f"epoch {epoch + 1}/{start_epoch + args.epochs} "
            f"games={total_games} buf={buffer.size} "
            f"new={n_games_this_cycle} steps={steps_this_cycle} "
            f"pl={train_metrics.get('loss/policy', float('nan')):.3f} "
            f"vl={train_metrics.get('loss/value', float('nan')):.3f} "
            f"plies={plies_mean:.1f} "
            f"age={log.get('buffer/age_p50', 0):.0f} "
            f"({epoch_time:.1f}s: gen={gen_time:.1f}s train={train_time:.1f}s)"
        )
        wr_bits = [
            f"{key[3:]}={log[f'eval/{key}_winrate']:.0%}"
            for spec in fast_specs + slow_specs
            for key in [_baseline_log_key(spec)]
            if f"eval/{key}_winrate" in log
        ]
        # Also surface any eval/* keys forwarded from eval_worker that aren't in our fast/slow lists.
        seen = {f"eval/{_baseline_log_key(s)}_winrate" for s in (fast_specs + slow_specs)}
        for k, v in log.items():
            if k.startswith("eval/") and k.endswith("_winrate") and k not in seen:
                wr_bits.append(f"{k[len('eval/'):-len('_winrate')]}={v:.0%}")
        if wr_bits:
            msg += " wr[" + " ".join(wr_bits) + "]"
        print(msg, flush=True)

        if run is not None:
            run.log(log)

        # --- save ---
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch{epoch + 1:04d}.pt")
            # Intermediate snapshots are weights+optimizer only (small, ~4 MB).
            # The replay buffer (~1.4 GB) only goes in latest.pt to support
            # resume — old "epochNNNN.pt" files don't need it.
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                epoch=epoch + 1,
                total_games=total_games,
                wandb_run_id=wandb_run_id,
            )
            # Write a separate latest.pt that includes the replay buffer.
            # Throttled by --save-buffer-every since this is the ~1.4 GB write.
            if (epoch + 1) % args.save_buffer_every == 0:
                latest = os.path.join(args.checkpoint_dir, "latest.pt")
                latest_tmp = latest + ".tmp"
                save_checkpoint(
                    latest_tmp,
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    total_games=total_games,
                    wandb_run_id=wandb_run_id,
                    extra={"replay_buffer": buffer.state_dict()},
                )
                try:
                    if os.path.islink(latest) or os.path.exists(latest):
                        os.remove(latest)
                    os.replace(latest_tmp, latest)
                except OSError:
                    pass
            # Publish a lean weights-only file for any worker processes
            # (self-play workers AND the eval_worker).
            _publish_worker_weights(epoch=epoch + 1)
            # Auto-prune old intermediate checkpoints to keep disk bounded.
            if args.keep_last_n > 0:
                ck_files = sorted(
                    Path(args.checkpoint_dir).glob("epoch*.pt"),
                    key=lambda p: p.stat().st_mtime,
                )
                to_drop = ck_files[: -args.keep_last_n] if len(ck_files) > args.keep_last_n else []
                for f in to_drop:
                    try:
                        f.unlink()
                    except OSError:
                        pass
            # Sweep .tmp leftovers and unknown files past the age threshold.
            _sweep_orphans()


if __name__ == "__main__":
    main()
