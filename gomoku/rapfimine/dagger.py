"""On-book DAgger for the idx-2 one-position experiment.

DAgger (Ross/Gordon/Bagnell 2011) fixes the covariate-shift wound that plain
behavioural cloning (the pretrain) and plain self-play both dodge: the expert
labels the states the STUDENT actually visits, those states are AGGREGATED into
the dataset, and the policy is re-fit by SUPERVISED IMITATION on the aggregate.

    D ← expert data (the BFS mine);  π̂ ← train on D
    for round i:
        roll out πᵢ to visit states     # student-vs-Rapfi, both seats  (gather)
        label every visited state w/ π*  # Rapfi's TIMED move (one-hot)  (label)
        D ← D ∪ {(s, Rapfi(s))}          # aggregate — never forget
        π̂ᵢ₊₁ ← train on D (warm-continue, gentle)                       (train)
    keep the round iff it does not regress vs the frozen parent          (gate)

This is PURE supervised imitation — NOT the AZ teacher-mix that collapsed the
student in #77 (a hot teacher CE fighting the self-play target at high LR
corrupted the trunk). With no competing objective, the failure mode is different:
the EXPERT MUST BE STRONGER THAN THE STUDENT or imitation drags the net DOWN. The
node-bounded SOFT winrate map (``analyze`` max_node) is NOT — fast_eval's own
calibration shows even 2M nodes loses to a matured net, so a soft label is a
sub-student teacher and round 0 (2026-06-25) regressed the net 0/48 vs the parent.
The strength dial is think-TIME: ``--label-timeout-ms`` labels with Rapfi's actual
move at a real time budget (a HARD one-hot — on-book DAgger's classifier target),
which IS stronger. See TRAINING_WIKI 2026-06-25.

Performance (the hard constraint): the two constituents use DISJOINT compute —
student MCTS on MPS, Rapfi on CPU cores — so the concurrent rollout overlaps them
and runs at ~1× the slower part (measured: 215 plies/s ≈ the 220 moves/s solo
student rate; Rapfi at ~870 labels/s hides under the MCTS bottleneck). Soft-
labelling the distinct states post-hoc adds <1 s on an ~8 s rollout. Well inside
the 2× budget.

Schema: dagger shards are byte-identical to the mine (``store.ShardWriter`` →
teacher-v2 soft npz), so ``pretrain.load_shards`` and the trainer read mine and
dagger dirs interchangeably. Canonical (D4-reduced, history-less) representatives,
exactly like the mine, so the trainer's sample-time D4 augment recovers the 8×.

CLI::

    GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.dagger round \\
        --checkpoint checkpoints/idx2_warmstart_final.pt \\
        --parent     checkpoints/idx2_warmstart_final.pt \\
        --dagger-dir /Users/jason/data/rapfimine/dagger_idx2 \\
        --mine-dir   /Users/jason/data/rapfimine/idx2_15x15 \\
        --target-new 30000 --train-epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from gomoku.board_config import BOARD_SIZE
from gomoku.eval_panel import (
    EvaluatorCache, IDX2_OPENING, Ruler, eval_vs_ruler, fixed_opening_state,
    result_to_row,
)
from gomoku.mcts import MCTSGame, policy_from_visits, run_batched_mcts
from gomoku.rapfi_pool import RapfiPool, rapfi_available
from gomoku.rapfimine.canonical import canonical_key, canonical_state
from gomoku.rapfimine.store import (
    ShardWriter, count_examples, iter_shard_paths, load_seen_keys,
)


# ==========================================================================
# GATHER — roll out the student vs Rapfi (both seats), concurrently.
# ==========================================================================
def _sample_move(mg, rng, ply, temp_until_ply, temperature):
    """Greedy after ``temp_until_ply``; before it, sample from the visit policy
    (opening variety → coverage). The student's policy is still what's executed —
    sampling its early moves just spreads the visited distribution."""
    if ply < temp_until_ply and temperature > 0:
        pi = policy_from_visits(mg.root, temperature=temperature)
        ssum = pi.sum()
        if ssum > 0:
            return int(rng.choice(len(pi), p=pi / ssum))
    return int(np.argmax(policy_from_visits(mg.root, temperature=0.0)))


def rollout_once(
    net_eval, pool, *, n_games: int, sims: int, c_puct: float, max_plies: int,
    seed: int, temp_until_ply: int = 10, temperature: float = 1.0,
) -> tuple[list, int, int]:
    """One batch of ``n_games`` student-vs-Rapfi games from idx-2, both seats.

    Reuses the project's two throughput tricks (``fast_eval``): net moves are one
    batched ``run_batched_mcts`` GPU pass across every net-to-move game; Rapfi
    (the opponent) moves are fanned across the warm pool in one ``label_states``.
    Early-ply temperature sampling on the student's own policy spreads the visited
    distribution (coverage). Returns ``(visited, n_finished, n_states)`` where
    ``visited`` is a list of ``(state, z)`` — every distinct board seen (natural,
    un-canonicalised) tagged with the **Monte-Carlo game outcome** ``z`` from the
    side-to-move's perspective (+1 won / −1 lost / 0 draw-or-unfinished). ``z`` is
    AlphaZero's value target, free from the games we already play to terminal:
    "P(my policy beats Rapfi from here)" — exactly the value to teach vs Rapfi.
    """
    rng = np.random.default_rng(seed)
    start = fixed_opening_state(IDX2_OPENING)
    games = [{"state": start, "net_black": (i % 2 == 0), "done": False,
              "winner": None, "traj": [start]} for i in range(n_games)]

    ply = start.move_count
    while any(not g["done"] for g in games) and ply < max_plies:
        active = [g for g in games if not g["done"]]
        side = ply % 2  # 0=black, 1=white to move
        net_games = [g for g in active if (side == 0) == g["net_black"]]
        rap_games = [g for g in active if (side == 0) != g["net_black"]]
        moves: dict[int, int] = {}
        if net_games:
            mgs = [MCTSGame(g["state"], c_puct=c_puct, rng=rng) for g in net_games]
            run_batched_mcts(mgs, net_eval, n_simulations=sims, add_root_noise=False)
            for g, mg in zip(net_games, mgs):
                moves[id(g)] = _sample_move(mg, rng, ply, temp_until_ply, temperature)
        if rap_games:
            acts = pool.label_states([g["state"] for g in rap_games])
            for g, a in zip(rap_games, acts):
                moves[id(g)] = int(a)
        for g in active:
            a = moves.get(id(g))
            if a is None:
                continue
            g["state"] = g["state"].apply(a)
            g["traj"].append(g["state"])
            done, v = g["state"].is_terminal()
            if done:
                g["done"] = True
                # v == -1.0 → the side that JUST moved won; else terminal draw.
                g["winner"] = side if v == -1.0 else None
        ply += 1

    # Tag every distinct board with its game's MC outcome (first occurrence wins
    # on a transposition — an unbiased, slightly noisy sample). z from the
    # side-to-move's perspective: +1 if that side won, −1 if it lost, 0 draw.
    visited: list = []
    raw_seen: set[bytes] = set()
    for g in games:
        w = g["winner"]
        for s in g["traj"]:
            k = s.board.tobytes()
            if k in raw_seen:
                continue
            raw_seen.add(k)
            stm = s.move_count % 2
            z = 0.0 if w is None else (1.0 if stm == w else -1.0)
            visited.append((s, z))
    n_finished = sum(1 for g in games if g["done"])
    return visited, n_finished, len(visited)


def gather_round(
    checkpoint: str, dagger_dir: str, *, target_new: int, n_games: int,
    sims: int, opponent_timeout_ms: int, max_node: int, max_pv: int | None,
    pool_size: int, c_puct: float, max_plies: int, round_id: int, seed: int,
    temp_until_ply: int = 10, temperature: float = 1.0, max_rolls: int = 400,
    stall_rolls: int = 12, stall_min_fresh: int = 8, label_timeout_ms: int = 0,
    device=None, on_log=print,
) -> int:
    """Roll out until ``target_new`` FRESH canonical states are labelled+stored.

    Dedups against everything already in ``dagger_dir`` (cumulative across rounds)
    so the student's heavily-revisited common lines are stored once. Each call
    writes its own shard stream (``worker_id=round_id``), so rounds never collide.

    Stall-guarded (Jason's "don't let it eat the day"): the student's canonical
    reachable set vs a fixed opponent is BOUNDED, so novelty decays. Stop early
    after ``stall_rolls`` consecutive rolls each yielding < ``stall_min_fresh``
    new states, or after ``max_rolls`` total — returning what was gathered. A
    small per-round harvest is the expected DAgger shape, not a failure.
    """
    net_eval = EvaluatorCache(device=device).evaluator(checkpoint)
    seen: set[bytes] = load_seen_keys(dagger_dir)
    start_n = len(seen)
    writer = ShardWriter(dagger_dir, worker_id=round_id)
    seat_new = {True: 0, False: 0}

    # The opponent pool (rollout) is FAST (coverage). The teacher signal is a
    # SEPARATE concern: node-bounded `analyze` (max_node) never reaches Rapfi's
    # real strength (fast_eval calibration: even 2M nodes loses to the net), so a
    # SOFT label is a teacher WEAKER than a matured student → distilling it
    # regresses the net (the round-0 wall, 2026-06-25). When ``label_timeout_ms``
    # > 0 we instead label with a STRONG, TIME-bounded `pick` (the expert's actual
    # move = HARD one-hot, on-book DAgger's classifier target) from a dedicated
    # strong pool; ``label_timeout_ms`` = 0 keeps the (weak) soft path.
    label_pool_cm = (
        RapfiPool(size=pool_size, timeout_ms=label_timeout_ms, board_size=BOARD_SIZE)
        if label_timeout_ms > 0 else None
    )
    with RapfiPool(size=pool_size, timeout_ms=opponent_timeout_ms,
                   board_size=BOARD_SIZE) as pool:
        label_pool = label_pool_cm.__enter__() if label_pool_cm is not None else None
        added = 0
        roll = 0
        stall = 0
        t0 = time.time()
        while added < target_new and roll < max_rolls and stall < stall_rolls:
            visited, fin, plies = rollout_once(
                net_eval, pool, n_games=n_games, sims=sims, c_puct=c_puct,
                max_plies=max_plies, seed=seed + roll,
                temp_until_ply=temp_until_ply, temperature=temperature,
            )
            roll += 1
            batch_before = added
            # Canonicalise (D4-reduced) and keep only the novel ones. CRUCIAL:
            # keep the REAL history (recency planes) — unlike the mine, which drops
            # it. Pure-supervised DAgger trains AND infers the same net, so the
            # stored planes MUST match inference-time planes; training on zero-
            # recency planes then playing with real recency is the train/inference
            # mismatch that regressed rounds 0a/0b to 0/48 (2026-06-25). The label
            # is still board-only (Rapfi is history-blind), so only the input
            # planes change. canonical_state transforms history under the same sym.
            fresh = []
            for s, z in visited:
                k = canonical_key(s)
                if k in seen:
                    continue
                seen.add(k)
                canon, _ = canonical_state(s)
                fresh.append((canon, k, z))
            if not fresh:
                continue
            # Label the novel states with the expert. STRONG (time-bounded) policy
            # when label_pool is set; else the (weak) node-bounded soft winrate map.
            # POLICY = Rapfi's move; VALUE = the MC game outcome z, encoded through
            # the trainer's value=2·max(soft)−1: a one-hot at Rapfi's move with
            # winrate (z+1)/2 gives policy=one-hot AND value=z, no trainer change.
            if label_pool is not None:
                acts = label_pool.label_states([c for c, _, _ in fresh])
                labels = [{int(a): (z + 1.0) / 2.0}
                          for a, (_, _, z) in zip(acts, fresh)]
            else:
                # Soft path: node-bounded winrate map (value from the map, not MC).
                labels = pool.analyze_states([c for c, _, _ in fresh],
                                             max_node=max_node, max_pv=max_pv)
            for (canon, k, _z), wr in zip(fresh, labels):
                if not wr:
                    continue  # engine refused / empty — skip, don't store an empty target
                writer.add(
                    planes=np.asarray(canon.to_planes(), dtype=np.float16),
                    winrates={int(a): float(w) for a, w in wr.items()},
                    key=k,
                    side=int(canon.move_count % 2),
                    ply=int(canon.move_count),
                )
                added += 1
                seat_new[bool(canon.move_count % 2 == 0)] += 1  # black-to-move == seat
            roll_new = added - batch_before
            stall = stall + 1 if roll_new < stall_min_fresh else 0
            dt = time.time() - t0
            on_log(f"  [gather r{round_id}] roll {roll}: +{added}/{target_new} new "
                   f"(this batch {roll_new} fresh, {fin}/{n_games} finished, "
                   f"stall {stall}/{stall_rolls}) {added/max(dt,1e-6):.0f} new/s")
    if label_pool_cm is not None:
        label_pool_cm.__exit__(None, None, None)
    writer.close()
    reason = ("target reached" if added >= target_new
              else "stalled (reachable set saturated)" if stall >= stall_rolls
              else "max rolls")
    on_log(f"  [gather r{round_id}] DONE ({reason}): stored {added} fresh canonical "
           f"states (black-to-move {seat_new[True]}, white-to-move {seat_new[False]}); "
           f"dagger store now {start_n + added} total")
    return added


# ==========================================================================
# TRAIN — warm-continue supervised imitation on the aggregate (D ∪ D_i).
# ==========================================================================
def _load_dirs(dagger_dir: str, mine_dir: str | None, mine_sample_shards: int,
               seed: int, on_log=print):
    """Load all dagger shards + (optionally) a random sample of mine shards.

    The aggregate is dagger ∪ a retention sample of the mine. Loading the full
    1.1M mine every round is wasteful (9 GB, minutes); the net already carries the
    mine in its warm-start weights, so a capped random sample is enough to fight
    catastrophic forgetting of the broad idx-2 tree while keeping the round fast.
    Set ``mine_sample_shards=0`` to train on dagger only, or a big number to use
    the whole mine.
    """
    from gomoku.board_config import N_ACTIONS

    paths = list(iter_shard_paths(dagger_dir))
    n_dag = len(paths)
    # mine_sample_shards: 0 = dagger only; >=len = whole mine; else a random sample.
    if mine_dir and mine_sample_shards != 0:
        mpaths = list(iter_shard_paths(mine_dir))
        if 0 < mine_sample_shards < len(mpaths):
            rng = np.random.default_rng(seed)
            mpaths = [mpaths[i] for i in
                      rng.choice(len(mpaths), size=mine_sample_shards, replace=False)]
        paths += mpaths
    if not paths:
        raise SystemExit(f"no shards in {dagger_dir!r}")
    counts, shape = [], None
    for p in paths:
        with np.load(p) as z:
            counts.append(int(z["moves"].shape[0]))
            if shape is None:
                shape = z["planes"].shape[1:]
    total = sum(counts)
    planes = np.empty((total, *shape), dtype=np.float16)
    soft = np.zeros((total, N_ACTIONS), dtype=np.float16)
    i = 0
    for p in paths:
        with np.load(p) as z:
            m = int(z["moves"].shape[0])
            planes[i:i + m] = z["planes"]
            soft[i:i + m] = z["soft_policy"]
            i += m
    on_log(f"  [train] aggregate {total} positions "
           f"({n_dag} dagger shards + {len(paths) - n_dag} mine shards, "
           f"{planes.nbytes / 1e9:.1f} GB f16)")
    return planes, soft


def train_round(
    resume_checkpoint: str, out_checkpoint: str, dagger_dir: str, *,
    mine_dir: str | None, mine_sample_shards: int, epochs: int, lr: float,
    batch_size: int, value_weight: float, teacher_temp: float,
    steps_per_epoch: int | None, size: str, global_pool, stem_padding: int,
    device=None, seed: int = 0, on_log=print,
) -> str:
    """Warm-continue the supervised pretrain loop on the aggregate, GENTLY.

    Loads ``resume_checkpoint`` weights and re-fits on D ∪ D_i at a low LR for a
    few epochs — nudge toward the aggregate, don't overwrite. Reuses the pretrain
    data pipeline + the trainer's policy/value losses; value is auxiliary
    (Rapfi's 2·winrate−1), policy carries the imitation load (#18/#44).
    """
    import torch

    from gomoku.model import build_model, load_checkpoint, n_params, save_checkpoint
    from gomoku.rapfimine.pretrain import PretrainData
    from gomoku.train import policy_loss, value_loss
    from gomoku.util import pick_device

    dev = pick_device(device or os.environ.get("GOMOKU_DEVICE")
                      or ("mps" if torch.backends.mps.is_available() else "cpu"))
    planes, soft = _load_dirs(dagger_dir, mine_dir, mine_sample_shards, seed, on_log)
    data = PretrainData(planes, soft, device=dev, teacher_temp=teacher_temp)
    steps = steps_per_epoch or max(1, data.n // batch_size)

    # Warm-start from the current net (the mine lives in these weights). Build with
    # the SAME architecture the checkpoint was trained with, then load its state.
    model = build_model(size, global_pool=global_pool, stem_padding=stem_padding).to(dev)
    src, _payload = load_checkpoint(resume_checkpoint, device=dev)
    model.load_state_dict(src.state_dict())
    on_log(f"  [train] warm-started {n_params(model):,} params from {resume_checkpoint}; "
           f"{steps} steps/epoch × {epochs} @ lr={lr:g} (gentle)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        pl_acc = torch.zeros((), device=dev)
        vl_acc = torch.zeros((), device=dev)
        for _ in range(steps):
            p, pi, z = data.sample(batch_size)
            logits, value = model(p)
            pl = policy_loss(logits, pi)
            vl = value_loss(value, z)
            loss = pl + value_weight * vl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pl_acc += pl.detach()
            vl_acc += vl.detach()
        on_log(f"  [train] epoch {ep+1}/{epochs} policy_ce={float(pl_acc)/steps:.4f} "
               f"value_mse={float(vl_acc)/steps:.4f} {time.time()-t0:.0f}s")
    os.makedirs(os.path.dirname(out_checkpoint) or ".", exist_ok=True)
    save_checkpoint(out_checkpoint, model, opt, epoch=0, total_games=0,
                    extra={"dagger": {"resume": resume_checkpoint,
                                      "dagger_dir": dagger_dir,
                                      "positions": int(data.n), "lr": lr,
                                      "epochs": epochs}})
    on_log(f"  [train] wrote {out_checkpoint}")
    return out_checkpoint


# ==========================================================================
# GATE — net-vs-frozen-parent H2H from idx-2, both seats (NOT vs Rapfi: the
# Rapfi cadence was non-discriminating in #77 — gate on the parent).
# ==========================================================================
def gate_vs_parent(checkpoint: str, parent: str, *, n_games: int, sims: int,
                   temp_until_ply: int = 8, seed: int = 0, device=None) -> dict:
    # Opening variety is REQUIRED: net-vs-net from a fixed opening is deterministic
    # at temp_until_ply=0, so n_games would collapse to ~2 distinct trajectories
    # and the score would be near-binary noise. Sampling the early plies makes the
    # n games genuinely varied → a real win-rate signal.
    start = fixed_opening_state(IDX2_OPENING)
    cache = EvaluatorCache(device=device)
    ruler = Ruler(label="parent", opponent=parent, n_games=n_games, sims=sims,
                  temp_until_ply=temp_until_ply, temperature=1.0)
    res = eval_vs_ruler(checkpoint, ruler, cache=cache, start_state=start, seed=seed)
    return result_to_row("parent", res)


# ==========================================================================
# ROUND — gather → train → gate, one on-book DAgger iteration.
# ==========================================================================
def run_round(args) -> dict:
    """One DAgger iteration. Returns a result dict with an ``rc`` field
    (0=KEEP, 1=REJECT/no-improvement, 3=gather saturated → a wall)."""
    dagger_dir = args.dagger_dir
    os.makedirs(dagger_dir, exist_ok=True)
    round_id = args.round_id
    if round_id < 0:
        # auto: one shard stream per round; next free worker id
        existing = {int(os.path.basename(p).split("_")[1][1:])
                    for p in iter_shard_paths(dagger_dir)} if iter_shard_paths(dagger_dir) else set()
        round_id = (max(existing) + 1) if existing else 0

    print(f"=== DAgger round {round_id} ===  checkpoint={args.checkpoint}")
    t0 = time.time()
    added = gather_round(
        args.checkpoint, dagger_dir, target_new=args.target_new,
        n_games=args.n_games, sims=args.sims,
        opponent_timeout_ms=args.opponent_timeout_ms, max_node=args.max_node,
        max_pv=args.max_pv, pool_size=args.pool_size, c_puct=args.c_puct,
        max_plies=args.max_plies, round_id=round_id, seed=args.seed,
        temp_until_ply=args.rollout_temp_until_ply,
        temperature=args.rollout_temperature, max_rolls=args.max_rolls,
        label_timeout_ms=args.label_timeout_ms, device=args.device,
    )
    if added == 0:
        print("  [round] gathered 0 new states — the student's reachable canonical "
              "set may be saturated vs this opponent. Stopping (a wall to discuss).")
        return {"rc": 3, "round": round_id, "added": 0}

    out_ckpt = args.out or os.path.join(
        os.path.dirname(args.checkpoint) or ".", f"dagger_r{round_id}.pt")
    train_round(
        args.checkpoint, out_ckpt, dagger_dir, mine_dir=args.mine_dir,
        mine_sample_shards=args.mine_sample_shards, epochs=args.train_epochs,
        lr=args.lr, batch_size=args.batch_size, value_weight=args.value_weight,
        teacher_temp=args.teacher_temp, steps_per_epoch=args.steps_per_epoch,
        size=args.size, global_pool=_gp(args.global_pool),
        stem_padding=args.stem_padding, device=args.device, seed=args.seed,
    )

    row = gate_vs_parent(out_ckpt, args.parent, n_games=args.gate_games,
                         sims=args.gate_sims, temp_until_ply=args.gate_temp_until_ply,
                         device=args.device)
    score = row.get("score", 0.0)
    verdict = "KEEP" if score >= args.gate_threshold else "REJECT"
    print(f"\n=== round {round_id} GATE vs parent ({args.parent}) ===")
    print(f"  score={score}  black={row.get('black_score')}  "
          f"white={row.get('white_score')}  (n={row.get('n_games')})  "
          f"threshold={args.gate_threshold} -> {verdict}")
    print(f"=== round {round_id} done in {time.time()-t0:.0f}s; "
          f"dagger store {count_examples(dagger_dir)} total ===")
    result = {"rc": 0 if verdict == "KEEP" else 1, "round": round_id,
              "checkpoint": out_ckpt, "added": added, "score": score,
              "gate": row, "verdict": verdict}
    if args.result_json:
        with open(args.result_json, "w") as f:
            json.dump(result, f, indent=2)
    return result


def _gp(v):
    return bool(v) if v in (0, 1) else v


# ==========================================================================
# LOOP — the DAgger flywheel: round i+1 rolls out from the BEST net so far,
# gates against the FROZEN parent, aggregates forever. State lives on disk
# (the per-round result JSONs + checkpoints), so a kill/restart resumes by
# re-reading them — a stateless reducer over the round log.
# ==========================================================================
import copy  # noqa: E402


def _scan_rounds(log_dir: str) -> list[dict]:
    import glob as _glob
    out = []
    for p in sorted(_glob.glob(os.path.join(log_dir, "dagger_r*_result.json"))):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except (OSError, ValueError):
            continue
    return out


def loop_driver(args) -> int:
    log_dir = args.log_dir or "mined"
    os.makedirs(log_dir, exist_ok=True)
    frozen_parent = args.parent
    ckpt_dir = os.path.dirname(args.checkpoint) or "."

    done = _scan_rounds(log_dir)
    # best net so far = highest gate score over completed rounds, else the start net.
    best_ckpt, best_score = args.checkpoint, -1.0
    for r in done:
        if r.get("score", -1) > best_score and os.path.isfile(r.get("checkpoint", "")):
            best_score, best_ckpt = r["score"], r["checkpoint"]
    next_id = (max((r["round"] for r in done), default=-1) + 1)
    print(f"=== DAgger LOOP === resuming at round {next_id}; "
          f"best so far = {best_ckpt} (score {best_score if best_score>=0 else 'n/a'}); "
          f"frozen parent = {frozen_parent}; budget {args.wall_secs}s / {args.rounds} rounds")

    t0 = time.time()
    ran = 0
    while ran < args.rounds and (time.time() - t0) < args.wall_secs:
        rid = next_id + ran
        ra = copy.copy(args)
        ra.checkpoint = best_ckpt                 # roll out from the BEST net
        ra.parent = frozen_parent                 # gate vs the FROZEN baseline
        ra.round_id = rid
        ra.out = os.path.join(ckpt_dir, f"dagger_r{rid}.pt")
        ra.result_json = os.path.join(log_dir, f"dagger_r{rid}_result.json")
        ra.seed = args.seed + rid * 1000          # decorrelate rollouts per round
        res = run_round(ra)
        ran += 1
        if res.get("rc") == 3:
            print(f"=== LOOP WALL: round {rid} gathered 0 new states (saturated). "
                  f"Stopping for discussion. ===")
            break
        score = res.get("score", 0.0)
        if score > best_score:
            best_score, best_ckpt = score, res["checkpoint"]
            print(f"=== LOOP: round {rid} NEW BEST score={score:.3f} -> {best_ckpt} ===")
        else:
            print(f"=== LOOP: round {rid} score={score:.3f} (best stays {best_score:.3f}); "
                  f"next round rolls out from the best, not this one ===")
        if score < args.abort_below:
            print(f"=== LOOP WALL: round {rid} score {score:.3f} < abort-below "
                  f"{args.abort_below} (hard regression). Stopping for discussion. ===")
            break
    print(f"=== DAgger LOOP done: ran {ran} round(s) in {time.time()-t0:.0f}s; "
          f"best = {best_ckpt} (score {best_score:.3f}) ===")
    return 0


# ==========================================================================
# CLI
# ==========================================================================
def _add_common(p):
    p.add_argument("--checkpoint", required=True, help="current net πᵢ (rollout + warm-start)")
    p.add_argument("--dagger-dir", required=True, help="aggregate store (grows each round)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    # gather
    p.add_argument("--target-new", type=int, default=30000)
    p.add_argument("--n-games", type=int, default=48)
    p.add_argument("--sims", type=int, default=32)
    p.add_argument("--opponent-timeout-ms", type=int, default=50,
                   help="Rapfi opponent think-time during rollout (the strength/coverage "
                        "knob; 50ms ≈ the net's current transition → competitive games)")
    p.add_argument("--max-node", type=int, default=5000,
                   help="Rapfi analyze node budget for the SOFT label (only used when "
                        "--label-timeout-ms 0)")
    p.add_argument("--label-timeout-ms", type=int, default=300,
                   help="STRONG teacher: label with a time-bounded one-hot pick at this "
                        "think-time (the expert's move). 0 = the OLD node-bounded soft "
                        "winrate map, which is WEAKER than a matured net (regresses it).")
    p.add_argument("--max-pv", type=int, default=None)
    p.add_argument("--pool-size", type=int, default=24)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--max-plies", type=int, default=80)
    p.add_argument("--rollout-temp-until-ply", type=int, default=10,
                   help="sample (not argmax) the student's moves before this ply "
                        "for trajectory variety → coverage")
    p.add_argument("--rollout-temperature", type=float, default=1.0)
    p.add_argument("--max-rolls", type=int, default=400,
                   help="hard cap on rollout batches per round (stall guard)")
    p.add_argument("--round-id", type=int, default=-1, help="-1 = auto (next shard stream)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="On-book DAgger for the idx-2 experiment")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _add_train_gate(p):
        p.add_argument("--parent", required=True, help="frozen baseline for the gate (the ep250 net)")
        p.add_argument("--mine-dir", default=None, help="BFS-mine dir to sample for retention (D∪Di)")
        p.add_argument("--mine-sample-shards", type=int, default=40,
                       help="random mine shards mixed in for retention (0 = dagger only)")
        p.add_argument("--out", default=None, help="output checkpoint (default dagger_r{id}.pt)")
        p.add_argument("--train-epochs", type=int, default=3)
        p.add_argument("--lr", type=float, default=1e-4, help="GENTLE warm-continue LR")
        p.add_argument("--batch-size", type=int, default=1024)
        p.add_argument("--steps-per-epoch", type=int, default=None)
        p.add_argument("--value-weight", type=float, default=0.5)
        p.add_argument("--teacher-temp", type=float, default=0.10)
        p.add_argument("--size", default="large")
        p.add_argument("--global-pool", type=int, default=1)
        p.add_argument("--stem-padding", type=int, default=1)
        p.add_argument("--gate-games", type=int, default=48)
        p.add_argument("--gate-sims", type=int, default=100)
        p.add_argument("--gate-temp-until-ply", type=int, default=8,
                       help="opening variety for the net-vs-net gate (0 = deterministic "
                            "→ near-binary noise; keep > 0)")
        p.add_argument("--gate-threshold", type=float, default=0.5)
        p.add_argument("--result-json", default=None)

    r = sub.add_parser("round", help="one DAgger iteration: gather → train → gate")
    _add_common(r)
    _add_train_gate(r)

    lp = sub.add_parser("loop", help="the flywheel: N rounds, best-net rollout, frozen gate")
    _add_common(lp)
    _add_train_gate(lp)
    lp.add_argument("--rounds", type=int, default=8, help="max rounds this invocation")
    lp.add_argument("--wall-secs", type=int, default=3000, help="wall-clock budget")
    lp.add_argument("--abort-below", type=float, default=0.35,
                    help="stop the loop if a round gates below this (hard regression wall)")
    lp.add_argument("--log-dir", default="mined", help="where per-round result JSONs live")

    g = sub.add_parser("gather", help="gather+label only (no train/gate)")
    _add_common(g)

    args = ap.parse_args(argv)
    if BOARD_SIZE != 15:
        print(f"WARNING: GOMOKU_BOARD_SIZE={BOARD_SIZE} (expected 15)", file=sys.stderr)
    if not rapfi_available():
        print("ERROR: native Rapfi not available", file=sys.stderr)
        return 2

    if args.cmd == "gather":
        os.makedirs(args.dagger_dir, exist_ok=True)
        rid = args.round_id if args.round_id >= 0 else 0
        n = gather_round(
            args.checkpoint, args.dagger_dir, target_new=args.target_new,
            n_games=args.n_games, sims=args.sims,
            opponent_timeout_ms=args.opponent_timeout_ms, max_node=args.max_node,
            max_pv=args.max_pv, pool_size=args.pool_size, c_puct=args.c_puct,
            max_plies=args.max_plies, round_id=rid, seed=args.seed,
            temp_until_ply=args.rollout_temp_until_ply,
            temperature=args.rollout_temperature, max_rolls=args.max_rolls,
            label_timeout_ms=args.label_timeout_ms, device=args.device,
        )
        return 0 if n > 0 else 3
    if args.cmd == "loop":
        return loop_driver(args)
    return run_round(args)["rc"]


if __name__ == "__main__":
    sys.exit(main())
