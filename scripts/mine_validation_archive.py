"""Mine a WL4 checkpoint into a frozen validation archive for WL5 diagnostics.

See wiki/topics/wl5-diagnostics-archive-start-design.md for the full design.

Produces a torch.save dict at --output:
    {
        "planes":     (N, N_INPUT_PLANES, BOARD, BOARD) float32,
        "pi_mcts":    (N, N_ACTIONS) float32,
        "z":          (N,) float32 in [-1, 1],
        "provenance": list[str] of length N (bucket tag per row),
        "side":       (N,) int8 (0=black to move at position, 1=white),
        "ply":        (N,) int16 (ply count when captured),
    }

Buckets populated:
    heuristic_loss, lookahead2_loss, lookahead4_loss
        — model-vs-baseline games; capture position BEFORE the move that lost.
    long_defense
        — self-play games with > 50 plies; sample positions where ply > 30.
    canonical_opening
        — first 5-10 plies of self-play games.
    high_kl
        — top-K WL4 buffer positions by KL(p_net || pi_mcts) under the WL4 model.

Uses the in-process batched self-play API (`generate_games`,
`generate_games_vs_baseline`) so MCTS is batched across many concurrent games
and the evaluator runs on MPS — typical end-to-end wall is a few minutes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root is importable when called as `python scripts/mine_validation_archive.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gomoku.baselines import heuristic_player, lookahead_player, random_player  # noqa: E402
from gomoku.game import BOARD_SIZE, GameState, N_ACTIONS, N_INPUT_PLANES  # noqa: E402
from gomoku.mcts import MCTSGame, make_torch_evaluator, policy_from_visits, run_batched_mcts  # noqa: E402
from gomoku.model import load_checkpoint  # noqa: E402
from gomoku.self_play import generate_games, generate_games_vs_baseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wl4-checkpoint", required=True,
                   help="Path to WL4 latest.pt (model + EMA + replay buffer).")
    p.add_argument("--output", required=True,
                   help="Path to write the validation archive .pt.")
    p.add_argument("--target-per-bucket", type=int, default=200,
                   help="Per-bucket target count (cap). Default 200.")
    p.add_argument("--mcts-sims", type=int, default=200,
                   help="MCTS sims used to label positions with pi_mcts.")
    p.add_argument("--c-puct", type=float, default=1.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="torch device. Default reads GOMOKU_DEVICE or cpu.")
    p.add_argument("--batch-games", type=int, default=64,
                   help="Number of games run concurrently per batched MCTS round.")
    p.add_argument("--wave-size", type=int, default=8,
                   help="MCTS wave size passed to run_batched_mcts_waves.")
    p.add_argument("--max-rounds", type=int, default=20,
                   help="Per-bucket cap on how many batches of size --batch-games to play.")
    return p.parse_args()


def _pick_device(arg: str | None) -> str:
    if arg:
        return arg
    env = os.environ.get("GOMOKU_DEVICE")
    if env:
        return env
    return "cpu"


def _score_kl(model, device, examples: list) -> np.ndarray:
    """For each SelfPlayExample, compute KL(p_net || pi_mcts). Higher = net's
    prior disagrees more with what MCTS concluded; that's a position the
    model "got wrong" in the prior sense even if MCTS rescued it."""
    if not examples:
        return np.zeros(0, dtype=np.float32)
    planes = np.stack([ex.planes for ex in examples]).astype(np.float32)
    pi = np.stack([ex.pi for ex in examples]).astype(np.float32)
    out = np.empty(len(examples), dtype=np.float32)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            chunk = 512
            for start in range(0, len(examples), chunk):
                stop = min(start + chunk, len(examples))
                pl = torch.from_numpy(planes[start:stop]).to(device)
                pi_b = torch.from_numpy(pi[start:stop]).to(device)
                logits, _ = model(pl)
                logp = F.log_softmax(logits, dim=-1)
                p = logp.exp()
                kl = (p * (logp - torch.log(pi_b.clamp_min(1e-9)))).sum(dim=-1)
                out[start:stop] = kl.cpu().numpy()
    finally:
        if was_training:
            model.train()
    return out


def _mine_hard_kl_live(model, evaluator, opp_picker, *, target: int, mcts_sims: int,
                       c_puct: float, rng: np.random.Generator,
                       batch_games: int, wave_size: int, max_rounds: int,
                       device: str) -> list[dict]:
    """Play batched games against `opp_picker` (None = selfplay); rank ALL
    model-move examples by KL(p_net || pi_mcts) and keep the global top-K.

    Works regardless of model strength — doesn't rely on losses, just on
    "positions where the net's prior is wrongest about MCTS's conclusion".
    """
    pool: list[tuple[float, dict]] = []  # (kl, row); kept as min-heap-ish
    games_played = 0
    for round_idx in range(max_rounds):
        if opp_picker is None:
            records = generate_games(
                n_games=batch_games, evaluator=evaluator,
                n_simulations=mcts_sims, c_puct=c_puct,
                temperature_moves=0, temperature_final=0.0, dirichlet_eps=0.0,
                augment_symmetries=False, wave_size=wave_size, rng=rng,
            )
        else:
            records = generate_games_vs_baseline(
                n_games=batch_games, evaluator=evaluator, opponent_picker=opp_picker,
                n_simulations=mcts_sims, c_puct=c_puct,
                temperature_moves=0, temperature_final=0.0, dirichlet_eps=0.0,
                augment_symmetries=False, wave_size=wave_size, rng=rng,
            )
        games_played += len(records)
        flat = [ex for r in records for ex in r.examples]
        if not flat:
            continue
        kls = _score_kl(model, device, flat)
        for ex, kl in zip(flat, kls):
            pool.append((float(kl), {
                "planes": ex.planes, "pi": ex.pi, "z": float(ex.z),
                "side": int(ex.side), "ply": int(ex.ply),
            }))
        # Keep only the top-`target` by KL so the list doesn't grow unbounded.
        pool.sort(key=lambda t: -t[0])
        pool = pool[: max(target, 2 * target if round_idx < max_rounds - 1 else target)]
        print(f"  round {round_idx + 1}/{max_rounds}: games={games_played}, "
              f"examples_scored={len(flat)}, pool_size={len(pool)}, "
              f"top_kl={pool[0][0]:.3f}", flush=True)
        if len(pool) >= target and round_idx >= 1:
            # Diminishing returns check: if the top-K is mostly stable, stop.
            # Simple heuristic: stop after 3 rounds.
            if round_idx >= 2:
                break
    pool.sort(key=lambda t: -t[0])
    return [row for _, row in pool[:target]]


def _mine_selfplay(evaluator, *, target_long: int, target_canonical: int,
                   mcts_sims: int, c_puct: float, rng: np.random.Generator,
                   batch_games: int, wave_size: int,
                   max_rounds: int) -> tuple[list[dict], list[dict]]:
    """Batched selfplay; harvest two buckets from each games batch.
       long_defense: positions with ply > 30 in games of plies > 50.
       canonical_opening: positions with ply < 10.
    """
    long_def: list[dict] = []
    canonical: list[dict] = []
    for round_idx in range(max_rounds):
        if len(long_def) >= target_long and len(canonical) >= target_canonical:
            break
        records = generate_games(
            n_games=batch_games,
            evaluator=evaluator,
            n_simulations=mcts_sims,
            c_puct=c_puct,
            temperature_moves=0,
            temperature_final=0.0,
            dirichlet_eps=0.0,
            augment_symmetries=False,
            wave_size=wave_size,
            rng=rng,
        )
        for r in records:
            for ex in r.examples:
                if ex.ply < 10 and len(canonical) < target_canonical:
                    canonical.append({
                        "planes": ex.planes, "pi": ex.pi, "z": float(ex.z),
                        "side": int(ex.side), "ply": int(ex.ply),
                    })
                if r.plies > 50 and ex.ply > 30 and len(long_def) < target_long:
                    long_def.append({
                        "planes": ex.planes, "pi": ex.pi, "z": float(ex.z),
                        "side": int(ex.side), "ply": int(ex.ply),
                    })
    return long_def[:target_long], canonical[:target_canonical]


def _mine_high_kl(model, payload: dict, device: str, *, target: int) -> list[dict]:
    buf = payload.get("replay_buffer")
    if buf is None:
        print("high_kl: WL4 checkpoint has no replay_buffer; skipping bucket")
        return []
    planes_all: torch.Tensor = buf["planes"]
    pi_all: torch.Tensor = buf["pi"]
    z_all: torch.Tensor = buf["z"]
    side_all = buf.get("side")
    ply_all = buf.get("ply")
    n = planes_all.shape[0]
    if n == 0:
        return []
    pool = min(n, max(target * 50, 20_000))
    idx = torch.randperm(n)[:pool]
    chunk = 256
    kls = torch.empty(pool, dtype=torch.float32)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, pool, chunk):
                stop = min(start + chunk, pool)
                pl = planes_all[idx[start:stop]].to(device)
                logits, _ = model(pl)
                logp = F.log_softmax(logits, dim=-1)
                pi_b = pi_all[idx[start:stop]].to(device)
                p = logp.exp()
                # KL(p_net || pi_mcts): penalizes positions where the net is
                # confidently wrong about what MCTS would do.
                pi_safe = pi_b.clamp_min(1e-9)
                kl = (p * (logp - torch.log(pi_safe))).sum(dim=-1)
                kls[start:stop] = kl.cpu()
    finally:
        if was_training:
            model.train()
    top = torch.topk(kls, k=min(target, pool)).indices
    sel = idx[top]
    out: list[dict] = []
    for j in sel.tolist():
        out.append({
            "planes": planes_all[j].cpu().numpy().astype(np.float32),
            "pi": pi_all[j].cpu().numpy().astype(np.float32),
            "z": float(z_all[j].cpu().item()),
            "side": int(side_all[j].cpu().item()) if side_all is not None else 0,
            "ply": int(ply_all[j].cpu().item()) if ply_all is not None else 0,
        })
    return out


def _validate(planes: np.ndarray, pi: np.ndarray, z: np.ndarray) -> None:
    if np.isnan(planes).any():
        raise RuntimeError("NaN in planes")
    if np.isnan(pi).any():
        raise RuntimeError("NaN in pi_mcts")
    if np.isnan(z).any():
        raise RuntimeError("NaN in z")
    sums = pi.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        bad = np.where(np.abs(sums - 1.0) > 1e-3)[0]
        raise RuntimeError(f"pi_mcts mass != 1 at {len(bad)} rows (first={bad[:5]})")


def main() -> None:
    args = parse_args()
    device = _pick_device(args.device)
    print(f"device = {device}")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model, payload = load_checkpoint(args.wl4_checkpoint, device=device)
    model.eval()
    evaluator = make_torch_evaluator(model, device)
    target = args.target_per_bucket
    rows: list[dict] = []
    provenance: list[str] = []

    # hard_kl_* buckets: live-mined top-K positions by KL(p_net || pi_mcts)
    # against each opponent type. Saturated models don't lose to weaker
    # baselines, but they still have positions where their prior diverges
    # from MCTS's conclusion — those are the actual "trouble states".
    hard_sources = [
        ("hard_kl_selfplay", None),
        ("hard_kl_heuristic", heuristic_player),
        ("hard_kl_lookahead2", lookahead_player(depth=2)),
        ("hard_kl_lookahead4", lookahead_player(depth=4)),
    ]
    for tag, opp in hard_sources:
        print(f"mining bucket {tag} (target={target}) ...", flush=True)
        items = _mine_hard_kl_live(
            model, evaluator, opp, target=target,
            mcts_sims=args.mcts_sims, c_puct=args.c_puct, rng=rng,
            batch_games=args.batch_games, wave_size=args.wave_size,
            max_rounds=args.max_rounds, device=device,
        )
        print(f"  -> {len(items)} positions", flush=True)
        rows.extend(items)
        provenance.extend([tag] * len(items))

    print(f"mining buckets long_defense + canonical_opening (target={target} each) ...", flush=True)
    long_def, canonical = _mine_selfplay(
        evaluator, target_long=target, target_canonical=target,
        mcts_sims=args.mcts_sims, c_puct=args.c_puct, rng=rng,
        batch_games=args.batch_games, wave_size=args.wave_size,
        max_rounds=args.max_rounds,
    )
    print(f"  -> long_defense {len(long_def)}, canonical_opening {len(canonical)}", flush=True)
    rows.extend(long_def)
    provenance.extend(["long_defense"] * len(long_def))
    rows.extend(canonical)
    provenance.extend(["canonical_opening"] * len(canonical))

    print(f"mining bucket high_kl (target={target}) ...", flush=True)
    items = _mine_high_kl(model, payload, device, target=target)
    print(f"  -> {len(items)} positions", flush=True)
    rows.extend(items)
    provenance.extend(["high_kl"] * len(items))

    if not rows:
        raise RuntimeError("no positions mined; aborting")

    planes = np.stack([r["planes"] for r in rows]).astype(np.float32)
    pi = np.stack([r["pi"] for r in rows]).astype(np.float32)
    z = np.asarray([r["z"] for r in rows], dtype=np.float32)
    side = np.asarray([r["side"] for r in rows], dtype=np.int8)
    ply = np.asarray([r["ply"] for r in rows], dtype=np.int16)
    _validate(planes, pi, z)

    if planes.shape[1:] != (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE):
        raise RuntimeError(
            f"plane shape mismatch: {planes.shape[1:]} != "
            f"({N_INPUT_PLANES}, {BOARD_SIZE}, {BOARD_SIZE})"
        )
    if pi.shape[1] != N_ACTIONS:
        raise RuntimeError(f"pi shape mismatch: {pi.shape[1]} != {N_ACTIONS}")

    archive = {
        "planes": torch.from_numpy(planes),
        "pi_mcts": torch.from_numpy(pi),
        "z": torch.from_numpy(z),
        "provenance": provenance,
        "side": torch.from_numpy(side),
        "ply": torch.from_numpy(ply),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(archive, out_path)
    print(f"wrote {planes.shape[0]} positions to {out_path}")
    from collections import Counter
    counts = Counter(provenance)
    for tag in sorted(counts):
        print(f"  {tag}: {counts[tag]}")


if __name__ == "__main__":
    main()
