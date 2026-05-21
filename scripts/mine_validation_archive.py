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

Run with GOMOKU_DEVICE=cpu to keep this script sequential and ~30 min wall.
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
    return p.parse_args()


def _pick_device(arg: str | None) -> str:
    if arg:
        return arg
    env = os.environ.get("GOMOKU_DEVICE")
    if env:
        return env
    return "cpu"


def _mcts_label(state: GameState, evaluator, *, n_sims: int, c_puct: float,
                rng: np.random.Generator) -> np.ndarray:
    """Run a single-game MCTS at `state` and return the visit-count policy at tau=1."""
    g = MCTSGame(state, c_puct=c_puct, rng=rng)
    run_batched_mcts([g], evaluator, n_simulations=n_sims, add_root_noise=False)
    return policy_from_visits(g.root, temperature=1.0)


def _play_game_vs_baseline(model_picker, opp_picker, *, rng: np.random.Generator,
                           max_plies: int = N_ACTIONS):
    """Play one game. Return (trajectory, winner_side, n_plies).

    trajectory is a list of (state_before_move, mover_side, action) — one entry
    per ply, in order.
    """
    model_is_black = bool(rng.integers(0, 2) == 0)
    state = GameState.initial()
    trajectory: list[tuple[GameState, int, int]] = []
    winner_side: int | None = None
    ply = 0
    while ply < max_plies:
        side = ply % 2
        model_turn = (side == 0) == model_is_black
        picker = model_picker if model_turn else opp_picker
        action = int(picker(state, rng))
        trajectory.append((state, side, action))
        state = state.apply(action)
        done, term_val = state.is_terminal()
        if done:
            if term_val == -1.0:
                winner_side = side
            break
        ply += 1
    return trajectory, winner_side, len(trajectory), model_is_black


def _play_selfplay_game(model_picker, *, rng: np.random.Generator,
                        max_plies: int = N_ACTIONS):
    state = GameState.initial()
    trajectory: list[tuple[GameState, int, int]] = []
    winner_side: int | None = None
    ply = 0
    while ply < max_plies:
        side = ply % 2
        action = int(model_picker(state, rng))
        trajectory.append((state, side, action))
        state = state.apply(action)
        done, term_val = state.is_terminal()
        if done:
            if term_val == -1.0:
                winner_side = side
            break
        ply += 1
    return trajectory, winner_side, len(trajectory)


def _z_from_perspective(winner_side: int | None, mover_side: int) -> float:
    if winner_side is None:
        return 0.0
    return 1.0 if winner_side == mover_side else -1.0


def _model_mcts_picker(evaluator, *, n_sims: int, c_puct: float):
    """Greedy MCTS picker — argmax visits, no Dirichlet noise."""
    def pick(state: GameState, rng: np.random.Generator) -> int:
        g = MCTSGame(state, c_puct=c_puct, rng=rng)
        run_batched_mcts([g], evaluator, n_simulations=n_sims, add_root_noise=False)
        pi = policy_from_visits(g.root, temperature=0.0)
        return int(np.argmax(pi))
    return pick


def _mine_baseline_losses(evaluator, opp_picker, *, target: int, mcts_sims: int,
                          c_puct: float, rng: np.random.Generator) -> list[dict]:
    """Play games until we have `target` positions captured from games the model
    lost. For each losing game, capture every position where it was the model's
    turn (so we can label pi_mcts there) — that's the position the model had to
    decide from before its last losing move.
    """
    model_picker = _model_mcts_picker(evaluator, n_sims=mcts_sims, c_puct=c_puct)
    out: list[dict] = []
    games_played = 0
    while len(out) < target and games_played < target * 6:
        traj, winner_side, n_plies, model_is_black = _play_game_vs_baseline(
            model_picker, opp_picker, rng=rng,
        )
        games_played += 1
        if winner_side is None:
            continue
        model_side = 0 if model_is_black else 1
        if winner_side == model_side:
            continue
        # Model lost. Capture the LAST model turn — the position right before
        # the final losing move (or the position from which the model failed to
        # block the opponent's winning sequence).
        model_positions = [(state, side, ply_idx) for ply_idx, (state, side, _action)
                           in enumerate(traj) if side == model_side]
        if not model_positions:
            continue
        state, side, ply_idx = model_positions[-1]
        pi = _mcts_label(state, evaluator, n_sims=mcts_sims, c_puct=c_puct, rng=rng)
        out.append({
            "planes": state.to_planes(),
            "pi": pi.astype(np.float32),
            "z": _z_from_perspective(winner_side, side),
            "side": int(side),
            "ply": int(ply_idx),
        })
    return out


def _mine_long_defense(evaluator, *, target: int, mcts_sims: int, c_puct: float,
                       rng: np.random.Generator) -> list[dict]:
    model_picker = _model_mcts_picker(evaluator, n_sims=mcts_sims, c_puct=c_puct)
    out: list[dict] = []
    games_played = 0
    while len(out) < target and games_played < target * 4:
        traj, winner_side, n_plies = _play_selfplay_game(model_picker, rng=rng)
        games_played += 1
        if n_plies <= 50:
            continue
        late_positions = [(s, side, p) for p, (s, side, _) in enumerate(traj) if p > 30]
        rng.shuffle(late_positions)
        for state, side, ply_idx in late_positions[:4]:
            if len(out) >= target:
                break
            pi = _mcts_label(state, evaluator, n_sims=mcts_sims, c_puct=c_puct, rng=rng)
            out.append({
                "planes": state.to_planes(),
                "pi": pi.astype(np.float32),
                "z": _z_from_perspective(winner_side, side),
                "side": int(side),
                "ply": int(ply_idx),
            })
    return out


def _mine_canonical_opening(evaluator, *, target: int, mcts_sims: int, c_puct: float,
                            rng: np.random.Generator) -> list[dict]:
    model_picker = _model_mcts_picker(evaluator, n_sims=mcts_sims, c_puct=c_puct)
    out: list[dict] = []
    while len(out) < target:
        traj, winner_side, n_plies = _play_selfplay_game(model_picker, rng=rng)
        early_positions = [(s, side, p) for p, (s, side, _) in enumerate(traj) if p < 10]
        rng.shuffle(early_positions)
        for state, side, ply_idx in early_positions[:3]:
            if len(out) >= target:
                break
            pi = _mcts_label(state, evaluator, n_sims=mcts_sims, c_puct=c_puct, rng=rng)
            out.append({
                "planes": state.to_planes(),
                "pi": pi.astype(np.float32),
                "z": _z_from_perspective(winner_side, side),
                "side": int(side),
                "ply": int(ply_idx),
            })
    return out


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
            "planes": planes_all[j].numpy().astype(np.float32),
            "pi": pi_all[j].numpy().astype(np.float32),
            "z": float(z_all[j].item()),
            "side": int(side_all[j].item()) if side_all is not None else 0,
            "ply": int(ply_all[j].item()) if ply_all is not None else 0,
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

    baselines = [
        ("heuristic_loss", heuristic_player),
        ("lookahead2_loss", lookahead_player(depth=2)),
        ("lookahead4_loss", lookahead_player(depth=4)),
    ]
    for tag, opp in baselines:
        print(f"mining bucket {tag} (target={target}) ...")
        items = _mine_baseline_losses(
            evaluator, opp, target=target,
            mcts_sims=args.mcts_sims, c_puct=args.c_puct, rng=rng,
        )
        print(f"  -> {len(items)} positions")
        rows.extend(items)
        provenance.extend([tag] * len(items))

    print(f"mining bucket long_defense (target={target}) ...")
    items = _mine_long_defense(
        evaluator, target=target, mcts_sims=args.mcts_sims, c_puct=args.c_puct, rng=rng,
    )
    print(f"  -> {len(items)} positions")
    rows.extend(items)
    provenance.extend(["long_defense"] * len(items))

    print(f"mining bucket canonical_opening (target={target}) ...")
    items = _mine_canonical_opening(
        evaluator, target=target, mcts_sims=args.mcts_sims, c_puct=args.c_puct, rng=rng,
    )
    print(f"  -> {len(items)} positions")
    rows.extend(items)
    provenance.extend(["canonical_opening"] * len(items))

    print(f"mining bucket high_kl (target={target}) ...")
    items = _mine_high_kl(model, payload, device, target=target)
    print(f"  -> {len(items)} positions")
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
