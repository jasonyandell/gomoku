"""FastAPI server: list checkpoints, play live, and generate self-play replays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gomoku.game import BOARD_SIZE, N_ACTIONS, GameState, action_to_str
from gomoku.mcts import MCTSGame, make_torch_evaluator, policy_from_visits, run_batched_mcts
from gomoku.model import load_checkpoint
from gomoku.self_play import generate_games
from gomoku.util import pick_device


STATIC_DIR = Path(__file__).parent / "static"

# ---- Model cache: load each checkpoint once, share across requests ----
_models: dict[str, tuple[Any, Any]] = {}  # path -> (model, evaluator)
_lock = Lock()


def _device() -> torch.device:
    return pick_device(os.environ.get("GOMOKU_DEVICE"))


def _load(ckpt_path: str):
    with _lock:
        if ckpt_path in _models:
            return _models[ckpt_path]
        device = _device()
        model, _payload = load_checkpoint(ckpt_path, device=device)
        model.eval()
        evaluator = make_torch_evaluator(model, device)
        _models[ckpt_path] = (model, evaluator)
        return _models[ckpt_path]


def _scan_checkpoints(root: Path) -> list[dict]:
    out = []
    if not root.exists():
        return out
    for p in sorted(root.glob("*.pt")):
        if p.is_symlink():
            continue
        try:
            stat = p.stat()
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            out.append({
                "name": p.name,
                "path": str(p.resolve()),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "mtime": stat.st_mtime,
                "epoch": int(ckpt.get("epoch", 0)),
                "total_games": int(ckpt.get("total_games", 0)),
                "model_size": _infer_size(ckpt.get("model_config", {})),
            })
        except Exception as e:
            out.append({"name": p.name, "path": str(p.resolve()), "error": str(e)})
    out.sort(key=lambda d: d.get("epoch", 0))
    return out


def _infer_size(cfg: dict) -> str:
    f = cfg.get("n_filters")
    b = cfg.get("n_blocks")
    if f == 32 and b == 2:
        return "tiny"
    if f == 64 and b == 4:
        return "small"
    if f == 96 and b == 6:
        return "medium"
    if f == 128 and b == 10:
        return "large"
    return f"custom(f={f},b={b})"


def _rebuild_state(moves: list[int]) -> GameState:
    state = GameState.initial()
    for m in moves:
        if not state.legal_mask()[m]:
            raise ValueError(f"illegal move {m} in history")
        state = state.apply(m)
    return state


def _xy_for(action: int) -> dict:
    return {"action": int(action), "r": action // BOARD_SIZE, "c": action % BOARD_SIZE, "name": action_to_str(action)}


class MoveRequest(BaseModel):
    checkpoint: str
    history: list[int]
    n_simulations: int = 200
    c_puct: float = 1.5
    top_k: int = 6


class SelfPlayRequest(BaseModel):
    checkpoint: str
    n_simulations: int = 200
    c_puct: float = 1.5
    temperature_moves: int = 0
    dirichlet_eps: float = 0.0
    seed: int | None = None


def create_app(checkpoints_dir: str = "checkpoints") -> FastAPI:
    app = FastAPI(title="gomoku")
    ckpt_root = Path(checkpoints_dir)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/checkpoints")
    def list_checkpoints():
        return {"dir": str(ckpt_root.resolve()), "checkpoints": _scan_checkpoints(ckpt_root)}

    @app.post("/api/move")
    def post_move(req: MoveRequest):
        try:
            state = _rebuild_state(req.history)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        done, term_val = state.is_terminal()
        if done:
            return {"done": True, "value": term_val}
        model, evaluator = _load(req.checkpoint)
        g = MCTSGame(state, c_puct=req.c_puct)
        run_batched_mcts([g], evaluator, n_simulations=req.n_simulations, add_root_noise=False)
        visits = g.root.N
        order = np.argsort(-visits)
        top = []
        for a in order[:req.top_k]:
            n = int(visits[a])
            if n == 0:
                continue
            q = float(g.root.W[a] / max(g.root.N[a], 1))
            top.append({**_xy_for(int(a)), "visits": n, "q": q})
        best = int(np.argmax(visits))
        pi = policy_from_visits(g.root, temperature=0.0)
        return {
            "done": False,
            "best": _xy_for(best),
            "value": float(g.root.W[best] / max(g.root.N[best], 1)) if visits.sum() > 0 else 0.0,
            "top": top,
            "total_visits": int(visits.sum()),
        }

    @app.post("/api/selfplay")
    def post_selfplay(req: SelfPlayRequest):
        _model, evaluator = _load(req.checkpoint)
        rng = np.random.default_rng(req.seed)
        records = generate_games(
            1,
            evaluator,
            n_simulations=req.n_simulations,
            c_puct=req.c_puct,
            temperature_moves=req.temperature_moves,
            dirichlet_alpha=0.3,
            dirichlet_eps=req.dirichlet_eps,
            rng=rng,
            augment_symmetries=False,
        )
        rec = records[0]
        # Reconstruct the move sequence by replaying the trajectory. The trajectory's
        # examples may include augmentation, but since we set augment_symmetries=False,
        # there's exactly one example per ply with the original board.
        # Easier: re-derive moves by replaying argmax(pi) from each example... but that's
        # only deterministic if temperature_moves=0. For generality, return all per-ply data
        # by re-generating the game while recording moves directly.
        # Simpler: do a second pass that captures the move list explicitly.
        return {
            "plies": rec.plies,
            "outcome": rec.outcome,
            "moves": _replay_moves_from_examples(rec.examples),
        }

    @app.post("/api/selfplay-detailed")
    def post_selfplay_detailed(req: SelfPlayRequest):
        """Generates a single game and returns per-ply (move, top_moves, value)."""
        _model, evaluator = _load(req.checkpoint)
        rng = np.random.default_rng(req.seed)
        state = GameState.initial()
        moves: list[dict] = []
        side = 0  # 0 = first mover (black-equivalent), 1 = second mover
        ply = 0
        while True:
            g = MCTSGame(state, c_puct=req.c_puct,
                         dirichlet_alpha=0.3, dirichlet_eps=req.dirichlet_eps,
                         rng=np.random.default_rng(rng.integers(0, 2**31)))
            run_batched_mcts([g], evaluator, n_simulations=req.n_simulations,
                             add_root_noise=(req.dirichlet_eps > 0))
            tau = 1.0 if ply < req.temperature_moves else 0.0
            pi = policy_from_visits(g.root, tau)
            if tau <= 0:
                action = int(np.argmax(g.root.N))
            else:
                s = pi.sum()
                action = int(rng.choice(len(pi), p=pi / s)) if s > 0 else int(np.argmax(g.root.N))

            visits = g.root.N
            order = np.argsort(-visits)
            top = []
            for a in order[:6]:
                n = int(visits[a])
                if n == 0:
                    continue
                q = float(g.root.W[a] / max(g.root.N[a], 1))
                top.append({**_xy_for(int(a)), "visits": n, "q": q})

            moves.append({
                **_xy_for(action),
                "side": side,
                "top": top,
                "value": float(g.root.W[action] / max(g.root.N[action], 1)) if visits.sum() > 0 else 0.0,
            })

            state = state.apply(action)
            done, term_val = state.is_terminal()
            if done:
                if term_val == -1.0:
                    outcome = 1.0 if side == 0 else -1.0  # +1 if first-mover won
                else:
                    outcome = 0.0
                return {
                    "plies": ply + 1,
                    "outcome": outcome,
                    "moves": moves,
                    "terminated_by": "win" if term_val == -1.0 else "draw",
                }
            side = 1 - side
            ply += 1
            if ply >= 81:
                return {"plies": ply, "outcome": 0.0, "moves": moves, "terminated_by": "max_plies"}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _replay_moves_from_examples(examples) -> list[dict]:
    """Fallback when /api/selfplay is called: derive moves by argmax of policy targets.

    This is only correct when temperature_moves=0 (greedy throughout). For temperature>0,
    use /api/selfplay-detailed instead.
    """
    out = []
    for i, ex in enumerate(examples):
        a = int(np.argmax(ex.pi))
        out.append({**_xy_for(a), "side": i % 2})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    import uvicorn
    app = create_app(args.checkpoints_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
