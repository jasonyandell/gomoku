"""The sensei daemon: an always-on, CPU-only eval service backed by a warm
Rapfi pool, plus a flatfile-driven cadence loop.

Two modes, one warm pool:

  * ``serve`` — a FastAPI service. Hold ``size`` warm Rapfi engines + a checkpoint
    evaluator cache; answer ``/eval`` (net vs one opponent), ``/panel`` (net vs the
    configured rulers), and ``/move`` (the master's move for a position) over HTTP.
    Because the pool and the fixed rulers stay warm across requests, a derby or an
    ad-hoc sweep can fire hundreds of evals without paying the engine-respawn /
    weight-reload tax — the 10×+ win that makes always-on eval practical.

  * ``cadence`` — a single long-lived loop that watches one checkpoint, and each
    time its epoch advances by ``--cadence-epochs`` runs the ruler panel and
    appends ONE per-color-split row to a JSONL series. This is the #34 deliverable:
    white is reported separately, and the loop is a pure reducer over the
    append-only series (``last_epoch`` is recovered from the file, never held as
    load-bearing in-process state) — kill it and restart and it picks up exactly
    where it left off.

Both modes run on CPU (``GOMOKU_DEVICE`` defaults to cpu here) so the daemon never
competes with an MPS trainer — the same property that makes the babysit cadence
safe to run during a live "Bruce" run.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

from gomoku.board_config import BOARD_SIZE
from gomoku.eval_panel import (
    EvaluatorCache,
    IDX2_OPENING,
    Ruler,
    eval_vs_ruler,
    fixed_opening_state,
    result_to_row,
    run_panel,
)
from gomoku.game import GameState
from gomoku.rapfi_pool import RapfiPool, default_rapfi_cmd, rapfi_available
from pydantic import BaseModel


# Request bodies live at module scope (FastAPI only treats a Pydantic model as a
# request BODY when it is module-level, not defined inside the app factory).
class EvalRequest(BaseModel):
    checkpoint: str
    opponent: str = "rapfi"
    n_games: int = 16
    sims: int = 160
    c_puct: float = 1.5
    opening: str = "idx2"
    temp_until_ply: int = 0
    temperature: float = 1.0
    rapfi_timeout_ms: int = 1000
    seed: int = 0


class PanelRequest(BaseModel):
    checkpoint: str
    epoch: int | None = None
    opening: str | None = None
    seed: int = 0


class MoveRequest(BaseModel):
    history: list[int] = []
    opening: str = "empty"


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def parse_opening(spec: str) -> GameState | None:
    """``"idx2"`` | ``"empty"`` | ``"r,c;r,c;..."`` → start state (None = initial)."""
    spec = (spec or "idx2").strip().lower()
    if spec in ("empty", "initial", "none", ""):
        return None
    if spec == "idx2":
        return fixed_opening_state(IDX2_OPENING)
    moves = []
    for part in spec.split(";"):
        r, c = part.split(",")
        moves.append((int(r), int(c)))
    return fixed_opening_state(tuple(moves))


def parse_ruler_specs(specs: list[str]) -> list[tuple[str, str]]:
    """Each spec is ``LABEL=OPPONENT`` (``rapfi``, a ``*.pt`` path, or a baseline
    spec like ``lookahead:depth=4``)."""
    out = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"ruler spec must be LABEL=OPPONENT, got {s!r}")
        label, opp = s.split("=", 1)
        out.append((label.strip(), opp.strip()))
    return out


def build_rulers(
    specs: list[tuple[str, str]],
    *,
    n_games: int,
    sims: int,
    temp_until_ply: int,
    temperature: float,
    rapfi_timeout_ms: int,
    drop_rapfi_if_unavailable: bool = True,
) -> list[Ruler]:
    rulers = []
    for label, opp in specs:
        if opp == "rapfi" and drop_rapfi_if_unavailable and not rapfi_available():
            print(f"  (skipping ruler {label}: Rapfi unavailable)", file=sys.stderr)
            continue
        # Rapfi supplies its own move variety via timeout wobble, so the net plays
        # deterministically against it; net-vs-net / baseline rulers need early-ply
        # sampling to get any game variety at a fixed opening.
        tup = 0 if opp == "rapfi" else temp_until_ply
        rulers.append(
            Ruler(
                label=label,
                opponent=opp,
                n_games=n_games,
                sims=sims,
                temp_until_ply=tup,
                temperature=temperature,
                rapfi_timeout_ms=rapfi_timeout_ms,
            )
        )
    return rulers


def needs_pool(rulers: list[Ruler]) -> bool:
    return any(r.opponent == "rapfi" for r in rulers)


# --------------------------------------------------------------------------
# serve mode — FastAPI
# --------------------------------------------------------------------------
def create_app(
    *,
    rulers: list[Ruler],
    opening: GameState | None,
    pool: RapfiPool | None,
    c_puct: float = 1.5,
):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException

    cache = EvaluatorCache()

    @asynccontextmanager
    async def lifespan(app: "FastAPI"):
        try:
            yield
        finally:
            if app.state.pool is not None:
                app.state.pool.close()

    app = FastAPI(title="gomoku-sensei", lifespan=lifespan)
    app.state.pool = pool
    app.state.cache = cache

    @app.get("/health")
    def health():
        p = app.state.pool
        return {
            "ok": True,
            "board_size": BOARD_SIZE,
            "pool_size": (p.size if p is not None else 0),
            "rapfi_available": rapfi_available(),
            "rulers": [r.label for r in rulers],
        }

    @app.post("/eval")
    def do_eval(req: EvalRequest):
        try:
            start = parse_opening(req.opening)
            ruler = Ruler(
                label=req.opponent,
                opponent=req.opponent,
                n_games=req.n_games,
                sims=req.sims,
                temp_until_ply=req.temp_until_ply,
                temperature=req.temperature,
                rapfi_timeout_ms=req.rapfi_timeout_ms,
            )
            res = eval_vs_ruler(
                req.checkpoint,
                ruler,
                cache=app.state.cache,
                pool=app.state.pool,
                start_state=start,
                c_puct=req.c_puct,
                seed=req.seed,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return result_to_row(req.opponent, res)

    @app.post("/panel")
    def do_panel(req: PanelRequest):
        try:
            start = parse_opening(req.opening) if req.opening is not None else opening
            return run_panel(
                req.checkpoint,
                rulers,
                cache=app.state.cache,
                pool=app.state.pool,
                start_state=start,
                c_puct=c_puct,
                seed=req.seed,
                epoch=req.epoch,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/move")
    def do_move(req: MoveRequest):
        """The teacher's move for a position (board reached by applying history
        from the given opening). Demonstrates the warm pool; handy for ad-hoc
        teacher labelling / debugging."""
        if app.state.pool is None:
            raise HTTPException(status_code=503, detail="no Rapfi pool")
        state = parse_opening(req.opening) or GameState.initial()
        try:
            for a in req.history:
                state = state.apply(int(a))
            move = app.state.pool.pick(state)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
        return {"move": int(move), "r": int(move) // BOARD_SIZE, "c": int(move) % BOARD_SIZE}

    return app


def serve(args: argparse.Namespace) -> int:
    os.environ.setdefault("GOMOKU_DEVICE", "cpu")
    specs = parse_ruler_specs(args.ruler) if args.ruler else _default_ruler_specs()
    rulers = build_rulers(
        specs,
        n_games=args.n_games,
        sims=args.sims,
        temp_until_ply=args.temp_until_ply,
        temperature=args.temperature,
        rapfi_timeout_ms=args.rapfi_timeout_ms,
    )
    opening = parse_opening(args.opening)
    # Warm a pool in serve mode whenever Rapfi is present, so /move and ad-hoc
    # rapfi /eval work regardless of the default rulers (not only when a rapfi
    # ruler is configured).
    pool = None
    if rapfi_available():
        pool = RapfiPool(
            size=args.pool_size,
            cmd=args.rapfi_cmd or default_rapfi_cmd(),
            timeout_ms=args.rapfi_timeout_ms,
            board_size=BOARD_SIZE,
        )
        print(f"warmed Rapfi pool: size={args.pool_size} @ {args.rapfi_timeout_ms}ms")
    elif needs_pool(rulers):
        print("WARNING: rapfi rulers configured but Rapfi unavailable", file=sys.stderr)
    import uvicorn

    print(f"serving sensei on {args.host}:{args.port} (board_size={BOARD_SIZE})")
    # The lifespan handler closes the pool on a clean shutdown; the try/finally
    # covers a startup failure (e.g. port in use) before lifespan ever runs.
    app = create_app(rulers=rulers, opening=opening, pool=pool, c_puct=args.c_puct)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if pool is not None:
            pool.close()
    return 0


# --------------------------------------------------------------------------
# cadence mode — the #34 series writer
# --------------------------------------------------------------------------
def _last_epoch_in_series(path: str) -> int | None:
    """Recover the last evaluated epoch from the append-only series (reducer)."""
    if not os.path.isfile(path):
        return None
    last = None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep = row.get("epoch")
                if isinstance(ep, int):
                    last = ep if last is None else max(last, ep)
    except OSError:
        return None
    return last


def _append_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def cadence(args: argparse.Namespace) -> int:
    os.environ.setdefault("GOMOKU_DEVICE", "cpu")
    specs = parse_ruler_specs(args.ruler) if args.ruler else _default_ruler_specs()
    rulers = build_rulers(
        specs,
        n_games=args.n_games,
        sims=args.sims,
        temp_until_ply=args.temp_until_ply,
        temperature=args.temperature,
        rapfi_timeout_ms=args.rapfi_timeout_ms,
    )
    if not rulers:
        print("ERROR: no usable rulers", file=sys.stderr)
        return 2
    opening = parse_opening(args.opening)
    cache = EvaluatorCache()
    pool = None
    if needs_pool(rulers):
        pool = RapfiPool(
            size=args.pool_size,
            cmd=args.rapfi_cmd or default_rapfi_cmd(),
            timeout_ms=args.rapfi_timeout_ms,
            board_size=BOARD_SIZE,
        )

    stop = {"req": False}

    def _sig(*_a):
        stop["req"] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    last_epoch = _last_epoch_in_series(args.series_out)
    print(
        f"cadence: watching {args.checkpoint} every {args.cadence_epochs} epochs "
        f"-> {args.series_out} (resuming from epoch {last_epoch}); "
        f"rulers={[r.label for r in rulers]}"
    )
    last_mtime = -1.0
    try:
        while not stop["req"]:
            try:
                mtime = os.path.getmtime(args.checkpoint)
            except OSError:
                time.sleep(args.interval_s)
                continue
            if mtime != last_mtime:
                last_mtime = mtime
                try:
                    ep = cache.epoch(args.checkpoint)
                except Exception as e:
                    print(f"  (skip: cannot read epoch: {e})", file=sys.stderr)
                    ep = None
                if ep is not None and (
                    last_epoch is None or ep - last_epoch >= args.cadence_epochs
                ):
                    t0 = time.time()
                    row = run_panel(
                        args.checkpoint,
                        rulers,
                        cache=cache,
                        pool=pool,
                        start_state=opening,
                        c_puct=args.c_puct,
                        seed=args.seed,
                        epoch=ep,
                    )
                    row["eval_s"] = round(time.time() - t0, 2)
                    _append_row(args.series_out, row)
                    last_epoch = ep
                    summary = " ".join(
                        f"{r.label}={row.get(f'{r.label}__score', '?')}"
                        f"(wL={row.get(f'{r.label}__white_loss_rate', '?')})"
                        for r in rulers
                    )
                    print(f"e{ep} [{row['eval_s']}s] {summary}", flush=True)
            # SIGTERM is checked between eval passes and every ~1s while idle
            # (a panel in progress runs to completion before the loop exits).
            slept = 0.0
            while slept < args.interval_s and not stop["req"]:
                time.sleep(min(1.0, args.interval_s - slept))
                slept += 1.0
    finally:
        if pool is not None:
            pool.close()
    print("cadence: stopped")
    return 0


# --------------------------------------------------------------------------
def _default_ruler_specs() -> list[tuple[str, str]]:
    """Always-available default panel (no machine-specific anchor paths)."""
    return [("rapfi", "rapfi"), ("heuristic", "heuristic"), ("lookahead4", "lookahead:depth=4")]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ruler", action="append", default=[],
                   help="repeatable LABEL=OPPONENT (rapfi | *.pt | baseline spec). "
                        "Defaults to rapfi+heuristic+lookahead4 if omitted.")
    p.add_argument("--opening", type=str, default="idx2",
                   help="'idx2' | 'empty' | 'r,c;r,c;...'")
    p.add_argument("--n-games", type=int, default=16)
    p.add_argument("--sims", type=int, default=160)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--temp-until-ply", type=int, default=9,
                   help="net-vs-net/baseline rulers sample moves before this ply "
                        "for opening variety (rapfi rulers ignore it)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--pool-size", type=int, default=4)
    p.add_argument("--rapfi-cmd", type=str, default=None)
    p.add_argument("--rapfi-timeout-ms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="gomoku sensei: always-on eval + teacher daemon")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="FastAPI eval service (warm Rapfi pool)")
    _add_common(s)
    s.add_argument("--host", type=str, default="127.0.0.1")
    s.add_argument("--port", type=int, default=8008)

    c = sub.add_parser("cadence", help="watch a checkpoint, append a per-color series")
    _add_common(c)
    c.add_argument("--checkpoint", required=True, help="checkpoint to watch + eval")
    c.add_argument("--series-out", required=True, help="JSONL series to append")
    c.add_argument("--cadence-epochs", type=int, default=50)
    c.add_argument("--interval-s", type=float, default=60.0)

    args = p.parse_args(argv)
    if args.cmd == "serve":
        return serve(args)
    if args.cmd == "cadence":
        return cadence(args)
    p.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
