"""Batched eval arena: play a whole match's games CONCURRENTLY (issue #105).

`play_match_pickers` plays games one at a time, so every model move runs MCTS
over a single-game list — every simulation is a batch-size-1 forward pass. A
40-game eval at sims=100 is ~10^5 sequential batch-1 MPS dispatches, each
paying full launch/sync latency at ~1% occupancy. The existing "parallel"
paths multiply *processes* (each re-loading the model, still batch-1 inside).

The arena flips eval into the self-play regime, generalizing the pattern
proven by `gomoku/rapfimine/fast_eval.py` (~minutes -> ~20s):

  * all games live at once in ONE process (one resident model per checkpoint);
  * each lockstep round partitions active games by which agent is to move:
      - net-to-move games -> ONE `run_batched_mcts_waves` call across the whole
        subset (batch ~= n_games x wave_size, exactly like self-play);
      - engine-to-move games (rapfi@Nms etc.) -> the warm engine pool's
        parallel `label_states`; CPU pickers (heuristic/lookahead) -> a loop;
  * the two sides run concurrently, so MPS (net) and CPU (engine) overlap and
    a round costs the slower side, not the sum.

Wall-clock ~= the longest single game's move chain, not the sum of games.

Semantics match `play_match_pickers`: A alternates colors per game, paired
random openings share a position across the color swap, draws = half-wins.
Per-game RNG is independent (seeded seed+idx+1, same as `play_match_parallel`)
rather than one RNG threaded across games — unbiased, not byte-identical to
the sequential path. Net moves are greedy (temperature 0, no root noise), a
fresh tree per move — the legacy eval default. Wave batching (`wave_size`>1)
is the same soft-virtual-loss search self-play uses; wave=1 recovers exactly
`run_batched_mcts` if strict fidelity is ever needed.

This is EVAL-ONLY. Self-play / generation / training paths are untouched.

CLI:
    gomoku-arena "model:checkpoint=checkpoints/latest.pt,sims=100" vs rapfi@50ms \
        --n-games 100
    gomoku-arena "model:checkpoint=a.pt" vs "model:checkpoint=b.pt" --n-games 120 \
        --random-opening-moves 4
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from gomoku.eval import MatchResult, _match_result_from_outcomes
from gomoku.game import BOARD_SIZE, GameState

Rng = np.random.Generator


# ---------------- Batched agents ----------------
#
# An agent answers "given these positions (all yours-to-move), what do you
# play in each?" — the batch seam that lets the arena batch net inference
# across games and fan engine moves across a warm pool.


class BatchedAgent:
    """Base: pick one action per state. Override `pick_batch`."""

    label: str = "?"

    def pick_batch(self, states: list[GameState], rngs: list[Rng]) -> list[int]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class NetAgent(BatchedAgent):
    """MCTS over a leaf evaluator, batched across all to-move games.

    Fresh tree per move, no root noise, greedy pick — identical decision rule
    to the legacy `mcts_picker` default, but the search itself batches leaves
    across every game in the subset (x wave_size with soft virtual loss).
    """

    def __init__(
        self,
        evaluator,
        *,
        sims: int = 100,
        c_puct: float = 1.5,
        wave_size: int = 16,
        label: str = "model",
    ) -> None:
        self.evaluator = evaluator
        self.sims = sims
        self.c_puct = c_puct
        self.wave_size = wave_size
        self.label = label

    def pick_batch(self, states: list[GameState], rngs: list[Rng]) -> list[int]:
        from gomoku.mcts import MCTSGame, policy_from_visits, run_batched_mcts_waves

        games = [
            MCTSGame(s, c_puct=self.c_puct, rng=r) for s, r in zip(states, rngs)
        ]
        run_batched_mcts_waves(
            games,
            self.evaluator,
            n_simulations=self.sims,
            wave_size=self.wave_size,
            add_root_noise=False,
        )
        return [
            int(np.argmax(policy_from_visits(g.root, temperature=0.0)))
            for g in games
        ]


class PickerAgent(BatchedAgent):
    """Adapter for a scalar `Picker` (random/heuristic/lookahead/...).

    Pure-Python pickers are GIL-bound, so a thread fan-out buys nothing —
    a plain loop keeps it simple and deterministic. For expensive pickers
    (lookahead) use `PooledPickerAgent`, which sidesteps the GIL with a
    process pool.
    """

    def __init__(self, picker, *, label: str) -> None:
        self.picker = picker
        self.label = label

    def pick_batch(self, states: list[GameState], rngs: list[Rng]) -> list[int]:
        return [int(self.picker(s, r)) for s, r in zip(states, rngs)]


# ---------------- Pooled CPU pickers (issue #110) ----------------
#
# Within an arena round every to-move game's pick is independent, so an
# expensive CPU picker (lookahead) fans across a persistent spawn pool.
# Workers build the picker from (kind, kwargs) at init and never import
# torch — `gomoku.baselines` is deliberately torch-free — so worker startup
# is cheap and there is no Metal/MPS anywhere near the pool.

_POOL_PICKER = None  # per-worker picker, set once by the pool initializer


def _torch_free_picker(kind: str, kwargs: dict[str, str]):
    """Build a picker from the torch-free subset of the `gomoku.match` spec
    kinds. Raises for kinds that need torch/subprocesses (model/external)."""
    from gomoku import baselines

    if kind == "random":
        return baselines.random_player
    if kind == "heuristic":
        return baselines.heuristic_player
    if kind == "defensive":
        return baselines.defensive_player
    if kind == "pacifist":
        max_own = int(kwargs.get("max_own_line", "3"))

        def _pacifist(state, rng, *, _m=max_own):
            return baselines.pacifist_blocker(state, rng, max_own_line=_m)

        return _pacifist
    if kind == "lookahead":
        return baselines.lookahead_player(depth=int(kwargs.get("depth", "4")))
    raise ValueError(f"no torch-free picker for kind {kind!r}")


def _picker_pool_init(kind: str, kwargs: dict[str, str]) -> None:
    global _POOL_PICKER
    _POOL_PICKER = _torch_free_picker(kind, kwargs)


def _picker_pool_pick(task: tuple[GameState, int]) -> int:
    state, seed = task
    return int(_POOL_PICKER(state, np.random.default_rng(seed)))


def default_picker_workers() -> int:
    """Worker count for pooled pickers: `GOMOKU_PICKER_WORKERS` env override,
    else leave headroom for the trainer/derby sharing this machine."""
    import os

    env = os.environ.get("GOMOKU_PICKER_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(12, (os.cpu_count() or 8) - 4))


class PooledPickerAgent(BatchedAgent):
    """A torch-free picker fanned across a persistent spawn-Pool.

    Each pick ships (state, seed) — a 9x9 state is tiny next to a lookahead4
    tree (~20k Python node visits), so IPC is noise. The seed is drawn from
    the slot's rng in the parent, so same-seed runs stay deterministic and
    results map back by index. The pool is created lazily on first use and
    persists across rounds (and matches, if the agent is reused).
    """

    def __init__(
        self,
        kind: str,
        kwargs: dict[str, str] | None = None,
        *,
        n_workers: int | None = None,
        label: str,
    ) -> None:
        _torch_free_picker(kind, dict(kwargs or {}))  # fail fast on bad kind
        self.kind = kind
        self.kwargs = dict(kwargs or {})
        self.n_workers = n_workers if n_workers else default_picker_workers()
        self.label = label
        self._pool = None

    def _ensure_pool(self):
        if self._pool is None:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                self.n_workers,
                initializer=_picker_pool_init,
                initargs=(self.kind, self.kwargs),
            )
        return self._pool

    def pick_batch(self, states: list[GameState], rngs: list[Rng]) -> list[int]:
        pool = self._ensure_pool()
        tasks = [(s, int(r.integers(2**63))) for s, r in zip(states, rngs)]
        # chunksize=1: tasks are ~ms-scale, so best load balance wins.
        return [int(a) for a in pool.map(_picker_pool_pick, tasks, chunksize=1)]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def picker_agent_from_spec(
    spec,
    *,
    picker_workers: int | None = None,
) -> BatchedAgent:
    """Arena agent for a non-model `PlayerSpec`: pool the expensive kinds
    (lookahead), plain loop for the cheap ones (heuristic is ~0.03 ms/move —
    IPC would cost more than it saves)."""
    workers = picker_workers if picker_workers is not None else default_picker_workers()
    if spec.kind == "lookahead" and workers > 1:
        return PooledPickerAgent(
            spec.kind, spec.kwargs, n_workers=workers, label=spec.label()
        )
    from gomoku.match import build_player

    return PickerAgent(build_player(spec), label=spec.label())


class EnginePoolAgent(BatchedAgent):
    """A warm pool of external Gomocup engines (`RapfiPool`), fanned per move.

    `label_states` runs one timed pick per engine in parallel threads (the
    GIL is released during blocking pipe reads), so a round costs ~one move's
    think time regardless of how many games the engine is to move in — up to
    the pool size.
    """

    def __init__(self, pool, *, label: str) -> None:
        self.pool = pool
        self.label = label

    def pick_batch(self, states: list[GameState], rngs: list[Rng]) -> list[int]:
        return [int(a) for a in self.pool.label_states(states)]

    def close(self) -> None:
        self.pool.close()


# ---------------- Spec parsing / agent construction ----------------

_RAPFI_SUGAR = re.compile(r"^rapfi@(\d+)ms$")


def net_agent_from_checkpoint(
    checkpoint: str,
    *,
    sims: int = 100,
    c_puct: float = 1.5,
    wave_size: int = 16,
    device: str | None = None,
    label: str | None = None,
) -> NetAgent:
    """Load a checkpoint into a warm `NetAgent` (load -> fuse -> warmup)."""
    import os

    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint
    from gomoku.util import pick_device

    dev = pick_device(device or os.environ.get("GOMOKU_DEVICE"))
    model, _ = load_checkpoint(checkpoint, device=dev)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, dev)
    # Warmup forward — eat the MPS graph-compile latency here, not on the
    # first move of the first game.
    evaluator([GameState.initial()])
    return NetAgent(
        evaluator,
        sims=sims,
        c_puct=c_puct,
        wave_size=wave_size,
        label=label or f"model:{checkpoint}",
    )


def build_agent(
    spec_str: str,
    *,
    device: str | None = None,
    default_sims: int = 100,
    wave_size: int = 16,
    engine_pool_size: int = 8,
    picker_workers: int | None = None,
) -> BatchedAgent:
    """Build a `BatchedAgent` from a player spec string.

    Accepts the `gomoku.match` spec grammar (`kind[:k=v,...]`) plus:
      * `rapfi[:timeout_ms=N,size=K,cmd=PATH]` — warm RapfiPool opponent
        (cmd resolves to the local build or the pinned HF snapshot);
      * `rapfi@50ms` — sugar for `rapfi:timeout_ms=50`;
      * `model:...` additionally takes `wave=K` (default `wave_size`).
    `external:cmd=...` also gets a warm pool (RapfiPool is engine-agnostic).
    """
    spec_str = spec_str.strip()
    m = _RAPFI_SUGAR.match(spec_str)
    if m:
        spec_str = f"rapfi:timeout_ms={m.group(1)}"

    if spec_str.split(":", 1)[0].strip().lower() == "rapfi":
        # Not a gomoku.match kind — parse kwargs by hand, same k=v grammar.
        kwargs: dict[str, str] = {}
        if ":" in spec_str:
            for part in spec_str.split(":", 1)[1].split(","):
                part = part.strip()
                if not part:
                    continue
                k, _, v = part.partition("=")
                kwargs[k.strip()] = v.strip()
        from gomoku.rapfi_pool import RapfiPool

        timeout_ms = int(kwargs.get("timeout_ms", "1000"))
        size = int(kwargs.get("size", str(engine_pool_size)))
        pool = RapfiPool(
            size=size,
            cmd=kwargs.get("cmd"),
            timeout_ms=timeout_ms,
            board_size=BOARD_SIZE,
        )
        return EnginePoolAgent(pool, label=f"rapfi@{timeout_ms}ms")

    from gomoku.match import parse_spec

    spec = parse_spec(spec_str)

    if spec.kind == "model":
        checkpoint = spec.kwargs.get("checkpoint")
        if not checkpoint:
            raise SystemExit(f"model spec needs checkpoint=PATH: {spec_str!r}")
        sims = int(spec.kwargs.get("sims", str(default_sims)))
        c_puct = float(spec.kwargs.get("c_puct", "1.5"))
        wave = int(spec.kwargs.get("wave", str(wave_size)))
        return net_agent_from_checkpoint(
            checkpoint,
            sims=sims,
            c_puct=c_puct,
            wave_size=wave,
            device=device,
            label=spec.label(),
        )

    if spec.kind == "external":
        from gomoku.rapfi_pool import RapfiPool

        cmd = spec.kwargs.get("cmd")
        if not cmd:
            raise SystemExit(f"external spec needs cmd=PATH: {spec_str!r}")
        pool = RapfiPool(
            size=int(spec.kwargs.get("size", str(engine_pool_size))),
            cmd=cmd,
            timeout_ms=int(spec.kwargs.get("timeout_ms", "1000")),
            board_size=int(spec.kwargs.get("size_board", str(BOARD_SIZE))),
            rule=int(spec.kwargs.get("rule", "0")),
        )
        return EnginePoolAgent(pool, label=spec.label())

    return picker_agent_from_spec(spec, picker_workers=picker_workers)


# ---------------- The arena ----------------


@dataclass
class _Slot:
    """One live game in the arena."""

    idx: int
    a_is_black: bool
    a_to_move: bool
    state: GameState
    ply: int
    rng: Rng
    done: bool = False
    winner_side: int | None = field(default=None)  # 0=black, 1=white, None=draw


def _build_slots(
    n_games: int,
    *,
    seed: int,
    random_opening_moves: int = 0,
    start_state: GameState | None = None,
    opening_states: list[GameState] | None = None,
) -> list[_Slot]:
    """Set up per-game slots with `play_match_pickers`-compatible semantics:
    A alternates colors by game index; a fixed `start_state` wins over random
    openings; each color-swap pair (2k, 2k+1) shares one random opening; an
    odd-ply opening means white is to move, flipping the base assignment.

    ``opening_states`` supplies the per-pair openings EXPLICITLY (pair k =
    games 2k and 2k+1) — the seam callers with their own opening derivation
    use (e.g. the delta-e H2H gate regenerates each pair's opening from
    ``seed + pair_idx``). Precedence: start_state > opening_states >
    random_opening_moves."""
    if opening_states is not None and start_state is None:
        need = (n_games + 1) // 2
        if len(opening_states) < need:
            raise ValueError(
                f"opening_states has {len(opening_states)} entries; "
                f"{n_games} games need {need} pairs"
            )
    elif start_state is None and random_opening_moves > 0:
        from gomoku.self_play import _random_opening_state

        opening_rng = np.random.default_rng(seed)
        generated: list[GameState] = []
        for _ in range((n_games + 1) // 2):
            opening_state, _moves = _random_opening_state(
                opening_rng, random_opening_moves
            )
            generated.append(opening_state)
        opening_states = generated

    slots: list[_Slot] = []
    for g_idx in range(n_games):
        a_is_black = g_idx % 2 == 0
        if start_state is not None:
            state = start_state
        elif opening_states is not None:
            state = opening_states[g_idx // 2]
        else:
            state = GameState.initial()
        ply = state.move_count
        a_to_move = a_is_black if ply % 2 == 0 else not a_is_black
        slots.append(
            _Slot(
                idx=g_idx,
                a_is_black=a_is_black,
                a_to_move=a_to_move,
                state=state,
                ply=ply,
                rng=np.random.default_rng(seed + g_idx + 1),
            )
        )
    return slots


def _advance_slots(group: list[_Slot], actions: list[int]) -> None:
    """Apply one picked action per slot, updating terminal/turn state."""
    for slot, action in zip(group, actions):
        side_just_moved = slot.ply % 2
        slot.state = slot.state.apply(int(action))
        done, v = slot.state.is_terminal()
        if done:
            slot.done = True
            if v == -1.0:
                slot.winner_side = side_just_moved
        else:
            slot.a_to_move = not slot.a_to_move
            slot.ply += 1


def _result_from_slots(slots: list[_Slot]) -> MatchResult:
    outcomes: list[tuple[bool, str]] = []
    for s in slots:
        if s.winner_side is None:
            outcomes.append((s.a_is_black, "draw"))
        else:
            a_side = 0 if s.a_is_black else 1
            outcomes.append(
                (s.a_is_black, "win" if s.winner_side == a_side else "loss")
            )
    return _match_result_from_outcomes(outcomes, len(slots))


def play_matches_batched_multi(
    agent_a: BatchedAgent,
    opponents: list[tuple[BatchedAgent, int, int]],
    *,
    random_opening_moves: int = 0,
    start_state: GameState | None = None,
    opening_states_per_opponent: list[list[GameState] | None] | None = None,
) -> list[MatchResult]:
    """Play A against SEVERAL opponents at once, all games live together
    (issue #110). `opponents` is a list of (agent, n_games, seed); returns one
    `MatchResult` per opponent, from A's perspective, with per-opponent slot
    seeding identical to a `play_matches_batched(agent_a, agent, ...)` call —
    interleaving changes wall-clock, not matchup semantics.

    Each round, ALL games where A is to move go to one `agent_a.pick_batch`
    (bigger net batches than per-opponent matches), and each opponent's
    to-move games go to that opponent's `pick_batch`. With at most one
    NetAgent in the field, every pick_batch of a round runs concurrently —
    MPS overlaps every CPU pool/engine at once; with two or more nets they'd
    contend for the device, so the round runs sequentially. Each agent is
    only ever inside one pick_batch call at a time.

    Cheap opponents finish their games early and simply drop out of later
    rounds — no per-opponent barrier.
    """
    if opening_states_per_opponent is None:
        opening_states_per_opponent = [None] * len(opponents)
    fields: list[list[_Slot]] = [
        _build_slots(
            n_games,
            seed=seed,
            random_opening_moves=random_opening_moves,
            start_state=start_state,
            opening_states=opening_states,
        )
        for (_agent, n_games, seed), opening_states in zip(
            opponents, opening_states_per_opponent
        )
    ]

    n_nets = sum(
        isinstance(ag, NetAgent)
        for ag in [agent_a] + [agent for agent, _n, _s in opponents]
    )
    overlap = n_nets <= 1

    # Safety valve — terminal draw detection ends games at a full board, so
    # this only trips on a buggy agent playing illegal/no-op moves.
    max_rounds = BOARD_SIZE * BOARD_SIZE + 1

    with ThreadPoolExecutor(max_workers=1 + len(opponents)) as ex:
        for _round in range(max_rounds):
            group_a: list[_Slot] = []
            work: list[tuple[list[_Slot], BatchedAgent]] = []
            for (agent_b, _n, _s), slots in zip(opponents, fields):
                active = [s for s in slots if not s.done]
                group_a.extend(s for s in active if s.a_to_move)
                group_b = [s for s in active if not s.a_to_move]
                if group_b:
                    work.append((group_b, agent_b))
            if not group_a and not work:
                break
            if group_a:
                work.insert(0, (group_a, agent_a))

            if overlap and len(work) > 1:
                futs = [
                    (
                        group,
                        ex.submit(
                            agent.pick_batch,
                            [s.state for s in group],
                            [s.rng for s in group],
                        ),
                    )
                    for group, agent in work
                ]
                moved = [(group, fut.result()) for group, fut in futs]
            else:
                moved = [
                    (
                        group,
                        agent.pick_batch(
                            [s.state for s in group], [s.rng for s in group]
                        ),
                    )
                    for group, agent in work
                ]

            for group, actions in moved:
                _advance_slots(group, actions)
        else:  # pragma: no cover - defensive
            stuck = [s.idx for slots in fields for s in slots if not s.done]
            raise RuntimeError(f"arena exceeded {max_rounds} rounds; stuck games: {stuck}")

    return [_result_from_slots(slots) for slots in fields]


def play_matches_batched(
    agent_a: BatchedAgent,
    agent_b: BatchedAgent,
    *,
    n_games: int,
    seed: int = 0,
    random_opening_moves: int = 0,
    start_state: GameState | None = None,
    opening_states: list[GameState] | None = None,
) -> MatchResult:
    """Play A vs B with every game live at once. Result from A's perspective.

    Single-opponent front door over `play_matches_batched_multi`: A's
    to-move games and B's to-move games run concurrently when at most one
    side is a net (net-on-MPS overlaps engine/pool-on-CPU; two torch agents
    would just contend for the device, so those run back to back).
    """
    return play_matches_batched_multi(
        agent_a,
        [(agent_b, n_games, seed)],
        random_opening_moves=random_opening_moves,
        start_state=start_state,
        opening_states_per_opponent=[opening_states],
    )[0]


def play_match_specs(
    spec_a: str,
    spec_b: str,
    *,
    n_games: int,
    seed: int = 0,
    device: str | None = None,
    random_opening_moves: int = 0,
    start_state: GameState | None = None,
    default_sims: int = 100,
    wave_size: int = 16,
    engine_pool_size: int = 8,
    picker_workers: int | None = None,
) -> MatchResult:
    """Spec-string front door: build both agents, play, tear down."""
    agent_a = build_agent(
        spec_a,
        device=device,
        default_sims=default_sims,
        wave_size=wave_size,
        engine_pool_size=engine_pool_size,
        picker_workers=picker_workers,
    )
    try:
        agent_b = build_agent(
            spec_b,
            device=device,
            default_sims=default_sims,
            wave_size=wave_size,
            engine_pool_size=engine_pool_size,
            picker_workers=picker_workers,
        )
        try:
            return play_matches_batched(
                agent_a,
                agent_b,
                n_games=n_games,
                seed=seed,
                random_opening_moves=random_opening_moves,
                start_state=start_state,
            )
        finally:
            agent_b.close()
    finally:
        agent_a.close()


# ---------------- CLI ----------------


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import time

    p = argparse.ArgumentParser(
        description="Batched eval arena: all games concurrent, net moves "
        "batched across games, engine moves fanned across a warm pool.",
    )
    p.add_argument(
        "specs",
        nargs="+",
        help="Two player specs (optional 'vs' separator). e.g. "
        "\"model:checkpoint=checkpoints/latest.pt,sims=100\" vs rapfi@50ms",
    )
    p.add_argument("--n-games", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu/mps (default: auto)")
    p.add_argument("--random-opening-moves", type=int, default=0, metavar="N")
    p.add_argument("--sims", type=int, default=100,
                   help="default sims for model specs that don't set sims=")
    p.add_argument("--wave-size", type=int, default=16)
    p.add_argument("--engine-pool-size", type=int, default=8,
                   help="warm engines per external opponent")
    p.add_argument("--picker-workers", type=int, default=None,
                   help="process-pool size for expensive CPU pickers "
                        "(lookahead); default GOMOKU_PICKER_WORKERS env or "
                        "cpu_count-4 capped at 12; 1 disables the pool")
    p.add_argument("--json", action="store_true", help="emit JSON result")
    args = p.parse_args(argv)

    raw = [s for s in args.specs if s.strip().lower() != "vs"]
    if len(raw) != 2:
        p.error(f"expected exactly two player specs, got {raw}")

    t0 = time.time()
    result = play_match_specs(
        raw[0],
        raw[1],
        n_games=args.n_games,
        seed=args.seed,
        device=args.device,
        random_opening_moves=args.random_opening_moves,
        default_sims=args.sims,
        wave_size=args.wave_size,
        engine_pool_size=args.engine_pool_size,
        picker_workers=args.picker_workers,
    )
    dt = time.time() - t0

    if args.json:
        print(
            json.dumps(
                {
                    "a": raw[0],
                    "b": raw[1],
                    "n_games": result.n_games,
                    "wins": result.wins,
                    "losses": result.losses,
                    "draws": result.draws,
                    "win_rate": result.win_rate,
                    "black": {"w": result.black_w, "l": result.black_l, "d": result.black_d},
                    "white": {"w": result.white_w, "l": result.white_l, "d": result.white_d},
                    "seconds": round(dt, 2),
                }
            )
        )
    else:
        print(
            f"{raw[0]} vs {raw[1]}: {result.wins}W-{result.losses}L-{result.draws}D "
            f"over {result.n_games} games (win rate {result.win_rate:.2%}) "
            f"in {dt:.1f}s ({dt / max(result.n_games, 1):.2f}s/game)"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
