"""Rapfi-as-teacher: distil a stronger engine's policy into the net.

The plateau verdict (TRAINING_WIKI 2026-06-23) was blunt: the three data-pipeline
levers (reuse, window, recency) are exhausted; self-play has no stronger gradient
to give. The only lever that points up is an EXTERNAL strength signal. This module
delivers it the principled way — **policy-side, one-hot distillation from Rapfi**:

  1. gather positions the net actually reaches (self-play from the fixed opening),
  2. ask the warm Rapfi pool "what's the master move here?" for each,
  3. store (rich net planes, one-hot Rapfi move) as a teacher dataset,
  4. the trainer mixes it into every SGD step as a policy CE (``--teacher-weight``).

Why policy-only one-hot: Rapfi exposes only a *move* (no value, no policy). And
per issues #18/#44 the policy must carry the load — value-only defense teaching is
structurally wrong. So we teach "in this position, play here" and never touch the
value head. This is expert-iteration / behavioural cloning of a stronger policy on
the net's own trajectory distribution.

Storage is compact: planes as float16 (binary-valued, exact), the target as a
single move index (not a 225-wide one-hot). D4 symmetry augmentation is applied at
SAMPLE time — one random transform per batch turns N labelled positions into 8×
effective diversity for free.

CLI::

    python -m gomoku.teacher generate \\
        --checkpoint .../latest.pt --out teacher_idx2.npz \\
        --n-positions 4000 --pool-size 6 --rapfi-timeout-ms 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from gomoku.board_config import BOARD_SIZE, N_ACTIONS
from gomoku.game import HISTORY_PLY, GameState, _sym_policy
from gomoku.model import fuse_model_for_inference, load_checkpoint
from gomoku.mcts import make_torch_evaluator
from gomoku.eval_panel import IDX2_OPENING, fixed_opening_state, make_net_picker
from gomoku.rapfi_pool import (
    RapfiPool,
    RapfiUnavailable,
    default_rapfi_cmd,
    rapfi_obtainable,
)
from gomoku.util import pick_device

TEACHER_NPZ_VERSION = 1


@dataclass
class TeacherExample:
    """One distillation target: the net's rich input + the master's move."""

    planes: np.ndarray  # (N_INPUT_PLANES, N, N) float32 — the network input
    move: int           # the teacher's chosen action (one-hot policy target)
    side: int           # 0=black to move, 1=white to move
    ply: int            # move_count at this position


# --------------------------------------------------------------------------
# Position gathering — play the net against itself from the opening and keep
# every distinct position it reaches (the net's own trajectory distribution).
# --------------------------------------------------------------------------
def gather_states(
    evaluator,
    start_state: GameState,
    *,
    n_positions: int,
    sims: int = 160,
    c_puct: float = 1.5,
    temp_until_ply: int = 12,
    temperature: float = 1.0,
    max_plies: int = 80,
    seed: int = 0,
) -> list[GameState]:
    """Self-play from ``start_state`` until ``n_positions`` distinct boards seen.

    Early-ply temperature sampling gives trajectory variety; positions are
    de-duplicated by board content so the (heavily revisited) opening doesn't
    swamp the set.
    """
    rng = np.random.default_rng(seed)
    picker = make_net_picker(
        evaluator,
        sims=sims,
        c_puct=c_puct,
        temp_until_ply=temp_until_ply,
        temperature=temperature,
    )
    seen: set[bytes] = set()
    states: list[GameState] = []
    # Stall guard: once the net goes deterministic (after temp_until_ply) its
    # reachable distinct-board set is bounded, so requesting more positions than
    # exist would loop forever. Bail out after many consecutive games that add
    # nothing new and return what we gathered.
    stall = 0
    max_stall_games = 300
    while len(states) < n_positions and stall < max_stall_games:
        before = len(states)
        s = start_state
        steps = 0
        while True:
            key = s.board.tobytes()
            if key not in seen:
                seen.add(key)
                states.append(s)
                if len(states) >= n_positions:
                    break
            action = picker(s, rng)
            s = s.apply(action)
            steps += 1
            done, _ = s.is_terminal()
            if done or s.move_count >= max_plies or steps >= max_plies:
                break
        stall = stall + 1 if len(states) == before else 0
    if len(states) < n_positions:
        print(
            f"  note: net reached only {len(states)} distinct positions "
            f"(requested {n_positions}); raise --temperature/--temp-until-ply for more"
        )
    return states


# --------------------------------------------------------------------------
# Labelling — ask the warm pool for the master move at each position.
# --------------------------------------------------------------------------
def label_states_with_pool(
    states: list[GameState],
    pool: RapfiPool,
    *,
    max_workers: int | None = None,
    on_progress=None,
) -> list[TeacherExample]:
    """Label each position with Rapfi (pool, in parallel). Skips any position
    the engine refuses/crashes on rather than sinking the whole batch."""
    workers = (
        min(pool.size, len(states)) if max_workers is None else max_workers
    )
    out: list[TeacherExample | None] = [None] * len(states)

    def _label(i: int) -> None:
        s = states[i]
        try:
            move = pool.pick(s)
        except Exception:
            return  # leave None; filtered below
        out[i] = TeacherExample(
            planes=np.asarray(s.to_planes(), dtype=np.float32),
            move=int(move),
            side=int(s.move_count % 2),
            ply=int(s.move_count),
        )

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for _ in ex.map(_label, range(len(states))):
            done += 1
            if on_progress is not None and done % 200 == 0:
                on_progress(done, len(states))
    return [e for e in out if e is not None]


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
def save_teacher_npz(path: str, examples: list[TeacherExample]) -> None:
    if not examples:
        raise ValueError("no teacher examples to save")
    planes = np.stack([e.planes for e in examples]).astype(np.float16)
    moves = np.asarray([e.move for e in examples], dtype=np.int32)
    side = np.asarray([e.side for e in examples], dtype=np.int8)
    ply = np.asarray([e.ply for e in examples], dtype=np.int16)
    np.savez_compressed(
        path,
        planes=planes,
        moves=moves,
        side=side,
        ply=ply,
        board_size=np.int32(BOARD_SIZE),
        n_actions=np.int32(N_ACTIONS),
        version=np.int32(TEACHER_NPZ_VERSION),
    )


def _action_perms(board_size: int, n_actions: int) -> np.ndarray:
    """Permutation of action indices under each of the 8 D4 symmetries.

    ``perm[s][a]`` = where action ``a`` lands under symmetry ``s`` — derived
    from the SAME transform :func:`gomoku.game._sym_policy` uses, so a one-hot
    move stays aligned with planes transformed by the identical symmetry.
    """
    perms = np.zeros((8, n_actions), dtype=np.int64)
    eye = np.eye(n_actions, dtype=np.float32)
    for s in range(8):
        for a in range(n_actions):
            perms[s, a] = int(np.argmax(_sym_policy(eye[a], s)))
    return perms


class TeacherDataset:
    """In-memory teacher set: planes + move targets, with on-the-fly D4 augment.

    ``sample(batch)`` returns ``(planes, pi)`` ready for ``train_step``'s teacher
    arms — ``planes`` is ``(B, N_INPUT_PLANES, N, N)`` float and ``pi`` is a
    one-hot ``(B, N_ACTIONS)``. With ``augment=True`` a single random D4 symmetry
    is applied per batch (planes rotated/flipped, the move index permuted to
    match), which is a valid stochastic augmentation over many batches.
    """

    def __init__(self, planes, moves, *, device, augment: bool = True):
        import torch

        self.device = device
        self.augment = bool(augment)
        self.planes = torch.as_tensor(
            np.asarray(planes), dtype=torch.float32, device=device
        )
        self.moves = torch.as_tensor(
            np.asarray(moves), dtype=torch.long, device=device
        )
        self.n = int(self.planes.shape[0])
        if self.n == 0:
            raise ValueError("empty teacher dataset")
        self._n_actions = int(self.planes.shape[-1]) * int(self.planes.shape[-2])
        perms = _action_perms(int(self.planes.shape[-1]), self._n_actions)
        self._perm = torch.as_tensor(perms, dtype=torch.long, device=device)

    @classmethod
    def load(cls, path: str, *, device, augment: bool = True) -> "TeacherDataset":
        d = np.load(path)
        bs = int(d["board_size"]) if "board_size" in d else BOARD_SIZE
        if bs != BOARD_SIZE:
            raise ValueError(
                f"teacher data {path!r} is board_size={bs}, but this process is "
                f"board_size={BOARD_SIZE}"
            )
        return cls(d["planes"].astype(np.float32), d["moves"], device=device, augment=augment)

    def sample(self, batch_size: int):
        import torch

        idx = torch.randint(0, self.n, (batch_size,), device=self.device)
        planes = self.planes[idx]
        moves = self.moves[idx]
        if self.augment:
            s = int(torch.randint(0, 8, (1,)).item())
            rot, flip = s % 4, s // 4
            if rot:
                planes = torch.rot90(planes, rot, dims=(-2, -1))
            if flip:
                planes = torch.flip(planes, dims=(-1,))
            moves = self._perm[s][moves]
        planes = planes.contiguous()
        pi = torch.zeros((batch_size, self._n_actions), device=self.device, dtype=planes.dtype)
        pi.scatter_(1, moves.view(-1, 1), 1.0)
        return planes, pi


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_opening(spec: str) -> tuple[tuple[int, int], ...] | None:
    """``"idx2"`` | ``"empty"`` | ``"r,c;r,c;..."`` → opening move list (or None)."""
    spec = spec.strip().lower()
    if spec == "empty":
        return None
    if spec == "idx2":
        return IDX2_OPENING
    moves = []
    for part in spec.split(";"):
        r, c = part.split(",")
        moves.append((int(r), int(c)))
    return tuple(moves)


def generate(args: argparse.Namespace) -> int:
    # The teacher REQUIRES Rapfi (it is the labeller). Fail fast — and resolve the
    # launch command up front, BEFORE the expensive self-play gather, so an
    # obtainable-but-unfetchable Rapfi gives one clean actionable error instead of
    # a traceback after we've already burned the gather. No silent continuation.
    if not rapfi_obtainable():
        print(
            "ERROR: Rapfi cannot be resolved (no local engines/rapfi build, no "
            "cached/fetchable HF snapshot). Build via engines/rapfi/build_rapfi.sh, "
            "pass --rapfi-cmd, or run on an arm64 mac with network for the one-time "
            "HF fetch.",
            file=sys.stderr,
        )
        return 2
    try:
        cmd = args.rapfi_cmd or default_rapfi_cmd()
    except RapfiUnavailable as e:
        print(
            f"ERROR: Rapfi could not be obtained ({e}). Build "
            "engines/rapfi/build_rapfi.sh, ensure network for the HF auto-fetch "
            "(jasonyandell/rapfi-arm64), or pass --rapfi-cmd.",
            file=sys.stderr,
        )
        return 2
    device = pick_device(os.environ.get("GOMOKU_DEVICE", "cpu"))
    model, _payload = load_checkpoint(args.checkpoint, device=device)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, device)

    opening = _parse_opening(args.opening)
    start = fixed_opening_state(opening) if opening else GameState.initial()

    t0 = time.time()
    print(f"gathering {args.n_positions} positions via self-play from {args.opening} ...")
    states = gather_states(
        evaluator,
        start,
        n_positions=args.n_positions,
        sims=args.sims,
        c_puct=args.c_puct,
        temp_until_ply=args.temp_until_ply,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(f"  gathered {len(states)} distinct positions in {time.time()-t0:.1f}s")

    print(f"labelling with {args.pool_size} warm Rapfi @ {args.rapfi_timeout_ms}ms: {cmd}")
    t1 = time.time()
    with RapfiPool(
        size=args.pool_size,
        cmd=cmd,
        timeout_ms=args.rapfi_timeout_ms,
        board_size=BOARD_SIZE,
    ) as pool:
        examples = label_states_with_pool(
            states,
            pool,
            on_progress=lambda d, n: print(f"  labelled {d}/{n}", flush=True),
        )
    dt = time.time() - t1
    rate = len(examples) / max(dt, 1e-6)
    print(f"  labelled {len(examples)} positions in {dt:.1f}s ({rate:.1f}/s)")

    save_teacher_npz(args.out, examples)
    print(f"wrote {len(examples)} teacher examples -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rapfi-as-teacher dataset tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="self-play + Rapfi-label a teacher npz")
    g.add_argument("--checkpoint", required=True, help="net used to gather positions")
    g.add_argument("--out", required=True, help="output npz path")
    g.add_argument("--n-positions", type=int, default=4000)
    g.add_argument("--opening", type=str, default="idx2",
                   help="'idx2' | 'empty' | 'r,c;r,c;...'")
    g.add_argument("--sims", type=int, default=160)
    g.add_argument("--c-puct", type=float, default=1.5)
    g.add_argument("--temp-until-ply", type=int, default=12,
                   help="sample (not argmax) net moves before this ply for variety")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--pool-size", type=int, default=6)
    g.add_argument("--rapfi-cmd", type=str, default=None)
    g.add_argument("--rapfi-timeout-ms", type=int, default=1000)
    g.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.cmd == "generate":
        return generate(args)
    p.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
