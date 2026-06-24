"""Rapfi-as-teacher: distil a stronger engine's policy into the net.

The plateau verdict (TRAINING_WIKI 2026-06-23) was blunt: the three data-pipeline
levers (reuse, window, recency) are exhausted; self-play has no stronger gradient
to give. The only lever that points up is an EXTERNAL strength signal. This module
delivers it policy-side, by distillation from Rapfi:

  1. gather positions the net actually reaches (self-play from the fixed opening),
  2. ask the warm Rapfi pool for its judgement at each,
  3. store (rich net planes, Rapfi target) as a teacher dataset,
  4. the trainer mixes it into every SGD step as a policy CE (``--teacher-weight``).

Two teacher signals:

* HARD (v1, the default): one-hot Rapfi BEST MOVE. ``pool.pick`` returns a single
  master move; ``moves`` is the target. This is the first thing we tried — and it
  COLLAPSED the student (policy -> uniform, plies 1.1->5.0, 0/96 H2H; #77/#86): a
  one-hot target on a draw-heavy distribution is too sharp a gradient.

* SOFT (v2, ``--soft``, the #86 fix): Rapfi's DENSE per-move WINRATE map.
  ``pool.analyze_state`` drives the existing binary's multiPV search to score
  every candidate root move with a side-to-move winrate in [0, 1]; we store the
  whole ``{action: winrate}`` map (dense ``soft_policy``) and the trainer distils
  a MASKED temperature-softmax of it (``--teacher-temp``) — a gentle KL toward
  Rapfi's value landscape rather than a hard "play exactly here". GPL untouched:
  we only DRIVE the binary over the gomocup protocol; nothing is rebuilt.

Why policy-only (never the value head): per issues #18/#44 the policy must carry
the load — value-only defense teaching is structurally wrong. This is
expert-iteration / behavioural cloning of a stronger policy on the net's own
trajectory distribution.

Storage is compact: planes as float16 (binary-valued, exact). v1 stores just a
move index; v2 adds a dense ``soft_policy`` (M, N_ACTIONS) float16 (winrate at each
scored action, 0 elsewhere) and keeps ``moves`` (=argmax) for back-compat. D4
symmetry augmentation is applied at SAMPLE time — one random transform per batch
permutes planes AND the (dense) target together, turning N labels into 8×
effective diversity for free.

CLI::

    # HARD (v1):
    python -m gomoku.teacher generate \\
        --checkpoint .../latest.pt --out teacher_idx2.npz \\
        --n-positions 4000 --pool-size 6 --rapfi-timeout-ms 1000

    # SOFT (v2):
    python -m gomoku.teacher generate --soft \\
        --checkpoint .../latest.pt --out teacher_soft_idx2.npz \\
        --n-positions 4000 --pool-size 6 --rapfi-timeout-ms 1000 \\
        --rapfi-max-node 20000
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

# v1: one-hot best-move teacher (`moves` only).
# v2: SOFT teacher — adds a dense per-move WINRATE map (`soft_policy`, (M, N_ACTIONS)
#     float16, winrate at each scored action, 0 elsewhere). `moves` is kept
#     (=argmax of soft_policy) for back-compat; load is version-gated.
TEACHER_NPZ_VERSION = 2

# Floor applied to a genuinely-scored winrate so the dense soft_policy's `> 0`
# entries are EXACTLY the scored support — distinguishing a "scored, ~0 winrate"
# move (a lost position) from an unscored/pruned cell. Just above float16's step
# at 0 (~6e-8); small enough not to perturb the temperature-softmax ordering.
SOFT_EPS = 1e-6


@dataclass
class TeacherExample:
    """One distillation target: the net's rich input + the master's signal.

    The HARD signal is ``move`` (the teacher's single best action — the v1
    one-hot target). The SOFT signal (v2) is ``winrates``: Rapfi's per-move
    side-to-move winrate over every scored candidate (``{action: winrate}``),
    distilled as a temperature-softmax policy target. ``winrates`` is None for a
    hard-only example; when present, ``move`` is its argmax.
    """

    planes: np.ndarray  # (N_INPUT_PLANES, N, N) float32 — the network input
    move: int           # the teacher's chosen action (one-hot policy target)
    side: int           # 0=black to move, 1=white to move
    ply: int            # move_count at this position
    winrates: dict[int, float] | None = None  # {action: winrate in [0,1]} or None


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


def soft_label_states_with_pool(
    states: list[GameState],
    pool: RapfiPool,
    *,
    max_node: int = 20000,
    max_workers: int | None = None,
    on_progress=None,
) -> list[TeacherExample]:
    """SOFT label: stamp each position with Rapfi's dense per-move WINRATE map.

    Each example carries ``winrates = {action: winrate}`` (Rapfi's side-to-move
    winrate over every scored candidate; pruned cells absent -> 0 mass), and
    ``move`` set to its argmax so the v1 hard target stays available. Positions
    the engine refuses / returns an empty analysis for are skipped rather than
    sinking the batch.
    """
    workers = (
        min(pool.size, len(states)) if max_workers is None else max_workers
    )
    out: list[TeacherExample | None] = [None] * len(states)

    def _label(i: int) -> None:
        s = states[i]
        try:
            wr = pool.analyze_state(s, max_node=max_node)
        except Exception:
            return  # leave None; filtered below
        if not wr:
            return  # no scored moves -> skip
        best = max(wr.items(), key=lambda kv: kv[1])[0]
        out[i] = TeacherExample(
            planes=np.asarray(s.to_planes(), dtype=np.float32),
            move=int(best),
            side=int(s.move_count % 2),
            ply=int(s.move_count),
            winrates={int(a): float(w) for a, w in wr.items()},
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
    arrays = dict(
        planes=planes,
        moves=moves,
        side=side,
        ply=ply,
        board_size=np.int32(BOARD_SIZE),
        n_actions=np.int32(N_ACTIONS),
        version=np.int32(TEACHER_NPZ_VERSION),
    )
    # SOFT teacher (v2): when ANY example carries a winrate map, store a dense
    # (M, N_ACTIONS) float16 soft_policy — the side-to-move winrate at each scored
    # action, 0 at unscored/pruned cells. The loader treats `> 0` as the scored
    # SUPPORT (masked-in for the temperature-softmax). A genuinely-scored move can
    # have winrate ~0 (a lost position); to keep it distinguishable from an
    # unscored cell we FLOOR every scored winrate to a tiny epsilon, so `> 0`
    # encodes membership exactly while the relative ordering of winrates is intact
    # (the float16 step at 0 is ~6e-8; SOFT_EPS sits just above it). Examples
    # WITHOUT a map fall back to a one-hot row at `move`, so a mixed batch
    # round-trips cleanly.
    if any(e.winrates is not None for e in examples):
        soft = np.zeros((len(examples), N_ACTIONS), dtype=np.float16)
        for i, e in enumerate(examples):
            if e.winrates:
                for a, w in e.winrates.items():
                    if 0 <= a < N_ACTIONS:
                        soft[i, a] = np.float16(max(float(w), SOFT_EPS))
            else:
                soft[i, int(e.move)] = np.float16(1.0)
        arrays["soft_policy"] = soft
    np.savez_compressed(path, **arrays)


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
    """In-memory teacher set: planes + targets, with on-the-fly D4 augment.

    ``sample(batch)`` returns ``(planes, pi)`` ready for ``train_step``'s teacher
    arm — ``planes`` is ``(B, N_INPUT_PLANES, N, N)`` float and ``pi`` is a
    ``(B, N_ACTIONS)`` policy target.

    HARD mode (v1, no ``soft_policy``): ``pi`` is a one-hot at the teacher move.

    SOFT mode (v2, ``soft_policy`` present): ``pi`` is a MASKED temperature-softmax
    over the scored support — Rapfi's per-move winrates are softmaxed at
    ``teacher_temp`` with the UNSCORED cells (stored as 0) masked OUT before the
    softmax, so the distribution lives only on candidate moves (no mass leaks
    onto pruned cells). As ``teacher_temp -> 0`` this collapses to the v1 one-hot
    at the highest-winrate move.

    With ``augment=True`` a single random D4 symmetry is applied per batch (planes
    rotated/flipped, the move index AND the dense winrate row permuted by the
    SAME symmetry, so target and planes stay aligned).
    """

    def __init__(
        self,
        planes,
        moves,
        *,
        device,
        augment: bool = True,
        soft_policy=None,
        teacher_temp: float = 0.10,
    ):
        import torch

        self.device = device
        self.augment = bool(augment)
        self.teacher_temp = float(teacher_temp)
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
        # SOFT target support: the dense (M, N_ACTIONS) winrate map (0 = unscored).
        self.soft_policy = None
        if soft_policy is not None:
            self.soft_policy = torch.as_tensor(
                np.asarray(soft_policy), dtype=torch.float32, device=device
            )
            if self.soft_policy.shape != (self.n, self._n_actions):
                raise ValueError(
                    f"soft_policy shape {tuple(self.soft_policy.shape)} != "
                    f"({self.n}, {self._n_actions})"
                )

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device,
        augment: bool = True,
        teacher_temp: float = 0.10,
    ) -> "TeacherDataset":
        d = np.load(path)
        bs = int(d["board_size"]) if "board_size" in d else BOARD_SIZE
        if bs != BOARD_SIZE:
            raise ValueError(
                f"teacher data {path!r} is board_size={bs}, but this process is "
                f"board_size={BOARD_SIZE}"
            )
        # Version-gated: a v2 npz carries a dense soft_policy; a v1 npz does not,
        # and loads exactly as before (hard one-hot targets).
        soft = d["soft_policy"] if "soft_policy" in d.files else None
        return cls(
            d["planes"].astype(np.float32),
            d["moves"],
            device=device,
            augment=augment,
            soft_policy=soft,
            teacher_temp=teacher_temp,
        )

    def _softmax_target(self, soft_rows):
        """Masked temperature-softmax of a (B, N_ACTIONS) winrate batch.

        Cells stored as 0 (unscored / pruned) are masked OUT before the softmax,
        so all probability mass stays on the scored support. ``teacher_temp -> 0``
        sharpens toward a one-hot at the max-winrate cell.
        """
        import torch

        temp = max(self.teacher_temp, 1e-6)
        support = soft_rows > 0.0
        logits = soft_rows / temp
        # Mask unscored cells with -inf so they get exactly zero softmax mass.
        logits = logits.masked_fill(~support, float("-inf"))
        # A row with no scored support (shouldn't happen — generation skips empty
        # analyses) would softmax to NaN; guard by falling back to uniform there.
        empty = ~support.any(dim=-1, keepdim=True)
        logits = torch.where(empty, torch.zeros_like(logits), logits)
        return torch.softmax(logits, dim=-1)

    def sample(self, batch_size: int):
        import torch

        idx = torch.randint(0, self.n, (batch_size,), device=self.device)
        planes = self.planes[idx]
        moves = self.moves[idx]
        soft_rows = self.soft_policy[idx] if self.soft_policy is not None else None
        if self.augment:
            s = int(torch.randint(0, 8, (1,)).item())
            rot, flip = s % 4, s // 4
            if rot:
                planes = torch.rot90(planes, rot, dims=(-2, -1))
            if flip:
                planes = torch.flip(planes, dims=(-1,))
            # Permute targets by the SAME D4 symmetry as the planes: a one-hot
            # move index, and (soft) the dense winrate row gathered through perm.
            moves = self._perm[s][moves]
            if soft_rows is not None:
                # perm[s][a] = destination cell of action a; place row[a] at perm[a].
                permuted = torch.zeros_like(soft_rows)
                permuted[:, self._perm[s]] = soft_rows
                soft_rows = permuted
        planes = planes.contiguous()
        if soft_rows is not None:
            return planes, self._softmax_target(soft_rows)
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

    mode = "SOFT (dense winrate)" if args.soft else "HARD (one-hot best move)"
    print(
        f"labelling [{mode}] with {args.pool_size} warm Rapfi @ "
        f"{args.rapfi_timeout_ms}ms"
        + (f", max_node={args.rapfi_max_node}" if args.soft else "")
        + f": {cmd}"
    )
    t1 = time.time()
    with RapfiPool(
        size=args.pool_size,
        cmd=cmd,
        timeout_ms=args.rapfi_timeout_ms,
        board_size=BOARD_SIZE,
    ) as pool:
        if args.soft:
            examples = soft_label_states_with_pool(
                states,
                pool,
                max_node=args.rapfi_max_node,
                on_progress=lambda d, n: print(f"  labelled {d}/{n}", flush=True),
            )
        else:
            examples = label_states_with_pool(
                states,
                pool,
                on_progress=lambda d, n: print(f"  labelled {d}/{n}", flush=True),
            )
    dt = time.time() - t1
    rate = len(examples) / max(dt, 1e-6)
    print(f"  labelled {len(examples)} positions in {dt:.1f}s ({rate:.1f}/s)")

    save_teacher_npz(args.out, examples)
    if args.soft and examples:
        supports = [len(e.winrates) for e in examples if e.winrates]
        if supports:
            mean_support = sum(supports) / len(supports)
            print(
                f"  soft targets: mean {mean_support:.0f} scored moves/position "
                f"(min {min(supports)}, max {max(supports)})"
            )
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
    g.add_argument("--soft", action="store_true",
                   help="SOFT teacher: label each position with Rapfi's DENSE "
                        "per-move winrate map (a soft policy target) instead of a "
                        "single one-hot best move. Distilling the one-hot move "
                        "collapsed the student (#77/#86); the soft winrate target "
                        "is the gentle fix. Writes a v2 npz with a dense "
                        "soft_policy array (and keeps `moves`=argmax for "
                        "back-compat). Pair with --teacher-temp at train time.")
    g.add_argument("--rapfi-max-node", type=int, default=20000,
                   help="Node budget for the SOFT whole-board analysis (smaller = "
                        "faster, coarser winrates). Default 20000.")
    g.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.cmd == "generate":
        return generate(args)
    p.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
