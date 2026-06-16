"""White-defense EVAL SUITE — CLI (GH #45).

A FIXED defend-as-white position set + a ``white_loss_rate`` metric WITH a
Wilson confidence interval — the gate signal that MOVES on the defense arc.
See ``gomoku/white_defense.py`` for the library + design rationale.

Two subcommands:

  mine   — (re)generate the checked-in fixture. Plays bounded CPU baseline games
           (NO GPU, NO net, NO wine), snapshots every white-to-move position
           facing a LIVE black threat, dedups, samples to a target count with a
           fixed seed, and writes the versioned fixture. The COMMITTED artifact
           is the fixture file; this command is its documented regenerator.

  score  — load a checkpoint, have it DEFEND as white from each fixture position
           against a fixed CPU attacker (default lookahead:depth=2), and report
           white_loss_rate + binomial_ci (a draw = a successful defense). The
           net is the only MPS consumer; --quick runs a tiny smoke subset.

REGEN COMMAND (committed fixture, 15x15) — also echoed by `mine`:
    GOMOKU_BOARD_SIZE=15 python scripts/white_defense_suite.py mine \
        --out fixtures/white_defense_15x15_v1.json --target 80 --seed 0

POSITIVE-CONTROL MEASUREMENT (orchestrator's GPU gate — NOT run here):
    GOMOKU_BOARD_SIZE=15 python scripts/white_defense_suite.py score \
        --checkpoint sweep_runs/g15_128x10_bigbuf_eval502.pt \
        --fixture fixtures/white_defense_15x15_v1.json \
        --attacker lookahead:depth=2 --sims 200 --n-seeds 3 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Repo root importable when run as `python scripts/white_defense_suite.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gomoku.game import BOARD_SIZE, GameState  # noqa: E402
from gomoku.white_defense import (  # noqa: E402
    black_threat_present,
    build_attacker,
    build_ckpt_defender,
    fixture_summary_json,
    load_fixture,
    save_fixture,
    score_white_defense,
)

DEFAULT_FIXTURE = f"fixtures/white_defense_{BOARD_SIZE}x{BOARD_SIZE}_v1.json"


# ---------------------------------------------------------------------------
# mine — build the checked-in fixture from bounded CPU baseline games.
# ---------------------------------------------------------------------------
def _mine_positions(
    *,
    target: int,
    seed: int,
    n_games: int,
    max_plies: int,
    require_four: bool,
    attacker_spec: str,
    defender_spec: str,
) -> list[dict]:
    """Play ``n_games`` baseline-vs-baseline games (black=attacker, white=
    defender), snapshot every WHITE-to-move position with a live black threat.
    Pure CPU, no net, fully deterministic given ``seed``. Dedups identical
    boards, then uniformly samples ``target`` positions with the seeded RNG."""
    rng = np.random.default_rng(seed)
    black = build_attacker(attacker_spec)
    white = build_attacker(defender_spec)
    seen: set[bytes] = set()
    found: list[dict] = []
    for g in range(n_games):
        state = GameState.initial()
        game_rng = np.random.default_rng(seed + 1 + g)
        moves: list[int] = []
        ply = 0
        while ply < max_plies:
            done, _ = state.is_terminal()
            if done:
                break
            # White to move (ply odd) + a live black threat = a defense position.
            # We store the MOVE SEQUENCE so far; replaying it reconstructs the
            # board + HISTORY_PLY frames + ply exactly (a tiny, diffable artifact).
            if state.move_count % 2 == 1 and black_threat_present(
                state, require_four=require_four
            ):
                key = state.board.tobytes()
                if key not in seen:
                    seen.add(key)
                    found.append(
                        {
                            "moves": list(moves),
                            "ply": int(state.move_count),
                            "provenance": f"{attacker_spec}_vs_{defender_spec}"
                            + ("_four" if require_four else "_three"),
                        }
                    )
            picker = black if (ply % 2 == 0) else white
            action = int(picker(state, game_rng))
            state = state.apply(action)
            moves.append(action)
            ply += 1
    if not found:
        raise RuntimeError(
            "mined 0 white-defense positions — loosen --require-four or raise --games"
        )
    # Deterministic uniform sample down to `target` (keep all if fewer).
    if len(found) > target:
        idx = rng.choice(len(found), size=target, replace=False)
        idx.sort()
        found = [found[i] for i in idx]
    return found


def cmd_mine(args: argparse.Namespace) -> int:
    # Mix open-three and four-threat positions across a couple of attacker pairs
    # so the fixture isn't monocultured to one opponent's style. All CPU, seeded.
    pairs = [
        ("heuristic", "defensive", False),
        ("lookahead:depth=2", "defensive", False),
        ("lookahead:depth=2", "heuristic", True),
    ]
    per_pair = max(1, args.target // len(pairs))
    positions: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for k, (atk, dfn, four) in enumerate(pairs):
        got = _mine_positions(
            target=per_pair + (args.target % len(pairs) if k == 0 else 0),
            seed=args.seed + 1000 * k,
            n_games=args.games,
            max_plies=args.max_plies,
            require_four=four,
            attacker_spec=atk,
            defender_spec=dfn,
        )
        for p in got:
            key = tuple(p["moves"])
            if key not in seen:
                seen.add(key)
                positions.append(p)
    if len(positions) > args.target:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(positions), size=args.target, replace=False)
        idx.sort()
        positions = [positions[i] for i in idx]
    regen = (
        f"GOMOKU_BOARD_SIZE={BOARD_SIZE} python scripts/white_defense_suite.py mine "
        f"--out {args.out} --target {args.target} --seed {args.seed}"
    )
    meta = {
        "regen_command": regen,
        "source": "baseline-vs-baseline CPU games (no net, no wine)",
        "attacker_pairs": [list(p) for p in pairs],
        "seed": args.seed,
        "games_per_pair": args.games,
    }
    save_fixture(args.out, positions, meta=meta)
    fx = load_fixture(args.out)  # round-trip-load to validate what we wrote
    print(f"wrote {len(fx)} positions to {args.out}")
    print(fixture_summary_json(fx))
    print(f"regen: {regen}")
    return 0


# ---------------------------------------------------------------------------
# score — defend the checkpoint as white and report white_loss_rate + CI.
# ---------------------------------------------------------------------------
def cmd_score(args: argparse.Namespace) -> int:
    fixture = load_fixture(args.fixture)
    if args.quick:
        # Tiny smoke subset: first few positions, single seed.
        k = min(args.quick_n, len(fixture))
        fixture.positions = fixture.positions[:k]
        n_seeds = 1
    else:
        n_seeds = args.n_seeds

    defender = build_ckpt_defender(
        args.checkpoint, sims=args.sims, c_puct=args.c_puct, device=args.device
    )
    attacker = build_attacker(args.attacker)

    total = len(fixture) * n_seeds
    state = {"last": 0}

    def progress(done: int, tot: int) -> None:
        if done == tot or done - state["last"] >= max(1, tot // 10):
            state["last"] = done
            print(f"  defended {done}/{tot} ...", flush=True)

    res = score_white_defense(
        defender, attacker, fixture,
        n_seeds=n_seeds, base_seed=args.seed,
        progress=None if args.quiet else progress,
    )
    out = {
        "checkpoint": args.checkpoint,
        "fixture": str(args.fixture),
        "board_size": BOARD_SIZE,
        "attacker": args.attacker,
        "sims": args.sims,
        "n_positions": len(fixture),
        "n_seeds": n_seeds,
        "n": res.n,
        "white_losses": res.white_losses,
        "white_loss_rate": round(res.white_loss_rate, 4),
        "ci95_lo": round(res.ci_lo, 4),
        "ci95_hi": round(res.ci_hi, 4),
        "ci95_half_width": round(res.ci_half, 4),
        "defense_score": res.defense_score,
        "per_provenance": {k: list(v) for k, v in res.per_provenance.items()},
        "quick": bool(args.quick),
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out_json}")
    return 0


# ---------------------------------------------------------------------------
# info — print a fixture summary without scoring.
# ---------------------------------------------------------------------------
def cmd_info(args: argparse.Namespace) -> int:
    fx = load_fixture(args.fixture)
    print(fixture_summary_json(fx))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="(re)generate the checked-in fixture (CPU)")
    m.add_argument("--out", default=DEFAULT_FIXTURE)
    m.add_argument("--target", type=int, default=80, help="target position count (~40-120)")
    m.add_argument("--games", type=int, default=60, help="games per attacker pair")
    m.add_argument("--max-plies", type=int, default=80)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=cmd_mine)

    s = sub.add_parser("score", help="defend a checkpoint as white; report white_loss_rate + CI")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--fixture", default=DEFAULT_FIXTURE)
    s.add_argument("--attacker", default="lookahead:depth=2",
                   help="random|heuristic|defensive|lookahead|lookahead:depth=K")
    s.add_argument("--sims", type=int, default=200)
    s.add_argument("--c-puct", type=float, default=1.5)
    s.add_argument("--n-seeds", type=int, default=1,
                   help="replay the whole fixture under this many seeds (tightens the CI)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", type=str, default=None)
    s.add_argument("--quick", action="store_true", help="tiny smoke run (first --quick-n positions, 1 seed)")
    s.add_argument("--quick-n", type=int, default=4)
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--out-json", type=str, default=None)
    s.set_defaults(func=cmd_score)

    i = sub.add_parser("info", help="print a fixture summary")
    i.add_argument("--fixture", default=DEFAULT_FIXTURE)
    i.set_defaults(func=cmd_info)

    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
