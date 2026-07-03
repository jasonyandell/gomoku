"""cap50->cap25 VCT-veto recall study (issue #114, attack-list item 2).

The sound-world self-play recipe solves, every ply, a merged batch of
attacker-terminus boards ("does side-to-move have a forced VCT?") and
defense escape-children ("if I play cell m, does the opponent then have a
VCT?" -> proven-losing moves are vetoed from play AND the recorded policy) at
node budget cap50 (``--vct-terminus-budget 50``). Day-1's census found 42.5%
of 13x13 veto boards grind to the cap, so halving the cap (cap25) would cut
most lanes' work -- but it is a SEMANTICS change: any blunder whose forced-win
proof needs 26-50 nodes stops being vetoed (a recall leak = a played-through
blunder).

THE QUANTITY THAT MATTERS: of the boards PROVEN WIN at cap50 (``win=True`` --
these are the terminus wins / blunder-vetoes), what fraction are STILL proven
at cap25? A clean no-win verdict (``win=False, hit_cap=False``) is
budget-independent; a capped verdict (``hit_cap=True``) is NOT a proof, so a
board that flips win50=True -> win25=False must be ``hit_cap`` at 25. Monotonicity
(proven@25 subset of proven@50) is asserted as a kernel sanity check.

Two modes (capture is untimed = contention-immune, no GPU lock needed; time IS
timed = wrap the invocation in the shared /tmp/gomoku-gpu.lock):

  # 1) capture live merged-veto batches from a real gen, then re-solve every
  #    board at cap-hi and cap-lo and report recall. Writes <out>.npz (raw
  #    batches, for the timing pass) + <out>.json (recall summary):
  uv run python scripts/vct_metal/cap25_recall_study.py \
      --board-size 13 --ckpt .../worker_weights.pt --batches 20 --out /tmp/r13

  # 2) time cap-hi vs cap-lo over the captured batches (RUN UNDER THE GPU LOCK):
  uv run python scripts/vct_metal/cap25_recall_study.py --time --out /tmp/r13
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board-size", type=int, default=9,
                    help="process-global board size (set before gomoku import)")
    ap.add_argument("--ckpt", help="worker_weights.pt / latest.pt (capture mode)")
    ap.add_argument("--out", required=True,
                    help="output prefix: writes <out>.npz + <out>.json")
    ap.add_argument("--batches", type=int, default=20,
                    help="minimum solver batches to collect in capture mode")
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--concurrent", type=int, default=32)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--wave-size", type=int, default=32)
    ap.add_argument("--cap-hi", type=int, default=50, help="baseline budget")
    ap.add_argument("--cap-lo", type=int, default=25, help="proposed budget")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--time", action="store_true",
                    help="timing mode: re-solve <out>.npz at cap-hi vs cap-lo")
    ap.add_argument("--time-reps", type=int, default=3)
    return ap.parse_args()


def capture(args) -> None:
    import torch
    from gomoku import self_play as sp
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint
    from gomoku.self_play import generate_games

    device = torch.device(args.device)
    model, _ = load_checkpoint(args.ckpt, device=device)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, device)

    # Exact sound-world gen semantics: cap-hi terminus + full-breadth veto,
    # overlap OFF so the solve is the workload (not hidden under search).
    sp.configure_vct_terminus(enabled=True, budget=args.cap_hi)
    sp.configure_oracle_veto(enabled=True, max_cands=0)
    sp.configure_oracle_overlap(enabled=False)
    sp._warm_mega_solver()

    real = sp._load_mega_solver()
    # each entry: (boards (B,2,N,N) bool, max_nodes, n_terminus). The streaming
    # path merges terminus + defense-children into ONE return_move=True solve
    # (_oracle_ply_solve): the first `n_terminus` boards are attacker-terminus
    # tests, the rest are defense escape-children. We recover the split by
    # wrapping _terminus_boards (called immediately before each merged solve).
    batches: list[tuple[np.ndarray, int, int]] = []
    pending_nt = {"n": 0}
    real_terminus_boards = sp._terminus_boards

    def recording_terminus_boards(arrs):
        seg = real_terminus_boards(arrs)
        pending_nt["n"] = int(seg.shape[0])
        return seg

    def recording_solver(boards, **kw):
        boards = np.asarray(boards)
        nt = pending_nt["n"]
        pending_nt["n"] = 0                       # consume; next merged call re-sets it
        batches.append((boards.copy(), int(kw.get("max_nodes", 0)),
                        min(nt, boards.shape[0])))
        return real(boards, **kw)

    sp._terminus_boards = recording_terminus_boards
    sp._vct_terminus_solver = recording_solver
    t0 = time.perf_counter()
    generate_games(args.games, evaluator, n_simulations=args.sims,
                   wave_size=args.wave_size,
                   rng=np.random.default_rng(args.seed),
                   concurrent_games=args.concurrent)
    gen_wall = time.perf_counter() - t0

    n = len(batches)
    if n < args.batches:
        print(f"WARNING: captured {n} < requested {args.batches} batches",
              file=sys.stderr)
    print(f"captured {n} batches "
          f"({sum(b.shape[0] for b, _, _ in batches)} boards, "
          f"gen_wall={gen_wall:.1f}s); re-solving for recall...",
          file=sys.stderr)

    summary = measure_recall(args, batches)
    summary["gen_wall_s"] = round(gen_wall, 1)

    # persist raw batches for the (lock-wrapped) timing pass
    npz = {"n_batches": np.int64(n),
           "budgets": np.array([m for _, m, _ in batches], dtype=np.int64),
           "n_terminus": np.array([t for _, _, t in batches], dtype=np.int64)}
    for i, (bd, _, _) in enumerate(batches):
        npz[f"batch_{i}"] = bd
    np.savez_compressed(args.out + ".npz", **npz)
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def _fresh():
    return {"n_boards": 0, "win_hi": 0, "win_lo": 0, "leak": 0,
            "mono_viol": 0, "leak_uncapped": 0}


def _accum(acc, wh, wl, hl):
    """Fold one solved slice (win@hi, win@lo, hit_cap@lo bool arrays) into acc.

    leak = win@hi & ~win@lo (proof needs 26-50 nodes -> not vetoed at cap-lo).
    mono_viol = win@lo & ~win@hi (must be 0: proven@lo subset of proven@hi).
    leak_uncapped = leak boards NOT hit_cap@lo (must be 0: a leak board is
    capped@lo, else the no-win at lo is a true proof that could not be win@hi)."""
    acc["n_boards"] += int(wh.size)
    acc["win_hi"] += int(wh.sum())
    acc["win_lo"] += int(wl.sum())
    leak_mask = wh & ~wl
    acc["leak"] += int(leak_mask.sum())
    acc["mono_viol"] += int((wl & ~wh).sum())
    acc["leak_uncapped"] += int((leak_mask & ~hl).sum())


def _finalize(acc):
    acc = dict(acc)
    acc["recall"] = (acc["win_lo"] / acc["win_hi"]) if acc["win_hi"] else float("nan")
    return acc


def measure_recall(args, batches) -> dict:
    """One solve per (batch, cap) -- slice the win/hit arrays into terminus
    prefix and defense suffix to accumulate all three views without re-solving."""
    from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb

    acc = {"all": _fresh(), "terminus": _fresh(), "defense": _fresh()}
    for bd, _, nt in batches:
        wh, _hh = solve_vct_mega_bb(bd, max_nodes=args.cap_hi, return_move=False)
        wl, hl = solve_vct_mega_bb(bd, max_nodes=args.cap_lo, return_move=False)
        wh = np.asarray(wh, dtype=bool)
        wl = np.asarray(wl, dtype=bool)
        hl = np.asarray(hl, dtype=bool)
        for key, sl in (("all", slice(None)),
                        ("terminus", slice(0, nt)),
                        ("defense", slice(nt, None))):
            _accum(acc[key], wh[sl], wl[sl], hl[sl])

    out = {
        "board_size": args.board_size, "ckpt": args.ckpt,
        "cap_hi": args.cap_hi, "cap_lo": args.cap_lo,
        "n_batches": len(batches),
        "all": _finalize(acc["all"]),
        "terminus": _finalize(acc["terminus"]),
        "defense": _finalize(acc["defense"]),
    }
    # monotonicity + leak-capped are hard kernel invariants -- fail loud
    for k in ("all", "terminus", "defense"):
        assert out[k]["mono_viol"] == 0, \
            f"MONOTONICITY VIOLATED in {k}: {out[k]['mono_viol']} boards win@lo & ~win@hi"
        assert out[k]["leak_uncapped"] == 0, \
            f"LEAK-UNCAPPED in {k}: {out[k]['leak_uncapped']} leak boards not hit_cap@lo"
    return out


def timing(args) -> None:
    from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb

    z = np.load(args.out + ".npz")
    nb = int(z["n_batches"])
    budgets = z["budgets"]
    batches = [z[f"batch_{i}"] for i in range(nb)]
    total_boards = sum(b.shape[0] for b in batches)
    print(f"timing {nb} batches, {total_boards} boards, "
          f"cap_hi={args.cap_hi} vs cap_lo={args.cap_lo}", file=sys.stderr)

    # warm both caps out of the timing
    solve_vct_mega_bb(batches[0][:8], max_nodes=args.cap_hi, return_move=False)
    solve_vct_mega_bb(batches[0][:8], max_nodes=args.cap_lo, return_move=False)

    def run(cap):
        best = None
        for _ in range(args.time_reps):
            t0 = time.perf_counter()
            for b in batches:
                solve_vct_mega_bb(b, max_nodes=cap, return_move=False)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        return best

    t_hi = run(args.cap_hi)
    t_lo = run(args.cap_lo)
    out = {"n_batches": nb, "n_boards": total_boards,
           "cap_hi": args.cap_hi, "cap_lo": args.cap_lo,
           "t_hi_s": round(t_hi, 3), "t_lo_s": round(t_lo, 3),
           "speedup": round(t_hi / t_lo, 3)}
    # merge into the recall json if present
    try:
        with open(args.out + ".json") as f:
            summary = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        summary = {}
    summary["timing"] = out
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(out, indent=2))


def main() -> None:
    args = _parse_args()
    # board size is a process-global constant resolved at gomoku import time
    os.environ["GOMOKU_BOARD_SIZE"] = str(args.board_size)
    sys.path.insert(0, os.getcwd())
    if args.time:
        timing(args)
    else:
        assert args.ckpt, "capture mode needs --ckpt"
        capture(args)


if __name__ == "__main__":
    main()
