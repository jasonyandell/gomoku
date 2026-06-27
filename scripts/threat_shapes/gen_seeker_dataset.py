"""Seeker BC dataset builder — the QUIET-PHASE moves of the side that reaches the first VCT.

The seek-VCT thesis (Jason, 2026-06-26): don't search toward 5-in-a-row; *steer* play
toward a position where you have a forced VCT, then hand the tactical finish to the exact
oracle. This builds the cheapest possible feasibility dataset for the STEERING half: can a
net learn to imitate how a strong player moves in the QUIET (pre-onset) phase on the way to
its own forced win — and does that generalize to UNSEEN games?

DEFINITIONS (all read off the miner's already-computed per-ply verdicts; NO re-solve):
  onset(game) = the FIRST ply where the side-to-move has a proven VCT (a ``puzzles.jsonl.gz``
                row with ``win=True and cap=False``). The mover at the onset ply is the
                SEEKER S — the side that first owns a forced win, converted or missed.
  STEERING EXAMPLE = every pre-onset ply ``p < onset`` with ``p % 2 == onset % 2`` (S to
                move). Input = the side-to-move-relative board ``bs[p]`` (``bs[p][0]`` = S's
                stones, since p shares S's parity). Target = ``moves[p]`` — the move S
                actually played, a flat row-major cell index in the SAME frame as ``bs[p]``.
                These are the moves that *led S to a position with a forced win*: the
                steering signal, as imitated from a strong engine.

  Games with NO onset (no proven VCT anywhere at the miner's budget) are EXCLUDED — there is
  no seeker to imitate. ``cap`` plies before the first clean win do NOT corrupt the onset
  (onset only counts ``win&~cap``); a deeper hidden-VCT prefix would make our onset a slight
  OVER-estimate (we'd include a few extra steering plies), never an under-estimate of S.

This reuses ``gen_isvct_dataset.py``'s exact Pass-B machinery: replay the manifest shards with
``mine_first_vct.all_boards`` (same constructor the miner used → identical side-to-move frame),
and cross-check every present puzzle key's board against the replay (frame guard vs the live
size-16 collection reusing a filename). A game that disagrees is dropped wholesale.

HONEST GENERALIZATION — SPLIT BY SHARD. Consecutive plies of a game differ by one stone, so a
position split leaks. Each *shard* goes wholesale to train or test by ``md5(basename)%10``
(==0 → TEST), identical to the isvct experiment so the two are directly comparable.

Output (under ``--out``):
  seeker_train.npz / seeker_test.npz : X (M,PB) uint8 packed bits (np.packbits of the 2*N*N
      board bits), mv (M,) int16 target cell, shard_id (M,) int32, game_idx (M,) int32,
      ply (M,) int16, onset (M,) int16 (the game's onset ply, for analysis)
  seeker_shards.json : {config, shards:[{id,name,split}], counts, onset_hist}

Run (no GPU; nice so it never starves the live collector):
  GOMOKU_BOARD_SIZE=15 nice -n 10 uv run python -m scripts.threat_shapes.gen_seeker_dataset \
      --puzzles ~/data/puzzle_miner/puzzles.jsonl.gz \
      --manifest ~/data/puzzle_miner/manifest.txt \
      --games-dir ~/data/games_raphi --out ~/data/puzzle_miner/seeker_exp --max-shards 400
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from collections import Counter

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from scripts.threat_shapes.read_games import iter_games
from scripts.threat_shapes.mine_first_vct import all_boards

NN = N * N
PB = int(np.ceil(2 * NN / 8))


def shard_split(basename: str, test_mod: int = 10) -> str:
    h = int(hashlib.md5(basename.encode()).hexdigest(), 16)
    return "test" if h % test_mod == 0 else "train"


def pack(boards: np.ndarray) -> np.ndarray:
    return np.packbits(boards.reshape(len(boards), -1), axis=1)


class SplitAcc:
    """Per-split staging buffer that flushes (packs) periodically to bound memory."""
    def __init__(self):
        self.Xc, self.mc, self.sc, self.gc, self.pc, self.oc = [], [], [], [], [], []
        self.bb, self.bm, self.bs, self.bg, self.bp, self.bo = [], [], [], [], [], []

    def add(self, board, mv, sid, gi, ply, onset):
        self.bb.append(board); self.bm.append(mv)
        self.bs.append(sid); self.bg.append(gi); self.bp.append(ply); self.bo.append(onset)
        if len(self.bb) >= 16384:
            self.flush()

    def flush(self):
        if not self.bb:
            return
        self.Xc.append(pack(np.stack(self.bb)))
        self.mc.append(np.array(self.bm, dtype=np.int16))
        self.sc.append(np.array(self.bs, dtype=np.int32))
        self.gc.append(np.array(self.bg, dtype=np.int32))
        self.pc.append(np.array(self.bp, dtype=np.int16))
        self.oc.append(np.array(self.bo, dtype=np.int16))
        self.bb, self.bm, self.bs, self.bg, self.bp, self.bo = [], [], [], [], [], []

    def finalize(self):
        self.flush()
        if not self.Xc:
            return None
        return (np.concatenate(self.Xc), np.concatenate(self.mc),
                np.concatenate(self.sc), np.concatenate(self.gc),
                np.concatenate(self.pc), np.concatenate(self.oc))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles", default=os.path.expanduser("~/data/puzzle_miner/puzzles.jsonl.gz"))
    ap.add_argument("--manifest", default=os.path.expanduser("~/data/puzzle_miner/manifest.txt"))
    ap.add_argument("--games-dir", default=os.path.expanduser("~/data/games_raphi"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shards", type=int, default=400, help="subset of manifest shards (0=all)")
    ap.add_argument("--test-mod", type=int, default=10, help="md5(shard)%%mod==0 -> TEST")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    manifest = [l.strip() for l in open(args.manifest) if l.strip()]
    if args.max_shards:
        manifest = manifest[:args.max_shards]
    sel = set(manifest)
    sid_of = {b: i for i, b in enumerate(manifest)}
    splits = {b: shard_split(b, args.test_mod) for b in manifest}
    n_test = sum(v == "test" for v in splits.values())
    print(f"[gen] N={N} manifest_shards={len(manifest)} train={len(manifest)-n_test} "
          f"test={n_test}", flush=True)

    # ---- Pass A: stream puzzles once. Record per-game onset (first win&~cap ply) and
    #              every present (sid,idx,ply) board for the frame guard.
    onset: dict[tuple[int, int], int] = {}            # (sid,idx) -> first win&~cap ply
    present: dict[tuple[int, int, int], tuple] = {}   # (sid,idx,ply) -> (atk_tuple, dfd_tuple)
    st = Counter()
    ta = time.time()
    with gzip.open(args.puzzles, "rt") as f:
        for line in f:
            if not line:
                continue
            r = json.loads(line)
            sh = r["shard"]
            if sh not in sel:
                continue
            sid = sid_of[sh]
            idx = int(r["idx"]); ply = int(r["ply"])
            present[(sid, idx, ply)] = (tuple(r["atk"]), tuple(r["dfd"]))
            if r["win"] and not r["cap"]:
                k = (sid, idx)
                if ply < onset.get(k, 1 << 30):
                    onset[k] = ply
    print(f"[gen] passA: present_keys={len(present)} games_with_onset={len(onset)} "
          f"({time.time()-ta:.0f}s)", flush=True)

    # ---- Pass B: replay manifest shards; emit S's pre-onset same-parity steering moves.
    acc = {"train": SplitAcc(), "test": SplitAcc()}
    onset_hist: Counter = Counter()
    tb = time.time()
    for si, base in enumerate(manifest):
        sid = sid_of[base]
        split = splits[base]
        sp = os.path.join(args.games_dir, base)
        if not os.path.exists(sp):
            st["shard_missing"] += 1
            continue
        a = acc[split]
        try:
            games = list(iter_games(args.games_dir, [sp]))
        except Exception:
            st["shard_read_err"] += 1
            continue
        for gi, g in enumerate(games):
            moves = g.get("moves", [])
            if any((not isinstance(m, int)) or m < 0 or m >= NN for m in moves):
                st["game_outofrange"] += 1
                continue
            try:
                bs = all_boards(moves)
            except Exception:
                st["game_build_err"] += 1
                continue
            if bs is None:
                st["game_too_short"] += 1
                continue
            # frame guard: any present key must match the replayed board exactly
            ok = True
            for p in range(len(bs)):
                k = (sid, gi, p)
                if k in present:
                    atk, dfd = present[k]
                    if (set(np.flatnonzero(bs[p][0].reshape(-1)).tolist()) != set(atk) or
                            set(np.flatnonzero(bs[p][1].reshape(-1)).tolist()) != set(dfd)):
                        ok = False
                        break
            if not ok:
                st["game_frame_mismatch"] += 1
                continue
            o = onset.get((sid, gi))
            if o is None:
                st["game_no_onset"] += 1
                continue
            o = min(o, len(bs))  # onset is always < len(bs) for an emitted game, but clamp
            onset_hist[o] += 1
            st["game_with_onset"] += 1
            par = o % 2
            for p in range(o):
                if p % 2 != par:
                    continue
                if p >= len(moves):
                    continue
                a.add(bs[p].astype(bool), int(moves[p]), sid, gi, p, o)
                st["examples"] += 1
        st["shards_replayed"] += 1
        if (si + 1) % 50 == 0:
            print(f"[gen] passB {si+1}/{len(manifest)} ex={st['examples']} "
                  f"onset_games={st['game_with_onset']} no_onset={st['game_no_onset']} "
                  f"mismatch={st['game_frame_mismatch']} ({time.time()-tb:.0f}s)", flush=True)
    print(f"[gen] passB done: examples={st['examples']} onset_games={st['game_with_onset']} "
          f"no_onset={st['game_no_onset']} replayed_shards={st['shards_replayed']} "
          f"missing={st['shard_missing']} frame_mismatch={st['game_frame_mismatch']} "
          f"oor_games={st['game_outofrange']} ({time.time()-tb:.0f}s)", flush=True)

    # ---- write npz per split
    counts = {}
    for split in ("train", "test"):
        fin = acc[split].finalize()
        if fin is None:
            counts[split] = {"n": 0}
            continue
        X, mv, sid, gi, ply, onset_arr = fin
        path = os.path.join(args.out, f"seeker_{split}.npz")
        np.savez_compressed(path, X=X, mv=mv, shard_id=sid, game_idx=gi, ply=ply, onset=onset_arr)
        counts[split] = {"n": int(len(mv)), "games": int(len(np.unique(gi * 100000 + sid)))}
        print(f"[gen] wrote {path}: n={len(mv)}", flush=True)

    out = {
        "board_size": N, "packed_bytes": PB, "test_mod": args.test_mod,
        "label_source": "mine_puzzles.py verdicts (onset=first win&~cap); no re-solve",
        "definition": "X=board[0]=seeker(side-to-move); mv=move S played at pre-onset ply (p<onset, p%2==onset%2)",
        "puzzles": os.path.abspath(os.path.expanduser(args.puzzles)),
        "games_dir": os.path.abspath(os.path.expanduser(args.games_dir)),
        "n_shards": len(manifest), "n_train_shards": len(manifest) - n_test,
        "n_test_shards": n_test,
        "stats": dict(st), "counts": counts,
        "onset_hist": {int(k): int(v) for k, v in sorted(onset_hist.items())},
        "shards": [{"id": sid_of[b], "name": b, "split": splits[b]} for b in manifest],
        "wall_secs": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "seeker_shards.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gen] DONE examples={st['examples']} onset_games={st['game_with_onset']} "
          f"wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
