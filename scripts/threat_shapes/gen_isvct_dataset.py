"""is-VCT dataset builder — per-ply "does the side-to-move have a forced VCT win?".

A feasibility dataset for ONE binary question: looking at a 15x15 board, does the
SIDE TO MOVE have a forced VCT (victory-by-continuous-threats) win? The verdict is
ALREADY computed for every ply of every game by the puzzle miner
(``mine_puzzles.py`` → ``puzzles.jsonl.gz`` + ``manifest.txt``); we DO NOT re-solve
(no GPU, no contention with the live game collection). We just turn the miner's
output into labelled boards.

LABELS (read straight off the miner's trit; no solve here):
  POSITIVE = a ``puzzles.jsonl.gz`` row with ``win=True and cap=False`` (proven VCT).
             Its board is stored directly as ``atk``/``dfd`` (side-to-move-relative,
             board[0]=attacker=mover) — used AS-IS, no replay, no frame ambiguity.
  EXCLUDE  = any row with ``cap=True`` (unknown at budget — not a clean label) — kept
             only as a key so it is never mislabelled a negative.
  NEGATIVE = any ply of a fully-solved (``manifest.txt``) shard that is ABSENT from
             ``puzzles.jsonl.gz`` entirely → proven no-VCT. The miner writes a row for
             EVERY non-null ply (win or cap), so an absent ply is provably win=False,
             cap=False. Absence is meaningful ONLY inside manifest shards.

NEGATIVE BOARDS come from a light CPU replay of the manifest shards using the SAME
constructor the miner used (``mine_first_vct.all_boards``) so neg boards are in the
EXACT side-to-move-relative frame as the stored positive boards. As a guard against the
live size-16 collection reusing a filename, every present-key ply in a replayed game is
cross-checked against its stored board; a game that disagrees (or whose moves are out of
range) is dropped — its plies never become negatives.

HONEST GENERALIZATION — SPLIT BY SHARD. Consecutive plies of one game differ by a single
stone, so a position split leaks. Each *shard* goes wholesale to train or test by
``md5(basename)%10`` (==0 → TEST). The test set is drawn from shards (games) never seen in
training; the disjoint shard counts are printed.

Output (under ``--out``):
  isvct_train.npz / isvct_test.npz : X (M,PB) uint8 packed bits (np.packbits of the 2*N*N
      board bits, 57 bytes/board at N=15), y (M,) uint8, shard_id (M,) int32,
      game_idx (M,) int32, ply (M,) int16
  shards.json : {config, shards:[{id,name,split}], counts}  (decode/provenance manifest)

Run (no GPU):
  GOMOKU_BOARD_SIZE=15 nice -n 10 uv run python -m scripts.threat_shapes.gen_isvct_dataset \
      --puzzles ~/data/puzzle_miner/puzzles.jsonl.gz \
      --manifest ~/data/puzzle_miner/manifest.txt \
      --games-dir ~/data/games_raphi --out ~/data/puzzle_miner/isvct_exp --max-shards 400
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


def board_from(atk, dfd) -> np.ndarray:
    b = np.zeros((2, N, N), dtype=bool)
    if atk:
        b[0].reshape(-1)[np.asarray(atk, dtype=np.int64)] = True
    if dfd:
        b[1].reshape(-1)[np.asarray(dfd, dtype=np.int64)] = True
    return b


def pack(boards: np.ndarray) -> np.ndarray:
    return np.packbits(boards.reshape(len(boards), -1), axis=1)


class SplitAcc:
    """Per-split staging buffer that flushes (packs) periodically to bound memory."""
    def __init__(self):
        self.Xc, self.yc, self.sc, self.gc, self.pc = [], [], [], [], []
        self.bb, self.by, self.bs, self.bg, self.bp = [], [], [], [], []

    def add(self, board, y, sid, gi, ply):
        self.bb.append(board); self.by.append(y)
        self.bs.append(sid); self.bg.append(gi); self.bp.append(ply)
        if len(self.bb) >= 16384:
            self.flush()

    def flush(self):
        if not self.bb:
            return
        self.Xc.append(pack(np.stack(self.bb)))
        self.yc.append(np.array(self.by, dtype=np.uint8))
        self.sc.append(np.array(self.bs, dtype=np.int32))
        self.gc.append(np.array(self.bg, dtype=np.int32))
        self.pc.append(np.array(self.bp, dtype=np.int16))
        self.bb, self.by, self.bs, self.bg, self.bp = [], [], [], [], []

    def finalize(self):
        self.flush()
        if not self.Xc:
            return None
        return (np.concatenate(self.Xc), np.concatenate(self.yc),
                np.concatenate(self.sc), np.concatenate(self.gc), np.concatenate(self.pc))


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

    # ---- selected manifest shards (fully-solved only; absence is meaningful only here)
    manifest = [l.strip() for l in open(args.manifest) if l.strip()]
    if args.max_shards:
        manifest = manifest[:args.max_shards]
    sel = set(manifest)
    sid_of = {b: i for i, b in enumerate(manifest)}
    splits = {b: shard_split(b, args.test_mod) for b in manifest}
    n_test = sum(v == "test" for v in splits.values())
    print(f"[gen] N={N} manifest_shards={len(manifest)} train={len(manifest)-n_test} "
          f"test={n_test}", flush=True)

    # ---- Pass A: stream puzzles.jsonl.gz once. Record present keys (exclusion set) and
    #              clean positives (board straight from atk/dfd).
    present: dict[tuple[int, int, int], tuple] = {}  # (sid,idx,ply) -> (atk_tuple, dfd_tuple)
    acc = {"train": SplitAcc(), "test": SplitAcc()}
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
            key = (sid, int(r["idx"]), int(r["ply"]))
            atk = tuple(r["atk"]); dfd = tuple(r["dfd"])
            present[key] = (atk, dfd)
            if r["win"] and not r["cap"]:
                st["pos"] += 1
                acc[splits[sh]].add(board_from(atk, dfd), 1, sid, key[1], key[2])
            elif r["cap"]:
                st["cap"] += 1
            else:
                st["win_capTrue_or_other"] += 1  # win=True&cap=True etc. -> excluded
    print(f"[gen] passA puzzles: present_keys={len(present)} POS={st['pos']} "
          f"cap_excluded={st['cap']} other_excluded={st['win_capTrue_or_other']} "
          f"({time.time()-ta:.0f}s)", flush=True)

    # ---- Pass B: replay manifest shards; emit a NEGATIVE for every ply absent from present.
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
            # cross-check this game's present keys against stored boards (frame guard)
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
            for p in range(len(bs)):
                if (sid, gi, p) not in present:
                    st["neg"] += 1
                    a.add(bs[p].astype(bool), 0, sid, gi, p)
        st["shards_replayed"] += 1
        if (si + 1) % 50 == 0:
            print(f"[gen] passB {si+1}/{len(manifest)} NEG={st['neg']} "
                  f"mismatch={st['game_frame_mismatch']} oor={st['game_outofrange']} "
                  f"({time.time()-tb:.0f}s)", flush=True)
    print(f"[gen] passB done: NEG={st['neg']} replayed_shards={st['shards_replayed']} "
          f"missing={st['shard_missing']} frame_mismatch={st['game_frame_mismatch']} "
          f"oor_games={st['game_outofrange']} ({time.time()-tb:.0f}s)", flush=True)

    # ---- write npz per split
    counts = {}
    for split in ("train", "test"):
        fin = acc[split].finalize()
        if fin is None:
            counts[split] = {"n": 0, "pos": 0, "neg": 0}
            continue
        X, y, sid, gi, ply = fin
        path = os.path.join(args.out, f"isvct_{split}.npz")
        np.savez_compressed(path, X=X, y=y, shard_id=sid, game_idx=gi, ply=ply)
        counts[split] = {"n": int(len(y)), "pos": int(y.sum()), "neg": int((y == 0).sum()),
                         "pos_rate": float(y.mean())}
        print(f"[gen] wrote {path}: n={len(y)} pos={int(y.sum())} neg={int((y==0).sum())} "
              f"pos_rate={y.mean():.4f}", flush=True)

    out = {
        "board_size": N, "packed_bytes": PB, "test_mod": args.test_mod,
        "label_source": "mine_puzzles.py (puzzles.jsonl.gz + manifest.txt); no re-solve",
        "label_rule": "POS=row(win&~cap); NEG=manifest ply absent from puzzles; EXCLUDE=cap",
        "puzzles": os.path.abspath(os.path.expanduser(args.puzzles)),
        "games_dir": os.path.abspath(os.path.expanduser(args.games_dir)),
        "n_shards": len(manifest), "n_train_shards": len(manifest) - n_test,
        "n_test_shards": n_test,
        "stats": dict(st), "counts": counts,
        "shards": [{"id": sid_of[b], "name": b, "split": splits[b]} for b in manifest],
        "wall_secs": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "shards.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gen] DONE pos={st['pos']} neg={st['neg']} excluded_cap={st['cap']} "
          f"wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
