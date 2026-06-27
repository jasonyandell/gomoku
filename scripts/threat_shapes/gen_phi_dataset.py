"""Phi distance-to-VCT field dataset builder — the FREE potential target for L2.

From `vct-reachability-mining.md` §1: the forward puzzle miner already wrote a per-ply
verdict for EVERY ply of EVERY game (`~/data/puzzle_miner/`): `win&~cap` plies present,
`cap` plies present, proven-no-VCT plies *absent* (within `manifest.txt` shards). So the
full per-game VCT-window structure is already on disk and we can read a distance field with
NO re-solve.

TARGET (per ply p, side-to-move = p%2; board `bs[p]` is side-to-move-relative so
`bs[p][0]` = the MOVER's stones):
  * OFFENSE  Phi_off(p) = gamma ** d_off,  d_off = the MOVER's own moves to its nearest
             future same-parity proven VCT (q>=p, q%2==p%2, win&~cap), measured (q-p)//2.
             d_off=0 at a VCT (Phi=1). Phi_off=0 if no future same-parity VCT in this game
             (the floor: "no forcing win reachable here, this game").
  * DEFENSE  Phi_def(p) = gamma ** d_def,  d_def = opponent-moves until the opponent first
             holds a VCT (q>=p, q%2!=p%2, win&~cap), measured (q-p+1)//2. 0 if none.

HONESTY (per §1): distance-along-Rapfi's-realized-play is an UPPER bound on the true
distance (the game found *a* path, maybe not the shortest); the Phi=0 floor is a LOWER
bound (misses wins Rapfi didn't take). The free target brackets the truth and is noisy —
good enough to (a) re-audition attention vs CNN on a GLOBAL target and (b) be the first L2
cut. `cap` plies are EXCLUDED at emit time (unknown verdict).

Reuses `gen_seeker_dataset.py`'s exact Pass-B machinery: replay manifest shards with
`mine_first_vct.all_boards` (identical side-to-move frame) and a per-game frame guard vs the
stored puzzle boards (drop a disagreeing game wholesale).

SPLIT BY SHARD (md5(basename)%10==0 -> TEST), identical rule to the isvct/seeker experiments
so all three are directly comparable.

Output (under --out):
  phi_train.npz / phi_test.npz : X (M,PB) uint8 packed board bits, phi_off (M,) f32,
      phi_def (M,) f32, d_off (M,) int16 (-1 if none), d_def (M,) int16, par (M,) int8,
      shard_id (M,) int32, game_idx (M,) int32, ply (M,) int16
  phi_shards.json : {config, shards:[{id,name,split}], counts, coverage stats}

Run (no GPU solve; nice so it never starves the live collector):
  GOMOKU_BOARD_SIZE=15 nice -n 10 uv run python -m scripts.threat_shapes.gen_phi_dataset \
      --puzzles ~/data/puzzle_miner/puzzles.jsonl.gz \
      --manifest ~/data/puzzle_miner/manifest.txt \
      --games-dir ~/data/games_raphi --out ~/data/puzzle_miner/phi_exp \
      --gamma 0.8 --max-shards 400
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
    """Per-split staging buffer; flushes (packs) periodically to bound memory."""
    FIELDS = ("bb", "boff", "bdef", "bdo", "bdd", "bpar", "bs", "bg", "bp")

    def __init__(self):
        self.cols = {f: [] for f in self.FIELDS}        # chunk lists (post-flush)
        self.buf = {f: [] for f in self.FIELDS}         # staging

    def add(self, board, phi_off, phi_def, d_off, d_def, par, sid, gi, ply):
        b = self.buf
        b["bb"].append(board); b["boff"].append(phi_off); b["bdef"].append(phi_def)
        b["bdo"].append(d_off); b["bdd"].append(d_def); b["bpar"].append(par)
        b["bs"].append(sid); b["bg"].append(gi); b["bp"].append(ply)
        if len(b["bb"]) >= 16384:
            self.flush()

    def flush(self):
        b = self.buf
        if not b["bb"]:
            return
        self.cols["bb"].append(pack(np.stack(b["bb"])))
        self.cols["boff"].append(np.array(b["boff"], dtype=np.float32))
        self.cols["bdef"].append(np.array(b["bdef"], dtype=np.float32))
        self.cols["bdo"].append(np.array(b["bdo"], dtype=np.int16))
        self.cols["bdd"].append(np.array(b["bdd"], dtype=np.int16))
        self.cols["bpar"].append(np.array(b["bpar"], dtype=np.int8))
        self.cols["bs"].append(np.array(b["bs"], dtype=np.int32))
        self.cols["bg"].append(np.array(b["bg"], dtype=np.int32))
        self.cols["bp"].append(np.array(b["bp"], dtype=np.int16))
        self.buf = {f: [] for f in self.FIELDS}

    def finalize(self):
        self.flush()
        if not self.cols["bb"]:
            return None
        return {f: np.concatenate(self.cols[f]) for f in self.FIELDS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles", default=os.path.expanduser("~/data/puzzle_miner/puzzles.jsonl.gz"))
    ap.add_argument("--manifest", default=os.path.expanduser("~/data/puzzle_miner/manifest.txt"))
    ap.add_argument("--games-dir", default=os.path.expanduser("~/data/games_raphi"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=0.8, help="discount per mover-move")
    ap.add_argument("--max-shards", type=int, default=400, help="subset of manifest shards (0=all)")
    ap.add_argument("--test-mod", type=int, default=10, help="md5(shard)%%mod==0 -> TEST")
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="keep this fraction of emitted plies (deterministic by hash)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    g = float(args.gamma)
    t0 = time.time()

    manifest = [l.strip() for l in open(args.manifest) if l.strip()]
    if args.max_shards:
        manifest = manifest[:args.max_shards]
    sel = set(manifest)
    sid_of = {b: i for i, b in enumerate(manifest)}
    splits = {b: shard_split(b, args.test_mod) for b in manifest}
    n_test = sum(v == "test" for v in splits.values())
    print(f"[gen] N={N} gamma={g} manifest_shards={len(manifest)} "
          f"train={len(manifest)-n_test} test={n_test}", flush=True)

    # ---- Pass A: stream puzzles once. Record per-(sid,idx,ply) verdict + board (frame guard).
    #      verdict: 1 = win&~cap (VCT), 2 = cap. absent = proven no-VCT (manifest shard).
    verdict: dict[tuple[int, int, int], int] = {}
    present: dict[tuple[int, int, int], tuple] = {}
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
            verdict[(sid, idx, ply)] = 1 if (r["win"] and not r["cap"]) else 2
    n_vct = sum(1 for v in verdict.values() if v == 1)
    print(f"[gen] passA: present={len(present)} vct={n_vct} cap={len(verdict)-n_vct} "
          f"({time.time()-ta:.0f}s)", flush=True)

    # ---- Pass B: replay shards; build the per-ply distance field; emit non-cap plies.
    acc = {"train": SplitAcc(), "test": SplitAcc()}
    cov = Counter()           # coverage tallies
    doff_hist: Counter = Counter()
    tb = time.time()
    sub = args.subsample
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
        for gi, gm in enumerate(games):
            moves = gm.get("moves", [])
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
            L = len(bs)
            # frame guard: any present key must match the replayed board exactly
            ok = True
            for p in range(L):
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

            # per-ply verdict array: 0 no-VCT, 1 VCT, 2 cap
            v = np.zeros(L, dtype=np.int8)
            for p in range(L):
                v[p] = verdict.get((sid, gi, p), 0)
            # nearest future VCT ply of each parity (walk backward, carry the min)
            next_vct = [-1, -1]   # next_vct[par] = nearest q>=p with v[q]==1 and q%2==par
            d_off_arr = np.full(L, -1, dtype=np.int32)
            d_def_arr = np.full(L, -1, dtype=np.int32)
            for p in range(L - 1, -1, -1):
                if v[p] == 1:
                    next_vct[p % 2] = p
                par = p % 2
                qo = next_vct[par]              # same-parity -> offense
                qd = next_vct[1 - par]          # opposite-parity -> defense
                if qo >= 0:
                    d_off_arr[p] = (qo - p) // 2
                if qd >= 0:
                    d_def_arr[p] = (qd - p + 1) // 2

            st["game_emit"] += 1
            for p in range(L):
                if v[p] == 2:                  # cap ply: unknown -> exclude
                    st["ply_cap"] += 1
                    continue
                if p >= len(moves) and p != L - 1:
                    pass  # terminal board with no move is fine; we still emit the position
                if sub < 1.0:
                    hh = int(hashlib.md5(f"{sid}:{gi}:{p}".encode()).hexdigest(), 16)
                    if (hh % 10000) >= int(sub * 10000):
                        continue
                do = int(d_off_arr[p]); dd = int(d_def_arr[p])
                phi_off = (g ** do) if do >= 0 else 0.0
                phi_def = (g ** dd) if dd >= 0 else 0.0
                a.add(bs[p].astype(bool), phi_off, phi_def, do, dd, p % 2, sid, gi, p)
                st["examples"] += 1
                cov["off_reach" if do >= 0 else "off_floor"] += 1
                cov["def_reach" if dd >= 0 else "def_floor"] += 1
                if do >= 0:
                    doff_hist[do] += 1
        st["shards_replayed"] += 1
        if (si + 1) % 50 == 0:
            print(f"[gen] passB {si+1}/{len(manifest)} ex={st['examples']} "
                  f"cap={st['ply_cap']} mismatch={st['game_frame_mismatch']} "
                  f"({time.time()-tb:.0f}s)", flush=True)
    print(f"[gen] passB done: examples={st['examples']} cap={st['ply_cap']} "
          f"emit_games={st['game_emit']} mismatch={st['game_frame_mismatch']} "
          f"missing={st['shard_missing']} ({time.time()-tb:.0f}s)", flush=True)

    # ---- write npz per split
    counts = {}
    for split in ("train", "test"):
        fin = acc[split].finalize()
        if fin is None:
            counts[split] = {"n": 0}
            continue
        path = os.path.join(args.out, f"phi_{split}.npz")
        np.savez_compressed(
            path, X=fin["bb"], phi_off=fin["boff"], phi_def=fin["bdef"],
            d_off=fin["bdo"], d_def=fin["bdd"], par=fin["bpar"],
            shard_id=fin["bs"], game_idx=fin["bg"], ply=fin["bp"])
        counts[split] = {"n": int(len(fin["boff"])),
                         "off_floor_frac": float((fin["bdo"] < 0).mean()),
                         "phi_off_mean": float(fin["boff"].mean()),
                         "phi_def_mean": float(fin["bdef"].mean())}
        print(f"[gen] wrote {path}: n={len(fin['boff'])} "
              f"off_floor={(fin['bdo']<0).mean():.3f}", flush=True)

    out = {
        "board_size": N, "packed_bytes": PB, "gamma": g, "test_mod": args.test_mod,
        "subsample": sub,
        "label_source": "mine_puzzles.py per-ply verdicts; no re-solve",
        "target": "phi_off=gamma**(mover-moves to nearest future same-parity VCT); "
                  "phi_def=gamma**(opp-moves to opponent's next VCT); 0=floor; cap excluded",
        "puzzles": os.path.abspath(os.path.expanduser(args.puzzles)),
        "games_dir": os.path.abspath(os.path.expanduser(args.games_dir)),
        "n_shards": len(manifest), "n_train_shards": len(manifest) - n_test,
        "n_test_shards": n_test, "stats": dict(st), "coverage": dict(cov),
        "counts": counts,
        "d_off_hist": {int(k): int(v) for k, v in sorted(doff_hist.items())},
        "shards": [{"id": sid_of[b], "name": b, "split": splits[b]} for b in manifest],
        "wall_secs": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "phi_shards.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gen] DONE examples={st['examples']} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
