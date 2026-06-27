"""Harvest the non-VCF "molecule" corpus — the combinational forced wins, move-labeled.

From vct-reachability-mining.md §3-§4: a **non-VCF VCT** is a forced win that needs a
*three* (not just fours) = a real combination, the molecule-shaped tactic
([shape-library-engine.md](shape-library-engine.md)). The GOLD concentrates on the WINNER's
side, surfaced by perturbing the DEFENDER: fan the opponent-to-move pre-onset non-VCT nodes,
and on each fanned board (now winner-to-move) keep the ones that are VCT but NOT VCF.

This is the corpus-scale *writer* of vct_fan.py's §3 probe. Each gold board pays twice
(§4): a non-trivial **offense terminus** (a combinational win) AND a hard **defense lesson**
(the defender just played a natural-looking move that walks into the combination). We
move-label each gold board with the catalyst (the winner's combinational-win first move) via
the kernel's passive `return_move` — no extra solver nodes.

Output (append-only under --out; jsonl.gz schema matches puzzles.jsonl.gz so it's reusable):
  gold.jsonl.gz : one row per gold board
      {atk:[occ idx, WINNER/side-to-move], dfd:[occ idx, defender incl the blunder],
       move:<catalyst cell>, dist:<plies before the real onset>, sid, gi, ply, alt}
  manifest.txt  : processed shard basenames (resume marker)
  harvest.log / README.md / stats.json

Kernels are Metal/GPU -> ZERO contention with the CPU-only collector fleet. Chunked +
incremental + time-capped so an unattended run banks what it got and exits cleanly.

Run:
  GOMOKU_BOARD_SIZE=15 nice -n 10 uv run python -m scripts.threat_shapes.harvest_molecules \
      --out ~/data/molecule_gold --max-shards 400 --max-nodes-sampled 12000 \
      --pre-band 6 --mn 500 --max-secs 3300
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import defaultdict

import numpy as np

from gomoku.board_config import BOARD_SIZE as N
from scripts.threat_shapes.read_games import iter_games
from scripts.threat_shapes.mine_first_vct import all_boards
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb
from scripts.vct_metal.mega_vcf_bb import solve_vcf_mega_bb

NN = N * N
GPU_BATCH = 16384


def load_verdicts(puzzles, sel, sid_of):
    win, cap = defaultdict(set), defaultdict(set)
    with gzip.open(puzzles, "rt") as f:
        for line in f:
            if not line:
                continue
            r = json.loads(line)
            sh = r["shard"]
            if sh not in sel:
                continue
            k = (sid_of[sh], int(r["idx"])); p = int(r["ply"])
            if r["win"] and not r["cap"]:
                win[k].add(p)
            elif r["cap"]:
                cap[k].add(p)
    return win, cap


def solve_batched(boards, fn, mn, want_move=False):
    """Run a GPU kernel over (M,2,N,N) in GPU_BATCH chunks. Returns arrays."""
    M = len(boards)
    w = np.zeros(M, bool); h = np.zeros(M, bool)
    mv = np.full(M, -1, np.int32) if want_move else None
    for s in range(0, M, GPU_BATCH):
        b = boards[s:s + GPU_BATCH]
        if want_move:
            ww, hh, mm = fn(b, max_nodes=mn, return_move=True)
            mv[s:s + len(ww)] = np.asarray(mm, np.int32)
        else:
            ww, hh = fn(b, max_nodes=mn)
        w[s:s + len(ww)] = np.asarray(ww, bool)
        h[s:s + len(hh)] = np.asarray(hh, bool)
    return (w, h, mv) if want_move else (w, h)


def process_chunk(meta, boards, mn, gf, stats, integrity=False):
    """VCT -> VCF on wins -> keep non-VCF gold -> move-label -> append rows. Returns n_gold."""
    n = len(boards)
    if n == 0:
        return 0
    if integrity:
        # sanity: the pre-fan NODES are non-VCT by construction; here we just confirm the
        # fanned boards have a sane VCT rate (defender-side ~80%); cheap signal only.
        pass
    vw, vh = solve_batched(boards, solve_vct_mega_bb, mn)
    widx = np.flatnonzero(vw)
    stats["fanned"] += n; stats["vct"] += int(vw.sum()); stats["vct_cap"] += int(vh.sum())
    if len(widx) == 0:
        return 0
    fw, fh = solve_batched(boards[widx], solve_vcf_mega_bb, mn)
    gold_local = widx[(~fw) & (~fh)]                      # VCT win, not VCF, not VCF-capped
    stats["vcf"] += int(fw.sum()); stats["vcf_cap"] += int(fh.sum())
    if len(gold_local) == 0:
        return 0
    # move-label the gold (passive return_move; cheap, gold set is small)
    gw, gh, gmv = solve_batched(boards[gold_local], solve_vct_mega_bb, mn, want_move=True)
    for j, gi_local in enumerate(gold_local):
        b = boards[gi_local]
        atk = np.flatnonzero(b[0].reshape(-1)).tolist()
        dfd = np.flatnonzero(b[1].reshape(-1)).tolist()
        m = meta[gi_local]
        gf.write(json.dumps({
            "atk": atk, "dfd": dfd, "move": int(gmv[j]),
            "dist": int(m[0]), "sid": int(m[1]), "gi": int(m[2]),
            "ply": int(m[3]), "alt": int(m[4]),
        }) + "\n")
    stats["gold"] += len(gold_local)
    return len(gold_local)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles", default=os.path.expanduser("~/data/puzzle_miner/puzzles.jsonl.gz"))
    ap.add_argument("--manifest", default=os.path.expanduser("~/data/puzzle_miner/manifest.txt"))
    ap.add_argument("--games-dir", default=os.path.expanduser("~/data/games_raphi"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shards", type=int, default=400)
    ap.add_argument("--max-nodes-sampled", type=int, default=12000, help="cap fanned NODES")
    ap.add_argument("--pre-band", type=int, default=6, help="fan nodes within this many plies before onset")
    ap.add_argument("--mn", type=int, default=500, help="solver max_nodes (miner budget)")
    ap.add_argument("--chunk", type=int, default=200000, help="fanned boards per GPU+write chunk")
    ap.add_argument("--max-secs", type=float, default=3300.0, help="wall-clock budget; stop collecting after")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest = [l.strip() for l in open(args.manifest) if l.strip()][:args.max_shards]
    sel = set(manifest); sid_of = {b: i for i, b in enumerate(manifest)}
    gold_path = os.path.join(args.out, "gold.jsonl.gz")
    manif_path = os.path.join(args.out, "manifest.txt")
    log_path = os.path.join(args.out, "harvest.log")
    done = set()
    if os.path.exists(manif_path):
        done = {l.strip() for l in open(manif_path) if l.strip()}

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    t0 = time.time()
    log(f"[harvest] DEFENDER-side fan; shards={len(manifest)} done={len(done)} "
        f"max_nodes={args.max_nodes_sampled} pre_band={args.pre_band} mn={args.mn} "
        f"max_secs={args.max_secs:.0f}")
    win, cap = load_verdicts(args.puzzles, sel, sid_of)
    log(f"[harvest] verdicts loaded: games_with_win={len(win)} ({time.time()-t0:.0f}s)")

    stats = defaultdict(int)
    node_count = 0
    buf_boards, buf_meta = [], []           # staging for the current chunk
    gf = gzip.open(gold_path, "at")          # append-only
    processed_shards = []
    stop = False

    def flush():
        if not buf_boards:
            return 0
        boards = np.concatenate(buf_boards)
        meta = np.concatenate(buf_meta)
        ng = process_chunk(meta, boards, args.mn, gf, stats)
        gf.flush()
        log(f"[harvest] chunk: fanned={len(boards)} -> gold+={ng} "
            f"(cum gold={stats['gold']} vct={stats['vct']} nodes={node_count} "
            f"{time.time()-t0:.0f}s)")
        buf_boards.clear(); buf_meta.clear()
        return ng

    for base in manifest:
        if stop:
            break
        if base in done:
            continue
        sid = sid_of[base]; sp = os.path.join(args.games_dir, base)
        if not os.path.exists(sp):
            continue
        for gi, g in enumerate(iter_games(args.games_dir, [sp])):
            moves = g.get("moves", [])
            if any((not isinstance(m, int)) or m < 0 or m >= NN for m in moves):
                continue
            k = (sid, gi); W = win.get(k, set())
            if not W:
                continue
            onset = min(W); Cg = cap.get(k, set()); bs = all_boards(moves)
            if bs is None:
                continue
            for p in range(max(0, onset - args.pre_band), onset):
                if p in W or p in Cg or p >= len(moves):
                    continue
                holder = (p % 2 == onset % 2)
                if holder:                      # DEFENDER-side only: keep opponent-to-move nodes
                    continue
                b = bs[p]; S = b[0]; opp = b[1]
                alts = np.flatnonzero((~(S | opp)).reshape(-1)); alts = alts[alts != moves[p]]
                A = len(alts)
                if A == 0:
                    continue
                s1 = np.empty((A, 2, N, N), dtype=bool)
                s1[:, 0] = opp                  # winner becomes side-to-move (attacker)
                s1[:, 1] = S
                s1[:, 1].reshape(A, -1)[np.arange(A), alts] = True   # defender's blunder alt
                meta = np.empty((A, 5), dtype=np.int64)
                meta[:, 0] = onset - p; meta[:, 1] = sid; meta[:, 2] = gi
                meta[:, 3] = p; meta[:, 4] = alts
                buf_boards.append(s1); buf_meta.append(meta)
                node_count += 1
                if sum(len(x) for x in buf_boards) >= args.chunk:
                    flush()
                if node_count >= args.max_nodes_sampled or (time.time() - t0) > args.max_secs:
                    stop = True; break
            if stop:
                break
        if not stop:
            processed_shards.append(base)
            with open(manif_path, "a") as mf:
                mf.write(base + "\n")
    flush()
    gf.close()

    st = dict(stats)
    nw = max(st.get("vct", 0), 1)
    summary = {
        "board_size": N, "side": "defender", "gamma": None,
        "max_nodes_sampled": args.max_nodes_sampled, "pre_band": args.pre_band,
        "mn": args.mn, "nodes_fanned": node_count, "shards_done": len(processed_shards),
        "stats": st,
        "gold_frac_of_vct": round(st.get("gold", 0) / nw, 4),
        "gold_frac_of_fanned": round(st.get("gold", 0) / max(st.get("fanned", 1), 1), 4),
        "wall_secs": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[harvest] DONE gold={st.get('gold',0)} of vct={st.get('vct',0)} "
        f"fanned={st.get('fanned',0)} nodes={node_count} wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
