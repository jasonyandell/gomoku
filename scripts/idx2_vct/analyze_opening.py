"""Build an explorable danger-map of the idx-2 opening, depths 1..D (default 7),
from the append-only frontier log. COMPUTE ONLY — no UI.

For every position in the fully-expanded shallow tree we emit, per node and per
move, a danger summary from BOTH sides, honest about caps and gaps:

  * oracle verdict at the node (black_win / black_loss / capped / quiet) — the
    only sound fact.
  * subtree danger DENSITIES (descriptive, NOT a proof — "danger we could find"):
      white_threat = fraction of the explored subtree (to depth D) that reaches a
                     black VCT  (black's attacking success = danger to WHITE)
      black_threat = fraction reaching a black-fumble loss (white VCT)
                     (danger to BLACK)
    plus nearest_black_win / nearest_black_loss (plies to the closest such leaf
    along any explored line).
  * HONESTY fields:
      cap_frac  = fraction of the subtree the solver could not decide at 250 nodes
      open_frac = fraction sitting on the depth-D frontier (we never looked deeper)
      n_children / n_legal = we only explored Rapfi's top-K; the rest is unseen.
    Low threat + high cap/open is UNKNOWN, not safe — keep them separate.
  * per-move ranking: at each node the explored child-moves are ranked by the
    MOVER's preference (maximise danger-to-opponent, minimise danger-to-self),
    and flagged when a move hands the opponent an immediate VCT (a sound blunder).
  * Rapfi prior (optional, --with-rapfi): each move's Rapfi winrate + rank, so a
    UI can show where the move-gen prior and the danger we found disagree.

This is a DESCRIPTIVE map under the Rapfi-top-K approximation, not a solve: with
both sides restricted to top-K, no internal node is soundly decided (every win
needs all-defences, every loss needs all-attacks — both have gaps). The leaves
and the immediate 1-ply refutations are the sound parts; the densities are the
heuristic "danger we could find".

Run (from worktree root):
  GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python -m scripts.idx2_vct.analyze_opening \
      --log ~/data/idx2_solve/run-a/nodes.jsonl --max-depth 7 --with-rapfi \
      --out ~/data/idx2_solve/run-a/analysis
"""
from __future__ import annotations

import os

os.environ.setdefault("GOMOKU_BOARD_SIZE", "15")

import argparse
import json
from collections import defaultdict
from pathlib import Path

from gomoku.board_config import BOARD_SIZE as N
from gomoku.eval_panel import IDX2_OPENING

# idx-2: black plays (3,2) & (4,5); white plays (5,4). After it, WHITE is to move,
# so the recorded move list alternates white, black, white, ... from idx-2.
IDX2_BLACK = [list(IDX2_OPENING[0]), list(IDX2_OPENING[2])]
IDX2_WHITE = [list(IDX2_OPENING[1])]


def rc(a: int) -> list[int]:
    return [a // N, a % N]


def stones(moves: list[int]) -> tuple[list, list]:
    """Absolute black/white stone (r,c) lists from the idx-2 move path."""
    white_played = [rc(a) for a in moves[0::2]]   # idx-2 = white to move
    black_played = [rc(a) for a in moves[1::2]]
    return IDX2_BLACK + black_played, IDX2_WHITE + white_played


def load_shallow(log: Path, D: int):
    """Load every node record at depth<=D. The log is strict-BFS by depth, so we
    cheap-parse the depth and STOP once depth>D appears (skip the `done` markers,
    which carry no depth)."""
    nodes: dict[str, dict] = {}
    children: dict[str, list] = defaultdict(list)
    with log.open() as f:
        for line in f:
            i = line.find('"depth": ')
            if i < 0:
                continue                      # `done` marker / non-node line
            d = int(line[i + 9: line.find(",", i + 9)])
            if d > D:
                break                          # BFS order => no more depth<=D
            r = json.loads(line)
            nodes[r["id"]] = r
            if r["parent"] is not None:
                children[r["parent"]].append(r["id"])
    return nodes, children


def aggregate(nodes, children, D: int):
    """Bottom-up subtree danger aggregation over depths D..0.
    Returns id -> dict(size, bw, bl, cap, open, dbw, dbl) with dbw/dbl = nearest
    black_win/black_loss distance in plies (None if none in subtree)."""
    by_depth = defaultdict(list)
    for nid, r in nodes.items():
        by_depth[r["depth"]].append(nid)
    agg: dict[str, dict] = {}

    def nearest(child_vals):
        finite = [1 + v for v in child_vals if v is not None]
        return min(finite) if finite else None

    for d in range(D, -1, -1):
        for nid in by_depth[d]:
            r = nodes[nid]
            ch = children.get(nid, [])
            if not ch:                          # leaf of the depth<=D map
                v = r["verdict"]
                a = dict(size=1, bw=0, bl=0, cap=0, open=0, dbw=None, dbl=None)
                if v == "black_win":
                    a["bw"], a["dbw"] = 1, 0
                elif v == "black_loss":
                    a["bl"], a["dbl"] = 1, 0
                elif v == "capped":
                    a["cap"] = 1
                else:                            # quiet leaf == depth-D frontier edge
                    a["open"] = 1
            else:                                # internal (was expanded)
                a = dict(size=1, bw=0, bl=0,
                         cap=(1 if r["verdict"] == "capped" else 0),
                         open=0, dbw=None, dbl=None)
                for c in ch:
                    ca = agg[c]
                    for k in ("size", "bw", "bl", "cap", "open"):
                        a[k] += ca[k]
                a["dbw"] = nearest([agg[c]["dbw"] for c in ch])
                a["dbl"] = nearest([agg[c]["dbl"] for c in ch])
            agg[nid] = a
    return agg


def rapfi_priors(nodes, children, rapfi_max_node, pool_size):
    """For every INTERNAL node, re-query Rapfi and return move->(winrate, rank)
    keyed by the child id. CPU pool; only call when the GPU/CPU is free."""
    from gomoku.rapfi_pool import RapfiPool
    from scripts.idx2_vct.frontier import state_from_moves

    internal = [nid for nid in nodes if children.get(nid)]
    out: dict[str, tuple] = {}
    CH = 4096
    with RapfiPool(size=pool_size, timeout_ms=1000, board_size=N) as pool:
        for i in range(0, len(internal), CH):
            batch = internal[i:i + CH]
            states = [state_from_moves(nodes[nid]["moves"]) for nid in batch]
            maps = pool.analyze_states(states, max_node=rapfi_max_node, max_pv=8)
            for nid, m in zip(batch, maps):
                order = sorted(m, key=m.get, reverse=True)
                rankof = {int(a): k for k, a in enumerate(order)}
                for cid in children[nid]:
                    mv = nodes[cid]["move"]
                    out[cid] = (m.get(mv), rankof.get(mv))
            print(f"  rapfi priors: {min(i+CH,len(internal))}/{len(internal)}",
                  flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-rapfi", action="store_true")
    ap.add_argument("--rapfi-max-node", type=int, default=2000)
    ap.add_argument("--pool", type=int, default=12)
    args = ap.parse_args()

    log = Path(os.path.expanduser(args.log))
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    D = args.max_depth

    print(f"[map] loading depth<= {D} from {log} ...", flush=True)
    nodes, children = load_shallow(log, D)
    print(f"[map] {len(nodes):,} nodes, {sum(len(v) for v in children.values()):,} edges",
          flush=True)

    agg = aggregate(nodes, children, D)

    priors = {}
    if args.with_rapfi:
        print("[map] re-querying Rapfi priors for internal nodes ...", flush=True)
        priors = rapfi_priors(nodes, children, args.rapfi_max_node, args.pool)

    # Per-node enrichment + per-parent danger ranking of explored moves.
    def threats(nid):
        a = agg[nid]
        s = a["size"]
        return a["bw"] / s, a["bl"] / s          # white_threat, black_threat

    enriched: dict[str, dict] = {}
    for nid, r in nodes.items():
        a = agg[nid]
        s = a["size"]
        wt, bt = threats(nid)
        bl_stones, wh_stones = stones(r["moves"])
        ch = children.get(nid, [])
        enriched[nid] = {
            "id": nid, "parent": r["parent"], "depth": r["depth"], "stm": r["stm"],
            "move": r["move"], "move_rc": (rc(r["move"]) if r["move"] is not None else None),
            "black": bl_stones, "white": wh_stones, "n_stones": len(bl_stones) + len(wh_stones),
            "verdict": r["verdict"],
            "vct_move_rc": (rc(r["vct_move"]) if r["vct_move"] >= 0 else None),
            "leaf": not ch,
            "frontier_edge": (not ch) and r["verdict"] == "quiet",
            "subtree": {"size": s, "black_win": a["bw"], "black_loss": a["bl"],
                        "capped": a["cap"], "open": a["open"]},
            "white_threat": round(wt, 5), "black_threat": round(bt, 5),
            "cap_frac": round(a["cap"] / s, 5), "open_frac": round(a["open"] / s, 5),
            "uncertainty": round((a["cap"] + a["open"]) / s, 5),
            "nearest_black_win": a["dbw"], "nearest_black_loss": a["dbl"],
            "n_children": len(ch), "n_legal": N * N - (len(bl_stones) + len(wh_stones)),
            "children": ch,
            # filled below:
            "rapfi_winrate": None, "rapfi_rank": None,
            "mover_value": None, "danger_rank": None, "gives_opponent_vct": False,
        }

    # Rank each parent's explored moves by the mover's danger preference.
    for pid, ch in children.items():
        pstm = nodes[pid]["stm"]
        sign = 1.0 if pstm == "b" else -1.0      # black maximises (wt-bt); white the opposite
        scored = []
        for cid in ch:
            e = enriched[cid]
            mv = sign * (e["white_threat"] - e["black_threat"])
            e["mover_value"] = round(mv, 5)
            # sound 1-ply blunder: the move hands the opponent an immediate VCT
            if (pstm == "b" and e["verdict"] == "black_loss") or \
               (pstm == "w" and e["verdict"] == "black_win"):
                e["gives_opponent_vct"] = True
            if cid in priors:
                e["rapfi_winrate"], e["rapfi_rank"] = priors[cid]
            scored.append((mv, cid))
        scored.sort(key=lambda t: t[0], reverse=True)
        for rank, (_mv, cid) in enumerate(scored):
            enriched[cid]["danger_rank"] = rank

    # Emit the flat map.
    mapfile = out / "map.jsonl"
    with mapfile.open("w") as f:
        for d in range(0, D + 1):
            for nid in (n for n in enriched if enriched[n]["depth"] == d):
                f.write(json.dumps(enriched[nid]) + "\n")
    print(f"[map] wrote {mapfile} ({mapfile.stat().st_size/1e6:.1f} MB)", flush=True)

    # Summary: per-depth rollup + the depth-1 entry index (white's first moves).
    per_depth = {}
    for d in range(0, D + 1):
        ids = [n for n in enriched if enriched[n]["depth"] == d]
        if not ids:
            continue
        vc = defaultdict(int)
        for n in ids:
            vc[enriched[n]["verdict"]] += 1
        per_depth[d] = {
            "nodes": len(ids),
            "verdicts": dict(vc),
            "mean_white_threat": round(sum(enriched[n]["white_threat"] for n in ids) / len(ids), 4),
            "mean_black_threat": round(sum(enriched[n]["black_threat"] for n in ids) / len(ids), 4),
            "mean_uncertainty": round(sum(enriched[n]["uncertainty"] for n in ids) / len(ids), 4),
        }

    root = next(n for n in enriched if enriched[n]["depth"] == 0)
    d1 = [enriched[c] for c in enriched[root]["children"]]
    d1.sort(key=lambda e: e["white_threat"], reverse=True)   # most dangerous for white first
    entry = [{
        "move_rc": e["move_rc"], "danger_rank_for_white": e["danger_rank"],
        "rapfi_rank": e["rapfi_rank"], "rapfi_winrate": e["rapfi_winrate"],
        "white_threat": e["white_threat"], "black_threat": e["black_threat"],
        "uncertainty": e["uncertainty"],
        "nearest_black_win": e["nearest_black_win"],
        "nearest_black_loss": e["nearest_black_loss"],
        "subtree": e["subtree"],
    } for e in d1]

    summary = {
        "log": str(log), "max_depth": D, "board_size": N,
        "idx2_opening": IDX2_OPENING, "total_nodes_mapped": len(enriched),
        "root_white_threat": enriched[root]["white_threat"],
        "root_black_threat": enriched[root]["black_threat"],
        "root_uncertainty": enriched[root]["uncertainty"],
        "per_depth": per_depth,
        "depth1_white_moves_by_danger_to_white": entry,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[map] wrote {out/'summary.json'}", flush=True)
    print(json.dumps(summary["depth1_white_moves_by_danger_to_white"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
