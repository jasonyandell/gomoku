"""VCT-terminus science A/B — finisher eval + head-to-head (issue #100).

One matchup per process, so the whole matrix fans across cores (MCTS is
single-core-bound; a lone process leaves the GPU idle between sims). Each
`--run ID` invocation appends its result JSON to --out under flock; `--collate`
prints the tables; `--list` enumerates ids.

Three nets (terminus e500, control e500, champion = HF 17-plane derby champ,
epoch 853 ~1876 elo), each runnable raw or with the #99 cap50 VCT-finisher.

Sequential-per-process (no fork) — the #99 guard forbids the finisher under
multi-worker fork (MLX-under-fork / Metal-wedge risk); separate processes are fine.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GOMOKU_BOARD_SIZE", "9")
import argparse
import fcntl
import json

CHAMP_REPO = "jasonyandell/gomoku-9x9"
CHAMP_REV = "0d2940d5340846322f1a5d1b1cbee8ad418ea480"  # 17-plane derby champ, epoch 853

# worker_weights.pt holds the EMA (averaged) weights — what self-play AND the
# internal eval actually use ("the model"). epoch0500.pt's model_state_dict is the
# RAW (unaveraged) net, which is dramatically weaker under ema_tau=0.99 on sharp
# short-game training (terminus: 6% vs heuristic raw, ~70% EMA). Eval the EMA.
CKPTS = {
    "terminus": "sweep_runs/vctsci-terminus/checkpoints/worker_weights.pt",
    "control": "sweep_runs/vctsci-control/checkpoints/worker_weights.pt",
    "champion": None,  # resolved from HF on demand (already the deployed weights)
}
BASELINES = ["heuristic", "lookahead:d2", "lookahead:d4"]


def build_matchups():
    """Flat ordered list of matchups. id = stable index."""
    M = []
    # (A) fixed baselines
    for net, cfgs in [("terminus", ("raw", "finisher")),
                      ("control", ("raw", "finisher")),
                      ("champion", ("raw",))]:
        for cfg in cfgs:
            for b in BASELINES:
                M.append({"kind": "fixed", "a": [net, cfg], "b": b,
                          "label": f"{net}+{cfg} vs {b}"})
    # (B) head-to-head
    h2h = [
        ("terminus", "finisher", "control", "raw", "Jason CRUSH: control on its own"),
        ("terminus", "finisher", "champion", "raw", "Jason BOLD: beats best-ever 9x9"),
        ("terminus", "finisher", "control", "finisher", "fair fight (both finish)"),
        ("terminus", "finisher", "champion", "finisher", "fair fight vs champ"),
        ("terminus", "raw", "control", "raw", "raw-vs-raw sibling"),
        ("champion", "raw", "control", "raw", "calib: champ >> control?"),
    ]
    for a, ac, b, bc, note in h2h:
        M.append({"kind": "h2h", "a": [a, ac], "bb": [b, bc], "note": note,
                  "label": f"{a}+{ac} vs {b}+{bc}"})
    for i, m in enumerate(M):
        m["id"] = i
    return M


def _champ_path():
    from huggingface_hub import hf_hub_download
    return hf_hub_download(CHAMP_REPO, "model.pt", revision=CHAMP_REV)


def run_one(mid, out_path, sims, n_games, budget):
    import numpy as np
    import torch
    from gomoku.model import load_checkpoint
    from gomoku.mcts import make_torch_evaluator
    from gomoku import eval as evmod
    from gomoku.eval import mcts_picker, play_match_pickers
    from gomoku.baselines import heuristic_player, lookahead_player

    m = build_matchups()[mid]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    # fire-counter
    _real = evmod._load_vct_solver
    fires = {"n": 0, "calls": 0}

    def loader():
        solver = _real()

        def wrapped(boards, *, max_nodes, return_move=False):
            out = solver(boards, max_nodes=max_nodes, return_move=return_move)
            fires["calls"] += int(boards.shape[0])
            fires["n"] += int(np.asarray(out[0]).sum())
            return out
        return wrapped
    evmod._load_vct_solver = loader

    _models = {}

    def picker(net, cfg):
        if net not in _models:
            path = CKPTS[net] or _champ_path()
            mdl, _ = load_checkpoint(path, device=dev)
            mdl.eval()
            _models[net] = make_torch_evaluator(mdl, dev)
        vf = budget if cfg == "finisher" else 0
        return mcts_picker(_models[net], n_simulations=sims, c_puct=1.5, vct_finish_nodes=vf)

    a = picker(*m["a"])
    if m["kind"] == "fixed":
        base = {"heuristic": heuristic_player,
                "lookahead:d2": lookahead_player(depth=2),
                "lookahead:d4": lookahead_player(depth=4)}[m["b"]]
        b = base
    else:
        b = picker(*m["bb"])

    r = play_match_pickers(a, b, n_games=n_games, seed=0)
    rec = {"id": mid, "label": m["label"], "kind": m["kind"],
           "winrate": r.win_rate, "wins": r.wins, "losses": r.losses, "draws": r.draws,
           "black": [r.black_w, r.black_l, r.black_d],
           "white": [r.white_w, r.white_l, r.white_d],
           "finfires": fires["n"], "fincalls": fires["calls"],
           "note": m.get("note", "")}
    with open(out_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"[done {mid}] {m['label']}: {r.win_rate:.1%} "
          f"({r.wins}-{r.losses}-{r.draws}) fires={fires['n']}/{fires['calls']}", flush=True)


def collate(out_path):
    rows = {}
    for line in open(out_path):
        rec = json.loads(line)
        rows[rec["id"]] = rec  # last-writer-wins (dedup re-runs)
    M = build_matchups()

    def split(rec):
        bw, bl, bd = rec["black"]
        ww, wl, wd = rec["white"]
        return f"b{bw}-{bl}-{bd}/w{ww}-{wl}-{wd}"

    print("=" * 84)
    print("(A) FIXED BASELINES (canonical, transitive) — winrate + black/white split")
    print("=" * 84)
    for m in M:
        if m["kind"] != "fixed":
            continue
        r = rows.get(m["id"])
        if not r:
            print(f"  [{m['id']:>2}] {m['label']:32s}: (pending)")
            continue
        tag = f"finfires={r['finfires']}/{r['fincalls']}" if "finisher" in m["label"] else ""
        print(f"  [{m['id']:>2}] {m['label']:32s}: {r['winrate']:5.1%}  {split(r):22s} {tag}")

    print("\n=== finisher lift (fin - raw) ===")
    for net in ("terminus", "control"):
        for b in BASELINES:
            raw = next((rows[m["id"]] for m in M if m["label"] == f"{net}+raw vs {b}" and m["id"] in rows), None)
            fin = next((rows[m["id"]] for m in M if m["label"] == f"{net}+finisher vs {b}" and m["id"] in rows), None)
            if raw and fin:
                print(f"  {net:8s} vs {b:12s}: {fin['winrate']-raw['winrate']:+.1%}  "
                      f"(raw {raw['winrate']:.0%} -> fin {fin['winrate']:.0%})")

    print("\n" + "=" * 84)
    print("(B) HEAD-TO-HEAD (prediction test, NON-transitive) — winrate from FIRST net")
    print("=" * 84)
    for m in M:
        if m["kind"] != "h2h":
            continue
        r = rows.get(m["id"])
        if not r:
            print(f"  [{m['id']:>2}] {m['label']:38s}: (pending)   [{m['note']}]")
            continue
        print(f"  [{m['id']:>2}] {m['label']:38s}: {r['winrate']:5.1%}  {split(r):22s} "
              f"fires={r['finfires']}/{r['fincalls']}   [{m['note']}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--collate", type=str, default=None)
    ap.add_argument("--out", type=str, default="vctsci_eval_results.jsonl")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--n-games", type=int, default=40)
    ap.add_argument("--budget", type=int, default=50)
    args = ap.parse_args()

    if args.list:
        for m in build_matchups():
            print(f"{m['id']}\t{m['label']}")
    elif args.collate:
        collate(args.collate)
    elif args.run is not None:
        run_one(args.run, args.out, args.sims, args.n_games, args.budget)
    else:
        ap.error("one of --list / --run / --collate required")


if __name__ == "__main__":
    main()
