"""E-B: walt as a SENSE — coverage defense.

walt's surviving essence is a coverage count: the share of sampled worlds that one
commitment survives. For the defender: per candidate c, pressure(c) = share of worlds
(attacker's next stone o1 ~ field via tickertape) in which, after c and o1, the attacker
would have a VCT if the defender passed (null-move solve: swap planes). One action must
cover every world, so the walt move minimizes uncovered worlds.

Stages (each caches to --work):
  positions  (GPU light)  decisive Rapfi games -> loser-to-move positions 1/3/5 plies before
                          the winner's first proven VCT (cap50, hit-caps re-solved at cap2000)
  sense      (GPU)        top-M candidates, veto (attacker VCT right after c), pressure over W
                          worlds, net prior, MCTS200 visits
  choose     (CPU)        choosers, all restricted to non-vetoed top-M (veto = existing stack)
  judge      (CPU)        Rapfi@50 plays both sides from after each distinct chosen move;
                          defender loss = 1. Primary metric: loss rate per chooser (paired).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from gomoku.game import GameState  # noqa: E402

CHAMP = "/Users/jason/data/sound-world-107b/sweep_runs/sound-world/checkpoints/worker_weights.pt"
N = 9


def load_net():
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint
    from gomoku.util import pick_device
    d = pick_device(None)
    m, _ = load_checkpoint(CHAMP, device=d)
    ev = make_torch_evaluator(fuse_model_for_inference(m), d)
    ev([GameState.initial()])
    return ev


def solver():
    from gomoku.eval import _load_vct_solver
    s = _load_vct_solver()
    s(np.zeros((1, 2, N, N), bool), max_nodes=1)
    return s


def replay(moves):
    s = GameState.initial()
    out = [s]
    for a in moves:
        s = s.apply(int(a))
        out.append(s)
    return out


def big_solve(solve, boards, nodes, chunk=16384):
    w = np.zeros(len(boards), bool); cap = np.zeros(len(boards), bool)
    for i in range(0, len(boards), chunk):
        r = solve(np.ascontiguousarray(boards[i:i + chunk]), max_nodes=nodes)
        w[i:i + chunk], cap[i:i + chunk] = np.asarray(r[0], bool), np.asarray(r[1], bool)
    return w, cap


# ------------------------------------------------------------------ positions
def stage_positions(args):
    sys.path.insert(0, os.path.join(REPO, "scripts", "threat_shapes"))
    from read_games import iter_games
    solve = solver()
    games = [g for g in iter_games(args.games) if g.get("winner") in (0, 1)]
    print(f"decisive games: {len(games)}", flush=True)
    # every position of every decisive game; side-to-move-first boards
    idx, boards = [], []
    for gi, g in enumerate(games):
        for p, s in enumerate(replay(g["moves"])[:-1]):
            idx.append((gi, p)); boards.append(s.board)
    boards = np.stack(boards).astype(bool)
    w, cap = big_solve(solve, boards, 50)
    hc = np.flatnonzero(cap & ~w)
    if hc.size:
        w2, _ = big_solve(solve, boards[hc], 2000)
        w[hc] |= w2
    win_at = {}
    for (gi, p), ww in zip(idx, w):
        win_at[(gi, p)] = bool(ww)
    rows = []
    for gi, g in enumerate(games):
        winner = g["winner"]  # 0 = black
        plies = len(g["moves"])
        onset = next((p for p in range(plies) if p % 2 == winner and win_at[(gi, p)]), None)
        if onset is None:
            continue
        for back in (1, 3, 5):
            p = onset - back
            if p < 6 or win_at.get((gi, p), False):
                continue
            rows.append({"game": gi, "ply": p, "back": back, "defender_black": p % 2 == 0,
                         "moves": g["moves"][:p], "rapfi_move": int(g["moves"][p])})
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    rows = rows[:args.max_positions]
    os.makedirs(args.work, exist_ok=True)
    json.dump(rows, open(os.path.join(args.work, "positions.json"), "w"))
    print(f"positions: {len(rows)}  (back=1/3/5: "
          f"{sum(r['back']==1 for r in rows)}/{sum(r['back']==3 for r in rows)}/{sum(r['back']==5 for r in rows)}; "
          f"defender black {sum(r['defender_black'] for r in rows)})", flush=True)


# ------------------------------------------------------------------ sense
def stage_sense(args):
    import walt_vec as wv
    from gomoku.mcts import MCTSGame, policy_from_visits, run_batched_mcts
    rows = json.load(open(os.path.join(args.work, "positions.json")))
    ev = load_net(); solve = solver()
    states = [replay(r["moves"])[-1] for r in rows]
    P, M, W = len(states), args.cand, args.worlds
    t0 = time.perf_counter()
    logits, _ = ev(states)
    prior = np.zeros((P, N * N))
    cands = np.zeros((P, M), int)
    for i, s in enumerate(states):
        z = np.where(s.legal_mask(), logits[i].astype(np.float64), -np.inf)
        z -= z.max(); e = np.exp(z); prior[i] = e / e.sum()
        cands[i] = np.argsort(-prior[i])[:M]
    # after c: attacker to move
    after = [states[i].apply(int(cands[i, j])) for i in range(P) for j in range(M)]
    ab, ah = wv.state_arrays(after)
    five_after = np.array([s.is_terminal()[0] for s in after])  # defender completed five (defender wins)
    veto, _ = big_solve(solve, ab, 50)                         # attacker VCT right after c
    veto &= ~five_after
    # worlds: attacker's o1 ~ net@T (tickertape); finisher if veto
    tapes = np.random.default_rng(args.seed).integers(1, 2**62, W).astype(np.uint64)
    probs = wv.field_probs("net", ev, ab, ah, args.temp)
    rep = np.repeat(np.arange(P * M), W)
    tp = np.tile(tapes, P * M)
    o1 = wv.draw(probs[rep], wv.tape_u(ab[rep], tp))
    b2 = ab[rep].copy()
    r_, c_ = np.divmod(o1, N)
    b2[np.arange(len(rep)), 0, r_, c_] = True
    att_five = wv.five(b2[:, 0])
    # null move: attacker to move again -> attacker-first board is b2 as-is (plane 0 = attacker)
    nul_w, nul_cap = big_solve(solve, b2, 50)
    lost = (att_five | nul_w).reshape(P * M, W)
    pressure = lost.mean(1)
    pressure[veto] = 1.0
    pressure[five_after] = 0.0
    pressure = pressure.reshape(P, M)
    veto = veto.reshape(P, M)
    t_sense = time.perf_counter() - t0
    # MCTS200 visits (same net), batched across positions
    games = [MCTSGame(s, c_puct=1.5, fpu_reduction_c=0.45, rng=np.random.default_rng(i)) for i, s in enumerate(states)]
    run_batched_mcts(games, ev, n_simulations=args.sims, add_root_noise=False)
    visits = np.stack([policy_from_visits(g.root, temperature=1.0) for g in games])
    np.savez(os.path.join(args.work, "sense.npz"), prior=prior, cands=cands, veto=veto,
             pressure=pressure, visits=visits, nul_cap=nul_cap.reshape(P, M, W).mean(2))
    print(f"sense: P={P} M={M} W={W} boards={P*M*W} in {t_sense:.1f}s; "
          f"veto rate {veto.mean():.2%}; pressure mean {pressure.mean():.3f}; "
          f"null-solve hitcap {nul_cap.mean():.2%}", flush=True)


# ------------------------------------------------------------------ choose
def stage_choose(args):
    rows = json.load(open(os.path.join(args.work, "positions.json")))
    z = np.load(os.path.join(args.work, "sense.npz"))
    prior, cands, veto, pr, visits = z["prior"], z["cands"], z["veto"], z["pressure"], z["visits"]
    P, M = cands.shape
    pc = np.take_along_axis(prior, cands, 1)
    vc = np.take_along_axis(visits, cands, 1)
    ok = ~veto
    ok[~ok.any(1)] = True  # everything vetoed -> no filter
    def pick(score):
        s = np.where(ok, score, -np.inf)
        return cands[np.arange(P), s.argmax(1)]
    beta = args.beta
    ch = {
        "prior+veto": pick(pc),
        "mcts+veto": pick(vc + 1e-9 * pc),
        "mcts*(1-p)^b": pick((vc + 1e-9 * pc) * (1 - pr) ** beta),
        "prior*(1-p)^b": pick(pc * (1 - pr) ** beta),
        "walt(min p)": pick(-pr + 1e-6 * pc),
    }
    for r_i, r in enumerate(rows):
        r["choices"] = {k: int(v[r_i]) for k, v in ch.items()}
    json.dump(rows, open(os.path.join(args.work, "choices.json"), "w"))
    names = list(ch)
    agree = {k: np.mean([r["choices"][k] == r["rapfi_move"] for r in rows]) for k in names}
    differ = np.mean([r["choices"]["mcts*(1-p)^b"] != r["choices"]["mcts+veto"] for r in rows])
    print("agreement with Rapfi's actual move (secondary):", {k: round(v, 3) for k, v in agree.items()})
    print(f"pressure changes the MCTS choice on {differ:.1%} of positions")


# ------------------------------------------------------------------ judge
def stage_judge(args):
    from gomoku.rapfi_pool import RapfiPool
    rows = json.load(open(os.path.join(args.work, "choices.json")))
    jobs = sorted({(i, m) for i, r in enumerate(rows) for m in r["choices"].values()})
    print(f"judge playouts: {len(jobs)}", flush=True)
    res = {}
    with RapfiPool(size=args.pool, timeout_ms=args.judge_ms, board_size=N) as pool:
        def play(job):
            i, m = job
            s = replay(rows[i]["moves"])[-1].apply(int(m))
            defender_moves_next = False  # attacker to move after c
            done, v = s.is_terminal()
            if done:
                return job, (0 if v == -1.0 else 0)  # defender's c completed five -> defender won
            rng = np.random.default_rng(i)
            for _ in range(N * N):
                s = s.apply(pool.pick(s, rng))
                done, v = s.is_terminal()
                if done:
                    if v == -1.0:  # side that just moved won
                        return job, (0 if defender_moves_next else 1)
                    return job, 0
                defender_moves_next = not defender_moves_next
            return job, 0
        t = time.perf_counter()
        with ThreadPoolExecutor(args.pool) as exe:
            for k, (job, lost) in enumerate(exe.map(play, jobs)):
                res[job] = lost
                if (k + 1) % 200 == 0:
                    print(f"  {k+1}/{len(jobs)} {time.perf_counter()-t:.0f}s", flush=True)
    for i, r in enumerate(rows):
        r["lost"] = {k: res[(i, m)] for k, m in r["choices"].items()}
    json.dump(rows, open(os.path.join(args.work, "judged.json"), "w"))
    report(rows)


def _judge_jobs(rows, jobs, pool_size, judge_ms):
    """Rapfi plays both sides from after the defender's move m; returns {job: lost(0/1)}."""
    from gomoku.rapfi_pool import RapfiPool
    out = {}
    with RapfiPool(size=pool_size, timeout_ms=judge_ms, board_size=N) as pool:
        def play(job):
            i, m = job
            s = replay(rows[i]["moves"])[-1].apply(int(m))
            dnext = False  # attacker moves next
            if s.is_terminal()[0]:
                return job, 0
            rng = np.random.default_rng(i)
            for _ in range(N * N):
                s = s.apply(pool.pick(s, rng))
                done, v = s.is_terminal()
                if done:
                    return job, ((0 if dnext else 1) if v == -1.0 else 0)
                dnext = not dnext
            return job, 0
        t = time.perf_counter()
        with ThreadPoolExecutor(pool_size) as exe:
            for k, (job, lost) in enumerate(exe.map(play, jobs)):
                out[job] = lost
                if (k + 1) % 400 == 0:
                    print(f"  {k+1}/{len(jobs)} {time.perf_counter()-t:.0f}s", flush=True)
    return out


def stage_detect(args):
    """Detector test: judge EVERY top-k candidate; AUC of each signal for predicting a lost move."""
    rows = json.load(open(os.path.join(args.work, "positions.json")))
    z = np.load(os.path.join(args.work, "sense.npz"))
    cands = z["cands"]
    P = min(args.detect_positions, len(rows)); K = args.detect_cand
    ev = load_net()
    after = [replay(rows[i]["moves"])[-1].apply(int(cands[i, j])) for i in range(P) for j in range(K)]
    _, v = ev(after)
    vdef = -np.asarray(v, np.float64).reshape(P, K)  # value after c, defender POV
    jobs = [(i, int(cands[i, j])) for i in range(P) for j in range(K)]
    print(f"detect playouts: {len(jobs)}", flush=True)
    res = _judge_jobs(rows, jobs, args.pool, args.judge_ms)
    lost = np.array([res[j] for j in jobs]).reshape(P, K)
    np.savez(os.path.join(args.work, "detect.npz"), lost=lost, vdef=vdef)
    detect_report(args.work)


def rankdata(a):
    """Average ranks (1-based), ties share the mean rank."""
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a)); r[order] = np.arange(1, len(a) + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / cnt)[inv]


def _logit_fit_predict(Xtr, ytr, Xte, iters=2000, lr=0.5, l2=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    A = np.c_[A, np.ones(len(A))]; B = np.c_[B, np.ones(len(B))]
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        w -= lr * (A.T @ (p - ytr) / len(A) + l2 * w)
    return B @ w


def _grouped_cv_auc(X, y, groups, k=5):
    ug = np.unique(groups)
    fold = {g: i % k for i, g in enumerate(np.random.default_rng(0).permutation(ug))}
    f = np.array([fold[g] for g in groups])
    aucs = [_auc(_logit_fit_predict(X[f != i], y[f != i], X[f == i]), y[f == i]) for i in range(k)]
    return float(np.nanmean(aucs)), float(np.nanstd(aucs))


def _auc(score, y):
    y = np.asarray(y, bool); s = np.asarray(score, float)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def detect_report(work):
    z = np.load(os.path.join(work, "sense.npz")); d = np.load(os.path.join(work, "detect.npz"))
    lost, vdef = d["lost"], d["vdef"]
    P, K = lost.shape
    cands = z["cands"][:P, :K]
    sig = {
        "walt pressure": z["pressure"][:P, :K],
        "-net prior": -np.take_along_axis(z["prior"][:P], cands, 1),
        "-MCTS200 visits": -np.take_along_axis(z["visits"][:P], cands, 1),
        "-net value after c": -vdef,
    }
    y = lost.ravel().astype(bool)
    mixed = (lost.min(1) == 0) & (lost.max(1) == 1)
    print(f"\n== DETECTOR: predicting a lost candidate (Rapfi judge), {P} positions x top-{K} ==")
    print(f"lost rate {y.mean():.3f}; positions where the choice matters (some hold, some lose): {mixed.mean():.1%}")
    for k, s in sig.items():
        wp = np.nanmean([_auc(s[i], lost[i]) for i in np.flatnonzero(mixed)]) if mixed.any() else float("nan")
        print(f"{k:>20}: AUC all {_auc(s.ravel(), y):.3f} | mixed pooled {_auc(s[mixed].ravel(), lost[mixed].ravel()):.3f}"
              f" | within-position {wp:.3f}")
    if mixed.sum() >= 10:
        groups = np.repeat(np.flatnonzero(mixed), K)
        X_net = np.stack([sig[k][mixed].ravel() for k in ("-net prior", "-MCTS200 visits", "-net value after c")], 1)
        X_all = np.concatenate([X_net, sig["walt pressure"][mixed].ravel()[:, None]], 1)
        yy = lost[mixed].ravel().astype(float)
        for name, X in (("net signals", X_net), ("net + pressure", X_all)):
            m, s = _grouped_cv_auc(X, yy, groups)
            print(f"  5-fold (grouped by position) logistic AUC, {name:>14}: {m:.3f} ± {s:.3f}")


def report(rows):
    names = list(rows[0]["lost"])
    base = "mcts+veto"
    print(f"\n== defender loss rate under Rapfi@judge (n={len(rows)}) ==")
    for k in names:
        L = np.array([r["lost"][k] for r in rows])
        by = {b: L[[r["back"] == b for r in rows]].mean() for b in (1, 3, 5)}
        col = {c: L[[r["defender_black"] == c for r in rows]].mean() for c in (True, False)}
        # paired vs base: discordant counts (McNemar)
        B = np.array([r["lost"][base] for r in rows])
        better, worse = int(((B == 1) & (L == 0)).sum()), int(((B == 0) & (L == 1)).sum())
        print(f"{k:>15}: loss {L.mean():.3f}  back1/3/5 {by[1]:.2f}/{by[3]:.2f}/{by[5]:.2f}  "
              f"def-black {col[True]:.2f} def-white {col[False]:.2f}  vs {base}: saved {better} / lost {worse}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["positions", "sense", "choose", "judge", "report", "detect", "detect_report"])
    ap.add_argument("--games", default="sweep_logs/walt_sense/rapfi9")
    ap.add_argument("--work", default="sweep_logs/walt_sense/expB")
    ap.add_argument("--max-positions", type=int, default=1500)
    ap.add_argument("--cand", type=int, default=16)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--judge-ms", type=int, default=50)
    ap.add_argument("--detect-positions", type=int, default=300)
    ap.add_argument("--detect-cand", type=int, default=8)
    args = ap.parse_args()
    if args.stage == "detect_report":
        detect_report(args.work)
    elif args.stage == "report":
        report(json.load(open(os.path.join(args.work, "judged.json"))))
    else:
        globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
