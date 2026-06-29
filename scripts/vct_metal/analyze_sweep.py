"""Analyze the mega-VCT throughput sweep (issue #95): turn results.jsonl +
profiles.jsonl into the throughput map, the per-pool hardness curves, and the
oracle-vs-blind gap (what routing by KNOWN hardness buys). CPU-only; safe to run
while the GPU sweep is still going (reads whatever rows exist so far).

  PYTHONPATH=. uv run python -m scripts.vct_metal.analyze_sweep --out ~/data/idx2_solve/sweep
"""
import argparse
import json
import os
from collections import defaultdict


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def strat_label(r):
    s = r["strategy"]
    if s == "base":
        return f"base@{r['budget']}"
    if s == "chunked":
        return f"chunked@{r['budget']}"
    if s == "worksteal":
        return f"worksteal@{r['budget']}/r{r['resident']}"
    if s == "deepen":
        return f"deepen[{r.get('ladder_name')}]"
    if s == "deepen_ws":
        return f"deepen_ws[{r.get('ladder_name')}]"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/data/idx2_solve/sweep"))
    ap.add_argument("--md", help="write a markdown report to this path")
    args = ap.parse_args()
    results = [r for r in load_jsonl(os.path.join(args.out, "results.jsonl")) if "error" not in r]
    profiles = load_jsonl(os.path.join(args.out, "profiles.jsonl"))
    errors = [r for r in load_jsonl(os.path.join(args.out, "results.jsonl")) if "error" in r]

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out(f"# mega-VCT throughput sweep — {len(results)} configs, {len(profiles)} profiles, "
        f"{len(errors)} errors\n")

    # ---- hardness profiles (the board-population shape per pool) ----
    out("## Hardness profiles — cumulative % resolved by budget\n")
    for p in profiles:
        hist = {int(k): v for k, v in p["hist_min_budget"].items()}
        N = p["N"]; ladder = p["ladder"]
        cum = 0
        cells = []
        for b in ladder:
            cum += hist.get(b, 0)
            cells.append(f"{b}:{100*cum/N:.0f}%")
        out(f"- **{p['pool']}** (N={N}, MAXD={p.get('maxd')}): " + "  ".join(cells)
            + f"  | still-capped@{ladder[-1]}: {100*p['still_capped']/N:.1f}%")
    out()

    # ---- throughput map: best strategy per pool ----
    by_pool = defaultdict(list)
    for r in results:
        by_pool[r["pool"]].append(r)
    out("## Throughput map — evals/min by strategy, per pool (top 8)\n")
    for pool, rs in sorted(by_pool.items()):
        rs = sorted(rs, key=lambda r: -r["evals_per_min"])
        out(f"### pool = {pool}")
        for r in rs[:8]:
            out(f"  {r['evals_per_min']:>10.0f} ev/min  {strat_label(r):28}  "
                f"resolved={r['resolved']}/{r['N']}  wins={r.get('wins','?')}  ({r['wall_s']}s)")
        # deepening vs a single dispatch at the SAME ceiling (the resolve-to-ceiling
        # job — same boards resolved, less wall = win). evals/min == resolved/min
        # here only matters via wall, so compare wall on equal resolved sets.
        base_by_b = {r["budget"]: r for r in rs if r["strategy"] == "base"}
        for dl in [r for r in rs if r["strategy"] in ("deepen", "deepen_ws")]:
            ceil = dl["ladder"][-1]
            b = base_by_b.get(ceil)
            if b:
                spd = b["wall_s"] / dl["wall_s"] if dl["wall_s"] else 0
                out(f"  -> {strat_label(dl)} vs base@{ceil}: wall {dl['wall_s']}s vs "
                    f"{b['wall_s']}s = {spd:.2f}x  (resolved {dl['resolved']} vs {b['resolved']})")
        out()

    # ---- oracle estimate: route each board to its min budget (no re-solve tax) ----
    # Cost model: aggregate node-rate R (nodes/s) fit from base rows; a board that
    # resolves at budget b costs ~b nodes (upper bound). Oracle work = sum_b hist[b]*b.
    out("## Oracle bound (estimate) — route each board to its min budget\n")
    out("Cost proxy: nodes = budget at resolution; rate from the profile's own walls. "
        "An IDEALIZED lower bound (ignores per-board variance + the single-board tail); "
        "the gap to the best blind strategy = what routing by known hardness can buy.\n")
    for p in profiles:
        hist = {int(k): v for k, v in p["hist_min_budget"].items()}
        N = p["N"]; ladder = p["ladder"]
        # fit nodes/s from this profile's rungs: each rung did `in` boards up to `bud` nodes
        tot_node = sum(min(pr["in"], N) * b for b, pr in
                       ((int(k), v) for k, v in p["per_rung"].items()))
        tot_wall = sum(pr["wall_s"] for pr in p["per_rung"].values())
        rate = tot_node / tot_wall if tot_wall else 0
        oracle_nodes = sum(hist.get(b, 0) * b for b in ladder) + p["still_capped"] * ladder[-1]
        ladder_nodes = tot_node                       # the ascending-ladder did exactly this
        oracle_wall = oracle_nodes / rate if rate else 0
        out(f"- **{p['pool']}**: ~{rate/1000:.0f}k nodes/s; ladder paid ~{ladder_nodes/1e6:.1f}M "
            f"node-units, oracle ~{oracle_nodes/1e6:.1f}M => oracle ~{ladder_nodes/max(oracle_nodes,1):.2f}x "
            f"cheaper than the low-ladder (re-solve tax avoided)")
    out()

    if errors:
        out("## Errors\n")
        for r in errors[:20]:
            out(f"  {r.get('pool')} {r.get('strategy')} b={r.get('budget')}: {r['error']}")

    if args.md:
        with open(args.md, "w") as f:
            f.write("\n".join(lines))
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
