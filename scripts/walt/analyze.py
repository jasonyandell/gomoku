"""Score phase JSONL against the pre-stated bets.

B1 leaves unknown -> rollout decides | B2 oracle-latency shape | B3' strength tracks the rollout policy.
"""
import json
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
summ = [r for r in rows if r.get("summary")]
games = [r for r in rows if not r.get("summary")]

print("== match table ==")
for s in summ:
    n = s["games"]
    score = (s["W"] + 0.5 * s["D"]) / n if n else float("nan")
    secs = [g["a_secs_mean"] for g in games if g["player"] == s["player"] and g["opp"] == s["opp"]]
    plies = [g["plies"] for g in games if g["player"] == s["player"] and g["opp"] == s["opp"]]
    print(f"{s['player']:>12} vs {s['opp']:<11} {s['W']:>2}-{s['L']:>2}-{s['D']:>2}  score={score:.2f}  "
          f"s/mv={np.mean(secs):.2f}  plies={np.mean(plies):.1f}  {'CAPPED' if s['capped'] else ''}")

print("\n== walt internals (search moves only) ==")
by = defaultdict(list)
for g in games:
    for m in g.get("walt_moves", []):
        by[(g["player"], g["opp"])].append(m)
for k, ms in by.items():
    fin = [m for m in ms if m["kind"] == "finisher"]
    sr = [m for m in ms if m["kind"] == "search"]
    if not sr:
        continue
    leaves = sum(m["leaves"] for m in sr)
    term = sum(m["leaf_terminal"] for m in sr)
    lv = sum(m["leaf_vct_resolved"] for m in sr)
    ro = sum(m["rollouts"] for m in sr)
    rv = sum(m["rollout_vct_resolved"] for m in sr)
    sat = np.mean([m["saturated"] for m in sr])
    tied = np.mean([m["n_tied"] > 1 for m in sr])
    agree = np.mean([m["choice"] == m["net_top"] for m in sr])
    vfrac = sum(m["vct_secs"] for m in sr) / sum(m["secs"] for m in sr)
    vb = np.mean([m["vct_boards"] / max(m["vct_calls"], 1) for m in sr])
    print(f"{k[0]} vs {k[1]}: moves={len(sr)} (+{len(fin)} finisher)")
    print(f"   B1 leaves: terminal {term/leaves:.1%}, VCT-resolved {lv/leaves:.1%}, -> rollout {(leaves-term-lv)/leaves:.1%}"
          f" | rollouts ended by VCT {rv/max(ro,1):.1%}, mean len {np.mean([m['rollout_len_mean'] for m in sr]):.1f}")
    print(f"   saturation: saturated {sat:.1%}, any tie {tied:.1%}, best_frac mean {np.mean([m['best_frac'] for m in sr]):.2f}")
    print(f"   B2 time: VCT {vfrac:.0%} of decision wall, {np.mean([m['vct_calls'] for m in sr]):.0f} calls/move, "
          f"{vb:.0f} boards/call, {np.mean([m['secs'] for m in sr]):.2f} s/move")
    print(f"   B3' agrees with net top-1 prior: {agree:.0%}")
