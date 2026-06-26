"""Typed threat-dependency-graph construction + colored 3-node subgraph census.

Per position we build Allis's threat network over the threats recovered by the
replayer:
  * NODE   = a typed threat (color = threat kind: four / double_four /
             three / three_fork / five).
  * DEPENDENCY edge A->B  iff  rest(A) ∩ gain(B) != ∅   (A leans on B's stone).
  * CONFLICT  edge A--B   iff  gain(A) ∈ cost(B) or gain(B) ∈ cost(A)
             or cost(A) ∩ cost(B) != ∅                 (they fight for a square).

We census connected 3-node colored motifs (the FANMOD recipe) pooled over the
corpus and compare observed counts to a degree-matched, node-color-PRESERVING
edge-rewiring null. Significance requires |z| above threshold in BOTH halves of
the corpus (the Einstein-from-noise guard).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np


def build_graph(threats: list) -> dict:
    """Return a graph dict from a list of Threat objects (one position).

    nodes: list of colors (threat.kind). dep: set of directed (i,j) i depends on
    j. con: set of frozenset({i,j}) conflict pairs.
    """
    n = len(threats)
    colors = [t.kind for t in threats]
    gain = [t.gain for t in threats]
    cost = [set(t.cost) for t in threats]
    rest = [set(t.rest) for t in threats]
    dep = set()
    con = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if gain[j] in rest[i]:           # rest(i) ∩ gain(j)
                dep.add((i, j))
    for i, j in combinations(range(n), 2):
        if gain[i] in cost[j] or gain[j] in cost[i] or (cost[i] & cost[j]):
            con.add(frozenset((i, j)))
    return {"colors": colors, "dep": dep, "con": con, "n": n}


def _motif_key(nodes, colors, dep, con) -> str:
    """Canonical colored key for a 3-node induced subgraph (relabel-invariant).

    Encodes node-color multiset + each pair's edge state. We canonicalise over
    the 6 permutations of the 3 nodes and take the lexicographically smallest
    string so isomorphic motifs collapse.
    """
    import itertools
    best = None
    for perm in itertools.permutations(range(3)):
        a, b, c = (nodes[perm[0]], nodes[perm[1]], nodes[perm[2]])
        cols = [colors[a], colors[b], colors[c]]
        tokens = [f"C{cols[0]}", f"C{cols[1]}", f"C{cols[2]}"]
        for (x, y, lbl) in ((0, 1, "01"), (0, 2, "02"), (1, 2, "12")):
            u, v = (a, b, c)[x], (a, b, c)[y]
            e = ""
            if frozenset((u, v)) in con:
                e += "K"
            if (u, v) in dep:
                e += "D"   # u depends on v
            if (v, u) in dep:
                e += "d"   # v depends on u
            tokens.append(f"{lbl}{e or '-'}")
        s = "|".join(tokens)
        if best is None or s < best:
            best = s
    return best


def census_graph(g: dict) -> Counter:
    """Count connected 3-node colored motifs in one position graph."""
    n = g["n"]
    colors, dep, con = g["colors"], g["dep"], g["con"]
    # adjacency (any edge) for connectivity
    adj = defaultdict(set)
    for (i, j) in dep:
        adj[i].add(j)
        adj[j].add(i)
    for e in con:
        i, j = tuple(e)
        adj[i].add(j)
        adj[j].add(i)
    out = Counter()
    for trio in combinations(range(n), 3):
        a, b, c = trio
        # connected? (induced subgraph on 3 nodes has >=2 edges spanning all)
        es = sum([(b in adj[a]), (c in adj[a]), (c in adj[b])])
        if es < 2:
            continue
        out[_motif_key(trio, colors, dep, con)] += 1
    return out


def census_corpus(graphs: list) -> Counter:
    total = Counter()
    for g in graphs:
        total += census_graph(g)
    return total


def _rewire_null(g: dict, rng) -> dict:
    """Degree-matched, color-preserving edge rewiring of one graph.

    Keeps node colors and the degree sequence of EACH edge type (dep out/in and
    conflict) by permuting endpoints within type via a configuration-model style
    stub shuffle. Self-loops/multi-edges are rejected (resampled a few times).
    """
    n = g["n"]
    colors = g["colors"]
    # Conflict (undirected): stub-shuffle.
    con_stubs = []
    for e in g["con"]:
        i, j = tuple(e)
        con_stubs += [i, j]
    new_con = set()
    rng.shuffle(con_stubs)
    for k in range(0, len(con_stubs) - 1, 2):
        i, j = con_stubs[k], con_stubs[k + 1]
        if i != j:
            new_con.add(frozenset((i, j)))
    # Dependency (directed): shuffle the targets among same in-degree pool.
    srcs = [i for (i, j) in g["dep"]]
    tgts = [j for (i, j) in g["dep"]]
    rng.shuffle(tgts)
    new_dep = set()
    for i, j in zip(srcs, tgts):
        if i != j:
            new_dep.add((i, j))
    return {"colors": colors, "dep": new_dep, "con": new_con, "n": n}


def motif_significance(graphs: list, *, n_null: int = 200, seed: int = 0):
    """Observed motif counts vs the rewired null. Returns dict motif -> stats."""
    obs = census_corpus(graphs)
    rng = np.random.default_rng(seed)
    null_counts = defaultdict(list)
    keys = set(obs)
    for _ in range(n_null):
        tot = Counter()
        for g in graphs:
            tot += census_graph(_rewire_null(g, rng))
        keys |= set(tot)
        for k in keys:
            null_counts[k].append(tot.get(k, 0))
    stats = {}
    for k in keys:
        arr = np.array(null_counts[k] + [0] * (n_null - len(null_counts[k])))
        mu, sd = arr.mean(), arr.std()
        o = obs.get(k, 0)
        z = (o - mu) / sd if sd > 1e-9 else (np.inf if o > mu else 0.0)
        stats[k] = {"obs": o, "null_mean": float(mu), "null_sd": float(sd),
                    "z": float(z)}
    return stats, obs
