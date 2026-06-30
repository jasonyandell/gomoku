"""Stage 3 — report: the deepening curve + per-budget throughput knees (issue #97).

Pure DuckDB over the append-only ledger. Answers the headline question — what
fraction of positions resolve at each node budget, and how many remain capped at
the ceiling — plus the empirical knee (best sustained boards/s) per rung.

Run:
  GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_cascade.stats --out ~/data/raphi_vct
"""
from __future__ import annotations

import argparse
import os

import duckdb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = os.path.expanduser(args.out)
    res_glob = os.path.join(out, "results", "**", "*.parquet")
    perf_glob = os.path.join(out, "perf", "*.parquet")

    con = duckdb.connect()

    print("\n=== DEEPENING CURVE (resolved at each node budget) ===")
    # one definitive (non-cap) row per id at its latching rung; capped-everywhere
    # ids have only cap rows -> solved_at IS NULL.
    con.execute(f"""
        CREATE TEMP VIEW solved AS
        SELECT id,
               min(cap) FILTER (verdict <> 'cap') AS solved_at,
               any_value(verdict) FILTER (verdict <> 'cap') AS final_verdict
        FROM read_parquet('{res_glob}', hive_partitioning=false)
        GROUP BY id;
    """)
    rows = con.execute("""
        SELECT
          coalesce(solved_at, -1)              AS rung,
          count(*)                              AS n,
          round(100.0*count(*)/sum(count(*)) OVER (), 3) AS pct,
          round(100.0*sum(count(*)) OVER (ORDER BY coalesce(solved_at, 1e18))
                / sum(count(*)) OVER (), 3)     AS cum_pct
        FROM solved GROUP BY solved_at ORDER BY coalesce(solved_at, 1e18);
    """).fetchall()
    print(f"  {'budget':>8} {'count':>14} {'pct':>8} {'cum%':>8}")
    for rung, n, pct, cum in rows:
        label = "CAP@ceil" if rung == -1 else str(rung)
        print(f"  {label:>8} {n:>14,} {pct:>7.2f}% {cum:>7.2f}%")

    print("\n=== FINAL VERDICT ===")
    v = con.execute("""
        SELECT coalesce(final_verdict, 'cap_at_ceiling') AS verdict, count(*) n,
               round(100.0*count(*)/sum(count(*)) OVER (),3) pct
        FROM solved GROUP BY 1 ORDER BY n DESC;
    """).fetchall()
    for verdict, n, pct in v:
        print(f"  {verdict:>16}: {n:>14,}  ({pct:.2f}%)")

    if os.path.exists(os.path.dirname(perf_glob)):
        print("\n=== THROUGHPUT KNEE per budget (best sustained boards/s) ===")
        # the knee lives across widths — ramp rows ARE the width sweep, so use all
        # dispatches; max() ignores the small-remainder tail dispatches.
        k = con.execute(f"""
            WITH p AS (SELECT * FROM read_parquet('{perf_glob}'))
            SELECT cap,
                   max(boards_per_s) AS best_bps,
                   arg_max(batch_size, boards_per_s) AS knee_width,
                   sum(n_boards) AS boards, round(sum(wall_s),1) AS wall_s
            FROM p GROUP BY cap ORDER BY cap;
        """).fetchall()
        print(f"  {'cap':>6} {'best b/s':>14} {'knee width':>12} {'boards':>14} {'wall(s)':>10}")
        for cap, bps, knee, boards, wall in k:
            print(f"  {cap:>6} {bps:>14,.0f} {knee:>12,} {boards:>14,} {wall:>10.1f}")

    con.close()


if __name__ == "__main__":
    main()
