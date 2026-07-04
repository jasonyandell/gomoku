import duckdb
c = duckdb.connect()
c.execute("PRAGMA memory_limit='32GB'")
c.execute("PRAGMA temp_directory='/Users/jason/.claude/jobs/09f23951/tmp/duckdb_spill'")
R = "/Users/jason/data/raphi_vct"
BUDGETS = [50, 100, 250, 500, 1000, 2000]

# win_budget: the smallest node budget at which each position was proven a VCT.
# (A win id appears as verdict='win' in exactly one rung — the rung that resolved
#  it — because once won it leaves the survivor stream. So budget is well-defined.)
union = " UNION ALL ".join(
    f"SELECT id, {b} AS budget FROM read_parquet('{R}/results/cap{b}/*.parquet') WHERE verdict='win'"
    for b in BUDGETS)
c.execute(f"CREATE TEMP TABLE win_budget AS {union}")
print("total VCT positions:", c.execute("SELECT count(*) FROM win_budget").fetchone()[0])
print("VCT positions by resolving budget:")
for b, n in c.execute("SELECT budget, count(*) FROM win_budget GROUP BY budget ORDER BY budget").fetchall():
    print(f"  cap{b}: {n}")

# every game-ply that is a VCT, tagged with the budget it needed
c.execute(f"""CREATE TEMP TABLE game_vct AS
  SELECT r.shard, r.game_idx, r.ply, wb.budget
  FROM read_parquet('{R}/positions_raw/*.parquet') r
  JOIN win_budget wb ON r.id = wb.id""")

# per game: earliest VCT (any budget) + its budget, and the earliest VCT reachable
# AT each cumulative budget ceiling
fcols = ",\n  ".join(
    f"min(ply) FILTER (WHERE budget<={b}) AS f{b}" for b in BUDGETS)
c.execute(f"""CREATE TEMP TABLE g AS
  SELECT shard, game_idx,
    min(ply) AS first_any,
    arg_min(budget, ply) AS budget_at_first,
    {fcols}
  FROM game_vct GROUP BY shard, game_idx""")

n_vct_games = c.execute("SELECT count(*) FROM g").fetchone()[0]
print("\ngames with >=1 VCT:", n_vct_games)

# (A) budget required by the game's OVERALL-earliest VCT (the true first decision)
print("\n=== (A) budget needed for the game's FIRST (earliest-ply) VCT ===")
for b, n in c.execute("SELECT budget_at_first, count(*) FROM g GROUP BY budget_at_first ORDER BY budget_at_first").fetchall():
    print(f"  cap{b}: {n}  ({100*n/n_vct_games:.1f}% of vct-games)  cumulative={'':<2}")
# cumulative %
cum = 0
print("  cumulative: first VCT resolved by budget B")
for b in BUDGETS:
    n = c.execute(f"SELECT count(*) FROM g WHERE budget_at_first<={b}").fetchone()[0]
    print(f"    <= cap{b}: {n} ({100*n/n_vct_games:.1f}%)")

# (B) for a budget-B seeker: does the game have ANY <=B VCT, and how early?
print("\n=== (B) first VCT reachable AT budget ceiling B (the seeker's view) ===")
print("  budget | %games w/ a <=B VCT | median ply | mean ply | p90 ply")
for b in BUDGETS:
    row = c.execute(f"""SELECT count(f{b}), median(f{b}), round(avg(f{b}),1),
                               quantile_cont(f{b},0.9)
                        FROM g""").fetchone()
    cov, med, mean, p90 = row
    print(f"   cap{b:<5}| {100*cov/n_vct_games:5.1f}% ({cov})      | {med:5} | {mean:6} | {p90}")

# (B') how much does the FIRST VCT move earlier as you pay more? (median over games
#      that have a <=50 VCT, so the same denominator)
print("\n=== (B') earlier-onset bought by budget, among games with a <=50 VCT ===")
base = c.execute("SELECT count(*) FROM g WHERE f50 IS NOT NULL").fetchone()[0]
for b in BUDGETS:
    med = c.execute(f"SELECT median(f{b}) FROM g WHERE f50 IS NOT NULL").fetchone()[0]
    print(f"   median first-VCT ply at cap{b}: {med}")
print(f"   (denominator: {base} games that have a 50-node VCT)")

# (C) the punchline for a 50-node seeker
g50 = c.execute("SELECT count(*) FROM g WHERE f50 IS NOT NULL").fetchone()[0]
print("\n=== (C) 50-node seeker coverage ===")
print(f"  games with at least one 50-node VCT: {g50} "
      f"({100*g50/n_vct_games:.1f}% of vct-games, {100*g50/1194662:.1f}% of ALL games)")
miss = c.execute("SELECT count(*) FROM g WHERE f50 IS NULL").fetchone()[0]
print(f"  vct-games where NO 50-node VCT ever exists (only deeper): {miss} ({100*miss/n_vct_games:.1f}%)")
print("DONE")
