import duckdb
c = duckdb.connect()
c.execute("PRAGMA memory_limit='32GB'")
c.execute("PRAGMA temp_directory='/Users/jason/.claude/jobs/09f23951/tmp/duckdb_spill'")
R = "/Users/jason/data/raphi_vct"
BUDGETS = [50, 100, 250, 500, 1000, 2000]

# budget at which each position was proven a VCT (unique per id)
union = " UNION ALL ".join(
    f"SELECT id, {b} AS budget FROM read_parquet('{R}/results/cap{b}/*.parquet') WHERE verdict='win'"
    for b in BUDGETS)
c.execute(f"CREATE TEMP TABLE win_budget AS {union}")

# unresolved deep-tail positions (still cap at 2000)
c.execute(f"CREATE TEMP TABLE deep_ids AS SELECT id FROM read_parquet('{R}/survivors/cap2000/*.parquet')")

# every game-ply that is a confirmed VCT, tagged with its resolving budget
c.execute(f"""CREATE TEMP TABLE game_vct AS
  SELECT r.shard, r.game_idx, r.ply, wb.budget
  FROM read_parquet('{R}/positions_raw/*.parquet') r
  JOIN win_budget wb ON r.id = wb.id""")

# per game: the FIRST (earliest-ply) confirmed VCT and the budget that resolved it
c.execute("""CREATE TEMP TABLE g AS
  SELECT shard, game_idx,
    min(ply) AS first_vct_ply,
    arg_min(budget, ply) AS first_vct_budget
  FROM game_vct GROUP BY shard, game_idx""")

# per game: earliest UNRESOLVED deep-tail ply (if any)
c.execute(f"""CREATE TEMP TABLE gd AS
  SELECT r.shard, r.game_idx, min(r.ply) AS earliest_deep_ply
  FROM read_parquet('{R}/positions_raw/*.parquet') r
  SEMI JOIN deep_ids d ON r.id = d.id
  GROUP BY r.shard, r.game_idx""")

n = c.execute("SELECT count(*) FROM g").fetchone()[0]
print("games with >=1 confirmed VCT:", n)

print("\n=== FIRST VCT per game — budget that resolved it (cap -> VCT by looking deeper) ===")
print("  budget | games | %    | cumulative%")
cum = 0
rows = {b: 0 for b in BUDGETS}
for b, k in c.execute("SELECT first_vct_budget, count(*) FROM g GROUP BY first_vct_budget ORDER BY first_vct_budget").fetchall():
    rows[b] = k
for b in BUDGETS:
    k = rows[b]; cum += k
    tag = "  (found at shallowest budget)" if b == 50 else "  <- needed deeper than cap50"
    print(f"  cap{b:<5}| {k:7} | {100*k/n:4.1f} | {100*cum/n:5.1f}{tag if b<=100 else ''}")

deeper = n - rows[50]
print(f"\n  FIRST VCT found at cap50 (shallowest)      : {rows[50]} ({100*rows[50]/n:.1f}%)")
print(f"  FIRST VCT only revealed by looking DEEPER  : {deeper} ({100*deeper/n:.1f}%)")
print(f"  (contrast: only ~1.2% of the TOTAL VCT population needs >50 nodes —")
print(f"   first VCTs are ~{100*deeper/n/1.2:.0f}x more likely to be deep than a random VCT)")

# distribution of how-deep among the ones that needed deeper than 50
print("\n=== of the first-VCTs that needed deeper than cap50, how deep? ===")
for b in BUDGETS[1:]:
    k = rows[b]
    print(f"  cap{b:<5}: {k} ({100*k/deeper:.1f}% of the deeper-needed)")

# ---- censoring caveat: an UNRESOLVED deep-tail position earlier than the first
#      confirmed VCT means the true first VCT could be even earlier/deeper ----
c.execute("""CREATE TEMP TABLE gj AS
  SELECT g.shard, g.game_idx, g.first_vct_ply, g.first_vct_budget,
         gd.earliest_deep_ply
  FROM g LEFT JOIN gd ON g.shard=gd.shard AND g.game_idx=gd.game_idx""")
censored = c.execute("SELECT count(*) FROM gj WHERE earliest_deep_ply IS NOT NULL AND earliest_deep_ply < first_vct_ply").fetchone()[0]
print("\n=== censoring caveat ===")
print(f"  vct-games with an UNRESOLVED deep-tail position BEFORE the first confirmed VCT: {censored} ({100*censored/n:.1f}%)")
print("    (their true first VCT could be earlier & deeper than what cap2000 confirmed)")

# games with NO confirmed VCT but WITH a deep-tail position (first VCT, if any, is
# entirely in the unresolved tail)
total_games = 1194662
novct_with_deep = c.execute(f"""
  SELECT count(*) FROM (
    SELECT DISTINCT shard, game_idx FROM read_parquet('{R}/positions_raw/*.parquet')
  ) ag
  LEFT JOIN g USING (shard, game_idx)
  SEMI JOIN gd USING (shard, game_idx)
  WHERE g.shard IS NULL
""").fetchone()[0]
print(f"  games with NO confirmed VCT but WITH a deep-tail position: {novct_with_deep}")
print("    (a first VCT for these, if it exists, lives entirely past cap2000)")
print("DONE")
