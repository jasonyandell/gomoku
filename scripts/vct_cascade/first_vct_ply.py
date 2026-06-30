import duckdb
c = duckdb.connect()
c.execute("PRAGMA memory_limit='32GB'")
c.execute("PRAGMA temp_directory='/Users/jason/.claude/jobs/09f23951/tmp/duckdb_spill'")

R = "/Users/jason/data/raphi_vct"

# A position is a VCT ("win") if ANY rung labeled it win (verdict is monotone:
# more budget only ever converts cap->win/no_win, never un-wins).
c.execute(f"""
CREATE TEMP TABLE win_ids AS
SELECT DISTINCT id FROM read_parquet('{R}/results/cap*/*.parquet') WHERE verdict='win'
""")
nwin = c.execute("SELECT count(*) FROM win_ids").fetchone()[0]
print("distinct win (VCT) position ids:", nwin)

# Per game: first ply whose position is a VCT-win. ply = #stones on board before
# that move, so ply=k means k moves already played (ply 0 = empty board).
c.execute(f"""
CREATE TEMP TABLE first_vct AS
SELECT r.shard, r.game_idx, min(r.ply) AS first_ply
FROM read_parquet('{R}/positions_raw/*.parquet') r
SEMI JOIN win_ids w ON r.id = w.id
GROUP BY r.shard, r.game_idx
""")

# total distinct games in the corpus
total_games = c.execute(f"""
SELECT count(*) FROM (
  SELECT DISTINCT shard, game_idx FROM read_parquet('{R}/positions_raw/*.parquet')
)
""").fetchone()[0]
games_with_vct = c.execute("SELECT count(*) FROM first_vct").fetchone()[0]

stats = c.execute("""
SELECT round(avg(first_ply),2), median(first_ply),
       min(first_ply), max(first_ply),
       quantile_cont(first_ply,0.25), quantile_cont(first_ply,0.75),
       quantile_cont(first_ply,0.10), quantile_cont(first_ply,0.90)
FROM first_vct
""").fetchone()

print("total games:", total_games)
print("games with >=1 VCT:", games_with_vct,
      f"({100*games_with_vct/total_games:.1f}%)")
print("games with NO VCT:", total_games - games_with_vct)
print("--- ply-to-first-VCT (over games that have one) ---")
print("mean   :", stats[0])
print("median :", stats[1])
print("min/max:", stats[2], "/", stats[3])
print("p10/p25/p75/p90:", stats[6], "/", stats[4], "/", stats[5], "/", stats[7])

# split by which side gets the first VCT (ply parity: even ply = black to move
# in a black-first game; report both buckets)
side = c.execute("""
SELECT first_ply % 2 AS parity, count(*), round(avg(first_ply),2)
FROM first_vct GROUP BY parity ORDER BY parity
""").fetchall()
print("--- by side-to-move parity (0=even ply, 1=odd ply) ---")
for p, n, a in side:
    print(f"  parity {p}: {n} games, mean first_ply {a}")

# histogram of first VCT ply in coarse buckets
hist = c.execute("""
SELECT CASE
         WHEN first_ply < 10 THEN '00-09'
         WHEN first_ply < 20 THEN '10-19'
         WHEN first_ply < 30 THEN '20-29'
         WHEN first_ply < 40 THEN '30-39'
         WHEN first_ply < 50 THEN '40-49'
         ELSE '50+' END AS bucket, count(*)
FROM first_vct GROUP BY bucket ORDER BY bucket
""").fetchall()
print("--- distribution of first-VCT ply ---")
for b, n in hist:
    print(f"  {b}: {n} ({100*n/games_with_vct:.1f}%)")
print("DONE")
