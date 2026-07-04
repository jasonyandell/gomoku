# idx-2 forward VCT frontier + danger map — "solve the Bruce-Lee board for black"

> **Status: HISTORICAL** *(2026-07-04)* — 2026-06-28 run record (run-a complete; deeper run unbuilt).

**One-line idea.** Forward-expand the idx-2 opening (15×15, 3-stone fair opener,
**white to move**) as an AND/OR frontier where **Rapfi's top-8 generates the moves
for both sides** and the **mega GPU VCT solver is the only oracle**, run on every
node both colours: a black-to-move VCT = **win-terminus** (harvest), a white-to-move
VCT = black-fumble **loss-terminus** (prune), anything else expands. No minimax, no
backup, no exhaustive defence — a chain of Rapfi-top-8 black moves that Rapfi-top-8
white can't refute. A **deliberately massive approximation**, run to learn what's
reachable and where the walls are. (Jason, 2026-06-28; "I don't expect to accomplish
this, but I want to run it, capture the data, and see what we learn.")

**Code:** `scripts/idx2_vct/frontier.py` (the append-only resumable expander) ·
`scripts/idx2_vct/probe_capped.py` (re-solve a capped sample at higher budgets) ·
`scripts/idx2_vct/analyze_opening.py` (build the depths-1→7 danger map).
**Builds on:** the on-device VCT megakernel `scripts/vct_metal/mega_vct_bb.py`
(`solve_vct_mega_bb`) — [mega-vct-solver.md](mega-vct-solver.md) — and the warm
`RapfiPool` (`gomoku.rapfi_pool`) — [rapfi-idx2-distillation-mine.md](rapfi-idx2-distillation-mine.md),
[eval-teacher-sensei.md](eval-teacher-sensei.md). Reducer-over-a-log design copied
from [shape-library-engine.md](shape-library-engine.md) §8's streaming minimizer.
**Root position:** `gomoku.eval_panel.IDX2_OPENING = ((3,2),(5,4),(4,5))` /
`fixed_opening_state()` — the same idx-2 the distillation campaign targets
([white-side-defense-plan.md](white-side-defense-plan.md)). Data (out-of-git):
`~/data/idx2_solve/run-a/`.

## 1. What "solve" means here, and the one honest caveat

From black's view it's an AND/OR tree: **white-to-move = AND** (black must beat every
reply), **black-to-move = OR** (black needs one winning move). The solver answers
"does the side to move have a forced VCT", so we read its verdict by parity:

| node | solver win | meaning | action |
|---|---|---|---|
| black-to-move (OR) | yes | black has a VCT | **win-terminus** (harvest) |
| white-to-move (AND) | yes | white has a VCT vs black | **loss-terminus** (black fumbled — prune) |
| either | `hit_cap` | inconclusive @250 nodes | record `capped`, keep expanding |
| either | no | quiet | expand: Rapfi top-8 |

**The unsoundness is at the AND (white) nodes.** We enumerate only Rapfi's top-8
defences, not all legal white moves, so even a fully black-won frontier proves only
"black wins vs Rapfi-top-8 white", never a true solve. Restricting black's OR moves
only makes us find *fewer* wins, never false ones — the gap is one-sided. Records keep
full parent pointers + verdicts so a line could be hardened later by exhaustive
defence expansion. **What we harvest are winning positions, NOT a backed-up strategy
from the root** (we never do the AND/OR backup — by design).

## 2. Architecture — append-only, resumable, bulk-synchronous

`frontier.py` is a level-synchronous BFS. Each wave: one **parallel Rapfi sweep**
(top-8 for every node's side-to-move) + one **≤16384 GPU solve** of all new children
(both colours) — both phases bulk, CPU and GPU never contend (the call-cost law,
[mega-vct-solver.md](mega-vct-solver.md)). State is a **reducer over a log**:

* `nodes.jsonl` is the only durable state. Each node is content-addressed
  `id = sha1(D4-canonical board)[:16]` — collapses transpositions **and** all 8 board
  symmetries (side-to-move implied by stone parity). Written once with its verdict;
  after a node's children are all flushed, a tiny `{"id":…,"done":1}` marker is appended.
* **Resume** = read the log: `seen` = every id (dedup), frontier = quiet/capped ids
  with no `done` marker; rebuild each GameState by replaying its stored move list from
  idx-2. Re-expanding a partially-expanded parent re-writes only its missing children
  (the rest are in `seen`) — crash-safe and idempotent. **Verified**: a stop-and-resume
  picked up at "frontier=1743" with zero re-expansion of completed depths.
* **Caps recorded, never dropped**: solver cap → verdict `capped`; a depth/wall/frontier
  cap leaves the unexpanded quiet nodes on disk = the frontier. The capped set is
  independently re-solvable at a higher budget.
* History is stripped from carried states (`light_state`): Rapfi and the solver read
  `board` only (full-board replay, `external_engine.py`), so the 8-ply history is dead
  weight — the difference between a ~0.7 GB and ~5 GB resident frontier.

Run: `GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python -m scripts.idx2_vct.frontier
--run-dir ~/data/idx2_solve/run-a --max-wall-secs 5400 --max-frontier 2000000`.
Resume/extend = re-run the identical command with higher caps.

## 3. The Rapfi cost knob — `max_node` binds, small-ms is the wrong tool (benchmarked)

Frontier-generation throughput is set by how Rapfi is budgeted. Measured (M5 Max,
12-engine pool, top-8): **`max_node` smoothly binds per-analyze wall-time**
(8.4 ms@200 → 51.7 ms@20000); **`timeout_ms` only bites below ~50 ms, and then it's a
guillotine** — at 25 ms Rapfi is cut off before it emits the multiPV block and the
top-8 map collapses to ~0.25 scored moves (Jaccard 0.25, garbage). Decisive tell:
`timeout_ms=200` and `max_node=20000` both land at ~51 ms (the search finishes on its
node budget, nowhere near 200 ms). **A small-ms timeout truncates the *output* and is
easily misread as "max_node doesn't work" — it does.** Chosen: **`max_node=2000`,
`timeout_ms=1000`** → peak realised **~4,700 children/s**, full top-8 support, Jaccard
0.77 vs the richest setting. Caveat: Rapfi *scores* only ~4–5 moves on most boards even
at `max_pv=8` (it omits dominated cells), so effective branching < 8.

## 4. run-a — scale, harvest, the flat-throughput wall

`run-a` (config above) hit the **90-min wall** mid-depth-11: a *full* depth-10
expansion + all of depth-11 recorded. **9,624,661 nodes**, 1.8 GB / 11.5 M log lines.
Per level (`[D→D+1]` creates depth-(D+1); wins on black-to-move/odd depths, losses on
white-to-move/even depths):

| depth | side | created | **black wins** | **black losses** | capped | → frontier | level time |
|--:|:--|--:|--:|--:|--:|--:|--:|
| 4 | white | 2.2k | – | 477 | 0.9k | 1.7k | 2.8s |
| 5 | black | 8.6k | 1,965 | – | 3.4k | 6.7k | 5.1s |
| 6 | white | 32k | – | 10,936 | 9.5k | 21k | 19.5s |
| 7 | black | 106k | 40,317 | – | 30k | 66k | 60.5s |
| 8 | white | 324k | – | 123,016 | 84k | 201k | 185s |
| 9 | black | 1.02M | 413,313 | – | 255k | 610k | 584s |
| 10 | white | 3.06M | – | 1,202,248 | 738k | 1.86M | 1704s |
| 11 | black | 5.06M | 1,962,704 | – | 1.21M | 3.10M | 2837s |

**Totals: 2.42M black-VCT wins · 1.34M black-fumble losses · 2.33M capped · 3.10M
frontier left unexpanded.** Three findings:

* **Throughput is dead flat** at ~1,750 children/s (1756→1749→1751→1799→1785). There
  is **no per-node slowdown** — each level takes ~3× longer purely because it has ~3×
  more nodes. Each node is ~0.57 ms amortised; the `max_nodes=250` cap clips the solver
  tail so a fuller, harder depth-10 board costs no more than a depth-6 one (the price is
  a steady ~24–28% capped fraction). **This is why "8 nodes is trivial and millions
  isn't a wall"**: bulk-synchronous flat throughput turns "exponential branching is
  hopeless" into "exponential branching costs exponential *time* at a flat per-node
  rate" — buying ~4 extra plies. A naïve solve-in-a-loop would eat the 25–70 s solver
  tail *per batch* and never clear depth 3.
* **Branching is decelerating and parity-asymmetric.** Frontier multiplier held
  ~3.0–3.1×/level through depth 10, then **halved to 1.66×** at depth 11: black
  (attacker) nodes get ~5.1 Rapfi candidate moves, white (defender) only ~2.8 (Rapfi
  prunes white's reasonable defences ~2× harder), and the terminus fraction climbs from
  ~22% (depth 4–5) to a ~39% plateau (wins and losses symmetric). Net effective
  branching **~2.25×/ply** and shrinking.
* **Dedup barely mattered (~2–3%).** D4 + transposition collapse saved only ~2–3%/level
  — at these short depths with distinct Rapfi-top-8 picks, move-orders rarely
  reconverge. The tree is nearly a tree, not a DAG; the sub-8 branching is Rapfi-pruning
  + terminus-pruning, not symmetry folding.

**The wall is TIME, not space.** Disk 1.8 GB, RAM ~5 GB peak — both had headroom (this
Mac could reach ~30–40 M nodes). Per-level time triples (60→185→584→1704→2837 s), so
depth 12 ≈ 2.5 hr, depth 13 ≈ 7 hr. By depth 10, **~39% of the frontier is black
hanging itself** (fumble-losses), and the surviving "quiet" frontier is mostly black
lines that are doomed-but-not-yet-proven-doomed — the approximation's seams, visible in
the data.

## 5. What the 250-node cap hides — almost nothing cheap

`probe_capped.py` reservoir-samples 3,000 capped nodes per depth and re-solves at up to
**16× budget**. `b=250` reproduces 100% capped (a reconstruction + determinism check).
Then, fraction of the capped sample still capped at `b=4000`: depth 7 **97.7%**, depth 9
**96.5%**, depth 10 **93.7%**, depth 11 **95.4%** — only <1% flip to a win, 2–5% to a
decided no-VCT. **Capped is a genuinely hard third regime** (intrinsically large VCT
search trees), not low-hanging fruit. So (1) the 2.42 M win harvest is a near-floor —
the cap is not hiding a big pool of missed wins — and (2) `max_nodes=250` is
well-calibrated: it decides the easy ~75% fast and quarantines the expensive ~25%
cheaply, exactly the tail the call-cost law warns about.

## 6. The danger map (depths 0→7) — `analyze_opening.py`

Depths 1–7 are fully expanded (**149,627 nodes**), so we build a complete, honest,
explorable danger map. Per node, the oracle `verdict` (the only sound fact) plus
**descriptive subtree danger densities** (to depth 7): `white_threat` = fraction of the
explored subtree reaching a black VCT (danger to white), `black_threat` = fraction
reaching a black fumble (danger to black); `nearest_black_win`/`nearest_black_loss`
(plies); and **honesty fields** `cap_frac` / `open_frac` / `uncertainty` /
`frontier_edge` / `n_children` vs `n_legal` — low threat + high uncertainty is
**unknown, not safe**, kept separate so a UI never paints it green. Each explored move
gets a `danger_rank` (mover maximises danger-to-opponent − danger-to-self), a
`gives_opponent_vct` flag (sound 1-ply blunder), and its `rapfi_winrate`/`rapfi_rank`
prior (re-queried, ~30 k internal nodes, seconds). Output: `analysis/map.jsonl` (flat,
parent+children pointers — drill-down ready) + `analysis/summary.json`. **No UI built
yet** (deliberately — compute first).

What it says:

* **idx-2 reads black-favourable but mostly unknown.** Root `white_threat 0.283` vs
  `black_threat 0.076` (black's found-attack density ~3.7× its found-fumble density),
  but `uncertainty 0.534` — over half the shallow tree is capped-or-frontier. Honest
  headline: *of what we could resolve, black's chances dominate ~4:1, but we resolved
  only ~47%.* (Densities are not proofs; white needs only one defence.)
* **Danger oscillates with the tempo** (mean over each depth): black-VCT density peaks
  on black's move (d5 0.44, d7 0.38), black-fumble density peaks on white's move
  (d4 0.39, d6 0.34). The initiative swings ply by ply, visible in the rollup.
* **Depth-1 entry grid — Rapfi vs the danger we found.** White's 8 first moves: Rapfi's
  top-2 ((5,5) rank 0, (4,4) rank 1) *are* the lowest-danger-to-white moves (the oracle
  agrees its best defences are safest), but its mid-ranking is **not** danger-calibrated
  — **(6,6), Rapfi's 3rd pick, is the single most black-dangerous first move**
  (`white_threat 0.319`). And the honesty catch: Rapfi's #1 (5,5) looks safest (0.190)
  partly because it is the **most unresolved** (uncertainty 0.617) — low danger there is
  "we couldn't find it", not "proven safe".
* **The per-move ranking works** (typical white-to-move node): the oracle's danger-best
  defence (contains black 89%, 0% black success) is **Rapfi's 2nd pick**, while Rapfi's
  #1 is only danger-rank 4 — concrete positions where "play the move that best contains
  the opponent" diverges from the engine prior.
* **Already-lost pockets exist.** Some depth-3 black positions have only 2 Rapfi-scored
  moves and *both lose* (every explored continuation is a white VCT); black is
  effectively dead there, and the map flags it (`gives_opponent_vct` on all children,
  `mover_value -1`).

## 7. Honest bounds — what this is NOT

* **Not a solve.** Top-8 at AND nodes ⇒ no internal node is soundly decided (every win
  needs all-defences, every loss needs all-attacks — both have move-space gaps). The
  *leaves* and the *1-ply refutations* are the sound parts; the densities are heuristic
  "danger we could find".
* **Harvest ≠ strategy.** 2.42 M winning positions are scattered leaves, not a
  backed-up plan from the root. The AND/OR backup (cheap reducer over the log) is the
  obvious next analysis — it would show how much *near the root* is actually decided
  (almost certainly "unknown", but the shape is informative).
* **Scored ≈ 4–5 moves, not 8.** "top-8" means "up to 8 scored candidates"; Rapfi omits
  dominated cells. The branching/coverage figures are top-K-restricted throughout.

## 8. Repro + next levers

Data: `~/data/idx2_solve/run-a/` (`nodes.jsonl`, `run.json`, `run.log`,
`analysis/{map.jsonl,summary.json}`). All three scripts: `scripts/idx2_vct/`.

* **Deeper run** — re-run `frontier.py` with higher `--max-frontier`/`--max-wall-secs`;
  it resumes from the 3.10 M frontier instantly (depth 12 ≈ 2.5 hr).
* **AND/OR backup over the log** — proven-win/loss/unknown bottom-up (honest about
  caps + the top-K gap).
* **Prune doomed black lines earlier** — don't expand a black move one ply from a white
  VCT; concentrate compute on viable lines instead of blunders.
* **The UI** — render `map.jsonl` as a drill-down grid (danger-shaded, uncertainty
  separated, Rapfi-prior overlay). Data is ready; nothing built.
* **Harden a line** — pick a black_win terminus and exhaustively enumerate white
  defences along its path to test soundness beyond Rapfi-top-8.
