# `probe_100pct.py` — eval-config sweep driver

Filed **2026-05-28** (derby-5xs). Extended **2026-05-28** (derby-u8d) to cover
the three additional MCTS search seams (FPU reduction, tree reuse, proven-
win/loss propagation + leaf-VCF) on top of the original sims × eval-VCF grid.
The one-command driver for the RESUME PLAYBOOK step 1 probe: does eval-config
alone close the v9 "100% target" gap?

## When to invoke

When a **matured checkpoint** already 100%s heuristic + lookahead2 vs the
in-repo evals but still draws (not loses) lookahead4-as-BLACK. That's the v8
champion's current shape: aggregate ≥1700 anchored elo, but the SOLE binding
gap to Jason's 100% target is lookahead4-black ~50% wins.

H2 finding (2026-05-28 code-walk): the ~50% ceiling is **search-depth at
eval**, not structural — black LOSSES are essentially zero, only the draw rate
moves. Our eval uses 100 MCTS sims while lookahead4 sees the full 4-ply
tactical horizon (depth-4 alpha-beta + threat-aware leaves + top-12
branching). At 12-candidate branching, our 100 sims = ~8 per relevant subtree —
too shallow for >4-ply forcing wins.

Five eval-config axes attack the depth gap without any training change:

1. **`--sims-grid`** (100 → {200, 400, 800}) — more total MCTS search.
2. **`--vcf-nodes-grid`** (0 → {200, 800}) — root VCF overlay; forced-fours
   found deterministically (same tactical class lookahead4 defends; merged in
   `derby-ehw`, commit `f26003b`).
3. **`--fpu-c-grid`** (derby-3w0) — KataGo FPU reduction at MCTS selection
   (`{0.0, 0.20, 0.45, 0.65}`); unvisited children inherit `parent_V − c × √Σ
   visited_priors` instead of Q=0. Default `0.0` is byte-identical OFF.
4. **`--reuse-tree-grid`** (derby-jmi) — cross-ply MCTS tree reuse at eval
   (`{0, 1}`); the picker holds a persistent `MCTSGame` across the match and
   advances the root on each played move, multiplying the effective per-move
   sim budget. Default `0` is byte-identical OFF.
5. **`--proven-prop-grid`** (derby-b3n) — KataGo proven-win/loss propagation
   in MCTS (`{0, 1}`); terminal outcomes (and optional bounded leaf-VCF
   results via `--proven-vcf-leaf-nodes-grid`) propagate upward and short-
   circuit root selection on a proven win. Default `0` is byte-identical OFF.
6. **`--proven-vcf-leaf-nodes-grid`** (derby-b3n companion) — bounded
   `solve_vcf` budget at MCTS leaf expansion (`{0, 200, 400}`). Requires
   `--proven-prop=1` to take effect. Default `0` is byte-identical OFF.

If ANY cell drives L4-black win-rate from 51% into the 70–90%+ range on the
existing champion, the 100% target is solved by eval-config alone — every
training-side proposal (reanalyze, draw-contempt, HL-Gauss) becomes a lower-
priority follow-on.

## Usage

```bash
# Default 4×3 = 12 cells (legacy derby-5xs shape; all new axes single-OFF).
# ~1–3 hours of serial GPU time at games-per-cell=40.
python scripts/probe_100pct.py \
    --checkpoint sweep_runs/derby_v8/_peaks/<champ>/peak.pt \
    --baseline lookahead:depth=4 \
    --games-per-cell 40 \
    --output probe_100pct_results.jsonl

# Full search-engine matrix (4×3×4×2×2×3 = 576 cells — pair with cheap-first
# + early-stop so we don't actually run them all):
python scripts/probe_100pct.py \
    --checkpoint sweep_runs/derby_v8/_peaks/<champ>/peak.pt \
    --baseline lookahead:depth=4 \
    --games-per-cell 40 \
    --sims-grid 100,200,400,800 \
    --vcf-nodes-grid 0,200,800 \
    --fpu-c-grid 0.0,0.20,0.45,0.65 \
    --reuse-tree-grid 0,1 \
    --proven-prop-grid 0,1 \
    --proven-vcf-leaf-nodes-grid 0,200,400 \
    --order cheap-first \
    --early-stop-on-target 0.1 \
    --output probe_100pct_full.jsonl

# Dry-run any grid (no eval):
python scripts/probe_100pct.py --dry-run --fpu-c-grid 0,0.45 --reuse-tree-grid 0,1
```

If `delo_derby.py` is running, the script refuses unless you pass
`--i-know-derby-is-running` — the probe is eval-only and doesn't touch the
derby's checkpoints / wandb, but the GPU lane is single-tenant.

## Cheap-first ordering (`--order cheap-first`, default)

The driver reorders cells by **number of active levers** (count of axes NOT
at their OFF value), with a lexicographic tiebreak so within each tier the
smallest values come first. The result:

- **Cell 0** — the all-OFF baseline (legacy sims=100, vcf=0, everything else
  at OFF). Always runs first so we have a baseline distance to compare against.
- **Cells 1..K** — each single lever isolated at its smallest non-OFF value
  (one axis flipped at a time).
- **Cells K+1..** — pair-combos, triples, etc., ordered by total active levers.

This is the right order when GPU time is finite and we expect a small lever or
a single-lever combo to close the gap: the moment the cheap cells hit the
target, `--early-stop-on-target` drops the expensive ones.

`--order full` keeps the canonical Cartesian enumeration (legacy derby-5xs
behavior, useful for reproducibility of pre-derby-u8d runs). `--order
custom-list` is reserved.

## Early-stop knobs

Both default OFF. Both may be combined.

- `--early-stop-on-target DIST` — first cell with distance ≤ DIST ends the
  sweep. Remaining cells are recorded with `early_stop="skipped:target-hit"`
  so the JSONL still has one row per requested cell (the analyst sees what
  WOULD have run).
- `--worse-than-baseline-margin MARGIN` — informational only. Once the all-
  OFF baseline has been measured, mark any later cell whose distance is ≥
  `baseline_distance + MARGIN` with `early_stop="worse"`. The sweep DOES
  continue (a single-lever cell that's slightly worse may combine with
  another into a winner), but the flag makes losing levers easy to spot in
  the JSONL.

## Output

- **stdout** — per-cell row table (`sims, vcf, fpu, reu, pp, pvcf, n,
  BlackW/L/D, WhiteW/L/D, Bwin%, Wloss%, dist, secs, stop`) + a markdown
  distance summary:
    - **≤2 axes vary** → classic 2D pivot table (rows = first-varying axis,
      cols = second-varying axis; other axes held at OFF).
    - **≥3 axes vary** → long-form top-N summary listing each cell's axis
      values explicitly (no ambiguous pivot).
  The best (lowest-distance) cell is always tagged with `*` and called out in
  a `best cell:` line.
- **JSONL** — line 1 is meta (`{"meta": {...}}`, including all 6 grids and
  the ordering / early-stop knobs); subsequent lines are one `CellResult` per
  cell (full color-split tallies + distance + wall_secs + `early_stop`).

Distance-to-100% per cell is computed by importing `report_100pct.score`
(`scripts/report_100pct.py`) — the single source of truth for the formula:

    dist = (1 - black_win_rate) + white_loss_rate    # per the baseline contribution

0.0 = win-all-black AND lose-none-white vs the baseline. The cheapest cell
runs first so failures fail fast.

## Reading the result

- **Any cell with dist < 0.1** ⇒ that eval-config is at the 100% target on
  this checkpoint. Adopt it as the default eval config; downgrade training-side
  100%-target proposals.
- **Best cell dist ~0.4–0.5** ⇒ eval-config helps but doesn't close it; training
  levers stay in play.
- **All cells ≈ same** ⇒ H2 partially refuted (depth wasn't the binding
  constraint); redirect to training-side levers.

## Wall-time (rough)

Order-of-magnitude per cell (CPU eval, n_games=40, baseline=lookahead4):
~1–2 min at `sims=100, vcf=0`; ~10–25 min at `sims=800, vcf=800`. Full 12-cell
sweep ~1–3 hours of serial GPU time. The driver is intentionally one-shot —
the orchestrator (GPU lane) runs it; this bead only delivers the script.

## NOT in scope

- New training run / new derby cell / training-side change. This is the
  eval-only probe.
- Extending eval game counts beyond the existing harness (that's `derby-563`).
- Auto-applying the best cell as a new default — that's a ranking decision the
  orchestrator makes after reading the table.
- A separate UI / dashboard for matrix results (output stays JSONL + stdout
  markdown).
- Replacing `report_100pct.py`'s distance formula (we re-use it via import —
  single source of truth).
