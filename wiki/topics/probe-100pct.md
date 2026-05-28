# `probe_100pct.py` — eval-sims × eval-VCF sweep driver

Filed **2026-05-28** (derby-5xs). The one-command driver for the RESUME PLAYBOOK
step 1 probe: does eval-config alone close the v9 "100% target" gap?

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

Two eval-config levers attack the depth gap without any training change:

1. **`--sims`** (100 → {200, 400, 800}) — more total MCTS search.
2. **`--eval-vcf-nodes`** (0 → {200, 800}) — root VCF overlay; forced-fours
   found deterministically (same tactical class lookahead4 defends; merged in
   `derby-ehw`, commit `f26003b`).

If ANY cell in the 4×3 grid drives L4-black win-rate from 51% into the
70–90%+ range on the existing champion, the 100% target is solved by
eval-config alone — every training-side proposal (reanalyze, draw-contempt,
HL-Gauss) becomes a lower-priority follow-on.

## Usage

```bash
# Default 4×3 = 12 cells; ~1–3 hours of serial GPU time at games-per-cell=40.
python scripts/probe_100pct.py \
    --checkpoint sweep_runs/derby_v8/_peaks/<champ>/peak.pt \
    --baseline lookahead:depth=4 \
    --games-per-cell 40 \
    --sims-grid 100,200,400,800 \
    --vcf-nodes-grid 0,200,800 \
    --output probe_100pct_results.jsonl

# Dry-run the grid (no eval):
python scripts/probe_100pct.py --dry-run
```

If `delo_derby.py` is running, the script refuses unless you pass
`--i-know-derby-is-running` — the probe is eval-only and doesn't touch the
derby's checkpoints / wandb, but the GPU lane is single-tenant.

## Output

- **stdout** — per-cell row table (`sims, vcf, n, BlackW/L/D, WhiteW/L/D,
  Bwin%, Wloss%, dist, secs`) + a 4×3 markdown distance grid with the best
  cell starred.
- **JSONL** — line 1 is meta (`{"meta": {...}}`); subsequent lines are one
  `CellResult` per cell (full color-split tallies + distance + wall_secs).

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
- Sweeping `--eval-vcf-nodes` finer than the default {0, 200, 800} (the runner
  can vary if needed once we see the table).
- Auto-applying the best cell as a new default — that's a ranking decision the
  orchestrator makes after reading the table.
