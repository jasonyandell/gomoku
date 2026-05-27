# Gomocup AlphaZero Implementation Technique Survey

Snapshot date: 2026-05-27. Researcher pass over the **AlphaZero-native** gomoku
engines for techniques we have **not yet raced in the Δelo Derby**. Companion to
[gomocup-external-engines-2026-05-22.md](gomocup-external-engines-2026-05-22.md)
(which surveyed engines as *eval baselines*); this one mines them for *training /
search levers*. Synthesis + registration status lives in
[../ops/research-board.md](../ops/research-board.md) "Open candidates"; the intake
mechanics are [../topics/derby-registration.md](../topics/derby-registration.md).

## Engines read (and what each is)

| Engine | Gomocup signal | What it is | Source |
|---|---|---|---|
| **AlphaGomoku (M. Kozarzewski)** | Freestyle 2256; Gomocup 2025 **2nd** | Pure AlphaZero-style net + MCTS, GPL-3.0 C++, actively developed (moves-left head added Oct 2025) | https://github.com/MaciejKozarzewski/AlphaGomoku |
| **KataGo / KataGomo** | KataGomo freestyle 2254 | KataGo (Go) adapted to gomoku/renju; the source of most of our already-mined levers | https://github.com/lightvector/KataGo , https://github.com/hzyhhzy/KataGomo |
| **junxiaosong AlphaZero_Gomoku** | — (teaching repo) | Vanilla AlphaZero; scalar value, root-only noise | https://github.com/junxiaosong/AlphaZero_Gomoku |
| **Xie & Fu (1809.10595)** | — (paper) | Curriculum board-size growth; otherwise vanilla | arXiv 1809.10595 |

**Already mined from these (do NOT re-propose):** PUCT + wave eval, Gumbel-root +
Sequential Halving, KataGo forced playouts + target pruning, playout-cap fraction,
KataGo **global-pooling residual blocks** (WON v5), KataGo **aux heads** (opp-reply
policy + per-cell ownership — middling in v4/v6), exact **VCF mate-teacher** (our
biggest win), **value-discount** (current champ lever), fixed-step trainer, EMA /
SWA self-play weights, random openings, temperature, dirichlet, recency buffer,
global opponent pool. The two teaching repos (junxiaosong, Xie-Fu) have nothing
untried.

## Headline finding

**AlphaGomoku uses a true WDL (win/draw/loss) value head as its PRIMARY value
representation** — not an aux head (`search/Value.hpp`: `struct Value { float
win_rate; float draw_rate; getExpectation(){return win_rate + 0.5*draw_rate;} }`;
MCTS edges store/average a `Value`). KataGo's value subhead is likewise a WDL
distribution (win/loss/no-result). Our value head is a **scalar tanh**. In our
**60-70%-draw** regime a scalar head squashes "solid draw" and "sharp 50/50
coin-flip" both to ~0.0 — exactly the distinction a WDL head preserves (LCZero's
WDL rationale, https://lczero.org/blog/2020/04/wdl-head/). This is the highest-
leverage untried lever and it **unlocks two cheap follow-ons** (draw-contempt, LCB
selection that needs a value distribution to be meaningful).

## Candidate levers (ranked; full per-idea detail)

### 1. WDL (win/draw/loss) categorical value head  — keystone
- **Source:** AlphaGomoku `Value.hpp`, `Edge.hpp updateValue`; KataGo value head
  (arXiv 1902.10565 §value head); LCZero WDL blog.
- **Hypothesis (drawish 9x9):** scalar tanh spends capacity regressing toward 0 in
  the draw-dominated middle and cannot separate dead-draw from coin-flip; a WDL head
  gives a confident `(0,1,0)` draw target and a `draw_rate` signal MCTS can act on →
  better-calibrated value, lower target variance where most games live.
- **Surface:** MODEL-ARCH + TRAINING-TARGET. Value head → 3 logits + softmax;
  loss → cross-entropy over {W,D,L}; z=+1→(1,0,0), 0→(0,1,0), −1→(0,0,1); MCTS backup
  carries a 3-vector, selection uses win+0.5·draw. Touches `model.py`, target-build in
  `train.py`/`self_play.py`, value plumbing in `mcts.py`. Composes with value-discount
  (discount W/L mass toward draw by γ^plies). **Code-heavy → bead.**
- **Cell sketch:** `--value-head wdl` (vs `scalar`); one lever, clone the champion.

### 2. Draw-contempt / draw-utility knob  — needs #1
- **Source:** KataGo `drawEquivalentWinsForWhite` (searchparams.h); AlphaGomoku's
  historical `style_factor` in `MaxValueSelector`.
- **Hypothesis:** a mild contempt (drawValue ~0.45) in self-play pushes the policy to
  keep probing for wins in equal positions → more decisive training data, separates
  siblings currently all "draw". Directly attacks drawishness.
- **Surface:** SEARCH, config-only **once #1 exists** (one float in Q + final
  selection).
- **Cell sketch:** `--draw-value 0.45` sweep {0.40,0.45,0.50,0.55} on a WDL champion.

### 3. LCB (lower-confidence-bound) root move selection
- **Source:** AlphaGomoku CHANGELOG 5.6.0 + `LCBSelector` (EdgeSelector.hpp); KataGo
  `useLcbForSelection`/`lcbStdevs`/`minVisitPropForLCB` (on by default, documented Elo).
- **Hypothesis:** with short noisy evals, max-visits can crown a move whose visits
  piled up before the value soured; LCB avoids the "looks busy, actually marginal"
  move — most valuable in balanced (our draw) positions where visits spread thin.
- **Surface:** SEARCH. Small; needs a per-edge value-stdev (KataGo uses running utility
  variance; cheap proxy 1/sqrt(visits)). Changes only root selection, not backup.
  **Code-heavy (MCTS) → bead.**
- **Cell sketch:** `--root-select lcb --lcb-stdevs 1.0` (vs max-visit).

### 4. Variance / uncertainty-scaled PUCT exploration
- **Source:** AlphaGomoku `PUCTvarianceSelector` (+ Thompson/KL-UCB/Bayes-UCB
  selectors); KataGo `KataGoMethods.md` cPUCT-scaled-by-utility-variance, `useUncertainty`.
- **Hypothesis:** most of a drawish tree is low-variance (everything equalizes) with a
  few sharp forcing lines that decide the game; variance-scaled cPUCT concentrates the
  cheap sims into the decisive lines instead of re-confirming dead-equal continuations →
  better sim-budget allocation for Δelo/wall.
- **Surface:** SEARCH. Medium; per-node running variance accumulator feeding cPUCT.
  Internal-only, no new head, byte-identical when scale=0. **Code-heavy → bead.**
- **Cell sketch:** `--puct-variance-scale 1.0` (0 = today).

### 5. Moves-left / plies-to-end head
- **Source:** AlphaGomoku `MovesLeftNetwork.hpp` (Oct 2025), `AGNetwork::unpackOutput`
  returns `movesLeft`; LCZero moves-left head.
- **Hypothesis:** decisive games are scarce and precious — convert won positions *fast*
  (don't dawdle a win into a draw), drag out lost ones; raises win-conversion and yields
  cleaner short decisive trajectories. Target (`plies_to_end`) is already computed for
  value-discount, so it's free.
- **Surface:** MODEL-ARCH (new head) + SEARCH (tie-break `Q ± ε·f(movesLeft)`).
  Medium-high. **Code-heavy → bead.** Lower priority (payoff concentrated in the
  minority of decisive games).
- **Cell sketch:** `--moves-left-head on --moves-left-weight 0.02` (off = byte-identical).

### 6. In-search VCF proven-score backup
- **Source:** AlphaGomoku `Edge.hpp` `Score`/`isProven()`/`ProvenValue`,
  `ThreatSpaceSearch.hpp`, `AlphaBetaSearch`; CHANGELOG 5.6.0 "Solver turned into
  alpha-beta search".
- **Hypothesis:** fold the exact solver INTO MCTS as proven W/L/D scores on edges
  (distinct from our *offline* VCF teacher that only rewrites targets): a proven node
  short-circuits selection and back-props a hard result, so MCTS never mis-weights a
  proven win as a soft 0.7, and a proven *draw* lets search stop wasting sims on dead
  lines (compounds with #4).
- **Surface:** SEARCH, expensive (most code of the six). We already have a VCF solver;
  the work is wiring proven results into per-edge scores + proven-aware selector/backup.
  **Code-heavy → bead.** Check overlap with the VCT-solver bead `derby-58f`.
- **Cell sketch:** `--in-search-vcf on --vcf-nodes 200` (off = today's pure-NN MCTS).

## Deferred / non-recommendations
- **Per-action (Q) WDL head** (AlphaGomoku `actionValues`): powerful selection prior
  but a big new head; only pays off after #1 — defer.
- **Squeeze-excitation / bottleneck residual blocks** (AG `blocks.hpp`), KataGo
  nested-bottleneck: pure-architecture levers adjacent to the already-won global-pooling;
  low priority vs the value-head family.
- **KataGo `policyOptimism`, `valueWeightExponent`/`useNoisePruning`, FPU**: real
  config-only-ish knobs worth a cheap sweep, but second-order next to the WDL family.

## Suggested derby order
#1 (WDL) first — keystone, unblocks #2 and #3, all three exploit drawishness directly.
#2/#3 are then near-free on the WDL champion. #4 is a self-contained search cell.
#5/#6 are new-code beads for the auto-factory once the WDL line proves out.

## Provenance
Researcher pass (Claude session `87a46d75-f155-4f81-a2db-769700ef5836`), web study of
the repos above + a background red-team review against our v1→v8 verdicts and the
derby-idea backlog (verdicts folded into the research-board "Open candidates" section).
