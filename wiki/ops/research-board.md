# Research Board — the Δelo Derby

A race between 8 fresh-start self-play training recipes ("ideas") to a fixed
**140-epoch budget**, run in **10-epoch chunks**, scored by the model's
**anchored elo** (vs `random`, `heuristic`, `lookahead:depth=2`, and the slow
`lookahead:depth=4` anchor). This is a research board that produces real models:
each idea gets a production-style title card, but the question it answers is
*which training recipe climbs fastest*.

## CURRENT (2026-05-27) — Derby v9 LIVE: the NET-CAPACITY axis

**v8 is CONCLUDED — the scalar `vcf + global-pool + value-discount 0.98` champion
("mate-discount", small 64f/4blk) decisively beat EVERY within-recipe lever** (WDL,
soft-policy, mish, gumbel-m8, dirichlet, defense, VCT, cross-game, recency, max-plies;
the full survey). Levers do not compound. So v9 changes the one axis left untouched:
**net capacity.** Same champion recipe on all three lanes; the *only* difference is
model size.

- **Live board:** `scripts/derby_v9_board.json` (base `sweep_runs/derby_v9`), 3 FRESH
  seed-0 lanes, identical champion recipe (gumbel-root + gumbel-m16 + vcf-teacher +
  value-discount 0.98 + global-pool + 64 sgd-steps):
  - `small` = `derby-v9-small` (64f / 4 blocks — the champion's net, fresh control).
  - `medium` = `derby-v9-medium` (96f / 6 blocks, ~2.3× params).
  - `large` = `derby-v9-large` (128f / 10 blocks, ~6× params).
- **North-star = Δelo/Δt:** a bigger net is *slower per epoch* (fewer epochs per 300s
  chunk) but may learn more per position — does the capacity win the **rate** race?
  Judge fresh lanes on climb-rate, not early anchored elo (which saturates ~1700 and
  diverges from H2H).
- **Cross-gen anchor:** the matured v8 small champion (`sweep_runs/derby_v8/_peaks/mate-discount/peak.pt`,
  anchored 1811) is preserved — the round-robin verdict also asks *"does a bigger net
  BEAT the old matured small champion?"* not just *"which size climbs fastest from
  scratch."*
- **Launched 2026-05-27** (Jason: "fresh v9, let's see what happens"); small lane up
  first, buffer filling clean. Same operating model as v8 (derby = sole GPU executor;
  300s Δelo-rate chunks; runner swaps by judgement; verdict = head-to-head round_robin).
- **⏳ JUDGMENT BAR (Jason, 2026-05-27): do NOT pass verdict before epoch 800+ AND a
  real plateau.** Judge on the **ceiling**, not the early rate. medium breaking through
  small's ceiling = capacity pays off; large failing to break through = large not worth
  investing in until medium is maxed out. "For now, let it cook." No premature
  round-robins, no swaps (all three are legit lanes climbing to their ceilings).
- **INTERIM H2H (23 chunks ≈ epochs 242–461, NOT a verdict — lanes far too young):**
  `round_robin_23chunks.json` partial — small≈medium (Δ+0.0), small≈large (Δ+14.5,
  ±135), so **no capacity edge at equal young training**; all three −137 vs the matured
  v8 champion = pure fresh-start lag (25–55 min vs ~22 h), not a recipe signal. The
  derby's anchored "large 1455 > small 1318" was the **anchored-vs-H2H mirage** —
  caught again. RR killed as premature; real verdict deferred to epoch 800+.
- **⚠️ wandb eval/model_elo is FROZEN for the large net (~epoch 45 / 788)** — the
  decoupled CPU eval-worker can't keep pace with the big net (small/medium track fine).
  Bead filed. Do NOT read large's strength from wandb; use derby_state.json (per-chunk)
  + the epoch-800 round-robin. New `scripts/wandb_workspace_v9.py` dashboard pins the 3
  lanes with this caveat baked into the panel titles.

## Derby v8 — CONCLUDED (2026-05-26 → 2026-05-27)

**The lab runs as a single GPU executor (the derby) + code-only beads (other sessions).**
The orchestrating session IS the "derby runner": it owns the GPU, runs the derby in
300s (5-min) chunks doled by Δelo-rate (peak-progress + patience), and **swaps
contestants in/out by judgement** (plateaued/starved lane → fresh cell; a climber is
never swapped; everything gets run). **Beads never run the GPU** — a bead = code-only
work for another session that lands a cell in `run_sweep.CELLS` "available for the
derby." Config-only levers (existing flags) skip beads — the runner just adds the cell
and races it. (Full model: `wiki/topics/research-loop.md`; memory
`project_derby_operating_model.md` is the resume index.)

- **Live board:** `scripts/derby_v8_board.json` (base `sweep_runs/derby_v8`), 4
  contestants on the **vcf + global-pool** base: `control`, `mate-discount`
  (`--value-discount 0.98`), `disc-recency` (`--value-discount 0.98` +
  `--buffer-recency-frac 0.5`), `buffer-comp` (`--buffer-recency-frac 0.5`).
  Pipelined eval on; cap = 4h backstop (not a hard 1h kill).
  - **SWAP (2026-05-27, commit `fa61aac`):** dropped `stack` (+`--max-plies 45`) —
    it regressed below its 1531 peak while its no-truncation twin mate-discount
    climbed to 1699; the 45-ply cap clips the 50–80-ply defensive games strong
    models play. Replaced with `disc-recency` to test whether the two *climbing*
    levers (value-discount + recency) compound.
- **v8 INTERIM H2H VERDICT (52 chunks ≈ 5h, `round_robin_52chunks.json`, 24g/pair):**
  mean-centered ratings — **mate-discount +115 🥇** (value-discount confirmed the
  key lever) · **buffer-comp +90 🥈** (recency curator is a genuine additive win over
  control) · disc-recency +16 (beats control, but **stacking value-discount+recency
  did NOT exceed either alone** — caveat: fresh seed-0 start, fewer chunks, so the
  combo is handicapped, not cleanly refuted) · control −81 (baseline) · **stack −139**
  (max-plies 45 actively harmful — swap vindicated). Takeaway: value-discount and
  recency are each real wins on the vcf+gp base; truncation hurts; the combo needs an
  equal-footing rematch before calling it.
- **v8 RR2 VERDICT (73 chunks, `round_robin_73chunks.json`):** the top THREE
  tightened into a cluster — **mate-discount +93** (0.98 value-discount, still #1) ·
  **disc-recency +82** (the combo — matured from +16 @52ch to +82, confirming it's
  competitive) · **buffer-comp +82** (recency). control −97; **vdisc-097 −160** (0.97).
  - **Sharper value-discount (0.97) is WORSE** than 0.98 by this peak — the discount
    optimum is at/above 0.98, not below. (Caveat below: vdisc-097's RR peak was its
    under-trained 1555; it later climbed to 1620, so −160 is partly stale.)
  - **METHOD CAVEAT — fresh-start H2H lag (recurring, now seen twice):** a lane that
    starts fresh seed-0 (`disc-recency`, `vdisc-097`) is *systematically undervalued*
    by round-robin until it matures — its saved `peak.pt` lags its live trajectory.
    disc-recency: +16→+82 across 21 chunks; vdisc-097: anchored 1555→1620 *after* its
    −160 RR. **Rule: never retire a still-climbing fresh lane on an H2H verdict; judge
    fresh lanes on climb-RATE, warm-resumed lanes on peak H2H.**
- **v8 RR3 — gomocup-survey levers (158 chunks, `round_robin_158chunks.json`):**
  **mate-discount +71 🥇** (champion still #1 — base recipe at maturity is hard to beat) ·
  **wdl +35 🥈** (WDL {w/d/l} value head — the strongest NEW lever, still maturing so
  possibly undervalued; the one to keep watching) · vdisc-099 +15 (0.99 anchor) ·
  gumbel-m8 −10 (m=16→8: neutral, no clear win) · **soft-policy −110** (soft-policy aux
  weight 0.15: clear dud, plateaued — retired for `derby-x-mish`). Interim (fresh lanes
  still maturing), but the ranking is clean: WDL is the promising survivor; soft-policy
  and gumbel-m8 don't beat the champion.
- **v8 RR4 — CLOSING verdict (174 chunks, `round_robin_174chunks.json`, all matured):**
  **mate-discount +80 🥇** (champion decisive #1) · **wdl +35 🥈** (WDL head — REPLICATED
  +35 across RR3 & RR4, the one reliable keeper, ~45 elo back) · vdisc-099 +22 · gumbel-m8
  −48 · **mish −89** (Mish activation — DUD; its fast anchored climb to 1634 was
  misleading, H2H it's last → another anchored-vs-H2H divergence, the C-cost-free
  nonlinearity doesn't help play strength). **v8 conclusion: of the whole gomocup-AZ
  survey, only the WDL value head survives as a carry-forward lever; gumbel-m≠16,
  Mish, soft-policy, max-plies, cross-game(perf-blocked) all rejected. Champion =
  vcf + global-pool + value-discount 0.98 (scalar), now anchored ~1811 and still
  climbing.** Next: stack the keeper — WDL + recency (config-only) racing as
  `derby-x-wdl-recency`.
- **v8 RR5 — POST-SURVEY verdict (215 chunks, `round_robin_215chunks.json`):**
  **mate-discount +154 🥇** (champion pulling AWAY — was +71/+80 in RR3/RR4, now +154 as
  it matures) · **wdl +56 🥈** (plain WDL still the lone #2) · wdl-recency −36 · wdl-max
  −41 · **defense −133**. **TWO conclusions: (1) stacking on WDL HURTS — wdl-recency and
  wdl-max (WDL+recency / WDL+deepVCF+recency) are both WORSE than plain WDL; the
  "aggressive combination" hypothesis is refuted (wdl-max's anchored 1649 was a mirage,
  H2H −41). (2) the scalar value-discount CHAMPION is decisively best and widening its
  lead — of the ENTIRE gomocup-AZ survey (WDL, gumbel-m, soft-policy, Mish, dirichlet,
  defense-teacher, +the perf-blocked crossgame/VCT), NOTHING beats it.** v8 research
  CONCLUSION: champion recipe `vcf + global-pool + value-discount 0.98 (scalar)` is the
  winner; WDL is the only additive lever but stacking degrades it and it trails the
  champion by ~100 elo. Decision point (Jason): v9 needs a NEW axis (bigger net / longer
  train / 15×15), not more single-lever sweeps; or wait for bounded crossgame/VCT
  (beads `derby-eda`/`derby-b6r`) to re-race the exact-solver family fairly.
- **v8 RR6 — VCT re-raced after the bound fix (245 chunks, `round_robin_vct245.json`):**
  mate-discount **+44 🥇** · vdisc-099 +25 · wdl +0 · **vct −69**. The now-bounded VCT
  (continuous-THREATS) teacher is ~110 elo BELOW the VCF champion — the "stronger" exact
  solver does NOT help (the aggressive depth-4/800 bound caps its extra power, and/or
  shallow non-forced-threat teaching adds noise). VCT rejected. (Caveat: vct csnp=3,
  semi-mature, so −69 may be mildly undervalued — but the gap is decisive.) **With VCT
  tested, the exact-solver family verdict is complete: VCF (champion) is the sweet spot;
  VCT over-reaches, crossgame (`derby-eda`) still perf-blocked. NOTHING in v8 beats the
  champion** — confirmed across the full survey + the two unblocked solvers.
  → confirm ONE `delo_derby` PID → `nohup bash scripts/derby_watchdog.sh
  scripts/derby_v8_board.json >/dev/null 2>&1 &`. A derby-runner **cron** (~30 min)
  drives swap/restock; the watchdog (startup-grace) keeps it alive.
- **Winner lineage** (rank by HEAD-TO-HEAD — anchored elo saturates ~1700, always
  `scripts/round_robin.py` over `_peaks/*/peak.pt`): vcf mate-teacher (v4) →
  **+global-pool** (v5 H2H win, compounds) → **+value-discount** (v6 H2H win, fixes the
  wins-anchored/loses-H2H overtraining gap). **sgd-steps sweep DEAD** (over-trains).
- **Backlog (beads, `derby-` prefix):** un-gated ideas = status `deferred` (hidden from
  `bd ready`); gate = `bd update <id> --status open`. Code-heavy epics: VCT solver
  (`derby-58f`, being built in a `gomoku-vct-solver` worktree), Rapfi opening-book
  (`derby-pyg`), reanalyze (`derby-3vs`).

## Open candidates — gomocup-AZ implementation survey (2026-05-27)

Researcher pass (session `87a46d75`) studying the AlphaZero-native gomocup engines
(**AlphaGomoku/MK** — Gomocup 2025 #2; **KataGo/KataGomo**) for levers we have NOT
raced. Full survey = [../sources/gomocup-az-techniques-2026-05-27.md](../sources/gomocup-az-techniques-2026-05-27.md);
intake mechanics = [../topics/derby-registration.md](../topics/derby-registration.md).
Six candidates were **red-teamed** (background reviewer) against the v1→v8 verdicts,
`TRAINING_WIKI.md`, the derby-idea backlog, and `model.py`/`mcts.py`. Verdicts:

| # | candidate | red-team verdict | why | status |
|---|---|---|---|---|
| 1 | **WDL (win/draw/loss) value head** | **PASS — #1 Δelo/wall bet** | only lever adding new *info capacity* (decisive-vs-drawn) vs our ~60-70%-draw data; purely TRAINING-side so it does NOT tax the Gumbel-SH generation hot path; aux-head precedent proves byte-identical-off is feasible | **BUILT + RACING** as `derby-x-wdl` (bead `derby-cgf` gated→built→raced by the factory in <20 min, commit 8100f87; scalar default byte-identical, 16 new tests). WDL ckpt can't warm-start a scalar champion → **FRESH-start lane** (judge on climb-RATE). Climb: 751→1567 peak. **✅ RR3 VERDICT (158ch, 72d0373): wdl +35 H2H — the STRONGEST NEW lever, still maturing (#1 bet VALIDATED). The drawish-fit hypothesis confirmed in head-to-head play. **✅✅ RR4 (174ch, 34c4b98): wdl +35 REPLICATED — "the one keeper"; of the WHOLE gomocup-AZ survey only the WDL head survives. The runner is now STACKING it (`derby-x-wdl-recency` = WDL + recency).** |
| — | Gumbel-`m` sweep (m=16→8) | red-team MISSED-idea: live flag frozen at v3 default, **never swept**, config-only/byte-identical | focus n=100 sims on fewer root candidates → sharper completed-Q targets | RACING as `derby-x-gumbel-m8`. RR3 −10 → **RR4 −48 = DUD** (m=16 was already optimal; the search-breadth tweak hurts). Ruled out. |
| 2 | draw-contempt (`drawValue`) | **KILL standalone** | a knob *on* the WDL head (follow-on sweep, not its own cell); also conflicts with the White "force-the-draw" objective (`derby-7ic`) → needs color-split eval, don't run blind | **WDL-sequenced follow-on family** (only if `derby-x-wdl` clears the field): draw-contempt (`--draw-value`); **per-action (Q) WDL head** (AlphaGomoku `actionValues` — a WDL Q per move as a selection prior, the last un-triaged AlphaGomoku value-axis lever; Class-C new head). Both wait on the WDL verdict. |
| 3 | LCB root move selection | **KILL** | written against visit-count selection; production is **Gumbel SH argmax over completed-Q** (`self_play.py:548`); no per-node variance accumulator (scalar `W` only); redundant with Gumbel's `sigma(q_hat)`; C-hot-path cost | rejected — recorded so it's not re-proposed |
| 4 | variance/uncertainty-scaled PUCT | **KILL** | no variance state + wrong engine (SH governs the root, not per-node cPUCT); closest analog v3 `forced` landed mid-tier *below* Gumbel; high C build cost, low expected Δelo | rejected |
| 5 | moves-left head | **KILL** | throughput goal already WON by adjudicate (`--max-plies 45`, +44 H2H v6); value-target half duplicated by the champion `--value-discount`; novel residue (search tie-break) is small + C-hot-path | rejected (revisit only if a delta-vs-value-discount is articulated) |
| 6 | in-search VCF proven-score backup | **KILL** | not a dup of `derby-58f` (different axis) but contradicts the explicit `derby-7ic` design ("RELABEL via teacher, NOT runtime alpha-beta"); a VCF solve per node on the generation hot path is the worst possible thing for Δelo/**hour** | rejected (a root-only cheap variant could be reconsidered later) |

**Round-2 candidates — from a deep read of actual KataGo + AlphaGomoku source (2026-05-27,
training-side, ZERO generation cost so Δelo/hr is protected):**

- **REGISTERED `derby-79l` → cell `derby-x-soft-policy`** (the pick): KataGo's **soft-policy
  auxiliary target** — a second policy-loss term against a 4th-root temperature-flattened copy
  of the *already-recorded* completed-Q `pi`, scaled by `--soft-policy-weight 0.15` (default
  0.0 = byte-identical). Under 60-70% draws the sharp target concentrates mass on 1-2 defensive
  moves and the net loses the search's runner-up structure; the soft target re-injects it (KataGo
  added it for exactly this under-taught-drawish reason). ~6 lines in `train.py:compute_loss`,
  orthogonal to value-discount (value head) + VCF (target). BUILT (7450019), RACED (210b105).
  **❌ RR3 VERDICT (158ch): soft-policy −110 H2H = CLEAR DUD, plateaued → retired for `derby-x-mish`.
  The policy-signal axis FAILED.** Climbed fast early (steepest fresh Δelo/hr) but couldn't beat
  the field at maturity — the fresh-start-climb-rate-is-not-a-verdict lesson, exactly why we wait
  for H2H.
- **Queued runners-up — DROPPED (sequencing discipline VINDICATED):** `derby-x-surprise-weight`
  (per-sample `1 + λ·KL(search_pi ‖ net_prior)`) and `derby-x-playout-weight` (SH visit-confidence)
  were *policy-signal-enrichment* levers correlated with soft-policy — **HELD pending its verdict,
  now DROPPED** since soft-policy was a −110 dud (the whole axis failed). Never filed → never flooded
  the board with correlated losers. This is the payoff of not firing correlated bets blind.
  - **🔁 RE-RED-TEAMED 2026-05-28 (loop tick, full reviewer pass) → KILL CONFIRMED, do NOT re-mine.**
    With the gomocup survey concluded and the v9 scale axis firming negative, training-side loss-weighting
    looked like the next un-mined lever (training-side = the WDL-keeper family that didn't tax the gen hot
    path). Two candidates, both **dead**: (b) **policy-surprise / KL target weighting is the EXACT dup** of
    `derby-x-surprise-weight` above (`1 + λ·KL(search_pi ‖ net_prior)`) — already dropped with the failed
    soft-policy axis. (a) **per-sample value-error / uncertainty weighting** is the value-axis sibling of the
    DOA "value-error head" (line 212) and would be a **second lever fighting `--value-discount 0.98`** (itself
    a per-sample target reshaping the derby already optimized); up-weighting high-value-error samples chases
    the noisiest draw/coin-flip positions = amplifies label noise in the 60–70%-draw regime. **No real prior
    art:** KataGo's per-sample weights are *data-curation* (playout-cap-fraction, already our `--playout-cap-frac`,
    v3 floor-stuck) + *search-side* (`utilityVariance`/`valueWeightExponent` in the tree), **not** a loss-magnitude
    SGD multiplier. The board's own pointer holds: the binding gap (convert lookahead4 black-DRAWS to wins) is a
    **decisiveness / search-and-target-sharpness** problem, not a loss-magnitude one — every "enrich the existing
    loss signal" lever (soft-policy, surprise-weight, playout-weight, recency-on-WDL) has lost. Injection point
    if ever revisited: `train_step` weighted-mean at `gomoku/train.py:157,181` (byte-identical when `w≡1.0`).
**Round-3 — ARCHITECTURE axis (deep source read of AlphaGomoku/KataGo net code, 2026-05-27):**

- **REGISTERED `derby-sib` → cell `derby-x-mish`** (the arch pick): swap ReLU→**Mish** activation
  (`--activation mish`, default relu = byte-identical). KataGo's `act()` factory offers it;
  zero added params, identical state_dict keys, **model.py-only**. *Load-bearing infra fact
  (verified): our native-C MCTS engine does NOT compute the forward — it does tree ops and calls
  back to a PyTorch `evaluate_planes` evaluator (`_mcts_native.c` has no conv/relu). So an
  activation swap needs NO native-C kernel* (correcting the initial worry). Fresh-start (ReLU-
  trained weights misbehave under Mish), judge on climb-rate. BUILT (793a86a), RACED (72d0373).
  **❌ RR4 VERDICT (174ch): mish −89 H2H = DUD, retired. The fast anchored climb (Δelo/hr 6124 at
  10min) MISLED — it finished H2H last. Third confirmation that fresh-start climb-rate ≠ verdict
  (soft-policy and mish both climbed fast then lost H2H). The activation axis is a wash for us.**
- **SE (squeeze-excitation) blocks — DEAD (do not file):** was gated on "file only if Mish clears
  the field." **Mish was a −89 dud → the activation axis failed → SE is dropped.** (Was already
  deprioritized: fresh-start + unfused per-block latency tax on our tiny 4×64 net.)
- **Nested-bottleneck / ConvNext depthwise — DOA (do NOT file):** KataGo's nested-bottleneck is a
  width/depth-amortization trick for deep wide trunks (b18c384); at 4×64 it has nothing to amortize
  and shrinks effective width, and it's multi-knob. ConvNext depthwise-7×7 is multi-knob + MPS-
  unfriendly. Wrong tools for our net size.

- **Multi-knob future study (NOT derby-shaped):** our optimizer is bare `AdamW(lr)` with no
  scheduler/warmup/grad-clip; both upstreams use SGD-momentum + LR warmup/decay + clip — real
  headroom but violates one-lever-per-cell, so it's a deliberate later study, not a single cell.
- **Confirmed dead-on-arrival (do NOT file):** soft *value* / TD-value / learned shortterm-
  value-error head (needs recorded bootstrap targets = pipeline change); extra symmetry aug
  (D4 = the full gomoku symmetry group, we're already complete); a score/margin head (degenerate
  for win/draw/loss).

**Loop status (2026-05-27, cron `4e4dcc03` — tick 31: v8 RESEARCH ARC CONCLUDED).** Full arc: gomocup-AZ
single-lever survey → aggressive WDL-stacking → blocked-bead loss-tail epic. **Final verdict (RR5 @215):
the scalar vcf+global-pool+value-discount champion (`mate-discount`, +154 and widening) beats everything
the survey produced** — WDL is the lone positive new lever (+56) but never the champion, and stacking
recency/vcf-deep on WDL HURTS (−36/−41). Plus the blocked-bead loss-tail epic shipped (gi7 + 1xf defense
arm racing; VCT attack arm bounded, awaiting the runner's GPU smoke). **The deliverable research is done;
the next axis (v9: bigger net / longer / 15×15) is a Jason call, not a within-v8 lever.** Detail of the
single-lever survey conclusion (RR3/RR4) follows; aggressive-combo + blocked-bead sections above.

Two H2H verdicts (RR3 @158, RR4 @174) closed the single-lever survey cleanly:

| lever (axis) | H2H verdict | outcome |
|---|---|---|
| **WDL (value-rep)** | RR3 **+35** → RR4 **+35 replicated** | ✅ **THE KEEPER** — sole survivor of the whole survey; runner is stacking it (`derby-x-wdl-recency`) |
| soft-policy (policy-signal) | −110 | ❌ dud, retired |
| mish (activation) | −89 | ❌ dud, retired (fast climb misled) |
| gumbel-m8 (search) | −10 → −48 | ❌ wash → dud |

**Result: of the entire gomocup-AZ survey (4 single-lever cells + the architecture & defense
axes), exactly ONE lever — the WDL win/draw/loss value head — is a net-positive keeper, and it
replicated.** That's precisely the cell I ranked #1 and the red-team flagged as the top Δelo/wall
bet → strong validation of the propose→red-team→race process. The duds were dropped without
flooding (sequencing discipline: soft-policy/mish follow-ons HELD then dropped; SE killed when Mish
failed). **Forward:** WDL is carried into stacking by the runner (`wdl-recency` = WDL + recency);
the WDL-native follow-ons (draw-contempt, per-action-Q) stay gated (White-objective conflict /
Class-C); the **defense teacher `derby-1xf`** is the highest-value future swing (blocked upstream).
Researcher monitors only — the derby-runner owns the GPU swaps. North star = **Δelo/wall**.

**AGGRESSIVE COMBINATION phase (tick 19, 2026-05-27, Jason: "try aggressive moves that combine
the best we found so far into new runs").** v4-style multi-lever stacks (deliberately dropping
one-lever-per-cell, as v4 did). Base = champion (vcf+global-pool+value-discount) + the survey
keeper WDL; stacked with the OTHER validated winners — buffer-recency (v8 buffer-comp +90) and
vcf-deep (v5 deeper exact-mate solver +44). A clean 2×2 over {recency, vcf-deep} on the WDL base:

| cell | = WDL + … | status |
|---|---|---|
| `derby-x-wdl` | (alone) | best new lever — **RR5 H2H +56** (lone #2, improved from +35) |
| `derby-x-wdl-recency` | + recency | **RR5 H2H −36** — below plain WDL ⇒ recency does NOT compound on WDL |
| `derby-x-wdl-max` | + recency + vcf-deep (maximal) | **RR5 H2H −41** — below plain WDL (anchored 1649 was a mirage) |
| `derby-x-wdl-deep` | + vcf-deep | (never needed a lane — the stacking hypothesis was refuted first) |

**❌ AGGRESSIVE-COMBO VERDICT (RR5, 215 chunks, `round_robin_215chunks.json`, commit `a3bad90`):
stacking on WDL HURTS — refuted.** Both stacks (`wdl-recency` −36, `wdl-max` −41) land *below* plain
WDL (+56); the maximal stack's anchored 1649 was a mirage (H2H −41 — the v4/v5 anchored-vs-H2H lesson,
again). And **`mate-discount` (the scalar vcf+global-pool+value-discount champion) is +154 and
WIDENING** — decisively best. **Bottom line: NOTHING in the entire gomocup-AZ survey, even aggressively
combined, beats the scalar value-discount champion.** WDL is a real positive lever (+56) but second to
the champion; the levers do not compound. (adjudicate was excluded — v8-regressed.) **v8 research is
CONCLUDED.** The next axis (v9: bigger net / longer training / 15×15) is a **Jason decision**, not a
within-v8 lever.

### Round-4 — DEFENSE axis: maps to EXISTING `derby-1xf` (no duplicate filed)

The gomocup-AZ survey's deepest pull is the wiki's **central failure mode**: the net learns
attack, not defense. None of the four v8 cells (value-rep / policy-signal / search / activation)
target it. The natural lever — a **defensive threat-block teacher** (mirror the VCF mate-teacher:
when the *opponent* has a proven forced win, relabel) — turns out to **already be filed as
`derby-1xf`** (P1, BLOCKED, the "novel half" of the loss-tail epic `derby-7ic`). Dedup discipline
caught it; no duplicate filed. A code-grounded validation pass adds three things the bead
under-specs, recorded here for whoever builds it:

1. **gomocup-AZ confirmation:** this IS the canonical strong-engine approach — AlphaGomoku/Rapfi
   run *two-sided* threat-space search (attack AND defense). Our teacher is one-sided (offense only).
   So `derby-1xf` is well-motivated by the external engines, not just our loss-tail theory.
2. **The Δelo/wall gen-cost GATE the bead under-specs (load-bearing):** naive value-only defense =
   a *second* full solve per position (opponent's perspective) ≈ ~2× the current VCF-teacher solver
   cost on the gen hot path — a real Δelo/hr risk. Cheapest sound formulation: **(a) skip the
   defensive solve when our offensive solve already fired** (we have a proven win → `z`=+1 already,
   no need to check the opponent), and **(b) gate it behind a near-free danger pre-scan**
   (`vcf._has_immediate_five` / a `baselines` window-count threat scan) so quiet positions cost ZERO
   solver calls. Reuse `solve_vcf` with **swapped planes** (`board = stack([planes[HISTORY_PLY],
   planes[0]])`, vcf.py:343-347) — no new solver code for value mode. **Measure the current
   VCF-teacher cost first** (the `vcf_solve_s`/`vcf_calls` profile counters in `_apply_vcf_teacher`
   exist but are unlogged — close that gap before racing).
3. **value-only first, policy-stamp deferred (the bead already lands here):** defense is genuinely
   non-unique (multiple blocks / non-block refutations), so a one-hot *policy* stamp is ill-defined;
   the *value* relabel (`z=-1` on a proven opponent win) is unambiguous and shippable. Correctness =
   the same 400-fuzz-vs-referee gate as VCF (a false positive poisons value the wrong way).
   **Correlated with VCF (same solver/seam) → test STACKED on the champion, not standalone**; depends
   on the color-split eval (`derby-gi7`) to measure the white-side loss-rate target.

**SHIPPING NOW (2026-05-27, Jason: "investigate blocked beads… review for staleness and ship it
— I set those to blocked because they were half-formed, not bad"):** the loss-tail epic's two
key beads are being de-staled + shipped:
- ✅ **`derby-gi7` (color-split eval) SHIPPED** — per-color (black/white) W/D/L in `eval_results.jsonl`,
  backward-compatible, 4 new tests + 16 eval tests green, CPU-only (merged `dc552c1`, branch
  `feat/color-split-eval`, closed). This is the epic's *acceptance instrument* (measures white-side
  loss-rate) → it **unblocked `derby-1xf`**.
- ✅ **`derby-1xf` (defensive teacher) SHIPPED** — value-only `--defense-teacher`: relabel z=−1 when the
  OPPONENT has a proven VCF win (swapped-plane `solve_vcf` reuse, no new solver). **Gen-cost gate**
  (skip-when-offense-fired + cheap `has_four_threat` pre-scan → quiet positions cost ZERO solver
  recursion) and **400-fuzz vs independent referee = ZERO false positives** (6 real lines verified,
  non-vacuous — the gate that matters, since a false positive poisons value targets). Byte-identical
  when off (verified: monkeypatch→call_count 0). Cell **`derby-x-defense`** (champion + `--defense-teacher`).
  Merged `22d8a82` (branch `feat/defense-teacher`), closed. Policy/refutation mode deferred (defense
  non-unique). ⚠ Live gen-cost under flooding to be confirmed by the runner's full-load epoch-50+
  re-race (the ingest-perf-flooding lesson).
  - **COLOR-SPLIT VERDICT (the shipped `gi7` instrument, used as intended — and it reframes the epic):**
    | vs | defense lane (elo 1356, fresh) | champion (elo 1665, mature) |
    |---|---|---|
    | heuristic | B 10-0-0 / **W 3-7-0** | B 10-0-0 / **W 10-0-0** |
    | lookahead2 | B 10-0-0 / W 0-0-10 | B 10-0-0 / **W 10-0-0** |
    | lookahead4 | B 3-1-6 / W 2-1-7 | B 4-0-6 / **W 2-0-8 (0 losses)** |

    **The MATURE champion has essentially NO white-side loss tail vs the in-repo baselines** (perfect
    white vs heuristic/lookahead2, zero white losses vs lookahead4). The defense lane's white losses are
    mostly its **~300-elo strength gap** (fresh vs mature), not a structural defense-blindness the
    champion shares. **So the loss-tail premise — "losses cluster on white" — largely does NOT hold for
    the mature value-discount champion vs these baselines**: it already defends white cleanly. This
    *vindicates retiring the defense lane* (RR5 −133 + no white-tail-to-fix) AND says the defense teacher
    targets a problem the champion has mostly solved. CAVEAT: the comparison is at different maturities,
    and the in-repo baselines may be too weak to expose a *residual* white tail in the mature champion —
    so the defense teacher's value, if any, would only show vs a STRONGER opponent. **Conditional
    follow-on: a Rapfi color-split eval** (champion vs Rapfi, per-color) — if even the champion leaks
    white losses vs Rapfi, the defense teacher earns a re-race; if not, the loss-tail epic is effectively
    closed by value-discount and the defense teacher is a solution without a problem (vs available opponents).

**Loss-tail epic `derby-7ic` — DEFENSE arm DELIVERED** (gi7 + 1xf shipped; the wiki's *central* problem
— never-lose-as-white — now has both its measurement instrument and its exact-relabel teacher).

**ATTACK arm (`derby-58f` VCT teacher) — code MERGED, race-readiness GATED on a gen-cost fix:**
- ✅ Wired `--vct-teacher` (`derby-rxf`, closed) using the already-built `solve_vct`; mirror of the VCF
  teacher, VCT replaces VCF (superset: +23 threes-only wins over VCF's 16, verified). **400-fuzz =
  ZERO false positives** (`derby-y8r`, closed). Byte-identical-off. Merged `896ced4`.
- ⚠️ **Gen-cost: the ungated VCT gen-STARVED** — the runner raced `derby-x-vct`, got buf=0/games=0
  (one wide-open position = ~5700 nodes / ~15s pure-Python at the loose depth-7/20k defaults; 8 workers
  each blocked per move → generation fully starved). Pulled it, filed bug `derby-b6r`. Confirms the
  in-search-VCF failure mode (exact-solve-on-the-gen-hot-path tanks Δelo/hr).
- ✅ **FIXED (`derby-b6r`, merged `a94b156`/`e8cee60`, branch `feat/vct-teacher-bound`):** aggressive
  per-move teacher cap (`_VCT_TEACHER_MAX_DEPTH=4`/`_VCT_TEACHER_MAX_NODES=800`; cell carries
  `--vct-max-depth 4 --vct-max-nodes 800`). CPU-proven: the wide-open position that took 5722 nodes/14.6s
  at 7/20k now bails in 131 nodes/320ms (~45×); short VCT wins still proven; byte-identical-off; 49 tests pass.
- ✅ **`derby-b6r` SMOKE-VERIFIED + CLOSED, then RACED:** runner smoked the bounded `derby-x-vct` (buf
  76k+, ~2.6s epochs — gen-starvation gone), closed b6r, raced it as a real A/B vs the VCF champion.
- ❌ **RR6 VERDICT (245 chunks, `round_robin_vct245.json`): VCT REJECTED — `vct −69` vs champion `+44`**
  (~110 elo below; `vdisc-099 +25`, `wdl +0`). The deeper continuous-threes teacher does NOT beat the
  continuous-fours (VCF) champion — the aggressive bound caps it and/or shallow threat-teaching adds noise.
  **EXACT-SOLVER TEACHER FAMILY COMPLETE: VCF is the sweet spot.** Neither the deeper offensive teacher
  (VCT) nor the defensive teacher (1xf) beats it. The whole loss-tail / exact-solver axis is exhausted at
  this scale — VCF mate-teaching is the keeper, full stop. (`derby-58f`/`derby-6us` closeable.)
- *(My parallel board-fill gate `feat/vct-gencost-gate` was HALTED as redundant once the node-bound
  merged — kept only as a conditional QUALITY follow-on if the aggressive cap proves to weaken VCT's
  deep-win detection too much. **Its profiling is preserved as ready-to-use evidence:** empirical
  threshold = **32 stones** (VCT/VCF cost ratio 24st→32×, 28st→13×, 32st→2.3×, 45+st→1×), and the
  gated cost re-measured at **1.12× the VCF baseline** over a realistic full-game distribution — so if
  the node-bound underperforms, the board-fill gate at 32 stones is a drop-in quality refinement.
  Honest lesson stands: a deeper exact solver is only a teacher lever if its per-position cost is
  bounded — the offensive analogue of the in-search-VCF kill.)

### Toward v9 — the gomocup-AZ STACKING thesis (planned ahead, gated on v8 verdicts)

The derby's winning pattern is **stacking orthogonal winners** (v5 stacked global-pool on vcf;
v6 added value-discount → the current champion). The four gomocup-AZ cells were deliberately
chosen on **four orthogonal axes** so they can compound rather than overlap:

| axis | cell | touches |
|---|---|---|
| value representation | `derby-x-wdl` | value head (3-way W/D/L) |
| policy signal | `derby-x-soft-policy` | policy-loss aux term |
| search breadth | `derby-x-gumbel-m8` | Gumbel-SH root width |
| activation | `derby-x-mish` | tower nonlinearity |

**v9 thesis:** whichever cells **beat the champion head-to-head** stack incrementally onto the
champion lineage, ONE added lever per cell (like v5→v6), exploiting the orthogonality. **Do NOT
pre-stack before the single-lever verdicts** — the v4 lesson (anchored leads were undertraining
artifacts; H2H reshuffled the order).

**UPDATED after RR3 (158ch) — the axes have spoken (interim, fresh lanes still maturing):**
- ✅ **value-rep (WDL) = the WINNER** (+35 H2H, strongest new lever, still maturing). `derby-x-wdl`
  IS already `champion + WDL` — so the first v9 stack is *on the board* and just needs to mature
  past the champion (+71). This is the lineage candidate to carry forward.
- ❌ **policy-signal (soft-policy) = OUT** (−110 dud). Its follow-ons dropped. Do not stack.
- ⚪ **search-breadth (gumbel-m8) = neutral** (−10) — not a stack ingredient.
- ⏳ **activation (Mish) = TBD** (just swapped in). If Mish beats the champion H2H, `WDL + Mish`
  is the clean next stack (both are independent of each other — value head vs nonlinearity).
- 🔬 **defense axis (`derby-1xf`)** remains the highest-value *future* lever (hits the core
  problem) but is blocked + correlated with VCF.

So the narrowed v9 lead = **carry WDL forward** (the one validated new lever), watch Mish for a
`WDL + Mish` stack, and treat defense (`derby-1xf`) as the big swing once unblocked. Stack only
verified H2H winners.

*(Loop operational gotcha, for future researcher-loop sessions: `bd create`/`bd update` stage
`.beads/issues.jsonl` on `main`, which blocks a `feat→main` merge with "local changes would be
overwritten" — always `git commit -- .beads/issues.jsonl` FIRST, then merge.)*

## v9 DIRECTION — SCALE-UP to 100% (Jason-GATED, 2026-05-27): bigger models, KEEP the evals, target 100%

**Goal (Jason):** do our best **at 9×9** — work out the process + recipes so the eventual 15×15 is an
experiment worth doing, not a wall-clock-burner. Stay at 9×9 until the recipe is *right*.

**CORRECTION to the v8 "evals saturated" framing (Jason, and he's right):** the evals are NOT saturated —
**anchored *elo* is** (it scores a draw 0.5, so a model that *draws* the hard games maxes elo without
*winning* them). The true objective is **100% across all three in-repo evals**, defined as **"always wins
as BLACK and never loses as WHITE."** By that bar the champion is NOT done:
- vs heuristic ✅ (B 10-0-0 / W 10-0-0) · vs lookahead2 ✅ (B 10-0-0 / W 10-0-0)
- **vs lookahead4 ✗ — B 4-0-6 (four wins, SIX DRAWS) / W 2-0-8.** White already meets "never lose"; the
  *entire* remaining gap is **converting the lookahead4 black-DRAWS into wins.**

**Why this re-opens the research (the load-bearing insight):** anchored elo is *why* v8 read as "nothing
moves the needle" — it saturates the instant a recipe can draw the hard games, so every near-ceiling recipe
looks tied. The ruler is too short, not the recipes equivalent. **The derby's success metric should track
distance-to-100% — specifically lookahead4 black-win-rate + white-non-loss (from the shipped color-split
eval) — not anchored elo.** That makes the research productive again.

**v9 plan (gated):**
1. **Bigger models** — scale `small` (64×4, ~0.8M) → `medium` (96×6) / `large` (128×10), toward
   AlphaGomoku's ~128×8. The small net very plausibly lacks the capacity to convert lookahead4 draws→wins,
   and it's likely *why* WDL/global-pool washed out (no spare capacity) — so **re-test the top gomocup-AZ
   levers (WDL, global-pool) at scale** (the v8 "stacking hurts" may be a small-net capacity artifact).
2. **KEEP the in-repo evals** (heuristic, lookahead2, lookahead4) — do NOT bail to Rapfi yet. Target =
   **100% (win-all-black, lose-none-white) across all three.** Rapfi becomes the *next* yardstick only once
   the in-repo 100% is hit (guards against over-fitting the fixed bots; until then the in-repo target has
   real, unmet headroom).
3. **Raise eval games-per-baseline** (esp. vs lookahead4) so "100% / never-loses" is statistically real,
   not a lucky 0-loss-in-10.
4. Target is **well-posed + reachable**: 9×9 freestyle is a first-player (black) win with strong play, and
   the second player can always draw — so "win-all-black, never-lose-white" is achievable, not asymptotic.

15×15 stays the *eventual* target, attempted only after the 9×9 recipe hits (or near-hits) 100%.

**🏁 LAUNCHED (2026-05-27, commit `3e02298`):** the derby transitioned off v8 to a dedicated
**`scripts/derby_v9_board.json`** — the live `delo_derby` PID now races the **net-capacity scale-ladder**:
`derby-v9-small` (64×4, fresh baseline rung the runner added) / `derby-v9-medium` (96×6) / `derby-v9-large`
(128×10) — champion recipe (vcf + global-pool + value-discount 0.98 + gumbel-m16 + sgd-64), only the net
`size` differs, all fresh-start for a clean A/B/C. **The v9 question is live:** does a bigger net convert
the lookahead4 BLACK-DRAWS into WINS (the binding gap to 100%)? Judge by **lookahead4 black-win-rate**, not
anchored elo.
- ⚠️ **READ AT MATURITY, not early (a premature-retirement guard for the runner):** bigger nets WARM UP
  SLOWER — more params to fit — so the scale-ladder's early standings show **small AHEAD** and medium/large
  trailing (5-chunk read 2026-05-27: small 893 / medium 751 / large 603). That is *expected warm-up lag*,
  NOT "scale doesn't help." Any capacity advantage of medium/large appears only once small plateaus and the
  bigger nets are still climbing. **Do NOT retire medium/large on early elo** (the net-scaling analogue of
  the fresh-start-H2H-lag rule). The verdict is the mature lookahead4-black-win-rate, after many chunks.
  - ✅ **GUARD VINDICATED + first scale signal (13 chunks, 2026-05-27 20:09Z):** the crossover happened —
    **`medium` 1318 OVERTOOK `small` 1253**, `large` 1187 climbing behind. The slow-warming bigger net
    ripened and passed small exactly as predicted; had it been retired on the 5-chunk read it'd have been a
    mistake. **Early positive sign that scale helps** (anchored ordering flipped medium > small) — but the
    real verdict is still the mature lookahead4-black-win-rate (does the bigger net WIN the games small DRAWS?).
  - 📊 **`small` PARKED + a clearer scale hint (52 chunks, commit c0ebe69):** the runner parked `small`
    (it plateaued at the small-net ceiling ~**1497**, met the 800+ bar ~2×, and was starving the bigger
    nets) — compute now flows to `medium`/`large`. **`large`'s peak (1580) already exceeds `small`'s
    ceiling (1497)** → suggestive that a bigger net raises the ceiling. STILL not a clean verdict (large
    fed more wall; anchored elo is noisy + saturating; the 100%-read can't be trusted until the eval is
    deepened per `derby-563`) — but the direction (bigger net > small's plateau) is the first real scale hint.
  - ↩️ **COURSE-CORRECTION (88 chunks, 2026-05-28 00:33Z) — the scale hint WEAKENED, now lukewarm/negative:**
    with heavy feeding, **`medium` (96×6) peaked only 1356 at 202 min wall — BELOW `small`'s parked 1555**
    despite ~2× the compute; `large` (128×10) is 1580 at 170 min — only marginally above small's *parked*
    peak and still below small's *true* potential (~1738). So bigger nets are **slower and not clearly
    exceeding small's known level** — the early anchored read is NOT pro-scale. NOT a verdict (anchored elo
    noisy; bigger nets genuinely slow to mature; the 100% metric still blocked on `derby-563`), but I'm
    explicitly retracting "scale helps" — the evidence so far leans the other way. Watch for medium/large to
    either climb past small's ~1738 with more maturation, or confirm scale doesn't pay at 9×9.
    - **Firming (93 chunks, 01:05Z): `medium` reached its 4h CAP (232/240 min) at peak 1356 — effectively
      final, and clearly below `small`'s parked 1555.** So on anchored elo, the 96×6 net does NOT beat the
      64×4 net even at-cap. `large` (128×10, 1580, not yet capped) is the last bigger-net hope but trending
      the same way (below small's ~1738 potential). CONFOUND: the run hit a **v9 wandb crash-loop** (runner
      hardened it, commits 0c83693/e1039da/91a7b93) which disrupted maturation. Still anchored-only + small
      under-trained — but the scale-doesn't-pay-at-9×9 read keeps firming.
    - **`large` (128×10) ERRORED — trainer CRASHED (97 chunks, 01:28Z):** milestones log — "no epoch
      progress (trainer epoch 628→628 in 9s) — trainer likely crashed"; derby retried once, errored it,
      tore down the lane (`large/chunk.log`: "trainer exited"). So the BIGGEST net isn't just slow — **it's
      unstable here** (crash, likely OOM / numerical at 128×10 on this box). Combined with `medium`
      underperforming `small` at-cap, the scale axis now reads doubly bad: 96×6 < 64×4, and 128×10 won't
      train stably. (Derby itself HEALTHY — `delo_derby`+watchdog up, 9 workers; a transient `pgrep`=0 mid-
      cycle briefly looked like a halt but was a false alarm, re-verified.) `large`'s teardown frees a lane —
      `derby-x-medium-signal` (the aux capacity-unlock probe) is the natural swap-in.
    - **(researcher obs, 2026-05-28 01:45Z, 99 chunks) `medium` cooked 220 more epochs past the raised cap
      (796→1016) and did NOT raise its ceiling** — `derby_state.json` peak still **1356 @ epoch 796**, now
      oscillating ~1280 (last_delo +37). So the plateau is well-confirmed across the extra training: the
      scale-doesn't-pay-at-9×9 read keeps firming on anchored elo. (Still the **ranking owner's verdict to
      call**, and anchored elo saturates / isn't strength — the real verdict remains the matured H2H
      round-robin including the v8 cross-gen anchor. Derby HEALTHY: `delo_derby` 58041 + watchdog 58155 +
      medium trainer + 8 workers; `medium-signal` landed but not yet swapped into the freed lane.)
  - 🔎 **IS `small` AT STANDARD? (asked 2026-05-28) — roughly yes, within noise, but PARKED EARLY + the
    "scale hint" is fragile.** `small` is byte-identical to the v8 champion (`mate-discount`) at the small
    net. At MATCHED training (~36 evals = small's parking point): small peaked **1555**, the champion peaked
    **1649** — small ~94 below, but the champion's own anchored elo SWINGS 1405→1649→1738→1434 eval-to-eval,
    so that gap is inside the noise. The champion's "1738" is a noisy TRANSIENT over 117 evals (3× small's
    training), NOT a stable bar. **Two consequences:** (1) `small` (36 evals) is UNDER-TRAINED — 1555 isn't
    its ceiling, and its true potential (champion ~1738 transient) likely sits ABOVE `large`'s current 1580,
    so the "large beats small" hint is **not safe** — small was parked before reaching its known level;
    (2) anchored elo is so noisy (champion alone spans 1405-1738) it CANNOT adjudicate small vs the scale
    ladder. Both point to the same fix (`derby-563`: the 100% metric + more games + a fair, equal-training
    comparison). Re-feeding `small` to its real ceiling for a fair baseline is the ranking owner's call.
  - ⚠️ **EVAL IS TOO NOISY TO READ THE 100% TARGET (20 chunks, direct color-split read):** at 10 games/color
    the lookahead4-black-win signal is noise-dominated — e.g. `derby-v9-small` showed L4-black **9-0-1** yet
    heuristic-black **2-0-8** (internally contradictory: can't be near-perfect vs lookahead4 and weak vs the
    *easier* heuristic — that's small-sample noise), and the three lanes are at mismatched maturities (`large`
    still at elo ~788, loses everything = warm-up). **So the metric bead's "raise eval games-per-baseline" is
    ESSENTIAL, not optional** — without ~50-100 games/color, "always-wins-black / never-loses-white" can't be
    distinguished from a lucky 10-game sample. The scale verdict needs mature, comparable-strength, larger-
    sample reads; the current evals can't support it.
  - 🔧 **INSTRUMENT BUILT: `scripts/report_100pct.py`** (read-only, no GPU) — pools the last K color-split
    eval rows per lane and prints per-baseline black-win% / white-loss% + an aggregate **distance-to-100%**
    (0.0 = win-all-black/lose-none-white). It's the concrete `derby-7ku` metric the runner can adopt.
    First pooled-6 read (2026-05-27 21:25Z): small **1.45** (elo 1243-1555) · medium **3.40** (711-1347) ·
    large **5.64** (339-788). **CAVEAT — this tracks MATURITY, not net size:** the lanes are at wildly
    different elos (large barely warmed, ~339-788), so the ordering just reflects who's more mature, NOT a
    scale verdict. Reading scale needs a **same-maturity** comparison (each net at ~equal elo, e.g. ~1400),
    which `large` hasn't reached. Also: **no lane is near 100%** (even mature small wins only 25% as black
    vs lookahead2) — the target is a long climb — and a lookahead2(25%)-vs-lookahead4(60%) black-win
    INVERSION persists even at pooled-60-games, reconfirming the eval-games increase is needed.
  - 🚧 **ROOT BLOCKER for reading v9 — the color-split eval LAGS training (the reporter exposed it):**
    `derby-v9-large` has only **5 color-split evals, all at elo 339-788, while training has reached ~1455**
    (peak). The full vs-baselines color-split eval runs FAR more sparsely than training — the recent
    `derby-i5j` change put the *frequent* elo readout on per-chunk wandb, but the color-split (the only
    source of the 100% metric) didn't follow. So `large`'s real 100%-distance at its current strength is
    simply **unmeasured**, and no amount of pooling fixes a lane with zero recent color-split rows.
    **So `derby-7ku` is really THREE things:** (1) rank by distance-to-100%, (2) raise games-per-color
    (~50-100), AND (3) **run the color-split eval at CURRENT strength / often enough** (a current-checkpoint
    color-split on each matured lane). Until (3), the scale verdict is unreadable regardless of maturity.
    [GPU/eval = runner's; researcher can't run it.]
- **The 100%-metric — a RESEARCHER SUBMISSION, not a researcher ranking decision (corrected 2026-05-27):**
  the researcher SUBMITS the read-only tool `scripts/report_100pct.py` (computes distance-to-100% from the
  shipped color-split) + the evidence (anchored elo saturates; eval noisy at 10 games/color + lags training)
  — and lets the **ranking owner (derby-runner) decide** whether/how to weight it + the eval games/cadence.
  Filed as the SUBMISSION bead **`derby-563`** (`for-ranking-owner`, not `derby-idea`). The earlier bead
  `derby-7ku` ("rank/allocate by distance-to-100%") was **closed as WRONG SCOPE** — it made a *ranking
  decision*, which is the ranking owner's call, not the researcher's. (Rule now in the `gomoku-derby-register`
  skill: submit research + read-only tools; never file a bead that rewires the success metric / `pick_priority`.)
- **Follow-on (gated on the scale result):** if a bigger net helps, RE-TEST WDL / global-pool *at that
  scale* (the v8 "levers wash out" may be a small-net capacity artifact). Don't pre-register before the
  scale signal.
- **REGISTERED — capacity-unlock probe `derby-x-medium-signal` (2026-05-28, Jason: "beef up medium —
  activate an old lever that didn't do much on its own but might compound where small didn't have the
  juice"):** = `derby-v9-medium` (champion @96×6) + the v4 KataGo **aux-supervision** lever (opp-reply
  policy head + per-cell ownership head, both @0.15). Rationale: aux heads were MIDDLING at small (v4
  `signal`) because they add 2 EXTRA prediction heads (representational load) a 64×4 net must starve its
  main heads to feed; a 96×6 net has the spare capacity to exploit the extra signal-per-position — the
  cleanest "compounds with more juice" bet. Config-only (flags exist, byte-identical-off), one-lever delta
  vs `derby-v9-medium`. A SUBMISSION (contestant) — the runner swaps it in + the ranking judges. (`medium`
  is tractable to train here, so a second medium lane is affordable.) WDL@medium is the secondary candidate
  (but WDL already did +56 on its own, so less of a "needed more juice" fit).

## Rules

- **Race to 140 epochs.** 140 is the milestone because that's roughly where a
  fresh model historically first beats the heuristic baseline. **Beat-heuristic
  (model_elo ≥ 800)** is the early checkpoint; the real prize is the **strongest
  model by epoch 140**.
- **10-epoch chunks.** Each scheduling step advances one idea by a 10-epoch
  increment (`gomoku.train --resume <idea>/latest.pt --epochs 10`).
- **Δelo/hour hill-climb priority** (Jason 2026-05-24: "never-run, then delta
  elo/hour — hill climb elo"). Order: **(1) never-run / entry-fee first** — an
  idea needs **2 elo points** to have a Δelo/hr slope, so round-0 then round-1 run
  for every idea (fewest points first); **(2) then highest Δelo/HOUR** over the
  most recent chunk — compute follows the *steepest recent climb*, not the highest
  absolute elo. Everyone caps; the steepest climbers get there first.
  - *History:* v1 ranked by last-chunk *raw Δelo* and pathologically fed the
    *worst* idea (a floored idea at Δ0 outranked a strong idea whose chunk dipped).
    v2 patched that with current-elo *level* (but that over-feeds an already-peaked
    champion and starves a faster challenger). The Δelo/**rate** rule is neither
    pathology: a floored idea sits at 0/hr and any genuine climber outranks it, and
    ranking the *rate* (not the level) is the literal hill-climb. The round-0/1
    entry fee avoids the floor-noise artifact (at the floor all ideas are ~equal, so
    a 1-point "rate" is meaningless).
- **Fresh self-play, shared init.** All ideas start from an identical fresh init
  (`--size small --seed 0`). No warm-start, no shared parent.
- **One lever each.** Every idea changes exactly ONE flag vs **C0-baseline** —
  clean attribution. C0-baseline is the control.
- **Anchored-elo scoring.** Score = the last `eval/model_elo` in
  `<idea_dir>/checkpoints/eval_results.jsonl`. Δelo for the queue = the change in
  that score across the idea's most recent chunk.

**Shared knobs** (held constant across all ideas):
`--games-per-epoch 64`, `--training-steps 400`, `--batch-size 256`,
`--replay-buffer-size 100000`, `--lr 1e-3`, `--temperature-moves 8`,
`--c-puct 1.25`, `--size small`, `--seed 0`. C0's `--n-simulations 200` is the
generation-strength control point.

**Out of scope (future board).** Curator / curriculum ideas
(recency-weighted, lru, gomocup-seed) are deferred — they require the
`train_replay` flywheel engine (curated in-RAM sampling over an archived buffer),
not the fresh self-play engine this derby races on. They get their own board once
a headroom parent exists.

---

## v1 FINAL — verdict (called 2026-05-24 at 5/8 capped; ranked by ELO, NOT wall-clock)

> **Wall-clock is busted for this run** and must NOT be used to rank: the derby ran
> single-process (`gomoku.train`, one stream, GPU ~30%), so every wall-time / Δelo/hr
> here is single-stream and unrepresentative of production (wave-mode, 8 workers,
> saturated). The under-counting trap ([[project-perf-bench-lesson]]). Rank by elo.

| rank | idea | lever | peak | final | beat-heur @ep |
|---:|---|---|---:|---:|:--:|
| 1 | **open-div4** | random_opening_moves 4 (WL3) | **1385** | 1385 | 90 |
| 2 | **temp-16** | temperature_moves 16 | **1340** | 1240 | 90 |
| 3 | sgd-800 | training_steps 800 | 1284 | 1081 | 70 |
| 4 | sims-400 | n_simulations 400 | 1265 | 1094 | 50 |
| 5 | buf-30k | replay_buffer 30k | 908 | 751 | 110 |
| — | C0-baseline | control | 567 (climbing, called @ep60) | — | — |
| — | ema-099 | ema_tau 0.99 | 405 (floor, ep50) | — | — |
| — | sims-100 | n_simulations 100 | 389 (NEVER grokked, ep110) | — | — |

**Findings:**
1. **Exploration/diversity levers win the ceiling.** Random openings (1385) and high temperature (1340) are the top 2 — *above* the compute levers (more sims 1265, more SGD 1284). Diversifying self-play raises reachable strength.
2. **Compute levers grok FASTER but peak LOWER.** Beat-heuristic timing tracks per-epoch compute: sims-400 @ep50 < sgd-800 @ep70 < open-div4/temp-16 @ep90 < buf-30k @ep110. More sims/SGD = earliest crossing; exploration = highest ceiling.
3. **Overtraining is real and lever-dependent.** sims-400/sgd-800 peaked ~ep90 then regressed ~180 elo by ep140; `open-div4` ended *at* its peak (openings sustain the climb, no overtrain); temp-16 mild (1340→1240).
4. **`sims-100` (100 sims) never groks** — floor 389 at ep110. Weak MCTS targets cap the climb (and it's the only *trainer-bound* recipe: gen<train).
5. **Generation-bound, not trainer-bound.** Train is a fixed ~10.5s/epoch floor; MCTS generation is 2–5× that and scales with sims. The trainer is cheap; Δelo/hr leverage is all on generation speed.
   > ⚠ **CORRECTED 2026-05-24 (Jason): this v1 reading is stale + was measured single-process.** It was taken on the SINGLE-PROCESS v1 derby (GPU ~30%, busted wall-clock), so the gen/train ratio itself is suspect. More importantly, after the perf wins (fp16-eval, V=512, native MCTS) **the regime FLIPPED: generation now OUTPACES the trainer — it FLOODS it.** Per [perf-bench-vs-real-training-cost.md](../topics/perf-bench-vs-real-training-cost.md): "maximizing generation throughput floods the trainer"; the LF1 runaway (per-epoch 20s→7min) is the generator producing positions faster than the trainer can use them, with `sgd_per_position` blowing up trying to consume the flood. **"Generation-bound" is NOT a standing truth — it's recipe-dependent (high-sims wave leans slower-gen; optimized fp16/high-V floods).** The v3 cards below that invoke "generation-bound" should be read through this correction. The fix is a fixed per-epoch SGD cap decoupled from inflow → the `derby-gumbel-fast5s` lane.
6. **Method fixes (mid-run):** priority must rank by *current elo*, not last-chunk-Δelo (the latter fed the *worst* idea); peak checkpoints were lost to `keep-last-n=3`.

## v2 — what's next (queued)

Re-run the **top 3** (`open-div4`, `temp-16`, `sgd-800`) **HEAD-TO-HEAD**, using the **production multiprocess recipe** (`run_sweep` wave-mode, 8 `selfplay_worker`s — saturates the GPU, so wall-clock is REAL). Eval = round-robin *direct matches* among the 3 (they're all >1280, so they'd saturate the anchor ladder — head-to-head via `delta_e_harness --head-to-head` is the correct eval for strong models). **Wall-native budget** (hours, not epochs; chunk = wall-slice; allocate+stop by Δelo/hr) measured on the saturated machine so Δelo/hr is finally honest. Carry the fixes: current-elo priority, peak-checkpoint snapshotting. (Verified 2026-05-24: the production recipe IS multiprocess; the single-stream drift was only in the v1 derby harness.)

---

## v3 — UNIFIED prior-art race (LAUNCHED 2026-05-24, `scripts/derby_v3_board.json`)

A **unified board**: Jason called it — rather than run v2 (the top-3 head-to-head) to
cap and *then* a separate v3, we **ported the v2 carryover recipes into v3** and race
everything at once. v2 was stopped at round-0 (all at the 389 floor → zero data lost),
which freed the box for the native `.so` rebuild. The roster (9 ideas, one lever each
vs the `c0` control, fresh `--size small --seed 0`, scored by anchored elo then
head-to-head at the top): the **v1/v2 carryovers** (open-div4, temp-16, sgd-800) +
the **4 prior-art levers** (playoutcap, forced, swa, gumbel) + a **sims100 control**
for gumbel. Each picked to attack a **specific v1 finding** — v1 told us we're
**generation-bound** and **exploration/diversity beats raw compute for the ceiling**,
so the new levers are biased toward *better targets per unit of generation*. All ran
through the lab's two-queue fan-out (5 worktree code lanes, opt-in flags, production
byte-identical when off, merged `--no-ff` serially with one native rebuild).

> Wall-fairness resolved: **gumbel + forced-playouts both run in the native C engine**
> (`_mcts_native.c`, rebuilt). Gumbel's first cut came back python-only (~5× slow =
> DOA per Jason); the **native C port** (per-game Sequential Halving inside the wave)
> made it **0.86–1.26× native PUCT** — wall-fair, raced wall-matched like the rest.
> (Gumbel + sims100 run at sims=100: gumbel's value-prop is good targets at *cheap*
> sims; sims100 is the plain-MCTS control that isolates whether gumbel rescues them.)
> **aux-head** (opponent-reply, Class-C model-arch) is built + verified but parked on
> its **own axis/board** — not in this search/recipe race.

### v3 FINAL — verdict (called as-is 2026-05-24, in prep for new contenders)

> Jason called v3 once `gumbel-fast5s` proved itself, to clear the board for a new
> round. Standings are the live fine-grained peaks (the slice-close state lags the
> mid-slice high-water mark — e.g. `gumbel-fast5s` touched 1620 mid-slice but its
> slice-close points were ~1455). 6 ideas after the mid-run prune of temp-16 /
> sgd-800 / playoutcap / swa (all stuck at the 389 floor).

| rank | idea | peak | wall→peak | grokked? | what it is |
|---:|---|---:|---:|:--:|---|
| 1 | **gumbel-fast5s** | **1620** ✓ | **~17 min** | yes | Gumbel@100 gen + **fixed-step trainer** (non-wave, `--sgd-steps-per-epoch 64`) |
| 2 | **gumbel** | **1580** ✓ | ~22 min | yes | Gumbel@100 gen + wave + `sgd_per_position` |
| 3 | forced | 1262 ✓ | ~21 min | yes | KataGo forced playouts + target pruning |
| 4 | open-div4 | 776 | ~21 min | ~ | v1's #1 (random openings) |
| 5 | sims100 | 697 | ~23 min | ~ | plain MCTS@100 — gumbel's control |
| 6 | c0 | 603 | ~21 min | ~ | no-lever control |

**Findings:**
1. **The Gumbel cheap-sims generator dominates.** `gumbel`/`gumbel-fast5s` (Gumbel@100) peaked ~1580–1620 — *more than 2× the control* `sims100` (plain MCTS@100, 697). So Gumbel doesn't just ride cheap sims, it **rescues** them: good targets at n=100 ≫ plain visit-count targets at n=100. This was the highest-leverage import.
2. **Fixed-step trainer is CO-EQUAL with wave — its wins are STRUCTURAL, not a Δelo-rate separation.** (Corrected post-Reviewer: an earlier draft claimed "~2× Δelo/hr / beats wave" — that mixed fast5s's time-to-peak against gumbel's *total* wall and is wrong.) Same Gumbel@100 generator; only the training mode differs. The two peaked **within eval noise** (fast5s 1620, gumbel 1580 — a ceiling tie), and on the derby's **canonical Δelo/hr** ((peak−389)/wall-to-peak, `standings.md`) **wave actually edges it: gumbel 3031 vs fast5s 2825** (~10% apart; like-for-like time-to-peak is ~1.5× at most — it flips with the wall basis). So **no clean rate separation.** Fixed-step's real advantages are structural: it reached its peak in a **single contiguous slice** (no multi-chunk resume), it is **structurally incapable of the inflow-driven runaway** (the whole point vs `sgd_per_position`), and it validated the gen-flooding fix — which is why it's the cleaner v4 **control**, not because it out-paced wave. Diagnostics (healthy): `reuse` ~1.4 settling to ~1.05, `pl` **4.39→1.72 descending** vs climbing `cumsteps` (productive SGD, *not* the redundant-flattening failure), `plies` ~67 (real defensive play).
3. **Prior-art compute-efficiency > v1 exploration levers.** `open-div4` (v1's ceiling champion, random openings) reached only 776 here vs gumbel's ~1600. The v1 headline "*exploration beats compute for the ceiling*" is **superseded once you have good cheap targets + efficient training**: Gumbel (target quality per sim) + fixed-step (training efficiency under a gen flood) beat the exploration knobs outright.
4. **Regime correction (the load-bearing reframe).** Generation now **floods** the trainer (`gen=0.4s ≪ train=3.0s`); "generation-bound" was stale (see correction in v1 finding #5 + `perf-bench-vs-real-training-cost.md`). The cure — a **fixed per-epoch SGD cap decoupled from inflow** (structurally can't run away) — is exactly the fixed-step mode, now empirically the best trainer.
5. **forced (KataGo) is a solid mid-tier search lever** (1262) — cheaper exploration than more sims, but well below the Gumbel-generator + fixed-step combo.

**Built this round (the durable artifacts, all merged + tested):** native Gumbel C port (`_mcts_native.c`, wall-fair 0.86–1.26× PUCT); the **fixed-step trainer mode** (`--sgd-steps-per-epoch`, non-wave async + non-blocking ingest + `sample_reuse_ratio`/`cumulative_sgd_steps` metrics); the **wave-mode SIGTERM deadlock fix** (+ test); the **Δelo/hr hill-climb scheduler** (never-run → entry-fee → Δelo/hr, peak tiebreaker); `scripts/watch_derby.py` (live elo/Δelo·hr/wall viewer) + `scripts/derby_dashboard.py` (wandb workspace) + `scripts/derby_sync_elo.py`; the discovery that **eval was already in wandb history** (trainer forwards the eval jsonl); and the **generation-flooding** correction (memory + wiki).

**Caveats:** anchored elo saturates ~1700 (the strong climbers are near the ceiling) → a **head-to-head** (`delta_e_harness --head-to-head`) is the rigorous tiebreak if a clean gumbel-vs-fast5s separation is ever needed; the two are within eval noise on ceiling. The fixed-step A/B conflates wave→non-wave + scaled→fixed (the training-*mode* fork, by design, not a single knob).

**Prep for new contenders (next round):** board is clear. Open candidates: the parked **aux-head** (opponent-reply, Class-C — built, awaiting sign-off); an **N-sweep on `--sgd-steps-per-epoch`** (the reuse-ratio knob: how hard can we push fixed-step before redundant SGD?); **reanalyze / curator** ideas (need the train_replay flywheel — their own board); and `gumbel-fast5s` itself is the **new baseline** to beat. Promotion of fixed-step+Gumbel to a production lineage is a deliberate ESCALATE (Jason's call), deferred.

## v4 — best-shot COMBINATIONS (LAUNCHED 2026-05-25, `scripts/derby_v4_board.json`)

The first **combination** round (Jason: "no more one-lever — put forth our top 3
combinations that we think have the best shot at being great gomoku players").
Every lane shares the **v3-winning base** (fixed-step `--sgd-steps-per-epoch 64` +
Gumbel@100, non-wave) so the only delta per lane is the headline lever. Wall-slice
engine (`run_sweep_wall_slice`), 600s slices, **10800s (3 hr) per-idea cap** (the
deeper bets need absorption room — see `absorption-phase` memory), Δelo/hr
hill-climb priority. All four lanes start **fresh + fair** (the control is a
distinct cell `derby-v4-control`, byte-identical to `gumbel-fast5s`, so it does not
resume v3's 8.8G checkpoint).

| lane | cell | lever (vs control) | source |
|---|---|---|---|
| **control** | `derby-v4-control` | none — fresh v3 winner (fixed-step + Gumbel@100) | Derby v3 |
| **signal** | `derby-signal` | KataGo aux supervision: opp-reply policy head + per-cell ownership head, both `@0.15` | KataGo |
| **wholeboard** | `derby-wholeboard` | KataGo global-pooling residual blocks (latter half; +4.79% params) | KataGo |
| **vcf** | `derby-vcf` | exact VCF mate-teacher (overwrites policy/value targets on forced wins; value disc 0.98/floor 0.90) | Rapfi/classical |

**Integration (all merged to `main`, tests green, smoke-validated):** global-pool
(`c5a81d7`), VCF solver (`3a6c6d9`, 400-fuzz vs independent referee = zero false
positives), opp-reply+ownership aux heads (`5eb6eec`/`36a446a`, byte-identical-off
verified 96/96 examples + both heads ENABLED in smoke), Rapfi yardstick
(`0c30427`, START 9 → OK). Cells + board `251d1bf`/`3acdb00`; watchdog `b85c113`.
**Yardstick:** Rapfi (Gomocup Elo 2625) runs separately on the leader (above-ladder),
not as a per-chunk anchor.

**Live:** dashboard `https://wandb.ai/jasonyandell-forge42/gomoku?nw=gv4fh2vq2rr`;
`scripts/derby_v4_watchdog.sh` supervises (restart-if-dead + `narration.log`);
`python scripts/watch_derby.py --board scripts/derby_v4_board.json` for the live
terminal board. **Known minor:** max-plies *draws* yield `ownership=None` (masked)
rather than the all-zeros the code comment promises — benign (no winner to credit;
trained models rarely draw at max_plies). Under Reviewer at launch. Reviewer
verdict: **PASS** (no BLOCK; merge resolution correct, control fair, aux NaN-safe,
VCF no false positives).

### v4 FINAL — verdict (stopped 2026-05-25 after ~12.3 hr / 67 chunks, 0 watchdog restarts)

Ran fully autonomously overnight (cap raised 3h→24h mid-run so the leader kept
getting fed). Anchored elo **saturated ~1700**, so the overnight peaks were a tight
66-elo cluster that anchored eval can't separate — resolved with a **head-to-head
round-robin** (`scripts/round_robin.py`, reuses `delta_e_harness.head_to_head_eval`;
120 games/pair, paired 4-ply openings, sims=100).

| metric | vcf | control | signal | wholeboard |
|---|---:|---:|---:|---:|
| anchored peak elo | **1784** | 1760 | 1738 | 1718 |
| H2H round-robin rating | **+31** | −29 | +6 | −8 |
| H2H rank | **1** | **4** | 2 | 3 |

**The head-to-head reshuffled the order — and that's the lesson.** On anchored elo
`control` (plain v3 winner, no extra lever) looked like #2 (peak 1760), but played
directly against its peers it **loses all three matchups** (−9/−29/−50) and ranks
**last** — its anchored score was overtrained inflation against the fixed ladder.
**`vcf` is the genuine champion on BOTH metrics**: highest anchored peak (1784) AND
beats every lane head-to-head (+14/+29/+50). The **exact VCF mate-teacher is the
standout v4 lever.** The two KataGo combos (signal=aux heads, wholeboard=global-pool)
land in the middle and did *not* cleanly separate from the baseline.

**Compute-fairness caveat (load-bearing):** early "leads" were undertraining
artifacts. `vcf` sat at ~1497 for hours looking like a clear 4th, then — once the
Δelo/hr hill-climb kept feeding the only lane still gaining — climbed straight to
#1. The scheduler's apparent "over-feeding" of the laggard was the most informative
allocation of the night. Lesson: **rank above the anchored ceiling with head-to-head,
and don't trust an early anchored lead before lanes have equal compute.**

**Statistical honesty:** H2H CIs are wide (±62 elo; high draw rates ~50%, i.e. good
defense), so the top-3 ordering (vcf > signal > wholeboard) is *not* airtight — but
the directional signals are clean and consistent: **vcf beats everyone, control loses
to everyone.** Peak checkpoints saved at `sweep_runs/derby_v4/_peaks/<lane>/peak.pt`.

**Above-ladder Rapfi yardstick (`vcf` champion, 9x9 freestyle, 20 games/budget,
`sweep_runs/derby_v4/rapfi_vcf.jsonl`):** vcf vs `pbrain-rapfi` (arm64-NEON,
build 6e0a132) — **100ms: 0W-0L-20D (50%); 500ms: 2W-0L-18D (55%); 1000ms:
1W-3L-16D (45%)**. Read: **roughly draw-parity** — near-total draws, Rapfi only
edges ahead at the longest control (3L vs 1W @ 1s). Surprisingly strong for a
~1700-anchored from-scratch net. CAVEATS: tiny draw-dominated sample (noisy ±10%
per 2-game swing); **Rapfi's 2625 is a 15x15 Gomocup rating that does NOT transfer
to 9x9 freestyle** — this says "competitive with Rapfi *on 9x9 freestyle*", NOT
"~2625 elo"; and 9x9 freestyle is intrinsically drawish under solid two-sided
defense. Model plays sims=100 vs Rapfi's time budgets (not a matched control).

## v5 — STACK THE WINNERS (LAUNCHED 2026-05-25, `scripts/derby_v5_board.json`)

v4's champion was the **exact VCF mate-teacher** (`--vcf-teacher`) on the fixed-step
+ Gumbel@100 base. v5 asks the **compounding question**: do the *other* v4 levers
ADD anything on top of vcf, or is bare vcf already the bar? Every lane = the vcf
base + exactly ONE added lever; `control` is the bare vcf base (the bar to clear).
All four start **fresh + fair** on the same wall budget (global-pool changes the
trunk so it can't warm-start from a non-global checkpoint — uniform fresh start
keeps it apples-to-apples). Engine `run_sweep_wall_slice`, Δelo/hr hill-climb
priority, peak checkpoints at `sweep_runs/derby_v5/_peaks/<lane>/peak.pt`.

| lane | lever (vs control = bare vcf base) | source |
|---|---|---|
| **control** | none — bare vcf base (VCF mate-teacher + fixed-step + Gumbel@100) | Derby v4 champion |
| **vcf-signal** | + KataGo aux heads: opp-reply policy + per-cell ownership, both `@0.15` | KataGo |
| **vcf-wholeboard** | + KataGo global-pooling residual blocks (latter half; +4.79% params) | KataGo |
| **vcf-deep** | + deeper VCF solver (`--vcf-max-depth 32 --vcf-max-nodes 500000` vs 16/200k) — proves longer forced wins → more exact mate labels | Rapfi/classical |

### v5 FINAL — verdict (stopped 2026-05-26 at 38 chunks; H2H is the verdict)

> **Read this from the head-to-head, not the anchored peaks.** Anchored elo
> SATURATES ~1700 and at 38 chunks the peaks are a NOISY, less-settled cluster
> than v4's 67-chunk run. v5 was also **restarted several times mid-race to ship
> infrastructure** (board cap 3h→24h, slice 600s→300s, PIPELINED eval, and the new
> peak-progress+patience priority metric) — so within-v5 wall-clock / Δelo-rate
> comparisons are **confounded**. The honest framing: v5 is BOTH a lever-compounding
> test AND the round where we built two durable infra wins (pipelined eval + the
> peak-progress metric). The clean signal is the post-race **H2H round-robin**
> (`scripts/round_robin.py`, 120 games/pair, paired 4-ply openings, sims=100,
> `sweep_runs/derby_v5/round_robin.json`).

| metric | vcf-wholeboard | vcf-deep | vcf-signal | control |
|---|---:|---:|---:|---:|
| anchored peak elo | **1634** | 1455 | 1606 | 1476 |
| H2H round-robin rating | **+81** | +44 | +7 | −132 |
| H2H rank | **1** | 2 | 3 | **4** |

Pairwise H2H Δelo (row vs column; + = row beats column):

| | control | vcf-signal | vcf-wholeboard | vcf-deep |
|---|---:|---:|---:|---:|
| **control** | · | −95 | −165 | −137 |
| **vcf-signal** | +95 | · | −44 | −29 |
| **vcf-wholeboard** | +165 | +44 | · | +35 |
| **vcf-deep** | +137 | +29 | −35 | · |

**Compounding verdict (from H2H): the levers DO compound on vcf — bare vcf is NOT
the bar, it's the floor.** `control` (bare vcf base) **loses all three matchups**
(−95/−165/−137) and ranks dead last; every +1-lever lane beats it head-to-head.
The standout add-on is **vcf-wholeboard** (KataGo global-pooling): it beats every
peer (+44/+35 over signal/deep, +165 over control) AND holds the top anchored peak
(1634). So whole-board structure stacked on exact mate labels is the v5 win.

**The H2H reshuffled the order (the v4 lesson, again):** on anchored elo `vcf-signal`
(1606) looked like the clear #2 and `vcf-deep` (1455) the clear last — but played
directly, **vcf-deep ranks #2 (+44) and signal drops to #3 (+7)**, and deep beats
signal head-to-head (+29). vcf-signal's high anchored peak was ladder-inflation it
couldn't cash against live opponents; vcf-deep's low anchored peak understated a
model that actually plays well. **Don't trust an anchored lead near the ceiling.**

**Honest caveats:** H2H CIs are wide (±62–69 elo, draw rates ~30–35% decisive→~65%
draws, i.e. strong two-sided defense), so the **middle order (deep > signal, +44 vs
+7) is inside noise** and not airtight; only the bookends are clean (wholeboard
clearly #1, control clearly #4). The anchored peaks are only ~38 chunks (vs v4's 67)
and the mid-race restarts mean the climb signal is muddier than v4's. What IS robust:
**every lever beats bare vcf, and global-pool is the strongest of the three.** The
load-bearing read is directional, not the exact middle ordering.

**Two durable infra wins shipped mid-race (the other half of v5's value):**
**(1) pipelined eval** — eval runs concurrently with the next training slice instead
of blocking it (the `(pipelined)` PEAK milestones), so the GPU queue stays fed; and
**(2) the peak-progress + patience priority metric** — the scheduler now ranks lanes
by recent peak-progress with a patience window rather than raw last-chunk Δelo/hr,
which is what kept feeding lanes still gaining (the v4 lesson that the laggard `vcf`
was the most informative allocation, now baked into the scheduler).

**Next = v6.** vcf-wholeboard is the new base to beat (vcf + global-pool). Open
questions: does **vcf + global-pool + aux** stack a fourth lever cleanly, or do the
two KataGo levers (signal/wholeboard) overlap? A **longer, restart-free** v5-rerun
(no mid-race infra churn) would settle the muddy middle order; and the vcf-wholeboard
champion is the natural candidate for the **Rapfi above-ladder yardstick** + a
promotion-to-lineage ESCALATE (Jason's call).

## v6 — RESEARCHER ROUND 1 (LAUNCHED 2026-05-26, `scripts/derby_v6_board.json`)

First batch gated from the **beads backlog** (researcher proposes / Jason gates,
2026-05-25). All five lanes ride the **vcf base** (the v4/v5 champion: exact VCF
mate-teacher + fixed-step `--sgd-steps-per-epoch 64` + Gumbel@100); `control` =
bare vcf base, the bar. Three independent levers, each a delta vs control:

| lane | lever (vs control = bare vcf base) | bead | source |
|---|---|---|---|
| **control** | none — bare vcf base | — | v4/v5 champion |
| **adjudicate** | + `--max-plies 45` — truncate dead/drifting games → more fresh openings/hr (attacks Δelo/hr); blunt-cap first cut, confidence-resign is the follow-on | derby-24a | AGZ resignation |
| **mate-discount** | + `--value-discount 0.98` — z = outcome·γ^plies_to_end; generalizes the VCF mate-distance discount to ALL outcomes; targets the v4 "wins anchored but loses H2H" overtraining gap | derby-2yn | our VCF path |
| **sgd128** | + `--sgd-steps-per-epoch 128` (2× the 64 baseline) — reuse-ratio sweep | derby-g2j | — |
| **sgd256** | + `--sgd-steps-per-epoch 256` (4× baseline) — reuse-ratio sweep | derby-g2j | — |

### v6 FINAL — verdict (closed 2026-05-26 at 168 chunks; H2H is the verdict)

> **Read this from the head-to-head.** Anchored elo SATURATES ~1700 and at 168
> chunks the lanes are well-explored but the anchored peaks are a noisy cluster near
> the ceiling (adjudicate 1682, control & mate-discount tied 1665, sgd256 1606, sgd128
> 1555). The real verdict is the post-race **H2H round-robin** (`scripts/round_robin.py`,
> 120 games/pair, paired 4-ply openings, sims=100, `sweep_runs/derby_v6/round_robin.json`).
> mate-discount's WHOLE POINT is to fix the anchored-vs-H2H gap, so for it the H2H
> verdict is the only one that counts.

| metric | mate-discount | adjudicate | control | sgd256 | sgd128 |
|---|---:|---:|---:|---:|---:|
| anchored peak elo | 1665 | **1682** | 1665 | 1606 | 1555 |
| H2H round-robin rating | **+46** | +20 | +7 | −31 | −41 |
| H2H rank | **1** | 2 | 3 | 4 | **5** |

Pairwise H2H Δelo (row vs column; + = row beats column):

| | control | adjudicate | mate-discount | sgd128 | sgd256 |
|---|---:|---:|---:|---:|---:|
| **control** | · | −44 | −20 | +50 | +41 |
| **adjudicate** | +44 | · | −58 | +53 | +41 |
| **mate-discount** | +20 | +58 | · | +44 | +61 |
| **sgd128** | −50 | −53 | −44 | · | −17 |
| **sgd256** | −41 | −41 | −61 | +17 | · |

**Per-lever verdicts (from the H2H, the row that matters):**

**(a) adjudicate — YES, games/hr translated to real strength.** `--max-plies 45`
beats control head-to-head (+44, control winrate 0.438 over 120 games) and ranks #2.
Truncating dead/drifting games didn't poison the value head with false early
terminations (the AGZ false-resign worry); the extra fresh openings/hr cashed into a
stronger model. The blunt cap is a clean win — and the confidence-resign + disable-frac
follow-on (still in derby-24a's design) is the natural next refinement, now de-risked.

**(b) mate-discount — YES, it beats control HEAD-TO-HEAD (+20) and TOPS the table
(+46).** This is the hypothesis the lever was built for: distance-discounting the value
target for ALL outcomes (not just VCF mates) closes the v4 "wins anchored but loses
H2H" overtraining gap. mate-discount is tied with control on anchored peak (both 1665)
yet **beats control when they actually play** (+20) AND beats adjudicate directly
(+58) despite adjudicate's higher anchored peak (1682) — exactly the anchored-understates-
H2H signature the discount was meant to produce. It's the **clear #1** and the v6 win.

**(c) more SGD/epoch — NO, it HURT.** Both `sgd128` (−41) and `sgd256` (−31) rank
below control (+7) and **lose to it head-to-head** (control +50 over sgd128, +41 over
sgd256). Pushing SGD steps/epoch above the 64 baseline over-trains on the same buffer
(redundant-sample flattening) and costs real strength. sgd256 edges sgd128 (+17) but
that's inside noise; the robust read is **64 is at or below the reuse-ratio optimum —
do NOT raise it.** (If anything the sweep wants a point *below* 64, not above.)

**Honest caveats:** H2H CIs are wide (±62 elo, ~30–40% decisive → ~60–70% draws, strong
two-sided defense), so the bookends are clean but the middle is soft. **What's robust:**
the two losers (sgd128/sgd256) clearly sit below the three winners (every cross-pair is
−41…−61, all outside or at the CI edge), and **mate-discount clearly beats sgd128/sgd256
and adjudicate** (+44/+61/+58). **What's inside noise:** the control↔mate-discount (−20)
and control↔adjudicate (−44) margins and the mate-discount-vs-adjudicate ordering are
each near one CI half-width — directionally mate-discount > adjudicate > control, but not
airtight at the top. The load-bearing reads (mate-discount #1, both sgd lanes lose to
control) survive the CIs.

**Carry-forward = mate-discount (`--value-discount 0.98`).** It's the cleanest single
win of the round: tops the H2H, fixes the exact overtraining artifact it was designed
for, and is a one-flag change that composes with the vcf base. adjudicate is the
**second** carry candidate — it's a pure generation lever (orthogonal to the value-target
lever), so mate-discount + adjudicate is the natural v7 stack to test for compounding.
**sgd128/sgd256 are dead** — the 64 baseline stays (or drops). Open: does mate-discount
stack on the v5 global-pool champion (vcf + global-pool + value-discount)? And the
adjudicate confidence-resign follow-on (derby-24a design) vs the blunt `--max-plies` cap.

### v3-gumbel  (HIGHEST leverage)
**Lever:** `--gumbel-root` (+ `--gumbel-m 16`, `--gumbel-c-visit 50`, `--gumbel-c-scale 1`) — Gumbel-top-k root sampling + Sequential Halving, completed-Q policy target. **Source:** Gumbel AlphaZero/MuZero, Danihelka et al. (DeepMind, 2022).

**Hypothesis:** Directly attacks v1's #1 finding (generation-bound) and its sharpest failure (`sims-100` "never grokked" under vanilla MCTS). Gumbel *provably* improves the policy even at tiny sim budgets (n=2..16) — so it should let self-play run far fewer sims per move yet still emit strong, low-variance targets, buying generation speed without the target-quality collapse that floored sims-100. The risk: the completed-Q target is a different target shape than visit-count policy; it could interact badly with our short-game distribution or need m/c-tuning to beat plain PUCT at our sim counts.

**Expected Δelo signature:** *Confirm* = at LOW sims (e.g. 100), Gumbel's Δelo/hr clears C0's and clears vanilla-MCTS-at-100 by a wide margin — strong targets cheap = the generation-bound win. *Refute* = Δelo ≈ vanilla PUCT at matched sims (the completed-Q target bought nothing at 9×9 scale) or instability from the target-shape change.

**Config delta vs C0:** `--n-simulations 100 --gumbel-root` (the point is cheap-sims-that-still-train; an A/B at 200 is a secondary cell).

### v3-playoutcap
**Lever:** `--playout-cap-frac 0.25 --playout-cap-fast-sims 50` — most moves run a small budget and are NOT recorded; ~25% run the full budget and ARE the training targets. **Source:** KataGo, Wu (2019); inherited by KataGomo (the engine we surveyed @ Gomocup 2254).

**Hypothesis:** Concentrate expensive search where it actually trains the net. Generation-bound says wall-clock is dominated by sims/move; spending the full budget on only ¼ of moves (and a cheap budget elsewhere just to keep the game progressing) should multiply games/wall at near-constant target quality — a different route to the same "more, fresher self-play per wall" that cheap-sims chases, but without weakening the *recorded* targets. The risk: the cheap moves still shape the game trajectory, so low-quality intermediate play could bias the distribution the recorded positions come from.

**Expected Δelo signature:** *Confirm* = Δelo/hr above C0 — same-or-better climb at materially less wall, because the recorded targets stay full-strength while the game advances cheaply. *Refute* = a shallower climb (cheap intermediate moves degraded the trajectory the targets are drawn from) or no wall win (the fast moves weren't cheap enough to matter).

**Config delta vs C0:** `--playout-cap-frac 0.25 --playout-cap-fast-sims 50` (full budget stays C0's `--n-simulations 200`).

### v3-forcedplayout
**Lever:** `--forced-playout-k 2.0` — force ≥ `ceil(sqrt(k·P(a)·N))` visits to each root child, then prune the forced visits back out of the policy target. **Source:** KataGo, Wu (2019).

**Hypothesis:** v1 found exploration beats compute for the ceiling, but "more sims" is the expensive way to explore. Forced playouts buy *root exploration* of promising-by-prior moves that PUCT would starve — without raising the sim budget — and target-pruning keeps the forced visits from polluting the trained policy. So: the ceiling benefit of exploration at the cost profile of a normal sim budget. The risk: at our small sim counts the forced minimums could *crowd out* the search's own signal, and the pruning rule could over/under-correct the target.

**Expected Δelo signature:** *Confirm* = a higher ceiling than C0 (echoing open-div4/temp-16's exploration win) at C0's wall cost — exploration without the sim tax. *Refute* = Δelo ≈ C0 (forcing didn't add useful exploration at 9×9) or instability from target-pruning artifacts.

**Config delta vs C0:** `--n-simulations 200 --forced-playout-k 2.0`.

### v3-swa
**Lever:** `--swa-window K` — publish self-play generator weights as the flat average of the last K checkpoints, instead of EMA/live. **Source:** Stochastic Weight Averaging / Leela Chess Zero weight-averaging practice.

**Hypothesis:** v1's `ema-099` was a *floor* — the exponential moving average LAGGED the learner on a fast climb, generating from a staler/weaker policy. SWA is the targeted fix: a flat tail-average smooths target generation (the stability EMA was reaching for) *without* the unbounded lag of exponential decay, since old weights fall out of the window entirely. So it should recover EMA's stability benefit on the climb without the lag penalty that sank ema-099. The risk: on a *fast* climb even a short flat window still mixes in too-old weights and lags; or the smoothing simply isn't worth anything when fresh-start gradients are large and directionally consistent.

**Expected Δelo signature:** *Confirm* = lower chunk-to-chunk Δelo variance than C0 AND a climb that stays at-or-above C0 (beating ema-099's floor) — stability without lag. *Refute* = lags like ema-099 (window too wide / climb too fast) or no variance reduction (the live policy was already fine).

**Config delta vs C0:** `--n-simulations 200 --swa-window 5` (tune K; contrast directly with v1's `--ema-tau 0.99`).

### v3-auxhead (Class-C, design-only — see `auxiliary-targets-design.md`)
**Lever:** an opponent-reply auxiliary policy head (recommended), opt-in via an aux-loss weight; predicts the opponent's next-ply policy for extra gradient per position. **Source:** KataGo auxiliary targets, adapted to 9×9.

**Hypothesis:** Attacks the laptop-scale thin-signal problem ([[az-at-scale-vs-laptop]]): short gomoku games yield few near-opening positions, so each scarce position should teach the net more. An opponent-reply head squeezes a second supervised signal from data we already generate. *This is a model-architecture change (Class C)* — design first, user sign-off before any model.py edit. Card finalized from the design doc.

**Expected Δelo signature:** *Confirm* = steeper Δelo/hr than C0 at equal generation (more signal per position = faster learning from scarce data), aux head dropped at inference so self-play/eval cost is unchanged. *Refute* = aux loss distracts the shared tower (policy/value regress) or the extra signal is redundant with the value target on short games.

**Config delta vs C0:** TBD from the design doc (e.g. `--aux-opponent-reply-weight 0.15`, default 0.0 = off).

---

## Title cards

### C0-baseline
**Lever:** control — sims 200, 64 games, 400 steps, buf 100k, temp 8, lr 1e-3.

**Hypothesis:** The reference climb. Every other card is read as a delta against
C0's anchored-elo trajectory. No claim of its own; it defines "did the lever
help or hurt the *rate*."

**Expected Δelo signature:** A monotone climb that crosses elo ≥ 800
(beat-heuristic) somewhere near epoch ~140 — by construction the milestone is
calibrated to roughly this recipe. Sets the per-chunk Δelo baseline the queue
sorts against.

**Config delta vs C0:** none (`--n-simulations 200`).

---

### sims-400
**Lever:** `n_simulations 400` (vs 200) — stronger policy targets, slower gen.

**Hypothesis:** Deeper MCTS per move yields sharper, lower-noise policy/value
targets, which could steepen the early climb — *if* target quality, not game
volume, is the binding constraint on the fresh-start ascent. The cost is ~2×
slower generation per game, so within a fixed 10-epoch (not fixed-wall) chunk it
trains on the same number of games but better-labeled ones. The risk: at low
model strength the extra sims mostly refine an already-cheap-to-estimate policy,
buying little while the wall-clock per chunk balloons.

**Expected Δelo signature:** *Confirm* = a higher Δelo per chunk than C0,
especially in the mid-climb (epochs 40–100) where target sharpness should matter
most; reaches 140 with a higher final elo. *Refute* = Δelo tracking C0 within
noise (target quality wasn't the bottleneck) while each chunk costs ~2× the wall.

**Config delta vs C0:** `--n-simulations 400`.

---

### sims-100
**Lever:** `n_simulations 100` (vs 200) — weaker targets, ~2× faster gen / more
games per wall.

**Hypothesis:** The LF1 lesson, inverted. LF1 showed that *fast generation
floods the trainer* — cheap gen pushed new-positions/epoch so high the trainer
fell behind and re-ground stale buffer. Here gen is cheap by design: does
cheaper/faster generation *win the climb* (more, fresher self-play per wall
overcomes weaker per-move targets), or does it just produce *noisier targets*
that slow the ascent? This is the early-climb test of the
volume-vs-quality tradeoff that the converged-model flywheel work couldn't isolate.

**Expected Δelo signature:** *Confirm (volume wins)* = Δelo per chunk meets or
beats C0 at a fraction of the wall — cheap gen is the efficiency frontier.
*Refute (noise dominates)* = a visibly shallower climb than C0, late or never
crossing elo 800, the weak targets capping reachable strength.

**Config delta vs C0:** `--n-simulations 100`.

---

### sgd-800
**Lever:** `training_steps 800` (vs 400) — more fit per epoch.

**Hypothesis:** delta-e run-2 found that **extra SGD bought nothing on a
*converged* model** — `lru,sgd=300` netted the identical chess-score to
`lru,sgd=100`, just played sharper, no net strength. But that was a net at its
optimum re-grinding a fixed curated slice. **Does more fit-per-epoch help on the
*climb***, where the net is far from convergence and each fresh batch carries
real un-learned signal? If the binding constraint early is "we under-fit the
data we generate," doubling SGD steps should steepen Δelo. The risk is the same
over-grinding seen at convergence reappears once the buffer is dominated by
stale self-play.

**Expected Δelo signature:** *Confirm* = steeper early Δelo than C0 (epochs
0–60), tapering as the net approaches the data's information limit. *Refute* =
the run-2 result generalizes — Δelo ≈ C0 within noise despite 2× the SGD,
extra grinding wasted even on the climb.

**Config delta vs C0:** `--n-simulations 200 --training-steps 800`.

---

### buf-30k
**Lever:** `replay_buffer_size 30k` (vs 100k) — faster turnover, fits recent
self-play harder.

**Hypothesis:** A smaller buffer turns over faster, so each epoch's SGD sees a
higher fraction of *recent* (stronger-policy) self-play and less stale
early-model garbage. On a fast climb where the policy is improving every few
epochs, weighting toward recent games could steepen Δelo — a fixed-buffer echo
of the recency-weighted curator finding (recency >> lru). The risk: 30k is small
enough to over-fit a narrow recent distribution and lose the diversity that
keeps targets honest, inducing instability.

**Expected Δelo signature:** *Confirm* = Δelo at or above C0 with the gap opening
mid-climb as recency compounds. *Refute* = higher per-chunk variance and/or a
shallower climb, recent-overfit eating the freshness gain.

**Config delta vs C0:** `--n-simulations 200 --replay-buffer-size 30000`.

---

### open-div4
**Lever:** `random_opening_moves 4` (the WL3 lever) — opening diversity,
better-balanced climb.

**Hypothesis:** WL3's diversity lever. Forcing 4 random opening moves spreads
self-play across a wider opening distribution, preventing the fresh net from
collapsing onto one or two dominant lines and over-fitting them. A
better-balanced game distribution should produce a steadier, less-degenerate
climb — and is the same mechanism that fixed delta-e run-2's near-50% decisive
rate (paired random openings made games decisive instead of replaying one line).
The risk: opening randomness adds early-game noise that slows the first chunks
before the diversity pays off.

**Expected Δelo signature:** *Confirm* = a smoother, more monotone climb than C0
with fewer regressions per chunk, and equal-or-better final elo. *Refute* =
slower early Δelo (noise tax) that the diversity never recoups by 140.

**Config delta vs C0:** `--n-simulations 200 --random-opening-moves 4`.

---

### ema-099
**Lever:** `ema_tau 0.99` (the WL2 lever) — EMA self-play weights, smoother
targets.

**Hypothesis:** WL2's stability lever. Generating self-play from an
exponential-moving-average of the weights (τ=0.99) instead of the live net gives
a slower-moving, lower-variance target-generation policy — the actor lags the
learner, so targets stop chasing every SGD wobble. This should reduce
self-play-target variance and smooth the climb, potentially raising Δelo by
keeping the net from training against its own noise. The risk on a *fast* climb:
the EMA lag could hold generation behind the learner's actual strength, slowing
how fast better targets become available.

**Expected Δelo signature:** *Confirm* = lower chunk-to-chunk Δelo variance than
C0 and a steady, regression-light climb. *Refute* = a visibly lagged climb —
Δelo tracking below C0 because the EMA actor keeps generating from a staler,
weaker policy than the learner has already reached.

**Config delta vs C0:** `--n-simulations 200 --ema-tau 0.99`.

---

### temp-16
**Lever:** `temperature_moves 16` (vs 8) — more opening exploration.

**Hypothesis:** Doubling the temperature-1 (sampling) window from 8 to 16 plies
keeps self-play exploratory deeper into the game before switching to greedy
selection. More exploration → broader state coverage and richer policy targets
in the early/mid game, which could steepen the climb the same way opening
diversity does — but via in-game sampling rather than forced random openings.
The risk: 16 plies of sampling injects weaker, higher-entropy moves into the
training data, diluting target quality and slowing convergence.

**Expected Δelo signature:** *Confirm* = Δelo at or above C0 with better coverage
showing as a steadier mid-climb and equal-or-higher final elo. *Refute* = a
shallower climb than C0 — the extra sampled-move noise outweighs the coverage
gain.

**Config delta vs C0:** `--n-simulations 200 --temperature-moves 16`.

---

<!-- STANDINGS:AUTO — delo_derby.py rewrites everything below this line -->

## Standings

_Last updated: 2026-05-24T15:31:05Z — 92 chunks run._

**Champion so far:** `open-div4` at 1385 elo (140/140 epochs).

| Rank | Idea | Epochs | Elo | Peak | Wall (min) | Δelo/hr | Beat heuristic? | Status |
|-----:|------|:------:|----:|-----:|-----------:|--------:|:---------------:|--------|
| 1 | open-div4 | 140/140 | 1385 | 1385 | 73.5 | 813 | ✓ | capped |
| 2 | temp-16 | 140/140 | 1240 | 1340 | 76.2 | 823 | ✓ | capped |
| 3 | sims-400 | 140/140 | 1094 | 1265 | 140.1 | 614 | ✓ | capped |
| 4 | sgd-800 | 140/140 | 1081 | 1284 | 105.0 | 874 | ✓ | capped |
| 5 | buf-30k | 140/140 | 751 | 908 | 77.9 | 488 | ✓ | capped |
| 6 | C0-baseline | 60/140 | 567 | 567 | 27.8 | 384 |  | queued |
| 7 | sims-100 | 110/140 | 389 | 389 | 33.2 | -8 |  | queued |
| 8 | ema-099 | 50/140 | 389 | 405 | 21.6 | 57 |  | queued |

_Δelo/hr = (peak elo − 389 floor) ÷ wall-hours-to-peak: real-strength gain per wall-clock hour, the north-star. Beat-heuristic ✓ = peak ≥ 800._
