# The sound-world recipe (#107) — oracle in the environment, not the loss

**One-line:** self-play where the GPU oracle makes blunders unplayable (veto) and both game-ends
exact (attacker VCT terminus + defender all-moves-lose terminus) — the punisher the plain self-play
twin never provided. Killed the 9-ply fast-attack attractor (#101) in one day after weeks of
target-side injection failed; validated on 9×9 2026-07-01/02 (TRAINING_WIKI #107 entries).

## Why it works (the one ML lesson)
AlphaZero's learning operator is: search improves the prior → net distills the search's visit
distribution + game outcome. Every failed VCT injection (#36/#42/#43/#77/#86/#102/#103) edited
TARGETS after the fact — off-policy, fighting the distribution. The veto edits the GAMES: targets
stay the net's own (constrained) search, on-policy by construction. Cap the veto breadth and the
attractor returns (K=24 ablation) — causal, not correlational.

## The levers (all byte-identical-off)
- `--vct-terminus --vct-terminus-budget 25` (worker): attacker end, one-hot on the oracle move (#98).
  **Cap25 as of 2026-07-03** (#114, Jason-approved; was cap50 through the 9×9 chapter). This single
  flag is the shared oracle node cap for BOTH the terminus test AND the per-ply veto escape-solves
  (`_VCT_TERMINUS_BUDGET`, `gomoku/self_play.py`), so it is the gen-throughput dial — see
  § Oracle budget: the cap50→cap25 flip. *(The eval-time finisher stays cap50.)*
- `--oracle-veto` (worker): per-ply bulk escape-solve at FULL breadth; proven-losing moves masked
  from played move AND recorded pi; all-legal-lose ⇒ defender terminus, z=−1, **NO example recorded
  for the doomed position** (the uniform-pi shrug collapsed white at scale — the #107 wound; see
  TRAINING_WIKI 2026-07-01 correction). Breadth caps are a 9×9 semantics trap; big-board staged
  escalation exists behind `--oracle-veto-max-cands` (leak rate must be measured).
- `--oracle-overlap` (worker): merged per-ply solve runs under the MPS search wave (1.18×, exonerated
  by the poison detector). Merged solve itself is default-on, bit-identical (1.07×).
- `--line-planes` (trainer): 8 in-forward line-potential channels; cross-line threats become local
  reads. In-model ⇒ 17-plane external contract untouched.
- Everything else: clone of `moonshot`/`vctsci-terminus` (WL2 stack, value-discount 0.98,
  global-pool, sgd-steps 64). Cell: `sound-world` in scripts/run_sweep.py.

## Guardrails (blood-bought)
1. **`uv run python scripts/gen_poison_check.py <ckpt>` after ANY gen-semantics change**: generates
   at live config and asserts NO recorded example carries policy mass on a proven-blunder cell
   (strict since the fix). Run A's poison was invisible to pl/vl/plies for 700 epochs.
2. **Gate on H2H + per-color columns, never internal metrics** (third confirmation of the #100
   lesson). The arena's `--json` gives the color split; the collapse signature is one-sided.
3. Never record a degenerate policy target "for the value signal" — drop the example; discounting
   carries z.

## Oracle budget: the cap50→cap25 flip (#114, 2026-07-03)
`--vct-terminus-budget` is the single node cap governing EVERY oracle solve in the loop — the attacker
terminus test AND the per-ply veto escape-child solves (`_VCT_TERMINUS_BUDGET`, `gomoku/self_play.py`
lines 263/315/459/480/529) — and the veto solve is [~91% of 13×13 gen wall](mega-vct-solver.md) (§5.5),
so the cap is the throughput dial. The perf blitz measured lowering it 50→25.

**The recall study (gate MET; receipts `scripts/vct_metal/cap25_recall_receipts.md` +
`cap25_recall_study.py`, #114).** *Recall* = the fraction of cap50-proven vetoes still proven at cap25:
**99.93%** (13×13 sound-world-scratch net; 30 leaks / 40,961) / **99.39%** (13×13 full-game
swap2/G-ladder; 525 / 85,884) / **98.64%** (9×9 champion `107b`, the worst case; 164 / 12,094). Solve
**~1.98×** at both sizes (halving the cap ≈ halves the solve wall). Every cap50-proven win recovered is
a DEFENSE escape-child — the attacker terminus fires **0** forced-VCT-for-mover at any self-play root,
so the recall question is entirely the soundness-critical defense-veto path. Two kernel invariants held
with 0 violations everywhere: monotonicity (`proven@25 ⊆ proven@50`) and leak-capped (every leak board
is `hit_cap@25`, i.e. a genuinely-harder proof, not a wrong verdict). Plumbing guardrail
`GOMOKU_POISON_BUDGET=25 gen_poison_check.py` = **0/174**. A missed veto is a *played blunder* (the
K-cap precedent, § Why it works); the residual ~0.07–1.36% leak is distribution-dependent.

**The decision (Jason, 2026-07-03).** *"cap25 is large savings and minimal cost. I'll happily pay 2%
gap on the high end in order to get the speedup."* **LANDED:** the `sound-world` cell's
`--vct-terminus-budget` 50→25 (one flag value; commit `8e2d9e1`, merge `09c067b`, `Closes #114`). This
is a **flat cap25 on the one cell**, not board-size-conditional logic — the study *recommended* cap25 at
13×13 / keep cap50 at 9×9, but what shipped is a single value plus the code note "a 9×9 rerun (closed
chapter) may prefer 50 — override per run." **The eval-time finisher stays cap50** (`vct_finish=50`):
one cheap call per round buys conversion strength. Composed with the (default-OFF) `lanes=K` kernel the
13×13 solver stack is ≈**2.7×** on the ~90%-of-wall component — see
[mega-vct-solver.md](mega-vct-solver.md) §5.3/§5.5 and [mcts-perf-ceiling.md](mcts-perf-ceiling.md).

## Known open edges
- White-vs-lookahead:4 softness at 9×9 (5/20 white losses @ e1368) — unsettled when the chapter closed.
- 13×13 gen **perf** prerequisite — **RESOLVED 2026-07-01/03** (dated correction; the original framing
  below, "cross-worker shared oracle solve, still open — width-is-free ⇒ ÷4 aggregate oracle time," was
  superseded by measurement). The blitz landed the levers: continuous-refill fleet-consolidation (#112),
  the `lanes=K` multi-thread-per-board kernel (#114, built + verified, default-OFF), and the cap50→cap25
  oracle-budget flip (#114, live in this cell — § Oracle budget above). Correction to the ÷4 intuition:
  streaming ≈ lockstep at EQUAL width in a single process (216 vs 216 games/min @13×13); #112's 3.4–4.6×
  was a *fleet → one-wide-process* comparison. See the § Perf isolation below and
  [mega-vct-solver.md](mega-vct-solver.md) §5.5. Product shape = net + **cap50** finisher (95% vs
  heuristic on 9×9 where bare-net draws; the finisher stays cap50 even though gen went cap25). **NB the
  13×13 graduation itself was attempted and is a STRUCTURAL NEGATIVE — see the #113 section below; this
  bullet is only the perf note, not an untried plan.**

## 13×13 graduation (#113, 2026-07-02) — the NEGATIVE result: an attack-only specialist

Carrying the validated 9×9 recipe up to 13×13 **failed**, and the failure is STRUCTURAL to the
cap50-terminus recipe, not a seed or poison artifact. This is the chapter's most important learning:
the sound world as built teaches OFFENSE and starves DEFENSE once the board is big enough for black
to force a fast VCT.

**Two independent runs, same verdict — white 0/20 everywhere.**
- **Warm-start** (wandb `8rp0gjpm`, `~/data/sound-world-13`, HF `jasonyandell/gomoku-13x13`):
  seed built offline = fresh 13×13 net + shape-matching copy of the 9×9 107b-e1540 conv tower
  (threat-shape features transfer; the board-flattened `policy_fc`/`value_fc1` heads REINIT fresh,
  strict-loadable via the production board-size guard). `pl` 3.36→2.18→2.01 (PLATEAUED, never
  marched to the 9×9 white-defense zone ~1.3). Every eval config: white **0/20**; black offense
  real (cap50 finisher 12→20 wins vs heuristic).
- **From-scratch control** (wandb `uublz536`, `~/data/sound-world-13-scratch`): the VALIDATED recipe,
  no transfer. `pl` converged FASTER (1.83→1.74, no warm-bias fight) — and bought PURELY offense
  (black bare 0→4, finisher 15) with white STILL **0/20** and plies STILL flat ~14. Better policy
  convergence, identical defensive collapse.

**The mechanism (confirmed, the headline).** Black forces a proven VCT by ply ~9–13 in self-play
(poison-gen: 31/32–32/32 games decisive, many at the 9-ply floor = fastest possible five). So white
is ALREADY LOST when threats appear → the veto masks all of white's moves → defender terminus fires →
**white's sharp-defense examples never enter the buffer**. White cannot learn to defend at any `pl`.
Contrast 9×9, where the veto made plies RISE (20s→50s) and white could draw — the walls capped trap
complexity. At 169 cells the walls stop saving white, so the same veto that CURED the 9×9 attractor
now STARVES 13×13 defense. `pl`<1.6 gates are moot: offense saturates, no gradient is left for white.

**Poison stayed clean throughout (0/612, 0/414, 0/411, 0/410).** The #107 uniform-pi fix holds at
13×13 — this is NOT the old wound. The collapse is on-policy net weakness, correctly attributed.

**Comprehensive eval (2026-07-02 morning, n=40, EMA, sims=100, our nets finisher-armed vct_finish=50):**
| # | matchup | A (w-l-d) | black | white |
|---|---|---|---|---|
| 1 | from-scratch+fin vs **rapfi@50ms** | 0-40-0 (0%) | 0/20 | 0/20 |
| 2 | warm-start+fin vs **rapfi@50ms** | 0-40-0 (0%) | 0/20 | 0/20 |
| 3 | from-scratch vs warm-start [H2H] | 20-20 (50%) | 20-0 | 0-20 |
| 4 | from-scratch+fin vs **OLD 128×10** bare | 0-40-0 | 0/20 | 0/20 |
| 5 | warm-start+fin vs **OLD 128×10** bare | 0-40-0 | 0/20 | 0/20 |
| 6 | OLD 128×10 bare vs **rapfi@50ms** [anchor] | 3-37 (7.5%) | 3/20 | 0/20 |
OLD = `G-ladder-13-board13` e424 (128×10, swap2/full-game recipe, NO cap50 terminus, self-play plies
50–64). The H2H (row 3) is EXACTLY 50/50 and purely color-determined — whoever is black forces the
VCT; the two nets are behaviorally identical attack specialists, no skill delta (refuted the
warm-start-wins prediction). **The "we-never-focused-on-it" OLD full-game net BEATS both sound-world
nets 40-0 AND scores 7.5% vs rapfi where ours score 0%** — because it trained on FULL games and
learned to DEFEND. Net+finisher is not a product at 13×13; it's a black-only party trick: the
finisher only fires on a forced VCT, and anything that defends (rapfi, or our own OLD net) never
hands one over.

**Perf isolation (bench_gen_refill, net size HELD at 64×4, so gen-path is isolated from net size).**
The overnight "14× faster epoch" was **~entirely NET SIZE** (128×10 train ~44s → 64×4 train ~2.7s;
train time is net-size-bound). The refill/streaming loop adds ~0 single-process throughput. At 13×13
the **VCT oracle veto is 91% of gen wall** and cuts throughput ~3× (640→216 games/min) — the lever
for faster sound-world gen is the solver kernel (#114 domain), NOT the gen loop. (#112's 3.4× was
8-proc fleet vs 1 wide proc, a different comparison.)

**Verdict:** the 9×9 sound-world recipe does NOT transfer white defense to 13×13 under a fixed cap50
terminus. For a STRONG 13×13 net the full-game (defense-learning) lineage beats the terminus recipe.
The recipe needs a real modification to teach defense before it earns more GPU. Runs PRESERVED as
evidence: `~/data/sound-world-13` (warm, `8rp0gjpm`), `~/data/sound-world-13-scratch` (scratch,
`uublz536`).

## New ideas / open directions (2026-07-02, live design discussion — not yet tried)

The 13×13 negative result reframes the whole line. The terminus was a scaffold that, at scale,
gates white out of its own training data. Four directions came out of the design discussion; the
first two are recipe rebuilds, the third is the pivotal unknown that decides which recipe suffices,
the fourth is the synthesis.

### 1. Role-invariant / symmetric rails — DROP the terminus
Replace both the attacker terminus AND the defender veto with ONE role-invariant rule applied to
BOTH sides every ply: **mask any move that degrades the side-to-move's forced-outcome class**
(win→draw, win→loss, draw→loss). Then just PLAY ON to a natural five / full board; `z` = the actual
result. This UNIFIES today's two asymmetric mechanisms — attacker-press (terminus) and defender-veto
— into a single statement: *the side to move plays soundly*. Crucially, **removing the terminus is
the proposed cure for white-starvation**: with no early game-end, black's forced win no longer
ejects white from the buffer — white stays ON-POLICY EVERYWHERE and generates the sharp-defense
examples the terminus recipe never let it see. (Falsifiable prediction, in the 9×9 idiom: plies
should RISE and white's H2H column should climb off 0 — the exact signature the 13×13 terminus runs
never produced.)

### 2. Attacker-preserve mask — learn CLOSING on-policy
Complementary to (1): mask attacker moves that fail to keep the win/VCT "MOVING" (i.e. that let the
forced-outcome class slip). Today the bare net ATTACKS but can't CONVERT — it reaches winning
positions and doesn't cash them (9×9: bare draws, finisher 95%; that's why the product needed the
oracle finisher bolted on). Teaching the forcing TAIL on-policy — every attacker move must preserve
the forced win — could fold the finisher INTO the net, making the bare net a deadly closer instead
of renting the conversion from the oracle at inference.

### 3. THE pivotal open question — is 13×13 freestyle a forced BLACK win from the empty board?
This is the fork the whole line now hinges on. Known: **15×15 freestyle is a PROVEN black win**
(Allis; wiki/topics/allis-threat-theory.md). **9×9 is recorded drawish** (two sound players draw;
and our own runs show black forcing wins only within the cap50 horizon, the walls capping trap
complexity). **13×13 is UNKNOWN.**
- If 13×13 is **drawish**, then rails ALONE (idea 1) should fix white — a sound white can hold, so
  removing the terminus lets it learn to.
- If 13×13 is a **forced black win**, then no defensive recipe can make white hold from the empty
  board — the game is unfair, and rails must be bolted onto **FAIR OPENINGS** (swap2 / the idx-2
  balanced-opening protocol; wiki/topics/swap2-opening-protocol.md) so both colors face winnable
  positions. This is the same fairness problem 15×15 solves with swap2.
- **We can PROBE it directly:** the GPU mega VCT/VCF oracle (the validated solver;
  wiki/topics/mega-vct-solver.md) can search whether black has a forced win from the 13×13 empty
  board (or from shallow openings) — turning "unknown" into a measured answer before we commit GPU
  to either recipe. (Cross-reference the fast-black-win evidence already logged: 9×9-in-horizon is
  a fast black win; 13×13 self-play forces wins by ply ~9–13.)

### 4. Rails + fair openings — the complete recipe
The synthesis: **role-invariant rails (1) + attacker-preserve closing (2), launched from fair
openings (3) if 13×13 proves a black win.** Rails keep every ply sound for BOTH colors and keep white
on-policy (curing starvation); fair openings guarantee white is defending a winnable game (curing
unfairness). Idea 3's oracle probe tells us whether the openings piece is REQUIRED or optional. This
is the candidate recipe that could carry the sound world from a 9×9 proof-of-concept to a genuinely
strong, defends-and-closes net at 13×13 and beyond.

## See also
- [vct-terminus-selfplay-result.md](vct-terminus-selfplay-result.md) — the #98 attacker terminus this recipe composes with.
- [vct-mining-research.md](vct-mining-research.md) — the VCT-mining reference (whole-net vs oracle split, the mining hub).
- [vct-backward-mining.md](vct-backward-mining.md) — VCT mining research lineage.
- [white-side-defense-plan.md](white-side-defense-plan.md) — the white-defense problem this line keeps colliding with.
- [swap2-opening-protocol.md](swap2-opening-protocol.md) — the fair-openings piece of idea #4 (and the idx-2 connection).
- [mega-vct-solver.md](mega-vct-solver.md) — the GPU oracle that can probe the 13×13 forced-win question (§5.3 `lanes=K`, §5.5 the veto = ~91% of gen wall).
- [batched-eval-arena.md](batched-eval-arena.md) — the eval-time cap50 VCT finisher (#109) that turns the bare net into the 95%-vs-heuristic product.
- [net-architecture-and-representation.md](net-architecture-and-representation.md) — the net + line-planes representation this recipe trains (param counts, presets).
- [the-claw.md](the-claw.md) — the double-threat / line-planes representation motivation (net-architecture context).
