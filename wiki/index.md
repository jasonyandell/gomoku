# Gomoku Wiki Index

This wiki is the maintained synthesis layer for the Gomoku training project.
It should compound what we learn from experiments instead of forcing each new
session to rediscover the same story from W&B runs, checkpoints, logs, and chat
history.

## Start Here

> ⚠️ **2026-06-15 RECKONING — read before trusting any 15×15 strength number.**
> The Rapfi yardstick the campaign ranked on is **broken** (weightless classical
> build; ignores its own search time → the "TC tiers" were one shallow engine
> measured twice). Direct head-to-head **overturns the rankings**: `128×10` beats
> the crowned `96×8 e499` champion **40-0**; the "capacity reversal" never happened;
> the deepgen search-axis experiment failed (specializes, not strengthens). The
> 96×8 "champion" is the *weakest* trained net. See **[lessons §8–§9](topics/alphazero-lessons-15x15-gomoku.md)**,
> handoff `handoffs/handoff-2026-06-15-yardstick-overturns-campaign.md`, issues
> **#28** (fix yardstick) and **#29** (re-crown + matched-epoch capacity ladder).
> **Gate every "did this help?" on head-to-head vs the preserved champion, not Rapfi.**
>
> ✅ **2026-06-18 UPDATE — the yardstick is FIXED (#40 closed, #28 root-caused).**
> The "broken Rapfi" above was the **weightless classical build**; native **Rapfi-NNUE**
> (Gomocup 2024+2025 winner) now loads its NNUE config and searches to its full time
> budget (depth 32 / 2M nodes / full budget), single-thread, **no wine** — a registered
> default panel anchor. First champion contact: **`eval502` vs Rapfi @5s, n=24 → 20.8%**,
> and the whole shortfall is the white-defense gap (**black 42% / white 0/12**). See
> [external-engine-baselines.md](topics/external-engine-baselines.md) § *Rapfi-NNUE NATIVE
> ANCHOR ONLINE*, [white-side-defense-plan.md](topics/white-side-defense-plan.md) §1B.2
> (2026-06-18), and `TRAINING_WIKI.md`. Absolute Gomocup-Elo calibration still pending
> (#35/#30); the champion-not-Rapfi gating rule above still stands until then.

Pick the doorway that matches the task. The big training notebook is still the
source of chronological evidence; these routes keep future sessions from
reading it front-to-back unless the work actually needs that.

> 🧭 **New here, or "what can this repo DO?"** → start with **[capabilities.md](capabilities.md)**,
> a one-screen synthesis of the toolbox (mine · pretrain/warm-start · train · evaluate ·
> search · operate), each capability linked to its deep doc. It's the capability view;
> the table below is the task view.

| Need | Start with | Then read |
|---|---|---|
| **"What can this repo do?" (the capability map)** | [capabilities.md](capabilities.md) | the deep doc each capability row links to |
| **Understand what AlphaZero taught us on this game (the learning artifact)** | [topics/alphazero-lessons-15x15-gomoku.md](topics/alphazero-lessons-15x15-gomoku.md) | [topics/15x15-training-campaign.md](topics/15x15-training-campaign.md), [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md). |
| Current training story or "how did we get here?" | [topics/training-run-lineage.md](topics/training-run-lineage.md) | [TRAINING_WIKI.md](../TRAINING_WIKI.md) tail, then [log.md](log.md). |
| Launch, resume, monitor, or stop a run | [topics/launch-sequence-runbook.md](topics/launch-sequence-runbook.md) | The relevant design page, then the latest run section in [TRAINING_WIKI.md](../TRAINING_WIKI.md). |
| Interpret training dynamics | [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md) and [topics/az-at-scale-vs-laptop.md](topics/az-at-scale-vs-laptop.md) | The run's evidence section in [TRAINING_WIKI.md](../TRAINING_WIKI.md). |
| Plan a WL-series follow-up | [topics/training-run-lineage.md](topics/training-run-lineage.md) | [topics/wave-of-lockstep-design.md](topics/wave-of-lockstep-design.md), [topics/wl2-scale-emulation-design.md](topics/wl2-scale-emulation-design.md), [topics/wl5-diagnostics-archive-start-design.md](topics/wl5-diagnostics-archive-start-design.md). |
| **Run/resume the 15×15 training campaign (the live autonomous push)** | [topics/15x15-training-campaign.md](topics/15x15-training-campaign.md) | [topics/15x15-era-feasibility-and-plan.md](topics/15x15-era-feasibility-and-plan.md), [topics/research-lab-charter.md](topics/research-lab-charter.md). |
| Plan or work the 15×15 era (port, feasibility, Gomocup path) | [topics/15x15-era-feasibility-and-plan.md](topics/15x15-era-feasibility-and-plan.md) | [topics/external-engine-baselines.md](topics/external-engine-baselines.md), [sources/gomocup-az-techniques-2026-05-27.md](sources/gomocup-az-techniques-2026-05-27.md), [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md). |
| Work on performance or hardware strategy | [topics/research-lab-charter.md](topics/research-lab-charter.md) | [topics/m5-max-as-mainframe.md](topics/m5-max-as-mainframe.md), [topics/research-lab-session-runbook.md](topics/research-lab-session-runbook.md), [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md), [topics/activity-monitor-perf-runbook.md](topics/activity-monitor-perf-runbook.md), [topics/ane-int8-inference.md](topics/ane-int8-inference.md), [topics/buffer-bit-packing.md](topics/buffer-bit-packing.md). |
| Operate the autonomous research lab | [topics/research-lab-charter.md](topics/research-lab-charter.md) | [topics/conventions.md](topics/conventions.md), [topics/research-lab-reviewer-role.md](topics/research-lab-reviewer-role.md), [ops/gpu-queue.md](ops/gpu-queue.md), [ops/best-cells.md](ops/best-cells.md), [ops/perf-log.md](ops/perf-log.md). |
| **Pick a wild research direction / seed the autolab** | [topics/idea-pile.md](topics/idea-pile.md) | [ops/research-board.md](ops/research-board.md), [topics/rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md), [topics/eval-teacher-sensei.md](topics/eval-teacher-sensei.md). |
| **Why a line-based representation is incomplete — "the claw"** (Jason's 1990s knight's-move defensive crystal = `2x+y≡0 mod 5`; *proven* unique optimal 5-in-a-row blocker AND *proven* invisible to Rapfi's line-organized eval; the defense-axis lever for idea #10 + a concrete #3 judo target) | [topics/the-claw.md](topics/the-claw.md) | [topics/idea-pile.md](topics/idea-pile.md) #10/#3, [topics/external-engine-baselines.md](topics/external-engine-baselines.md). |
| **Rapfi `mix9svq` — exact NNUE architecture** (source-grounded teardown of the engine that made the `~/data/games_raphi` corpus + our native yardstick: a **quantized line-shape CNN**, not classic NNUE. Input = 11-cell ternary line shapes per cell×4 dirs → **VQ codebook** [442503 raw shapes → ≤65536 learned 64-d prototypes, 10-bit packed = the "svq"]; EU accumulator `mapSum[64]` via versioned snapshots [reducer-shaped, undo=`version--`]; 3×3 depthwise-conv trunk → `mapConv[32]`; value head = 3×3 group-pool → **StarBlock** multiplicative gates → MLP → (win,loss,draw); policy head = **hypernet-generated 1×1 conv** → per-cell logit. **The seek-VCT payoff:** the net never confirms a win — **VCF/VCT is pure search, never calls the network** [cheap approx recognition vs exact expensive confirmation, separated by design]; the line-organized codebook is exactly what the claw is blind to) | [topics/rapfi-mix9svq-architecture.md](topics/rapfi-mix9svq-architecture.md) | [topics/the-claw.md](topics/the-claw.md), [topics/vct-recognition-learnability.md](topics/vct-recognition-learnability.md), [topics/seeker-steering-learnability.md](topics/seeker-steering-learnability.md), [topics/external-engine-baselines.md](topics/external-engine-baselines.md). |
| **Discover non-line "molecules" / offensive fields — methods raided from computational genetics** (DCA bond-maps, cryo-EM 2-D class averaging, reciprocal-lattice spectral claw-detector, TF-MoDISco importance motifs, MAP-Elites; the v0 "blocking-task" probe was NEGATIVE because blocking is itself line-shaped — pivot to position-dependent objectives or the training-free detectors) | [topics/molecule-discovery-toolkit.md](topics/molecule-discovery-toolkit.md) | [topics/idea-pile.md](topics/idea-pile.md) #10, [topics/the-claw.md](topics/the-claw.md). |
| **Allis's threat formalism — the gomoku threat theory (reference)** (taxonomy: five/straight-four/four/three/broken-three; the gain/cost/rest-square formalism; dependency vs conflict relations; winning-combination/fork condition; VCF OR-only vs VCT AND/OR; 15×15 freestyle = first-player win, 138,790-node solution tree) | [topics/allis-threat-theory.md](topics/allis-threat-theory.md) | [topics/molecule-discovery-toolkit.md](topics/molecule-discovery-toolkit.md), [topics/idea-pile.md](topics/idea-pile.md) #10, [topics/the-claw.md](topics/the-claw.md). |
| **Standard human gomoku strategy — the "rule of priorities" (external reference)** (GomokuTV video, transcript-distilled: the 5-depth priority ladder — overline/five → four/VCF → three/fukumi/VCT → two/yobi/VC2 → sh-win/positional/"ear-reddening"; the fukumi/yobi/cut/sh-win glossary; the standard line-and-threat view of the game, and explicitly where it ALIGNS with Allis and where the line ladder runs out of names = the molecule ⊋ line foil) | [topics/gomoku-standard-strategy.md](topics/gomoku-standard-strategy.md) | [topics/allis-threat-theory.md](topics/allis-threat-theory.md), [topics/idea-pile.md](topics/idea-pile.md) #10, [topics/the-claw.md](topics/the-claw.md). |
| **GPU/MPS-batched VCT solver** — ⚡ **THE CALL-COST LAW: one call = one _tail_, the wall set by the *single hardest board*, ~flat in batch size (B=16 → 24.6 s, B=16384 → 71.7 s; ~350× per-board swing); compile ~0.1 s, not the cost. Throughput is free, latency is fixed ⇒ every caller must be BULK-SYNCHRONOUS (gather all boards into one ≤16k call, never solve-in-a-loop) and cap `max_nodes`. This flat-in-B wall is what makes million-position VCT/ablation tractable at all — it is the enabling constraint of the whole approach; read the top banner.** Correctness: the §1–7 spike "CPU-bound v0" was **OVERTURNED by §8** — the bitboard `ulong[4]` megakernel `mega_vct_bb` runs the *whole* AND/OR search **on-device**, **~1600× CPU throughput** (~850–1020 solves/s @ B=16k–32k), **0 FP / 0 FN over 320 VCT + 360 VCF real positions**. Read the banner + §8 first; §1–7 are the superseded v0 narrative | [topics/gpu-vct-feasibility.md](topics/gpu-vct-feasibility.md) | [topics/mega-vct-solver.md](topics/mega-vct-solver.md), [topics/shape-library-engine.md](topics/shape-library-engine.md), [topics/vct-backward-mining.md](topics/vct-backward-mining.md), [topics/allis-threat-theory.md](topics/allis-threat-theory.md). |
| **`mega_vct_bb` — the on-device GPU VCT solver (CANONICAL API REFERENCE / the contract)** — the production solver every threat-shape/mining/labeling consumer calls: one GPU thread/board, the whole AND/OR search on-device as a `ulong[4]` bitboard, ~1600× CPU, 0 FP/0 FN over 320 VCT + 360 VCF real positions. **`solve_vct_mega_bb(boards, *, max_nodes, return_move, return_support, complete, return_carriers)`** → default `(win, hit_cap)`; optional outputs append in FIXED order `move`, `support`, `winmask`, `carriers` (existing callers never break, each flag = its own compiled kernel variant, **default verdict byte-identical**). **`move`** = passive sound VCT first move (no extra search). **`support`** (2026-06-27) = the proof's relevance window / stencil seed via return-path accumulation (winning branches only ⇒ no pollution; ⊆ root EMPTY cells = played cells, over-inclusive vs minimal) — the required-OPENINGS, NOT the stones. **`carriers`** (2026-06-27, #88) = the complementary load-bearing OWN stones (root-own collinear-within-4 of any support cell) = the `B` channel; a typed stencil is support ∪ carriers (`.BBBB.` → carriers = the four B, support = the two ends). **`complete`** (2026-06-27, slower) = `winmask` of ALL winning FIRST MOVES (forcing moves only = the VCT first moves; sound + complete, validated 0 unsound / 0 missing — the tempo guard is the subtlety). **`w`** (2026-06-27, #90) = the OPP mirror of carriers (over-inclusive load-bearing DEFENDER stones = the `W` channel). **`max_depth`** (2026-06-28, #91) = a per-board FRAME cap (clean no-win past it, no hit_cap, default byte-identical) → the wrapper **`solve_md_min`** binary-searches it to the **order-independent mate distance** md_min — the §3 ablation enabler (the named blocker, now gone); drives `md_minimize.py`. Plane convention side-to-move-relative `board[0]`=attacker NO swap; free-style; bulk-synchronous only (call-cost law). Test budget `max_nodes=500` ≈ 90% of VCTs, seconds | [topics/mega-vct-solver.md](topics/mega-vct-solver.md) | [topics/gpu-vct-feasibility.md](topics/gpu-vct-feasibility.md), [topics/vct-backward-mining.md](topics/vct-backward-mining.md), [topics/shape-library-engine.md](topics/shape-library-engine.md). |
| **The Shape-Library Engine — the gomoku AI Jason wants to build** (mine the guaranteed-**first-VCT** from real games [forward scan, both colors, trit verdict + defer-on-CAP] → reduce each to a **stencil**: a relative, explicitly-typed [`B`/`.`/`p`], single-orientation, minimal reduction, via **context ablation** on the real legal board — *the all-white "meanest board" certificate FAILED [defender-win → NO-WIN; corrected 2026-06-26], so no monotone-DNF; we ablate legal boards* → a library = a fast **candidate index** matched by structural bitmask fit, with **L0 verifying every match** [L1 proposes, L0 proves ⇒ **never hallucinates** — only ever blind]. **CERTIFICATE PROPERTY MEASURED 2026-06-27 [#88]:** a stencil that wins *in isolation* transfers by the SAME forcing line — **660/660** attacker VCTs self-contained [carriers alone win], **0/2913** tempo-safe placements refute; the *only* breaker is defender **counter-tempo** [= Allis dependency-based-search soundness, now operational on GPU-mined stencils, enabled by the new solver `carriers` output] → a player that finds the min *forcing* path to an **un-blockable FORK** of stencils [two-player pursuit / df-pn, **not** Dijkstra]. **L2 = the AlphaZero layer**: regress the stencil-reachability potential into the fog on *verifiable* targets. Conservative on D4 [translation only for now]; go-all-the-way, negative-result-welcome) | [topics/shape-library-engine.md](topics/shape-library-engine.md) | [topics/vct-backward-mining.md](topics/vct-backward-mining.md), [topics/gpu-vct-feasibility.md](topics/gpu-vct-feasibility.md) §8, [topics/allis-threat-theory.md](topics/allis-threat-theory.md), [topics/molecule-discovery-toolkit.md](topics/molecule-discovery-toolkit.md) + [topics/idea-pile.md](topics/idea-pile.md) #10. |
| **VCT-backward enabling-shape mining — walk won games back to the "first true VCT move"** (stage 1: bank strong Rapfi-vs-Rapfi games fast ~9 g/s; stage 2: walk each won game BACKWARD to the earliest winner-to-move position still a forced VCT win = the *setup*, not the line-bound kill — substrate for non-line molecule discovery. Plane convention VERIFIED no-swap; 0 FP/FN/extra over 258 clean GPU-vs-CPU. CPU depth-10 sweet spot = 184 move-labeled oracle shapes; the **flat-batch GPU** miner — sloppy "solve everything from the back" beats the clever level-walk because the megakernel is tail-bound — hits ~20 g/s, **63k shapes**, run-lengths to **15**. **RESOLVED 2026-06-26 (§5): GPU root-move output** (`solve_vct_mega_bb(return_move=True)`) — passive, no extra nodes; 2.38M forward puzzles move-labeled, 400/400 moves independently verified) | [topics/vct-backward-mining.md](topics/vct-backward-mining.md) | [topics/gpu-vct-feasibility.md](topics/gpu-vct-feasibility.md), [topics/molecule-discovery-toolkit.md](topics/molecule-discovery-toolkit.md), [topics/idea-pile.md](topics/idea-pile.md) #10. |
| **Can a net SEE a VCT? — is-VCT recognition learnability** (feasibility, 2026-06-26: a net classifies "side-to-move has a forced VCT" on **held-out, shard-disjoint** games at **AUROC 0.92+** — real generalization. But for these *local, translation-equivariant* shapes a **CNN (0.971, 168k params) beats attention (0.924, 339k)**, and even **logreg-on-threat-counts (0.946) beats attention** — recognition is easy + count-dominated ⇒ leave it to the exact oracle; **attention's real bet is the SEEKER, not the recognizer**. Labels reused from the miner [absence = no-VCT], no re-solve. The seek-VCT thesis: learn the approximation-tolerant *steering*, solve the approximation-intolerant *forcing finish* [anti-correlated tractability]) | [topics/vct-recognition-learnability.md](topics/vct-recognition-learnability.md) | [topics/seeker-steering-learnability.md](topics/seeker-steering-learnability.md), [topics/gpu-vct-feasibility.md](topics/gpu-vct-feasibility.md) §8, [topics/shape-library-engine.md](topics/shape-library-engine.md), [topics/vct-backward-mining.md](topics/vct-backward-mining.md) §5. |
| **Can a net STEER toward a VCT? — seeker BC learnability** (the seek-VCT thesis's steering half, Phase A, 2026-06-26: a net behaviorally-clones the **quiet-phase (pre-onset) moves of the side that reaches the first forced VCT** and **generalizes to unseen games** — held-out shard-disjoint **top-1 0.386 / top-5 0.696** vs adjacency-to-stones 0.025/0.121 vs random-legal 0.005/0.023 ⇒ steering is **learnable + real**, the cheap green light before the hybrid player. **CNN (224k) beats attention (339k) again** at *next-move* imitation [0.386 vs 0.263] — but attention was **still climbing at the epoch cap** [undertrained, not capped] and this BC proxy is *local*, so it does **NOT** settle attention's *global-receptive-field* bet for sequential seeking. Honest weak proxy: top-1 match ≠ strong play, and conflates seeking with general strength. onset/labels reused from the miner, no re-solve; 500k examples / 38,927 onset games, 0 frame mismatches. **Next = the GPU-spending real tests: Phase B oracle-labeled reachability + Phase C hybrid-play eval vs a fixed baseline [gate w/ Jason]**) | [topics/seeker-steering-learnability.md](topics/seeker-steering-learnability.md) | [topics/vct-reachability-mining.md](topics/vct-reachability-mining.md), [topics/vct-recognition-learnability.md](topics/vct-recognition-learnability.md), [topics/gpu-vct-feasibility.md](topics/gpu-vct-feasibility.md) §8, [topics/shape-library-engine.md](topics/shape-library-engine.md). |
| **Can a net REGRESS the proof-frontier? — Φ distance-to-VCT field learnability** (the trilogy's third leg + **first real L2 model**, 2026-06-27: regress the dual potential Φ — `phi_off=γ^(my-moves-to-my-VCT)` + `phi_def=γ^(opp-moves-to-their-VCT)`, the gradient of which **is** Jason's "which moves move the proof frontier toward mine vs theirs". Held-out shard-disjoint **CNN offense ρ=0.72/R²=0.76/reach-AUROC=0.91, defense ρ=0.76/R²=0.69/0.92** ⇒ the frontier field is **learnable + generalizes**. Two sharp results: **(1) Φ is NOT count-dominated** — CNN nearly *doubles* a ridge-on-board baseline [ρ 0.36→0.72], unlike recognition ⇒ closeness-to-a-fork is genuinely **spatial**, structure lives in *distance* not *presence*; **(2) CNN beats attention a THIRD time** — now param-matched [376k vs 348k] on the **global** target [attention's claimed home turf] with **3× the epochs** [CNN early-stops ep6, attn plateaus ep20] ⇒ the global-receptive-field bet **does not cash out** at this scale. Defense reads *better* than offense [net sees incoming danger best → the white wound]. Verifiable non-bootstrapped target [oracle distances, no self-play value]. **Default L2 arch = the CNN.** Still single-position regression, not whole-game seeking ⇒ strong play = Phase C hybrid eval) | [topics/phi-distance-field-learnability.md](topics/phi-distance-field-learnability.md) | [topics/vct-reachability-mining.md](topics/vct-reachability-mining.md) §1, [topics/seeker-steering-learnability.md](topics/seeker-steering-learnability.md), [topics/shape-library-engine.md](topics/shape-library-engine.md) §4, [topics/white-side-defense-plan.md](topics/white-side-defense-plan.md). |
| **Mining VCT-reachability from the Rapfi corpus — distance field, off-path fan, the knife-edge, non-VCF gold** (2026-06-26: cheap ways to mine seek-VCT steering signal from the 500k Rapfi-v-Rapfi games. **THESIS UPDATE [the headline]:** the pre-onset band we assumed was the net's *forgiving, quiet* region is a **knife-edge** — ~**80% of alternative moves lose by force**, ~99% one ply before onset, ~half even 6 plies out; **both players walk a tightrope**, sharpness ramps *before* the VCT ⇒ the net/solver boundary is fuzzy + earlier than assumed, the net's safe domain is further back than onset−6. **The off-path fan** [ride a game, fan the moves a side didn't play, solve VCT] is a **defense/blunder + VCT-board miner, NEVER an offense detector** [a VCT belongs to the side-to-move; after S moves it's the opponent's VCT] — code-verified [0.000% of fanned nodes are VCT]. **Triviality split via the VCF kernel:** of the 81% fanned VCT-wins, **96.1% are trivial VCF** [four-blocks], only **3.5% are non-VCF VCT** [need a *three* = the combinational **molecules**] — and the gold is almost all the **WINNER's** wins [defender-perturbed; combinations belong to the side with initiative]. **Harvest = perturb the *defender*** → 100k+ non-VCF VCT boards [free GPU] = non-trivial offense termini + hard **defense** lessons **[RAN 2026-06-27: `harvest_molecules.py` banked 146,655 move-labeled gold to `~/data/molecule_gold/` — 99% distinct, sparse [winner mean 6.2 stones], gold grows with distance; 68/400 shards ⇒ ~60× headroom]**. Plus the **free distance-to-VCT field** [from existing verdicts, no re-solve]: terminal-VCT 99%, multi-window 11.6%, offense coverage 49% [an upper-bound, censored target]; proposed Φ=γ^dist potential, offense+defense channels. Banked negatives: both our yield predictions wrong; 81% looks rich but is mostly trivial) | [topics/vct-reachability-mining.md](topics/vct-reachability-mining.md) | [topics/seeker-steering-learnability.md](topics/seeker-steering-learnability.md), [topics/shape-library-engine.md](topics/shape-library-engine.md), [topics/gpu-vct-feasibility.md](topics/gpu-vct-feasibility.md) §8, [topics/white-side-defense-plan.md](topics/white-side-defense-plan.md). |
| **Build/operate the autolab** (the self-driving lab: one out-of-git ledger spine + trainer/arena/research/worker loops; epic #53 — **LIVE 2026-06-19**: ran 6 real 9×9 slices then a full 15×15 lane unattended, crowned the first 9×9 **and** first 15×15 champion, 0 failures; #65 board-size passthrough, #67 arena artifact-ref fix) | [topics/autolab-architecture.md](topics/autolab-architecture.md) | [topics/cockpit-vs-autopilot.md](topics/cockpit-vs-autopilot.md), [topics/research-lab-charter.md](topics/research-lab-charter.md), [topics/m5-max-cross-engine-coupling.md](topics/m5-max-cross-engine-coupling.md). |
| **Run the autolab unattended overnight** (the P5–P7 operating contract: launchd plists, the seed config/cell/cap, the monitor digest + notification, research-lite tick, and the `autolab up`/`down` + attended-PROD-slice-proof runbook) | [topics/autolab-supervisor-and-monitor.md](topics/autolab-supervisor-and-monitor.md) | [topics/autolab-architecture.md](topics/autolab-architecture.md), [topics/cockpit-vs-autopilot.md](topics/cockpit-vs-autopilot.md). |
| Understand cross-cutting conventions (autonomy, merge-commits, memories-to-wiki) | [topics/conventions.md](topics/conventions.md) | [topics/research-lab-charter.md](topics/research-lab-charter.md) for lab-specific rules. |
| Run a perf cell, training slice, or sweep (procedure) | [topics/research-lab-session-runbook.md](topics/research-lab-session-runbook.md) | [ops/perf-log.md](ops/perf-log.md), [ops/experiment-ledger.md](ops/experiment-ledger.md), [ops/baselines.md](ops/baselines.md). |
| Run frontier-lab perf fanout | [ops/status.md](ops/status.md) | [ops/frontier.md](ops/frontier.md), [ops/baselines.md](ops/baselines.md), [ops/experiment-ledger.md](ops/experiment-ledger.md), [ops/test-ledger.md](ops/test-ledger.md), [ops/perf-log.md](ops/perf-log.md). |
| Add or interpret external engine baselines | [topics/external-engine-baselines.md](topics/external-engine-baselines.md) | [sources/gomocup-external-engines-2026-05-22.md](sources/gomocup-external-engines-2026-05-22.md), then `gomoku.match` / `gomoku.eval_worker`. |
| Find/run external gomoku engines to test against (which have open source?) | [topics/gomocup-engines-catalog.md](topics/gomocup-engines-catalog.md) | [topics/external-engine-baselines.md](topics/external-engine-baselines.md). |
| **What's a RELIABLE eval?** (wine engines SHELVED 2026-06-16 — opt-in only, #35; default = net-vs-net pure-torch + pure-python `heuristic`/`lookahead`; native Rapfi anchor = #40/#28) | [topics/reliable-eval-set.md](topics/reliable-eval-set.md) | [topics/gomocup-engines-catalog.md](topics/gomocup-engines-catalog.md), issues #35 / #40 / #28. |
| **Build the calibrated engine-panel derby / eval ladder** (fix the broken-yardstick wound; runner + brain wrapper BUILT — register nets with `incremental=1`; **wine run BROKE calibration (#35); 2026-06-18: native Rapfi-NNUE anchor now ONLINE (#40) + first champion-vs-Rapfi run DONE (eval502 20.8% @5s, n=24) — relative Elo works, absolute calibration still pending #35/#30**) | [topics/engine-panel-derby-design.md](topics/engine-panel-derby-design.md) | [topics/alphazero-lessons-15x15-gomoku.md](topics/alphazero-lessons-15x15-gomoku.md) §8–§14, [topics/external-engine-baselines.md](topics/external-engine-baselines.md), issues #30 / #35 / #40. |
| **Run the always-on Rapfi eval / set up a Rapfi teacher** (the sensei: warm Rapfi pool behind HTTP, CPU-only non-competing; the #34 white-split cadence series; policy-side one-hot Rapfi distillation `--teacher-weight` — the lever up after the self-play knobs were exhausted) | [topics/eval-teacher-sensei.md](topics/eval-teacher-sensei.md) | [topics/swap2-opening-protocol.md](topics/swap2-opening-protocol.md), [topics/external-engine-baselines.md](topics/external-engine-baselines.md), issues #34 / #46 / #18 / #30. |
| **Quantify / fix white-side (defense) weakness** (the #33/#18 "never lose as white" arm — pure-analysis reader over the panel JSONL: `scripts/panel_white_elo.py`; **2026-06-18: real Rapfi-NNUE confirms it — champion white 0/12 vs the #1 engine, the whole shortfall**) | [topics/white-side-defense-plan.md](topics/white-side-defense-plan.md) | [topics/engine-panel-derby-design.md](topics/engine-panel-derby-design.md), [topics/external-engine-baselines.md](topics/external-engine-baselines.md), issues #33 / #18 / #37 / #43 / #49. |
| **Work the swap2 opening protocol (#72 — the REAL white fix)** (white weakness is the first-player-win THEOREM, not a net flaw → delete the doomed role; swap2 rebalances the GAME so self-play data goes ~50/50 and white becomes winnable. **BUILT full Path A (100+ tests), LIVE wandb `8nq1a7cm`; CORE BET CONFIRMED at the data level — white wins 27% in swap2 self-play vs ~0% empty-board. Strength-vs-champion still at parity (H2H 51.6%, early). Gate on H2H-vs-frozen-champion, NOT noisy Rapfi**) | [topics/swap2-opening-protocol.md](topics/swap2-opening-protocol.md) | [topics/white-side-defense-plan.md](topics/white-side-defense-plan.md) banner, [TRAINING_WIKI.md](../TRAINING_WIKI.md) 2026-06-20, issue #72. |
| **Mine Rapfi teacher data at scale / pretrain then warm-start ("Bruce Lee one-position": master idx-2 ONLY)** (`gomoku/rapfimine` — multiprocess flat-file mine, D4-canonical dedup, crash-robust shards; fixed the Rapfi multiPV mate-crash + a thread-per-line perf bug → ~700 moves/s on the M5 Max; then `rapfimine.pretrain` → standard checkpoint → `run_sweep --resume`) | [topics/rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md) | [topics/white-side-defense-plan.md](topics/white-side-defense-plan.md), [topics/eval-teacher-sensei.md](topics/eval-teacher-sensei.md), issues #86 / #46 / #18. |
| Mine or use validation archives | [topics/mining-validation-archives.md](topics/mining-validation-archives.md) | [topics/wl5-diagnostics-archive-start-design.md](topics/wl5-diagnostics-archive-start-design.md) and [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md). |
| Play a checkpoint | [topics/playing-the-model.md](topics/playing-the-model.md) | Latest plateau/run-end notes in [topics/training-run-lineage.md](topics/training-run-lineage.md). |
| Maintain the wiki | [topics/wiki-operating-model.md](topics/wiki-operating-model.md) | [sources/karpathy-llm-wiki.md](sources/karpathy-llm-wiki.md) and [log.md](log.md). |

## Current Synthesis

*(refreshed 2026-06-25)*

The era is now **15×15**, and the question has sharpened from "teach 9×9 to defend"
to: **can we make WHITE (the defender) winnable, and can a net stand against native
Rapfi-NNUE?** The 9×9 work (Z → WL series) is preserved *lineage*, not the live front.
Chronological evidence lives in [TRAINING_WIKI.md](../TRAINING_WIKI.md) (read the
tail); the reusable toolbox is in [capabilities.md](capabilities.md). High-level read:

- **The binding wound is the white-defense gap.** The 15×15 champion ("Bruce",
  128×10) plateaued ~50 Δelo; vs native Rapfi-NNUE @idx-2 it scores **black ~42% /
  white 0/12** — the whole shortfall is white. See
  [topics/white-side-defense-plan.md](topics/white-side-defense-plan.md).
- **The yardstick was broken, now fixed** (#28/#40): native **Rapfi-NNUE** is the
  registered anchor — but **gate "did this help?" on H2H vs the preserved champion,
  not Rapfi** (the Rapfi number is a hint; absolute calibration pending #35/#30). See
  [topics/external-engine-baselines.md](topics/external-engine-baselines.md).
- **Two principled white fixes, both promising-not-proven.** (a) **swap2** rebalances
  the GAME so white becomes winnable — confirmed at the DATA level (white ~27% of
  decisive swap2 self-play vs ~0% empty-board), strength-vs-champion still ~parity
  (early). (b) **fixed-fair-openings** (the live recipe) starts every game from a
  known-fair Rapfi opening, sidestepping the unfair-opener black edge.
  [topics/swap2-opening-protocol.md](topics/swap2-opening-protocol.md),
  [cards/gomoku-15x15-fixed-fair-openings.md](cards/gomoku-15x15-fixed-fair-openings.md).
- **Teacher distillation: one-hot HARMS, soft-target is the fix.** One-hot Rapfi
  distillation flattens the policy head and regresses the net even gentle (#77/#86,
  with a matched control). Soft-target winrate distillation (the designed fix) was
  finally mined at scale + warm-started 2026-06-25 (the idx-2 "one-position" bet);
  the over-specialized net climbs the low rungs but did **not** beat strong Rapfi by
  ep250 — a multi-day climb, banked. Gate teacher runs on **H2H-vs-frozen-parent**.
  [topics/rapfi-idx2-distillation-mine.md](topics/rapfi-idx2-distillation-mine.md),
  [topics/eval-teacher-sensei.md](topics/eval-teacher-sensei.md).
- **Infrastructure matured into reusable capabilities**: mine Rapfi at scale (~700
  moves/s), fast eval-gradient (~20 s; **think-time, not node budget, is the Rapfi
  strength dial**), warm-start from distillation, the self-driving autolab, the uv
  loop. See [capabilities.md](capabilities.md).

Durable lessons (era-independent):

- **Fast-attack collapse** is the main training failure mode: policy sharpens on
  attacks, self-play opponents fail to punish missing defense, fixed
  heuristic/lookahead opponents expose it. Watch `selfplay/plies_mean` — falling +
  concave buffer-fill = collapse.
- **Short evals are noisy** — strength claims need fixed baselines, enough games, and
  explicit checkpoint/run IDs. Sibling H2H is non-transitive; use fixed rulers.
- **Native MCTS + wave-batched eval** moved the self-play bottleneck off Python tree
  churn toward the evaluator/engine boundary; Core ML/ANE is scouted, not yet a win.
  [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md),
  [topics/ane-int8-inference.md](topics/ane-int8-inference.md).
- **Preserve evidence** in every new entry: command, config, W&B run ID, checkpoint
  path, metrics, and the working-theory change.

## Page Catalog

### Model Cards

| Card | Role |
|---|---|
| [cards/gomoku-15x15-fixed-fair-openings.md](cards/gomoku-15x15-fixed-fair-openings.md) | **era-3 / the fairness experiment (IN TRAINING, 2026-06-21):** fresh-init 15×15 that starts every game from one of Rapfi's 9 **known-fair** balanced openings (no negotiation, net plays only post-opening) — sidesteps the unfair opener (#73, [swap2 §10/§11](topics/swap2-opening-protocol.md)). Tests: **does white engage ~50/50 on a fair board?** Run `nbctsiua`. |
| [cards/gomoku-9x9-swap2-era2.md](cards/gomoku-9x9-swap2-era2.md) | **era-2 / Path-A phase 1 (2026-06-20):** the first swap2 net where **white is not doomed** — fresh-init 9×9, aggression `value-discount 0.95`, v2a OFF; white ~45% of decisive self-play games (vs era-1's white-0%-vs-Rapfi basin); phase-1 seed for the 9×9→15×15 warm-start. Run `lywhy1ba`. |

### Core

| Page | Role |
|---|---|
| [AGENTS.md](../AGENTS.md) | Schema for agents: wiki rules, repo map, and working conventions. |
| [TRAINING_WIKI.md](../TRAINING_WIKI.md) | Primary append-oriented training notebook: run history, hypotheses, results, and corrections. |
| [log.md](log.md) | Chronological wiki maintenance log. |
| [topics/wiki-operating-model.md](topics/wiki-operating-model.md) | Gomoku-specific adaptation of the LLM wiki pattern. |
| [topics/training-run-lineage.md](topics/training-run-lineage.md) | Compact route map for the Z and WL-series run sequence. |
| [sources/karpathy-llm-wiki.md](sources/karpathy-llm-wiki.md) | Source record for the LLM wiki charter that inspired this structure. |
| [sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md](sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md) | Transcript of Sid Bidasaria's "Stop babysitting your agents" talk (verification → multi-Claude → background loops). Locally Whisper-transcribed because the video has no captions. |
| [sources/gomocup-az-techniques-2026-05-27.md](sources/gomocup-az-techniques-2026-05-27.md) | Frozen survey (2026-05-27) of AlphaGomoku/KataGo training/search techniques considered for the lab: WDL value head, LCB/variance-PUCT, moves-left, in-search-VCF, SE blocks, and others. The synthesis-and-verdict layer for each lever lives in [ops/research-board.md](ops/research-board.md) ("Open candidates" + the v8/v9 verdicts). |

### Training Dynamics

| Page | Role |
|---|---|
| [topics/az-at-scale-vs-laptop.md](topics/az-at-scale-vs-laptop.md) | Why exploration arcs, plies swings, and age oscillations are laptop-scale artifacts before they are bug evidence. |
| [topics/loss-floor-bouncing.md](topics/loss-floor-bouncing.md) | Why low-floor loss bounces can be healthy in small-scale AZ, and when to suspect a real bug. |

### Run Designs

| Page | Role |
|---|---|
| [topics/wave-of-lockstep-design.md](topics/wave-of-lockstep-design.md) | WL1 design: per-version uniformity via wave-of-lockstep + greedy fill. Now a preserved design record plus WL1 status pointer. |
| [topics/wl2-scale-emulation-design.md](topics/wl2-scale-emulation-design.md) | WL2 design: EMA self-play + past-checkpoint mix + worker poll jitter + grad accumulation. Now a preserved design record plus WL2 status pointer. |
| [topics/wl5-diagnostics-archive-start-design.md](topics/wl5-diagnostics-archive-start-design.md) | WL5 design: fixed validation archive, H/KL decomposition, per-color/ply metrics, and archive-start. Now a preserved design record plus WL5 status pointer. |

### Performance And Hardware

| Page | Role |
|---|---|
| [topics/15x15-era-feasibility-and-plan.md](topics/15x15-era-feasibility-and-plan.md) | 2026-06-12 report: measured board/net scaling on MPS (96×8 @ 15×15 costs only 2.32× at wave=64), week-scale feasibility envelope, phased plan (Rapfi certify → rules decision → port → smoke → first run → derby). |
| [topics/mcts-perf-ceiling.md](topics/mcts-perf-ceiling.md) | Where MCTS gen-time wins are and are not. Do not re-port "v2 storage"; we are already there. |
| [topics/activity-monitor-perf-runbook.md](topics/activity-monitor-perf-runbook.md) | Practical knobs and interpretation rules for Mac Activity Monitor perf experiments. |
| [topics/m5-max-as-mainframe.md](topics/m5-max-as-mainframe.md) | Guiding philosophy and sequence for the post-WL5 perf era on Jason's M5 Max. |
| [topics/ane-int8-inference.md](topics/ane-int8-inference.md) | Engine-isolation plan and first scout for Core ML / ANE / CPU lanes around MPS training. |
| [topics/coreml-ane-residency-lab.md](topics/coreml-ane-residency-lab.md) | Rail-proof lab for Core ML / ANE residency claims; caps `CPU_AND_NE` label checks below ANE-backed unless powermetrics shows nonzero ANE rail. |
| [topics/buffer-bit-packing.md](topics/buffer-bit-packing.md) | Post-WL5 replay-buffer compression plan: bit-packed planes plus FP16 policy, with cheap-test gate. |
| [topics/conventions.md](topics/conventions.md) | Cross-cutting project conventions: deny-list autonomy (Class A/B/C), merge-commits-never-rebase, memories-also-go-to-wiki, Opus-minutes-not-human-days. Source of truth mirrored in memory. |
| [topics/workflow-orchestration.md](topics/workflow-orchestration.md) | How the Claude Code *Workflow* feature maps onto the lab: deterministic agent-chaining (not the `/loop` looper) = the everything-else lane of the two-queue scheduler, never the GPU lane. Fit/misfit table, the cockpit/verify-gate why, and the first real workflow (`.claude/workflows/reviewer-gated-fanout.js`). Includes the **workflow-master** session discipline (run/tune workflows, stay lean). |
| [topics/cockpit-vs-autopilot.md](topics/cockpit-vs-autopilot.md) | The lab's operating lens (synthesis of Sid Bidasaria's "stop babysitting your agents" + Jason's framing): **autopilot** = machinery that runs without you; **cockpit** = the thin attention layer (trustworthy gate · one-glance status surface · escalation line) that makes it supervisable. More autopilot without cockpit = more to babysit. The judging question for new lab infra. |
| [topics/research-lab-charter.md](topics/research-lab-charter.md) | Charter for the autonomous research lab: mission, two research areas (perf + training-recipe), GPU-required vs everything-else queues, training-slice protocol, smoke-first doctrine, operating loop, priority function, tier system, reviewer gate, autonomy boundaries, worktree discipline, stop conditions. |
| [topics/autolab-architecture.md](topics/autolab-architecture.md) | **The autolab (epic #53) — LIVE 2026-06-19:** the self-driving lab as one out-of-git append-only ledger spine (`gomoku/lab/{ledger,daemon,trainer,arena}.py`; `~/data/autolab/` home) read by same-shape loops — trainer (1-epoch→1h singleton slices → HuggingFace per-slice revisions), arena (gate vs the HF champion tag, co-tenancy guard), research, worker. Ran 6 real 9×9 slices then a full 15×15 lane unattended; crowned the first 9×9 + first 15×15 champion (#65 board-size passthrough, #67 arena artifact-ref fix). No-claim flock singleton, the code-shape contract, the measured M5 co-tenancy envelope, the 15×15 capability, the **arena-yardstick gap** + Rapfi-readout plan, the cockpit overlay, the phased plan. Supersedes the #2 three-tier framing. |
| [topics/wall-clock-to-elo-metric.md](topics/wall-clock-to-elo-metric.md) | LF1-followup #4 design: wall-clock-to-elo as a first-class metric family (MTTE primary, EPWH/Δelo·Δt⁻¹ secondary) the throughput proxies must be checked against; protocol, val/policy_ce gate, gap analysis vs `delta_e_harness.py`, proposed charter diff. |
| [topics/research-lab-reviewer-role.md](topics/research-lab-reviewer-role.md) | Codified Reviewer role: when it fires (post-lane + mid-loop), the audit prompts, the three verdicts (APPROVE/REVISE/BLOCK), and what it does NOT do. |
| [topics/research-lab-session-runbook.md](topics/research-lab-session-runbook.md) | End-to-end procedure for running a GPU-required lab item (perf cell, training slice, or sweep): pre-flight, naming, command surfaces, receipt, surfaces to update. |
| [topics/probe-100pct.md](topics/probe-100pct.md) | `scripts/probe_100pct.py` — one-command driver for the RESUME PLAYBOOK step 1 sweep (eval-sims × eval-VCF vs lookahead4) on a matured checkpoint; per-cell distance-to-100% via the existing `report_100pct.py` formula. |

### Operations And Use

| Page | Role |
|---|---|
| [topics/launch-sequence-runbook.md](topics/launch-sequence-runbook.md) | Reusable playbook for launching, smoking, monitoring, and ending training runs. |
| [topics/mining-validation-archives.md](topics/mining-validation-archives.md) | Operational recipe for `scripts/mine_validation_archive.py`: buckets, knobs, throughput, anti-patterns. |
| [topics/playing-the-model.md](topics/playing-the-model.md) | How to play a trained checkpoint through the local web UI or live SPA. |
| [topics/external-engine-baselines.md](topics/external-engine-baselines.md) | Rated OSS/source-available Gomocup engine candidates and the Piskvork wrapper plan. |
| [topics/containerize-training-runs.md](topics/containerize-training-runs.md) | **Backlog (for soon):** containerize the training run, one container at a time, refine the skill for lower startup friction/time. Open question: no Metal/MPS in Docker on macOS — targets off-Mac/at-scale or a non-Docker run unit. |
| [../scripts/wandb_workspace.py](../scripts/wandb_workspace.py) | Creates W&B workspaces for run overlays. Regenerate when a new run joins the comparison set. |

### Frontier Lab Ops

| Page | Role |
|---|---|
| [ops/status.md](ops/status.md) | Current ML performance frontier control-room summary. |
| [ops/frontier.md](ops/frontier.md) | Human-readable board projected from `.frontier/lanes.json`. |
| [ops/baselines.md](ops/baselines.md) | Benchmark command surfaces and reference results. |
| [ops/experiment-ledger.md](ops/experiment-ledger.md) | Receipt ledger for promote/reject/block decisions. |
| [ops/test-ledger.md](ops/test-ledger.md) | Validation command ledger for frontier decisions. |
| [ops/perf-log.md](ops/perf-log.md) | Day-by-day narrative timeline for the M5 Max perf era. |
| [ops/gpu-queue.md](ops/gpu-queue.md) | Live, ordered queue for GPU-required lab items (perf cells and training slices). Source of truth for the autonomous lab loop. |
| [ops/best-cells.md](ops/best-cells.md) | Current best cell per quality reference point; promotion log. |
| [topics/fleet-management.md](topics/fleet-management.md) | The agent-management toolchain north star (roadmap): land the too-many-sessions problem; log-based, append-only; cockpit not autopilot. |

## Layers

- **Evidence sources**: W&B histories, local logs, checkpoint files, match
  outputs, scripts, raw command output, and external source records under
  [sources/](sources/).
- **Maintained synthesis**: this index, topic pages under [topics/](topics/),
  and the training notebook.
- **Schema**: [AGENTS.md](../AGENTS.md), which tells future sessions how to
  maintain and use the wiki.

## Maintenance Rules

- Read this index first, then drill into the pages it names. For a capability-first
  overview (vs the task-first table above), read [capabilities.md](capabilities.md).
- **Close the spine on every significant run**: append a dated `TRAINING_WIKI.md`
  entry AND update [capabilities.md](capabilities.md) if a capability changed — a
  topic page + index row alone is not discoverable via the chronological read path.
- Keep source records and artifacts immutable unless the user explicitly asks
  for cleanup.
- Keep the training notebook append-oriented. When a conclusion changes, add a
  dated correction with evidence instead of polishing the old entry.
- File useful answers back into the wiki when they would save a future session
  from recomputing the same synthesis.
- Update [log.md](log.md) whenever the wiki structure, index, or synthesis pages
  change in a meaningful way.
