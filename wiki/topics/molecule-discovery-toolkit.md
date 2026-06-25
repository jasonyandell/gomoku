# Molecule-discovery toolkit — what we steal from computational genetics

Methods raided from computational genetics / bioinformatics (+ cryo-EM, crystallography,
evolutionary ML) to **discover non-line "molecules" — including unknown "offensive fields" —** in
gomoku. The actionable companion to [idea-pile.md](idea-pile.md) **#10** (*molecule ⊋ line; represent
relations, let lines emerge; the rep as a discovery engine*) and [the-claw.md](the-claw.md) (the
canonical non-line molecule). Scouted 2026-06-25 (Jason's "those folks have incredible tools" nose);
research-agent credited. Citations inline.

## The killer reframe (why these transfer *better* to us than to biology)
A 9×9/15×15 board is **already a fixed-size grid pre-registered to a common coordinate frame.** The
step that dominates and bedevils genomics — multiple sequence alignment (MSA) — is **free** for us:
cell *k* is permanently "column *k*." And the latent "pose" that cryo-EM fights (continuous
SO(2)×ℝ²) is, for us, the **finite D4 group (8 elements)**. So several bio tools are *easier* here
than at home. The one tax we pay that biology doesn't: **translation/D4 invariance** (a fork is a fork
anywhere) — fix by re-centering on a local frame (last move / threat gain-square) and/or D4-augment.

## The 5 highest-value steals (ranked by insight-per-compute on the M5)

### #3 — Spectral / reciprocal-lattice claw detector ★★★★★ (cheapest; seconds, no training)
Genomics finds periodicity (codon 3-period, ~10.5bp helical) by Fourier of an indicator encoding
(**Voss representation**), no trained model; **Stoffer's spectral envelope** (Biometrika 80:611, 1993)
adds a significance threshold. **2-D upgrade = the reciprocal lattice (crystallography):** a periodic
point-lattice's Fourier transform is itself a sharp lattice of **Bragg peaks**. The claw
`L={2x+y≡0 mod 5}` (sublattice of ℤ², index 5) has its signature at normalized 2-D frequency
**(f_x,f_y)=(2/5, 1/5)** (the plane-wave e^{2πi(2x+y)/5}) + reciprocal partners. **Detector:**
`P=|FFT2(defender-occupancy)|²`, read the bin nearest (2/5,1/5). A *line* lights up the axis/diagonal
frequencies — **different bins** — which is *exactly why the claw is invisible to line engines but loud
to a 2-D spectral detector.* Caveats: 9 isn't a multiple of 5 → zero-pad to 45 / window / Goertzel for
the off-grid bin; most positions won't carry the frequency (hunt the enriched few); use a permutation
null (shuffle colors ×200, z-score). *Seed (seconds):* `claw_score(board)` on a synthesized perfect
claw as positive control, then random density-1/5 boards and lines-of-5 as negatives, then real
defensive positions + an FFT2 of a learned value/threat field.

### #1 — Direct-Coupling Analysis (DCA) ★★★★★ (training-free "bond map" of cells)
Fit a **global max-entropy Potts/MRF** over board cells from a corpus of high-value positions; read off
**direct couplings J_ij**. The decisive trick (Morcos et al. 2011, PNAS 108:E1293): it **disentangles
direct from indirect correlation** — a completed five induces a blizzard of indirect pairwise
correlations that raw mutual information lights up; DCA collapses the line to its chain of direct bonds
and **leaves the genuinely off-axis bonds as residual couplings.** That residual is the non-line
bond map: the claw's knight partner, the split-two, the broken-three gap-cell — high coupling, on no
single line. (This coevolution→contact-map pipeline is what AlphaFold2's Evoformer absorbed.) Transfers
*unusually well* (no MSA needed); handle invariance via relative-offset coords J(Δr,Δc,s_i,s_j) = a
**shift-invariant convolutional MRF** (the bridge straight to a CNN/attention edge predictor). Impls to
copy: mfDCA (fast), plmDCA/CCMpred/GREMLIN (pseudolikelihood, GPU-friendly), `pydca`. *Seed A (hours,
no net):* MI + mean-field DCA on 10–50k self-play states → plot top non-adjacent, non-collinear
couplings. Falsifiable: strong known shapes reproduce; **strong-and-unnamed = candidate species.**

### #2 — Cryo-EM 2-D class averaging ★★★★★ (unsupervised template dictionary)
Cryo-EM distills crisp 2-D **class averages** from tens of thousands of noisy, pose-varied projections
via EM, *zero labels* (Scheres RELION/ML2D, J. Struct. Biol. 180:519, 2012). For us this is *literally*
"find a small set of canonical 2-D templates from many noisy instances up to translation + the **finite,
tiny D4 pose group** (8-way argmax, not a continuous search)." Corpus = local patches around high-value
events (top moves; or the loser's last-N positions to hunt collapse-motifs). E-step = assign each patch
to its best template under best-of-8 D4; M-step = D4-aligned mean; ~30 iters → a dictionary of soft
probabilistic stencils, **the place non-line offensive *fields* fall out as their own classes.** Most
honestly-2-D tool on the list. Mandatory mitigations (cryo-EM's own): multiple restarts; keep only
templates reproducible across two independent corpus halves (the **"Einstein-from-noise"** trap —
averaging noise manufactures plausible templates). *Seed (~150 LOC, 1–2 days):* k≈32 templates over
~50k 7×7 / 9×9 one-hot patches; render as information-weighted opacity heatmaps; eyeball for fields.

### #4 — TF-MoDISco + the DeepBind conv-filter≡PWM bridge ★★★★ (value-grounded molecules)
DeepBind (Alipanahi 2015, Nat. Biotech. 33:825): **a CNN's first-layer conv filter IS a PWM** — recover
it by averaging its top-activating patches (our length-5 detectors are already PWMs over cells; this is
the read-out recipe). The deeper steal, **TF-MoDISco** (Shrikumar 2018, arXiv:1811.00416): discover
motifs in **importance space, not data space** — attribute the **value head** back onto the board
(DeepLIFT/integrated-gradients via Captum), threshold high-importance "seqlets" into 2-D patches,
cluster (under D4, = #2's engine) into a non-redundant vocabulary. Result: **value-grounded** molecules
— provably the chunks the net *uses* to judge the position, not merely frequent ones. "Discover in
importance space" is the single most transferable idea here, and the field's own caveat is on-point:
first-layer filters under-report exactly the **combinatorial non-line** structure (it lives in deeper
layers) — so filter-reading is the warm-up, importance-clustering is the real tool. *Seed (1 day atop
#2):* IG-attribute the value head on ~2k wins → seqlet patches → #2 D4 k-means. Motifs in
importance-but-not-frequency space = **valuable-but-rare** (the interesting ones).

### #5 — MAP-Elites + Novelty Search ★★★★ (illuminate species we can't name)
*Illumination*, not optimization. **Novelty Search** (Lehman & Stanley 2011) rewards behavioral novelty
alone — and beats objective-driven search on **deceptive** problems (ours is deceptive: pure
win-prediction over-rewards known line threats and starves weird non-line forks). **MAP-Elites** (Mouret
& Clune 2015, arXiv:1504.04909) keeps the best solution per cell of a **behavior-descriptor space** →
a whole archive of diverse high-performers = **a periodic table of offensive fields.** Evolve 2-D
typed-cell stencils (require-black / -white / -empty / don't-care + weight); fitness = correlation of
firing with state-value; **descriptors = (bounding-box area, #don't-care holes, off-axis/non-collinearity
score, color-mix)** — the non-collinearity axis (our domain knowledge) *forces* the archive to populate
non-line regions. Fully transferable (domain-agnostic evo-ML; the genetics framing is the bitter-lesson
connection, not a dependency). Ref impl: `pymap_elites`. *Seed:* a few thousand CPU evals; each filled
archive cell = a candidate species.

**Honorable mention — network-motif discovery on the threat graph ★★★** (Milo 2002, Science 298:824;
tool **FANMOD**, supports typed nodes/edges). On our threat reaction-graph (nodes = threats w/ gain+cost
squares, edges = shared-square interactions), census colored size-3/4 subgraphs vs a degree-matched null:
**the fork = a 3-node shared-gain-square motif; a VCF chain = the feed-forward-loop analog.** Lower only
because it presupposes the threat graph is already extracted — it *names* known tactical molecules more
than it *discovers* non-line fields. (GNN message-passing = the learned version.)

## Honest caveat — Reaction–Diffusion / Turing is a LENS, not a mechanism (correction)
Earlier I (Claude) got excited that "chemistry + Fourier = Turing reaction–diffusion explains the claw."
**The raid checked it rigorously and that's wrong — own it.** RD captures the *shape* of
local-reinforcement + lateral-inhibition and nothing more. Gomoku is discrete, adversarial, turn-based,
finite: **no diffusion constant → no diffusion ratio → the central Turing knob that selects a wavelength
k\* does not exist**, and the "inhibitor" is an optimizing minimax antagonist, not a passive
faster-diffusing chemical. Decisively, **the claw's period-5 is number-theoretic, not dynamical** — it
falls out of run-length 5 + the coprimality of all four line-direction steps mod 5 (a covering-congruence
argument) and would not budge if you changed any diffusion-like quantity. **The honest tool that explains
the 5 is the reciprocal-lattice math (#3), not RD.** Salvage: diffusion is real for **Go-style
influence/territory fields** (Bouzy's mathematical morphology, CG2002; Go-as-Ising arXiv:1710.07360) —
usable only as a **pre-smoothing front-end** that turns sparse stones into a continuous field for cleaner
spectral peak-hunting (#3); and "sweep a diffusion ratio and watch the peak NOT move" is itself clean
evidence the claw is arithmetic, not Turing. *The chemistry/reactivity lens (activator–inhibitor,
catalysts, quenching) stays useful as intuition + maps to influence fields; it just doesn't predict the 5.*

## Other honest caveats
- **Einstein-from-noise (#2/#4):** clustering/averaging noisy patches manufactures plausible templates →
  multiple restarts + cross-half reproducibility, always.
- **Significance, always (#1/#3):** small-n + finite boards make spurious peaks/couplings easy →
  permutation nulls (#1/#3), degree-matched random-graph nulls (FANMOD) before believing anything.
- **First-layer-filter ceiling (#4):** combinatorial non-line fields live in deeper layers; read
  importance, not filters.

## Skip / overrated for us
- **Smith-Waterman / BLAST / PSI-BLAST** — fundamentally 1-D-ordered with indels; a board has no linear
  order, "insert a cell" is meaningless. Useful crumb (fuzzy template-match score) is just 2-D
  correlation, already in #1/#5.
- **Profile HMMs (HMMER) at board level** — a linear Markov backbone can't carry off-chain far
  dependencies (= the whole non-line thesis). Narrow salvage: along a *single ray* it's a principled
  variable-length *line*-threat recognizer (open/broken/split three) — a baseline component, the opposite
  of what we hunt.
- **MEME / Gibbs / sequence-logos as code** — don't port (their 1-D sliding-window blows up to a 2-D
  anchor + D4 search); borrow their **scoring math** instead: information-content-in-bits (Schneider &
  Stephens 1990; max log₂3 ≈ 1.58 bits for our 3-symbol alphabet) for per-cell conservation/degeneracy +
  rendering, and MEME's ZOOPS/OOPS occupancy models ("does this board contain 0/1/several copies of
  molecule X?"). Run the EM skeleton as the 2-D cryo-EM variant (#2).

## Recommended order of attack (all M5-cheap; none touch the GPU training lane)
1. **#3 spectral claw-detector** (seconds, no training) — validate the frequency-domain thesis with a
   synthesized-claw positive control before any corpus work.
2. **#1 Seed A: MI + mean-field DCA bond map** (hours, no training) — fastest "do non-line bonds exist,
   and which?" Highest insight-per-compute.
3. **#2 board class-averaging** (1–2 days) — the unsupervised template dictionary; eyeball for fields.
4. **#4 value-grounded molecules** (1 day; reuses #2's clustering + a checkpoint) — rank the dictionary
   by what the net actually uses.
5. **#5 MAP-Elites / #1 Seed B Potts kernel / FANMOD census** — enumerate & name species once bonds are
   believed.

## Key sources
Morcos 2011 DCA (PNAS 108:E1293); Ekeberg plmDCA (arXiv:1401.4832); CCMpred (PMC4201158); pydca.
Scheres RELION/ML2D 2012 (J. Struct. Biol. 180:519). Voss 3-base periodicity; Stoffer spectral envelope
1993 (Biometrika 80:611); reciprocal lattice (IUCr). Alipanahi DeepBind 2015 (Nat. Biotech. 33:825);
Kelley Basset 2016; Shrikumar TF-MoDISco 2018 (arXiv:1811.00416); Captum. Lehman & Stanley 2011 Novelty;
Mouret & Clune 2015 MAP-Elites (arXiv:1504.04909); pymap_elites. Milo 2002 network motifs (Science
298:824); FANMOD (Wernicke 2006). Bouzy morphology CG2002; Go-as-Ising arXiv:1710.07360; Schneider &
Stephens 1990 (info content).
