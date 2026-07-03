# Source: incognito Claude on "how to beat Rapfi" (2026-07-03)

**Provenance:** Jason asked a fresh, incognito Claude chat (no project context,
no wiki, no memory) how to beat Rapfi, over lunch 2026-07-03. Recorded verbatim
below. **Caveat on independence:** the lab's resident Claude and the incognito
one are the same model — agreement between them is the same prior sampled
twice, NOT independent confirmation. Treat convergence as "the model finds
this line attractive," not "two experts agree."

**Reaction / synthesis:** resident-Claude reaction appended below the verbatim
(same day); ladder/feedback context in [legomoku.md](../topics/legomoku.md).

---

The trick is to notice the asymmetry you already spotted: Rapfi is a
masterpiece of playing well in 5 seconds and 350MB. You have unlimited
compute. So don't build a better player — build something that makes
"playing" the wrong frame entirely. Three layers:

**Layer 1: Solve, don't evaluate.** Freestyle gomoku is a proven first-player
win (Allis, 1993, via threat-space search), and the game is brutally tactical
— forced wins via continuous fours and threats (VCF/VCT) are everywhere.
Gomocup games start from a fixed pool of balanced openings, which is a gift:
the opening set is finite and known in advance. So spend GPU-years offline
running neural-guided proof-number search (df-pn) on every tournament opening,
the way checkers was solved — forward search meeting terminal proofs. For many
openings you get an actual proof tree: a mathematical certificate of win or
draw. Against a proof tree, Rapfi's tuning is irrelevant; it's playing poker
against someone who can see the deck. The "engine" at match time is partly
just a lookup.

**Layer 2: A KataGo-scale learned prior for everything the proofs don't
cover.** Where positions aren't solved, run the modern recipe at absurd scale:
AlphaZero-style self-play with KataGo's efficiency tricks (auxiliary
ownership/score targets, playout cap randomization), Gumbel-style search for
sample efficiency, a network 100–1000x bigger than anything Gomocup hardware
could dream of, and at play time an MCTS with millions of playouts per move
plus a dedicated VCF/VCT solver thread running in parallel so no tactic ever
slips through. This alone likely beats Rapfi-at-Gomocup — Katagomo-style
engines with big nets and long thinks already outclass alpha-beta engines
given hardware parity, and you have a hardware chasm. But it's layer two
because raw strength is the boring win.

**Layer 3: Exploit Rapfi specifically.** Rapfi is open source and
deterministic enough to model. Download it, run it at exact Gomocup settings
as an oracle, and query it hundreds of millions of times. Now you can:

- Mine its blind spots: search for positions where your deep analysis says one
  thing and Rapfi-at-5-seconds says another. Those disagreement positions are
  a map of its horizon effects and NNUE eval biases.
- Train an exploiter: population-based training where one adversary in the
  league is literally Rapfi. You're not learning gomoku anymore, you're
  learning the function "Rapfi's mistakes."
- Steer, don't just play: when your proof tree says a position is drawn or
  lost, switch objective from "best move" to "move maximizing the probability
  Rapfi errs," estimated from your opponent model. This is the
  Chinook-vs-Tinsley insight — perfect play and maximally winning play against
  a specific bounded opponent are different things.

The philosophical shift that makes this "think different": every component
treats uncertainty as something to eliminate rather than estimate. Search
until proven, and where you can't prove, model the opponent's specific bounded
rationality instead of assuming a perfect adversary. Rapfi is optimized for
the minimax world; you'd be playing a different game — part number theory,
part psychology, with a superhuman player as mere fallback.

Honest caveats: 20x20 freestyle proofs are vastly harder than 15x15, so full
solves of every opening may not land even with big compute — but partial proof
trees (all VCF/VCT lines certified, deep df-pn frontiers) still convert
directly into unbeatable tactics. And Rapfi with a Yixin-class opening book
has prep of its own — yours is just deeper by orders of magnitude, because you
knew the opening pool and had no clock.

---

## Resident-Claude reaction (2026-07-03, same day — with full lab context)

*The verbatim above is the raw source; everything below is synthesis by the
in-lab Claude. Same independence caveat applies in reverse.*

### The convergence is the headline (discounted appropriately)

With zero context — no wiki, no memory, no project — the incognito instance
landed on the house thesis: "solve, don't evaluate," a dedicated VCF/VCT
solver thread beside the net, uncertainty as something to *eliminate* rather
than estimate. Layer 1 is the seek-VCT program, reinvented blind. Either the
thesis is attractive from first principles, or the model family shares a bias.
Hold both.

### Scorecard against the lab

- **Layer 1 (solve the openings): already built, one upgrade suggested.** The
  mega-VCT solver *is* partial proof trees / certified tactics; "the opening
  pool is known in advance" is trivially true here (pool = idx-2, size one),
  and the parked "does black have an in-horizon win from idx-2" probe is
  precisely a Layer-1 act. The genuine upgrade: **neural-guided df-pn** as the
  general-purpose prover beyond threat-space (see primer below).
- **Layer 2 (KataGo scale): premise fails locally.** It assumes a hardware
  chasm; this lab is one M5 Max — at our scale *Rapfi* is the efficient one.
  Strip the scale and the steal-worthy residue is the **sample-efficiency
  kit**: KataGo auxiliary targets, playout-cap randomization, Gumbel search.
  All laptop-sized, all serve Δelo/Δt directly. Most actionable paragraph in
  the source.
- **Layer 3 (exploit Rapfi): half built, one new idea, one landmine.** The
  Rapfi-as-oracle infrastructure exists (pool, sensei, distillation mine).
  **New: disagreement mining** — and locally it's *better* than described,
  because we have a prover: positions where mega-VCT certifies a win that
  Rapfi-at-budget misplays are **certified blind spots**, zero ambiguity.
  **Landmine: the exploiter league / Chinook-style steering.** Rapfi is our
  measuring stick; training against the anchor corrupts the instrument
  (reliable-eval doctrine). If that thread ever runs, a held-out anchor comes
  first.

### The disagreement mine is a molecule detector (LeGomoku tie-in)

Rapfi's perception is line-organized (mix9svq teardown), and the claw proved
real "molecules" can be *invisible to line-organized eval*. Therefore Rapfi's
certified blind spots are exactly where molecules live. Disagreement mining
isn't just an exploit map — it's a **claw-detector at scale**, the instrument
the molecule-toolkit v0 lacked. Stated as a shape prior: hunt where
certificate-distance and eval disagree.

### The deepest miss: no feedback

The three layers are one-directional — proofs flow down into play, full stop.
But neural-guided df-pn *is* a loop (net guides prover, certificates train
net), and the disagreement mine *is* a return channel (proofs correcting the
learned prior). Wired into a cycle, the layers become the "chemistry" of the
[legomoku.md](../topics/legomoku.md) ladder. Per the ceiling argument recorded
there: the source's program minus feedback approaches Rapfi; only the loop
version has a shot at exceeding it. (Pre-stated framing, not a claim.)

### Primer: proof-number search / df-pn (the fun learning)

Allis again — PNS is from his thesis, same one as the gomoku proof. Minimax
asks "how *good* is this position?"; PNS asks "how close to *settled*?" Every
node carries a **proof number** (fewest leaves that must still go my way to
prove a win) and a **disproof number** (fewest to refute). The only rule:
expand the *most-proving node* — descend wherever the tree is thinnest, toward
the cheapest certificate. No eval function in the pure form. Why it eats
tactical games: a forcing move leaves few legal replies → AND-nodes stay
narrow → proof numbers stay tiny → the search *autonomously dives down forcing
lines*; nobody told it forcing moves matter, the arithmetic flows downhill
into them. **df-pn** (Nagai 2002) is the depth-first reformulation: recurse
with pn/dn thresholds ("stay down here until you exceed these bounds") + a
transposition table — same search, a fraction of the memory. It solved
checkers. Resonance: the pn/dn pair is a *quantified* "well, who friggin
knows," and the search's whole job is driving it to zero where that's
cheapest. The mega-VCT solver is a specialized cousin (threat-space ≈ PNS with
the move menu pre-restricted to threats); df-pn is the general instrument.

### What shape would the resident sniff for? (asked directly, answered on record)

One shape, three faces:

1. **Provably hot, perceptually cold.** Rapfi's blindness has a *geometry*:
   it cannot smell value living *between* lines — conjunctions whose parts are
   individually innocuous (the claw is the existence proof). Measurable
   signature: proof number collapsing while NNUE eval sits flat. We can
   compute both gradients — mine the divergence.
2. **Wins with a quiet prefix.** Alpha-beta at 5 s lives on selectivity;
   quiet moves get pruned hard. A forced win whose first 2–3 moves are
   non-forcing preparations (cascade starts at move 4) sits beyond the
   selective horizon regardless of nominal depth. Certify those and you have
   moves that look like positional genius and are actually receipts.
3. **The meta-shape: bias–truth gaps.** Every engine *is* its inductive bias;
   Rapfi is lines + incrementality + selectivity. You don't out-play a
   masterpiece at its own prior — you find where its prior stops paying rent
   and move the game there.

### On the gaps Jason named (no shapes, no feedback, no new ML)

A provenance observation, not a criticism: asked "how do I beat X," the model
retrieves the canonical playbook — solve, scale, exploit. All excellent, all
*known*. Shapes, feedback loops, and genuinely new learning mechanisms aren't
in the answer because they aren't in any playbook to retrieve — they're the
part that would have to be *discovered*. There's also a question-shaped
component: "beat Rapfi" pulls the competitive frame; "what does Rapfi fail to
understand?" would likely pull the science frame and a shapes-adjacent answer.
The gap, named without prejudging its value (Jason's framing): the source
answered with engineering; the molecule ladder is a bet that there's *science*
left in this game.
