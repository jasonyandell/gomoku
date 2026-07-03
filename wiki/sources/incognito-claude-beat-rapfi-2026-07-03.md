# Source: incognito Claude on "how to beat Rapfi" (2026-07-03)

**Provenance:** Jason asked a fresh, incognito Claude chat (no project context,
no wiki, no memory) how to beat Rapfi, over lunch 2026-07-03. Recorded verbatim
below. **Caveat on independence:** the lab's resident Claude and the incognito
one are the same model — agreement between them is the same prior sampled
twice, NOT independent confirmation. Treat convergence as "the model finds
this line attractive," not "two experts agree."

**Reaction / synthesis:** see [legomoku.md](../topics/legomoku.md) and the
session notes of 2026-07-03; actionable extractions filed where noted.

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
