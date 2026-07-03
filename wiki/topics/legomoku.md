# LeGomoku — latent-space world models for a rule-following game

**Status: brainstorm** (2026-07-03, Jason + Claude, no code yet). An experiment
hub-in-waiting — expected to grow child pages if/when spikes run. **No strong
claims anywhere on this page; predictions are pre-stated bets, not conclusions.**
Named goal is double: (a) see if a world model can make a *better search*, and
(b) it's a totally-new-to-Jason thing to learn — the learning is a first-class
deliverable, so a splat that's written down is a win.

> Hub: [Experiments](../experiments.md) · siblings: [seek-VCT](../seek-vct.md) ·
> related: [sound-world-recipe.md](sound-world-recipe.md),
> [mega-vct-solver.md](mega-vct-solver.md), [allis-threat-theory.md](allis-threat-theory.md)

## The idea

Reasoning in latent space about a thing that follows rules. Not "predict the
next board" — predict the next *representation*, JEPA-style: let the model learn
what about the future is worth predicting, then search over that instead of
over moves.

## The weak version (pre-killed)

A learned simulator of gomoku dynamics is a worse copy of `game.py` — the
transition function is free, deterministic, and perfect. MuZero is the
precedent: it matched AlphaZero on Go while *learning* the dynamics, a stunning
result about learning that bought ~zero search improvement where rules are
known. "MuZero for gomoku" spends compute re-deriving stone placement. Not the
experiment.

## The strong version (the experiment)

Gomoku is an unusually good sandbox for latent-space search because **the right
abstraction is already known — Allis hand-built it**. Threat-space search *is*
a world model: collapse the board into threats, search over forcing sequences
instead of moves. It's why the GPU VCT oracle demolishes move-space MCTS in its
domain. So the crisp question:

> Can a net *learn* the threat-space abstraction it was never told, and does
> search in that learned latent beat move-space MCTS at a fixed node budget?

Preferred first target is **temporal** abstraction over spatial: a latent model
that jumps a whole forcing exchange in one step (state + "launch this threat" →
representation after the forced reply sequence). Search over macro-moves =
fewer, deeper nodes — aimed at vanilla MCTS's actual weakness here (burning
simulations ply-by-ply through sequences the threat structure already decided).

## The prior: the Texas 42 splat

Jason ran a by-the-books world model (initial paper + improv) on Texas 42 —
his first love, infinity-times more meaningful than gomoku (literal grandma's
knee) — and it went splat on a representational wall: **no encodable
representation of "well, who friggin knows"**. These models have a terrible
time with non-geometrical things — things that happen *behind their back*,
i.e. imperfect information.

The gomoku connection is real but not identical: gomoku's information is
perfect, but there's a **fog of war we currently believe you only penetrate by
more, better search** — the value of a position hides behind combinatorial
depth, not hidden state. If deep-search truths are as "non-geometrical" to a
latent model as hidden cards were, this goes splat the same way. **Directly,
this may go splat. Probably goes splat.** But: maybe there are things
predictable about the *shape* of future play even when the exact line isn't —
that's the residual hope, and it's testable.

## Unfair local advantages

- **The VCT oracle as a free ground-truth probe for the latent**: do positions
  the oracle proves equivalent (same threat class / same win status) land near
  each other in latent space? A learned-abstraction quality metric no other
  domain hands you for free.
- **Trajectories already on disk**: 234k games of rails-v0 (2026-07-03,
  `~/data/sweep_runs/rails-v0/`, wandb `vraf0b6e`) including the collapse tail
  (lots of forcing sequences), plus the Rapfi mining rig for stronger-play
  corpora.
- **A ready-made eval from the rails-v0 night**: the momentum-swing signal
  (white blunder → black flips the game). Does latent-space search *see* the
  blunder-punish that move-space search mostly misses?

## Cheapest first spike (falsifiable, small)

Train a dynamics head `g(latent, move) → latent'` on existing trajectories.
Unroll k steps in latent only; read the value head off the unrolled latent;
compare to the real-position value. **If latent unrolls hold value fidelity
for 3–5 plies on contested positions, there's something to search over; if
fidelity decays immediately, that's a fast honest negative.** Either way, new
machinery learned.

## Pre-stated predictions (bets, not claims — stakes-free, PVE rules)

- **Claude ~60%**: the naive spike shows value fidelity decaying fast (<3
  plies) on *contested* positions while looking deceptively fine on quiet ones
  — i.e. the fog-of-war wall is real and shows up exactly where search matters.
- **Claude ~25–30%**: a temporal-abstraction (macro-move) variant produces a
  measurable search-quality win at fixed node budget within a few spike
  cycles. Low, but that's the payoff branch.
- **Claude ~high**: the latent-geometry probe (oracle-equivalent positions
  clustering) yields something interesting *regardless* of whether search
  improves — representation findings tend to outlive their motivating
  application.
- **Jason (standing, from the brainstorm)**: "probably goes splat" directly —
  but maybe shape-of-future-play prediction gets something going. (To be
  sharpened into a concrete bet when the first spike is scoped.)

## The pattern ladder (Jason's metaphor, same-day addendum)

It's not just Allis. The intuitive picture: **stones are quantum particles;
Allis threats are atoms** — reliable, provable structures that pop out of the
foam; **what's missing is molecules, and then chemistry.** At least maybe.

Introspection on human play says the missing layers are real: "hmm, *that*
region looks dangerous, this one looks promising" → the hmm *allocates* a
human-search; inside it, shape knowledge prunes ("open 3 is bad"; "if I play
this end it could lead to one of those shapes"; "if I play there I could give
them an H, which is death"). The H is just one kind of VCT pattern — the
Rapfi-games→VCT distillation attempt was shooting at exactly this extraction,
and it did NOT obviously fall out; **book still open** (see
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md), whose v0
blocking-probe negative and pivot are the same hunt from the genetics side).

The load-bearing observation underneath the metaphor: what's being described
is **different kinds of searches where one informs the other** — a cheap
pattern-level search proposing where to look and what to avoid, an expensive
concrete search verifying, and (in humans) a return channel where verified
discoveries become reusable named patterns ("H = death"). Current AZ has only
a flat version of this: the policy prior is the "hmm," but there's no explicit
molecule vocabulary between stones and moves, and nothing search discovers
mid-game gets *named and reused* — each subtree relearns it. LeGomoku's latent,
if it earns its keep, would BE the molecule layer, and the architecture target
shifts from "a better search" to **a dialogue of searches**.

Existing local evidence the molecule layer is non-trivial:
- **The claw** ([the-claw.md](the-claw.md)) — a proven-optimal defensive
  structure that is *provably invisible to line-organized eval*. A molecule
  that doesn't decompose into the atom vocabulary we score with.
- The molecule-toolkit v0 negative (blocking is itself line-shaped) — naive
  extraction finds atoms again, not molecules.
- The H-from-distillation open book, above.

(No claims; this section sharpens what the latent should be *for* if the
spikes get anything to hold onto.)

## Open questions

- What's the right latent target — EMA-teacher representation (JEPA-style),
  value/policy-sufficient embedding (MuZero-style), or oracle-supervised
  threat features (cheating on purpose, as a ceiling probe)?
- Macro-move vocabulary: hand-defined (threat launches, from Allis's taxonomy)
  first, or learned options from the start?
- Does the 42 wall generalize? I.e. is "deep-search truth" representationally
  equivalent to "hidden information" for these models — worth writing up as
  its own question if evidence accumulates either way.
- Where does this sit vs the seek-VCT thesis (net steers / oracle finishes)?
  A working latent threat model would be a *soft, learnable* oracle — 
  complement or competitor to the GPU solver. No claim yet.

## Child pages

None yet — expected: spike write-ups, the 42-wall question, latent-geometry
probe results. Add them to the [Experiments hub](../experiments.md) index as
they land.
