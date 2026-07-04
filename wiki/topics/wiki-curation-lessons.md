# Wiki-curation lessons — the portable meta

**Status: LIVE (2026-07-04).** Written at the close of the great curation pass
as the transferable distillation: what we learned about *keeping an LLM-wiki
alive*, stated so it can be lifted into any other project. The gomoku wiki is
the evidence base; every lesson cites what actually happened here. Sibling of
[wiki-operating-model](wiki-operating-model.md) (this wiki's philosophy) and
[../curation.md](../curation.md) (this wiki's playbook); this page is the
*why* behind both, for export.

## The one-sentence version

A knowledge base stays alive when **using it improves it** (query files back),
**maintaining it is mechanized** (scripts with built-in verification, closed
vocabularies a linter can check), and **history is delegated to version
control** (cut with tombstones, never hoard) — everything else is vigilance,
and vigilance always loses eventually.

## 1. The disease is chronicle-itis, and it lives at the leaves

The append-only instinct is correct for *evidence* and fatal for *synthesis*.
Left alone, synthesis pages become stacks of dated updates and retractions
where the settled truth sits at the BOTTOM under superseded claims preserved
"for the trail" — here, a reader had to walk 1,256 lines of
alphazero-lessons to learn that §2 was retracted by §8–9 and re-confirmed by
§11. Cure: **settled-verdict-first** — every synthesis page leads with its
conclusion as-of a date; the trail is compressed below or cut. The hub layer,
rebuilt earlier, was healthy; the disease was invisible until a full-leaf
inventory. *Audit the leaves; the doorways always look fine.*

## 2. Staleness flows upward, truth lives at the leaves

The systemic failure found by the fresh-eyes audit — and confirmed by a live
experiment: a result page recorded that rails-v0 RAN while three hubs still
called it untried. A flat-footed agent reading the stale wiki still reached
the right conclusions *because leaf pages were current*. Leaves get updated
by whoever does the work; hubs only get updated by discipline. Cure: make
"update the owning hub row" an explicit step of every ingest (blast-radius
check), and lint index-vs-log freshness.

## 3. Query is a write operation (the Karpathy loop)

The single highest-leverage structural idea, and the one we adopted last: if
answering a question required synthesizing across 2+ pages or reading raw
evidence, **that assembly work is the signal a page is missing or stale —
file the answer back before the session ends.** Ingest feeds the wiki only
when someone remembers; query feeds it every time it's used. Proof it works
day one: a training-proposal question produced two independent agent
syntheses that mapped the proposal onto an already-run experiment — filed as
an idea-pile addendum instead of evaporating in chat. A recurring question
that costs >1 fetch is a lint finding, not a retrieval failure.

## 4. Mechanism over vigilance — every failure here was a vigilance failure

The day's honor roll of failures and their fixes:
- A **freehand journal rotation silently dropped 21 entries** → the fix is a
  rotation tool that *refuses to write* unless entry counts + bytes
  reconcile (`scripts/wiki_rotate.py`), not a rule saying "be careful."
- A **retired tool documented in present-tense procedural voice got
  re-summoned** by a later session → the fix is past-tense + banner, and a
  lint check for stopped-systems-in-present-tense.
- **9 pages used ad-hoc status prose** (BANKED, CHAPTER CLOSED) that no
  script could check → the fix is a CLOSED vocabulary of six markers
  (LIVE / HISTORICAL / SUPERSEDED-BY / DORMANT / DESIGN-NEVER-BUILT /
  DEAD-END) that `scripts/wiki_lint.py` enforces.
- A **subagent silently worked in the wrong checkout** because "we're in the
  worktree" was conversational, not mechanical → the fix is pinning the
  session cwd, not passing paths carefully.
The pattern never varies: encode the lesson in a tool or a closed vocabulary,
because the next session won't remember the prose.

## 5. Git is the archive — never maintain a parallel history tree

An `_archive/` directory feels safe and is actually a second wiki that
competes with the first and rots. Ours (528 KB, 17 files) was audited —
adversarially, with random mid-file spot-checks — found to contain exactly
THREE archive-only details worth keeping (hoisted to live pages), and
deleted. Cut content gets a dated **tombstone naming the recovery commit**
(`*(removed 2026-07-04; recover: git show ca76350:<path>)*`). Corollary:
**audit before you delete** — the audit is cheap, and it found real (if
minor) items every sweep missed.

## 6. Test the system with cold agents, and count fetches

The only honest test of a knowledge base is a **flat-footed reader**: verbatim
question, zero scaffolding, whatever the repo ambiently provides. Two live
results here: a cold agent answered a five-part narrative question ("tell us
about Bruce") in 4 fetches with zero guessing; a cold agent given a training
proposal self-routed via the entry-file → wiki path and matched a
heavily-scaffolded agent's conclusions run-ID-for-run-ID. The metric is
fetches-to-answer; the pass criterion is "the knowledge base, not the prompt
scaffolding, carried the answer." A guided prompt buys coverage, not
correctness — if flat-footed fails, fix the wiki, not the prompt.

## 7. The load-bearing separations

- **Evidence vs synthesis** (hard split): append-only dated records vs
  maintained conclusions. Never let a synthesis page grow a queue; never
  rewrite evidence, correct it with dated entries.
- **Tell each story once**: a finding gets ONE canonical home, everyone else
  links. Duplication is where contradictions breed (the white-defense theorem
  was told 3× with drift before consolidation).
- **Story ≠ timeline ≠ hubs**: a milestone table is not a narrative. The
  missing layer here was the *story* — what we believed, what the machine did
  to our theories, how each era's defeat became the next era's premise.
  Humans and agents both read it first once it existed.
- **Routing table for ingest**: every input class (run, perf receipt, idea,
  external source, decision, machine fact, synthesized answer) has ONE
  evidence home and ONE synthesis home, written down. "Where does this go?"
  should never be a judgment call.

## 8. The schema must live inside the system it governs

The playbook (curation.md) is itself a wiki page, found unaided by cold
agents given only "curate this into the wiki." The agent-instruction files
(CLAUDE.md/AGENTS.md) carry one pointer line, not the rules. This is what
makes the maintainer improve through the artifact it maintains — and it's
testable: the **minimal-instruction bar** ("does a fresh agent with five
words find the playbook and do it right?") is the acceptance test for the
whole system.

## 9. Curation is a pass, maintenance is a loop

The big-bang reorganization (inventory → work packages → adversarial sweeps)
was necessary ONCE, to pay down the debt. Sustainability is the small loop:
ingest per the routing table, file query answers back, run the lint at
session end, rotate when the smell threshold trips, adjudicate size triggers
and RECORD the adjudication (so the same notices aren't re-litigated), banner
status changes the moment work stops. If the small loop runs, there is never
a second big bang.

---

*Bootstrapping a new project's wiki from these:* start with the entry-file
pointer + a one-page playbook (layer map, routing table, the six banners, the
query rule, rotation ritual) + the two scripts (`wiki_rotate.py`,
`wiki_lint.py` — both portable, stdlib-only) + an index and an append-only
log. Write the story page as soon as there's a story. Test with a flat-footed
agent before trusting it.
