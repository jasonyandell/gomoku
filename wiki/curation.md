# Curation — how to put knowledge into this wiki

**This is the playbook.** When someone says *"curate this into the wiki"*, this
page is the whole instruction. It extends
[wiki-operating-model](topics/wiki-operating-model.md) (the philosophy) with
the concrete rules the 2026-07-04 reorganization established. If this page and
a habit disagree, this page wins.

## The layer map (where things live)

| Layer | Files | What belongs there |
|---|---|---|
| **Doorways** | [index.md](index.md) · [capabilities.md](capabilities.md) | One screen each: you-are-here / capability map + the doors. No leaf content. |
| **Story** | [story.md](story.md) · [training-timeline.md](training-timeline.md) | The narrative arc + the milestone index. Updated at **era boundaries**. |
| **Hubs** | [alphazero](alphazero.md) · [experiments](experiments.md) · [derby](derby.md) · [m5-mainframe](m5-mainframe.md) · [reference](reference.md) · [ops](ops.md) (+[seek-vct](seek-vct.md) sub-hub) | One fetch: tables + short status bullets. Essay prose lives in leaves. |
| **Workflows** | [train-](train-a-model.md)/[eval-](eval-a-model.md)/[publish-a-model](publish-a-model.md) | Pinned "just get me going" pages. |
| **Topics** | `topics/*.md` | Maintained synthesis, one subject each (~10 KB median; >25 KB triggers the chronicle-itis check below). |
| **Evidence** | `ops/*` ledgers/logs, [TRAINING_WIKI.md](../TRAINING_WIKI.md), `sources/` (verbatim external), `cards/` (era model-card artifacts), W&B, checkpoints | Append-only; dated corrections, never rewrites. |
| **Archive** | **git history** (no directory) | Cut/rotated content is preserved by the version control, not a parallel tree. Live pages carry a dated tombstone naming the recovery commit (`git show <sha>:<path>`). The old `wiki/_archive/` side-by-side was audited (zero lost facts) and deleted 2026-07-04; recover via `ca76350`. |

## The five hard rules

1. **Settled-verdict-first.** Every topic page leads with its conclusion as of
   a date. The chronicle/trail lives below it, compressed — or is cut with a
   tombstone (git preserves it). A reader must never parse a retraction stack
   bottom-up to learn what's true.
2. **Status banner on every topic page.** First line under the title, dated:
   `**LIVE**` · `**HISTORICAL**` (correct record, closed era) ·
   `**SUPERSEDED-BY(page)**` · `**DORMANT**` (paused, reactivatable) ·
   `**DESIGN-NEVER-BUILT**` · `**DEAD-END (lesson kept)**`.
   If work stops or moves on, updating the banner IS the curation.
3. **Tell each story once.** A finding gets ONE canonical home; everywhere
   else gets 2–3 lines + a link. (The white-defense theorem lives in
   [alphazero-lessons](topics/alphazero-lessons-15x15-gomoku.md); swap2 and
   the defense plan just point at it.)
4. **Evidence vs synthesis is a hard split.** Dated lane catalogs, receipts,
   run logs, launchd plists → ledgers, or cut-with-tombstone (git preserves).
   Topics carry the maintained conclusion plus pointers. Don't let a synthesis
   page grow a queue.
5. **Cut, never lose — git is the archive.** Superseded bulk is REMOVED from
   live pages; git history preserves every byte. The live page keeps a dated
   tombstone naming the recovery commit
   (`*(removed YYYY-MM-DD; recover: git show <sha>:<path>)*`). Never maintain
   a parallel archive tree — it competes with git and rots. Update inbound
   links whenever you move or remove anything (`grep -rn '<name>' wiki/`).

## Ingest — "curate this info into the wiki"

1. **Classify it — the routing table.** Whatever the input, it has one
   evidence home (append-only, dated) and one synthesis home (the maintained
   conclusion). Search for an existing page before creating one
   (`grep -ril <keyword> wiki/topics/`).

   | You have… | Evidence home (append, dated) | Synthesis home |
   |---|---|---|
   | Training run / result | [TRAINING_WIKI.md](../TRAINING_WIKI.md) | owning topic verdict + [alphazero](alphazero.md) hub row |
   | Perf / benchmark receipt | [ops/experiment-ledger.md](ops/experiment-ledger.md) (new dated **era header**, don't interleave with May receipts) | owning M5 topic + [m5-mainframe](m5-mainframe.md) hub row |
   | Eval / match outcome | [TRAINING_WIKI.md](../TRAINING_WIKI.md) (cmd, checkpoint, n, result) | [eval-suite](topics/eval-suite.md) or owning topic |
   | Idea / hunch (un-run) | [idea-pile](topics/idea-pile.md) (dated seed) | graduates to its own topic when run |
   | External material (paper, post, engine, chat) | `sources/` verbatim + dated provenance note | reaction/synthesis in the owning topic, never in the source file |
   | Decision / pivot / era turn | [log.md](log.md) + a [training-timeline](training-timeline.md) row | [index § You are here](index.md) + a [story.md](story.md) paragraph |
   | New subject with no home | — | new `topics/<kebab-slug>.md` **with status banner**, hub-linked in the same edit |
   | This-Mac / working-with-Jason fact | **not the wiki** → `~/.claude/.../memory/` | ([conventions](topics/conventions.md) § memory-vs-wiki) |
   | An answer you synthesized from the wiki | — | see **Query** below — file it back |
2. **Update the owning hub row** (the "now"/finding column) if the headline
   changed. Hubs stay tables.
3. **Check the blast radius.** Does this supersede/kill another page's claim?
   Fix that page's banner + add a dated correction line. Don't leave two pages
   asserting different truths.
4. **If an era turned** (chapter closed, program pivoted): update
   [index.md § You are here](index.md), [training-timeline](training-timeline.md)
   (one milestone row), and [story.md](story.md) (a paragraph, not a rewrite).
5. **Log it:** dated entry in [log.md](log.md) (structure/synthesis changes)
   — and TRAINING_WIKI if the training story changed.

## Query — answering from the wiki files back

The wiki's second write path, and the loop that makes it self-reinforcing
(per the [Karpathy source](sources/karpathy-llm-wiki.md): answers filed back
"compound in the knowledge base just like ingested sources do"). Ingest feeds
the wiki when someone remembers to curate; **query feeds it every time the
wiki is used** — asking is writing.

**Trigger:** you answered a question and it took synthesizing across 2+ pages
or reading raw evidence (W&B, logs, checkpoints, git). That assembly work IS
the signal a page is missing or stale. Before the session ends, file it:

- The answer **refined a settled verdict** → update the owning topic's
  verdict + date.
- The answer **exposed a missing hub row or link** → add it.
- The question had **no one-fetch home** → create the topic that should have
  existed (banner + hub link), so next time it *is* one fetch.
- A question that recurs at >1 fetch is a **lint finding** — a structural
  gap, not a retrieval failure.

A clean one-fetch answer (read one page, answered) files nothing — the wiki
already worked.

## Rotation thresholds (the giants stay caged)

- **log.md / any ops journal**: rotate months belonging to a *closed era* out
  of the live file (~60 KB is the smell threshold, not a hard cap — keep the
  live era readable in place). **Use the hardened tool, never rotate by hand,
  and let git be the archive — the three-step ritual:**
  ```bash
  # 1. split with the reconcile guarantee (archive named for the era it contains)
  uv run python scripts/wiki_rotate.py wiki/log.md \
      --before YYYY-MM --archive wiki/_archive/log-YYYY-MM.md [--dry-run]
  git add -A && git commit -m "rotate <journal> <era>"   # 2. verbatim split enters git
  git rm -r wiki/_archive && git commit -m "drop rotation staging"  # 3. git keeps it
  ```
  Then rewrite the journal's "Older eras" pointer as a tombstone naming the
  step-2 commit (`recover: git show <sha>:wiki/_archive/log-YYYY-MM.md`).
  The tool **refuses to write unless entry counts + byte totals across
  (live + archive) reconcile to the pre-rotation totals**, then re-verifies
  after writing — a freehand rotation on 2026-07-04 silently dropped 21
  entries and had to be redone from git; the reconcile check lives in the
  tool, not in vigilance. `wiki/_archive/` exists only transiently between
  steps 2 and 3; never let it accumulate.
- **Queue/board files**: when a race/era concludes, cut its closed verdicts
  with a tombstone (same three-step ritual); keep durable synthesis + open
  intake live.
- **Topic pages** > ~25 KB: run the chronicle-itis *check* (a trigger, not a
  violation) — hoist verdict, compress superseded sections to summaries,
  archive the cut text. Pages that pass the check stay big legitimately:
  reference dictionaries ([training-run-reference](topics/training-run-reference.md)),
  API contracts ([mega-vct-solver](topics/mega-vct-solver.md)), and
  verdict-first synthesis pages whose length is evidence density.

## Lint (run one of these passes when curating broadly)

**The mechanized half is a script — run it first, at the end of any curation
session:** `uv run python scripts/wiki_lint.py` (add `--json` for machine
output; exit 1 on errors). It checks banners (presence + date), broken relative
links (live pages; `_archive/` exempt), topic orphans, rotation smells
(>60 KB journals without an ARCHIVED/FROZEN head marker), chronicle-itis
triggers (>25 KB topics), and index § You-are-here staleness vs log.md. The
judgment-tier checks below can't be scripted — do them by reading:

- Banners: does every topics/ page have a current status line? (`head -5`)
- Links: script-extract all relative `.md` links → any broken? any topics/
  page with zero hub inbound? (orphans)
- Staleness: pages presenting stopped systems in present tense (the
  janitor/worktree-hygiene incident — a retired tool documented procedurally
  WILL get re-summoned).
- Duplication: the same finding narrated in 2+ pages → consolidate per rule 3.
- Index honesty: does § You are here match the newest evidence?
- Query gaps: questions that recurred at >1 fetch (see **Query**) → the page
  that would make them one fetch doesn't exist yet.

## Style

- Honest provenance ("reconstructed from logs" says so); dated corrections,
  never silent rewrites; negatives banked as carefully as wins.
- Match the wiki's voice: dense, linked, evidence-cited (run IDs, issue #s).
- Prefer editing an existing page over creating a near-duplicate; prefer a
  hub-row edit over a new hub section.
