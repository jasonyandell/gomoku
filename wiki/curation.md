# Curation — how to put knowledge into this wiki

**This is the playbook.** When someone says *"curate this into the wiki"*, this
page is the whole instruction. It extends
[wiki-operating-model](topics/wiki-operating-model.md) (the philosophy) with
the concrete rules the 2026-07-04 reorganization established. If this page and
a habit disagree, this page wins.

## The layer map (where things live)

| Layer | Files | What belongs there |
|---|---|---|
| **Doorway** | [index.md](index.md) | One screen: you-are-here + the doors. Never prose. |
| **Story** | [story.md](story.md) | The narrative arc. Updated at **era boundaries only**. |
| **Hubs** | [alphazero](alphazero.md) · [experiments](experiments.md) · [derby](derby.md) · [m5-mainframe](m5-mainframe.md) · [reference](reference.md) · [ops](ops.md) (+[seek-vct](seek-vct.md) sub-hub) | One-fetch tables: started → now → learned + links. **No prose bodies.** |
| **Workflows** | [train-](train-a-model.md)/[eval-](eval-a-model.md)/[publish-a-model](publish-a-model.md) | Pinned "just get me going" pages. |
| **Topics** | `topics/*.md` | Maintained synthesis, one subject each, ~10 KB target. |
| **Evidence** | `ops/*` ledgers/logs, [TRAINING_WIKI.md](../TRAINING_WIKI.md), W&B, checkpoints | Append-only; dated corrections, never rewrites. |
| **Archive** | `_archive/` | Full-fidelity history rotated out of live pages. Nothing is deleted. |

## The five hard rules

1. **Settled-verdict-first.** Every topic page leads with its conclusion as of
   a date. The chronicle/trail lives below it or in `_archive/`. A reader must
   never parse a retraction stack bottom-up to learn what's true.
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
   run logs, launchd plists → ledgers/`_archive/`. Topics carry the maintained
   conclusion plus pointers. Don't let a synthesis page grow a queue.
5. **Archive, never delete.** Superseded bulk moves verbatim to `_archive/`
   with a pointer from the live page; update inbound links when you move
   anything (`grep -rn 'filename' wiki/`).

## Ingest — "curate this info into the wiki"

1. **Classify it.** New result/run/decision → which layer?
   - A dated event or receipt → [TRAINING_WIKI.md](../TRAINING_WIKI.md) (training
     evidence) or the relevant ops ledger. Append-only, dated.
   - A durable conclusion → the owning **topic page** (update its settled
     verdict + date; push displaced detail down or to archive). Search for an
     existing page before creating one (`grep -ril <keyword> wiki/topics/`).
   - A new subject with no home → new `topics/<kebab-slug>.md` **with a status
     banner**, linked from its owning hub the same edit.
   - Verbatim external material → `sources/` (never rewrite it; add a dated
     provenance note + reaction separately).
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

## Rotation thresholds (the giants stay caged)

- **log.md / any ops journal** > ~60 KB or an era old → rotate closed months
  to `_archive/log-YYYY-MM.md`, leave the pointer. **Split by date-prefix with a
  script, not by hand, and reconcile before/after** — count entries and sum bytes
  across (live + archive) and confirm both equal the pre-rotation totals. A
  freehand rotation on 2026-07-04 silently dropped 21 entries and had to be redone
  from git; a two-line count/byte check would have caught it.
- **Queue/board files**: when a race/era concludes, move its closed verdicts
  to `_archive/`, keep durable synthesis + open intake live.
- **Topic pages** > ~25 KB: check for chronicle-itis — hoist verdict, compress
  superseded sections to summaries, archive the cut text. (Reference
  dictionaries like [training-run-reference](topics/training-run-reference.md)
  are exempt — length is their job.)

## Lint (run one of these passes when curating broadly)

- Banners: does every topics/ page have a current status line? (`head -5`)
- Links: script-extract all relative `.md` links → any broken? any topics/
  page with zero hub inbound? (orphans)
- Staleness: pages presenting stopped systems in present tense (the
  janitor/worktree-hygiene incident — a retired tool documented procedurally
  WILL get re-summoned).
- Duplication: the same finding narrated in 2+ pages → consolidate per rule 3.
- Index honesty: does § You are here match the newest evidence?

## Style

- Honest provenance ("reconstructed from logs" says so); dated corrections,
  never silent rewrites; negatives banked as carefully as wins.
- Match the wiki's voice: dense, linked, evidence-cited (run IDs, issue #s).
- Prefer editing an existing page over creating a near-duplicate; prefer a
  hub-row edit over a new hub section.
