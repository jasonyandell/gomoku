# Wiki Operating Model

This repo's wiki should behave like a maintained codebase for knowledge. The
goal is not to make a nicer pile of notes; it is to compile experiment evidence
into a durable working model that future sessions can use immediately.

## What Counts As A Source

Sources are evidence the wiki reads from but should not casually rewrite:

- W&B run histories and summaries.
- Local `wandb/` files and training logs.
- Checkpoints and match outputs.
- Commands and configs used to launch experiments.
- Code state when it materially affects an experiment.
- External references recorded under [../sources/](../sources/).

The source layer can live partly outside `wiki/` because large artifacts already
have natural homes in this repo. The wiki should point to them clearly instead
of copying or cleaning them up.

## What Counts As Synthesis

Synthesis pages are maintained explanations that help the next session move
faster:

- What we believe about the training dynamics.
- Which hypotheses have been tried and what killed or supported them.
- Which runs or checkpoints are meaningful.
- Which metrics are misleading.
- Which next experiments are justified by evidence.

[../../TRAINING_WIKI.md](../../TRAINING_WIKI.md) remains the main chronological
lab notebook. Topic pages should emerge when a recurring question needs a stable
home or when an answer would otherwise disappear into chat history.

## Workflows

### Ingest

When a new run, source, or result matters:

- Read the source evidence directly.
- Add or update synthesis pages that the evidence changes.
- Add a dated entry to [../log.md](../log.md) if the wiki structure or maintained
  synthesis changed.
- Add a dated entry to [../../TRAINING_WIKI.md](../../TRAINING_WIKI.md) when the
  evidence changes the training story.

### Query

When answering a training question:

- Start from [../index.md](../index.md).
- Read the relevant synthesis page or training notebook section.
- Verify against raw evidence when the claim is recent, surprising, or important.
- If the answer creates reusable synthesis, file it back into the wiki.

### Lint

Periodically health-check the wiki for:

- stale claims contradicted by newer evidence;
- headings that hide important current conclusions near the bottom;
- missing links between index, notebook, and source records;
- orphan topic pages;
- hypotheses without run IDs or checkpoint evidence;
- answers in chat that should become durable wiki pages.

## Local Principle

The wiki should make the training investigation cumulative. We should not need
to rediscover, for the third time, whether a heuristic win-rate spike was a real
crossing or an `n=4` artifact. If the repo has learned something, the wiki should
carry it forward.
