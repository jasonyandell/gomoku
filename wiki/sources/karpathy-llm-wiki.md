# Source Record: Karpathy LLM Wiki

Source URL:
https://gist.githubusercontent.com/karpathy/442a6bf555914893e9891c11519de94f/raw/ac46de1ad27f92b28ac95459c782c07f6b8c964a/llm-wiki.md

Retrieved: 2026-05-19

Status: External organizing charter. Treat the URL as the source of record; this
page records how the idea applies locally without duplicating the full text.

## Relevant Ideas For This Repo

- A wiki should be a persistent, compounding artifact, not a fresh retrieval pass
  over raw documents each time someone asks a question.
- The agent maintains the wiki: summarizing, cross-referencing, filing,
  correcting stale claims, and keeping the index useful.
- The human curates sources, guides emphasis, asks questions, and decides what
  matters.
- The useful architecture has three layers:
  - raw or source evidence, kept stable;
  - wiki synthesis, maintained by the agent;
  - schema, captured in an agent-facing instruction file.
- Two special files keep the system navigable:
  - an index for content-oriented navigation;
  - an append-only log for chronological maintenance history.
- Good answers should be filed back into the wiki when they represent reusable
  synthesis.
- Periodic linting should look for contradictions, stale claims, missing links,
  orphan pages, and gaps that need new evidence.

## Local Mapping

- Raw/source evidence: W&B runs, local logs, checkpoint files, match outputs,
  scripts, command output, and records in this directory.
- Wiki synthesis: [../index.md](../index.md), [../topics/](../topics/), and
  [../../TRAINING_WIKI.md](../../TRAINING_WIKI.md).
- Schema: [../../AGENTS.md](../../AGENTS.md).
