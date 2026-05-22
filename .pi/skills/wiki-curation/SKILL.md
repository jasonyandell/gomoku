---
name: wiki-curation
description: Curate worker receipts and experiment evidence into the project wiki while keeping raw evidence separate from maintained synthesis.
---

# Wiki Curation

Use this skill when updating wiki control-room pages after frontier workers finish.

## Inputs

- `AGENTS.md`
- `wiki/index.md`
- `TRAINING_WIKI.md` when training claims are involved
- `.frontier/config.json`
- `.frontier/lanes.json`
- run state under `.frontier/runs/<run-id>/state.json`
- worker receipts under `.frontier/runs/<run-id>/workers/*/receipt.md`
- open notes under `wiki/ops/open-notes/`

## Curation Rules

- Keep evidence and synthesis separate.
- Do not rewrite old notebook conclusions to make the story cleaner.
- Append dated corrections when new evidence contradicts old notes.
- Keep `wiki/ops/*` concise and current.
- Link to artifacts instead of copying giant logs.
- Update `wiki/log.md` when wiki structure or maintained synthesis materially changes.

## Control-Room Pages

- `wiki/ops/status.md`: current focus and latest read.
- `wiki/ops/frontier.md`: human-readable lane board.
- `wiki/ops/baselines.md`: benchmark command and reference-number catalog.
- `wiki/ops/experiment-ledger.md`: concise receipts and decisions.
- `wiki/ops/test-ledger.md`: commands/results supporting claims.

## Output

Finish with:

- files changed,
- receipts integrated,
- blockers,
- promoted next lane,
- any stale claims trimmed or corrected.
