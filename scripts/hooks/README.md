# Opt-in git hooks (`scripts/hooks/`)

Committed, **opt-in** hooks. They are not active until you point git at this
directory (one line, per clone/worktree — does not override anyone else's
config):

```bash
git config core.hooksPath scripts/hooks
```

To stop using them: `git config --unset core.hooksPath`.

## `pre-commit` — workflow resilience gauge (#51)

Fires `node scripts/check_workflow_resilience.mjs` **only** when the commit
stages a change to a top-level `.claude/workflows/*.js`; otherwise it exits 0
immediately and never spawns node, so ordinary commits are untouched.

This is the self-firing half of the lab's janitor+gauge rule for workflow
entropy: #50 shipped the gauge + a wiki note but it was run by hand. With the
hook enabled, an edit that reintroduces an unguarded `agent()` dereference is
caught at commit time. A non-zero checker exit blocks the commit; a missing
`node` warns and lets the commit through (the deploy/CI path can still enforce).

Bypass for a single commit (e.g. an unrelated emergency): `git commit -n`.
