// implement-backlog — the autonomous "runner / researcher / implementer / reviewer"
// recipe Jason called the winning one. A standing Workflow that picks its OWN work
// from the live GitHub backlog and drives each safe code-only issue from open →
// reviewed-and-ready-to-merge, every step in an isolated worktree.
//
// FLOW (the four roles):
//   RUNNER     one agent triages the live `gh` ready queue → the safest code-only,
//              bounded, unit-testable, no-GPU, no-live-validation issues (ranked,
//              capped at maxPicks). It is the gate on WHAT we touch.
//   RESEARCHER per issue: read the issue + the relevant code, produce a concrete
//              plan (files, the exact change, the test that proves it).
//   IMPLEMENTER per issue, in its OWN worktree on feat/gh-<N>-<slug>: make the
//              change, add/run the targeted test, commit. CPU only — never the GPU
//              (the sliding derby owns MPS; code-only issues don't need it).
//   REVIEWER   a fresh, read-only auditor per issue: is the claim supported by the
//              test shown? scope creep? broken math? APPROVE / REVISE (bounded
//              loop) / BLOCK (escalate).
//
// This STOPS at "APPROVED, branch ready" and returns the branch list. The serial
// `git merge --no-ff` + full-suite gate + push stays with the orchestrator (me),
// so parallel agents never race on shared `main` and a red suite can be reverted
// in one hand. NOT the GPU lane; never launches run_sweep / training.
//
// INVOKE: Workflow({ name: 'implement-backlog', args: { maxPicks: 4 } })

export const meta = {
  name: 'implement-backlog',
  description: 'Autonomously triage the live GitHub ready queue and drive the safest code-only issues from open → reviewed-and-ready-to-merge, each in an isolated worktree. Runner/researcher/implementer/reviewer. Never touches the GPU lane.',
  whenToUse: 'Idle capacity + a backlog of code-only GitHub issues. Picks its own work, implements + tests + reviews each in isolation, returns merge-ready branches for a serial operator merge.',
  phases: [
    { title: 'Triage', detail: 'one runner agent ranks the ready queue into safe picks' },
    { title: 'Implement', detail: 'research → implement+test in an isolated worktree, per issue' },
    { title: 'Review', detail: 'a fresh reviewer per issue; REVISE loops, BLOCK escalates' },
  ],
}

const MAX_PICKS = (args && Number(args.maxPicks)) || 4
const MAX_REVISE = 2

// Issues the live overnight lab is ALREADY driving (the sliding derby / defense
// arc) or that are not code-only — the runner must not pick these.
const EXCLUDE = [33, 34, 36, 37, 38, 39, 30, 21, 2, 19, 18]

const TRIAGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['picks', 'skipped'],
  properties: {
    picks: {
      type: 'array',
      description: `Up to ${MAX_PICKS} issues, ranked safest-first, that are code-only, bounded, unit-testable WITHOUT GPU or a live external engine, and safe to auto-implement+merge tonight.`,
      items: {
        type: 'object', additionalProperties: false,
        required: ['number', 'title', 'slug', 'rationale', 'sketch', 'risk'],
        properties: {
          number: { type: 'integer' },
          title: { type: 'string' },
          slug: { type: 'string', description: 'short kebab-case branch slug (no "gh-N" prefix; the workflow adds it)' },
          rationale: { type: 'string', description: 'why this is genuinely code-only + bounded + testable + safe to merge unsupervised' },
          sketch: { type: 'string', description: 'concrete approach: which file(s), the exact change, and the test that proves it' },
          risk: { enum: ['low', 'medium'], description: 'merge risk to main' },
        },
      },
    },
    skipped: {
      type: 'array',
      description: 'Ready-queue issues deliberately NOT picked, each with a one-line reason (needs GPU / needs live validation / too large / ambiguous / epic / being handled by the live derby).',
      items: {
        type: 'object', additionalProperties: false,
        required: ['number', 'reason'],
        properties: { number: { type: 'integer' }, reason: { type: 'string' } },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['verdict', 'oneLine', 'details'],
  properties: {
    verdict: { enum: ['APPROVE', 'REVISE', 'BLOCK'], description: 'APPROVE=ready to merge; REVISE=fixable, loop back; BLOCK=escalate.' },
    oneLine: { type: 'string' },
    details: { type: 'string', description: 'APPROVE: what was checked. REVISE: exactly what to fix. BLOCK: why it must escalate.' },
  },
}

// ---------------------------------------------------------------- Triage
phase('Triage')
const triage = await agent(
  `You are the RUNNER for the gomoku research lab's autonomous backlog. Find the safest code-only GitHub issues to implement unsupervised tonight.

Run (Bash):
  gh issue list --state open --limit 60 --search 'no:assignee -label:blocked -label:deferred -label:in-progress -label:epic -label:runner-domain -label:human-gated'
Then \`gh issue view <N>\` on each plausible candidate to read its body.

PICK (ranked safest-first, at most ${MAX_PICKS}) issues that are ALL of:
  - genuinely CODE-ONLY (no GPU training, no live external engine, no W&B run needed),
  - BOUNDED (a localized change in ~1-3 files with a clear definition of done),
  - UNIT-TESTABLE on CPU (you can write/run a pytest or equivalent that proves it),
  - safe to MERGE to main unsupervised tonight (low blast radius).
EXCLUDE these numbers (already owned by the live derby / not code-only / epics): ${EXCLUDE.join(', ')}.
Also exclude anything labeled gpu / needs-live-validation, anything that is an epic, and anything whose acceptance genuinely needs a training run or a wine/external engine.

For each pick give a concrete sketch: the file(s), the exact change, and the test that proves it. Everything in the ready queue you did NOT pick goes in \`skipped\` with a one-line reason. Be conservative: a smaller set of truly-safe picks beats a big risky list. Do NOT implement anything yet — triage only.`,
  { label: 'runner:triage', phase: 'Triage', schema: TRIAGE_SCHEMA },
)

const picks = (triage.picks || []).slice(0, MAX_PICKS)
log(`Runner picked ${picks.length}: ${picks.map(p => `#${p.number}(${p.slug})`).join(', ') || 'none'}`)
log(`Runner skipped ${triage.skipped?.length || 0}.`)
if (!picks.length) {
  return { approved: [], blocked: [], reviseExhausted: [], triage }
}

// ---------------------------------------------------- Implement → Review (pipeline)
const results = await pipeline(
  picks,
  // STAGE 1: research + implement + test, in an isolated worktree on a feat branch.
  (pick) => agent(
    `You own GitHub issue #${pick.number} — "${pick.title}". Work ONLY in your own git worktree, on branch \`feat/gh-${pick.number}-${pick.slug}\`. CPU only — NEVER touch the GPU (no run_sweep / training / delo_derby); the sliding derby owns MPS.

FIRST research: \`gh issue view ${pick.number}\` and read the relevant code so your plan is grounded in what's actually there (the triage sketch below may be imperfect — trust the code over the sketch).
Triage sketch (a starting point, verify it): ${pick.sketch}

THEN implement the smallest correct change that closes the issue, and ADD a test that genuinely proves the behavior (not a tautology). Run the touched-area tests (targeted pytest / the relevant suite) — NOT the full suite (avoid GPU contention). Commit to your feat branch with a clear message. Do NOT add \`Closes #${pick.number}\` (the orchestrator adds it at merge). Do NOT merge.

Return a receipt: the issue, the exact change (files + what), the test you added and its PASS/FAIL output, and any caveat or scope you deliberately left out.`,
    { label: `impl:gh-${pick.number}`, phase: 'Implement', isolation: 'worktree' },
  ),
  // STAGE 2: fresh reviewer, bounded REVISE loop.
  async (receipt, pick) => {
    if (!receipt) return { number: pick.number, slug: pick.slug, branch: `feat/gh-${pick.number}-${pick.slug}`, verdict: 'BLOCK', oneLine: 'implementer returned nothing (died/skipped)', details: 'No receipt produced.' }
    let verdict, attempt = 0
    while (true) {
      verdict = await agent(
        `You are a FRESH, read-only REVIEWER for GitHub issue #${pick.number} — "${pick.title}". You did not write this. Grade what was DELIVERED, not what was intended.

RECEIPT:
${receipt}

Audit hard: Does the change actually close the issue? Is the test REAL (would it fail before the fix)? Any scope creep, broken logic, convention violation (worktree/branch discipline, GPU avoidance), or unsupported claim? If the receipt shows tests failing or no test, that is at most REVISE (or BLOCK if fundamentally misguided). Return APPROVE / REVISE / BLOCK.`,
        { label: `review:gh-${pick.number}`, phase: 'Review', schema: VERDICT_SCHEMA },
      )
      if (verdict.verdict !== 'REVISE' || attempt++ >= MAX_REVISE) break
      log(`#${pick.number}: REVISE ${attempt}/${MAX_REVISE} — ${verdict.oneLine}`)
      receipt = await agent(
        `Reviewer returned REVISE on issue #${pick.number} (branch feat/gh-${pick.number}-${pick.slug}):
${verdict.details}

Fix it in your worktree and return an updated receipt (same format: change, test PASS/FAIL, caveats).`,
        { label: `revise:gh-${pick.number}`, phase: 'Implement', isolation: 'worktree' },
      )
    }
    const exhausted = verdict.verdict === 'REVISE'
    return {
      number: pick.number, slug: pick.slug,
      branch: `feat/gh-${pick.number}-${pick.slug}`,
      verdict: exhausted ? 'REVISE_EXHAUSTED' : verdict.verdict,
      oneLine: verdict.oneLine, details: verdict.details,
    }
  },
)

const done = results.filter(Boolean)
const approved = done.filter(r => r.verdict === 'APPROVE')
const blocked = done.filter(r => r.verdict === 'BLOCK')
const stuck = done.filter(r => r.verdict === 'REVISE_EXHAUSTED')

log(`Done: ${approved.length} APPROVE · ${blocked.length} BLOCK · ${stuck.length} revise-exhausted`)
log(approved.length
  ? `Merge-ready (serial --no-ff + full-suite gate, orchestrator's call): ${approved.map(r => `#${r.number} ${r.branch}`).join(', ')}`
  : 'Nothing cleared review — no branches ready to merge.')

return { approved, blocked, reviseExhausted: stuck, triage }
