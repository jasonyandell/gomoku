// reviewer-gated-fanout — the lab's first real Claude Code Workflow.
//
// WHAT THIS IS: the *everything-else* lane of the two-queue scheduler, encoded
// as deterministic JS instead of prose. It fans out N independent CPU sub-lanes
// (code/analysis/doc work) — each in its own git worktree — and chains every
// lane straight into a FRESH Reviewer that grades the receipt with a structured
// APPROVE / REVISE / BLOCK verdict. REVISE loops back (bounded); BLOCK surfaces
// for escalation. The verify gate is the control flow, so it cannot be skipped.
//
// WHAT THIS IS NOT: it is NOT the GPU lane. Workflow agents are LLM reasoning
// agents, not MPS processes — they cannot hold the serial GPU lock and must
// never launch run_sweep / delo_derby / training. CPU/analysis/doc work only.
// It is also NOT a cross-session daemon (that stays cron + watchdog); it runs
// once per invocation and returns.
//
// It deliberately STOPS at "APPROVED, branch ready". The serial `git merge
// --no-ff` + push stays the operator's call (Class-B-adjacent), so this never
// touches shared `main`.
//
// INVOKE:
//   Workflow({ name: "reviewer-gated-fanout", args: [
//     { id: "curate-buffer", task: "Add a recency+diversity curator to ..." },
//     { id: "wiki-audit",    task: "Audit wiki/topics/ for stale run IDs ..." },
//   ]})
// Each lane: { id: <kebab slug, becomes feat/<id>>, task: <full instructions> }

export const meta = {
  name: 'reviewer-gated-fanout',
  description: 'Fan out CPU code/analysis/doc lanes in worktrees; gate each with a fresh schema-validated Reviewer (APPROVE/REVISE/BLOCK). Never touches the GPU lane.',
  whenToUse: 'You have 2+ independent code-only / analysis / doc sub-lanes to run in parallel and want every one verified by a fresh Reviewer before it is eligible to merge. The research-lab everything-else queue.',
  phases: [
    { title: 'Implement', detail: 'one isolated-worktree agent per sub-lane' },
    { title: 'Review', detail: 'a fresh Reviewer per lane; REVISE loops, BLOCK escalates' },
  ],
}

const MAX_REVISE = 2 // bounded retry; matches the lab's "don't churn forever" rule

// The Reviewer's verdict IS a schema — the lab's VERDICT / ONE-LINE / DETAILS block.
const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    verdict: { enum: ['APPROVE', 'REVISE', 'BLOCK'], description: 'APPROVE=ready to merge; REVISE=fixable, loop back; BLOCK=escalate to a human.' },
    oneLine: { type: 'string', description: 'One-line summary of the verdict.' },
    details: { type: 'string', description: 'For REVISE: exactly what to change. For BLOCK: why it must escalate. For APPROVE: what was checked.' },
  },
  required: ['verdict', 'oneLine', 'details'],
}

const lanes = Array.isArray(args) ? args.filter(l => l && l.id && l.task) : []
if (!lanes.length) {
  throw new Error('reviewer-gated-fanout needs args: [{id, task}, ...] — one entry per CPU sub-lane. See the header for an example.')
}
log(`Fanning out ${lanes.length} sub-lane(s): ${lanes.map(l => l.id).join(', ')}`)

const IMPLEMENT = lane => agent(
  `You own sub-lane "${lane.id}". Work ONLY inside your own git worktree on branch feat/${lane.id}; never edit a checkout outside it, never touch the GPU (no run_sweep / training).\n\nTASK:\n${lane.task}\n\nSmoke-test what you change (pytest the touched area, or explain why no test applies). Do NOT merge to main. Return a receipt: what changed (files), how you verified it, and any caveats.`,
  { label: `impl:${lane.id}`, phase: 'Implement', isolation: 'worktree' },
)

// pipeline (no barrier): each lane hits its Reviewer the moment it finishes
// implementing — lane A is being audited while lane B is still being built.
const results = await pipeline(
  lanes,
  IMPLEMENT,
  async (receipt, lane) => {
    let verdict, attempt = 0
    while (true) {
      verdict = await agent(
        `You are a FRESH, read-only Reviewer for sub-lane "${lane.id}". You did not write this; grade what was DELIVERED, not what was intended.\n\nRECEIPT:\n${receipt}\n\nAudit: is the claim supported by the verification shown? Any premature promotion, confounded knobs, broken math, or charter/convention violation? Did it stay in its worktree and off the GPU? Return your verdict.`,
        { label: `review:${lane.id}`, phase: 'Review', schema: VERDICT_SCHEMA },
      )
      if (verdict.verdict !== 'REVISE' || attempt++ >= MAX_REVISE) break
      log(`${lane.id}: REVISE (${attempt}/${MAX_REVISE}) — ${verdict.oneLine}`)
      receipt = await agent(
        `Reviewer returned REVISE on "${lane.id}":\n${verdict.details}\n\nFix it in your worktree (branch feat/${lane.id}) and return an updated receipt.`,
        { label: `revise:${lane.id}`, phase: 'Implement', isolation: 'worktree' },
      )
    }
    const exhausted = verdict.verdict === 'REVISE' // hit the retry cap still asking for changes
    return { lane: lane.id, branch: `feat/${lane.id}`, verdict: exhausted ? 'REVISE_EXHAUSTED' : verdict.verdict, oneLine: verdict.oneLine, details: verdict.details }
  },
)

const done = results.filter(Boolean)
const approved = done.filter(r => r.verdict === 'APPROVE')
const blocked = done.filter(r => r.verdict === 'BLOCK')
const stuck = done.filter(r => r.verdict === 'REVISE_EXHAUSTED')

log(`Done: ${approved.length} APPROVE · ${blocked.length} BLOCK · ${stuck.length} revise-exhausted`)
log(approved.length
  ? `Ready to merge (serial --no-ff, operator's call): ${approved.map(r => r.branch).join(', ')}`
  : 'Nothing cleared the Reviewer — no branches ready to merge.')

// Returned to the operator: APPROVE branches are merge-ready; BLOCK/stuck need a human.
return { approved, blocked, reviseExhausted: stuck }
