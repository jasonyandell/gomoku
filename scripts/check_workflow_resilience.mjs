#!/usr/bin/env node
// check_workflow_resilience.mjs (#50) — deterministic resilience smoke test for
// the lab's Claude Code Workflows. It loads each `.claude/workflows/*.js`, injects
// STUB harness globals (no network, no real agents), and runs each workflow under
// three failure modes, asserting it never throws and always returns a result
// object:
//
//   • "agent-death"  — every agent() returns null (a terminal API error / a
//     user-skip). This is the regression guard for the `null is not an object`
//     crash that an API-overload window produced (#50).
//   • "happy"        — every agent() returns a valid stub: the normal path still
//     runs end-to-end.
//   • "stage2-death" — stage-1 agents succeed but the review/score agents die.
//     Exercises the per-item reviewer-null / score-null guards specifically.
//
// It mirrors how the real Workflow harness runs a script (strip `export`, wrap the
// body in an async IIFE, inject the globals), so a parse error here is a real
// syntax error there. Pure deterministic JS — no Date.now/Math.random.
//
//   Run:  node scripts/check_workflow_resilience.mjs    (exit 0 = all pass)
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const WF = join(ROOT, '.claude', 'workflows')

// --- stub harness primitives (match the documented agent()/parallel()/pipeline()
//     semantics: a throwing/erroring agent resolves to null, never rejects) -----
function stubParallel(thunks) {
  return Promise.all(thunks.map((t) => Promise.resolve().then(t).catch(() => null)))
}
function stubPipeline(items, ...stages) {
  return Promise.all(
    items.map(async (it, i) => {
      let v = it
      for (const s of stages) {
        try { v = await s(v, it, i) } catch { return null }
      }
      return v
    }),
  )
}
const noop = () => {}

// happy-path agent: return a shape keyed off opts.label so each workflow's reads
// succeed (the stub does not enforce the schema; it just returns plausible data).
function happyAgent(_prompt, opts) {
  const label = (opts && opts.label) || ''
  if (label.startsWith('runner:triage')) {
    return { picks: [{ number: 1, title: 't', slug: 's', rationale: 'r', sketch: 'sk', risk: 'low' }], skipped: [] }
  }
  if (label.startsWith('review')) return { verdict: 'APPROVE', oneLine: 'ok', details: 'checked', positive_control_cmd: 'cmd' }
  if (label.startsWith('train:')) {
    return { launched: true, pid: '123', last_trainer_line: 'epoch 1 vl 0.15 plies 30', latest_ckpt: '/tmp/x.pt', slice_seconds_elapsed: 1, notes: '' }
  }
  if (label.startsWith('score:')) return { gate_verdict: 'AMBIGUOUS', win_rate: 0.5, vl: 0.15, plies: 30, measured_outcome: 'confirm', reasoning: 'r' }
  return 'receipt: changed foo.py; added test_foo (PASS)' // impl / revise / build receipts
}
const deathAgent = () => null
const stage2DeathAgent = (prompt, opts) => {
  const label = (opts && opts.label) || ''
  if (label.startsWith('review') || label.startsWith('score')) return null // stage-2 dies
  return happyAgent(prompt, opts) // stage-1 succeeds
}

// --- load + run a workflow body with injected globals (mirrors the real harness)
async function runWorkflow(file, args, agentImpl) {
  const src = readFileSync(join(WF, file), 'utf8').replace(/^export\s+const\s+meta/m, 'const meta')
  const body = `return (async () => {\n${src}\n})()`
  const fn = new Function('agent', 'parallel', 'pipeline', 'phase', 'log', 'args', 'budget', body)
  const budget = { total: null, spent: () => 0, remaining: () => Infinity }
  return fn(agentImpl, stubParallel, stubPipeline, noop, noop, args, budget)
}

// --- cases (one per workflow, with args that pass its own input validation) ----
const CASES = [
  { file: 'implement-backlog.js', args: { maxPicks: 2 } },
  { file: 'sliding-derby-composite.js', args: {} },
  { file: 'reviewer-gated-fanout.js', args: [{ id: 'lane-a', task: 'do a thing' }, { id: 'lane-b', task: 'do another' }] },
  { file: 'build-issue-45-white-defense.js', args: {} },
]
const MODES = [['agent-death', deathAgent], ['happy', happyAgent], ['stage2-death', stage2DeathAgent]]

let failures = 0
console.log('Workflow resilience smoke test (#50): no real agents, deterministic.\n')
for (const c of CASES) {
  for (const [mode, impl] of MODES) {
    try {
      const result = await runWorkflow(c.file, c.args, impl)
      if (result === undefined || result === null || typeof result !== 'object') {
        throw new Error(`returned ${result === undefined ? 'undefined' : JSON.stringify(result)} (expected a result object)`)
      }
      console.log(`  PASS  ${c.file.padEnd(36)} [${mode}]`)
    } catch (e) {
      failures++
      console.log(`  FAIL  ${c.file.padEnd(36)} [${mode}] — ${e && e.message ? e.message : e}`)
    }
  }
}
console.log(failures ? `\n${failures} resilience check(s) FAILED.` : '\nAll workflow resilience checks passed (no crash on agent death).')
process.exit(failures ? 1 : 0)
