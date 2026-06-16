// sliding-derby-composite — the measured-outcome sliding derby (RFC v2).
//
// MVP = the TRAIN-supervisor measured-outcome CYCLE, run on a known-answer
// hypothesis first (champion-continuation) to validate the machinery before we
// trust it on anything we don't already know — the methodology applied to its own
// construction. ONE cycle per invocation (launch detached slice → score the result
// vs its PRE-STATED title-card outcome → record); the "forever" loop is a chain of
// re-invocations (a dumb re-kicker), each re-adopting running work from state —
// the shape the harness probe (workflow-harness-capabilities.md) confirmed.
//
// GROWTH PATH (not yet wired — this session/next builds it): wrap this train lane +
// a RESEARCH supervisor (fresh skill-loading proposer) + a WORK supervisor (code
// issues) in a top `parallel([...])`, loop, and re-rank a multi-hypothesis queue.
//
// INVOKE: Workflow({ name: 'sliding-derby-composite' })  // uses the seeded champ-cont hypothesis
//   or args:{hypothesis:{...}, slice_secs, gate_n} to race a different one.

export const meta = {
  name: 'sliding-derby-composite',
  description: 'Measured-outcome sliding derby (RFC v2) — train-supervisor cycle: launch a detached time-capped slice for the top hypothesis, score the result vs its PRE-STATED title-card outcome (3-way gate + vl/plies death-tell), record confirm/refute/ambiguous. First hypothesis = champion-continuation (known-answer self-test + #37 control).',
  phases: [
    { title: 'TrainCycle', detail: 'launch detached time-capped slice → poll to completion' },
    { title: 'Score', detail: '3-way gate + trainer vl/plies → measured outcome vs the pre-stated card' },
  ],
}

const REPO = '/Users/jason/code/gomoku'

// Hypothesis #1 (seed): champion-continuation — the known-answer self-test. A
// no-teacher continuation of eval502 should NOT degrade → the gate should be
// AMBIGUOUS (can't distinguish from eval502 = stable) with vl healthy. If the loop
// scores THIS correctly, the machinery is real. Doubles as the #37 death-spiral
// control (does eval502 degrade with no teacher at all?).
const HYP = (args && args.hypothesis) || {
  id: 'champ-cont',
  title: 'champion-continuation (known-answer self-test + #37 death-spiral control)',
  cell: 'G15-128x10-bigbuf',
  resume: `${REPO}/sweep_runs/g15_128x10_bigbuf_eval502.pt`,
  peer: `${REPO}/sweep_runs/g15_128x10_bigbuf_eval502.pt`,
  gate_on: 'h2h',
  slice_secs: (args && args.slice_secs) || 400,
  gate_n: (args && args.gate_n) || 40,
  // The PRE-STATED OUTCOME (the title card) — the scorer applies this verbatim:
  outcome: 'CONFIRM = did NOT degrade vs eval502: gate verdict in {PROMOTE, AMBIGUOUS} AND vl >= 0.10 AND plies >= 25 (stable — the EXPECTED no-teacher result; AMBIGUOUS just means "cannot distinguish from eval502", which IS stability, i.e. confirm). REFUTE = gate == REVERT (proven statistically worse) OR (vl < 0.10 AND plies >= 25 = value-poisoning death) OR (plies < 25 sustained = fast-attack collapse). Otherwise → AMBIGUOUS-OUTCOME (inconclusive, would earn one more lap).',
}

const CELLDIR = `${REPO}/sweep_runs/${HYP.cell}-board15`
const TLOG = `${REPO}/sweep_logs/${HYP.cell}-board15/trainer.log`
const BOARD = `${REPO}/sweep_runs/composite_derby_board.jsonl`

// --- Resilience helper (deterministic; #50) ---------------------------------
// `agent()` returns null when a subagent dies on a terminal API error (or the
// user skips it). agentTry re-spawns IDEMPOTENT, read-only agents up to `tries`
// times (a fresh internal-retry budget) to ride out a transient API blip. Use it
// ONLY for the SCORE agent (dry-run gate + log reads — no side effects). NEVER for
// the train-LAUNCH agent: re-spawning it could double-launch a detached GPU slice.
// On persistent failure it returns null and the caller degrades gracefully — and
// because this workflow is built to be re-invoked (the dumb re-kicker re-adopts
// running work from state), a clean abort here loses nothing.
async function agentTry(prompt, opts, tries = 3) {
  for (let i = 1; i <= tries; i++) {
    const r = await agent(prompt, opts)
    if (r) return r
    log(`  agent ${(opts && opts.label) || '?'} returned null (attempt ${i}/${tries}) — API overload? re-spawning…`)
  }
  log(`  agent ${(opts && opts.label) || '?'} gave up after ${tries} attempts — degrading gracefully.`)
  return null
}

phase('TrainCycle')
const slice = await agent(
  `You run ONE measured-outcome TRAIN cycle for hypothesis "${HYP.id}" — ${HYP.title}. You ARE the train-supervisor; launching a GPU slice is your job (no other GPU tenant should be running — verify with pgrep first; if one is, wait/abort and say so). In ${REPO}:
1. Clean the cell so it warm-starts fresh: \`.venv/bin/python scripts/run_sweep.py --cell ${HYP.cell} --clean\`.
2. Launch a DETACHED, time-capped slice (reparents to init; survives you):
   GOMOKU_BOARD_SIZE=15 nohup .venv/bin/python scripts/run_sweep.py --cell ${HYP.cell} --resume ${HYP.resume} --max-wall-secs ${HYP.slice_secs} --foreground >> ${REPO}/sweep_logs/${HYP.cell}-board15/slice.log 2>&1 &
   Record the PID.
3. POLL to completion (the slice self-terminates at the --max-wall-secs cap (~${HYP.slice_secs}s) and force-saves latest.pt). Loop: \`sleep 30\`; check ${TLOG} epoch is advancing AND gen is flowing (new=>0 / games climbing — flag if STARVED new=0). When pgrep -f 'run_sweep.py --cell ${HYP.cell}' returns nothing, the slice is done. Hard cap your wait at ${HYP.slice_secs + 300} seconds.
4. Confirm ${CELLDIR}/checkpoints/latest.pt exists and is newer than slice start.
Return: launched, pid, the LAST line of ${TLOG} (carries epoch/vl/plies/new/games for the scorer), latest_ckpt path, slice_seconds_elapsed, and notes (esp. any gen-starvation / NaN / crash).`,
  { label: `train:${HYP.id}`, phase: 'TrainCycle',
    schema: { type:'object', additionalProperties:false,
      required:['launched','last_trainer_line','latest_ckpt','notes'],
      properties:{ launched:{type:'boolean'}, pid:{type:'string'}, last_trainer_line:{type:'string'},
        latest_ckpt:{type:'string'}, slice_seconds_elapsed:{type:'number'}, notes:{type:'string'} } } }
)

phase('Score')
if (!slice) log(`Train-supervisor agent returned null (API overload?) — a detached slice may or may not have launched; scoring the cell's latest.pt by fallback path and flagging in notes.`)
const candidate = slice && slice.latest_ckpt ? slice.latest_ckpt : `${CELLDIR}/checkpoints/latest.pt`
const score = await agentTry(
  `You SCORE the just-finished slice for hypothesis "${HYP.id}" against its PRE-STATED outcome — the heart of the measured-outcome derby. Do NOT launch any training. In ${REPO}:
1. Run the 3-way gate in DRY-RUN (it must NOT auto-slide the peak — the derby decides promotion):
   GOMOKU_BOARD_SIZE=15 .venv/bin/python scripts/sliding_gate.py --dry-run --device mps --gate-on ${HYP.gate_on} --n-games ${HYP.gate_n} --sims 200 --seed 0 --candidate ${candidate} --peak ${HYP.peer} --verdict-log /tmp/composite_${HYP.id}_gate.jsonl --board /tmp/composite_${HYP.id}_gateboard.json
   Capture verdict ∈ {PROMOTE, REVERT, AMBIGUOUS}, win_rate (or white_defense_rate), CI, delta_elo.
2. Death-tell signals: \`tail -5 ${TLOG}\` → extract vl (value loss) and plies (mean) from the latest epoch lines.
3. Apply the hypothesis's PRE-STATED outcome VERBATIM: "${HYP.outcome}"
   → decide measured_outcome ∈ {confirm, refute, ambiguous}, naming WHICH clause fired with the numbers (gate verdict + vl + plies).
4. Append one JSONL line to ${BOARD}: echo a JSON object with ts ($(date -u +%Y-%m-%dT%H:%M:%SZ)), id "${HYP.id}", gate_verdict, win_rate, vl, plies, measured_outcome, reasoning.
Return: gate_verdict, win_rate, vl, plies, measured_outcome, reasoning.`,
  { label: `score:${HYP.id}`, phase: 'Score',
    schema: { type:'object', additionalProperties:false,
      required:['gate_verdict','measured_outcome','reasoning'],
      properties:{ gate_verdict:{enum:['PROMOTE','REVERT','AMBIGUOUS','UNKNOWN']}, win_rate:{type:'number'},
        vl:{type:'number'}, plies:{type:'number'}, measured_outcome:{enum:['confirm','refute','ambiguous']},
        reasoning:{type:'string'} } } }
)

if (!score) {
  log(`Score agent unavailable after retries (API overload?) for ${HYP.id}. Aborting cleanly — the slice result stands; re-invoke to re-score (the gate is idempotent).`)
  return { hypothesis: HYP.id, slice, score: null, aborted: true }
}
log(`Cycle done: ${HYP.id} → gate ${score.gate_verdict} → measured outcome ${score.measured_outcome}`)
return { hypothesis: HYP.id, slice, score }
